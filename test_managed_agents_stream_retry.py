"""Tests for managed_agents.api_client.stream_events retry/resume behavior."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx

from managed_agents import api_client


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_request() -> httpx.Request:
    return httpx.Request("GET", "https://api.anthropic.com/v1/sessions/sess_x/stream")


class _FakeStreamResponse:
    """Mimics the context-manager returned by httpx.Client.stream()."""

    def __init__(self, *, status_code: int = 200, chunks=None, raise_at: int | None = None):
        self.status_code = status_code
        self._chunks = list(chunks or [])
        self._raise_at = raise_at

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_text(self):
        for i, chunk in enumerate(self._chunks):
            if self._raise_at is not None and i == self._raise_at:
                raise httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body",
                    request=_make_request(),
                )
            yield chunk

    def read(self) -> bytes:
        return b""


class _FakeHttpxClient:
    """Drop-in for httpx.Client(...) used by stream_events.

    Each .stream(...) call pops one queued FakeStreamResponse off the list and
    records the headers it was called with.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.captured_headers: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, headers=None, **kwargs):
        self.captured_headers.append(dict(headers or {}))
        if not self._responses:
            raise AssertionError("FakeHttpxClient: no more queued stream responses")
        return self._responses.pop(0)


class _PatchHttpx:
    """Helper context manager: patches api_client.httpx.Client → factory."""

    def __init__(self, client_instances):
        self._instances = list(client_instances)

    def __enter__(self):
        self._patcher = patch.object(api_client, "httpx")
        self._mock_httpx = self._patcher.start()
        # Preserve real exception classes so isinstance/except clauses work
        for attr in (
            "RemoteProtocolError",
            "ReadError",
            "ConnectError",
            "TimeoutException",
            "NetworkError",
            "Request",
        ):
            setattr(self._mock_httpx, attr, getattr(httpx, attr))
        # httpx.Timeout is also referenced; preserve it
        self._mock_httpx.Timeout = httpx.Timeout
        # The factory yields next queued client per call
        idx = {"i": 0}

        def _client_factory(*args, **kwargs):
            i = idx["i"]
            idx["i"] += 1
            if i >= len(self._instances):
                raise AssertionError(
                    f"FakeHttpxClient: factory called {i + 1} times but only "
                    f"{len(self._instances)} instances were queued"
                )
            return self._instances[i]

        self._mock_httpx.Client.side_effect = _client_factory
        return self

    def __exit__(self, *args):
        self._patcher.stop()


def _sse(data_dict: dict, sse_id: str | None = None) -> str:
    """Format a single SSE message as one chunk."""
    import json

    parts = []
    if sse_id is not None:
        parts.append(f"id: {sse_id}")
    parts.append(f"data: {json.dumps(data_dict)}")
    parts.append("")  # blank line terminator
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamEventsHappyPath(unittest.TestCase):
    """Baseline: single connection terminates on session.status_idle."""

    def test_terminates_on_idle(self):
        chunks = [
            _sse({"type": "agent.tool_use", "name": "code"}),
            _sse({"type": "session.status_idle"}),
        ]
        clients = [_FakeHttpxClient([_FakeStreamResponse(chunks=chunks)])]
        with _PatchHttpx(clients), patch.object(api_client, "time"):
            events = list(api_client.stream_events("sess_x"))

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "agent.tool_use")
        self.assertEqual(events[1]["type"], "session.status_idle")


class TestStreamEventsRetry(unittest.TestCase):
    """stream_events should retry on RemoteProtocolError mid-stream."""

    def test_recovers_from_remote_protocol_error(self):
        # First connection: yield one event then raise.
        # Second connection: yield idle.
        first = _FakeStreamResponse(
            chunks=[
                _sse({"type": "agent.tool_use", "name": "tool_a"}),
                _sse({"type": "agent.message", "text": "hello"}),
            ],
            raise_at=1,  # raise on the 2nd chunk
        )
        second = _FakeStreamResponse(
            chunks=[_sse({"type": "session.status_idle"})],
        )
        clients = [_FakeHttpxClient([first]), _FakeHttpxClient([second])]
        with _PatchHttpx(clients), \
             patch.object(api_client.time, "sleep") as mock_sleep, \
             patch.object(api_client, "get_session", return_value={"status": "running"}):
            events = list(api_client.stream_events("sess_x"))

        # First connection yielded one event before erroring; reconnect yields idle.
        types = [e["type"] for e in events]
        self.assertIn("agent.tool_use", types)
        self.assertIn("session.status_idle", types)
        self.assertEqual(types[-1], "session.status_idle")
        mock_sleep.assert_called()  # backoff should have slept at least once

    def test_sends_last_event_id_on_reconnect(self):
        # First connection emits one event with SSE id, then errors.
        first = _FakeStreamResponse(
            chunks=[
                _sse({"type": "agent.tool_use", "name": "tool_a"}, sse_id="evt-7"),
                _sse({"type": "agent.message", "text": "x"}, sse_id="evt-8"),
            ],
            raise_at=1,
        )
        second = _FakeStreamResponse(
            chunks=[_sse({"type": "session.status_idle"})],
        )
        c1 = _FakeHttpxClient([first])
        c2 = _FakeHttpxClient([second])
        with _PatchHttpx([c1, c2]), \
             patch.object(api_client.time, "sleep"), \
             patch.object(api_client, "get_session", return_value={"status": "running"}):
            list(api_client.stream_events("sess_x"))

        # Second connection must have received the Last-Event-ID header
        self.assertEqual(len(c2.captured_headers), 1)
        sent = c2.captured_headers[0]
        # case-insensitive lookup
        normalized = {k.lower(): v for k, v in sent.items()}
        self.assertEqual(normalized.get("last-event-id"), "evt-7")

    def test_dedupes_replayed_events_by_json_id(self):
        # First connection yields events with ids "a", "b", then errors.
        # Second connection replays "a", "b", and adds "c" + idle.
        first = _FakeStreamResponse(
            chunks=[
                _sse({"id": "a", "type": "agent.message", "text": "1"}),
                _sse({"id": "b", "type": "agent.message", "text": "2"}),
            ],
            raise_at=2,  # raise after both chunks delivered
        )
        # raise_at=2 won't fire (only 2 chunks indices 0,1) — use sentinel chunk
        first = _FakeStreamResponse(
            chunks=[
                _sse({"id": "a", "type": "agent.message", "text": "1"}),
                _sse({"id": "b", "type": "agent.message", "text": "2"}),
                "",  # trigger the raise
            ],
            raise_at=2,
        )
        second = _FakeStreamResponse(
            chunks=[
                _sse({"id": "a", "type": "agent.message", "text": "1"}),
                _sse({"id": "b", "type": "agent.message", "text": "2"}),
                _sse({"id": "c", "type": "agent.message", "text": "3"}),
                _sse({"type": "session.status_idle"}),
            ],
        )
        clients = [_FakeHttpxClient([first]), _FakeHttpxClient([second])]
        with _PatchHttpx(clients), \
             patch.object(api_client.time, "sleep"), \
             patch.object(api_client, "get_session", return_value={"status": "running"}):
            events = list(api_client.stream_events("sess_x"))

        # We should see "a" once, "b" once, "c" once, then idle.
        ids = [e.get("id") for e in events if e.get("id")]
        self.assertEqual(ids, ["a", "b", "c"])
        self.assertEqual(events[-1]["type"], "session.status_idle")

    def test_returns_cleanly_when_session_already_idle(self):
        # First connection errors immediately. get_session reports idle.
        # No further reconnect attempt should happen (only 1 client queued).
        first = _FakeStreamResponse(chunks=[""], raise_at=0)
        clients = [_FakeHttpxClient([first])]
        with _PatchHttpx(clients), \
             patch.object(api_client.time, "sleep") as mock_sleep, \
             patch.object(api_client, "get_session", return_value={"status": "idle"}):
            events = list(api_client.stream_events("sess_x"))

        self.assertEqual(events, [])
        mock_sleep.assert_not_called()  # no backoff before short-circuit return

    def test_exhausts_retries_and_raises(self):
        # Every connection errors. After max attempts, RemoteProtocolError propagates.
        always_err = lambda: _FakeStreamResponse(chunks=[""], raise_at=0)
        clients = [_FakeHttpxClient([always_err()]) for _ in range(10)]
        with _PatchHttpx(clients), \
             patch.object(api_client.time, "sleep"), \
             patch.object(api_client, "get_session", return_value={"status": "running"}):
            with self.assertRaises(httpx.RemoteProtocolError):
                list(api_client.stream_events("sess_x"))


if __name__ == "__main__":
    unittest.main()
