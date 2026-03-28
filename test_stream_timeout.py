"""Tests for _create_message stream timeout and retry behaviour."""
import time
import unittest
from unittest.mock import MagicMock, patch

import anthropic

import memo_automator
from memo_automator import _create_message, _is_api_error
from memo_chef.pipeline import (
    TokenTracker,
    _MessagesProxy,
    _TrackedStream,
    _retry,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text="ok"):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = "end_turn"
        self.usage = _FakeUsage()


class _HangingStream:
    """Simulates a stream that never completes get_final_message()."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_final_message(self):
        # Block for much longer than the timeout we'll use in the test
        time.sleep(30)
        return _FakeMessage()


class _NormalStream:
    """Simulates a healthy stream that returns immediately."""

    def __init__(self, message=None):
        self._msg = message or _FakeMessage()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_final_message(self):
        return self._msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateMessageTimeout(unittest.TestCase):
    """_create_message should raise APITimeoutError when stream hangs."""

    @patch.object(memo_automator, "_STREAM_TIMEOUT", 2)
    def test_hanging_stream_raises_timeout_without_blocking(self):
        """Timeout must fire AND return promptly — not block on pool shutdown."""
        client = MagicMock()
        client.messages.stream.return_value = _HangingStream()

        start = time.monotonic()
        with self.assertRaises(anthropic.APITimeoutError):
            _create_message(client, model="test", messages=[])
        elapsed = time.monotonic() - start

        # Must complete within a few seconds of the timeout, NOT wait 30s
        # for the hanging thread to finish (the old broken behaviour).
        self.assertLess(elapsed, 8, f"Took {elapsed:.1f}s — pool.shutdown likely blocked")

    def test_normal_stream_returns_message(self):
        expected = _FakeMessage("hello")
        client = MagicMock()
        client.messages.stream.return_value = _NormalStream(expected)

        result = _create_message(client, model="test", messages=[])
        self.assertIs(result, expected)


class TestTimeoutIsRetryable(unittest.TestCase):
    """APITimeoutError raised by _create_message should be retried by _retry."""

    def test_api_timeout_is_recognised(self):
        err = anthropic.APITimeoutError(request=None)
        self.assertTrue(_is_api_error(err))

    def test_retry_recovers_from_transient_hang(self):
        """First call raises timeout, second call succeeds."""
        expected = _FakeMessage("recovered")
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise anthropic.APITimeoutError(request=None)
            return expected

        result = _retry(flaky_func, retries=3, base_delay=0.01, jitter=0)
        self.assertEqual(result, expected)
        self.assertEqual(call_count, 2)

    def test_retry_exhaustion_raises(self):
        """All retries timeout -> error propagates."""

        def always_fail():
            raise anthropic.APITimeoutError(request=None)

        with self.assertRaises(anthropic.APITimeoutError):
            _retry(always_fail, retries=2, base_delay=0.01, jitter=0)


# ---------------------------------------------------------------------------
# _TrackedStream tests
# ---------------------------------------------------------------------------

class TestTrackedStream(unittest.TestCase):
    """_TrackedStream should forward to the raw stream and track tokens."""

    def test_get_final_message_tracks_tokens(self):
        """Tokens from the inner stream's message are added to the tracker."""
        raw_stream = _NormalStream(_FakeMessage("tracked"))
        client = MagicMock()
        tracker = TokenTracker(client)
        kwargs = {"model": "claude-sonnet-4-6"}

        self.assertEqual(tracker.input_tokens, 0)
        self.assertEqual(tracker.output_tokens, 0)

        wrapped = _TrackedStream(raw_stream, tracker, kwargs)
        with wrapped:
            msg = wrapped.get_final_message()

        self.assertEqual(msg.content[0].text, "tracked")
        self.assertEqual(tracker.input_tokens, 10)
        self.assertEqual(tracker.output_tokens, 5)
        self.assertGreater(tracker.estimated_cost_usd, 0.0)

    def test_get_final_message_no_usage_attr(self):
        """If the message has no usage attribute, tracker stays at zero."""
        msg_no_usage = MagicMock(spec=[])  # no attributes at all
        msg_no_usage.content = [_FakeTextBlock("bare")]
        raw_stream = _NormalStream(msg_no_usage)
        # Manually set get_final_message to return our custom msg
        raw_stream._msg = msg_no_usage

        client = MagicMock()
        tracker = TokenTracker(client)
        wrapped = _TrackedStream(raw_stream, tracker, {"model": "test"})

        with wrapped:
            wrapped.get_final_message()

        self.assertEqual(tracker.input_tokens, 0)
        self.assertEqual(tracker.output_tokens, 0)

    def test_getattr_delegates_to_raw_stream(self):
        """Unknown attribute access falls through to the raw stream."""
        raw_stream = MagicMock()
        raw_stream.some_custom_attr = "custom_value"
        client = MagicMock()
        tracker = TokenTracker(client)

        wrapped = _TrackedStream(raw_stream, tracker, {})
        self.assertEqual(wrapped.some_custom_attr, "custom_value")


# ---------------------------------------------------------------------------
# _MessagesProxy tests
# ---------------------------------------------------------------------------

class TestMessagesProxy(unittest.TestCase):
    """_MessagesProxy.stream() should return a _TrackedStream wrapper."""

    def test_stream_returns_tracked_stream(self):
        """Calling .stream() on the proxy returns a _TrackedStream instance."""
        client = MagicMock()
        client.messages.stream.return_value = _NormalStream()
        tracker = TokenTracker(client)
        proxy = _MessagesProxy(client, tracker)

        result = proxy.stream(model="test", messages=[], max_tokens=100)
        self.assertIsInstance(result, _TrackedStream)

    def test_stream_wrapper_tracks_tokens_end_to_end(self):
        """The _TrackedStream from proxy.stream() tracks tokens on get_final_message."""
        client = MagicMock()
        raw = _NormalStream(_FakeMessage("proxied"))
        client.messages.stream.return_value = raw
        tracker = TokenTracker(client)
        proxy = _MessagesProxy(client, tracker)

        stream = proxy.stream(model="claude-sonnet-4-6", messages=[], max_tokens=100)
        with stream:
            msg = stream.get_final_message()

        self.assertEqual(msg.content[0].text, "proxied")
        self.assertEqual(tracker.input_tokens, 10)
        self.assertEqual(tracker.output_tokens, 5)

    def test_getattr_delegates_to_real_messages(self):
        """Unknown attrs on the proxy fall through to the real messages object."""
        client = MagicMock()
        client.messages.batches = "batches_obj"
        tracker = TokenTracker(client)
        proxy = _MessagesProxy(client, tracker)

        self.assertEqual(proxy.batches, "batches_obj")


# ---------------------------------------------------------------------------
# _retry callback tests
# ---------------------------------------------------------------------------

class TestRetryCallback(unittest.TestCase):
    """_retry should optionally emit progress via the callback parameter."""

    def test_callback_emitted_on_retry(self):
        """When callback and retry_percent are provided, callback is called on retry."""
        call_count = 0
        callback = MagicMock()

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise anthropic.APITimeoutError(request=None)
            return "ok"

        result = _retry(
            flaky,
            retries=3,
            base_delay=0.01,
            jitter=0,
            callback=callback,
            retry_percent=50,
            stage="mapping",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)
        # callback should have been called exactly once (one retry)
        callback.assert_called_once()
        # The callback receives a StageUpdate object
        stage_update = callback.call_args[0][0]
        self.assertEqual(stage_update.key, "mapping")
        self.assertEqual(stage_update.percent, 50)
        self.assertIn("Retry 1/3", stage_update.label)

    def test_callback_none_does_not_raise(self):
        """When callback is None (default), retries still work without error."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise anthropic.APITimeoutError(request=None)
            return "ok"

        result = _retry(
            flaky,
            retries=3,
            base_delay=0.01,
            jitter=0,
            callback=None,
            retry_percent=50,
            stage="mapping",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)

    def test_callback_not_called_without_retry_percent(self):
        """When retry_percent is None, callback is never invoked even if provided."""
        call_count = 0
        callback = MagicMock()

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise anthropic.APITimeoutError(request=None)
            return "ok"

        result = _retry(
            flaky,
            retries=3,
            base_delay=0.01,
            jitter=0,
            callback=callback,
            retry_percent=None,
            stage="mapping",
        )

        self.assertEqual(result, "ok")
        callback.assert_not_called()

    def test_callback_called_multiple_times_on_multiple_retries(self):
        """Two consecutive failures should emit two callback calls."""
        call_count = 0
        callback = MagicMock()

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise anthropic.APITimeoutError(request=None)
            return "ok"

        result = _retry(
            flaky,
            retries=3,
            base_delay=0.01,
            jitter=0,
            callback=callback,
            retry_percent=60,
            stage="validation",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(callback.call_count, 2)
        # Verify the retry labels increment
        first_update = callback.call_args_list[0][0][0]
        second_update = callback.call_args_list[1][0][0]
        self.assertIn("Retry 1/3", first_update.label)
        self.assertIn("Retry 2/3", second_update.label)


# ---------------------------------------------------------------------------
# _create_message pool cleanup tests
# ---------------------------------------------------------------------------

class TestCreateMessagePoolCleanup(unittest.TestCase):
    """_create_message should shut down the ThreadPoolExecutor on success."""

    @patch("memo_automator.concurrent.futures.ThreadPoolExecutor")
    def test_pool_shutdown_called_on_success(self, mock_pool_cls):
        """On a successful stream, pool.shutdown(wait=False) is called (no leak)."""
        expected = _FakeMessage("success")
        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = expected
        mock_pool.submit.return_value = mock_future
        mock_pool_cls.return_value = mock_pool

        result = _create_message(MagicMock(), model="test", messages=[])

        self.assertIs(result, expected)
        mock_pool.shutdown.assert_called_once_with(wait=False)

    @patch("memo_automator.concurrent.futures.ThreadPoolExecutor")
    def test_pool_shutdown_called_on_timeout(self, mock_pool_cls):
        """On timeout, pool.shutdown(wait=False, cancel_futures=True) is called."""
        import concurrent.futures

        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()
        mock_pool.submit.return_value = mock_future
        mock_pool_cls.return_value = mock_pool

        with self.assertRaises(anthropic.APITimeoutError):
            _create_message(MagicMock(), model="test", messages=[])

        mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    @patch("memo_automator.concurrent.futures.ThreadPoolExecutor")
    def test_pool_shutdown_not_wait_true(self, mock_pool_cls):
        """pool.shutdown must never be called with wait=True (would block)."""
        expected = _FakeMessage("no-block")
        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = expected
        mock_pool.submit.return_value = mock_future
        mock_pool_cls.return_value = mock_pool

        _create_message(MagicMock(), model="test", messages=[])

        for c in mock_pool.shutdown.call_args_list:
            # wait should never be True
            if c.kwargs.get("wait") is True:
                self.fail("pool.shutdown was called with wait=True — would block on hung stream")


if __name__ == "__main__":
    unittest.main()
