"""Tests for _create_message stream timeout and retry behaviour."""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import anthropic

import memo_automator
from memo_automator import _create_message, _is_api_error
from memo_chef.pipeline import _retry


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

    @patch.object(memo_automator, "_STREAM_TIMEOUT", 1)  # 1-second timeout for test speed
    def test_hanging_stream_raises_timeout(self):
        client = MagicMock()
        client.messages.stream.return_value = _HangingStream()

        with self.assertRaises(anthropic.APITimeoutError):
            _create_message(client, model="test", messages=[])

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

    @patch.object(memo_automator, "_STREAM_TIMEOUT", 1)
    def test_retry_recovers_from_transient_hang(self):
        """First call hangs (timeout), second call succeeds."""
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
        """All retries timeout → error propagates."""

        def always_fail():
            raise anthropic.APITimeoutError(request=None)

        with self.assertRaises(anthropic.APITimeoutError):
            _retry(always_fail, retries=2, base_delay=0.01, jitter=0)


if __name__ == "__main__":
    unittest.main()
