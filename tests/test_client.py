"""
Tests for TelemetryClient (client.py).

Covers: capture, flush, queue limits, prompt redaction,
HTTP retry/circuit-breaker, and shutdown.
"""

import time
import urllib.error
from unittest.mock import patch

import pytest

from ai_telemetry.client import TelemetryClient, TelemetryEvent
from ai_telemetry.context import _current_trace_id


class TestCapture:
    def test_is_non_blocking(self, client):
        event = TelemetryEvent(event_type="llm_call", model="gpt-4o")
        start = time.perf_counter()
        client.capture(event)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1.0, f"capture() took {elapsed_ms:.2f}ms — expected < 1ms"

    def test_sets_app_id(self, client):
        event = TelemetryEvent(event_type="llm_call")
        client.capture(event)
        assert event.app_id == "test-app"

    def test_auto_populates_trace_id_from_context_var(self, client):
        _current_trace_id.set("trace-xyz")
        event = TelemetryEvent(event_type="llm_call")
        client.capture(event)
        assert event.trace_id == "trace-xyz"

    def test_does_not_overwrite_explicit_trace_id(self, client):
        _current_trace_id.set("from-context")
        event = TelemetryEvent(event_type="llm_call", trace_id="pre-set")
        client.capture(event)
        assert event.trace_id == "pre-set"

    def test_assigns_span_id(self, client):
        event = TelemetryEvent(event_type="llm_call")
        client.capture(event)
        assert event.span_id is not None and event.span_id != ""

    def test_enqueues_event(self, client):
        client.capture(TelemetryEvent(event_type="llm_call"))
        assert client._queue.qsize() == 1


class TestPromptRedaction:
    def test_strips_content_when_capture_prompts_false(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url,
            capture_prompts=False, flush_interval_s=999,
        )
        event = TelemetryEvent(
            event_type="llm_call",
            input_messages=[{"role": "user", "content": "secret"}],
            output_message={"role": "assistant", "content": "answer"},
            tool_input={"query": "secret query"},
            tool_output="secret result",
        )
        c.capture(event)
        assert event.input_messages is None
        assert event.output_message is None
        assert event.tool_input is None
        assert event.tool_output is None
        c.shutdown()

    def test_preserves_content_when_capture_prompts_true(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url,
            capture_prompts=True, flush_interval_s=999,
        )
        event = TelemetryEvent(
            event_type="llm_call",
            input_messages=[{"role": "user", "content": "hello"}],
        )
        c.capture(event)
        assert event.input_messages is not None
        c.shutdown()


class TestFlush:
    def test_delivers_events_to_server(self, client, mock_server):
        client.capture(TelemetryEvent(event_type="llm_call"))
        client.capture(TelemetryEvent(event_type="tool_call"))
        client.flush()
        assert len(mock_server.received) == 1
        assert len(mock_server.received[0]["events"]) == 2

    def test_empty_queue_is_noop(self, client, mock_server):
        client.flush()
        assert mock_server.received == []

    def test_clears_queue_after_flush(self, client):
        client.capture(TelemetryEvent(event_type="llm_call"))
        client.flush()
        assert client._queue.qsize() == 0


class TestQueueOverflow:
    def test_drops_events_not_oom(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url,
            max_queue_size=5, flush_interval_s=999,
        )
        for i in range(20):
            c.capture(TelemetryEvent(event_type="llm_call", metadata={"i": i}))
        assert c._queue.qsize() <= 5
        c.shutdown()

    def test_does_not_raise_on_overflow(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url,
            max_queue_size=3, flush_interval_s=999,
        )
        for _ in range(10):
            c.capture(TelemetryEvent(event_type="llm_call"))  # must not raise
        c.shutdown()


class TestHTTPRetry:
    def test_4xx_no_retry_failure_counter_stays_zero(self, client):
        err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            client.capture(TelemetryEvent(event_type="llm_call"))
            client.flush()
        assert client._consecutive_failures == 0

    def test_5xx_increments_failure_counter(self, client):
        err = urllib.error.HTTPError(url="", code=503, msg="Unavailable", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            client.capture(TelemetryEvent(event_type="llm_call"))
            client.flush()
        assert client._consecutive_failures == 1

    def test_network_error_increments_failure_counter(self, client):
        err = urllib.error.URLError("network down")
        with patch("urllib.request.urlopen", side_effect=err):
            client.capture(TelemetryEvent(event_type="llm_call"))
            client.flush()
        assert client._consecutive_failures == 1

    def test_circuit_opens_after_threshold(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url, flush_interval_s=999,
        )
        c._consecutive_failures = 4  # one short of threshold=5
        err = urllib.error.HTTPError(url="", code=503, msg="", hdrs={}, fp=None)
        with patch("urllib.request.urlopen", side_effect=err):
            c.capture(TelemetryEvent(event_type="llm_call"))
            c.flush()
        assert c._circuit_open_until > time.time()
        c.shutdown()

    def test_circuit_open_skips_send(self, client, mock_server):
        client._circuit_open_until = time.time() + 60  # force circuit open
        client.capture(TelemetryEvent(event_type="llm_call"))
        client.flush()
        assert mock_server.received == []


class TestShutdown:
    def test_flushes_pending_events(self, client, mock_server):
        client.capture(TelemetryEvent(event_type="llm_call"))
        client.shutdown()
        assert len(mock_server.received) == 1

    def test_background_thread_stops(self, server_url):
        c = TelemetryClient(
            app_id="app", api_key="key", endpoint=server_url, flush_interval_s=999,
        )
        assert c._thread.is_alive()
        c.shutdown()
        c._thread.join(timeout=2)
        assert not c._thread.is_alive()


class TestFlushBatching:
    """flush() delivers batch_size events at a time, not the whole queue at once."""

    def test_sends_in_batch_size_chunks(self, server_url):
        c = TelemetryClient(app_id="app", api_key="key", endpoint=server_url,
                            batch_size=20, flush_interval_s=999)
        sent = []
        with patch.object(c, "_send_http", side_effect=lambda evs: sent.append(len(evs))):
            for _ in range(45):
                c.capture(TelemetryEvent(event_type="tool_call"))
            c.flush()
        assert sent == [20, 20, 5]
        assert c._queue.qsize() == 0
        c.shutdown()


class _SlowPiiClassifier:
    """Stands in for PiiApiClassifier: a 0.2 s round trip, counting overlap."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.active = self.peak = self.calls = 0

    def entity_types(self, text, content_hash):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.2)
        with self._lock:
            self.active -= 1
        return {"SSN"}


class TestPiiLabellingConcurrency:
    def _client(self, server_url, concurrency):
        c = TelemetryClient(app_id="app", api_key="key", endpoint=server_url,
                            flush_interval_s=999, pii_concurrency=concurrency)
        c.pii_classifier = _SlowPiiClassifier()
        return c

    def _events(self, n):
        return [TelemetryEvent(event_type="tool_call", tool_output=f"row {i}") for i in range(n)]

    def test_calls_run_concurrently_and_label_every_event(self, server_url):
        c = self._client(server_url, 4)
        events = self._events(8)
        start = time.perf_counter()
        c._label_batch_with_pii_service(events)
        elapsed = time.perf_counter() - start
        assert c.pii_classifier.calls == 8
        assert c.pii_classifier.peak == 4
        assert elapsed < 1.0, f"8 calls of 0.2 s at concurrency 4 took {elapsed:.2f}s"
        assert all(e.pii_detected is True and "SSN" in e.pii_types for e in events)
        c.shutdown()

    def test_concurrency_one_is_sequential(self, server_url):
        c = self._client(server_url, 1)
        c._label_batch_with_pii_service(self._events(3))
        assert c.pii_classifier.peak == 1
        c.shutdown()

    def test_a_failing_call_does_not_stop_the_batch(self, server_url):
        c = self._client(server_url, 4)
        calls = []

        def flaky(text, content_hash):
            calls.append(text)
            if text == "row 0":
                raise RuntimeError("pii api down")
            return set()
        c.pii_classifier.entity_types = flaky
        events = self._events(5)
        c._label_batch_with_pii_service(events)
        assert len(calls) == 5
        assert all(e.pii_detected is False for e in events[1:])
        c.shutdown()
