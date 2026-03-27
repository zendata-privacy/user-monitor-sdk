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
