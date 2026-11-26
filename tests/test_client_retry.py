"""Delivery retries (Scalability.md Phase 1): 429/5xx/network failures are
retried later by the flush loop, honouring Retry-After, without sleeping in it;
batch_id is stable across retries; the backlog is bounded."""

import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from ai_telemetry.client import TelemetryClient, TelemetryEvent


class _Server(HTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.script = []          # status codes to answer, in order; then 202
        self.retry_after = "0"
        self.received = []        # (status, batch_id header, body)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        code = self.server.script.pop(0) if self.server.script else 202
        self.server.received.append((code, self.headers.get("X-Batch-Id"), body))
        self.send_response(code)
        if code == 429:
            self.send_header("Retry-After", self.server.retry_after)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    s = _Server()
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.shutdown()


def _client(server, **kw):
    host, port = server.server_address
    return TelemetryClient(app_id="app", api_key="k", endpoint=f"http://{host}:{port}",
                           flush_interval_s=999, **kw)


def _due_now(c):
    for b in c._retry:
        b["due"] = 0.0


def test_batch_id_in_body_and_header(server):
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    [(code, header, body)] = server.received
    assert code == 202 and len(body["events"]) == 1
    assert header and body["batch_id"] == header
    c.shutdown()


def test_429_is_retried_with_the_same_batch_id(server):
    server.script = [429]
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    assert c.stats()["retry_pending_batches"] == 1 and c.stats()["sent_events"] == 0
    assert c._consecutive_failures == 0          # throttled, not down
    _due_now(c)
    c.flush()
    assert [r[0] for r in server.received] == [429, 202]
    assert server.received[0][1] == server.received[1][1]
    s = c.stats()
    assert s["sent_events"] == 1 and s["retry_pending_batches"] == 0 and s["retried_batches"] == 1
    c.shutdown()


def test_retry_after_is_honoured(server):
    server.script, server.retry_after = [429], "30"
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    [b] = list(c._retry)
    assert b["due"] >= time.time() + 29
    c.flush()                                     # not due yet: nothing sent
    assert len(server.received) == 1
    c.shutdown()


def test_5xx_does_not_block_the_flush(server):
    server.script = [503, 503, 503]
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    start = time.perf_counter()
    c.flush()
    assert time.perf_counter() - start < 1.0      # no sleeping between attempts
    assert len(server.received) == 1 and c.stats()["retry_pending_events"] == 1
    c.shutdown()


def test_other_4xx_is_final(server):
    server.script = [400]
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    assert c.stats()["retry_pending_batches"] == 0 and c.stats()["send_failures"] == 1
    c.shutdown()


def test_circuit_open_parks_and_delivers_after_it_closes(server):
    c = _client(server)
    c._circuit_open_until = time.time() + 60
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    assert server.received == [] and c.stats()["retry_pending_events"] == 1
    c._circuit_open_until = 0.0
    _due_now(c)
    c.flush()
    assert len(server.received) == 1 and c.stats()["sent_events"] == 1
    c.shutdown()


def test_backlog_is_bounded_by_max_queue_size(server):
    server.script = [503] * 10
    c = _client(server, max_queue_size=3, batch_size=1)
    for _ in range(3):
        c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    c._circuit_open_until = 0.0
    for _ in range(2):
        c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    s = c.stats()
    assert s["retry_pending_events"] <= 3 and s["dropped_after_retry"] >= 2
    with patch.object(c, "_attempt", return_value=True):
        c.shutdown()


def test_gives_up_after_retry_window(server):
    server.script = [503, 503]
    c = _client(server)
    c.capture(TelemetryEvent(event_type="llm_call"))
    c.flush()
    [b] = list(c._retry)
    b["first_try"] -= TelemetryClient.RETRY_FOR_S + 1
    b["due"] = 0.0
    c.flush()
    assert c.stats()["retry_pending_batches"] == 0 and c.stats()["dropped_after_retry"] == 1
    c.shutdown()


def test_shutdown_with_endpoint_down_drops_and_returns_quickly():
    c = TelemetryClient(app_id="app", api_key="k", endpoint="http://127.0.0.1:9/x",
                        flush_interval_s=999, batch_size=1)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        for _ in range(3):
            c.capture(TelemetryEvent(event_type="llm_call"))
        c.flush()
        assert c.stats()["retry_pending_batches"] == 3
        start = time.perf_counter()
        c.shutdown()
    assert time.perf_counter() - start < 2.0
    s = c.stats()
    assert s["retry_pending_batches"] == 0 and s["dropped_after_retry"] == 3
