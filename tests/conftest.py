"""
Shared fixtures for ai_telemetry test suite.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest

from ai_telemetry.client import TelemetryClient
from ai_telemetry.context import current_user_context, _current_trace_id, _current_span_id


# ---------------------------------------------------------------------------
# Minimal local HTTP server that collects inbound POST bodies
# ---------------------------------------------------------------------------

class _CollectingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress output during tests


class MockHTTPServer(HTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received: list = []


@pytest.fixture
def mock_server():
    """Start a real local HTTP server; yield it; shut it down."""
    server = MockHTTPServer(("127.0.0.1", 0), _CollectingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture
def server_url(mock_server):
    host, port = mock_server.server_address
    return f"http://{host}:{port}"


@pytest.fixture
def client(server_url):
    """TelemetryClient pointing at the local mock server with auto-flush disabled."""
    c = TelemetryClient(
        app_id="test-app",
        api_key="sk-test",
        endpoint=server_url,
        flush_interval_s=999,
        debug=True,
    )
    yield c
    c.shutdown()


@pytest.fixture
def mock_telemetry():
    """Mock TelemetryClient that collects captured events in .captured."""
    t = MagicMock()
    t.captured = []
    t.capture.side_effect = lambda e: t.captured.append(e)
    return t


@pytest.fixture(autouse=True)
def reset_context():
    """Reset all ContextVars after every test to prevent cross-test leakage."""
    tok_user = current_user_context.set({})
    tok_trace = _current_trace_id.set("")
    tok_span = _current_span_id.set("")
    yield
    current_user_context.reset(tok_user)
    _current_trace_id.reset(tok_trace)
    _current_span_id.reset(tok_span)
