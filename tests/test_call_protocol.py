"""call_protocol: the transport protocol (call type) of captured calls."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from ai_telemetry.transport import (
    protocol_from_serialized, protocol_from_url, protocol_of_client,
)


@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1", "https"),
    ("http://localhost:8080/v1", "http"),
    ("HTTPS://Example.com", "https"),
    ("grpcs://model.internal:443", "grpcs"),
    ("wss://realtime.example.com/ws", "wss"),
    ("http+unix://%2Fvar%2Frun%2Fdocker.sock/v1", "unix"),
    ("arn:aws:dynamodb:us-east-1:123:table/T", None),
    ("api.openai.com", None),
    (None, None),
])
def test_protocol_from_url(url, expected):
    assert protocol_from_url(url) == expected


def test_protocol_of_sdk_resource_and_boto3_client():
    resource = SimpleNamespace(_client=SimpleNamespace(base_url="https://api.anthropic.com"))
    assert protocol_of_client(resource) == "https"
    boto_client = SimpleNamespace(meta=SimpleNamespace(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"))
    assert protocol_of_client(boto_client) == "https"
    assert protocol_of_client(SimpleNamespace()) is None


def test_protocol_from_langchain_serialized():
    assert protocol_from_serialized({"kwargs": {"openai_api_base": "http://proxy:4000"}}) == "http"
    assert protocol_from_serialized({"kwargs": {"model": "gpt-4o"}}) is None


@pytest.fixture
def fake_openai(monkeypatch):
    resp = MagicMock()
    resp.choices = []
    resp.usage = None
    mod = MagicMock()
    mod.resources.chat.completions.Completions.create.return_value = resp
    monkeypatch.setitem(sys.modules, "openai", mod)
    yield mod
    sys.modules.pop("ai_telemetry.integrations.openai_integration", None)


@pytest.mark.parametrize("base_url,expected", [
    ("https://api.openai.com/v1", "https"),
    ("http://localhost:11434/v1", "http"),     # e.g. a local proxy: insecure transport is visible
])
def test_openai_patch_records_call_protocol(fake_openai, mock_telemetry, base_url, expected):
    from ai_telemetry.integrations.openai_integration import patch_openai
    patch_openai(mock_telemetry)
    resource = SimpleNamespace(_client=SimpleNamespace(base_url=base_url))
    fake_openai.resources.chat.completions.Completions.create(
        resource, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert mock_telemetry.captured[0].call_protocol == expected


def test_trace_tool_takes_protocol_from_source_uri(mock_telemetry):
    from ai_telemetry.decorators import trace_tool

    @trace_tool(mock_telemetry, tool_name="crm_lookup", source_uri="https://crm.example.com/api")
    def crm_lookup(q):
        return {"ok": True}

    @trace_tool(mock_telemetry, tool_name="local_calc")
    def local_calc(x):
        return x * 2

    @trace_tool(mock_telemetry, tool_name="grpc_tool", call_protocol="grpc")
    def grpc_tool():
        return 1

    crm_lookup("a"); local_calc(2); grpc_tool()
    by_name = {e.tool_name: e for e in mock_telemetry.captured if e.event_type == "tool_call"}
    assert by_name["crm_lookup"].call_protocol == "https"
    assert by_name["local_calc"].call_protocol is None
    assert by_name["grpc_tool"].call_protocol == "grpc"


def test_langchain_handler_records_protocol(mock_telemetry):
    pytest.importorskip("langchain_core")
    from ai_telemetry.integrations.langchain_integration import create_langchain_handler
    handler = create_langchain_handler(mock_telemetry)
    run_id = uuid4()
    handler.on_chat_model_start({"kwargs": {"openai_api_base": "https://gw.example.com/v1"}},
                                [[]], run_id=run_id)
    handler.on_llm_end(SimpleNamespace(llm_output={}, generations=[]), run_id=run_id)
    llm_events = [e for e in mock_telemetry.captured if e.event_type == "llm_call"]
    assert llm_events and llm_events[-1].call_protocol == "https"


def test_event_serialises_call_protocol():
    from ai_telemetry.client import TelemetryEvent
    assert TelemetryEvent(event_type="llm_call", call_protocol="https").to_dict()["call_protocol"] == "https"
    assert "call_protocol" not in TelemetryEvent(event_type="llm_call").to_dict()
