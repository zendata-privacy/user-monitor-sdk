"""
Capture fixes for multi-step / multi-agent applications.

1. One user submission is one turn, however many LLM calls answer it.
2. Tools and sources recorded through a per-request accumulator survive the
   copied contexts agent frameworks run tools in.
3. trace_tool resolves its source at call time when given a callable.
4. api_key_hash is the CALLER's credential when the host supplies one.
5. The LangChain handler captures the prompt messages and system prompt hash.
6. The mock evaluator merges PII labels instead of overwriting them, and only
   sensitive identifiers move the composite risk.
7. The PII service labels event content on the flush thread.
"""
import contextvars
import sys
import threading
import types
import uuid
from unittest.mock import MagicMock

import pytest

from ai_telemetry.client import TelemetryEvent, _sha256
from ai_telemetry.context import (current_user_context, record_tool, request_sources,
                                  request_tools, start_request)
from ai_telemetry.decorators import trace_llm, trace_tool


# ── 1. one submission, one turn ─────────────────────────────────────────────

def test_llm_calls_for_one_submission_share_one_turn(client):
    from ai_telemetry.session_state import get_session_store
    start_request(user_id="u-turns", session_id="s-turns", prompt_text="what is my balance",
                  prompt_submission_time_ms=1_700_000_000_000)
    first = TelemetryEvent(event_type="user_message")
    client.capture(first)
    for _ in range(3):                       # router + tool loop + final answer
        e = TelemetryEvent(event_type="llm_call")
        client.capture(e)
        assert e.session_turn_index == first.session_turn_index
        assert e.entity_type == first.entity_type
    state = get_session_store().get_or_create_session("s-turns", "u-turns")
    assert len(state.turns) == 1


def test_a_new_submission_is_a_new_turn(client):
    from ai_telemetry.session_state import get_session_store
    for i, ms in enumerate((1_700_000_100_000, 1_700_000_160_000)):
        start_request(user_id="u-two", session_id="s-two", prompt_text=f"question {i}",
                      prompt_submission_time_ms=ms)
        client.capture(TelemetryEvent(event_type="user_message"))
    assert len(get_session_store().get_or_create_session("s-two", "u-two").turns) == 2


# ── 2. accumulator across copied contexts ───────────────────────────────────

def test_tool_recorded_in_a_copied_context_is_visible_to_the_parent():
    start_request(user_id="u1")
    ctx = contextvars.copy_context()
    t = threading.Thread(target=lambda: ctx.run(record_tool, "search_records"))
    t.start(); t.join()
    assert request_tools() == "search_records"


def test_trace_llm_reports_tools_run_in_another_thread(mock_telemetry):
    start_request(user_id="u2")

    @trace_tool(mock_telemetry, tool_name="get_customer", source_uri="crm/customers")
    def get_customer(cid):
        return {"id": cid}

    ctx = contextvars.copy_context()
    t = threading.Thread(target=lambda: ctx.run(get_customer, "C1"))
    t.start(); t.join()

    @trace_llm(mock_telemetry, model="m")
    def answer():
        return "ok"

    answer()
    llm = mock_telemetry.captured[-1]
    assert llm.event_type == "llm_call"
    assert llm.tools_called == "get_customer"
    assert llm.data_sources == "crm/customers"


def test_without_accumulator_previous_behaviour_is_kept(mock_telemetry):
    current_user_context.set({"user_id": "legacy"})

    @trace_tool(mock_telemetry, tool_name="t1")
    def t1():
        return 1

    t1()
    assert current_user_context.get()["tools_called"] == "t1"


# ── 3. callable source_uri ──────────────────────────────────────────────────

def test_callable_source_uri_single(mock_telemetry):
    start_request(user_id="u3")

    @trace_tool(mock_telemetry, tool_name="search_records",
                source_uri=lambda a, kw, r: f"{kw['resource']}/records")
    def search_records(resource):
        return []

    search_records(resource="payroll_db")
    e = mock_telemetry.captured[-1]
    assert e.source_uri == "payroll_db/records"
    assert e.data_sources == "payroll_db/records"
    assert request_sources() == "payroll_db/records"


def test_callable_source_uri_several(mock_telemetry):
    start_request(user_id="u4")

    @trace_tool(mock_telemetry, tool_name="join",
                source_uri=lambda a, kw, r: ["crm/customers", "loan_db/loans"])
    def join():
        return []

    join()
    e = mock_telemetry.captured[-1]
    assert e.source_uri is None                 # several sources ride data_sources
    assert e.data_sources == "crm/customers|loan_db/loans"


# ── 4. caller credential ────────────────────────────────────────────────────

def test_caller_api_key_is_hashed_into_api_key_hash(client):
    start_request(user_id="svc_bot", caller_api_key="svc-key-123")
    e = TelemetryEvent(event_type="tool_call")
    client.capture(e)
    assert e.api_key_hash == _sha256("svc-key-123")


def test_without_caller_key_the_sdk_key_is_used(client):
    start_request(user_id="human")
    e = TelemetryEvent(event_type="tool_call")
    client.capture(e)
    assert e.api_key_hash == _sha256("sk-test")


# ── 5. LangChain prompt capture ─────────────────────────────────────────────

@pytest.fixture
def lc_stub(monkeypatch):
    base_mod = types.ModuleType("langchain_core.callbacks.base")
    base_mod.BaseCallbackHandler = object
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    callbacks_mod.base = base_mod
    lc_mod = types.ModuleType("langchain_core")
    lc_mod.callbacks = callbacks_mod
    monkeypatch.setitem(sys.modules, "langchain_core", lc_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.base", base_mod)
    sys.modules.pop("ai_telemetry.integrations.langchain_integration", None)
    yield


def _msg(kind, content):
    m = MagicMock()
    m.type, m.content = kind, content
    return m


def test_langchain_captures_messages_and_system_prompt_hash(lc_stub, mock_telemetry):
    from ai_telemetry.integrations.langchain_integration import create_langchain_handler
    start_request(user_id="u5", session_id="s5")
    h = create_langchain_handler(mock_telemetry, capture_tools=False, agent_name="hr_agent")
    run = uuid.uuid4()
    h.on_chat_model_start({}, [[_msg("system", "You are HR."), _msg("human", "ssn of bob?")]], run_id=run)
    resp = MagicMock()
    resp.llm_output = {"model_name": "gpt-4o"}
    gen = MagicMock(); gen.text = "no"
    resp.generations = [[gen]]
    h.on_llm_end(resp, run_id=run)
    e = mock_telemetry.captured[-1]
    assert e.input_messages == [{"role": "system", "content": "You are HR."},
                                {"role": "user", "content": "ssn of bob?"}]
    assert e.system_prompt_hash == _sha256("You are HR.")
    assert e.session_id == "s5"
    assert e.metadata == {"agent_name": "hr_agent"}


def test_langchain_capture_tools_false_disables_tool_events(lc_stub, mock_telemetry):
    from ai_telemetry.integrations.langchain_integration import create_langchain_handler
    h = create_langchain_handler(mock_telemetry, capture_tools=False)
    run = uuid.uuid4()
    h.on_tool_start({}, "x", run_id=run)
    h.on_tool_end("out", run_id=run)
    assert mock_telemetry.captured == []


# ── 6. evaluator merges labels ──────────────────────────────────────────────

def test_mock_evaluator_keeps_existing_labels_and_bumps_on_sensitive(monkeypatch):
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", sailpoint_risk="20", pii_types="SSN|NAME",
                       output_message={"role": "assistant", "content": "done"})
    _mock_evaluate(e)
    assert "SSN" in e.pii_types and "NAME" in e.pii_types
    assert e.zendata_composite_risk == "50"


def test_mock_evaluator_does_not_bump_on_generic_entities():
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", sailpoint_risk="20", pii_types="NAME|DATE",
                       output_message={"role": "assistant", "content": "done"})
    _mock_evaluate(e)
    assert e.pii_detected is True
    assert e.zendata_composite_risk == "20"


# ── 7. PII service labelling ────────────────────────────────────────────────

def test_pii_service_labels_are_merged_on_flush(client):
    svc = MagicMock()
    svc.entity_types.return_value = {"SSN", "NAME"}
    client.pii_classifier = svc
    e = TelemetryEvent(event_type="tool_call", tool_output="Bob 123-45-6789", pii_types="PHONE")
    client._label_with_pii_service(e)
    assert e.pii_types == "NAME|PHONE|SSN"
    assert e.pii_detected is True
    text = svc.entity_types.call_args[0][0]
    assert "Bob 123-45-6789" in text


def test_pii_service_skips_system_prompt(client):
    svc = MagicMock()
    svc.entity_types.return_value = set()
    client.pii_classifier = svc
    e = TelemetryEvent(event_type="llm_call",
                       input_messages=[{"role": "system", "content": "SECRET SYSTEM"},
                                       {"role": "user", "content": "hello"}])
    client._label_with_pii_service(e)
    assert "SECRET SYSTEM" not in svc.entity_types.call_args[0][0]


# ── 8. trust boundary from declared egress ──────────────────────────────────

def test_internal_email_is_not_a_boundary_crossing(client):
    from ai_telemetry.context import record_egress
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    start_request(user_id="u8", tools_called="send_email")
    record_egress("colleague@bank.example", external=False)
    e = TelemetryEvent(event_type="llm_call", tools_called="send_email", sailpoint_risk="20")
    client.capture(e)
    _mock_evaluate(e)
    assert e.trust_boundary_crossed is False
    assert e.zendata_composite_risk == "20"


def test_external_email_is_a_boundary_crossing(client):
    from ai_telemetry.context import record_egress
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    start_request(user_id="u9")
    record_egress("colleague@bank.example", external=False)
    record_egress("someone@partner.example", external=True)
    e = TelemetryEvent(event_type="llm_call", tools_called="send_email", sailpoint_risk="20")
    client.capture(e)
    _mock_evaluate(e)
    assert e.trust_boundary_crossed is True
    assert e.zendata_composite_risk == "60"


def test_without_declared_egress_the_heuristic_still_applies():
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", tools_called="send_email", sailpoint_risk="20")
    _mock_evaluate(e)
    assert e.trust_boundary_crossed is True


# ── 9. PII scored by clearance ──────────────────────────────────────────────

@pytest.mark.parametrize("permitted, expected", [
    (None, "50"),                                   # no declared clearance: +30 as before
    ({"PII", "INTERNAL", "PUBLIC"}, "30"),           # cleared for SSN's class: +10
    ({"INTERNAL", "PUBLIC"}, "50"),                  # not cleared: +30
])
def test_sensitive_pii_scored_by_clearance(permitted, expected):
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", sailpoint_risk="20", pii_types="SSN",
                       output_message={"role": "assistant", "content": "done"})
    _mock_evaluate(e, permitted)
    assert e.zendata_composite_risk == expected


def test_account_number_needs_credentials_clearance():
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", sailpoint_risk="20", pii_types="BANK_ACCOUNT",
                       output_message={"role": "assistant", "content": "done"})
    _mock_evaluate(e, {"PII", "INTERNAL", "PUBLIC"})
    assert e.zendata_composite_risk == "50"


def test_client_passes_the_callers_clearance(client):
    from ai_telemetry.manifest import AgentCapabilityManifest
    store = MagicMock()
    store.get.side_effect = lambda k: AgentCapabilityManifest(
        agent_id=k, permitted_classifications=["PII", "INTERNAL"]) if k == "hr_user" else None
    client.manifest_store = store
    assert client._clearance_for(TelemetryEvent(user_id="hr_user")) == {"PII", "INTERNAL"}
    assert client._clearance_for(TelemetryEvent(user_id="nobody")) is None


def test_a_date_alone_does_not_raise_risk():
    from ai_telemetry.integrations.zendata_integration import _mock_evaluate
    e = TelemetryEvent(event_type="llm_call", sailpoint_risk="41", pii_types="DOB",
                       output_message={"role": "assistant", "content": "as of 2026-11-02"})
    _mock_evaluate(e, {"INTERNAL", "PUBLIC"})
    assert e.zendata_composite_risk == "41"
    assert "DOB" in e.pii_types
