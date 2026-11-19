"""ai_telemetry.persistence: learned state survives a restart (a batch job per day)."""
import json

import pytest

from ai_telemetry import persistence
from ai_telemetry.agent_fingerprint import AgentFingerprintStore, SystemPromptVersion
from ai_telemetry.drift_detector import AgentDriftStore, TimedSample
from ai_telemetry.role_inference import HumanRoleStore
from ai_telemetry.session_state import SessionStore


@pytest.fixture
def fresh_stores(monkeypatch):
    import ai_telemetry.agent_fingerprint as af
    import ai_telemetry.drift_detector as dd
    import ai_telemetry.role_inference as ri
    import ai_telemetry.session_state as ss
    stores = {"drift": AgentDriftStore(), "fp": AgentFingerprintStore(),
              "roles": HumanRoleStore(), "sessions": SessionStore()}
    monkeypatch.setattr(dd, "_global_drift_store", stores["drift"])
    monkeypatch.setattr(af, "_global_store", stores["fp"])
    monkeypatch.setattr(ri, "_human_store", stores["roles"])
    monkeypatch.setattr(ss, "_global_store", stores["sessions"])
    return stores


def _populate(s):
    b = s["drift"].get_or_create("agt_x")
    b.total_calls = 150
    b.full_baseline_active = True
    b.baseline_tool_scope = {"search_records", "list_resources"}
    b.safety_samples = [TimedSample(1, 0.9), TimedSample(2, 0.8)]
    r = s["fp"].get_or_create("agt_x", "abcd")
    r.tool_scope = ["search_records"]
    r.system_prompt_versions = [SystemPromptVersion("h1", 1, 2, 5)]
    r.call_count = 150
    h = s["roles"].get_or_create("usr_1")
    h.role_hits["FINANCE"] = 4
    s["roles"].record_session("usr_1", "sess_a")
    s["sessions"].update_baseline("usr_1", 23.0)


def test_round_trip_through_a_file(fresh_stores, tmp_path, monkeypatch):
    _populate(fresh_stores)
    path = tmp_path / "state" / "sdk_state.json"
    persistence.save_state(str(path))
    json.loads(path.read_text())                         # plain JSON

    # a "restart": brand-new empty stores
    import ai_telemetry.agent_fingerprint as af
    import ai_telemetry.drift_detector as dd
    import ai_telemetry.role_inference as ri
    import ai_telemetry.session_state as ss
    monkeypatch.setattr(dd, "_global_drift_store", AgentDriftStore())
    monkeypatch.setattr(af, "_global_store", AgentFingerprintStore())
    monkeypatch.setattr(ri, "_human_store", HumanRoleStore())
    monkeypatch.setattr(ss, "_global_store", SessionStore())

    counts = persistence.load_state(str(path))
    assert counts == {"drift_baselines": 1, "agent_fingerprints": 1, "human_roles": 1,
                      "entity_risk_baselines": 1}
    b = dd.get_drift_store().get_or_create("agt_x")
    assert b.total_calls == 150 and b.full_baseline_active
    assert b.baseline_tool_scope == {"search_records", "list_resources"}
    assert b.safety_samples[1].value == 0.8 and isinstance(b.safety_samples[0], TimedSample)
    r = af.get_fingerprint_store().get("agt_x")
    assert r.call_count == 150 and r.system_prompt_versions[0].prompt_hash == "h1"
    h = ri.get_human_role_store().get_or_create("usr_1")
    assert h.role_hits["FINANCE"] == 4
    assert ri.get_human_role_store().record_session("usr_1", "sess_b") == 2
    assert ss.get_session_store().get_baseline("usr_1").risk_scores == [23.0]


def test_missing_or_bad_file_restores_nothing(fresh_stores, tmp_path):
    assert persistence.load_state(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert persistence.load_state(str(bad)) == {}
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": 999}))
    assert persistence.load_state(str(other)) == {}
