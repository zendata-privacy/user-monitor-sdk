"""The scope layer's contribution to the SRM score.

SRM itself is computed by ZASP: the scanner posts an evaluation and reads
``srm_score`` back. What the SDK owes that computation is a self-describing
block of evidence, and a safety score that actually moves when a boundary is
crossed. Both are tested here, because a violation that leaves the ZASP score
untouched is a violation nothing downstream can act on.
"""
import os
import tempfile

import pytest

import ai_telemetry
from ai_telemetry import classification as cls
from ai_telemetry.client import TelemetryEvent
from ai_telemetry.context import current_user_context
from ai_telemetry.drift_detector import derive_zasp_score
from ai_telemetry.scope_checker import (
    ScopeBoundaryChecker, srm_contribution, safety_penalty, SRM_WEIGHTS,
    SC_OUT_OF_SCOPE_SOURCE, SC_CLASSIFICATION_ESCALATION,
    SC_COLLUSION_MANIFEST_VIOLATION, SC_MANIFEST_DRIFT,
)
from ai_telemetry.manifest import AgentCapabilityManifest
from ai_telemetry.source_provenance import SourceProvenanceTag, InterAgentMessage

CUSTOMER_ACCOUNTS = "arn:aws:dynamodb:us-east-1:123456789012:table/CustomerAccounts"


@pytest.fixture(autouse=True)
def _fresh():
    from ai_telemetry import scope_runtime
    scope_runtime._seen_sessions.clear()
    scope_runtime._checker._session_hashes.clear()
    current_user_context.set({})
    yield


def _qa_manifest():
    return AgentCapabilityManifest(
        agent_id="document-qa-agent", manifest_version=3,
        permitted_sources=["arn:aws:s3:::internal-docs"],
        permitted_classifications=[cls.INTERNAL, cls.PUBLIC])


def _collusion_violations():
    checker = ScopeBoundaryChecker()
    msg = InterAgentMessage(
        from_agent="customer-service-agent", to_agent="document-qa-agent",
        source_provenance=[SourceProvenanceTag(
            source_uri=CUSTOMER_ACCOUNTS, retrieved_by="customer-service-agent",
            classification=cls.PII, classifications_detected=[cls.PII])])
    return checker.check_inter_agent_handoff("document-qa-agent", msg, _qa_manifest())


# ---------------------------------------------------------------------------
# The SRM input block
# ---------------------------------------------------------------------------

class TestSrmContribution:

    def test_no_violations_contributes_nothing(self):
        assert srm_contribution([]) == {}

    def test_block_is_self_describing(self):
        block = srm_contribution(_collusion_violations())
        assert block["dominant_signal"] == SC_COLLUSION_MANIFEST_VIOLATION
        assert block["weight"] == 0.92
        assert block["severity"] == "CRITICAL"
        assert block["violation_count"] == 2
        assert block["manifest_version"] == 3
        assert [s["signal"] for s in block["signals"]] == [
            SC_COLLUSION_MANIFEST_VIOLATION, SC_CLASSIFICATION_ESCALATION]

    def test_origin_agents_name_who_handed_the_data_over(self):
        """The field that turns 'something odd reached B' into 'A handed it
        over' — the edge the collusion graph is drawn from."""
        block = srm_contribution(_collusion_violations())
        assert block["origin_agents"] == ["customer-service-agent"]

    def test_direct_access_has_no_origin_agent(self):
        checker = ScopeBoundaryChecker()
        v = checker.check_direct_access(
            "document-qa-agent",
            SourceProvenanceTag(source_uri=CUSTOMER_ACCOUNTS), _qa_manifest())
        assert srm_contribution(v)["origin_agents"] == []

    def test_weight_is_the_heaviest_not_a_sum(self):
        """SRM takes a 0–1 input. Summing would put two MEDIUMs (0.45 + 0.40)
        above a CRITICAL out-of-scope access (0.88), inverting the ranking the
        weights exist to express."""
        checker = ScopeBoundaryChecker()
        drift = checker.check_manifest_drift(
            "a", "changed prompt",
            AgentCapabilityManifest(system_prompt_hash="0" * 64, manifest_version=1))
        direct = checker.check_direct_access(
            "document-qa-agent",
            SourceProvenanceTag(source_uri=CUSTOMER_ACCOUNTS), _qa_manifest())

        block = srm_contribution([drift] + direct)
        assert block["weight"] == 0.88
        assert block["dominant_signal"] == SC_OUT_OF_SCOPE_SOURCE
        assert block["violation_count"] == 2

    def test_repeated_signal_is_counted_not_duplicated(self):
        checker = ScopeBoundaryChecker()
        m = _qa_manifest()
        v = []
        for uri in (CUSTOMER_ACCOUNTS, "arn:aws:s3:::hr-records/pay.csv"):
            v += checker.check_direct_access(
                "document-qa-agent", SourceProvenanceTag(source_uri=uri), m)

        block = srm_contribution(v)
        entry = next(s for s in block["signals"]
                     if s["signal"] == SC_OUT_OF_SCOPE_SOURCE)
        assert entry["count"] == 2
        assert sorted(entry["sources"]) == sorted(
            [CUSTOMER_ACCOUNTS, "arn:aws:s3:::hr-records/pay.csv"])

    def test_weights_match_the_spec_table(self):
        assert SRM_WEIGHTS[SC_COLLUSION_MANIFEST_VIOLATION] == 0.92
        assert SRM_WEIGHTS[SC_CLASSIFICATION_ESCALATION] == 0.90
        assert SRM_WEIGHTS[SC_OUT_OF_SCOPE_SOURCE] == 0.88
        assert SRM_WEIGHTS[SC_MANIFEST_DRIFT] == 0.45


# ---------------------------------------------------------------------------
# The ZASP safety score has to move
# ---------------------------------------------------------------------------

class TestZaspSafetyScore:

    def test_penalty_maps_weight_onto_safety_points(self):
        assert safety_penalty(0.92) == pytest.approx(9.2)
        assert safety_penalty(0.45) == pytest.approx(4.5)
        assert safety_penalty(None) == 0.0
        assert safety_penalty(0.0) == 0.0

    def test_a_clean_event_without_an_engine_verdict_still_scores_nothing(self):
        """No risk data at all must stay None — a clean event is not evidence."""
        assert derive_zasp_score(TelemetryEvent(event_type="llm_call")) is None

    def test_a_violation_alone_produces_a_score(self):
        """zendata_composite_risk needs the Zendata engine and is often absent.
        Returning None there would discard a finding the SDK made on its own
        evidence and is certain of."""
        ev = TelemetryEvent(event_type="tool_call", scope_srm_weight=0.92)
        assert derive_zasp_score(ev) == pytest.approx(0.8)

    def test_a_violation_depresses_an_existing_score(self):
        clean = TelemetryEvent(event_type="llm_call", zendata_composite_risk="20")
        dirty = TelemetryEvent(event_type="llm_call", zendata_composite_risk="20",
                               scope_srm_weight=0.45)
        assert derive_zasp_score(clean) == pytest.approx(8.0)
        assert derive_zasp_score(dirty) == pytest.approx(3.5)

    def test_the_score_never_goes_negative(self):
        ev = TelemetryEvent(event_type="llm_call", zendata_composite_risk="90",
                            scope_srm_weight=0.92)
        assert derive_zasp_score(ev) == 0.0

    def test_a_clean_engine_verdict_is_unchanged(self):
        ev = TelemetryEvent(event_type="llm_call", zendata_composite_risk="10")
        assert derive_zasp_score(ev) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# End to end through the client
# ---------------------------------------------------------------------------

def _client(**kw):
    kw.setdefault("app_id", "document-qa-agent")
    kw.setdefault("api_key", "sk-test")
    c = ai_telemetry.init(
        endpoint="http://127.0.0.1:1", enable_local_log=True,
        log_path=os.path.join(tempfile.mkdtemp(), "events.csv"), **kw)
    c._send_http = lambda events: None
    return c


def test_srm_block_reaches_the_event_and_the_wire():
    client = _client(permitted_sources=["arn:aws:s3:::internal-docs"],
                     permitted_classifications=["INTERNAL", "PUBLIC"],
                     manifest_version=3)
    ev = TelemetryEvent(event_type="tool_call", tool_name="lookup",
                        source_uri=CUSTOMER_ACCOUNTS, tool_output="{}")
    client.capture(ev)
    client._send([ev])

    block = ev.metadata["scope_srm_input"]
    assert block["dominant_signal"] == SC_OUT_OF_SCOPE_SOURCE
    assert block["weight"] == 0.88
    assert block["manifest_version"] == 3

    # The individual records travel too, so a finding can be reviewed rather
    # than only counted.
    detail = ev.metadata["scope_violations"][0]
    assert detail["signal"] == SC_OUT_OF_SCOPE_SOURCE
    assert detail["source_uri"] == CUSTOMER_ACCOUNTS
    assert detail["detected_at"]

    import json
    json.dumps(ev.to_dict())      # the HTTP sink must still be able to send it


def test_a_collusion_finding_lands_on_the_zasp_score():
    """The full path: handoff → violation → SRM weight → ZASP safety score.

    The llm_call branch of _send() calls derive_zasp_score after evaluate_scope,
    so the score it computes has to already reflect the violation.
    """
    client = _client(permitted_sources=["arn:aws:s3:::internal-docs"],
                     permitted_classifications=["INTERNAL", "PUBLIC"])
    handoff = InterAgentMessage(
        from_agent="customer-service-agent", to_agent="document-qa-agent",
        source_provenance=[SourceProvenanceTag(
            source_uri=CUSTOMER_ACCOUNTS, retrieved_by="customer-service-agent",
            classification=cls.PII, classifications_detected=[cls.PII])])
    current_user_context.set({"session_id": "s-srm", "inter_agent_message": handoff})

    ev = TelemetryEvent(event_type="llm_call", zendata_composite_risk="10")
    client.capture(ev)
    client._send([ev])

    assert ev.scope_srm_weight == 0.92
    assert ev.metadata["scope_srm_input"]["origin_agents"] == ["customer-service-agent"]
    # Would have been 9.0 without the violation.
    assert ev.zasp_score == 0.0
