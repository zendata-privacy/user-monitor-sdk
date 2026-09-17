"""Phase 4 — the behavioural baseline and the soft boundary it gates.

Two properties carry the weight here. The threshold must be counted in
*sessions*, because counting calls activates the soft boundary one to two orders
of magnitude early on data far too thin to trust. And a familiar bucket must
stay familiar when the agent reads a new object inside it, because otherwise
normal traffic produces a HIGH per object key within a day of activation.
"""
import json

import pytest

import ai_telemetry
from ai_telemetry.baseline import (
    AgentBehavioralBaseline, BaselineStore, SourceProfile,
    MIN_SESSIONS_FOR_BASELINE, source_container,
)
from ai_telemetry.baseline_builder import (
    build, build_all, reconstruct_sessions, read_events, _subsequences,
)
from ai_telemetry.client import TelemetryEvent
from ai_telemetry.context import current_user_context
from ai_telemetry.scope_checker import (
    ScopeBoundaryChecker, SC_BASELINE_NEW_SOURCE, SC_COLLUSION_BASELINE_DEVIATION,
)
from ai_telemetry.manifest import AgentCapabilityManifest
from ai_telemetry.source_provenance import SourceProvenanceTag, InterAgentMessage

DOCS = "arn:aws:s3:::internal-docs"
HR = "arn:aws:s3:::hr-records"


def _events(sessions=1, turns=2, agent="cs-agent", sources=(f"{DOCS}/a.pdf",),
            start=1_700_000_000_000):
    """A flat event stream shaped like what the SDK actually ships."""
    out = []
    ts = start
    for s in range(sessions):
        sid = f"conv-{s}"
        for t in range(turns):
            out.append({"event_type": "llm_call", "app_id": agent,
                        "session_id": sid, "timestamp_ms": ts})
            ts += 1000
            for uri in sources:
                out.append({
                    "event_type": "tool_call", "app_id": agent, "session_id": sid,
                    "timestamp_ms": ts, "tool_name": "retrieve",
                    "source_provenance": [{"source_uri": uri, "classification": "INTERNAL",
                                           "content_length": 120,
                                           "retrieved_at": "2026-01-01T00:00:00Z"}],
                })
                ts += 1000
    return out


# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------

class TestReconstruction:

    def test_sessions_are_grouped_by_session_id(self):
        by_agent = reconstruct_sessions(_events(sessions=3, turns=2))
        assert len(by_agent["cs-agent"]) == 3

    def test_turns_split_on_each_llm_call(self):
        sessions = reconstruct_sessions(_events(sessions=1, turns=3))["cs-agent"]
        assert len(sessions[0].turns) == 3
        assert sessions[0].turns[0].tool_calls == ["retrieve"]

    def test_events_without_a_session_id_are_dropped_not_synthesised(self):
        """Telemetry predating the schema fix cannot contribute to a session
        count. Bucketing it under a synthetic id would bring the baseline ready
        early on exactly the data that is least trustworthy."""
        evs = _events(sessions=2) + [
            {"event_type": "tool_call", "app_id": "cs-agent",
             "timestamp_ms": 1, "tool_name": "x"}]
        assert len(reconstruct_sessions(evs)["cs-agent"]) == 2

    def test_events_are_ordered_by_timestamp(self):
        evs = list(reversed(_events(sessions=1, turns=3)))
        s = reconstruct_sessions(evs)["cs-agent"][0]
        assert s.started_ms < s.ended_ms

    def test_malformed_events_do_not_break_the_build(self):
        evs = _events(sessions=1) + [None, "junk", 7, {}]
        assert reconstruct_sessions(evs)["cs-agent"]


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------

class TestThreshold:

    def test_below_the_threshold_is_not_ready(self):
        b = build("cs-agent", reconstruct_sessions(
            _events(sessions=99, turns=1))["cs-agent"])
        assert b.sessions_observed == 99
        assert b.baseline_ready is False

    def test_at_the_threshold_it_is_ready(self):
        b = build("cs-agent", reconstruct_sessions(
            _events(sessions=100, turns=1))["cs-agent"])
        assert b.baseline_ready is True

    def test_the_threshold_counts_sessions_not_calls(self):
        """Ten sessions of fifty turns is 500+ events. Counting calls — as
        drift_detector._FULL_THRESHOLD does — would call this ready; counting
        sessions correctly does not."""
        evs = _events(sessions=10, turns=50)
        assert len(evs) > 500
        b = build("cs-agent", reconstruct_sessions(evs)["cs-agent"])
        assert b.sessions_observed == 10
        assert b.baseline_ready is False

    def test_a_stored_ready_flag_cannot_override_the_count(self):
        b = AgentBehavioralBaseline.from_dict(
            {"agent_id": "a", "sessions_observed": 5, "baseline_ready": True})
        assert b.baseline_ready is False


# ---------------------------------------------------------------------------
# has_seen_source
# ---------------------------------------------------------------------------

class TestFamiliarity:

    def _ready(self, *uris):
        return AgentBehavioralBaseline(
            agent_id="a", sessions_observed=MIN_SESSIONS_FOR_BASELINE,
            baseline_ready=True,
            observed_sources={u: SourceProfile(source_uri=u) for u in uris})

    def test_a_new_object_in_a_familiar_bucket_is_familiar(self):
        """Otherwise an agent working normally through a known bucket produces
        a HIGH per new object key."""
        assert self._ready(f"{DOCS}/reports/q3.pdf").has_seen_source(f"{DOCS}/reports/q4.pdf")

    def test_an_unfamiliar_bucket_is_not(self):
        assert not self._ready(f"{DOCS}/a.pdf").has_seen_source(f"{HR}/salaries.csv")

    def test_an_index_on_a_familiar_table_is_familiar(self):
        t = "arn:aws:dynamodb:us-east-1:1:table/Customers"
        assert self._ready(t).has_seen_source(f"{t}/index/by-email")

    def test_a_different_table_is_not(self):
        assert not self._ready(
            "arn:aws:dynamodb:us-east-1:1:table/Customers").has_seen_source(
            "arn:aws:dynamodb:us-east-1:1:table/Salaries")

    def test_an_unfamiliar_secret_is_not_excused_by_a_familiar_one(self):
        """Secrets have no sibling relationship worth inferring — guessing one
        would make an unread secret look familiar."""
        assert not self._ready(
            "arn:aws:secretsmanager:us-east-1:1:secret:app-AbCdEf").has_seen_source(
            "arn:aws:secretsmanager:us-east-1:1:secret:root-XyZwQr")

    @pytest.mark.parametrize("uri,container", [
        (f"{DOCS}/a/b/c.pdf", DOCS),
        ("arn:aws:dynamodb:us-east-1:1:table/T/index/I", "arn:aws:dynamodb:us-east-1:1:table/T"),
        ("vector-index://kb/chunk_id:vx_1", "vector-index://kb"),
        ("https://api.example.com/v1/x", "https://api.example.com"),
        ("arn:aws:secretsmanager:us-east-1:1:secret:a-AbCdEf",
         "arn:aws:secretsmanager:us-east-1:1:secret:a-AbCdEf"),
    ])
    def test_container_extraction(self, uri, container):
        assert source_container(uri) == container


# ---------------------------------------------------------------------------
# Profiles and sequences
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_access_counts_and_frequency(self):
        b = build("cs-agent", reconstruct_sessions(
            _events(sessions=4, turns=2))["cs-agent"])
        p = b.observed_sources[f"{DOCS}/a.pdf"]
        assert p.access_count == 8
        assert p.sessions_seen == 4
        assert p.access_frequency == 2.0          # per session, not per day
        assert p.classifications_seen == ["INTERNAL"]
        assert p.avg_content_length == 120

    def test_tools_and_sequences(self):
        evs = []
        ts = 1
        for s in range(3):
            evs.append({"event_type": "llm_call", "app_id": "a",
                        "session_id": f"c{s}", "timestamp_ms": ts}); ts += 1
            for tool in ("lookup", "fetch", "summarise"):
                evs.append({"event_type": "tool_call", "app_id": "a",
                            "session_id": f"c{s}", "timestamp_ms": ts,
                            "tool_name": tool}); ts += 1
        b = build("a", reconstruct_sessions(evs)["a"])
        assert b.observed_tools == {"lookup": 3, "fetch": 3, "summarise": 3}
        assert b.has_seen_sequence(["lookup", "fetch"])
        assert not b.has_seen_sequence(["summarise", "lookup"])

    def test_sequences_are_bounded(self):
        """A run longer than the cap is a whole session, not a pattern — storing
        it would make the anomaly signal fire on any run of a different length."""
        long_run = [f"t{i}" for i in range(20)]
        assert max(len(s) for s in _subsequences(long_run)) == 6

    def test_turn_and_duration_statistics(self):
        b = build("cs-agent", reconstruct_sessions(
            _events(sessions=5, turns=3))["cs-agent"])
        assert b.typical_turn_count[0] == 3.0
        assert b.typical_session_secs[0] > 0


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------

class TestStore:

    def test_jsonl_round_trip(self, tmp_path):
        store = build_all_from(_events(sessions=100, turns=2))
        out = tmp_path / "baselines.jsonl"
        assert store.to_jsonl(str(out)) == 1

        loaded = BaselineStore.from_jsonl(str(out))
        b = loaded.get("cs-agent")
        assert b.sessions_observed == 100
        assert b.baseline_ready is True
        assert b.has_seen_source(f"{DOCS}/a.pdf")
        assert loaded.ready_agents == ["cs-agent"]

    def test_a_missing_file_yields_an_empty_store(self):
        """Failing closed would turn an unreadable file into a stream of HIGHs
        claiming the agent has never seen anything."""
        assert len(BaselineStore.from_jsonl("/nonexistent/b.jsonl")) == 0

    def test_malformed_lines_are_skipped(self, tmp_path):
        f = tmp_path / "b.jsonl"
        f.write_text('{"agent_id":"a","sessions_observed":200}\nnot json\n\n')
        assert len(BaselineStore.from_jsonl(str(f))) == 1

    def test_builder_reads_a_jsonl_directory(self, tmp_path):
        f = tmp_path / "events.jsonl"
        f.write_text("\n".join(json.dumps(e) for e in _events(sessions=100)))
        store = build_all(tmp_path)
        assert store.get("cs-agent").baseline_ready is True


def build_all_from(events):
    store = BaselineStore()
    for agent, sessions in reconstruct_sessions(events).items():
        store.put(build(agent, sessions))
    return store


# ---------------------------------------------------------------------------
# The soft boundary, end to end
# ---------------------------------------------------------------------------

class TestSoftBoundaryActivation:

    def _manifest(self):
        # Both sources permitted — so anything that fires here is the SOFT
        # boundary, not the hard one.
        return AgentCapabilityManifest(
            agent_id="cs-agent", permitted_sources=[DOCS, HR],
            permitted_classifications=["INTERNAL", "PUBLIC"])

    def test_an_unready_baseline_fires_nothing(self):
        b = build("cs-agent", reconstruct_sessions(_events(sessions=99))["cs-agent"])
        v = ScopeBoundaryChecker().check_direct_access(
            "cs-agent", SourceProvenanceTag(source_uri=f"{HR}/pay.csv"),
            self._manifest(), baseline=b)
        assert v == []

    def test_a_ready_baseline_fires_on_a_genuinely_new_source(self):
        b = build("cs-agent", reconstruct_sessions(_events(sessions=100))["cs-agent"])
        v = ScopeBoundaryChecker().check_direct_access(
            "cs-agent", SourceProvenanceTag(source_uri=f"{HR}/pay.csv"),
            self._manifest(), baseline=b)
        assert [x.signal for x in v] == [SC_BASELINE_NEW_SOURCE]
        assert v[0].severity == "HIGH"

    def test_a_ready_baseline_stays_quiet_on_a_familiar_source(self):
        b = build("cs-agent", reconstruct_sessions(_events(sessions=100))["cs-agent"])
        v = ScopeBoundaryChecker().check_direct_access(
            "cs-agent", SourceProvenanceTag(source_uri=f"{DOCS}/b.pdf"),
            self._manifest(), baseline=b)
        assert v == []

    def test_collusion_baseline_deviation(self):
        b = build("cs-agent", reconstruct_sessions(_events(sessions=100))["cs-agent"])
        msg = InterAgentMessage(
            from_agent="other-agent", to_agent="cs-agent",
            source_provenance=[SourceProvenanceTag(
                source_uri=f"{HR}/pay.csv", retrieved_by="other-agent")])
        v = ScopeBoundaryChecker().check_inter_agent_handoff(
            "cs-agent", msg, self._manifest(), baseline=b)
        assert [x.signal for x in v] == [SC_COLLUSION_BASELINE_DEVIATION]
        assert v[0].origin_agent == "other-agent"


def test_baseline_reaches_the_checker_through_init(tmp_path):
    """The whole Phase 4 path: builder → jsonl → init → a HIGH on a new source."""
    from ai_telemetry import scope_runtime
    scope_runtime._seen_sessions.clear()
    scope_runtime._checker._session_hashes.clear()

    telemetry = tmp_path / "events.jsonl"
    telemetry.write_text("\n".join(json.dumps(e) for e in _events(sessions=100)))
    baselines = tmp_path / "baselines.jsonl"
    build_all(telemetry).to_jsonl(str(baselines))

    client = ai_telemetry.init(
        app_id="cs-agent", api_key="sk", endpoint="http://127.0.0.1:1",
        permitted_sources=[DOCS, HR],
        permitted_classifications=["INTERNAL", "PUBLIC"],
        baseline_path=str(baselines))
    client._send_http = lambda e: None
    current_user_context.set({"session_id": "new-conv"})

    familiar = TelemetryEvent(event_type="tool_call", source_uri=f"{DOCS}/z.pdf",
                              tool_output="x")
    client.capture(familiar); client._send([familiar])
    assert familiar.scope_signals is None

    novel = TelemetryEvent(event_type="tool_call", source_uri=f"{HR}/pay.csv",
                           tool_output="x")
    client.capture(novel); client._send([novel])
    assert novel.scope_signals == SC_BASELINE_NEW_SOURCE
    assert novel.scope_severity == "HIGH"
    assert novel.scope_srm_weight == 0.65


class TestToolSequenceAnomaly:
    """SC_TOOL_SEQUENCE_ANOMALY — the signal is about ORDER, not vocabulary."""

    def _baseline(self):
        """100 sessions of lookup → fetch → summarise, always that order."""
        evs, ts = [], 1
        for s in range(100):
            evs.append({"event_type": "llm_call", "app_id": "a",
                        "session_id": f"c{s}", "timestamp_ms": ts}); ts += 1
            for tool in ("lookup", "fetch", "summarise"):
                evs.append({"event_type": "tool_call", "app_id": "a",
                            "session_id": f"c{s}", "timestamp_ms": ts,
                            "tool_name": tool}); ts += 1
        return build("a", reconstruct_sessions(evs)["a"])

    def test_the_familiar_order_is_quiet(self):
        b, ck = self._baseline(), ScopeBoundaryChecker()
        assert ck.check_tool_sequence("a", "s1", "lookup", b) == []
        assert ck.check_tool_sequence("a", "s1", "fetch", b) == []
        assert ck.check_tool_sequence("a", "s1", "summarise", b) == []

    def test_a_reversed_order_fires(self):
        """Same two calls, opposite direction — unremarkable one way, an
        exfiltration shape the other."""
        from ai_telemetry.scope_checker import SC_TOOL_SEQUENCE_ANOMALY
        b, ck = self._baseline(), ScopeBoundaryChecker()
        ck.check_tool_sequence("a", "s2", "summarise", b)
        v = ck.check_tool_sequence("a", "s2", "lookup", b)
        assert [x.signal for x in v] == [SC_TOOL_SEQUENCE_ANOMALY]
        assert v[0].severity == "MEDIUM"
        assert "summarise → lookup" in v[0].detail

    def test_an_unfamiliar_tool_is_not_an_ordering_anomaly(self):
        """A vocabulary change is the manifest's permitted_tools question, not
        this one. Reporting it here would mislabel what actually happened."""
        b, ck = self._baseline(), ScopeBoundaryChecker()
        ck.check_tool_sequence("a", "s3", "lookup", b)
        assert ck.check_tool_sequence("a", "s3", "exfiltrate", b) == []

    def test_an_unready_baseline_fires_nothing(self):
        b = build("cs-agent",
                  reconstruct_sessions(_events(sessions=5))["cs-agent"])
        assert b.baseline_ready is False
        ck = ScopeBoundaryChecker()
        ck.check_tool_sequence("cs-agent", "s4", "x", b)
        assert ck.check_tool_sequence("cs-agent", "s4", "y", b) == []

    def test_the_first_call_of_a_session_cannot_be_an_ordering_anomaly(self):
        b, ck = self._baseline(), ScopeBoundaryChecker()
        assert ck.check_tool_sequence("a", "s5", "summarise", b) == []

    def test_only_the_latest_pair_is_reported(self):
        """Testing every window would emit one violation per suffix and turn a
        single oddity into a burst."""
        b, ck = self._baseline(), ScopeBoundaryChecker()
        ck.check_tool_sequence("a", "s6", "summarise", b)
        v = ck.check_tool_sequence("a", "s6", "fetch", b)
        assert len(v) == 1

    def test_sessions_do_not_bleed_and_end_session_clears(self):
        b, ck = self._baseline(), ScopeBoundaryChecker()
        ck.check_tool_sequence("a", "s7", "summarise", b)
        assert ck.check_tool_sequence("a", "s8", "lookup", b) == []
        ck.end_session("s7")
        assert ck.check_tool_sequence("a", "s7", "lookup", b) == []

    def test_it_fires_through_the_real_client(self, tmp_path):
        from ai_telemetry.scope_checker import SC_TOOL_SEQUENCE_ANOMALY
        from ai_telemetry import scope_runtime
        scope_runtime._seen_sessions.clear()
        scope_runtime._checker._session_hashes.clear()
        scope_runtime._checker._session_tools.clear()

        out = tmp_path / "b.jsonl"
        BaselineStore({"qa": self._baseline()}).to_jsonl(str(out))
        # The builder keyed it "a"; re-key so the client's app_id finds it.
        loaded = BaselineStore.from_jsonl(str(out))
        base = loaded.get("a")
        base.agent_id = "qa"
        store = BaselineStore({"qa": base})

        client = ai_telemetry.init(app_id="qa", api_key="sk",
                                   endpoint="http://127.0.0.1:1",
                                   permitted_sources=[DOCS],
                                   permitted_classifications=["INTERNAL", "PUBLIC"],
                                   baseline_store=store)
        client._send_http = lambda e: None
        current_user_context.set({"session_id": "live"})

        for name in ("summarise", "lookup"):
            ev = TelemetryEvent(event_type="tool_call", tool_name=name,
                                source_uri=f"{DOCS}/a.pdf", tool_output="x")
            client.capture(ev)
            client._send([ev])
        # The baseline was built from tool-call events carrying no sources, so
        # reading one is legitimately novel too. Both signals are correct here;
        # this test owns the ordering one.
        assert SC_TOOL_SEQUENCE_ANOMALY in ev.scope_signals.split("|")
        assert ev.scope_severity == "HIGH"   # SC_BASELINE_NEW_SOURCE outranks it


class TestConfigurableThreshold:
    """The threshold is tunable, and the chosen value travels with the record."""

    def _sessions(self, n):
        return reconstruct_sessions(_events(sessions=n, turns=1))["cs-agent"]

    def test_default_is_the_spec_value(self):
        assert MIN_SESSIONS_FOR_BASELINE == 100

    def test_a_lower_threshold_activates_earlier(self):
        s = self._sessions(60)
        assert build("cs-agent", s).baseline_ready is False
        assert build("cs-agent", s, min_sessions=60).baseline_ready is True

    def test_the_threshold_is_written_onto_the_record(self):
        """Without it the SDK re-applies its own compiled-in default and can
        silently disagree with the builder — a baseline built at 60 would read
        as not-ready to an SDK defaulting to 100, and the soft boundary would go
        quiet for reasons nobody could see."""
        b = build("cs-agent", self._sessions(60), min_sessions=60)
        loaded = AgentBehavioralBaseline.from_dict(b.to_dict())
        assert loaded.min_sessions == 60
        assert loaded.baseline_ready is True

    def test_an_older_record_without_the_field_falls_back_to_the_default(self):
        b = AgentBehavioralBaseline.from_dict(
            {"agent_id": "a", "sessions_observed": 150})
        assert b.baseline_ready is True
        b2 = AgentBehavioralBaseline.from_dict(
            {"agent_id": "a", "sessions_observed": 60})
        assert b2.baseline_ready is False

    def test_a_zero_threshold_cannot_make_an_empty_baseline_ready(self):
        """Guarding the degenerate case: 0 would make every agent instantly
        'ready' against a baseline that has observed nothing, and every source
        it ever touches would be novel."""
        b = build("cs-agent", [], min_sessions=0)
        assert b.baseline_ready is False
