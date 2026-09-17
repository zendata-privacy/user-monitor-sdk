"""Manifests from an identity/entitlement export, and the demo fixtures.

Most enterprises already hold a capability manifest in their IGA system: a
principal and the entitlements someone approved for it. This is that adapter,
and the tests are anchored on the real demo fixture because the failure this
guards against was found there and not in the abstract.
"""
import collections
import csv
from pathlib import Path

import pytest

from ai_telemetry.identity_manifest import (
    IdentityManifestStore, build_manifest, load_csv,
    parse_entitlements, split_entitlement,
)
from ai_telemetry.baseline_builder import build_all, reconstruct_sessions
from ai_telemetry.scope_checker import ScopeBoundaryChecker, SC_OUT_OF_SCOPE_SOURCE
from ai_telemetry.source_provenance import SourceProvenanceTag

FIXTURES = Path("/media/t8000/Novovolume/Zendata/Clinical-Assistant-Graph-Database"
                "/app/static/autonomous_workflow")
IDENTITIES = FIXTURES / "identities_100.csv"
EVENTS = FIXTURES / "events_fallback.csv"

requires_fixtures = pytest.mark.skipif(
    not IDENTITIES.is_file(), reason="demo fixtures not present")


class TestEntitlementParsing:

    @pytest.mark.parametrize("raw,expected", [
        ("payroll_db.read", ("payroll_db", "read")),
        ("fx_platform.write", ("fx_platform", "write")),
        ("cash_mgmt", ("cash_mgmt", "read")),
        ("", ("", "")),
    ])
    def test_split(self, raw, expected):
        assert split_entitlement(raw) == expected

    def test_a_bare_resource_is_treated_as_readable(self):
        """An export naming a bare resource is granting access to it. Assuming
        otherwise drops it from permitted_sources and turns every legitimate
        read into a CRITICAL."""
        entitled, _ = parse_entitlements("crm")
        assert entitled == ["crm"]

    def test_write_grants_are_permitted_sources_too(self):
        """THE false-positive fix, and it is counter-intuitive enough to pin.

        The observed vocabulary — source_uri, and the older data_sources — names
        a source that was TOUCHED and carries no action. Comparing a
        source-touch against a read-only list compares two different things, and
        fires CRITICAL every time an agent reads something it holds a write
        grant on. On the demo fixture that was 274 identities whose scenario is
        'normal', against 4 that genuinely drifted.
        """
        entitled, by_action = parse_entitlements("a.read|b.write|c.admin")
        assert entitled == ["a", "b", "c"]
        assert by_action["b"] == ["write"]

    def test_comma_and_pipe_are_both_accepted(self):
        assert parse_entitlements("a.read,b.read")[0] == ["a", "b"]


class TestBuildManifest:

    def test_from_a_row(self):
        m = build_manifest({
            "id": "agt_x", "entitlements": "crm.read|payroll_db.write",
            "description": "Summarises accounts", "approved_by": "sarah@x.com",
        })
        assert m.agent_id == "agt_x"
        assert m.permitted_sources == ["crm", "payroll_db"]
        assert m.approved_by == "sarah@x.com"
        assert m.derived_from["permitted_sources"] == "identity_entitlements"
        assert m.is_enforceable()

    def test_write_capable_sources_are_recorded_as_an_observation(self):
        """An agent entitled to write where its purpose is to summarise is worth
        a conversation — and this is the only place both facts are in hand."""
        m = build_manifest({"id": "a", "entitlements": "crm.read|gl_db.write"})
        assert m.scope_gaps["write_capable_sources"] == ["gl_db"]

    def test_alternative_column_names(self):
        m = build_manifest({"identity_id": "a", "permissions": "crm.read"})
        assert m.agent_id == "a" and m.permitted_sources == ["crm"]

    def test_an_identity_with_no_entitlements_yields_no_manifest(self, tmp_path):
        f = tmp_path / "i.csv"
        f.write_text("id,entitlements\na,\nb,crm.read\n")
        assert list(load_csv(str(f))) == ["b"]


@requires_fixtures
class TestDemoFixtures:
    """Anchored on the real files a demo would use."""

    def test_every_identity_yields_a_manifest(self):
        store = IdentityManifestStore.from_csv(str(IDENTITIES))
        assert len(store) == 411
        assert store.get("agt_risk_summarizer").is_enforceable()

    def test_the_observed_vocabulary_is_fully_covered_by_entitlements(self):
        """If a source name appears in telemetry that no entitlement mentions,
        the two vocabularies have diverged and every read of it is a false
        CRITICAL. This is the check that would catch that."""
        store = IdentityManifestStore.from_csv(str(IDENTITIES))
        entitled = set()
        for aid in store.agent_ids:
            entitled.update(store.get(aid).permitted_sources)

        observed = set()
        with open(EVENTS, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for src in (row.get("data_sources") or "").split("|"):
                    if src.strip():
                        observed.add(src.strip())
        assert observed - entitled == set()

    def test_only_the_planted_attack_fires(self):
        """The whole demo, in one assertion: the hard boundary picks out exactly
        the scope_drift identities and stays silent on the 384 normal ones."""
        store = IdentityManifestStore.from_csv(str(IDENTITIES))
        scenario = {}
        with open(IDENTITIES, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                scenario[row["id"]] = row["scenario"]

        checker = ScopeBoundaryChecker()
        flagged = set()
        with open(EVENTS, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                manifest = store.get(row["user_id"])
                if manifest is None:
                    continue
                for src in (row.get("data_sources") or "").split("|"):
                    if not src.strip():
                        continue
                    for v in checker.check_direct_access(
                            row["user_id"],
                            SourceProvenanceTag(source_uri=src.strip()),
                            manifest):
                        if v.signal == SC_OUT_OF_SCOPE_SOURCE:
                            flagged.add(row["user_id"])

        assert {scenario[u] for u in flagged} == {"scope_drift"}
        assert len(flagged) == 4

    def test_the_fixture_cannot_activate_the_soft_boundary(self):
        """Recorded deliberately rather than worked around.

        Every principal in this fixture appears in exactly one session, so no
        baseline can reach MIN_SESSIONS_FOR_BASELINE. Demonstrating the soft
        boundary needs telemetry where the same principal returns across many
        sessions; lowering the threshold to make it fire would be demonstrating
        the thing the threshold exists to prevent.
        """
        store = build_all(EVENTS, session_key="user_id", agent_key="user_id")
        assert len(store) == 411
        assert store.ready_agents == []
        assert all(store.get(a).sessions_observed == 1 for a in store._baselines)

    def test_agent_key_rescues_events_with_no_agent_id(self):
        """Without it, the 28,670 events carrying no agent_id are dropped and
        only the 12 fingerprinted agents get a baseline at all."""
        without = build_all(EVENTS, session_key="user_id")
        with_key = build_all(EVENTS, session_key="user_id", agent_key="user_id")
        assert len(without) == 12
        assert len(with_key) == 411

    def test_data_sources_reaches_the_baseline(self):
        """The legacy pipe-delimited field is the only source vocabulary this
        fixture has; a builder that ignored it would produce empty baselines."""
        store = build_all(EVENTS, session_key="user_id", agent_key="user_id")
        b = store.get("agt_risk_summarizer")
        assert len(b.observed_sources) == 8


class TestSessionKeyGuards:

    def test_trace_id_is_refused(self):
        """Every turn has its own trace, so it would meet a 100-session
        threshold after about a day — silently making it meaningless."""
        with pytest.raises(ValueError, match="trace_id cannot stand in"):
            reconstruct_sessions([], session_key="trace_id")

    def test_user_id_fallback_groups_correctly(self):
        evs = [{"event_type": "llm_call", "user_id": "u1", "agent_id": "a",
                "timestamp_ms": i} for i in range(3)]
        sessions = reconstruct_sessions(evs, session_key="user_id")["a"]
        assert len(sessions) == 1 and len(sessions[0].turns) == 3


class TestCertifiedPromptHash:
    """The roster's system_prompt_hash is what makes SC_MANIFEST_DRIFT possible.

    Without it a roster-derived manifest covers sources and classifications but
    not the prompt — and the prompt is the one thing an observed baseline can
    never catch on its own, because a baseline learns whatever it is shown,
    including an unauthorised edit that was live when observation began.
    """

    def test_the_column_is_read(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "system_prompt_hash": "a" * 64})
        assert m.system_prompt_hash == "a" * 64
        assert m.derived_from["system_prompt_hash"] == "identity_export"

    def test_absence_is_recorded_as_absence(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read"})
        assert m.system_prompt_hash == ""
        assert m.derived_from["system_prompt_hash"] == "not_declared"

    def test_drift_fires_from_a_roster_manifest(self):
        from ai_telemetry.scope_checker import ScopeBoundaryChecker, SC_MANIFEST_DRIFT
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "system_prompt_hash": "a" * 64})
        ck = ScopeBoundaryChecker()
        assert ck.check_manifest_drift_hash("a", "a" * 64, m) is None
        v = ck.check_manifest_drift_hash("a", "b" * 64, m)
        assert v.signal == SC_MANIFEST_DRIFT and v.severity == "MEDIUM"

    def test_no_certified_hash_means_no_drift_check(self):
        """A human has no system prompt. Treating that absence as a change would
        fire MEDIUM against every person in the roster."""
        from ai_telemetry.scope_checker import ScopeBoundaryChecker
        m = build_manifest({"id": "usr_1", "entitlements": "crm.read"})
        assert ScopeBoundaryChecker().check_manifest_drift_hash(
            "usr_1", "b" * 64, m) is None

    def test_the_two_drift_entry_points_agree(self):
        """check_manifest_drift hashes a prompt; check_manifest_drift_hash takes
        the digest. Telemetry never ships the prompt itself, only the hash, so
        both exist — and they must not be able to disagree."""
        from ai_telemetry.scope_checker import ScopeBoundaryChecker
        from ai_telemetry.manifest import _sha256
        prompt = "You are a risk summary agent."
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "system_prompt_hash": _sha256(prompt)})
        ck = ScopeBoundaryChecker()
        assert ck.check_manifest_drift("a", prompt, m) is None
        assert ck.check_manifest_drift_hash("a", _sha256(prompt), m) is None
        edited = prompt + " Also consult the HR store."
        assert ck.check_manifest_drift("a", edited, m).signal == \
            ck.check_manifest_drift_hash("a", _sha256(edited), m).signal


@requires_fixtures
class TestRosterCoherence:
    """The roster is a synthetic dataset built to tell a story. These pin that
    the prompt-hash column tells the SAME story the other columns tell."""

    def _roster(self):
        return list(csv.DictReader(open(IDENTITIES, newline="", encoding="utf-8")))

    def test_only_machines_carry_a_certified_hash(self):
        for row in self._roster():
            has = bool(row.get("system_prompt_hash"))
            assert has == (row["identity_type"] == "machine"), row["id"]

    def test_every_agent_hash_is_distinct(self):
        """Keyed on identity, not job title. Sharing one hash across eleven
        AIAgent identities would mean a drift on any one implicated all of
        them — and a SIEM triage agent and an HR onboarding bot plainly do not
        run the same system prompt."""
        hashes = [r["system_prompt_hash"] for r in self._roster()
                  if r["identity_type"] == "machine"]
        assert len(hashes) == len(set(hashes)) == 12

    def test_exactly_the_scope_drift_agent_drifts(self):
        """Its description says the access was granted "via ad-hoc config
        changes with no formal review" — an un-reviewed config change IS the
        prompt edit. No other scenario describes one, and firing drift for a
        compromised credential or an unused grant would make the signal mean
        "something is wrong" rather than "the prompt changed"."""
        roster = {r["id"]: r for r in self._roster()}
        runtime = collections.defaultdict(list)
        with open(EVENTS, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ident = roster.get(row["user_id"])
                if ident and ident["identity_type"] == "machine" and row.get("system_prompt_hash"):
                    runtime[row["user_id"]].append(row["system_prompt_hash"])

        drifted = {u for u, seen in runtime.items() if len(set(seen)) > 1}
        assert {roster[u]["scenario"] for u in drifted} == {"scope_drift"}
        assert len(drifted) == 1

        # And a clean agent's runtime hash must equal its certified one, or the
        # roster and the telemetry are describing different agents.
        for uid, seen in runtime.items():
            if uid not in drifted:
                assert set(seen) == {roster[uid]["system_prompt_hash"]}, uid

    def test_the_drift_starts_at_the_config_change_not_before(self):
        """A drift that predated the change it explains would be telling a
        different story from the one the story columns tell."""
        rows = [r for r in csv.DictReader(open(EVENTS, newline="", encoding="utf-8"))
                if r["user_id"] == "agt_risk_summarizer"]
        rows.sort(key=lambda r: r["timestamp_iso"])
        hashes = [r["system_prompt_hash"] for r in rows]
        first_new = next(i for i, h in enumerate(hashes) if h != hashes[0])
        assert rows[first_new]["story_phase"] in ("RECON", "PROBE", "ESCALATION")
        assert "config change" in rows[first_new]["story_note"].lower()


class TestPerIdentityManifests:
    """One process, many principals — scope is a property of the CALLER.

    A gateway or a demo assistant serves hundreds of identities from a single
    client. Judging every identity's retrievals against one agent's manifest is
    meaningless in both directions: it invents violations for identities granted
    more, and hides them for identities granted less.
    """

    def _client(self, tmp_path, **kw):
        import ai_telemetry
        c = ai_telemetry.init(
            app_id="gateway", api_key="sk", endpoint="http://127.0.0.1:1",
            enable_local_log=True, log_path=str(tmp_path / "e.csv"), **kw)
        c._send_http = lambda e: None
        return c

    def _roster(self, tmp_path):
        f = tmp_path / "roster.csv"
        f.write_text("id,entitlements,identity_type\n"
                     "alice,crm.read|reporting.read,human\n"
                     "bob,payroll_db.read,human\n")
        return f

    def test_each_identity_is_judged_against_its_own_scope(self, tmp_path):
        from ai_telemetry.client import TelemetryEvent
        from ai_telemetry.context import current_user_context
        from ai_telemetry.scope_checker import SC_OUT_OF_SCOPE_SOURCE
        from ai_telemetry import scope_runtime
        scope_runtime._seen_sessions.clear()

        client = self._client(tmp_path, identity_roster=str(self._roster(tmp_path)))
        assert len(client.manifest_store) == 2

        # alice reading crm is fine; bob reading crm is not.
        out = {}
        for uid in ("alice", "bob"):
            current_user_context.set({"user_id": uid, "session_id": uid})
            ev = TelemetryEvent(event_type="tool_call", user_id=uid,
                                data_sources="crm", tool_output="x")
            client.capture(ev); client._send([ev])
            out[uid] = ev.scope_signals
        assert out["alice"] is None
        assert out["bob"] == SC_OUT_OF_SCOPE_SOURCE

    def test_coverage_is_reported_for_every_known_identity(self, tmp_path):
        """manifest_version on the event is what the dashboard counts as
        'scoped'. An identity with a manifest must always carry it, violation
        or not — otherwise coverage reads 0% and the tab says 'not measured'
        for traffic that WAS measured."""
        from ai_telemetry.client import TelemetryEvent
        from ai_telemetry.context import current_user_context
        client = self._client(tmp_path, identity_roster=str(self._roster(tmp_path)))
        current_user_context.set({"user_id": "alice", "session_id": "s"})
        ev = TelemetryEvent(event_type="tool_call", user_id="alice",
                            data_sources="crm", tool_output="x")
        client.capture(ev); client._send([ev])
        assert ev.manifest_version == 1

    def test_an_unknown_identity_falls_back_rather_than_inventing_scope(self, tmp_path):
        """Somebody absent from the roster has no declared scope. Judging them
        against another identity's manifest would manufacture a violation."""
        from ai_telemetry.client import TelemetryEvent
        from ai_telemetry.context import current_user_context
        client = self._client(tmp_path, identity_roster=str(self._roster(tmp_path)))
        current_user_context.set({"user_id": "stranger", "session_id": "s2"})
        ev = TelemetryEvent(event_type="tool_call", user_id="stranger",
                            data_sources="anything", tool_output="x")
        client.capture(ev); client._send([ev])
        assert ev.scope_signals is None
        assert ev.manifest_version is None

    def test_the_single_manifest_case_is_unchanged(self, tmp_path):
        """No store configured — the original one-agent behaviour must hold."""
        from ai_telemetry.client import TelemetryEvent
        from ai_telemetry.context import current_user_context
        from ai_telemetry.scope_checker import SC_OUT_OF_SCOPE_SOURCE
        client = self._client(tmp_path, permitted_sources=["crm"],
                              permitted_classifications=["INTERNAL", "PUBLIC"])
        assert client.manifest_store is None
        current_user_context.set({"user_id": "anyone", "session_id": "s3"})
        ev = TelemetryEvent(event_type="tool_call", user_id="anyone",
                            data_sources="payroll_db", tool_output="x")
        client.capture(ev); client._send([ev])
        assert ev.scope_signals == SC_OUT_OF_SCOPE_SOURCE

    def test_the_store_wins_over_the_single_manifest(self, tmp_path):
        from ai_telemetry.client import TelemetryEvent
        from ai_telemetry.context import current_user_context
        client = self._client(tmp_path, identity_roster=str(self._roster(tmp_path)),
                              permitted_sources=["nothing-relevant"],
                              permitted_classifications=["PUBLIC"])
        current_user_context.set({"user_id": "alice", "session_id": "s4"})
        ev = TelemetryEvent(event_type="tool_call", user_id="alice",
                            data_sources="crm", tool_output="x")
        client.capture(ev); client._send([ev])
        assert ev.scope_signals is None      # alice's own manifest permits crm


class TestClearanceFromRoster:
    """permitted_classifications must be READ, not just accepted as an argument.

    The column existed in the roster and build_manifest ignored it, taking the
    value only from a parameter nothing passed. Every manifest came back with an
    empty clearance, which the SDK reads as "undeclared" and which therefore
    suppresses SC_CLASSIFICATION_ESCALATION entirely — a boundary silently
    missing an axis while the data to check it sat in the file.
    """

    def test_the_column_is_read(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "permitted_classifications": "PII|INTERNAL|PUBLIC"})
        assert m.permitted_classifications == ["INTERNAL", "PII", "PUBLIC"]
        assert m.declares_classifications()
        assert m.derived_from["permitted_classifications"] == "identity_export"

    def test_comma_separated_and_mixed_case(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "permitted_classifications": "pii, internal"})
        assert m.permitted_classifications == ["INTERNAL", "PII"]

    def test_an_explicit_argument_still_wins(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "permitted_classifications": "PII"},
                           permitted_classifications=["PUBLIC"])
        assert m.permitted_classifications == ["PUBLIC"]

    def test_absence_stays_undeclared(self):
        m = build_manifest({"id": "a", "entitlements": "crm.read"})
        assert not m.declares_classifications()
        assert m.derived_from["permitted_classifications"] == "not_declared"

    def test_escalation_fires_from_a_roster_clearance(self):
        from ai_telemetry.scope_checker import ScopeBoundaryChecker, SC_CLASSIFICATION_ESCALATION
        from ai_telemetry.source_provenance import SourceProvenanceTag
        m = build_manifest({"id": "a", "entitlements": "crm.read",
                            "permitted_classifications": "INTERNAL|PUBLIC"})
        v = ScopeBoundaryChecker().check_direct_access(
            "a", SourceProvenanceTag(source_uri="crm", classification="PII",
                                     classifications_detected=["PII"]), m)
        assert [x.signal for x in v] == [SC_CLASSIFICATION_ESCALATION]


@requires_fixtures
class TestRosterProducesViolations:
    """The demo must actually demonstrate. These pin both halves: the seeded
    identities are caught, and the 384 normal ones are not."""

    def _roster(self):
        return {r["id"]: r for r in csv.DictReader(
            open(IDENTITIES, newline="", encoding="utf-8"))}

    def test_every_identity_declares_a_clearance(self):
        for row in self._roster().values():
            assert row["permitted_classifications"], row["id"]

    def test_normal_identities_are_cleared_for_what_the_assistant_returns(self):
        """The assistant hands the same content to every caller, so a clearance
        narrower than that fires on nearly all of them — 216 normal identities
        in the run that prompted this. A per-identity signal is the wrong shape
        for a property of the corpus."""
        for row in self._roster().values():
            if row["scenario"] == "normal":
                assert {"PII", "CREDENTIALS"} <= set(
                    row["permitted_classifications"].split("|")), row["id"]

    def test_seeded_identities_are_cleared_for_less_than_they_touch(self):
        """Their narrowness IS the finding — the roster already says these 27
        handle data above their grade."""
        narrow = [r for r in self._roster().values()
                  if r["scenario"] != "normal"
                  and not {"PII", "CREDENTIALS"} <= set(
                      r["permitted_classifications"].split("|"))]
        assert len(narrow) >= 20
