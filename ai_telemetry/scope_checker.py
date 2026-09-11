"""Scope boundary checks — the SC_* signal family.

Runs at every tool call return (direct access) and every inter-agent message
receipt (indirect access), producing ``ScopeViolation`` records that feed the
SRM score.

────────────────────────────────────────────────────────────────────────────────
THIS LAYER OBSERVES. IT DOES NOT BLOCK.

Every entry point here returns a list of violations and mutates nothing outside
its own return value. That is not a limitation worked around — it is what makes
the layer safe to run at all. ``TelemetryClient.capture()`` enqueues and returns;
all evaluation happens later on the background flush thread, alongside the
Zendata engine call, agent fingerprinting and drift detection. Nothing on that
thread is on the host application's request path, so the 4ms budget quoted in
the spec is a budget for work that no user is waiting on.

The consequence worth being explicit about: an agent that violates its manifest
still receives the data. The violation is recorded, scored into SRM and
dispatched; it is not prevented. Preventing it needs a synchronous call the
orchestrator makes *before* delivering a handoff, which is a different API with
a real latency budget and a fail-open decision attached. Nothing in this module
should ever be called from that path without that work being done first.

────────────────────────────────────────────────────────────────────────────────
TWO SEVERITY VOCABULARIES

The spec grades these signals CRITICAL / HIGH / MEDIUM. The SDK's
``alert_dispatcher`` grades everything CRITICAL / WARNING / INVESTIGATE /
MONITOR, and its PagerDuty, OpsGenie, CEF and LEEF integrations are configured
against that ordering by customers who already have runbooks keyed to it.

Extending ``_SEVERITY_ORDER`` with new tiers would silently re-grade every
existing drift alert routed through it. So the spec's vocabulary is kept on the
violation record — it is what the platform stores and what an auditor reads —
and mapped at the dispatch boundary instead. Two vocabularies, one translation,
no existing integration disturbed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ai_telemetry import classification as cls
from ai_telemetry.manifest import AgentCapabilityManifest
from ai_telemetry.source_provenance import SourceProvenanceTag, InterAgentMessage

# ---------------------------------------------------------------------------
# Signal identifiers and their SRM weights (spec §5)
# ---------------------------------------------------------------------------

SC_COLLUSION_MANIFEST_VIOLATION = "SC_COLLUSION_MANIFEST_VIOLATION"
SC_CLASSIFICATION_ESCALATION    = "SC_CLASSIFICATION_ESCALATION"
SC_OUT_OF_SCOPE_SOURCE          = "SC_OUT_OF_SCOPE_SOURCE"
SC_COLLUSION_BASELINE_DEVIATION = "SC_COLLUSION_BASELINE_DEVIATION"
SC_BASELINE_NEW_SOURCE          = "SC_BASELINE_NEW_SOURCE"
SC_SOURCE_MANIPULATION_DELTA    = "SC_SOURCE_MANIPULATION_DELTA"
SC_MANIFEST_DRIFT               = "SC_MANIFEST_DRIFT"
SC_TOOL_SEQUENCE_ANOMALY        = "SC_TOOL_SEQUENCE_ANOMALY"

SRM_WEIGHTS: Dict[str, float] = {
    SC_COLLUSION_MANIFEST_VIOLATION: 0.92,
    SC_CLASSIFICATION_ESCALATION:    0.90,
    SC_OUT_OF_SCOPE_SOURCE:          0.88,
    SC_COLLUSION_BASELINE_DEVIATION: 0.72,
    SC_SOURCE_MANIPULATION_DELTA:    0.70,
    SC_BASELINE_NEW_SOURCE:          0.65,
    SC_MANIFEST_DRIFT:               0.45,
    SC_TOOL_SEQUENCE_ANOMALY:        0.40,
}

# Tool calls retained per session for ordering checks. Matches the builder's
# MAX_SEQUENCE_LEN so the two ends compare like with like.
_MAX_TRACKED_TOOLS = 6

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"

# Spec severity → the dispatcher's existing tiers. See module docstring.
_DISPATCH_SEVERITY: Dict[str, str] = {
    CRITICAL: "CRITICAL",
    HIGH:     "WARNING",
    MEDIUM:   "INVESTIGATE",
}

# Signals produced by the soft boundary. Suppressed wholesale until a baseline
# is ready — see the note in check_direct_access.
SOFT_BOUNDARY_SIGNALS = frozenset({
    SC_BASELINE_NEW_SOURCE,
    SC_COLLUSION_BASELINE_DEVIATION,
    SC_TOOL_SEQUENCE_ANOMALY,
})


@dataclass
class ScopeViolation:
    """One scope boundary violation. Feeds SRM; never blocks anything."""

    signal: str = ""
    severity: str = ""
    detail: str = ""
    agent_id: str = ""
    source_uri: str = ""
    # Which agent originally retrieved the data. Set only for violations that
    # arrived through a handoff — it is what turns "agent B saw something odd"
    # into "agent A handed it to them", and it is the field the collusion graph
    # is drawn from.
    origin_agent: Optional[str] = None
    manifest_version: Optional[int] = None
    classification: Optional[str] = None
    matched_rule: Optional[str] = None
    srm_weight: float = 0.0
    detected_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dispatch_severity(self) -> str:
        """This violation's severity in the alert dispatcher's vocabulary."""
        return _DISPATCH_SEVERITY.get(self.severity, "INVESTIGATE")


def _violation(signal: str, severity: str, detail: str, **kw: Any) -> ScopeViolation:
    return ScopeViolation(
        signal=signal,
        severity=severity,
        detail=detail,
        srm_weight=SRM_WEIGHTS.get(signal, 0.0),
        **kw,
    )


# ---------------------------------------------------------------------------

class ScopeBoundaryChecker:
    """Evaluates provenance tags against an agent's declared and observed scope.

    Stateless with respect to violations — it holds only the per-session content
    hashes needed for manipulation-delta detection, which by definition cannot
    be answered from a single retrieval.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        # (session_id, source_uri) -> (content_hash, hash_scope). Bounded per
        # session by the number of distinct sources a session touches, and
        # dropped by end_session(); a long-lived process serving many sessions
        # would otherwise accumulate one entry per source per session forever.
        self._session_hashes: Dict[str, Dict[str, tuple]] = {}
        # Tool call order per session, for SC_TOOL_SEQUENCE_ANOMALY. Bounded to
        # the window the baseline stores sequences at: keeping the whole session
        # would mean comparing a 200-call transcript against 6-call patterns,
        # which can only ever mismatch.
        self._session_tools: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Hard boundary — direct access
    # ------------------------------------------------------------------

    def check_direct_access(
        self,
        agent_id: str,
        tag: SourceProvenanceTag,
        manifest: Optional[AgentCapabilityManifest],
        baseline: Optional[Any] = None,
    ) -> List[ScopeViolation]:
        """Violations raised by *agent_id* reading *tag.source_uri* itself.

        ``baseline`` is accepted so the signature matches the spec and Phase 4
        needs no call-site change, but soft-boundary signals stay suppressed
        until a baseline exists and reports itself ready. A baseline built on a
        handful of sessions produces enough false positives to get the whole
        signal family muted by the people receiving it, which costs more than
        the detections it would have bought.
        """
        violations: List[ScopeViolation] = []

        if manifest is None:
            return violations

        # ── Hard boundary: source outside the manifest ────────────────────
        # An unenforceable manifest — no sources declared, or IAM Resource "*" —
        # must not produce a violation for every read. It has no boundary to
        # cross. The condition is surfaced at build time instead.
        if manifest.is_enforceable() and not manifest.permits_source(tag.source_uri):
            denied_by = manifest.prohibiting_rule(tag.source_uri)
            reason = (f"explicitly prohibited by '{denied_by}'" if denied_by
                      else "not in its manifest")
            violations.append(_violation(
                SC_OUT_OF_SCOPE_SOURCE, CRITICAL,
                f"{agent_id} accessed {tag.source_uri} which is {reason}",
                agent_id=agent_id,
                source_uri=tag.source_uri,
                manifest_version=manifest.manifest_version,
                matched_rule=denied_by,
            ))

        # ── Hard boundary: classification exceeds clearance ───────────────
        # Every detected class is tested, not only the dominant one: content
        # carrying both PHI and PII reaching an agent cleared for PHI alone
        # would otherwise pass on the strength of its own label.
        for detected in _classes_to_test(tag) if manifest.declares_classifications() else ():
            if cls.exceeds(detected, manifest.permitted_classifications):
                violations.append(_violation(
                    SC_CLASSIFICATION_ESCALATION, CRITICAL,
                    f"{detected} data from {tag.source_uri} reached {agent_id}: "
                    f"{cls.describe_escalation(detected, manifest.permitted_classifications)}",
                    agent_id=agent_id,
                    source_uri=tag.source_uri,
                    classification=detected,
                    manifest_version=manifest.manifest_version,
                ))

        # ── Tamper detection ──────────────────────────────────────────────
        if tag.manipulation_delta:
            violations.append(_violation(
                SC_SOURCE_MANIPULATION_DELTA, HIGH,
                f"Content of {tag.source_uri} differs from its last retrieval "
                f"in this session — possible injection into the retrieval corpus",
                agent_id=agent_id,
                source_uri=tag.source_uri,
            ))

        # ── Soft boundary — suppressed until Phase 4 delivers a baseline ──
        if _baseline_ready(baseline) and not baseline.has_seen_source(tag.source_uri):
            violations.append(_violation(
                SC_BASELINE_NEW_SOURCE, HIGH,
                f"{agent_id} accessed {tag.source_uri}, a source not in its "
                f"{getattr(baseline, 'baseline_period_days', 30)}-day behavioural baseline",
                agent_id=agent_id,
                source_uri=tag.source_uri,
            ))

        return violations

    # ------------------------------------------------------------------
    # Hard boundary — indirect access (collusion)
    # ------------------------------------------------------------------

    def check_inter_agent_handoff(
        self,
        receiving_agent: str,
        message: InterAgentMessage,
        manifest: Optional[AgentCapabilityManifest],
        baseline: Optional[Any] = None,
    ) -> List[ScopeViolation]:
        """Violations raised by *receiving_agent* being handed *message*.

        This is where collusion becomes visible. Neither agent did anything
        individually unauthorised: the sender was entitled to retrieve, and a
        message is not a retrieval. The violation exists only in the join
        between the provenance carried on the payload and the receiver's
        manifest, which is why the tag has to travel with the data.
        """
        violations: List[ScopeViolation] = []
        if manifest is None:
            return violations

        for tag in _deduped(message.source_provenance):
            origin = tag.retrieved_by or message.from_agent

            if manifest.is_enforceable() and not manifest.permits_source(tag.source_uri):
                violations.append(_violation(
                    SC_COLLUSION_MANIFEST_VIOLATION, CRITICAL,
                    f"Data derived from {tag.source_uri} (retrieved by {origin}) "
                    f"arrived at {receiving_agent} whose manifest does not permit "
                    f"this source",
                    agent_id=receiving_agent,
                    source_uri=tag.source_uri,
                    origin_agent=origin,
                    manifest_version=manifest.manifest_version,
                ))

            for detected in (_classes_to_test(tag)
                             if manifest.declares_classifications() else ()):
                if cls.exceeds(detected, manifest.permitted_classifications):
                    violations.append(_violation(
                        SC_CLASSIFICATION_ESCALATION, CRITICAL,
                        f"{detected} data from {tag.source_uri} reached "
                        f"{receiving_agent} via {origin}: "
                        f"{cls.describe_escalation(detected, manifest.permitted_classifications)}",
                        agent_id=receiving_agent,
                        source_uri=tag.source_uri,
                        origin_agent=origin,
                        classification=detected,
                        manifest_version=manifest.manifest_version,
                    ))

            if _baseline_ready(baseline) and not baseline.has_seen_source(tag.source_uri):
                violations.append(_violation(
                    SC_COLLUSION_BASELINE_DEVIATION, HIGH,
                    f"Data from {tag.source_uri}, a source never seen by "
                    f"{receiving_agent}, arrived via {origin}",
                    agent_id=receiving_agent,
                    source_uri=tag.source_uri,
                    origin_agent=origin,
                ))

        return violations

    # ------------------------------------------------------------------
    # Manifest drift
    # ------------------------------------------------------------------

    def check_manifest_drift(
        self,
        agent_id: str,
        current_system_prompt: str,
        manifest: Optional[AgentCapabilityManifest],
    ) -> Optional[ScopeViolation]:
        """The runtime system prompt against the hash that was certified.

        Fires on the first turn of a session whose prompt does not match the
        approved manifest — a developer adding a data source to the prompt
        without going back through certification, which is exactly the change no
        observed baseline can catch, because a baseline learns whatever it is
        shown.
        """
        if manifest is None or not manifest.system_prompt_hash:
            return None
        if not current_system_prompt:
            return None

        from ai_telemetry.manifest import _sha256
        return self.check_manifest_drift_hash(
            agent_id, _sha256(current_system_prompt), manifest)

    def check_manifest_drift_hash(
        self,
        agent_id: str,
        current_hash: str,
        manifest: Optional[AgentCapabilityManifest],
    ) -> Optional[ScopeViolation]:
        """Manifest drift when the runtime hash is already known.

        ``check_manifest_drift`` hashes a prompt the SDK has in hand. Telemetry
        that has already left the process carries only the digest — the prompt
        itself is deliberately never shipped — so replaying it, or evaluating it
        at a gateway, needs the comparison without the hashing step. Same rule,
        one input earlier.
        """
        if manifest is None or not manifest.system_prompt_hash or not current_hash:
            return None
        if current_hash == manifest.system_prompt_hash:
            return None
        return _violation(
            SC_MANIFEST_DRIFT, MEDIUM,
            f"System prompt for {agent_id} has changed since manifest "
            f"v{manifest.manifest_version} was approved by "
            f"{manifest.approved_by or 'an unrecorded approver'} — "
            f"re-run ZASP certification and update the manifest",
            agent_id=agent_id,
            manifest_version=manifest.manifest_version,
        )

    # ------------------------------------------------------------------
    # Manipulation delta bookkeeping
    # ------------------------------------------------------------------

    def note_content_hash(
        self, session_id: str, source_uri: str, content_hash: str, hash_scope: str
    ) -> bool:
        """Record this retrieval's hash; return True if it changed.

        Comparison is refused across differing hash scopes. A source read in
        full on one turn and truncated on the next would otherwise report a
        manipulation that is entirely an artefact of the size cap — the one
        failure mode guaranteed to discredit the signal.
        """
        if not (session_id and source_uri and content_hash):
            return False
        seen = self._session_hashes.setdefault(session_id, {})
        previous = seen.get(source_uri)
        seen[source_uri] = (content_hash, hash_scope)
        if previous is None:
            return False
        prev_hash, prev_scope = previous
        if prev_scope != hash_scope:
            return False
        return prev_hash != content_hash

    def check_tool_sequence(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        baseline: Optional[Any] = None,
    ) -> List[ScopeViolation]:
        """Record a tool call and flag an order this agent has never used.

        The signal is about *order*, not vocabulary. An agent that calls
        ``lookup_account`` then ``send_email`` every day is unremarkable; the
        same two calls in the opposite order may be an exfiltration shape. A
        tool the agent has never called at all is a different question and one
        the manifest's ``permitted_tools`` answers better, so it is not raised
        here.

        Only the most recent pair is tested. Testing every window would emit one
        violation per suffix of an anomalous run — six for a six-call sequence —
        and inflate a single behavioural oddity into a burst.
        """
        if not (session_id and tool_name):
            return []

        recent = self._session_tools.setdefault(session_id, [])
        recent.append(tool_name)
        if len(recent) > _MAX_TRACKED_TOOLS:
            del recent[:-_MAX_TRACKED_TOOLS]

        if not _baseline_ready(baseline) or len(recent) < 2:
            return []

        pair = recent[-2:]
        # An unfamiliar tool makes any pair containing it unfamiliar, which
        # would report a vocabulary change as an ordering anomaly. Both halves
        # have to be known before their order is worth judging.
        if not (baseline.has_seen_tool(pair[0]) and baseline.has_seen_tool(pair[1])):
            return []
        if baseline.has_seen_sequence(pair):
            return []

        return [_violation(
            SC_TOOL_SEQUENCE_ANOMALY, MEDIUM,
            f"{agent_id} called {pair[0]} → {pair[1]}, an order not observed in "
            f"its {getattr(baseline, 'baseline_period_days', 30)}-day baseline",
            agent_id=agent_id,
        )]

    def end_session(self, session_id: str) -> None:
        """Drop per-session hash state. Manipulation delta is a within-session
        signal by definition, so retaining it past the session only leaks."""
        self._session_hashes.pop(session_id, None)
        self._session_tools.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classes_to_test(tag: SourceProvenanceTag) -> List[str]:
    try:
        return tag.classes()
    except AttributeError:      # a plain dict-derived tag
        return [tag.classification] if getattr(tag, "classification", None) else []


def _baseline_ready(baseline: Optional[Any]) -> bool:
    return bool(baseline is not None and getattr(baseline, "baseline_ready", False))


def _deduped(tags: List[SourceProvenanceTag]) -> List[SourceProvenanceTag]:
    """One tag per (source_uri, content_hash).

    A provenance list that has crossed several hops carries the same source
    repeatedly — and a cycle (A→B→A) re-appends it on every pass. Without this,
    one source produces N identical CRITICALs and inflates the SRM score by an
    accident of routing rather than by anything the agents did.
    """
    seen = set()
    out = []
    for t in tags or ():
        key = (getattr(t, "source_uri", ""), getattr(t, "content_hash", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def summarize(violations: List[ScopeViolation]) -> Dict[str, Any]:
    """Fold violations into the fields an event and the SRM input carry."""
    if not violations:
        return {}
    signals = []
    for v in violations:
        if v.signal not in signals:
            signals.append(v.signal)
    worst = min(violations, key=lambda v: ("CRITICAL", "HIGH", "MEDIUM").index(v.severity)
                if v.severity in ("CRITICAL", "HIGH", "MEDIUM") else 99)
    return {
        "scope_signals": "|".join(signals),
        "scope_severity": worst.severity,
        # The heaviest single weight, not a sum: SRM takes a 0–1 input, and
        # adding weights would push two MEDIUMs above one CRITICAL.
        "scope_srm_weight": max(v.srm_weight for v in violations),
        "scope_violation_count": len(violations),
    }


def srm_contribution(violations: List[ScopeViolation]) -> Dict[str, Any]:
    """The scope layer's input to the SRM score (spec §2 step 6).

    SRM itself is computed by ZASP — the scanner's ``zasp_adapter`` posts an
    evaluation request and reads ``srm_score`` back off the response. What the
    SDK owes that computation is a self-describing block: which signals fired,
    how heavily each is weighted, and which agent handed the data over.

    ``weight`` is the heaviest single violation, not a sum. SRM takes a 0–1
    input, so adding weights would put two MEDIUMs (0.45 + 0.40) above a
    CRITICAL out-of-scope access (0.88) — inverting the ranking the weights
    exist to express. The count travels alongside it so a consumer that wants
    to account for volume still can, without the ordering being corrupted by
    default.

    ``origin_agents`` is the field that makes a collusion finding actionable.
    A violation record naming only the receiving agent says something odd
    reached it; naming the sender says who handed it over, which is the edge
    the collusion graph is drawn from.
    """
    if not violations:
        return {}
    by_signal: Dict[str, Dict[str, Any]] = {}
    for v in violations:
        entry = by_signal.setdefault(v.signal, {
            "signal": v.signal,
            "severity": v.severity,
            "weight": v.srm_weight,
            "count": 0,
            "sources": [],
        })
        entry["count"] += 1
        if v.source_uri and v.source_uri not in entry["sources"]:
            entry["sources"].append(v.source_uri)

    origins = []
    for v in violations:
        if v.origin_agent and v.origin_agent not in origins:
            origins.append(v.origin_agent)

    worst = max(violations, key=lambda v: v.srm_weight)
    return {
        "weight": worst.srm_weight,
        "severity": worst.severity,
        "dominant_signal": worst.signal,
        "signals": sorted(by_signal.values(), key=lambda e: -e["weight"]),
        "violation_count": len(violations),
        "origin_agents": origins,
        "manifest_version": worst.manifest_version,
    }


def safety_penalty(srm_weight: Optional[float]) -> float:
    """ZASP safety points to deduct for a scope violation of this weight.

    ZASP's safety score runs 0–10 with higher meaning safer, while SRM weights
    run 0–1 with higher meaning worse. Mapping one onto the other linearly means
    a CRITICAL collusion violation (0.92) costs 9.2 of the 10 available points —
    which is the right magnitude: an agent grounding an answer in data another
    agent was never permitted to hand it is not marginally less safe, it has
    failed the property the score exists to measure.
    """
    if not srm_weight:
        return 0.0
    return max(0.0, min(10.0, float(srm_weight) * 10.0))


__all__ = [
    "ScopeBoundaryChecker", "ScopeViolation", "summarize",
    "srm_contribution", "safety_penalty",
    "SRM_WEIGHTS", "SOFT_BOUNDARY_SIGNALS",
    "CRITICAL", "HIGH", "MEDIUM",
    "SC_COLLUSION_MANIFEST_VIOLATION", "SC_CLASSIFICATION_ESCALATION",
    "SC_OUT_OF_SCOPE_SOURCE", "SC_COLLUSION_BASELINE_DEVIATION",
    "SC_BASELINE_NEW_SOURCE", "SC_SOURCE_MANIPULATION_DELTA",
    "SC_MANIFEST_DRIFT", "SC_TOOL_SEQUENCE_ANOMALY",
]
