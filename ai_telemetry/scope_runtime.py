"""Runs the scope boundary checks against telemetry events.

The seam between ``scope_checker`` (which knows the rules and holds no
telemetry) and ``TelemetryClient`` (which knows the transport and holds no
rules). Called from ``TelemetryClient._send()`` — the background flush thread —
alongside the Zendata engine call, agent fingerprinting and drift detection.

Nothing here runs on the caller's thread. That is what makes it safe to do a
SHA-256 over retrieved content and, when a PII API key is configured, an HTTPS
round trip to classify it: no user is waiting on either.

────────────────────────────────────────────────────────────────────────────────
WHERE source_uri COMES FROM

The SDK cannot infer it. ``trace_tool`` sees a function name, its arguments and
its return value; nothing in that tells it the result came from a DynamoDB
table. Four producers, and this module accepts all of them so that adopting the
cheap one first costs nothing later:

1. ``@trace_tool(..., source_uri="arn:...")`` — declared on the decorator.
   Static sources, one line, no runtime cost.
2. ``with source_context("arn:...")`` — for sources only known at runtime, such
   as a vector chunk id.
3. ``event.source_uri`` set directly by the host application.
4. The boto3 interceptor (Phase 2), which registers an ``after-call`` hook and
   reports the real table, bucket and key with no customer code change at all.

Producers 1–3 require somebody to have annotated the call. Producer 4 is the one
that makes the feature true in an account that will not annotate four hundred
tool functions, and it reports ARNs — the same vocabulary the manifest holds,
because the manifest is built from IAM.

Until a producer supplies a URI there is nothing to check, and this module does
nothing rather than guessing. A guessed source identity would put a fabricated
CRITICAL on an agent's record, which is worse than the silence it replaces.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_telemetry import classification as cls
from ai_telemetry.scope_checker import ScopeBoundaryChecker, summarize, srm_contribution
from ai_telemetry.source_provenance import (
    SourceProvenanceTag, InterAgentMessage,
    hash_content, make_source_id, infer_source_type,
)

# One checker per process. It holds only per-session content hashes for
# manipulation-delta detection; violations are returned, never accumulated.
_checker = ScopeBoundaryChecker()


def get_checker() -> ScopeBoundaryChecker:
    return _checker


def _content_text(event) -> str:
    """The retrieved content to hash and classify.

    ``tool_output`` is what a retrieval returns. It has already been through
    ``_safe_serialize`` in the decorator, so it is JSON-shaped rather than raw
    bytes — which is fine for both purposes here: the hash only needs to be
    stable and comparable against itself, and the classifier only needs text.
    """
    out = getattr(event, "tool_output", None)
    if out is None:
        return ""
    return out if isinstance(out, str) else str(out)


def build_tag(
    event,
    source_uri: str,
    agent_id: str,
    session_id: str,
    manifest=None,
    pii_classifier=None,
    max_hash_bytes: Optional[int] = None,
) -> SourceProvenanceTag:
    """Construct the provenance tag for one retrieval."""
    from ai_telemetry.source_provenance import DEFAULT_MAX_HASH_BYTES

    content = _content_text(event)
    content_hash, hash_scope, length = hash_content(
        content, max_hash_bytes or DEFAULT_MAX_HASH_BYTES)

    tag = SourceProvenanceTag(
        source_id=make_source_id(source_uri),
        source_type=infer_source_type(source_uri),
        source_uri=source_uri,
        retrieved_by=agent_id or "",
        retrieved_at=getattr(event, "timestamp_ms", None) and _iso(event.timestamp_ms) or "",
        content_hash=content_hash,
        hash_scope=hash_scope,
        content_length=length,
        tool_name=getattr(event, "tool_name", None),
    )

    if manifest is not None:
        tag.in_manifest = manifest.permits_source(source_uri)
        tag.matched_rule = manifest.matching_rule(source_uri)

    tag.classifications_detected = sorted(
        _classify(content, content_hash, event, pii_classifier))
    tag.classification = cls.dominant(tag.classifications_detected)

    # note_content_hash refuses to compare a container against itself — see
    # its docstring. The rule lives there, not here, so no caller can bypass it.
    tag.manipulation_delta = _checker.note_content_hash(
        session_id, source_uri, content_hash, hash_scope)
    return tag


def _classify(content: str, content_hash: str, event, pii_classifier) -> set:
    """Clearance classes present in the retrieved content.

    The PII API is authoritative when configured. The built-in regex classifier
    in ``detection.py`` is the floor — it has already labelled this event's
    ``pii_types`` on the caller's thread for free, so reusing that costs nothing
    and means an unconfigured deployment still detects the categories it can.
    """
    found: set = set()

    if pii_classifier is not None and content:
        found |= pii_classifier.classify(content, content_hash)

    existing = getattr(event, "pii_types", None)
    if existing:
        found |= cls.classes_for_builtin_types(existing.split("|"))

    return found


def _iso(timestamp_ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(
        timestamp_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Entry point from TelemetryClient._send()
# ---------------------------------------------------------------------------

def evaluate_scope(client, event) -> None:
    """Attach source provenance and scope violations to *event*, in place.

    Never raises. A scope layer that can break the telemetry path it rides on
    would take down observability to report a policy finding, which is a trade
    nobody would accept if it were offered explicitly.
    """
    agent_id = getattr(event, "agent_id", None) or client.app_id
    session_id = _session_id(event, client)
    manifest = _manifest_for(client, event, agent_id)

    violations: List[Any] = []

    # ── Manifest drift, once per session ─────────────────────────────────
    if event.event_type in ("llm_call", "user_message"):
        prompt = _system_prompt_for(event)
        if prompt and _first_turn_of_session(session_id, client):
            drift = _checker.check_manifest_drift(agent_id, prompt, manifest)
            if drift:
                violations.append(drift)

    # ── Direct access, on every retrieval that names a source ────────────
    tags = _tags_for(event, client, agent_id, session_id, manifest)
    for tag in tags:
        violations.extend(
            _checker.check_direct_access(agent_id, tag, manifest,
                                         baseline=_baseline_for(client, agent_id)))

    # ── Tool ordering, on every tool call ────────────────────────────────
    if event.event_type == "tool_call" and event.tool_name:
        violations.extend(_checker.check_tool_sequence(
            agent_id, session_id, event.tool_name,
            baseline=_baseline_for(client, agent_id)))

    # ── Indirect access, when the event carries a handoff ────────────────
    handoff = _handoff_for(event)
    if handoff is not None:
        violations.extend(
            _checker.check_inter_agent_handoff(
                agent_id, handoff, manifest,
                baseline=_baseline_for(client, agent_id)))
        tags = list(tags) + list(handoff.source_provenance)

    # ── Write the results onto the event ─────────────────────────────────
    if tags:
        primary = tags[0]
        event.source_uri = event.source_uri or primary.source_uri
        event.source_type = event.source_type or primary.source_type
        event.source_classification = primary.classification
        event.source_in_manifest = primary.in_manifest
        event.content_hash = event.content_hash or primary.content_hash
        if primary.manipulation_delta:
            event.manipulation_delta = True
        event.source_provenance = [t.to_dict() for t in tags]

    if manifest is not None:
        event.manifest_version = manifest.manifest_version
        if manifest.asset_arn and not event.asset_arn:
            event.asset_arn = manifest.asset_arn

    if violations:
        for key, value in summarize(violations).items():
            setattr(event, key, value)
        origins = [v.origin_agent for v in violations if v.origin_agent]
        if origins:
            event.origin_agent = origins[0]

        # The SRM input block (spec §2 step 6). Rides metadata rather than a
        # flat column because it is structured — per-signal weights and the
        # origin agents behind a collusion finding — and flattening it would
        # leave the SRM consumer with a severity label and no way to see what
        # produced it. The scalar summary fields above carry the same verdict
        # for consumers that only need the headline.
        event.metadata = dict(event.metadata or {})
        event.metadata["scope_srm_input"] = srm_contribution(violations)
        event.metadata["scope_violations"] = [v.to_dict() for v in violations]

        # Land the violation on the ZASP safety score here, unconditionally.
        #
        # _send() also derives that score, but only inside the agent
        # fingerprinting branch — which requires event_type == "llm_call", an
        # entity_type of API CALL / AGENT / AMBIGUOUS, and a resolved agent_id.
        # A tool_call never satisfies it, and the retrieval that crosses a
        # boundary IS a tool_call. Left to that branch alone, the most severe
        # signal this layer can raise would leave zasp_score empty on exactly
        # the events that carry it.
        #
        # derive_zasp_score is pure with respect to the event, so the later
        # branch recomputing it is harmless: it arrives at the same number.
        from ai_telemetry.drift_detector import derive_zasp_score
        scored = derive_zasp_score(event)
        if scored is not None:
            event.zasp_score = scored

        _dispatch(violations, client)


# ---------------------------------------------------------------------------
# Sources of each input
# ---------------------------------------------------------------------------

def _session_id(event, client) -> str:
    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}
    return (ctx.get("session_id") or ctx.get("user_id")
            or getattr(event, "trace_id", None) or "unknown")


def _system_prompt_for(event) -> str:
    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}
    prompt = ctx.get("system_prompt")
    if prompt:
        return prompt
    for msg in getattr(event, "input_messages", None) or ():
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


_seen_sessions: set = set()


def _first_turn_of_session(session_id: str, client) -> bool:
    """Manifest drift is a session-start check, not a per-turn one.

    Re-hashing an unchanged prompt on every turn would emit the same MEDIUM
    dozens of times per conversation, which is how a real signal becomes noise
    somebody writes a filter for.
    """
    if session_id in _seen_sessions:
        return False
    _seen_sessions.add(session_id)
    if len(_seen_sessions) > 10_000:
        _seen_sessions.clear()
        _seen_sessions.add(session_id)
    return True


def _tags_for(event, client, agent_id, session_id, manifest) -> List[SourceProvenanceTag]:
    """Provenance tags for this event, from whichever producer supplied a URI."""
    if getattr(event, "source_provenance", None):
        return [SourceProvenanceTag.from_dict(t) for t in event.source_provenance]

    uris = _source_uris(event)
    if not uris:
        return []

    return [
        build_tag(event, uri, agent_id, session_id, manifest,
                  pii_classifier=getattr(client, "pii_classifier", None),
                  max_hash_bytes=getattr(client, "max_hash_bytes", None))
        for uri in uris
    ]


def _source_uris(event) -> List[str]:
    """Every source this event names, in producer precedence order."""
    if getattr(event, "source_uri", None):
        return [event.source_uri]

    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}

    # Producer 2 — source_context()
    scoped = ctx.get("source_uri")
    if scoped:
        return [scoped]

    # Producer 4 — the boto3 interceptor. Ahead of the hand-maintained
    # data_sources field because it observed the actual call rather than
    # repeating what somebody typed, and one tool call may have read several
    # places: a DynamoDB lookup then an S3 fetch both belong on this chain.
    accumulated = ctx.get("boto3_sources")
    if accumulated:
        from ai_telemetry.integrations.boto3_integration import drain_sources
        drained = drain_sources()
        if drained:
            return drained

    # Producer 3 — the pre-existing data_sources field. Pipe-delimited and set
    # by hand, but it is what deployments already populate, so honouring it
    # means the hard boundary works on existing instrumentation.
    declared = getattr(event, "data_sources", None) or ctx.get("data_sources")
    if declared:
        return [s.strip() for s in str(declared).split("|") if s.strip()]

    return []


def _handoff_for(event) -> Optional[InterAgentMessage]:
    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}
    raw = ctx.get("inter_agent_message") or (event.metadata or {}).get("inter_agent_message")
    if not raw:
        return None
    if isinstance(raw, InterAgentMessage):
        return raw
    if isinstance(raw, dict):
        return InterAgentMessage.from_dict(raw)
    return None


def _manifest_for(client, event, agent_id):
    """The declared scope to judge THIS event against.

    One process, one manifest is the normal case and stays the default. But a
    gateway or a demo assistant serves many principals from a single process,
    and there the manifest is a property of the caller rather than of the
    deployment — checking every identity's retrievals against one agent's scope
    would be meaningless in both directions.

    So a ``manifest_store`` is consulted first, keyed by the event's own
    identity. Falling back to ``client.manifest`` keeps the single-agent case
    working unchanged.
    """
    store = getattr(client, "manifest_store", None)
    if store is not None:
        for key in (getattr(event, "user_id", None), getattr(event, "agent_id", None), agent_id):
            if key:
                found = store.get(key)
                if found is not None:
                    return found
    return getattr(client, "manifest", None)


def _baseline_for(client, agent_id):
    """This agent's observed scope, or None.

    None keeps every soft-boundary signal suppressed, and so does a baseline
    whose session count has not reached the threshold — the check is in
    ScopeBoundaryChecker, deliberately, so there is exactly one place that
    decides whether the soft boundary is live.

    Keyed on agent_id with app_id as the fallback, matching how the builder
    keys them: agent_id when fingerprinting resolved one, app_id otherwise.
    """
    store = getattr(client, "baseline_store", None)
    if store is None:
        return None
    return store.get(agent_id) or store.get(client.app_id)


def _dispatch(violations: List[Any], client) -> None:
    """Route violations through the existing alert dispatcher.

    Severity is translated at this boundary — the spec's CRITICAL/HIGH/MEDIUM
    into the dispatcher's own tiers — so customers' existing PagerDuty and SIEM
    routing rules keep working unchanged.
    """
    try:
        from ai_telemetry.alert_dispatcher import get_dispatcher
        dispatcher = get_dispatcher()
        if dispatcher is None:
            return
        for v in violations:
            dispatcher.dispatch({
                "alert_name": v.signal,
                "drift_class": v.dispatch_severity(),
                "scope_severity": v.severity,
                "agent_id": v.agent_id,
                "origin_agent": v.origin_agent,
                "source_uri": v.source_uri,
                "detail": v.detail,
                "srm_weight": v.srm_weight,
                "manifest_version": v.manifest_version,
                "detected_at": v.detected_at,
            })
    except Exception as exc:                            # noqa: BLE001
        if getattr(client, "debug", False):
            print(f"[AI Telemetry] Scope alert dispatch skipped: {exc}")


__all__ = ["evaluate_scope", "build_tag", "get_checker"]
