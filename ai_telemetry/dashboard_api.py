"""
Dashboard Data API — Module 6

Aggregates data from all SDK stores into structured JSON payloads
for each dashboard panel.  All functions are pure reads; they never
mutate store state and are safe to call from any thread.

Panels
------
  identity_overview()       Agent record table (Identity Overview)
  caller_split()            Human vs. Agent score distribution
  role_distribution()       Inferred role breakdown + confidence
  safety_timeline()         Rolling ZASP per agent + drift markers
  tool_scope_monitor()      Expected vs. observed tools + scope gaps
  drift_alert_feed()        Active alerts sorted by severity
  benchmark_coverage()      ZASP baseline vs. behavioral-only agents
  iam_enrichment_status()   Matched vs. pseudonymous agent counts
  agent_topology()          Orchestration graph (edges + nodes)

Alert channels (see alert_dispatcher.py):
  Webhook · SIEM CEF/LEEF · Email digest · PagerDuty / OpsGenie
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Severity ordering for alert feed sorting
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INVESTIGATE": 2, "MONITOR": 3, "STABLE": 4}

_DRIFT_RECOMMENDED_ACTION = {
    "SAFETY_SCORE_CRITICAL":     "Suspend agent; trigger incident response.",
    "SAFETY_SCORE_WARNING":      "Review recent interactions; notify security team.",
    "TOOL_EXPANSION_CRITICAL":   "Revoke HIGH-risk tool immediately; audit agent config.",
    "TOOL_EXPANSION_ALERT":      "Confirm tool addition is authorized; update baseline.",
    "PROMPT_VERSION_CRITICAL":   "Compare certified vs. current system prompt; recertify or rollback.",
    "PROMPT_VERSION_AUDIT":      "Log system prompt change; confirm authorization.",
    "REFUSAL_RATE_WARNING":      "Investigate why agent is producing more unsafe outputs.",
    "CROSS_AGENT_ANOMALY":       "Isolate agent; compare to cluster baseline.",
    "RESPONSE_LENGTH_SHIFT":     "Review response patterns; check for prompt injection.",
    "UNEXPLAINED_DRIFT":         "PRIORITY: behavior changed with no config change — likely adversarial.",
    "ROLE_SCOPE_GAP":            "Tool scope exceeds stated role; verify authorization chain.",
    "SYSTEM_PROMPT_VERSION_CHANGE": "Audit system prompt lineage; confirm authorized change.",
    "TOOL_SCOPE_EXPANSION":      "New tool detected; verify scope is authorized.",
}

_DEFAULT_ACTION = "Monitor; escalate if drift persists."


# ---------------------------------------------------------------------------
# Panel: Identity Overview
# ---------------------------------------------------------------------------

def identity_overview() -> List[Dict]:
    """Agent record table — one row per known agent."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store = get_fingerprint_store()
    drift_store = get_drift_store()
    rows = []
    for rec in fp_store.all_records():
        drift_b = drift_store.get_or_create(rec.agent_id)
        rows.append({
            "agent_id":               rec.agent_id,
            "display_name":           rec.display_name or "—",
            "agent_type":             rec.agent_type,
            "agent_role":             rec.agent_role or rec.inferred_role or "—",
            "orchestration_role":     rec.orchestration_role,
            "risk_level":             rec.risk_level or "—",
            "role_confidence":        round(rec.role_confidence, 3),
            "role_scope_gap":         rec.role_scope_gap,
            "first_seen_iso":         _ms_to_iso(rec.first_seen_ms),
            "last_seen_iso":          _ms_to_iso(rec.last_seen_ms),
            "call_count":             rec.call_count,
            "tool_scope":             list(rec.tool_scope),
            "tool_scope_size":        len(rec.tool_scope),
            "system_prompt_versions": len(rec.system_prompt_versions),
            "iam_identity":           rec.iam_identity,
            "drift_score":            round(drift_b.latest_drift_score, 4),
            "drift_classification":   drift_b.latest_drift_classification,
            "drift_alerts":           list(drift_b.latest_drift_signals),
            "baseline_active":        drift_b.full_baseline_active,
        })
    return sorted(rows, key=lambda r: r["drift_score"], reverse=True)


# ---------------------------------------------------------------------------
# Panel: Human vs. Agent Split
# ---------------------------------------------------------------------------

def caller_split(events: List[Dict]) -> Dict:
    """Distribution of entity types + caller_type_score histogram."""
    counts: Dict[str, int] = {"HUMAN": 0, "AUTOMATED": 0, "AGENT": 0,
                               "AMBIGUOUS": 0, "UNKNOWN": 0}
    score_buckets = [0] * 10  # 10 buckets: 0.0-0.1, 0.1-0.2, …

    llm_events = [e for e in events if e.get("event_type") == "llm_call"]
    for e in llm_events:
        et = e.get("entity_type", "UNKNOWN")
        counts[et] = counts.get(et, 0) + 1
        score = e.get("caller_type_score")
        if score is not None:
            try:
                bucket = min(9, int(float(score) * 10))
                score_buckets[bucket] += 1
            except (ValueError, TypeError):
                pass

    total = max(sum(counts.values()), 1)
    return {
        "counts": counts,
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
        "score_histogram": score_buckets,   # 10 buckets 0.0→1.0
        "total_llm_calls": total,
    }


# ---------------------------------------------------------------------------
# Panel: Role Distribution
# ---------------------------------------------------------------------------

def role_distribution(events: List[Dict]) -> Dict:
    """Inferred role breakdown — human and agent roles with confidence."""
    human_roles: Dict[str, List[float]] = {}
    agent_roles: Dict[str, List[float]] = {}

    for e in events:
        if e.get("human_inferred_role"):
            role = e["human_inferred_role"]
            conf = float(e.get("human_role_confidence") or 0)
            human_roles.setdefault(role, []).append(conf)

        if e.get("agent_role"):
            role = e["agent_role"]
            conf = float(e.get("role_confidence") or 0)
            agent_roles.setdefault(role, []).append(conf)

    def summarise(role_map: Dict[str, List[float]]) -> List[Dict]:
        out = []
        for role, confs in sorted(role_map.items(), key=lambda x: -len(x[1])):
            out.append({
                "role": role,
                "count": len(confs),
                "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
                "max_confidence": round(max(confs), 3) if confs else 0,
            })
        return out

    return {
        "human_roles": summarise(human_roles),
        "agent_roles": summarise(agent_roles),
    }


# ---------------------------------------------------------------------------
# Panel: Safety Score Timeline
# ---------------------------------------------------------------------------

def safety_timeline() -> List[Dict]:
    """Per-agent rolling ZASP mean, baseline band, and drift alert markers."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store = get_fingerprint_store()
    drift_store = get_drift_store()
    rows = []
    for rec in fp_store.all_records():
        b = drift_store.get_or_create(rec.agent_id)
        recent = b.recent_safety_scores or []
        baseline_mean = b.baseline_safety_mean

        rows.append({
            "agent_id":         rec.agent_id,
            "display_name":     rec.display_name or rec.agent_id,
            "agent_role":       rec.agent_role or "—",
            "current_mean":     round(sum(recent) / len(recent), 3) if recent else None,
            "baseline_mean":    round(baseline_mean, 3) if baseline_mean is not None else None,
            "baseline_std":     round(b.baseline_safety_std, 3) if b.baseline_safety_std else None,
            "recent_scores":    [round(s, 2) for s in recent[-20:]],  # last 20 for sparkline
            "drift_score":      b.latest_drift_score,
            "drift_class":      b.latest_drift_classification,
            "drift_alerts":     [a for a in b.latest_drift_signals if "SAFETY" in a or "UNEXPLAINED" in a],
            "baseline_active":  b.full_baseline_active,
            "call_count":       b.total_calls,
        })
    return sorted(rows, key=lambda r: r["drift_score"], reverse=True)


# ---------------------------------------------------------------------------
# Panel: Tool Scope Monitor
# ---------------------------------------------------------------------------

def tool_scope_monitor() -> List[Dict]:
    """Expected vs. observed tool scope per agent with gap detection."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store
    from ai_telemetry.role_inference import _AGENT_ROLE_CONFIGS

    fp_store = get_fingerprint_store()
    drift_store = get_drift_store()
    rows = []

    for rec in fp_store.all_records():
        role_cfg = _AGENT_ROLE_CONFIGS.get(rec.agent_role or "", {})
        expected = list(role_cfg.get("expected_tools", set()))
        forbidden = list(role_cfg.get("forbidden_tools", set()))
        observed = list(rec.tool_scope)
        b = drift_store.get_or_create(rec.agent_id)
        new_post_baseline = list(b.post_baseline_new_tools)

        # Scope gap: forbidden tools in observed scope
        gap_tools = [t for t in observed if t in forbidden]
        # Missing expected: expected tools not yet observed
        missing = [t for t in expected if t not in observed]

        rows.append({
            "agent_id":           rec.agent_id,
            "display_name":       rec.display_name or rec.agent_id,
            "agent_role":         rec.agent_role or "—",
            "risk_level":         rec.risk_level or "—",
            "observed_tools":     observed,
            "expected_tools":     expected,
            "forbidden_tools":    forbidden,
            "missing_expected":   missing,
            "gap_tools":          gap_tools,
            "role_scope_gap":     rec.role_scope_gap,
            "new_post_baseline":  new_post_baseline,
            "scope_gap_alerts":   [a for a in b.latest_drift_signals if "TOOL" in a],
        })
    return sorted(rows, key=lambda r: (r["role_scope_gap"], len(r["new_post_baseline"])), reverse=True)


# ---------------------------------------------------------------------------
# Panel: Drift Alert Feed
# ---------------------------------------------------------------------------

def drift_alert_feed(max_alerts: int = 100) -> List[Dict]:
    """Active drift alerts sorted by severity, newest first within tier."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store = get_fingerprint_store()
    drift_store = get_drift_store()
    feed: List[Dict] = []

    for rec in fp_store.all_records():
        b = drift_store.get_or_create(rec.agent_id)
        if not b.latest_drift_signals:
            continue
        for signal in b.latest_drift_signals:
            action = _DRIFT_RECOMMENDED_ACTION.get(signal.split(":")[0], _DEFAULT_ACTION)
            feed.append({
                "agent_id":         rec.agent_id,
                "display_name":     rec.display_name or rec.agent_id,
                "agent_role":       rec.agent_role or "—",
                "risk_level":       rec.risk_level or "—",
                "drift_score":      b.latest_drift_score,
                "drift_class":      b.latest_drift_classification,
                "primary_signal":   signal,
                "last_seen_iso":    _ms_to_iso(rec.last_seen_ms),
                "recommended_action": action,
                "iam_identity":     rec.iam_identity,
            })

    # Sort: severity tier first, then drift_score descending
    feed.sort(key=lambda x: (
        _SEVERITY_ORDER.get(x["drift_class"], 99),
        -x["drift_score"],
    ))
    return feed[:max_alerts]


# ---------------------------------------------------------------------------
# Panel: Benchmark Coverage
# ---------------------------------------------------------------------------

def benchmark_coverage() -> Dict:
    """ZASP red-team baseline vs. behavioral-only agents."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store = get_fingerprint_store()
    drift_store = get_drift_store()

    total = 0
    with_zasp_baseline = 0
    preliminary_only = 0
    no_baseline = 0
    by_role: Dict[str, Dict] = {}

    for rec in fp_store.all_records():
        total += 1
        b = drift_store.get_or_create(rec.agent_id)
        role = rec.agent_role or "UNKNOWN"
        by_role.setdefault(role, {"total": 0, "with_baseline": 0})
        by_role[role]["total"] += 1

        if b.full_baseline_active:
            with_zasp_baseline += 1
            by_role[role]["with_baseline"] += 1
        elif b.preliminary_active:
            preliminary_only += 1
        else:
            no_baseline += 1

    coverage_pct = round(with_zasp_baseline / max(total, 1) * 100, 1)
    return {
        "total_agents":        total,
        "with_zasp_baseline":  with_zasp_baseline,
        "preliminary_only":    preliminary_only,
        "no_baseline":         no_baseline,
        "coverage_pct":        coverage_pct,
        "by_role":             [
            {
                "role": role,
                "total": d["total"],
                "with_baseline": d["with_baseline"],
                "coverage_pct": round(d["with_baseline"] / max(d["total"], 1) * 100, 1),
            }
            for role, d in sorted(by_role.items())
        ],
    }


# ---------------------------------------------------------------------------
# Panel: IAM Enrichment Status
# ---------------------------------------------------------------------------

def iam_enrichment_status() -> Dict:
    """Agents matched to IAM identities vs. pseudonymous-only."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store

    fp_store = get_fingerprint_store()
    records = fp_store.all_records()

    matched = [r for r in records if r.iam_identity]
    pseudonymous = [r for r in records if not r.iam_identity]
    total = max(len(records), 1)

    return {
        "total_agents":      len(records),
        "iam_matched":       len(matched),
        "pseudonymous_only": len(pseudonymous),
        "enrichment_rate":   round(len(matched) / total * 100, 1),
        "matched_agents":    [
            {"agent_id": r.agent_id, "display_name": r.display_name, "iam_identity": r.iam_identity}
            for r in matched
        ],
        "iam_mode": "IAM-CONNECTED" if matched else "NO-IAM",
    }


# ---------------------------------------------------------------------------
# Panel: Agent Topology — Module 8 (full implementation)
# ---------------------------------------------------------------------------

def agent_topology(events: Optional[List[Dict]] = None) -> Dict:
    """Directed graph of agent relationships with authority scores.

    Delegates to topology_mapper (Module 8) for full kill-chain analysis.
    Pass pre-loaded event dicts to enable session-cooccurrence and
    API-key-sequence edge inference.
    """
    try:
        from ai_telemetry.topology_mapper import get_topology_risk_report
        return get_topology_risk_report(events=events)
    except ImportError:
        pass

    # Fallback: basic orchestration-only graph (pre-Module-8 behaviour)
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    fp_store = get_fingerprint_store()
    records  = fp_store.all_records()
    nodes    = []
    edges    = []
    orchestrators = [r for r in records if r.orchestration_role == "ORCHESTRATOR"]
    subagents     = [r for r in records if r.orchestration_role == "SUBAGENT"]
    for rec in records:
        nodes.append({
            "id": rec.agent_id, "label": rec.display_name or rec.agent_id[:12],
            "role": rec.orchestration_role, "agent_role": rec.agent_role or "—",
            "risk_level": rec.risk_level or "—", "drift_score": round(rec.drift_score, 3),
            "authority_score": 0.0,
        })
    for orch in orchestrators:
        for sub in subagents:
            edges.append({"source": orch.agent_id, "target": sub.agent_id, "inferred": True})
    return {"nodes": nodes, "edges": edges, "kill_chains": [], "escalated_alerts": [],
            "summary": {}, "note": "Fallback topology (Module 8 not loaded)."}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ms_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None
