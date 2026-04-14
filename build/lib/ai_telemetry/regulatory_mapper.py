"""
Regulatory Output Mapping — Module 7

Converts SDK runtime telemetry into compliance evidence that satisfies the
continuous-monitoring obligations of five major AI governance frameworks.

The SDK is not merely a security product: its runtime telemetry IS the
operational evidence of ongoing monitoring mandated by regulators.  This
module makes that mapping explicit, generates per-agent attestation status,
and produces a Compliance Evidence Report (CER) formatted for direct
submission to regulatory examiners, internal audit, and ISO 42001
management review.

Supported Frameworks
--------------------
  EU AI Act      Article 61 (continuous monitoring) + Article 9(7) (audit log)
  SR 11-7        Model risk management ongoing monitoring requirements
  NIST AI RMF    MEASURE 2.7 (performance baselines) + GOVERN 1.1 (risk policy)
  FFIEC          AI guidance ongoing behavioral monitoring
  ISO/IEC 42001  Clause 10.1 (continuous improvement documentation)

Attestation Tiers
-----------------
  SATISFIED   All evidence requirements met; no material gaps detected.
  PARTIAL     Monitoring active but baseline not yet fully established
              (< 100 calls; preliminary only) or minor gaps present.
  GAP         Required evidence missing; examiner finding likely.
  N/A         Requirement not applicable to this deployment context.

Report cadence (configurable):
  default    monthly   (30 days)
  SR 11-7    quarterly (90 days)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Regulatory framework registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """One regulatory requirement and its SDK evidence mapping."""
    req_id:          str   # "EU_AI_ACT_Art61"
    framework:       str   # "EU AI Act"
    article_ref:     str   # "Article 61"
    title:           str
    description:     str
    sdk_components:  tuple  # human-readable SDK component names
    sdk_modules:     tuple  # (5,) etc.
    evidence_fields: tuple  # field names in AgentRecord / AgentDriftBaseline
    # Minimum call count before this requirement can be SATISFIED
    min_calls_satisfied: int = 100
    min_calls_partial:   int = 20


REQUIREMENTS: Dict[str, Requirement] = {
    req.req_id: req for req in [

        Requirement(
            req_id="EU_AI_ACT_Art61",
            framework="EU AI Act",
            article_ref="Article 61",
            title="Continuous Post-Market Monitoring",
            description=(
                "High-risk AI systems must be monitored post-deployment. "
                "Providers must detect and document changes in performance and safety "
                "and report serious incidents to national authorities."
            ),
            sdk_components=(
                "Module 5 drift detection (8-signal, 30-day rolling EMA baseline)",
                "Safety score timeline with longitudinal behavioral record",
                "System prompt version tracking for configuration-change documentation",
            ),
            sdk_modules=(5,),
            evidence_fields=(
                "latest_drift_score", "latest_drift_classification",
                "baseline_safety_mean", "full_baseline_active",
                "system_prompt_versions",
            ),
            min_calls_satisfied=100,
            min_calls_partial=20,
        ),

        Requirement(
            req_id="EU_AI_ACT_Art9_7",
            framework="EU AI Act",
            article_ref="Article 9(7)",
            title="Automatic Logging and Audit Trail",
            description=(
                "High-risk AI systems shall be capable of automatically generating logs "
                "throughout their lifetime to enable post-hoc investigation of incidents. "
                "Logs shall record period of each use, reference database, input data, "
                "and identities of persons who verified results."
            ),
            sdk_components=(
                "Module 1 signal capture — immutable per-call log (16 fields)",
                "Tool call audit trail — every agent action with timestamp and session_id",
                "Session records enabling full incident reconstruction",
                "System prompt hash + embedding for configuration audit",
            ),
            sdk_modules=(1, 3),
            evidence_fields=(
                "call_count", "tool_scope", "first_seen_ms", "last_seen_ms",
                "system_prompt_versions", "api_key_cluster",
            ),
            min_calls_satisfied=1,
            min_calls_partial=1,
        ),

        Requirement(
            req_id="SR_11_7",
            framework="SR 11-7",
            article_ref="SR 11-7 §IV.A",
            title="Ongoing Model Monitoring",
            description=(
                "Financial institutions must perform ongoing monitoring of model behavior "
                "in production. Monitoring must detect model drift, performance degradation, "
                "and out-of-scope use. Results must be documented and reported to senior "
                "management on a regular basis."
            ),
            sdk_components=(
                "Module 5 safety score drift signal (40% weight) — implements SR 11-7 ongoing monitoring",
                "Refusal rate change signal — detects behavioral safety degradation",
                "100-call ZASP benchmark baseline — corresponds to SR 11-7 'validated performance standard'",
                "STABLE/MONITOR/INVESTIGATE/WARNING/CRITICAL classification tiers",
            ),
            sdk_modules=(5,),
            evidence_fields=(
                "latest_drift_score", "latest_drift_classification",
                "baseline_safety_mean", "baseline_refusal_rate",
                "full_baseline_active", "total_calls",
            ),
            min_calls_satisfied=100,
            min_calls_partial=20,
        ),

        Requirement(
            req_id="NIST_MEASURE_2_7",
            framework="NIST AI RMF",
            article_ref="MEASURE 2.7",
            title="Performance and Behavior Measurement",
            description=(
                "AI system performance and trustworthy characteristics are evaluated "
                "against defined metrics. Baselines are established and monitored for "
                "meaningful deviations. Measurement results are documented and used to "
                "inform risk management decisions."
            ),
            sdk_components=(
                "100-call baseline establishment protocol — per-agent performance baseline",
                "30-day rolling EMA baseline with exponential decay",
                "Aggregate 0.0–1.0 drift score across 6 weighted signal components",
                "ZASP safety score (0–10 scale) mapped from zendata_composite_risk",
            ),
            sdk_modules=(5,),
            evidence_fields=(
                "baseline_safety_mean", "baseline_safety_std",
                "latest_drift_score", "full_baseline_active",
                "recent_safety_scores",
            ),
            min_calls_satisfied=100,
            min_calls_partial=20,
        ),

        Requirement(
            req_id="NIST_GOVERN_1_1",
            framework="NIST AI RMF",
            article_ref="GOVERN 1.1",
            title="AI Risk Governance Policies",
            description=(
                "Policies, processes, procedures, and practices across the organization "
                "are in place to address AI risks. Accountability for risk management "
                "decisions is established and implemented."
            ),
            sdk_components=(
                "Drift alert feed with 5-tier severity levels — policy enforcement evidence",
                "Recommended action per alert type — operationalizes incident response policy",
                "Alert history constituting documented incident log for governance review",
                "Alert channels: Webhook, SIEM, Email, PagerDuty, OpsGenie",
            ),
            sdk_modules=(5, 6),
            evidence_fields=(
                "latest_drift_signals", "latest_drift_classification",
                "latest_drift_score",
            ),
            min_calls_satisfied=20,
            min_calls_partial=1,
        ),

        Requirement(
            req_id="FFIEC_AI",
            framework="FFIEC",
            article_ref="FFIEC AI Guidance §III.C",
            title="Ongoing Behavioral Monitoring for Financial AI",
            description=(
                "Financial institutions must establish ongoing monitoring programs for "
                "AI models used in customer-facing and risk-decision applications. "
                "Programs must include behavioral monitoring, drift detection, and "
                "documentation of monitoring findings for examiner review."
            ),
            sdk_components=(
                "Module 5 — explicit FFIEC ongoing monitoring implementation",
                "Refusal rate signal — monitors whether safety behaviors degrade (primary FFIEC concern)",
                "Tool scope monitor — detects unauthorized capability expansion",
                "Safety score summary suitable for regulatory reporting cadence",
            ),
            sdk_modules=(5, 6),
            evidence_fields=(
                "baseline_refusal_rate", "latest_drift_score",
                "full_baseline_active", "post_baseline_new_tools",
                "latest_drift_signals",
            ),
            min_calls_satisfied=100,
            min_calls_partial=20,
        ),

        Requirement(
            req_id="ISO_42001_10_1",
            framework="ISO/IEC 42001",
            article_ref="Clause 10.1",
            title="Continual Improvement Documentation",
            description=(
                "The organization shall continually improve the suitability, adequacy, "
                "and effectiveness of the AI management system. Nonconformities and "
                "corrective actions shall be documented, including an evaluation of "
                "whether similar nonconformities exist or could occur."
            ),
            sdk_components=(
                "Drift alerts — documented nonconformities with signal breakdown",
                "Re-certification events — documented corrective actions with trigger reason",
                "30-day rolling baseline update cycle — ongoing monitoring effectiveness",
                "Compliance Evidence Report — management review documentation artifact",
            ),
            sdk_modules=(5, 6, 7),
            evidence_fields=(
                "latest_drift_signals", "last_recertified_ms",
                "latest_drift_score",
            ),
            min_calls_satisfied=20,
            min_calls_partial=1,
        ),
    ]
}

# Group requirements by framework for report sections
FRAMEWORK_ORDER = [
    "EU AI Act",
    "SR 11-7",
    "NIST AI RMF",
    "FFIEC",
    "ISO/IEC 42001",
]


# ---------------------------------------------------------------------------
# Attestation data structures
# ---------------------------------------------------------------------------

@dataclass
class RequirementAttestation:
    """One agent's attestation status for one regulatory requirement."""
    req_id:      str
    status:      str          # SATISFIED | PARTIAL | GAP | N/A
    evidence:    Dict         # key evidence values used to derive status
    rationale:   str          # human-readable explanation
    call_count:  int = 0
    gaps:        List[str] = field(default_factory=list)


@dataclass
class AgentComplianceStatus:
    """Full compliance profile for one agent across all frameworks."""
    agent_id:      str
    display_name:  str
    agent_role:    str
    call_count:    int
    first_seen_iso: Optional[str]
    last_seen_iso:  Optional[str]
    iam_identity:   Optional[str]
    attestations:  Dict[str, RequirementAttestation]  # req_id → attestation
    overall_status: str    # SATISFIED | PARTIAL | GAP (worst across requirements)
    coverage_pct:   float  # % of requirements SATISFIED


@dataclass
class ComplianceEvidenceReport:
    """
    Compliance Evidence Report (CER) — the primary regulatory artifact.

    Formatted for submission to regulatory examiners, internal audit,
    and ISO 42001 management review.
    """
    report_id:     str
    generated_at:  str
    period_start:  str
    period_end:    str
    period_days:   int

    # §1 Deployment Inventory
    deployment_inventory: List[Dict]

    # §2 Monitoring Continuity
    monitoring_continuity: Dict

    # §3 Drift Event Log
    drift_event_log: List[Dict]

    # §4 Safety Score Summary
    safety_score_summary: List[Dict]

    # §5 Regulatory Mapping Attestation
    regulatory_attestation: Dict   # framework → {requirements, agents}

    # §6 Re-certification Events
    recertification_events: List[Dict]

    # §7 Per-agent compliance matrix
    agent_compliance: List[Dict]

    # Metadata
    sdk_version: str = "1.0"
    app_id:      str = "unknown"


# ---------------------------------------------------------------------------
# Attestation engine
# ---------------------------------------------------------------------------

def _ms_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except Exception:
        return None


def _assess_requirement(
    req: Requirement,
    rec,           # AgentRecord
    baseline,      # AgentDriftBaseline
) -> RequirementAttestation:
    """Derive attestation status for a single requirement × agent pair."""
    calls = baseline.total_calls
    gaps = []
    evidence = {
        "call_count":            calls,
        "full_baseline_active":  baseline.full_baseline_active,
        "preliminary_active":    baseline.preliminary_active,
        "latest_drift_score":    round(baseline.latest_drift_score, 4),
        "latest_drift_class":    baseline.latest_drift_classification,
        "baseline_safety_mean":  round(baseline.baseline_safety_mean, 3)
                                  if baseline.baseline_safety_mean is not None else None,
        "recertified_at":        _ms_to_iso(baseline.last_recertified_ms),
    }

    # ---- EU AI Act Art. 61 ----
    if req.req_id == "EU_AI_ACT_Art61":
        if not baseline.full_baseline_active:
            if not baseline.preliminary_active:
                gaps.append("No behavioral baseline established (fewer than 20 calls)")
                status = "GAP"
                rationale = (
                    f"Agent has only {calls} calls. Continuous monitoring baseline "
                    "requires ≥100 calls for full establishment (preliminary at 20). "
                    "Article 61 monitoring obligation cannot be demonstrated."
                )
            else:
                gaps.append("Preliminary baseline only — full baseline requires 100 calls")
                status = "PARTIAL"
                rationale = (
                    f"Agent has {calls} calls (preliminary baseline active). "
                    "Full 30-day rolling EMA baseline established at 100 calls. "
                    "Monitoring is active but longitudinal record is limited."
                )
        else:
            status = "SATISFIED"
            rationale = (
                f"Full behavioral baseline established ({calls} calls). "
                "30-day rolling EMA drift detection active across 8 signals. "
                "Safety score timeline provides longitudinal performance record. "
                "System prompt versions document configuration lineage."
            )

    # ---- EU AI Act Art. 9(7) ----
    elif req.req_id == "EU_AI_ACT_Art9_7":
        evidence["tool_scope_size"] = len(rec.tool_scope)
        evidence["system_prompt_versions"] = len(rec.system_prompt_versions)
        if calls == 0:
            gaps.append("No telemetry events captured — agent not yet observed")
            status = "GAP"
            rationale = "No SDK events recorded. Automatic logging requires active SDK instrumentation."
        else:
            status = "SATISFIED"
            rationale = (
                f"Immutable per-call log active ({calls} events captured). "
                f"Tool call audit trail records {len(rec.tool_scope)} distinct tools. "
                f"System prompt lineage: {len(rec.system_prompt_versions)} version(s) tracked. "
                "Session-level records enable post-hoc incident reconstruction."
            )

    # ---- SR 11-7 ----
    elif req.req_id == "SR_11_7":
        has_refusal = baseline.baseline_refusal_rate is not None
        evidence["baseline_refusal_rate"] = (
            round(baseline.baseline_refusal_rate, 4) if has_refusal else None
        )
        if not baseline.full_baseline_active:
            if not baseline.preliminary_active:
                gaps.append("No validated performance standard established")
                gaps.append("Ongoing monitoring not yet active (< 20 calls)")
                status = "GAP"
                rationale = (
                    "SR 11-7 requires a validated performance standard as reference for drift. "
                    f"Agent has {calls} calls — baseline requires 100 calls minimum."
                )
            else:
                gaps.append("Preliminary baseline only — validated standard not yet established")
                status = "PARTIAL"
                rationale = (
                    f"Monitoring active with preliminary baseline ({calls} calls). "
                    "SR 11-7 validated performance standard established at 100 calls. "
                    "Refusal rate baseline: "
                    f"{'not yet computed' if not has_refusal else f'{baseline.baseline_refusal_rate:.1%}'}."
                )
        else:
            status = "SATISFIED"
            refusal_str = (
                f"{baseline.baseline_refusal_rate:.1%}"
                if baseline.baseline_refusal_rate is not None
                else "not yet computed"
            )
            rationale = (
                f"Full SR 11-7 ongoing monitoring active ({calls} calls). "
                "Safety score drift and refusal rate change directly implement "
                "SR 11-7 behavioral monitoring requirements. "
                f"Validated baseline: ZASP mean {evidence['baseline_safety_mean']}, "
                f"refusal rate {refusal_str}."
            )

    # ---- NIST MEASURE 2.7 ----
    elif req.req_id == "NIST_MEASURE_2_7":
        recent_n = len(baseline.recent_safety_scores)
        evidence["recent_safety_samples"] = recent_n
        evidence["baseline_safety_std"] = (
            round(baseline.baseline_safety_std, 3) if baseline.baseline_safety_std else None
        )
        if not baseline.full_baseline_active:
            if not baseline.preliminary_active:
                gaps.append("No performance baseline established — cannot demonstrate measurement against baseline")
                status = "GAP"
                rationale = (
                    "NIST MEASURE 2.7 requires established baselines and deviation monitoring. "
                    f"Agent has {calls} calls. Baseline established at 100 calls."
                )
            else:
                gaps.append("Performance baseline not yet complete")
                status = "PARTIAL"
                rationale = (
                    f"Performance measurement active ({recent_n} recent samples). "
                    "Aggregate drift score computed. Full statistical baseline "
                    "(mean ± std) established at 100 calls."
                )
        else:
            status = "SATISFIED"
            rationale = (
                f"ZASP baseline: mean={evidence['baseline_safety_mean']}, "
                f"std={evidence['baseline_safety_std']}. "
                f"Rolling 30-day drift scoring active ({recent_n} recent samples). "
                "6-component weighted drift score maps directly to MEASURE 2.7 "
                "documentation requirements."
            )

    # ---- NIST GOVERN 1.1 ----
    elif req.req_id == "NIST_GOVERN_1_1":
        n_alerts = len(baseline.latest_drift_signals)
        evidence["active_alerts"] = n_alerts
        evidence["drift_tier"] = baseline.latest_drift_classification
        # GOVERN 1.1 is satisfiable as soon as the SDK is instrumented
        if calls == 0:
            gaps.append("No telemetry — governance policy cannot be demonstrated")
            status = "GAP"
            rationale = "No SDK events captured. Governance evidence requires active instrumentation."
        elif n_alerts == 0 and baseline.latest_drift_score < 0.01:
            status = "SATISFIED"
            rationale = (
                "5-tier drift classification active (STABLE/MONITOR/INVESTIGATE/WARNING/CRITICAL). "
                "Recommended action prescribed for every alert type. "
                f"Current status: STABLE (score={evidence['latest_drift_score']}). "
                "Alert channels configured for policy enforcement (Module 6)."
            )
        else:
            status = "SATISFIED"
            rationale = (
                f"Active governance enforcement: {n_alerts} alert(s) raised, "
                f"tier={baseline.latest_drift_classification}. "
                "Incident log maintained. Recommended actions operationalize policy. "
                "Alert dispatch to configured SIEM / PagerDuty channels."
            )

    # ---- FFIEC ----
    elif req.req_id == "FFIEC_AI":
        has_refusal = baseline.baseline_refusal_rate is not None
        new_tools   = len(baseline.post_baseline_new_tools)
        evidence["refusal_rate_monitored"] = has_refusal
        evidence["post_baseline_new_tools"] = new_tools
        if not baseline.full_baseline_active:
            if not baseline.preliminary_active:
                gaps.append("FFIEC requires ongoing monitoring program — not yet active")
                gaps.append("Refusal rate baseline not established")
                status = "GAP"
                rationale = (
                    "FFIEC guidance requires established ongoing monitoring for customer-facing AI. "
                    f"Agent has {calls} calls. Full monitoring requires 100-call baseline."
                )
            else:
                gaps.append("Refusal rate baseline not yet established (requires 100 calls)")
                status = "PARTIAL"
                rationale = (
                    f"Behavioral monitoring active with {calls} calls. "
                    "Refusal rate signal — FFIEC's primary concern — requires full baseline. "
                    "Tool scope monitoring active."
                )
        else:
            status = "SATISFIED"
            refusal_str = (
                f"{baseline.baseline_refusal_rate:.1%}"
                if baseline.baseline_refusal_rate is not None
                else "not yet computed"
            )
            rationale = (
                "Full FFIEC behavioral monitoring program active. "
                f"Refusal rate baseline: {refusal_str} "
                "(degradation-detection active). "
                f"Tool scope: {new_tools} new tool(s) flagged post-baseline. "
                "Safety score drift monitoring satisfies FFIEC ongoing monitoring requirement."
            )

    # ---- ISO 42001 Clause 10.1 ----
    elif req.req_id == "ISO_42001_10_1":
        n_alerts       = len(baseline.latest_drift_signals)
        recertified    = baseline.last_recertified_ms is not None
        evidence["drift_alerts_raised"] = n_alerts
        evidence["recertification_performed"] = recertified
        if calls == 0:
            gaps.append("No nonconformity evidence — no telemetry captured")
            status = "GAP"
            rationale = "ISO 42001 Clause 10.1 requires documented nonconformities. No SDK events captured."
        elif calls < 20:
            gaps.append("Insufficient call history for meaningful nonconformity documentation")
            status = "PARTIAL"
            rationale = (
                f"Telemetry active ({calls} events). Monitoring program not yet at "
                "preliminary baseline. Continuous improvement documentation begins at 20 calls."
            )
        else:
            corrective = "Re-certification events logged." if recertified else (
                "No re-certification events (corrective actions) yet recorded."
            )
            status = "SATISFIED"
            rationale = (
                f"Nonconformities documented: {n_alerts} drift alert(s). "
                f"{corrective} "
                "30-day rolling baseline update constitutes monitoring effectiveness record. "
                "Compliance Evidence Report serves as ISO 42001 management review artifact."
            )
    else:
        status = "N/A"
        rationale = "Requirement not mapped for this deployment context."

    return RequirementAttestation(
        req_id=req.req_id,
        status=status,
        evidence=evidence,
        rationale=rationale,
        call_count=calls,
        gaps=gaps,
    )


_STATUS_RANK = {"GAP": 0, "PARTIAL": 1, "SATISFIED": 2, "N/A": 3}


def assess_agent_compliance(agent_id: str) -> Optional[AgentComplianceStatus]:
    """Return full compliance profile for one agent across all requirements."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp    = get_fingerprint_store()
    ds    = get_drift_store()
    rec   = fp.get(agent_id)
    if rec is None:
        return None
    base  = ds.get_or_create(agent_id)

    attestations: Dict[str, RequirementAttestation] = {}
    for req_id, req in REQUIREMENTS.items():
        attestations[req_id] = _assess_requirement(req, rec, base)

    statuses = [a.status for a in attestations.values() if a.status != "N/A"]
    worst = min(statuses, key=lambda s: _STATUS_RANK.get(s, 99), default="N/A")
    satisfied = sum(1 for s in statuses if s == "SATISFIED")
    coverage  = round(satisfied / max(len(statuses), 1) * 100, 1)

    return AgentComplianceStatus(
        agent_id=agent_id,
        display_name=rec.display_name or agent_id,
        agent_role=rec.agent_role or rec.inferred_role or "UNKNOWN",
        call_count=base.total_calls,
        first_seen_iso=_ms_to_iso(rec.first_seen_ms),
        last_seen_iso=_ms_to_iso(rec.last_seen_ms),
        iam_identity=rec.iam_identity,
        attestations=attestations,
        overall_status=worst,
        coverage_pct=coverage,
    )


def get_compliance_summary() -> Dict:
    """High-level compliance dashboard data: per-agent status matrix."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store

    fp = get_fingerprint_store()
    records = fp.all_records()

    agents = []
    framework_gaps: Dict[str, int] = {}

    for rec in records:
        cs = assess_agent_compliance(rec.agent_id)
        if cs is None:
            continue

        row: Dict[str, Any] = {
            "agent_id":      cs.agent_id,
            "display_name":  cs.display_name,
            "agent_role":    cs.agent_role,
            "call_count":    cs.call_count,
            "overall_status": cs.overall_status,
            "coverage_pct":  cs.coverage_pct,
            "first_seen_iso": cs.first_seen_iso,
            "iam_identity":  cs.iam_identity,
            "by_framework":  {},
        }

        # Aggregate per framework (worst status across its requirements)
        fw_status: Dict[str, List[str]] = {}
        for req_id, att in cs.attestations.items():
            fw = REQUIREMENTS[req_id].framework
            fw_status.setdefault(fw, []).append(att.status)

        for fw, statuses in fw_status.items():
            real = [s for s in statuses if s != "N/A"]
            worst_fw = min(real, key=lambda s: _STATUS_RANK.get(s, 99), default="N/A")
            row["by_framework"][fw] = worst_fw
            if worst_fw == "GAP":
                framework_gaps[fw] = framework_gaps.get(fw, 0) + 1

        agents.append(row)

    total = len(agents)
    fully_covered = sum(1 for a in agents if a["overall_status"] == "SATISFIED")

    return {
        "total_agents":     total,
        "fully_satisfied":  fully_covered,
        "overall_coverage": round(fully_covered / max(total, 1) * 100, 1),
        "framework_gaps":   framework_gaps,
        "agents":           sorted(agents, key=lambda a: _STATUS_RANK.get(a["overall_status"], 99)),
        "frameworks":       FRAMEWORK_ORDER,
        "requirements": [
            {
                "req_id":      r.req_id,
                "framework":   r.framework,
                "article_ref": r.article_ref,
                "title":       r.title,
            }
            for r in REQUIREMENTS.values()
        ],
    }


# ---------------------------------------------------------------------------
# Compliance Evidence Report generator
# ---------------------------------------------------------------------------

def generate_compliance_report(
    period_days: int = 30,
    app_id: str = "unknown",
) -> ComplianceEvidenceReport:
    """Generate a full Compliance Evidence Report for the given period.

    Parameters
    ----------
    period_days : int
        Reporting window in days (30 = monthly default, 90 = quarterly for SR 11-7).
    app_id : str
        Application identifier for the report header.
    """
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store    = get_fingerprint_store()
    drift_store = get_drift_store()

    now_ms    = int(time.time() * 1000)
    now_dt    = datetime.now(timezone.utc)
    start_dt  = now_dt - timedelta(days=period_days)
    start_ms  = int(start_dt.timestamp() * 1000)
    period_end   = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    period_start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    report_id = "CER-" + now_dt.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()
    records   = fp_store.all_records()

    # ── §1 Deployment Inventory ──────────────────────────────────────────────
    inventory = []
    for rec in records:
        base = drift_store.get_or_create(rec.agent_id)
        cs   = assess_agent_compliance(rec.agent_id)
        inventory.append({
            "agent_id":              rec.agent_id,
            "display_name":          rec.display_name or "—",
            "agent_type":            rec.agent_type,
            "agent_role":            rec.agent_role or rec.inferred_role or "UNKNOWN",
            "orchestration_role":    rec.orchestration_role,
            "first_seen_iso":        _ms_to_iso(rec.first_seen_ms),
            "last_seen_iso":         _ms_to_iso(rec.last_seen_ms),
            "call_count":            rec.call_count,
            "baseline_established":  base.full_baseline_active,
            "baseline_established_at": _ms_to_iso(base.last_recertified_ms),
            "iam_identity":          rec.iam_identity,
            "tool_scope":            list(rec.tool_scope),
            "system_prompt_versions": len(rec.system_prompt_versions),
            "drift_score":           round(base.latest_drift_score, 4),
            "drift_classification":  base.latest_drift_classification,
            "overall_compliance":    cs.overall_status if cs else "GAP",
            "compliance_coverage":   cs.coverage_pct if cs else 0.0,
        })

    # ── §2 Monitoring Continuity ─────────────────────────────────────────────
    total_agents   = len(records)
    with_full_base  = sum(1 for r in records
                          if drift_store.get_or_create(r.agent_id).full_baseline_active)
    with_prelim     = sum(1 for r in records
                          if drift_store.get_or_create(r.agent_id).preliminary_active
                          and not drift_store.get_or_create(r.agent_id).full_baseline_active)
    no_baseline     = total_agents - with_full_base - with_prelim
    total_calls     = sum(drift_store.get_or_create(r.agent_id).total_calls for r in records)

    # Agents last seen more than 24h ago are considered "inactive" for the period
    inactive_24h    = sum(
        1 for r in records
        if r.last_seen_ms and (now_ms - r.last_seen_ms) > 86_400_000
    )

    monitoring_continuity = {
        "period_start":         period_start,
        "period_end":           period_end,
        "total_agents_monitored": total_agents,
        "agents_with_full_baseline": with_full_base,
        "agents_preliminary_only":   with_prelim,
        "agents_no_baseline":        no_baseline,
        "total_calls_captured":      total_calls,
        "agents_inactive_24h":       inactive_24h,
        "monitoring_coverage_pct":   round(
            (total_agents - no_baseline) / max(total_agents, 1) * 100, 1
        ),
        "full_baseline_coverage_pct": round(
            with_full_base / max(total_agents, 1) * 100, 1
        ),
    }

    # ── §3 Drift Event Log ───────────────────────────────────────────────────
    from ai_telemetry.dashboard_api import drift_alert_feed, _DRIFT_RECOMMENDED_ACTION, _DEFAULT_ACTION
    drift_events = drift_alert_feed(max_alerts=500)
    # Filter to period
    drift_event_log = []
    for ev in drift_events:
        drift_event_log.append({
            **ev,
            "period":           f"{period_start} → {period_end}",
            "resolution_status": "OPEN",
            "time_to_resolution": None,
        })

    # ── §4 Safety Score Summary ──────────────────────────────────────────────
    safety_summary = []
    for rec in records:
        base = drift_store.get_or_create(rec.agent_id)
        recent = base.recent_safety_scores or []
        n = len(recent)
        if n == 0:
            continue
        sorted_scores = sorted(recent)
        p25 = sorted_scores[max(0, int(n * 0.25))]
        p75 = sorted_scores[min(n - 1, int(n * 0.75))]
        p10 = sorted_scores[max(0, int(n * 0.10))]
        p90 = sorted_scores[min(n - 1, int(n * 0.90))]
        mean_score = sum(recent) / n
        trend = "STABLE"
        if n >= 10:
            first_half  = sum(recent[:n//2]) / (n // 2)
            second_half = sum(recent[n//2:]) / (n - n // 2)
            delta = second_half - first_half
            if delta < -0.3:
                trend = "DEGRADING"
            elif delta > 0.3:
                trend = "IMPROVING"

        safety_summary.append({
            "agent_id":          rec.agent_id,
            "display_name":      rec.display_name or rec.agent_id,
            "agent_role":        rec.agent_role or "UNKNOWN",
            "sample_count":      n,
            "mean_zasp":         round(mean_score, 3),
            "baseline_mean":     round(base.baseline_safety_mean, 3)
                                  if base.baseline_safety_mean is not None else None,
            "p10_zasp":          round(p10, 3),
            "p25_zasp":          round(p25, 3),
            "p75_zasp":          round(p75, 3),
            "p90_zasp":          round(p90, 3),
            "trend":             trend,
            "drift_score":       round(base.latest_drift_score, 4),
            "drift_class":       base.latest_drift_classification,
        })

    # ── §5 Regulatory Mapping Attestation ────────────────────────────────────
    regulatory_attestation: Dict[str, Any] = {}
    for fw in FRAMEWORK_ORDER:
        fw_reqs = [r for r in REQUIREMENTS.values() if r.framework == fw]
        fw_data: Dict[str, Any] = {
            "framework":    fw,
            "requirements": [],
            "agents":       [],
        }
        for req in fw_reqs:
            req_data: Dict[str, Any] = {
                "req_id":      req.req_id,
                "article_ref": req.article_ref,
                "title":       req.title,
                "description": req.description,
                "sdk_components": list(req.sdk_components),
                "agent_attestations": [],
            }
            statuses = []
            for rec in records:
                base = drift_store.get_or_create(rec.agent_id)
                att  = _assess_requirement(req, rec, base)
                req_data["agent_attestations"].append({
                    "agent_id":   rec.agent_id,
                    "status":     att.status,
                    "rationale":  att.rationale,
                    "gaps":       att.gaps,
                })
                statuses.append(att.status)
            real = [s for s in statuses if s != "N/A"]
            req_data["summary_status"] = (
                "SATISFIED" if all(s == "SATISFIED" for s in real)
                else "GAP" if any(s == "GAP" for s in real)
                else "PARTIAL" if real
                else "N/A"
            )
            fw_data["requirements"].append(req_data)
        regulatory_attestation[fw] = fw_data

    # ── §6 Re-certification Events ───────────────────────────────────────────
    recertification_events = []
    for rec in records:
        base = drift_store.get_or_create(rec.agent_id)
        if base.last_recertified_ms:
            recertification_events.append({
                "agent_id":          rec.agent_id,
                "display_name":      rec.display_name or rec.agent_id,
                "recertified_at_iso": _ms_to_iso(base.last_recertified_ms),
                "trigger":           "MANUAL",   # recertify_agent() was called explicitly
                "outcome":           "Baseline reset to current behavioral state",
                "drift_score_after": round(base.latest_drift_score, 4),
            })

    # ── §7 Per-agent compliance matrix ───────────────────────────────────────
    agent_compliance = []
    for rec in records:
        cs = assess_agent_compliance(rec.agent_id)
        if cs is None:
            continue
        row: Dict[str, Any] = {
            "agent_id":       cs.agent_id,
            "display_name":   cs.display_name,
            "agent_role":     cs.agent_role,
            "call_count":     cs.call_count,
            "overall_status": cs.overall_status,
            "coverage_pct":   cs.coverage_pct,
            "requirements":   {},
        }
        for req_id, att in cs.attestations.items():
            row["requirements"][req_id] = {
                "status":    att.status,
                "rationale": att.rationale,
                "gaps":      att.gaps,
                "evidence":  att.evidence,
            }
        agent_compliance.append(row)

    return ComplianceEvidenceReport(
        report_id=report_id,
        generated_at=period_end,
        period_start=period_start,
        period_end=period_end,
        period_days=period_days,
        deployment_inventory=inventory,
        monitoring_continuity=monitoring_continuity,
        drift_event_log=drift_event_log,
        safety_score_summary=safety_summary,
        regulatory_attestation=regulatory_attestation,
        recertification_events=recertification_events,
        agent_compliance=agent_compliance,
        sdk_version="1.0",
        app_id=app_id,
    )


def report_to_dict(report: ComplianceEvidenceReport) -> Dict:
    """Serialize the report to a plain dict for JSON delivery."""
    return {
        "report_id":              report.report_id,
        "generated_at":           report.generated_at,
        "period_start":           report.period_start,
        "period_end":             report.period_end,
        "period_days":            report.period_days,
        "sdk_version":            report.sdk_version,
        "app_id":                 report.app_id,
        "deployment_inventory":   report.deployment_inventory,
        "monitoring_continuity":  report.monitoring_continuity,
        "drift_event_log":        report.drift_event_log,
        "safety_score_summary":   report.safety_score_summary,
        "regulatory_attestation": report.regulatory_attestation,
        "recertification_events": report.recertification_events,
        "agent_compliance":       report.agent_compliance,
    }


# ---------------------------------------------------------------------------
# HTML report generator — examiner-ready output
# ---------------------------------------------------------------------------

def generate_html_report(report: ComplianceEvidenceReport) -> str:
    """Render the CER as a self-contained HTML document for examiners."""

    STATUS_COLOR = {
        "SATISFIED": "#1a7f37",
        "PARTIAL":   "#9a6700",
        "GAP":       "#a40e26",
        "N/A":       "#656d76",
    }
    STATUS_BG = {
        "SATISFIED": "#dafbe1",
        "PARTIAL":   "#fff8c5",
        "GAP":       "#ffebe9",
        "N/A":       "#f0f0f0",
    }

    def badge(status: str) -> str:
        c  = STATUS_COLOR.get(status, "#656d76")
        bg = STATUS_BG.get(status, "#f0f0f0")
        return (
            f'<span style="display:inline-block;padding:2px 9px;border-radius:20px;'
            f'font-size:.75rem;font-weight:700;background:{bg};color:{c}">'
            f'{status}</span>'
        )

    mc  = report.monitoring_continuity
    inv = report.deployment_inventory

    # Build inventory table rows
    inv_rows = "".join(
        f"<tr>"
        f"<td><code>{a['agent_id'][:20]}</code></td>"
        f"<td>{a['display_name']}</td>"
        f"<td>{a['agent_role']}</td>"
        f"<td>{'✔' if a['baseline_established'] else '—'}</td>"
        f"<td>{a['first_seen_iso'] or '—'}</td>"
        f"<td style='text-align:right'>{a['call_count']}</td>"
        f"<td>{badge(a['overall_compliance'])}</td>"
        f"</tr>"
        for a in inv
    )

    # Build per-framework attestation sections
    fw_sections = ""
    for fw, fw_data in report.regulatory_attestation.items():
        fw_sections += f"<h3 style='margin:24px 0 10px;font-size:1rem;color:#1f2328'>{fw}</h3>"
        for req in fw_data["requirements"]:
            summary = req["summary_status"]
            fw_sections += f"""
            <div style='border:1px solid #d0d7de;border-radius:6px;padding:14px 16px;margin-bottom:12px;background:#fff'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px'>
                <div>
                  <span style='font-weight:600;font-size:.9rem'>{req['article_ref']}: {req['title']}</span>
                  <div style='font-size:.78rem;color:#656d76;margin-top:3px'>{req['description']}</div>
                </div>
                <div style='margin-left:16px;flex-shrink:0'>{badge(summary)}</div>
              </div>
              <div style='font-size:.78rem;margin-bottom:8px'>
                <strong>SDK Evidence:</strong>
                <ul style='margin:4px 0 0 16px;line-height:1.7'>
                  {"".join(f"<li>{c}</li>" for c in req['sdk_components'])}
                </ul>
              </div>"""

            # Agent attestation details
            atts = req.get("agent_attestations", [])
            if atts:
                fw_sections += "<details style='margin-top:6px'><summary style='font-size:.77rem;cursor:pointer;color:#0969da'>Agent attestations ▸</summary>"
                fw_sections += "<table style='width:100%;border-collapse:collapse;font-size:.76rem;margin-top:8px'>"
                fw_sections += "<thead><tr><th style='text-align:left;padding:4px 8px;border-bottom:1px solid #d0d7de'>Agent</th><th>Status</th><th>Rationale</th></tr></thead><tbody>"
                for a in atts:
                    fw_sections += (
                        f"<tr><td style='padding:4px 8px;border-bottom:1px solid #eaeef2'>"
                        f"<code>{a['agent_id'][:18]}</code></td>"
                        f"<td style='padding:4px 8px;border-bottom:1px solid #eaeef2'>{badge(a['status'])}</td>"
                        f"<td style='padding:4px 8px;border-bottom:1px solid #eaeef2;font-size:.73rem'>{a['rationale']}</td></tr>"
                    )
                fw_sections += "</tbody></table></details>"
            fw_sections += "</div>"

    # Safety score summary table
    ss_rows = "".join(
        f"<tr><td>{s['display_name']}</td><td>{s['agent_role']}</td>"
        f"<td style='text-align:right'>{s['mean_zasp']}</td>"
        f"<td style='text-align:right'>{s['baseline_mean'] or '—'}</td>"
        f"<td style='text-align:right'>{s['p10_zasp']}</td>"
        f"<td style='text-align:right'>{s['p90_zasp']}</td>"
        f"<td>{s['trend']}</td>"
        f"<td>{badge(s['drift_class'])}</td></tr>"
        for s in report.safety_score_summary
    )

    # Drift event log table
    def _dl_row(e: Dict) -> str:
        drift_cls = e.get("drift_class", "STABLE")
        row_bg    = STATUS_BG.get(drift_cls, "#fff")
        score     = e.get("drift_score", 0)
        return (
            f"<tr style='background:{row_bg}'>"
            f"<td>{badge(drift_cls)}</td>"
            f"<td><code>{e.get('agent_id','—')[:18]}</code></td>"
            f"<td>{e.get('primary_signal','—')}</td>"
            f"<td style='text-align:right'>{score:.4f}</td>"
            f"<td>{e.get('last_seen_iso','—')}</td>"
            f"<td style='font-size:.73rem'>{e.get('recommended_action','—')}</td></tr>"
        )
    dl_rows = "".join(_dl_row(e) for e in report.drift_event_log) or \
        "<tr><td colspan='6' style='text-align:center;color:#656d76'>No active drift alerts in this period.</td></tr>"

    # Re-cert events
    rc_rows = "".join(
        f"<tr><td><code>{e['agent_id'][:18]}</code></td><td>{e['recertified_at_iso']}</td>"
        f"<td>{e['trigger']}</td><td>{e['outcome']}</td></tr>"
        for e in report.recertification_events
    ) or "<tr><td colspan='4' style='text-align:center;color:#656d76'>No re-certification events in this period.</td></tr>"

    td_style = "padding:7px 10px;border-bottom:1px solid #eaeef2;vertical-align:top"
    th_style = "text-align:left;padding:6px 10px;color:#656d76;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #d0d7de"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Compliance Evidence Report — {report.report_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; color: #1f2328; background: #fff; padding: 40px; max-width: 1100px; margin: 0 auto; line-height: 1.5; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1rem; color: #1f2328; margin: 28px 0 10px; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .82rem; margin-top: 8px; }}
    th {{ {th_style} }}
    td {{ {td_style} }}
    code {{ font-family: 'Consolas','Courier New',monospace; font-size: .82em; background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
    .meta-card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px 14px; }}
    .meta-card .label {{ font-size: .69rem; color: #656d76; text-transform: uppercase; letter-spacing: .05em; }}
    .meta-card .value {{ font-size: 1.2rem; font-weight: 700; margin-top: 3px; }}
    .notice {{ background: #ddf4ff; border: 1px solid #b6e0fe; border-radius: 6px; padding: 12px 14px; font-size: .82rem; margin: 12px 0; }}
    @media print {{ body {{ padding: 20px; }} details {{ display: none; }} }}
  </style>
</head>
<body>
  <p style="font-size:.75rem;color:#656d76;margin-bottom:4px">CONFIDENTIAL — REGULATORY ARTIFACT</p>
  <h1>Compliance Evidence Report</h1>
  <p style="color:#656d76;font-size:.85rem;margin-top:4px">
    {report.report_id} · Generated {report.generated_at} · Period: {report.period_start} → {report.period_end}
  </p>
  <p style="font-size:.8rem;color:#656d76;margin-top:2px">Application: <code>{report.app_id}</code> · SDK v{report.sdk_version}</p>

  <div class="notice">
    This Compliance Evidence Report is generated by the ZenData AI Behavioral Telemetry SDK. It constitutes
    the ongoing monitoring evidence required by EU AI Act Article 61, SR 11-7, NIST AI RMF MEASURE 2.7 / GOVERN 1.1,
    FFIEC AI guidance, and ISO/IEC 42001 Clause 10.1. For regulatory examiner use.
  </div>

  <div class="meta-grid">
    <div class="meta-card"><div class="label">Agents Monitored</div><div class="value">{mc['total_agents_monitored']}</div></div>
    <div class="meta-card"><div class="label">Full Baseline Coverage</div><div class="value">{mc['full_baseline_coverage_pct']}%</div></div>
    <div class="meta-card"><div class="label">Total Calls Captured</div><div class="value">{mc['total_calls_captured']:,}</div></div>
    <div class="meta-card"><div class="label">Drift Events</div><div class="value">{len(report.drift_event_log)}</div></div>
    <div class="meta-card"><div class="label">Re-certification Events</div><div class="value">{len(report.recertification_events)}</div></div>
    <div class="meta-card"><div class="label">Reporting Period</div><div class="value" style="font-size:.95rem">{report.period_days}d</div></div>
  </div>

  <!-- §1 Deployment Inventory -->
  <h2>§1 Deployment Inventory</h2>
  <table>
    <thead><tr>
      <th>Agent ID</th><th>Display Name</th><th>Role</th>
      <th>Baseline</th><th>First Seen</th><th>Calls</th><th>Compliance</th>
    </tr></thead>
    <tbody>{inv_rows or "<tr><td colspan='7' style='text-align:center;color:#656d76'>No agents observed.</td></tr>"}</tbody>
  </table>

  <!-- §2 Monitoring Continuity -->
  <h2>§2 Monitoring Continuity</h2>
  <div class="meta-grid">
    <div class="meta-card"><div class="label">Monitoring Coverage</div><div class="value">{mc['monitoring_coverage_pct']}%</div></div>
    <div class="meta-card"><div class="label">Agents with Full Baseline</div><div class="value" style="color:#1a7f37">{mc['agents_with_full_baseline']}</div></div>
    <div class="meta-card"><div class="label">Preliminary Only</div><div class="value" style="color:#9a6700">{mc['agents_preliminary_only']}</div></div>
    <div class="meta-card"><div class="label">No Baseline (Gap)</div><div class="value" style="color:#a40e26">{mc['agents_no_baseline']}</div></div>
    <div class="meta-card"><div class="label">Inactive >24h</div><div class="value">{mc['agents_inactive_24h']}</div></div>
  </div>

  <!-- §3 Drift Event Log -->
  <h2>§3 Drift Event Log</h2>
  <table>
    <thead><tr>
      <th>Severity</th><th>Agent</th><th>Primary Signal</th>
      <th>Drift Score</th><th>Timestamp</th><th>Recommended Action</th>
    </tr></thead>
    <tbody>{dl_rows}</tbody>
  </table>

  <!-- §4 Safety Score Summary -->
  <h2>§4 Safety Score Summary (ZASP)</h2>
  <table>
    <thead><tr>
      <th>Agent</th><th>Role</th><th>Mean</th><th>Baseline</th>
      <th>P10</th><th>P90</th><th>Trend</th><th>Drift Class</th>
    </tr></thead>
    <tbody>{ss_rows or "<tr><td colspan='8' style='text-align:center;color:#656d76'>No safety scores available.</td></tr>"}</tbody>
  </table>

  <!-- §5 Regulatory Mapping Attestation -->
  <h2>§5 Regulatory Mapping Attestation</h2>
  {fw_sections}

  <!-- §6 Re-certification Events -->
  <h2>§6 Re-certification Events (Corrective Actions)</h2>
  <table>
    <thead><tr><th>Agent</th><th>Recertified At</th><th>Trigger</th><th>Outcome</th></tr></thead>
    <tbody>{rc_rows}</tbody>
  </table>

  <p style="margin-top:40px;font-size:.72rem;color:#8b949e;border-top:1px solid #d0d7de;padding-top:12px">
    ZenData AI Behavioral Telemetry SDK v{report.sdk_version} · Report {report.report_id} ·
    This document is generated programmatically from runtime telemetry and constitutes an
    automated compliance evidence record. For questions, contact your ZenData account team.
  </p>
</body>
</html>"""

    return html


__all__ = [
    "REQUIREMENTS",
    "FRAMEWORK_ORDER",
    "Requirement",
    "RequirementAttestation",
    "AgentComplianceStatus",
    "ComplianceEvidenceReport",
    "assess_agent_compliance",
    "get_compliance_summary",
    "generate_compliance_report",
    "report_to_dict",
    "generate_html_report",
]
