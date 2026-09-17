"""
Behavioral Drift Detection — Module 5

Computes a continuous aggregate drift score (0.0–1.0) per agent from six
weighted signal components.  Activates at 20 calls (preliminary baseline),
reaches full accuracy at 100 calls.  Baseline evolves on a 30-day rolling
window with exponential decay; a re-certification event hard-resets it.

Drift Signal Taxonomy (spec §Module 5)
---------------------------------------
  Safety score drift      40%  Rolling mean ZASP score vs. established baseline
  Tool scope expansion    25%  New / HIGH-risk tools not present at baseline
  System prompt version   15%  Hash diverges from certified version
  Refusal rate change     10%  Rate of unsafe responses (score ≤ 2.0) vs. baseline
  Cross-agent anomaly      5%  Deviation from behavioral cluster of similar agents
  Other signals            5%  Response length, prompt pattern, temporal shift

Aggregate drift classifications
--------------------------------
  STABLE       0.00 – 0.20
  MONITOR      0.20 – 0.40
  INVESTIGATE  0.40 – 0.60
  WARNING      0.60 – 0.80
  CRITICAL     0.80 – 1.00

The highest-value alert: drift_score ≥ 0.4 without a corresponding system
prompt change → adversarial manipulation, unexpected model update, or context
window poisoning most likely.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THIRTY_DAYS_MS: int = 30 * 24 * 60 * 60 * 1000
_DECAY_TIME_CONSTANT_DAYS: float = 30.0   # exponential decay constant (τ)
_PRELIMINARY_THRESHOLD: int = 20          # calls before preliminary baseline
_FULL_THRESHOLD: int = 100                # calls before full baseline
_RECENT_WINDOW: int = 50                  # rolling window for recent ZASP mean
_REFUSAL_SAFETY_THRESHOLD: float = 2.0   # ZASP score ≤ this → unsafe / refusal

# Signal component weights (must sum to 1.0)
_WEIGHTS: Dict[str, float] = {
    "safety_score":     0.40,
    "tool_expansion":   0.25,
    "prompt_version":   0.15,
    "refusal_rate":     0.10,
    "cross_agent":      0.05,
    "other":            0.05,
}

# 5-tier classification thresholds
_DRIFT_TIERS = [
    (0.20, "STABLE"),
    (0.40, "MONITOR"),
    (0.60, "INVESTIGATE"),
    (0.80, "WARNING"),
    (1.01, "CRITICAL"),
]

# HIGH-risk tool names — any new appearance triggers CRITICAL regardless of role
_HIGH_RISK_TOOLS: Set[str] = {
    "transfer_funds", "execute_trade", "place_order",
    "bash", "code_interpreter", "run_script", "python_repl",
    "delete_record", "drop_table", "truncate_table",
    "revoke_access", "grant_permission", "create_user", "modify_acl",
    "approve_invoice", "process_payment",
    "send_email", "send_sms", "send_message",
    "exfiltrate_data", "write_file", "upload_file",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TimedSample:
    """One timestamped measurement used for rolling-window and decay computation."""
    timestamp_ms: int
    value: float


@dataclass
class AgentDriftBaseline:
    """All drift tracking state for a single agent (keyed by agent_id)."""
    agent_id: str

    # ---- Lifecycle ----
    total_calls: int = 0
    last_sample_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    preliminary_active: bool = False    # True once 20 calls reached
    full_baseline_active: bool = False  # True once 100 calls reached

    # ---- Safety score (ZASP, 0–10, higher = safer) ----
    safety_samples: List[TimedSample] = field(default_factory=list)
    # Baseline EMA (decays toward new observations; established at call 100)
    baseline_safety_mean: Optional[float] = None
    baseline_safety_std: Optional[float] = None
    # Rolling window of last 50 scores for comparison
    recent_safety_scores: List[float] = field(default_factory=list)

    # ---- Refusal rate (responses with safety_score ≤ 2.0) ----
    baseline_refusal_rate: Optional[float] = None   # set at call 100
    baseline_refusal_count: int = 0
    baseline_total_scored: int = 0
    recent_refusal_count: int = 0
    recent_total_for_refusal: int = 0               # recent scored calls (max 50)

    # ---- Tool scope ----
    baseline_tool_scope: Set[str] = field(default_factory=set)
    high_risk_tools_at_baseline: Set[str] = field(default_factory=set)
    post_baseline_new_tools: List[str] = field(default_factory=list)   # audit trail

    # ---- System prompt ----
    certified_prompt_hashes: Set[str] = field(default_factory=set)
    last_recertified_ms: Optional[int] = None
    # Track if a prompt change co-occurred with a safety drop
    prompt_changed_this_window: bool = False

    # ---- Response length (other signals) ----
    baseline_response_length_mean: Optional[float] = None
    recent_response_lengths: List[int] = field(default_factory=list)

    # ---- Prompt template hashes (other signals) ----
    baseline_template_hashes: Set[str] = field(default_factory=set)
    recent_template_hashes: Set[str] = field(default_factory=set)

    # ---- Temporal pattern (other signals) ----
    # 24-bucket call distribution — copied from AgentRecord temporal_rhythm
    baseline_temporal_rhythm: List[int] = field(default_factory=lambda: [0] * 24)

    # ---- Latest computed result (cached for dashboard reads) ----
    latest_drift_score: float = 0.0
    latest_drift_classification: str = "STABLE"
    latest_drift_signals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AgentDriftStore:
    """Thread-safe, in-memory store for per-agent drift baselines."""

    def __init__(self) -> None:
        self._baselines: Dict[str, AgentDriftBaseline] = {}
        self._lock = threading.Lock()

    def get_or_create(self, agent_id: str) -> AgentDriftBaseline:
        with self._lock:
            if agent_id not in self._baselines:
                self._baselines[agent_id] = AgentDriftBaseline(agent_id=agent_id)
            return self._baselines[agent_id]

    def update(self, baseline: AgentDriftBaseline) -> None:
        with self._lock:
            self._baselines[baseline.agent_id] = baseline

    def all_baselines(self) -> List[AgentDriftBaseline]:
        with self._lock:
            return list(self._baselines.values())


_global_drift_store = AgentDriftStore()


def get_drift_store() -> AgentDriftStore:
    return _global_drift_store


# ---------------------------------------------------------------------------
# Helper: ZASP score derivation from TelemetryEvent
# ---------------------------------------------------------------------------


def derive_zasp_score(event: "TelemetryEvent") -> Optional[float]:
    """Convert zendata_composite_risk (0–100 string) → ZASP safety score (0–10).

    Higher ZASP score = safer.  Returns None when no risk data is available at
    all.

    Scope boundary violations are risk evidence and are deducted here. Without
    this they would be invisible to the ZASP score and to every drift component
    that reads it — an agent could ground an answer in data another agent was
    never permitted to hand it and still report a clean safety score, which is
    precisely the outcome the scope layer exists to prevent.

    Note that a scope violation alone is enough to produce a score. The
    composite risk field is populated by the Zendata evaluation engine and is
    frequently absent; returning None in that case would discard a finding the
    SDK made on its own evidence and is certain of.
    """
    composite_score: Optional[float] = None
    try:
        composite = float(event.zendata_composite_risk or "")
        composite_score = max(0.0, min(10.0, 10.0 - composite / 10.0))
    except (ValueError, TypeError):
        composite_score = None

    penalty = 0.0
    try:
        from ai_telemetry.scope_checker import safety_penalty
        penalty = safety_penalty(getattr(event, "scope_srm_weight", None))
    except Exception:                                   # pragma: no cover
        penalty = 0.0

    if composite_score is None:
        # No engine verdict. A clean event tells us nothing; a violation tells
        # us a great deal, so only the second produces a score.
        if penalty <= 0.0:
            return None
        return max(0.0, 10.0 - penalty)

    return max(0.0, composite_score - penalty)


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def _decay_weight(age_ms: int) -> float:
    """Exponential decay weight for an observation of given age (milliseconds).

    w = exp(-age_days / τ)  where τ = _DECAY_TIME_CONSTANT_DAYS.
    At 30 days: w ≈ 0.37 (37 % of fresh weight).
    """
    age_days = age_ms / 86_400_000
    return math.exp(-age_days / _DECAY_TIME_CONSTANT_DAYS)


def _prune_old_samples(samples: List[TimedSample], now_ms: int) -> List[TimedSample]:
    """Remove samples older than 30 days."""
    cutoff = now_ms - _THIRTY_DAYS_MS
    return [s for s in samples if s.timestamp_ms >= cutoff]


def _weighted_mean_std(samples: List[TimedSample], now_ms: int) -> Tuple[Optional[float], Optional[float]]:
    """Compute weighted mean and std using exponential decay weights."""
    if not samples:
        return None, None
    weights = [_decay_weight(now_ms - s.timestamp_ms) for s in samples]
    total_w = sum(weights)
    if total_w == 0:
        return None, None
    mean = sum(w * s.value for w, s in zip(weights, samples)) / total_w
    variance = sum(w * (s.value - mean) ** 2 for w, s in zip(weights, samples)) / total_w
    return mean, math.sqrt(variance)


def _update_baseline_ema(
    current_ema: Optional[float],
    new_value: float,
    elapsed_ms: int,
) -> float:
    """Advance the EMA baseline by one observation using time-based decay.

    alpha = 1 - exp(-elapsed_days / τ)
    alpha → 1 for large gaps (observation dominates), → 0 for immediate calls.
    """
    if current_ema is None:
        return new_value
    elapsed_days = elapsed_ms / 86_400_000
    alpha = 1.0 - math.exp(-elapsed_days / _DECAY_TIME_CONSTANT_DAYS)
    alpha = max(0.01, min(0.5, alpha))   # clamp: [1 % – 50 %] per call
    return alpha * new_value + (1.0 - alpha) * current_ema


# ---------------------------------------------------------------------------
# Component 1: Safety score drift  (weight 40%)
# ---------------------------------------------------------------------------


def _check_safety_score_drift(
    baseline: AgentDriftBaseline,
    zasp_score: float,
    now_ms: int,
) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names])."""
    alerts: List[str] = []

    # Update rolling recent window (last 50 scores)
    baseline.recent_safety_scores.append(zasp_score)
    if len(baseline.recent_safety_scores) > _RECENT_WINDOW:
        baseline.recent_safety_scores.pop(0)

    # Update time-series samples (for decayed baseline)
    baseline.safety_samples.append(TimedSample(now_ms, zasp_score))
    baseline.safety_samples = _prune_old_samples(baseline.safety_samples, now_ms)

    # Track refusal
    baseline.recent_total_for_refusal += 1
    if baseline.recent_total_for_refusal > _RECENT_WINDOW:
        baseline.recent_total_for_refusal = _RECENT_WINDOW  # cap for rate calculation
    if zasp_score <= _REFUSAL_SAFETY_THRESHOLD:
        baseline.baseline_refusal_count += 1  # also used for baseline rate
        baseline.recent_refusal_count += 1

    # Establish full baseline at call 100
    if baseline.total_calls == _FULL_THRESHOLD:
        wm, ws = _weighted_mean_std(baseline.safety_samples, now_ms)
        baseline.baseline_safety_mean = wm
        baseline.baseline_safety_std = ws or 0.5
        # Baseline refusal rate
        if baseline.baseline_total_scored > 0:
            baseline.baseline_refusal_rate = (
                baseline.baseline_refusal_count / baseline.baseline_total_scored
            )
        else:
            baseline.baseline_refusal_rate = 0.0
        baseline.recent_refusal_count = 0
        baseline.recent_total_for_refusal = 0

    baseline.baseline_total_scored += 1

    # No baseline → no drift score
    ref_mean = baseline.baseline_safety_mean
    if ref_mean is None:
        # Preliminary: use all-time mean as reference
        if baseline.preliminary_active and len(baseline.recent_safety_scores) >= 5:
            ref_mean = sum(baseline.recent_safety_scores) / len(baseline.recent_safety_scores)
        else:
            return 0.0, []

    # Update baseline EMA with current observation (rolling 30-day decay)
    elapsed_ms = now_ms - baseline.last_sample_ms
    baseline.baseline_safety_mean = _update_baseline_ema(
        baseline.baseline_safety_mean, zasp_score, elapsed_ms
    )

    # Recent mean (last 50 calls)
    if len(baseline.recent_safety_scores) < 5:
        return 0.0, []
    recent_mean = sum(baseline.recent_safety_scores) / len(baseline.recent_safety_scores)

    # Delta: positive = baseline was safer (current is worse)
    delta = ref_mean - recent_mean
    if delta <= 0:
        return 0.0, []

    # Thresholds per spec: drop > 0.5 = WARNING, > 1.0 = CRITICAL
    if delta > 1.0:
        alerts.append("SAFETY_SCORE_CRITICAL")
    elif delta > 0.5:
        alerts.append("SAFETY_SCORE_WARNING")

    component = min(1.0, delta / 1.5)  # 1.5 delta → max score
    return component, alerts


# ---------------------------------------------------------------------------
# Component 2: Tool scope expansion  (weight 25%)
# ---------------------------------------------------------------------------


def _check_tool_scope_expansion(
    baseline: AgentDriftBaseline,
    current_tool_scope: List[str],
    now_ms: int,
) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names])."""
    alerts: List[str] = []
    component = 0.0

    if not current_tool_scope:
        return 0.0, []

    # During baseline period: record the baseline tool scope
    if not baseline.full_baseline_active:
        baseline.baseline_tool_scope.update(current_tool_scope)
        baseline.high_risk_tools_at_baseline.update(
            t for t in current_tool_scope if t in _HIGH_RISK_TOOLS
        )
        return 0.0, []

    # Post-baseline: check for new tools
    new_tools = [t for t in current_tool_scope if t not in baseline.baseline_tool_scope]
    for tool in new_tools:
        if tool in baseline.post_baseline_new_tools:
            continue  # already alerted
        baseline.post_baseline_new_tools.append(tool)
        if tool in _HIGH_RISK_TOOLS:
            alerts.append(f"TOOL_EXPANSION_CRITICAL:{tool}")
            component = max(component, 1.0)
        else:
            alerts.append(f"TOOL_EXPANSION_ALERT:{tool}")
            component = max(component, 0.5)

    return component, alerts


# ---------------------------------------------------------------------------
# Component 3: System prompt version change  (weight 15%)
# ---------------------------------------------------------------------------


def _check_prompt_version(
    baseline: AgentDriftBaseline,
    prompt_hash: Optional[str],
    safety_dropping: bool,
) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names]).

    safety_dropping = True when safety score component > 0 (scores are worsening).
    """
    if not prompt_hash:
        return 0.0, []

    alerts: List[str] = []

    # During baseline: register certified hashes
    if not baseline.full_baseline_active:
        baseline.certified_prompt_hashes.add(prompt_hash)
        return 0.0, []

    if prompt_hash in baseline.certified_prompt_hashes:
        baseline.prompt_changed_this_window = False
        return 0.0, []

    # New prompt hash detected
    baseline.prompt_changed_this_window = True
    if safety_dropping:
        alerts.append("PROMPT_VERSION_CRITICAL")
        component = 1.0
    else:
        alerts.append("PROMPT_VERSION_AUDIT")
        component = 0.5

    return component, alerts


# ---------------------------------------------------------------------------
# Component 4: Refusal rate change  (weight 10%)
# ---------------------------------------------------------------------------


def _check_refusal_rate(baseline: AgentDriftBaseline) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names]).

    Drop > 20% in refusal rate → WARNING (agents refusing less = less safe).
    """
    ref_rate = baseline.baseline_refusal_rate
    if ref_rate is None or baseline.recent_total_for_refusal < 10:
        return 0.0, []

    recent_rate = (
        baseline.recent_refusal_count / baseline.recent_total_for_refusal
        if baseline.recent_total_for_refusal > 0
        else 0.0
    )

    drop = ref_rate - recent_rate  # positive = fewer refusals (worse)
    if drop <= 0:
        return 0.0, []

    alerts: List[str] = []
    if drop > 0.20:
        alerts.append("REFUSAL_RATE_WARNING")

    component = min(1.0, drop / 0.40)  # 40% drop → max score
    return component, alerts


# ---------------------------------------------------------------------------
# Component 5: Cross-agent anomaly  (weight 5%)
# ---------------------------------------------------------------------------


def _check_cross_agent_anomaly(
    baseline: AgentDriftBaseline,
    cluster_means: List[float],
) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names]).

    cluster_means: recent safety means of other agents with the same role.
    Deviation > 0.6 from cluster mean → ALERT.
    """
    if not cluster_means or baseline.baseline_safety_mean is None:
        return 0.0, []
    if len(baseline.recent_safety_scores) < 5:
        return 0.0, []

    cluster_mean = sum(cluster_means) / len(cluster_means)
    recent_mean = sum(baseline.recent_safety_scores) / len(baseline.recent_safety_scores)
    deviation = abs(recent_mean - cluster_mean)

    if deviation <= 0:
        return 0.0, []

    alerts: List[str] = []
    if deviation > 0.6:
        alerts.append("CROSS_AGENT_ANOMALY")

    component = min(1.0, deviation / 2.0)
    return component, alerts


# ---------------------------------------------------------------------------
# Component 6: Other signals (response length, prompt pattern, temporal)  (5%)
# ---------------------------------------------------------------------------


def _check_other_signals(
    baseline: AgentDriftBaseline,
    response_length: Optional[int],
    template_hash: Optional[str],
) -> Tuple[float, List[str]]:
    """Return (component_score 0–1, [alert_names]).

    Aggregates sub-threshold signals that don't warrant their own component
    but contribute in combination.
    """
    alerts: List[str] = []
    sub_scores: List[float] = []

    # Response length deviation
    if response_length is not None:
        if not baseline.full_baseline_active:
            baseline.recent_response_lengths.append(response_length)
        else:
            baseline.recent_response_lengths.append(response_length)
            if len(baseline.recent_response_lengths) > _RECENT_WINDOW:
                baseline.recent_response_lengths.pop(0)

            if baseline.baseline_response_length_mean is not None and len(baseline.recent_response_lengths) >= 5:
                recent_len_mean = sum(baseline.recent_response_lengths) / len(baseline.recent_response_lengths)
                ref = baseline.baseline_response_length_mean
                if ref > 0:
                    deviation_pct = abs(recent_len_mean - ref) / ref
                    if deviation_pct > 0.50:
                        alerts.append("RESPONSE_LENGTH_SHIFT")
                    sub_scores.append(min(1.0, deviation_pct / 1.0))

    # Prompt template diversity shift
    if template_hash and baseline.full_baseline_active:
        baseline.recent_template_hashes.add(template_hash)
        new_templates = baseline.recent_template_hashes - baseline.baseline_template_hashes
        if len(new_templates) > 5:
            sub_scores.append(min(1.0, len(new_templates) / 20))

    component = sum(sub_scores) / max(len(sub_scores), 1) if sub_scores else 0.0
    return component, alerts


# ---------------------------------------------------------------------------
# Aggregate scoring and classification
# ---------------------------------------------------------------------------


def _aggregate_drift_score(component_scores: Dict[str, float]) -> float:
    """Weighted sum of component scores → 0.0–1.0."""
    return min(1.0, sum(
        _WEIGHTS[name] * score
        for name, score in component_scores.items()
    ))


def classify_drift(score: float) -> str:
    """Map drift score to 5-tier label."""
    for threshold, label in _DRIFT_TIERS:
        if score < threshold:
            return label
    return "CRITICAL"


def _detect_unexplained_drift(drift_score: float, prompt_changed: bool) -> List[str]:
    """Emit the highest-value alert: significant drift with no prompt change."""
    if drift_score >= 0.40 and not prompt_changed:
        return ["UNEXPLAINED_DRIFT"]
    return []


# ---------------------------------------------------------------------------
# Re-certification
# ---------------------------------------------------------------------------


def recertify_agent(agent_id: str, now_ms: Optional[int] = None) -> None:
    """Re-certification event: reset the baseline to current state.

    Call this when an agent receives a deliberate, authorized update so that
    the new configuration becomes the new reference point.
    """
    now_ms = now_ms or int(time.time() * 1000)
    store = get_drift_store()
    baseline = store.get_or_create(agent_id)

    # Snapshot current safety mean as new baseline
    if baseline.recent_safety_scores:
        baseline.baseline_safety_mean = (
            sum(baseline.recent_safety_scores) / len(baseline.recent_safety_scores)
        )

    # Promote recent state to certified
    baseline.certified_prompt_hashes.update(
        # These will be re-added from the next event
    )
    baseline.baseline_tool_scope.update(baseline.post_baseline_new_tools)
    baseline.post_baseline_new_tools.clear()

    if baseline.recent_response_lengths:
        baseline.baseline_response_length_mean = (
            sum(baseline.recent_response_lengths) / len(baseline.recent_response_lengths)
        )

    baseline.baseline_refusal_rate = (
        baseline.recent_refusal_count / max(baseline.recent_total_for_refusal, 1)
    )
    baseline.recent_refusal_count = 0
    baseline.recent_total_for_refusal = 0
    baseline.prompt_changed_this_window = False
    baseline.last_recertified_ms = now_ms
    baseline.latest_drift_score = 0.0
    baseline.latest_drift_classification = "STABLE"
    baseline.latest_drift_signals = []

    store.update(baseline)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_drift(
    event: "TelemetryEvent",
    agent_id: str,
    current_tool_scope: List[str],
    cluster_means: Optional[List[float]] = None,
) -> Tuple[float, str, List[str]]:
    """Compute the aggregate drift score for this agent on this call.

    Returns (drift_score, classification, [all_alert_names]).

    Called from TelemetryClient._send() after fingerprint_agent().
    Requires at least 20 calls before returning non-zero scores.
    """
    store = get_drift_store()
    baseline = store.get_or_create(agent_id)

    now_ms = event.timestamp_ms
    baseline.total_calls += 1

    # Activation gates
    if baseline.total_calls < _PRELIMINARY_THRESHOLD:
        # Collect data but produce no drift signal yet
        _accumulate_pre_baseline(baseline, event, current_tool_scope, now_ms)
        store.update(baseline)
        return 0.0, "STABLE", []

    if not baseline.preliminary_active:
        baseline.preliminary_active = True

    if baseline.total_calls >= _FULL_THRESHOLD and not baseline.full_baseline_active:
        baseline.full_baseline_active = True
        # Establish baseline response length mean from collected data
        if baseline.recent_response_lengths:
            baseline.baseline_response_length_mean = (
                sum(baseline.recent_response_lengths) / len(baseline.recent_response_lengths)
            )
        # Lock in template hashes from baseline period
        baseline.baseline_template_hashes = set(baseline.recent_template_hashes)
        # Freeze initial tool scope
        baseline.baseline_tool_scope.update(current_tool_scope)

    # ---- Derive ZASP score ----
    zasp = derive_zasp_score(event)

    # ---- Response length ----
    response_length: Optional[int] = None
    if event.output_message and isinstance(event.output_message.get("content"), str):
        response_length = len(event.output_message["content"])

    # ---- Template hash ----
    template_hash = event.prompt_template_hash

    # ---- Certify prompt hash during baseline ----
    if not baseline.full_baseline_active and event.system_prompt_hash:
        baseline.certified_prompt_hashes.add(event.system_prompt_hash)

    # ---- Run component detectors ----
    c1_score, c1_alerts = (0.0, [])
    if zasp is not None:
        c1_score, c1_alerts = _check_safety_score_drift(baseline, zasp, now_ms)

    c2_score, c2_alerts = _check_tool_scope_expansion(baseline, current_tool_scope, now_ms)

    # Component 3 uses whether safety is currently dropping as a co-occurrence test
    safety_dropping = c1_score > 0
    c3_score, c3_alerts = _check_prompt_version(
        baseline, event.system_prompt_hash, safety_dropping
    )

    c4_score, c4_alerts = _check_refusal_rate(baseline)
    c5_score, c5_alerts = _check_cross_agent_anomaly(baseline, cluster_means or [])
    c6_score, c6_alerts = _check_other_signals(baseline, response_length, template_hash)

    # ---- Aggregate ----
    component_scores = {
        "safety_score":   c1_score,
        "tool_expansion": c2_score,
        "prompt_version": c3_score,
        "refusal_rate":   c4_score,
        "cross_agent":    c5_score,
        "other":          c6_score,
    }
    drift_score = _aggregate_drift_score(component_scores)
    classification = classify_drift(drift_score)

    # Collect all alerts
    all_alerts: List[str] = c1_alerts + c2_alerts + c3_alerts + c4_alerts + c5_alerts + c6_alerts

    # Highest-value alert: unexplained drift
    all_alerts += _detect_unexplained_drift(drift_score, baseline.prompt_changed_this_window)

    # ---- Update cached result ----
    baseline.latest_drift_score = round(drift_score, 4)
    baseline.latest_drift_classification = classification
    baseline.latest_drift_signals = all_alerts
    baseline.last_sample_ms = now_ms
    store.update(baseline)

    return round(drift_score, 4), classification, all_alerts


# ---------------------------------------------------------------------------
# Pre-baseline accumulation (before call 20)
# ---------------------------------------------------------------------------


def _accumulate_pre_baseline(
    baseline: AgentDriftBaseline,
    event: "TelemetryEvent",
    current_tool_scope: List[str],
    now_ms: int,
) -> None:
    """Collect observations during the pre-baseline warm-up period."""
    zasp = derive_zasp_score(event)
    if zasp is not None:
        baseline.safety_samples.append(TimedSample(now_ms, zasp))
        baseline.recent_safety_scores.append(zasp)
        baseline.baseline_total_scored += 1
        if zasp <= _REFUSAL_SAFETY_THRESHOLD:
            baseline.baseline_refusal_count += 1

    baseline.baseline_tool_scope.update(current_tool_scope)

    if event.system_prompt_hash:
        baseline.certified_prompt_hashes.add(event.system_prompt_hash)

    if event.output_message and isinstance(event.output_message.get("content"), str):
        baseline.recent_response_lengths.append(len(event.output_message["content"]))

    if event.prompt_template_hash:
        baseline.recent_template_hashes.add(event.prompt_template_hash)

    baseline.last_sample_ms = now_ms
