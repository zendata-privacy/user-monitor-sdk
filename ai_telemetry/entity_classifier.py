"""
Entity Classifier — differentiates human users from AI agents.

Implements all behavioral rules across 6 categories and writes results
directly onto the TelemetryEvent passed in.

Categories
----------
  1  Timing & Latency    T1  T2  T3
  2  Prompt Structure    P1  P4
  3  Session Behavior    S1  S4
  4  API Access Patterns A1  A2
  5  Multi-Agent Topology M1
  6  Behavioral Drift    D1  D2  D3  D4  D5
"""

import re
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryEvent
    from ai_telemetry.session_state import SessionState, SessionStore

# ---------------------------------------------------------------------------
# Compiled regular expressions (module-level for performance)
# ---------------------------------------------------------------------------

_RE_IMPERATIVE = re.compile(
    r"\b(return only|do not include|format as|your task is|respond in json"
    r"|output only|provide only|do not add|output as|response must be"
    r"|always respond|never include|you must return|you must respond)\b",
    re.IGNORECASE,
)

_RE_DELEGATION = re.compile(
    r"\b(your task is|you are a sub.?agent|previous agent determined"
    r"|tool result|agent output|pass this to|as instructed by"
    r"|the orchestrator|delegating to|sub.?agent|from the orchestrator"
    r"|handoff from|task assigned by)\b",
    re.IGNORECASE,
)

_RE_PERSONA = re.compile(
    r"\b(pretend you are|act as|you are now|from now on you are"
    r"|roleplay as|play the role|assume the identity|your new persona"
    r"|forget your (previous|prior)|ignore your (previous|prior)"
    r"|disregard your|jailbreak|DAN mode|developer mode)\b",
    re.IGNORECASE,
)

_RE_REGULATORY = re.compile(
    r"\b(structuring|smurfing|below the threshold|under \$10[,\s]?000"
    r"|reporting requirement|currency transaction report|CTR\b|SAR\b"
    r"|avoid detection|split transaction|layering|placement stage"
    r"|integration stage|cash intensive|nominee account|bulk cash"
    r"|cash equivalent|threshold avoidance)\b",
    re.IGNORECASE,
)

_RE_SENSITIVE_ENTITY = re.compile(
    r"\b(account number|routing number|SSN|social security"
    r"|IBAN|tax id|EIN|beneficiary|offshore|shell company"
    r"|wire transfer|correspondent bank|hawala)\b",
    re.IGNORECASE,
)

_RE_ORIENTATION = re.compile(
    r"\b(hello|hi |hey |thanks|thank you|can you|could you"
    r"|what is|how does|i need|i have a question"
    r"|i.?m looking|i.?d like|please help|good morning|good afternoon)\b",
    re.IGNORECASE,
)

_RE_SPECIFICITY = re.compile(
    r"\b(exactly|specifically|precise|actual|real|give me the|tell me the"
    r"|what is the exact|i need the exact)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Category 1 — Timing & Latency
# ---------------------------------------------------------------------------

def _check_T1(state: "SessionState", submission_ms: int) -> List[str]:
    """T1 — Submission velocity.

    < 500 ms after session start (first turn) or < 200 ms after last
    response → flag AUTOMATED.
    """
    if not state.turns:
        if submission_ms - state.created_at_ms < 500:
            return ["T1_SUBMISSION_VELOCITY"]
    else:
        last_resp = state.turns[-1].response_time_ms
        if last_resp is not None and submission_ms - last_resp < 200:
            return ["T1_SUBMISSION_VELOCITY"]
    return []


def _check_T2(state: "SessionState") -> List[str]:
    """T2 — Inter-turn latency variance.

    stddev < 200 ms over 5+ gaps → AUTOMATED;
    stddev > 3000 ms → HUMAN indicator.
    """
    gaps: List[float] = []
    for i in range(1, len(state.turns)):
        prev_resp = state.turns[i - 1].response_time_ms
        curr_sub = state.turns[i].submission_time_ms
        if prev_resp is not None:
            gaps.append(float(curr_sub - prev_resp))

    if len(gaps) < 5:
        return []

    mean = sum(gaps) / len(gaps)
    stddev = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5

    if stddev < 200:
        return ["T2_LOW_LATENCY_VARIANCE"]   # → AUTOMATED
    if stddev > 3000:
        return ["T2_HIGH_LATENCY_VARIANCE"]  # → HUMAN indicator
    return []


def _check_T3(prompt_text: str, inter_turn_ms: int) -> List[str]:
    """T3 — Typing speed.

    chars/sec > 15 → AUTOMATED (human average is 4–6 chars/sec).
    """
    if inter_turn_ms <= 0:
        return []
    cps = len(prompt_text) / (inter_turn_ms / 1000.0)
    return ["T3_TYPING_SPEED"] if cps > 15 else []


# ---------------------------------------------------------------------------
# Category 2 — Prompt Structure
# ---------------------------------------------------------------------------

def _structural_skeleton(text: str) -> str:
    """Reduce a prompt to its structural skeleton by replacing variable content."""
    s = re.sub(r'"[^"]*"', '"__V__"', text)
    s = re.sub(r"'[^']*'", "'__V__'", s)
    s = re.sub(r"\b\d[\d,.\s]*\b", "__N__", s)
    s = re.sub(r"\b[A-Z]{2,6}\b", "__S__", s)   # ticker symbols / acronyms
    return s.strip()


def _check_P1(state: "SessionState", current_prompt: str) -> List[str]:
    """P1 — Template marker detection.

    If 3+ turns (including current) share an identical structural skeleton
    in their first 80 characters → flag AUTOMATED.
    """
    all_prompts = [t.prompt_text for t in state.turns] + [current_prompt]
    if len(all_prompts) < 3:
        return []
    openings = [_structural_skeleton(p)[:80] for p in all_prompts]
    most_common = max(set(openings), key=openings.count)
    if openings.count(most_common) >= 3:
        return ["P1_TEMPLATE_STRUCTURE"]
    return []


def _check_P4(prompt_text: str) -> List[str]:
    """P4 — Imperative / instruction framing.

    2+ imperative command phrases in one prompt → flag AUTOMATED.
    """
    matches = _RE_IMPERATIVE.findall(prompt_text)
    return ["P4_IMPERATIVE_FRAMING"] if len(matches) >= 2 else []


# ---------------------------------------------------------------------------
# Category 3 — Session Behavior
# ---------------------------------------------------------------------------

def _check_S1(prompt_text: str, state: "SessionState") -> List[str]:
    """S1 — Warm-up absence.

    First prompt is long (> 80 chars) and contains no orientation language
    → flag AUTOMATED.
    """
    if state.turns:
        return []
    has_orientation = bool(_RE_ORIENTATION.search(prompt_text))
    if len(prompt_text) > 80 and not has_orientation:
        return ["S1_WARMUP_ABSENT"]
    return []


def _check_S4(store: "SessionStore", entity_key: str) -> List[str]:
    """S4 — Simultaneous sessions.

    Same entity active in 2+ sessions → flag AUTOMATED definitively.
    """
    if store.count_active_sessions(entity_key) >= 2:
        return ["S4_SIMULTANEOUS_SESSIONS"]
    return []


# ---------------------------------------------------------------------------
# Category 4 — API & Access Patterns
# ---------------------------------------------------------------------------

def _check_A1(store: "SessionStore", entity_key: str) -> List[str]:
    """A1 — Call volume per key.

    > 1 000 calls in 24 h → AUTOMATED;  < 20 calls → lean HUMAN.
    """
    count = len(store.get_call_timestamps(entity_key))
    if count > 1000:
        return ["A1_HIGH_CALL_VOLUME"]
    if count < 20:
        return ["A1_LOW_CALL_VOLUME"]
    return []


def _check_A2(store: "SessionStore", entity_key: str, now_ms: int) -> List[str]:
    """A2 — Burst detection.

    > 5 calls in any 10-second window → flag AUTOMATED.
    """
    window_start = now_ms - 10_000
    burst = sum(1 for t in store.get_call_timestamps(entity_key) if t >= window_start)
    return ["A2_BURST_CALLS"] if burst > 5 else []


# ---------------------------------------------------------------------------
# Category 5 — Multi-Agent Topology
# ---------------------------------------------------------------------------

def _check_M1(prompt_text: str, turn_index: int) -> Tuple[List[str], Optional[str]]:
    """M1 — Delegation language detection.

    Returns (signals, subtype) where subtype is SUBAGENT_RECEIVER or
    ORCHESTRATOR_SENDER based on turn position.
    """
    if not _RE_DELEGATION.search(prompt_text):
        return [], None
    # First turn with delegation language → this entity is receiving a delegated task
    subtype = "SUBAGENT_RECEIVER" if turn_index == 0 else "ORCHESTRATOR_SENDER"
    return ["M1_DELEGATION_LANGUAGE"], subtype


# ---------------------------------------------------------------------------
# Category 6 — Behavioral Drift
# ---------------------------------------------------------------------------

def _check_D1(store: "SessionStore", entity_key: str, current_risk: float) -> List[str]:
    """D1 — Baseline deviation.

    Current risk score deviates > 1.5 SD from rolling average → DRIFT_ALERT.
    """
    baseline = store.get_baseline(entity_key)
    mean, std = baseline.mean_and_std()
    if mean is None or std is None or std == 0:
        return []
    if abs(current_risk - mean) / std > 1.5:
        return ["D1_BASELINE_DRIFT"]
    return []


def _check_D2(state: "SessionState", submission_ms: int) -> List[str]:
    """D2 — Human-in-loop anomaly.

    Session classified as AUTOMATED/AGENT suddenly shows human timing
    (> 2 000 ms gap) after at least 2 turns → TAKEOVER_ALERT.
    """
    if state.entity_type not in ("AUTOMATED", "AGENT"):
        return []
    if len(state.turns) < 2:
        return []
    last_resp = state.turns[-1].response_time_ms
    if last_resp is not None and submission_ms - last_resp > 2000:
        return ["D2_TAKEOVER_ALERT"]
    return []


def _check_D3(prompt_text: str, state: "SessionState") -> List[str]:
    """D3 — Persona switching / role-play injection.

    Any persona-switch instruction → ROLE_PLAY_INJECTION; increments
    session counter.
    """
    if _RE_PERSONA.search(prompt_text):
        state.role_play_count += 1
        return ["D3_ROLE_PLAY_INJECTION"]
    return []


def _probe_score(prompt_text: str) -> float:
    """Score how specifically a prompt targets a sensitive financial topic (0–10)."""
    score = 0.0
    score += min(len(prompt_text) / 200.0, 2.0)          # length heuristic
    if _RE_SENSITIVE_ENTITY.search(prompt_text):
        score += 4.0
    if _RE_REGULATORY.search(prompt_text):
        score += 3.0
    if _RE_SPECIFICITY.search(prompt_text):
        score += 1.0
    return min(score, 10.0)


def _check_D4(state: "SessionState", current_prompt: str) -> List[str]:
    """D4 — Escalating probe pattern.

    3 consecutive turns with strictly increasing sensitivity scores and
    the latest score ≥ 5 → PROGRESSIVE_PROBE.
    """
    current_score = _probe_score(current_prompt)
    state.probe_scores.append(current_score)
    if len(state.probe_scores) < 3:
        return []
    recent = state.probe_scores[-3:]
    if recent[0] < recent[1] < recent[2] and recent[2] >= 5.0:
        return ["D4_PROGRESSIVE_PROBE"]
    return []


def _check_D5(prompt_text: str) -> List[str]:
    """D5 — Threshold language / regulatory probe detection."""
    if _RE_REGULATORY.search(prompt_text):
        return ["D5_REGULATORY_PROBE"]
    return []


# ---------------------------------------------------------------------------
# Signal → entity type voting
# ---------------------------------------------------------------------------

_AUTOMATED_SIGNALS = frozenset({
    "T1_SUBMISSION_VELOCITY", "T2_LOW_LATENCY_VARIANCE", "T3_TYPING_SPEED",
    "P1_TEMPLATE_STRUCTURE", "P4_IMPERATIVE_FRAMING",
    "S1_WARMUP_ABSENT", "S4_SIMULTANEOUS_SESSIONS",
    "A1_HIGH_CALL_VOLUME", "A2_BURST_CALLS",
})

_HUMAN_SIGNALS = frozenset({
    "T2_HIGH_LATENCY_VARIANCE",
    "A1_LOW_CALL_VOLUME",
})

_AGENT_SIGNALS = frozenset({
    "M1_DELEGATION_LANGUAGE",
})

# Signals that produce alerts rather than entity-type votes
_ALERT_MAP = {
    "D1_BASELINE_DRIFT":    "DRIFT_ALERT",
    "D2_TAKEOVER_ALERT":    "TAKEOVER_ALERT",
    "D3_ROLE_PLAY_INJECTION": "ROLE_PLAY_INJECTION",
    "D4_PROGRESSIVE_PROBE": "PROGRESSIVE_PROBE",
    "D5_REGULATORY_PROBE":  "REGULATORY_PROBE",
}


def _vote_entity_type(signals: List[str]) -> Tuple[str, Optional[str]]:
    """
    Vote on entity type from collected signals.

    Returns (entity_type, entity_subtype).
    AGENT always wins if M1 is present (multi-agent topology is definitive).
    """
    if any(s in _AGENT_SIGNALS for s in signals):
        return "AGENT", None
    auto_votes = sum(1 for s in signals if s in _AUTOMATED_SIGNALS)
    human_votes = sum(1 for s in signals if s in _HUMAN_SIGNALS)
    if auto_votes > 0 and auto_votes > human_votes:
        return "AUTOMATED", None
    if human_votes > 0:
        return "HUMAN", None
    return "UNKNOWN", None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_event(
    event: "TelemetryEvent",
    store: "SessionStore",
    session_id: str,
    entity_key: str,
    prompt_text: str,
    submission_ms: int,
) -> None:
    """
    Run all classification rules and annotate *event* in-place.

    Called from TelemetryClient.capture() before the event is queued.
    """
    from ai_telemetry.session_state import TurnRecord

    now_ms = event.timestamp_ms
    state = store.get_or_create_session(session_id, entity_key)

    # Record this call for rate-limit tracking (A1/A2)
    store.record_call(entity_key, now_ms)

    # Calculate inter-turn latency (time from last response to this submission)
    inter_turn_ms: int = 0
    if state.turns:
        last = state.turns[-1]
        ref = last.response_time_ms if last.response_time_ms is not None else last.submission_time_ms
        inter_turn_ms = max(0, submission_ms - ref)
    else:
        inter_turn_ms = max(0, submission_ms - state.created_at_ms)

    turn_index = len(state.turns)

    # ----------------------------------------------------------------
    # Run all rules
    # ----------------------------------------------------------------
    all_signals: List[str] = []

    # Category 1
    all_signals += _check_T1(state, submission_ms)
    all_signals += _check_T2(state)
    if inter_turn_ms > 0:
        all_signals += _check_T3(prompt_text, inter_turn_ms)

    # Category 2
    all_signals += _check_P1(state, prompt_text)
    all_signals += _check_P4(prompt_text)

    # Category 3
    all_signals += _check_S1(prompt_text, state)
    all_signals += _check_S4(store, entity_key)

    # Category 4
    all_signals += _check_A1(store, entity_key)
    all_signals += _check_A2(store, entity_key, now_ms)

    # Category 5
    m1_signals, m1_subtype = _check_M1(prompt_text, turn_index)
    all_signals += m1_signals

    # Category 6 — use sailpoint_risk as the current risk score (composite
    # risk is not yet available at capture time; it's assigned later in _send)
    current_risk = 0.0
    try:
        current_risk = float(event.sailpoint_risk or 0)
    except (ValueError, TypeError):
        pass

    all_signals += _check_D1(store, entity_key, current_risk)
    all_signals += _check_D2(state, submission_ms)
    all_signals += _check_D3(prompt_text, state)
    all_signals += _check_D4(state, prompt_text)
    all_signals += _check_D5(prompt_text)

    # ----------------------------------------------------------------
    # Determine entity type
    # ----------------------------------------------------------------
    entity_type, _ = _vote_entity_type(all_signals)
    entity_subtype: Optional[str] = m1_subtype

    # Preserve an established non-UNKNOWN classification if no new signals fired
    if entity_type == "UNKNOWN" and state.entity_type != "UNKNOWN":
        entity_type = state.entity_type
        entity_subtype = state.entity_subtype

    # ----------------------------------------------------------------
    # Separate classification signals from behavioral alerts
    # ----------------------------------------------------------------
    alerts = [_ALERT_MAP[s] for s in all_signals if s in _ALERT_MAP]
    classification_signals = [s for s in all_signals if s not in _ALERT_MAP]

    # ----------------------------------------------------------------
    # Update session state
    # ----------------------------------------------------------------
    turn = TurnRecord(
        turn_index=turn_index,
        prompt_text=prompt_text,
        prompt_chars=len(prompt_text),
        submission_time_ms=submission_ms,
        # Approximate response time as the event timestamp (event is created
        # right after the LLM call completes)
        response_time_ms=now_ms,
    )
    state.turns.append(turn)
    state.entity_type = entity_type
    if entity_subtype:
        state.entity_subtype = entity_subtype
    for sig in all_signals:
        if sig not in state.signals:
            state.signals.append(sig)
    for alert in alerts:
        if alert not in state.alerts:
            state.alerts.append(alert)

    # Update rolling baseline
    if current_risk > 0:
        store.update_baseline(entity_key, current_risk)

    store.update_session(state)

    # ----------------------------------------------------------------
    # Annotate the event
    # ----------------------------------------------------------------
    event.entity_type = entity_type
    event.entity_subtype = entity_subtype
    event.classification_signals = "|".join(classification_signals) if classification_signals else None
    event.behavioral_alerts = "|".join(alerts) if alerts else None
    event.session_turn_index = turn_index
    event.inter_turn_latency_ms = inter_turn_ms
    event.prompt_submission_time_ms = submission_ms
