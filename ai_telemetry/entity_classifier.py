"""
Entity Classifier — differentiates human users from AI agents.

Implements all behavioral rules across 6 categories and writes results
directly onto the TelemetryEvent passed in.

Categories
----------
  1  Timing & Latency      T1  T2  T3  T5  T6
  2  Prompt Structure      P1  P4
  3  Session Behavior      S1  S2  S4
  4  API Access Patterns   A1  A2
  5  Multi-Agent Topology  M1
  6  Behavioral Drift      D1  D2  D3  D4  D5
  7  Temporal Context      B1

Scoring
-------
Every rule emits a named signal.  Each signal carries a weight that pushes
the ``caller_type_score`` toward 0.0 (definite agent) or 1.0 (definite human).
Weights are defined in ``_SIGNAL_WEIGHTS``.

Per-turn score = 0.5 + Σ(weights of signals fired this turn), clamped [0, 1].
Session score  = EMA(per_turn_score, alpha=0.4) — recent turns count more.

Thresholds (spec §Module 2):
  score < 0.25          → AUTOMATED or AGENT (see M1 flag)
  0.25 ≤ score ≤ 0.75  → AMBIGUOUS  (re-evaluated each call)
  score > 0.75          → HUMAN

Most callers resolve in 3–5 calls.
"""

import re
from datetime import datetime, timezone
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

# T5 — Human writing artifacts: typos, self-corrections, informal language
_RE_TYPO_CORRECTION = re.compile(
    r"\b(wait[,\s]+i meant|i meant to (say|type|write)|oops[,!\s]|my bad"
    r"|typo[,\s]|sorry[,\s]+(i meant|let me|i was)|lol\b|gonna\b|wanna\b"
    r"|dunno\b|kinda\b|sorta\b|nvm\b|idk\b|tbh\b|imo\b|fyi\b|btw\b"
    r"|thx\b|pls\b|tho\b|rn\b|irl\b)\b"
    r"|\*correction\*|\*edit\*|\*update\*"
    r"|\.{3,}",   # overuse of ellipsis is a human writing artifact
    re.IGNORECASE,
)

# S2 — Session continuity: references to prior turns / pronoun resolution
_RE_CONTINUITY = re.compile(
    r"\b(you (just |previously |earlier )?(said|mentioned|told me|suggested|recommended)"
    r"|as (you |we )?(said|mentioned|discussed|talked about)"
    r"|earlier (you|we|i)|the (previous|last) (message|response|answer|question|point)"
    r"|going back to|referring to|what (i|you) (said|meant|asked)"
    r"|like (i|you) (mentioned|said|asked)|that (thing|point|option|idea) (you|i)"
    r"|from (before|earlier)|following up (on)?|as (we |i |you )?discussed"
    r"|as (we |i |you )?mentioned|you (remember|recall)|i (remember|recall)"
    r"|to clarify (what|that)|building on (that|what)|in addition to (what|that)"
    r")\b",
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


def _check_T5(prompt_text: str) -> List[str]:
    """T5 — Typos, self-corrections, and informal language (HIGH human).

    Human writing artifacts: conversational repair ('wait I meant'), informal
    contractions (gonna, wanna), filler abbreviations (idk, lol), and
    overuse of ellipsis.  These patterns are virtually absent in programmatic
    prompt generation.
    """
    return ["T5_TYPO_CORRECTION"] if _RE_TYPO_CORRECTION.search(prompt_text) else []


def _check_T6(inter_turn_ms: int) -> List[str]:
    """T6 — Human reading/typing latency band (HIGH human).

    5 000–60 000 ms between the last response and this submission is
    consistent with a human reading the reply and composing a response.
    Sub-5 s favours automation; above 60 s may indicate the user stepped
    away (ambiguous), so we only assert human in this window.
    """
    if inter_turn_ms <= 0:
        return []
    return ["T6_HUMAN_LATENCY_BAND"] if 5_000 <= inter_turn_ms <= 60_000 else []


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


def _check_S2(prompt_text: str, state: "SessionState") -> List[str]:
    """S2 — Session continuity / pronoun resolution (HIGH human).

    References to previous messages ('you mentioned', 'as we discussed',
    'earlier you said') require the caller to maintain context across turns.
    Pipelines and stateless agents do not naturally produce these phrases.
    Only checked from turn 1 onwards (first turn has no prior context).
    """
    if not state.turns:
        return []
    return ["S2_SESSION_CONTINUITY"] if _RE_CONTINUITY.search(prompt_text) else []


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
# Category 7 — Temporal Context
# ---------------------------------------------------------------------------

def _check_B1(submission_ms: int) -> List[str]:
    """B1 — Business hours clustering (MEDIUM human).

    Human callers concentrate in working hours; batch pipelines and agents
    run at any hour.  A call arriving Mon–Fri 08:00–19:00 UTC leans human.
    This signal is MEDIUM weight — it confirms but cannot override strong
    timing signals pointing the other way.
    """
    dt = datetime.fromtimestamp(submission_ms / 1000.0, tz=timezone.utc)
    if dt.weekday() < 5 and 8 <= dt.hour < 19:  # Mon=0 … Fri=4
        return ["B1_BUSINESS_HOURS"]
    return []


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
# Signal weights — continuous caller_type scoring (Module 2)
# ---------------------------------------------------------------------------
# Each signal carries a weight in (-1, +1).
# Negative weights push the score toward 0.0 (agent).
# Positive weights push the score toward 1.0 (human).
# Magnitude: HIGH ≈ ±0.25, MEDIUM ≈ ±0.12.

_SIGNAL_WEIGHTS: dict = {
    # --- Agent signals (negative) ---
    "T1_SUBMISSION_VELOCITY":   -0.25,  # HIGH  — sub-500 ms latency
    "T2_LOW_LATENCY_VARIANCE":  -0.25,  # HIGH  — clock-like regularity
    "T3_TYPING_SPEED":          -0.25,  # HIGH  — >15 chars/sec
    "M1_DELEGATION_LANGUAGE":   -0.30,  # HIGH  — orchestration phrases
    "S1_WARMUP_ABSENT":         -0.12,  # MEDIUM — no warm-up / stateless
    "P1_TEMPLATE_STRUCTURE":    -0.12,  # MEDIUM — repeating template
    "P4_IMPERATIVE_FRAMING":    -0.12,  # MEDIUM — command framing
    "S4_SIMULTANEOUS_SESSIONS": -0.25,  # HIGH  — parallel sessions same key
    "A1_HIGH_CALL_VOLUME":      -0.12,  # MEDIUM — >1 000 calls/24 h
    "A2_BURST_CALLS":           -0.12,  # MEDIUM — >5 calls/10 s
    # --- Human signals (positive) ---
    "T5_TYPO_CORRECTION":       +0.25,  # HIGH  — informal / repair language
    "T6_HUMAN_LATENCY_BAND":    +0.25,  # HIGH  — 5–60 s response window
    "S2_SESSION_CONTINUITY":    +0.25,  # HIGH  — refers to prior turns
    "B1_BUSINESS_HOURS":        +0.12,  # MEDIUM — Mon–Fri 08:00–19:00 UTC
    "T2_HIGH_LATENCY_VARIANCE": +0.20,  # existing — high stddev across turns
    "A1_LOW_CALL_VOLUME":       +0.12,  # existing — <20 calls/24 h
}

# Signals that produce behavioral alerts rather than classification votes
_ALERT_MAP = {
    "D1_BASELINE_DRIFT":      "DRIFT_ALERT",
    "D2_TAKEOVER_ALERT":      "TAKEOVER_ALERT",
    "D3_ROLE_PLAY_INJECTION": "ROLE_PLAY_INJECTION",
    "D4_PROGRESSIVE_PROBE":   "PROGRESSIVE_PROBE",
    "D5_REGULATORY_PROBE":    "REGULATORY_PROBE",
}

# EMA smoothing factor: 0.4 weights the current turn, 0.6 retains session history.
# A caller typically resolves (exits AMBIGUOUS) after 3–5 turns.
_EMA_ALPHA = 0.4

# Spec thresholds
_THRESHOLD_AGENT  = 0.25
_THRESHOLD_HUMAN  = 0.75


def _compute_caller_type(
    signals: List[str],
    session_score: float,
    m1_subtype: Optional[str],
) -> Tuple[float, str, Optional[str]]:
    """Compute the updated session caller_type_score and derive entity_type.

    Returns (new_session_score, entity_type, entity_subtype).

    Algorithm
    ---------
    1. Per-turn score = 0.5 + Σ(signal weights for this turn), clamped [0, 1].
    2. EMA update: session_score = α·per_turn + (1-α)·session_score.
    3. Entity type from thresholds; M1 overrides < 0.25 → AGENT vs AUTOMATED.
    """
    per_turn_delta = sum(_SIGNAL_WEIGHTS.get(s, 0.0) for s in signals)
    per_turn_score = max(0.0, min(1.0, 0.5 + per_turn_delta))

    new_score = _EMA_ALPHA * per_turn_score + (1.0 - _EMA_ALPHA) * session_score
    new_score = max(0.0, min(1.0, new_score))

    if new_score < _THRESHOLD_AGENT:
        # M1 delegation language distinguishes AGENT from AUTOMATED
        entity_type = "AGENT" if "M1_DELEGATION_LANGUAGE" in signals else "AUTOMATED"
        entity_subtype = m1_subtype
    elif new_score > _THRESHOLD_HUMAN:
        entity_type = "HUMAN"
        entity_subtype = None
    else:
        entity_type = "AMBIGUOUS"
        entity_subtype = m1_subtype  # retain subtype if already known

    return new_score, entity_type, entity_subtype


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

    # Category 1 — Timing & Latency
    all_signals += _check_T1(state, submission_ms)
    all_signals += _check_T2(state)
    if inter_turn_ms > 0:
        all_signals += _check_T3(prompt_text, inter_turn_ms)
        all_signals += _check_T6(inter_turn_ms)      # NEW: human latency band
    all_signals += _check_T5(prompt_text)             # NEW: typos/corrections

    # Category 2 — Prompt Structure
    all_signals += _check_P1(state, prompt_text)
    all_signals += _check_P4(prompt_text)

    # Category 3 — Session Behavior
    all_signals += _check_S1(prompt_text, state)
    all_signals += _check_S2(prompt_text, state)      # NEW: session continuity
    all_signals += _check_S4(store, entity_key)

    # Category 4 — API & Access Patterns
    all_signals += _check_A1(store, entity_key)
    all_signals += _check_A2(store, entity_key, now_ms)

    # Category 5 — Multi-Agent Topology
    m1_signals, m1_subtype = _check_M1(prompt_text, turn_index)
    all_signals += m1_signals

    # Category 6 — Behavioral Drift
    # Use sailpoint_risk as the current risk score; composite risk is
    # assigned later in _send() and not yet available at capture time.
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

    # Category 7 — Temporal Context
    all_signals += _check_B1(submission_ms)           # NEW: business hours

    # ----------------------------------------------------------------
    # Separate classification signals from behavioral alerts
    # ----------------------------------------------------------------
    alerts = [_ALERT_MAP[s] for s in all_signals if s in _ALERT_MAP]
    classification_signals = [s for s in all_signals if s not in _ALERT_MAP]

    # ----------------------------------------------------------------
    # Continuous caller_type scoring (Module 2)
    # ----------------------------------------------------------------
    new_score, entity_type, entity_subtype = _compute_caller_type(
        classification_signals,
        state.caller_type_score,
        m1_subtype,
    )

    # If AMBIGUOUS and session has a more definitive prior classification,
    # preserve it (don't regress to AMBIGUOUS on a signal-quiet turn).
    if entity_type == "AMBIGUOUS" and state.entity_type not in ("UNKNOWN", "AMBIGUOUS"):
        entity_type = state.entity_type
        entity_subtype = state.entity_subtype or entity_subtype

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
    state.caller_type_score = new_score
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
    # Module 4 — human role evidence accumulation
    # Accumulate regardless of current classification; emit role only
    # when entity_type is confirmed HUMAN so no evidence is lost during
    # the AMBIGUOUS resolution window.
    # ----------------------------------------------------------------
    from ai_telemetry.role_inference import update_human_role, get_human_role
    update_human_role(entity_key, session_id, prompt_text)
    if entity_type == "HUMAN":
        human_role, human_conf = get_human_role(entity_key)
        event.human_inferred_role = human_role
        event.human_role_confidence = round(human_conf, 4) if human_conf else None
    else:
        event.human_inferred_role = None
        event.human_role_confidence = None

    # ----------------------------------------------------------------
    # Annotate the event
    # ----------------------------------------------------------------
    event.entity_type = entity_type
    event.entity_subtype = entity_subtype
    event.caller_type_score = round(new_score, 4)
    event.classification_signals = "|".join(classification_signals) if classification_signals else None
    event.behavioral_alerts = "|".join(alerts) if alerts else None
    event.session_turn_index = turn_index
    event.inter_turn_latency_ms = inter_turn_ms
    event.prompt_submission_time_ms = submission_ms
