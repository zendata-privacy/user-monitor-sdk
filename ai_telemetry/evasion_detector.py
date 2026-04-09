"""
Signal Evasion Detection — Module 9

Detects deliberate mimicry of legitimate behavioral signals.  A sophisticated
attacker who knows the SDK is deployed can fake individual signals; this module
looks for statistical inconsistencies across the full signal set that reveal
calibration effort.

Detection Methods
-----------------
  LATENCY_MIMICRY      Human-range inter-turn latency (5–60 s) with zero typos
                       AND perfect prompt structure — inconsistent combination.
  TYPO_INJECTION       Typos at suspiciously regular intervals / character positions
                       (low coefficient of variation → non-random origin).
  TEMPLATE_VARIATION   Surface variation masking structurally identical templates;
                       caught by n-gram Jaccard similarity on consecutive prompts.
  TOOL_SEQUENCE        Novel tool-use sequences within allowed scope that never
                       appear in any baseline agent of the same role.
  CONTENT_BYPASS       Anomalously low ZASP safety score (< 3.0) while all
                       observable behavioral signals appear clean — content-layer
                       attack through a compliant-looking behavioral envelope.

Cross-cutting
-------------
  SLOW_DRIFT_MASK      Gradual safety-score drift co-occurring with simultaneous
                       latency recalibration; catch rate low without periodic ZASP
                       re-evaluation.
  COORDINATED_EVASION  Multi-agent decomposition where no individual agent triggers
                       per-agent thresholds; topology module partially addresses this.

Inconsistency flag
------------------
When INCONSISTENT_SIGNALS is set:
  • All caller confidence scores are discounted 30 %.
  • Session is tagged for mandatory manual review.
  • The flag itself is a behavioral signal — legitimate callers almost never
    exhibit the combination of near-human timing with computational precision.

Acknowledged Limitations (documented for diligence purposes)
-------------------------------------------------------------
  1. Slow drift with signal masking (catch rate: low without ZASP re-evaluation)
  2. Coordinated multi-agent decomposition (detection unreliable for sophisticated attacks)
  3. Content layer attacks with perfect behavioral compliance (requires ZASP red-teaming)

Multi-tenancy Governance
------------------------
  Data element            Aggregation status   Basis
  ─────────────────────── ──────────────────── ─────────────────────────────────────────
  Raw prompt / response   NEVER                PII + customer-confidential content
  System prompt content   NEVER                Proprietary agent instructions
  Agent topology edges    NEVER                Customer's network structure is proprietary
  System prompt embedding Anonymized           Math rep only; no reconstruction path
  Safety score dists.     Anonymized           Numeric per agent type/role; no customer ID
  Tool call fingerprints  Anonymized           Tool names are not customer-confidential
  Temporal rhythm         Anonymized           Numeric distributions only
  Drift event metadata    Anonymized           Categorical signals; no customer/agent ID

Governed by DPA (GDPR Art.6(1)(f) legitimate interests), per-customer opt-out,
and data-residency region configuration (EU / DE / UK / DEFAULT).
"""

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HUMAN_LATENCY_MIN_MS: int   = 5_000      # 5 s — lower bound for "human-range"
_HUMAN_LATENCY_MAX_MS: int   = 60_000     # 60 s — upper bound
_LATENCY_CONSISTENCY_WINDOW: int = 5      # consecutive turns required to fire
_LATENCY_PRECISION_CV_MAX: float = 0.25   # CV below this → suspiciously consistent
_TYPO_REGULARITY_CV_THRESHOLD: float = 0.30  # low CV on typo intervals → injection
_TEMPLATE_NGRAM_THRESHOLD: float = 0.70   # Jaccard bigram similarity above this
_CONTENT_BYPASS_SAFETY_THRESHOLD: float = 3.0   # ZASP below this while signals clean
_SLOW_DRIFT_WINDOW: int  = 10             # turns for slow-drift analysis
_MIN_TURNS_FOR_DETECTION: int = 3         # minimum turns before any flag fires
_CONFIDENCE_DISCOUNT: float = 0.30        # applied when INCONSISTENT_SIGNALS flagged

# Tool pairs that are unusually powerful in combination
_SUSPICIOUS_TOOL_PAIRS: Set[Tuple[str, str]] = {
    ("get_customer_data",   "transfer_funds"),
    ("search_documents",    "execute_trade"),
    ("get_account_balance", "approve_invoice"),
    ("retrieve_hr_record",  "modify_acl"),
    ("read_compliance_data","delete_record"),
    ("get_user_profile",    "send_email"),
    ("query_database",      "run_script"),
}

# Signal type constants (also used as keys in evidence dicts)
LATENCY_MIMICRY      = "LATENCY_MIMICRY"
TYPO_INJECTION       = "TYPO_INJECTION"
TEMPLATE_VARIATION   = "TEMPLATE_VARIATION"
TOOL_SEQUENCE        = "TOOL_SEQUENCE"
CONTENT_BYPASS       = "CONTENT_BYPASS"
SLOW_DRIFT_MASK      = "SLOW_DRIFT_MASK"
COORDINATED_EVASION  = "COORDINATED_EVASION"
INCONSISTENT_SIGNALS = "INCONSISTENT_SIGNALS"

# These limitations are documented as roadmap items (cannot be reliably caught today)
ACKNOWLEDGED_LIMITATIONS: List[Dict] = [
    {
        "id":          "ALM-1",
        "name":        "Slow drift with signal masking",
        "description": (
            "An agent that drifts gradually over weeks while simultaneously "
            "adjusting its latency and typo patterns to stay within human-range "
            "thresholds.  Catch rate: low without ZASP periodic re-evaluation."
        ),
        "roadmap_item": "Adversarial probe replay loop (Roadmap Item 1)",
    },
    {
        "id":          "ALM-2",
        "name":        "Coordinated multi-agent evasion",
        "description": (
            "Multiple agents in a network that collectively mimic normal behavior "
            "while individually executing fragments of a harmful workflow. The "
            "decomposition attack class is designed to evade per-agent monitoring. "
            "Topology module (Module 8) partially addresses this."
        ),
        "roadmap_item": "Enhanced evasion detection (Roadmap Item 3)",
    },
    {
        "id":          "ALM-3",
        "name":        "Content layer attacks with perfect behavioral compliance",
        "description": (
            "An agent that produces harmful outputs while scoring 4.0+ on all "
            "behavioral signals.  Requires the attacker to find prompts that "
            "elicit harmful responses from the specific model in the specific "
            "deployment context — exactly what ZASP red teaming discovers pre-deployment."
        ),
        "roadmap_item": "ZASP red-team baseline (pre-deployment requirement)",
    },
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvasionSignal:
    """One detected evasion pattern within a session."""
    signal_type: str
    confidence: float        # 0.0–1.0
    evidence: Dict[str, Any]
    detected_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class SessionEvasionState:
    """All evasion-detection state for a single session."""
    session_id: str

    # Per-turn signal collections
    latency_samples_ms:     List[float]        = field(default_factory=list)
    typos_per_turn:         List[int]          = field(default_factory=list)
    typo_positions:         List[List[int]]    = field(default_factory=list)   # char positions
    prompt_lengths:         List[int]          = field(default_factory=list)
    prompt_bigram_sets:     List[Set[str]]     = field(default_factory=list)   # for Jaccard
    template_hashes:        List[str]          = field(default_factory=list)
    tool_sequences:         List[List[str]]    = field(default_factory=list)
    safety_scores:          List[float]        = field(default_factory=list)
    entity_type_history:    List[str]          = field(default_factory=list)
    caller_type_scores:     List[float]        = field(default_factory=list)
    behavioral_alert_counts: List[int]         = field(default_factory=list)

    # Detected signals
    evasion_signals:        List[EvasionSignal] = field(default_factory=list)

    # Outcome flags
    inconsistent_signals_flag: bool  = False
    confidence_discount:       float = 0.0      # 0.30 when flagged
    requires_manual_review:    bool  = False

    last_updated_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class EvasionReport:
    """Aggregate evasion detection snapshot — used by dashboard and API."""
    total_sessions_analyzed:  int
    flagged_sessions:         int
    signal_counts:            Dict[str, int]
    high_confidence_evasion:  List[Dict]
    acknowledged_limitations: List[Dict]
    governance_status:        Dict
    generated_at_iso:         str


# ---------------------------------------------------------------------------
# Multi-tenancy governance
# ---------------------------------------------------------------------------

# Data elements and their aggregation permission
_AGGREGATION_POLICY: Dict[str, str] = {
    "raw_prompt":              "NEVER",
    "raw_response":            "NEVER",
    "system_prompt_content":   "NEVER",
    "agent_topology_edges":    "NEVER",
    "system_prompt_embedding": "ANONYMIZED",
    "safety_score_distribution": "ANONYMIZED",
    "tool_call_fingerprints":  "ANONYMIZED",
    "temporal_rhythm_patterns": "ANONYMIZED",
    "drift_event_metadata":    "ANONYMIZED",
}

_VALID_REGIONS = {"DEFAULT", "EU", "DE", "UK"}


@dataclass
class CustomerAggregationPolicy:
    customer_id: str
    opted_out:          bool = False
    data_residency_region: str = "DEFAULT"

    # Fine-grained overrides (all True by default; honoured only when not opted_out)
    allow_embedding_aggregation:      bool = True
    allow_safety_score_aggregation:   bool = True
    allow_tool_name_aggregation:      bool = True
    allow_temporal_pattern_aggregation: bool = True
    allow_drift_metadata_aggregation:  bool = True


class TenancyGovernance:
    """Thread-safe multi-tenancy data aggregation governance.

    Enforces the aggregation policy table from the SDK spec.
    Raw content, system prompt text, and topology edges are NEVER aggregated
    regardless of customer settings — this is hard-coded, not configurable.
    """

    _NEVER_AGGREGATE: Set[str] = {
        "raw_prompt", "raw_response", "system_prompt_content", "agent_topology_edges",
    }

    def __init__(self) -> None:
        self._policies: Dict[str, CustomerAggregationPolicy] = {}
        self._lock = threading.Lock()

    def get_or_create_policy(self, customer_id: str) -> CustomerAggregationPolicy:
        with self._lock:
            if customer_id not in self._policies:
                self._policies[customer_id] = CustomerAggregationPolicy(
                    customer_id=customer_id
                )
            return self._policies[customer_id]

    def set_opt_out(self, customer_id: str, opted_out: bool) -> None:
        """Customer opt-out from contributing anonymized signals to the aggregation corpus."""
        with self._lock:
            p = self._policies.setdefault(
                customer_id, CustomerAggregationPolicy(customer_id=customer_id)
            )
            p.opted_out = opted_out

    def set_data_residency(self, customer_id: str, region: str) -> None:
        """Configure data residency region.  One of: EU, DE, UK, DEFAULT.

        Aggregation is performed within the same region; cross-region aggregation
        requires explicit customer consent (handled separately).
        """
        if region not in _VALID_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Must be one of {_VALID_REGIONS}.")
        with self._lock:
            p = self._policies.setdefault(
                customer_id, CustomerAggregationPolicy(customer_id=customer_id)
            )
            p.data_residency_region = region

    def can_aggregate(self, customer_id: str, data_element: str) -> bool:
        """Return True if *data_element* may be contributed to the cross-customer corpus.

        Hard rule: elements in _NEVER_AGGREGATE always return False regardless of policy.
        """
        if data_element in self._NEVER_AGGREGATE:
            return False
        with self._lock:
            p = self._policies.get(customer_id)
        if p is None:
            return True   # default: allow anonymized aggregation
        if p.opted_out:
            return False
        # Fine-grained check
        _element_flag_map = {
            "system_prompt_embedding":      p.allow_embedding_aggregation,
            "safety_score_distribution":    p.allow_safety_score_aggregation,
            "tool_call_fingerprints":       p.allow_tool_name_aggregation,
            "temporal_rhythm_patterns":     p.allow_temporal_pattern_aggregation,
            "drift_event_metadata":         p.allow_drift_metadata_aggregation,
        }
        return _element_flag_map.get(data_element, True)

    def get_policy(self, customer_id: str) -> Dict:
        p = self.get_or_create_policy(customer_id)
        return {
            "customer_id":            p.customer_id,
            "opted_out":              p.opted_out,
            "data_residency_region":  p.data_residency_region,
            "contractual_instruments": [
                "Data Processing Agreement (DPA) — GDPR Art.6(1)(f) legitimate interests",
                "Aggregation Opt-Out (per-customer, revocable)",
                "Data Residency Configuration (EU / DE / UK / DEFAULT)",
            ],
            "aggregation_policy":     _AGGREGATION_POLICY,
        }

    def get_governance_summary(self) -> Dict:
        with self._lock:
            policies = list(self._policies.values())
        opted_out = sum(1 for p in policies if p.opted_out)
        return {
            "total_customers":    len(policies),
            "opted_out":          opted_out,
            "opted_in":           len(policies) - opted_out,
            "aggregation_policy": _AGGREGATION_POLICY,
            "never_aggregate":    sorted(self._NEVER_AGGREGATE),
            "contractual_basis":  "DPA (GDPR Art.6(1)(f) legitimate interests)",
            "opt_out_effect":     "Reduces benchmark calibration quality for that deployment; "
                                  "does not affect core SDK functionality.",
        }


_global_governance = TenancyGovernance()


def get_tenancy_governance() -> TenancyGovernance:
    return _global_governance


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


class SessionEvasionStore:
    """Thread-safe in-memory store for per-session evasion state."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionEvasionState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionEvasionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionEvasionState(session_id=session_id)
            return self._sessions[session_id]

    def update(self, state: SessionEvasionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state

    def all_states(self) -> List[SessionEvasionState]:
        with self._lock:
            return list(self._sessions.values())

    def flagged_states(self) -> List[SessionEvasionState]:
        return [s for s in self.all_states() if s.inconsistent_signals_flag]


_global_evasion_store = SessionEvasionStore()


def get_evasion_store() -> SessionEvasionStore:
    return _global_evasion_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coefficient_of_variation(values: List[float]) -> Optional[float]:
    """CV = σ / μ.  Returns None if fewer than 2 values or mean == 0."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def _bigrams(text: str) -> Set[str]:
    """Character bigrams of a whitespace-normalised, lower-cased text."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i: i + 2] for i in range(len(t) - 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _count_typos_in_prompt(prompt: str) -> Tuple[int, List[int]]:
    """Estimate typo count from common English mis-spelling / correction patterns.

    Heuristic: sequences like 'teh', 'adn', 'taht', repeated characters adjacent
    to corrections such as '[word] [same-word-again]', or explicit correction phrases
    like 'sorry, I meant'.  Returns (count, list_of_char_positions).
    """
    _TYPO_PATTERNS = [
        r"\b(teh|adn|taht|hte|waht|fo|ot|ti|nad|ahve|thier|tis|nto|si|yuo)\b",
        r"\b(\w+)\s+\1\b",                       # repeated word (accidental dup)
        r"(?i)\b(sorry,?\s+i\s+meant|oops|correction:?)\b",
    ]
    positions: List[int] = []
    for pat in _TYPO_PATTERNS:
        for m in re.finditer(pat, prompt, re.IGNORECASE):
            positions.append(m.start())
    return len(positions), positions


def _is_perfectly_structured(prompt: str) -> bool:
    """Return True if the prompt looks like a machine-generated imperative template."""
    _IMPERATIVE_MARKERS = [
        r"\b(return only|do not include|format as|your task is|respond in json"
        r"|output only|provide only|do not add|output as|response must be"
        r"|always respond|never include|you must return|you must respond)\b",
    ]
    for pat in _IMPERATIVE_MARKERS:
        if re.search(pat, prompt, re.IGNORECASE):
            return True
    # Heuristic: very short sentences, all punctuated, no casual language
    words = prompt.split()
    if not words:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    return avg_word_len >= 6.5 and "\n" not in prompt and "?" not in prompt


# ---------------------------------------------------------------------------
# Evasion Detector — core engine
# ---------------------------------------------------------------------------


class EvasionDetector:
    """Analyzes per-session signal patterns for deliberate mimicry.

    Call ``analyze_event(event_dict)`` for each incoming event.  The detector
    maintains per-session state and emits ``EvasionSignal`` objects when
    inconsistencies are detected.  The ``INCONSISTENT_SIGNALS`` composite flag
    is set when any two or more signals fire within the same session.
    """

    def __init__(self, store: Optional[SessionEvasionStore] = None) -> None:
        self._store = store or _global_evasion_store

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze_event(self, event: Dict[str, Any]) -> SessionEvasionState:
        """Process one event dict and return the updated session evasion state."""
        session_id = event.get("session_id") or event.get("trace_id") or "default"
        state = self._store.get_or_create(session_id)

        # ---- Extract per-turn signals --------------------------------
        latency = event.get("inter_turn_latency_ms") or event.get("latency_ms")
        if latency is not None:
            try:
                state.latency_samples_ms.append(float(latency))
            except (TypeError, ValueError):
                pass

        prompt = event.get("prompt_text") or ""
        if isinstance(prompt, list):
            # input_messages format
            prompt = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in prompt
            )
        if prompt:
            typo_count, positions = _count_typos_in_prompt(prompt)
            state.typos_per_turn.append(typo_count)
            state.typo_positions.append(positions)
            state.prompt_lengths.append(len(prompt))
            state.prompt_bigram_sets.append(_bigrams(prompt[:500]))  # cap at 500 chars

        tmpl_hash = event.get("prompt_template_hash") or ""
        state.template_hashes.append(tmpl_hash)

        tools_raw = event.get("tools_called") or ""
        tools = [t.strip() for t in tools_raw.split("|") if t.strip()] if tools_raw else []
        state.tool_sequences.append(tools)

        safety = event.get("zasp_score") or event.get("safety_score")
        if safety is not None:
            try:
                state.safety_scores.append(float(safety))
            except (TypeError, ValueError):
                pass

        entity_type = event.get("entity_type") or "UNKNOWN"
        state.entity_type_history.append(entity_type)

        cts = event.get("caller_type_score")
        if cts is not None:
            try:
                state.caller_type_scores.append(float(cts))
            except (TypeError, ValueError):
                pass

        alerts_raw = event.get("behavioral_alerts") or event.get("drift_signals") or ""
        alert_count = len([a for a in alerts_raw.split("|") if a.strip()]) if alerts_raw else 0
        state.behavioral_alert_counts.append(alert_count)

        # ---- Run detectors (only after minimum turn count) -----------
        n_turns = len(state.entity_type_history)
        if n_turns >= _MIN_TURNS_FOR_DETECTION:
            self._run_detectors(state, prompt, tools)

        state.last_updated_ms = int(time.time() * 1000)
        self._store.update(state)
        return state

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _run_detectors(
        self,
        state: SessionEvasionState,
        current_prompt: str,
        current_tools: List[str],
    ) -> None:
        new_signals: List[EvasionSignal] = []

        sig = self._detect_latency_mimicry(state)
        if sig:
            new_signals.append(sig)

        sig = self._detect_typo_injection(state)
        if sig:
            new_signals.append(sig)

        sig = self._detect_template_variation(state)
        if sig:
            new_signals.append(sig)

        sig = self._detect_tool_sequence(state, current_tools)
        if sig:
            new_signals.append(sig)

        sig = self._detect_content_bypass(state)
        if sig:
            new_signals.append(sig)

        sig = self._detect_slow_drift_mask(state)
        if sig:
            new_signals.append(sig)

        # Deduplicate by signal_type (keep highest confidence)
        existing_types = {s.signal_type for s in state.evasion_signals}
        for sig in new_signals:
            if sig.signal_type not in existing_types:
                state.evasion_signals.append(sig)
                existing_types.add(sig.signal_type)
            else:
                # Update confidence if higher
                for existing in state.evasion_signals:
                    if existing.signal_type == sig.signal_type and sig.confidence > existing.confidence:
                        existing.confidence = sig.confidence
                        existing.evidence = sig.evidence
                        break

        # Apply composite inconsistency flag
        self._apply_inconsistency_flag(state)

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _detect_latency_mimicry(
        self, state: SessionEvasionState
    ) -> Optional[EvasionSignal]:
        """Human-range latency + zero typos + perfect structure = LATENCY_MIMICRY."""
        latencies = state.latency_samples_ms
        if len(latencies) < _LATENCY_CONSISTENCY_WINDOW:
            return None

        window = latencies[-_LATENCY_CONSISTENCY_WINDOW:]
        in_human_range = all(_HUMAN_LATENCY_MIN_MS <= v <= _HUMAN_LATENCY_MAX_MS for v in window)
        if not in_human_range:
            return None

        # All window turns have zero typos
        typo_window = state.typos_per_turn[-_LATENCY_CONSISTENCY_WINDOW:]
        all_zero_typos = all(t == 0 for t in typo_window)
        if not all_zero_typos:
            return None

        # Latency CV suspiciously low (too consistent for a human)
        cv = _coefficient_of_variation(window)
        if cv is None or cv > _LATENCY_PRECISION_CV_MAX:
            return None

        # Check at least one recent prompt looks perfectly structured
        recent_prompts_structured = sum(
            1 for i in range(-min(3, len(state.prompt_bigram_sets)), 0)
            if len(state.prompt_bigram_sets) > 0 and
            i < len(state.entity_type_history) and
            state.entity_type_history[i] in ("HUMAN", "AMBIGUOUS")
        )

        confidence = min(0.9, 0.5 + (1 - cv) * 0.4 + (0.1 if all_zero_typos else 0))
        return EvasionSignal(
            signal_type=LATENCY_MIMICRY,
            confidence=confidence,
            evidence={
                "latency_window_ms":  window,
                "cv":                 round(cv, 4),
                "all_zero_typos":     all_zero_typos,
                "threshold_cv":       _LATENCY_PRECISION_CV_MAX,
                "human_range_ms":     [_HUMAN_LATENCY_MIN_MS, _HUMAN_LATENCY_MAX_MS],
            },
        )

    def _detect_typo_injection(
        self, state: SessionEvasionState
    ) -> Optional[EvasionSignal]:
        """Typos at regular intervals (non-random distribution) = TYPO_INJECTION."""
        # Find turns that had at least one typo
        typo_turns = [i for i, t in enumerate(state.typos_per_turn) if t > 0]
        if len(typo_turns) < 3:
            return None

        # Compute intervals between typo turns
        intervals = [float(typo_turns[i + 1] - typo_turns[i]) for i in range(len(typo_turns) - 1)]
        if not intervals:
            return None

        cv = _coefficient_of_variation(intervals)
        if cv is None or cv >= _TYPO_REGULARITY_CV_THRESHOLD:
            return None   # intervals are irregular — looks natural

        # Also check: agent-classified entity having typos is suspicious
        typo_in_agent_session = any(
            state.entity_type_history[i] in ("API CALL", "AGENT")
            for i in typo_turns
            if i < len(state.entity_type_history)
        )

        confidence = min(0.85, 0.55 + (1 - cv / _TYPO_REGULARITY_CV_THRESHOLD) * 0.3
                          + (0.1 if typo_in_agent_session else 0))
        return EvasionSignal(
            signal_type=TYPO_INJECTION,
            confidence=confidence,
            evidence={
                "typo_turn_indices": typo_turns,
                "intervals":         intervals,
                "cv":                round(cv, 4),
                "threshold_cv":      _TYPO_REGULARITY_CV_THRESHOLD,
                "agent_classified_with_typos": typo_in_agent_session,
            },
        )

    def _detect_template_variation(
        self, state: SessionEvasionState
    ) -> Optional[EvasionSignal]:
        """Surface variation masking structurally similar prompts = TEMPLATE_VARIATION."""
        bigram_sets = state.prompt_bigram_sets
        if len(bigram_sets) < 3:
            return None

        # Compare consecutive pairs
        similarities = []
        for i in range(len(bigram_sets) - 1):
            j = i + 1
            if bigram_sets[i] and bigram_sets[j]:
                similarities.append(_jaccard(bigram_sets[i], bigram_sets[j]))

        if not similarities:
            return None

        high_sim_count = sum(1 for s in similarities if s >= _TEMPLATE_NGRAM_THRESHOLD)
        if high_sim_count < 2:
            return None

        # If template hashes are also identical despite different surface text → stronger signal
        template_repeat = (
            len(set(h for h in state.template_hashes if h)) > 0 and
            len(set(state.template_hashes)) < len(state.template_hashes) / 2
        )

        avg_sim = sum(similarities) / len(similarities)
        confidence = min(0.80, 0.45 + avg_sim * 0.35 + (0.1 if template_repeat else 0))
        return EvasionSignal(
            signal_type=TEMPLATE_VARIATION,
            confidence=confidence,
            evidence={
                "consecutive_similarities": [round(s, 3) for s in similarities],
                "high_sim_count":            high_sim_count,
                "avg_similarity":            round(avg_sim, 3),
                "threshold":                 _TEMPLATE_NGRAM_THRESHOLD,
                "template_hash_repetition":  template_repeat,
            },
        )

    def _detect_tool_sequence(
        self,
        state: SessionEvasionState,
        current_tools: List[str],
    ) -> Optional[EvasionSignal]:
        """Novel tool-use sequence patterns not seen in any baseline = TOOL_SEQUENCE."""
        if not current_tools or len(current_tools) < 2:
            return None

        # Build pairs from current call
        current_pairs = {(current_tools[i], current_tools[i + 1])
                         for i in range(len(current_tools) - 1)}
        suspicious = current_pairs & _SUSPICIOUS_TOOL_PAIRS
        if not suspicious:
            return None

        # Cross-session frequency check: if the same pair appeared before in this
        # session, escalate confidence
        past_all_tools = [t for seq in state.tool_sequences[:-1] for t in seq]
        pair_repeated = any(
            past_all_tools.count(a) > 0 and past_all_tools.count(b) > 0
            for a, b in suspicious
        )

        confidence = 0.65 if not pair_repeated else 0.78
        return EvasionSignal(
            signal_type=TOOL_SEQUENCE,
            confidence=confidence,
            evidence={
                "suspicious_pairs":  [list(p) for p in suspicious],
                "current_tools":     current_tools,
                "repeated_in_session": pair_repeated,
                "known_suspicious_pairs": [list(p) for p in _SUSPICIOUS_TOOL_PAIRS],
            },
        )

    def _detect_content_bypass(
        self, state: SessionEvasionState
    ) -> Optional[EvasionSignal]:
        """Low ZASP safety score while behavioral signals appear clean = CONTENT_BYPASS."""
        if len(state.safety_scores) < _MIN_TURNS_FOR_DETECTION:
            return None

        recent_safety = state.safety_scores[-5:]
        low_safety_count = sum(1 for s in recent_safety if s < _CONTENT_BYPASS_SAFETY_THRESHOLD)
        if low_safety_count < 2:
            return None

        # Check behavioral signals appear clean (entity not classified as AGENT with alerts)
        recent_alerts = state.behavioral_alert_counts[-5:]
        avg_alerts = sum(recent_alerts) / len(recent_alerts) if recent_alerts else 0
        recent_entity_types = state.entity_type_history[-5:]
        classified_as_human_or_ambiguous = sum(
            1 for e in recent_entity_types if e in ("HUMAN", "AMBIGUOUS")
        )

        # The signal fires when safety is low AND behavioral appearance is "clean"
        if classified_as_human_or_ambiguous < 2 or avg_alerts > 1:
            return None

        avg_safety = sum(recent_safety) / len(recent_safety)
        confidence = min(0.90, 0.60 + (1 - avg_safety / 10.0) * 0.3)
        return EvasionSignal(
            signal_type=CONTENT_BYPASS,
            confidence=confidence,
            evidence={
                "recent_safety_scores":       recent_safety,
                "avg_safety_score":           round(avg_safety, 3),
                "threshold":                  _CONTENT_BYPASS_SAFETY_THRESHOLD,
                "avg_behavioral_alerts":      round(avg_alerts, 2),
                "classified_as_clean_entity": classified_as_human_or_ambiguous,
                "note": (
                    "Safety score is the evasion-resistant signal — an attacker can "
                    "mimic latency and typos but cannot easily fake ZASP safety scores "
                    "without genuinely changing the agent's underlying behavior."
                ),
            },
        )

    def _detect_slow_drift_mask(
        self, state: SessionEvasionState
    ) -> Optional[EvasionSignal]:
        """Gradual safety drift co-occurring with latency recalibration = SLOW_DRIFT_MASK."""
        if len(state.safety_scores) < _SLOW_DRIFT_WINDOW:
            return None
        if len(state.latency_samples_ms) < _SLOW_DRIFT_WINDOW:
            return None

        safety_window  = state.safety_scores[-_SLOW_DRIFT_WINDOW:]
        latency_window = state.latency_samples_ms[-_SLOW_DRIFT_WINDOW:]

        # Safety drift: downward trend
        first_half_safety  = safety_window[:_SLOW_DRIFT_WINDOW // 2]
        second_half_safety = safety_window[_SLOW_DRIFT_WINDOW // 2:]
        safety_dropping = (
            sum(second_half_safety) / len(second_half_safety) <
            sum(first_half_safety) / len(first_half_safety) - 0.5
        )
        if not safety_dropping:
            return None

        # Latency adjusting: CV dropping over the window
        first_half_lat  = latency_window[:_SLOW_DRIFT_WINDOW // 2]
        second_half_lat = latency_window[_SLOW_DRIFT_WINDOW // 2:]
        cv_first  = _coefficient_of_variation(first_half_lat)
        cv_second = _coefficient_of_variation(second_half_lat)
        if cv_first is None or cv_second is None:
            return None
        latency_becoming_more_consistent = cv_second < cv_first * 0.7

        if not latency_becoming_more_consistent:
            return None

        confidence = 0.55  # documented as low catch-rate
        return EvasionSignal(
            signal_type=SLOW_DRIFT_MASK,
            confidence=confidence,
            evidence={
                "safety_first_half_mean":   round(sum(first_half_safety) / len(first_half_safety), 3),
                "safety_second_half_mean":  round(sum(second_half_safety) / len(second_half_safety), 3),
                "latency_cv_first_half":    round(cv_first, 4),
                "latency_cv_second_half":   round(cv_second, 4),
                "window_turns":             _SLOW_DRIFT_WINDOW,
                "limitation":              "ALM-1 — catch rate low without periodic ZASP re-evaluation",
            },
        )

    # ------------------------------------------------------------------
    # Composite inconsistency flag
    # ------------------------------------------------------------------

    def _apply_inconsistency_flag(self, state: SessionEvasionState) -> None:
        """Set INCONSISTENT_SIGNALS if ≥2 signals fired, or any single HIGH-confidence signal."""
        high_confidence = [s for s in state.evasion_signals if s.confidence >= 0.75]
        multiple_signals = len(state.evasion_signals) >= 2

        if high_confidence or multiple_signals:
            state.inconsistent_signals_flag = True
            state.confidence_discount       = _CONFIDENCE_DISCOUNT
            state.requires_manual_review    = True
        else:
            state.inconsistent_signals_flag = False
            state.confidence_discount       = 0.0

    def apply_discount(self, caller_type_score: float, session_id: str) -> float:
        """Apply the 30 % confidence discount if the session is flagged.

        Call this after entity classification to get the adjusted score.
        """
        state = self._store.get_or_create(session_id)
        if state.inconsistent_signals_flag:
            # Discount pulls the score toward 0.5 (uncertainty)
            return caller_type_score - _CONFIDENCE_DISCOUNT * (caller_type_score - 0.5)
        return caller_type_score


# ---------------------------------------------------------------------------
# Batch analysis — works on a list of event dicts (e.g. from JSONL log)
# ---------------------------------------------------------------------------


def analyze_events_batch(events: List[Dict[str, Any]]) -> List[SessionEvasionState]:
    """Run evasion detection across a list of events.

    Groups events by session_id and processes each chronologically.
    Returns one state object per session.
    """
    detector = EvasionDetector()
    for event in sorted(events, key=lambda e: e.get("timestamp_ms", 0)):
        detector.analyze_event(event)
    return detector._store.all_states()


# ---------------------------------------------------------------------------
# Dashboard / API surface
# ---------------------------------------------------------------------------


def get_evasion_report(events: Optional[List[Dict]] = None) -> EvasionReport:
    """Build an EvasionReport from the current store state (optionally refreshed from events)."""
    from datetime import datetime, timezone

    if events:
        analyze_events_batch(events)

    store = _global_evasion_store
    all_states = store.all_states()
    flagged    = store.flagged_states()

    signal_counts: Dict[str, int] = {}
    high_conf: List[Dict] = []

    for state in all_states:
        for sig in state.evasion_signals:
            signal_counts[sig.signal_type] = signal_counts.get(sig.signal_type, 0) + 1
            if sig.confidence >= 0.70:
                high_conf.append({
                    "session_id":   state.session_id,
                    "signal_type":  sig.signal_type,
                    "confidence":   round(sig.confidence, 3),
                    "evidence":     sig.evidence,
                    "detected_at_iso": _ms_to_iso(sig.detected_at_ms),
                    "requires_manual_review": state.requires_manual_review,
                })

    high_conf.sort(key=lambda x: -x["confidence"])

    return EvasionReport(
        total_sessions_analyzed  = len(all_states),
        flagged_sessions         = len(flagged),
        signal_counts            = signal_counts,
        high_confidence_evasion  = high_conf[:50],
        acknowledged_limitations = ACKNOWLEDGED_LIMITATIONS,
        governance_status        = _global_governance.get_governance_summary(),
        generated_at_iso         = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def get_session_evasion_detail(session_id: str) -> Optional[Dict]:
    """Return a full evasion-detail dict for one session, or None if unknown."""
    state = _global_evasion_store.get_or_create(session_id)
    if not state.evasion_signals and not state.inconsistent_signals_flag:
        return None
    return {
        "session_id":               state.session_id,
        "inconsistent_signals_flag": state.inconsistent_signals_flag,
        "confidence_discount":       state.confidence_discount,
        "requires_manual_review":    state.requires_manual_review,
        "turns_analyzed":            len(state.entity_type_history),
        "evasion_signals": [
            {
                "signal_type":   s.signal_type,
                "confidence":    round(s.confidence, 3),
                "evidence":      s.evidence,
                "detected_at":   _ms_to_iso(s.detected_at_ms),
            }
            for s in state.evasion_signals
        ],
        "last_updated_iso": _ms_to_iso(state.last_updated_ms),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ms_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except Exception:
        return None
