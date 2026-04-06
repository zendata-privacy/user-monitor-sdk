"""
Session state tracking for entity classification.

Maintains per-session turn history (timing, prompt text, response times)
and per-entity call rate history used by the classifier rules.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TurnRecord:
    """One user→assistant exchange within a session."""
    turn_index: int
    prompt_text: str
    prompt_chars: int
    submission_time_ms: int    # when the user submitted the prompt
    response_time_ms: Optional[int] = None  # when the system returned its response


@dataclass
class SessionState:
    """Per-session mutable state accumulated across turns."""
    session_id: str
    entity_key: str            # user_id or API key — the identity being tracked
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    turns: List[TurnRecord] = field(default_factory=list)
    entity_type: str = "UNKNOWN"
    entity_subtype: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    # D3: counts how many persona-switch injections have been seen this session
    role_play_count: int = 0
    # D4: sensitivity score per turn (float 0–10); used to detect escalating probes
    probe_scores: List[float] = field(default_factory=list)
    # Module 2 — continuous caller_type score (0.0 = definite agent, 1.0 = definite human)
    # Maintained as an exponential moving average across turns; resolves in 3–5 calls.
    caller_type_score: float = 0.5


@dataclass
class EntityBaseline:
    """Rolling historical baseline per entity for drift detection (D1)."""
    entity_key: str
    risk_scores: List[float] = field(default_factory=list)

    def add_risk_score(self, score: float) -> None:
        self.risk_scores.append(score)
        # Keep only the last 100 observations
        if len(self.risk_scores) > 100:
            self.risk_scores.pop(0)

    def mean_and_std(self) -> tuple:
        """Return (mean, std) or (None, None) if fewer than 3 data points."""
        if len(self.risk_scores) < 3:
            return None, None
        mean = sum(self.risk_scores) / len(self.risk_scores)
        variance = sum((v - mean) ** 2 for v in self.risk_scores) / len(self.risk_scores)
        return mean, math.sqrt(variance)


class SessionStore:
    """Thread-safe, in-memory store for session states and entity-level tracking."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._baselines: Dict[str, EntityBaseline] = {}
        # Maps entity_key → list of call timestamp_ms (pruned to last 24 h)
        self._entity_calls: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str, entity_key: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    entity_key=entity_key,
                )
            return self._sessions[session_id]

    def update_session(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state

    def count_active_sessions(self, entity_key: str) -> int:
        """Number of sessions currently associated with this entity key (S4)."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.entity_key == entity_key)

    # ------------------------------------------------------------------
    # Baseline access
    # ------------------------------------------------------------------

    def get_baseline(self, entity_key: str) -> EntityBaseline:
        with self._lock:
            if entity_key not in self._baselines:
                self._baselines[entity_key] = EntityBaseline(entity_key=entity_key)
            return self._baselines[entity_key]

    def update_baseline(self, entity_key: str, risk_score: float) -> None:
        with self._lock:
            if entity_key not in self._baselines:
                self._baselines[entity_key] = EntityBaseline(entity_key=entity_key)
            self._baselines[entity_key].add_risk_score(risk_score)

    # ------------------------------------------------------------------
    # Call rate tracking (A1 / A2)
    # ------------------------------------------------------------------

    def record_call(self, entity_key: str, timestamp_ms: int) -> None:
        """Record one call for this entity; prunes calls older than 24 hours."""
        with self._lock:
            if entity_key not in self._entity_calls:
                self._entity_calls[entity_key] = []
            self._entity_calls[entity_key].append(timestamp_ms)
            cutoff = timestamp_ms - 86_400_000  # 24 h in ms
            self._entity_calls[entity_key] = [
                t for t in self._entity_calls[entity_key] if t >= cutoff
            ]

    def get_call_timestamps(self, entity_key: str) -> List[int]:
        with self._lock:
            return list(self._entity_calls.get(entity_key, []))


# Module-level singleton — shared across all TelemetryClient instances
_global_store = SessionStore()


def get_session_store() -> SessionStore:
    return _global_store
