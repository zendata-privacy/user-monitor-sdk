"""Behavioural baseline — the soft boundary.

Where the manifest declares what an agent *may* reach, this records what it
actually *has* reached: which sources, how often, in what tool sequences, over a
rolling window. A source that is inside the manifest but has never once appeared
in thirty days is not a violation, but it is worth a look — so these signals are
HIGH, not CRITICAL, and they are gated behind a session count that has to be
earned.

────────────────────────────────────────────────────────────────────────────────
WHY THIS IS LOADED, NOT LEARNED IN PROCESS

Every store in this SDK — ``SessionStore``, ``AgentFingerprintStore``, the drift
store — is a module-level dict behind a lock. Nothing survives a restart and
nothing is shared between replicas. A thirty-day, hundred-session baseline built
that way would be reset by every deploy and would see only the fraction of
traffic that happened to land on one process, which is worse than no baseline:
it would report sources as "never seen" that the agent reads constantly on
another replica, and every one of those is a false HIGH.

So the baseline is built OFFLINE, by ``baseline_builder`` running over shipped
telemetry, and loaded here as read-only state. The SDK observes and reports; it
does not accumulate.

────────────────────────────────────────────────────────────────────────────────
THE THRESHOLD IS SESSIONS, AND IT IS NOT THE ONE ALREADY IN THE SDK

``drift_detector._FULL_THRESHOLD`` is 100 *calls*. This one is 100 *sessions*,
and they are different by one to two orders of magnitude for a conversational
agent. Using the call count would activate the soft boundary after roughly a day
of traffic rather than a month, on a sample too thin to distinguish "this agent
does not read that bucket" from "this agent has not read that bucket yet".

Counting sessions is also what makes the threshold work at both ends of the
range the spec cares about: an agent serving five sessions a day earns it in
twenty days, one serving five hundred earns it in hours. A calendar duration
would be wrong for both.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Sessions required before any soft-boundary signal may fire. Below this,
# ``baseline_ready`` is False and ScopeBoundaryChecker suppresses all three.
#
# Configurable per build, but 100 is the default for a reason worth stating: the
# soft boundary's whole job is to distinguish "this agent does not read that
# source" from "this agent has not read that source yet", and that distinction
# is a sample-size question. Lower it and every source an agent touches rarely —
# a quarter-end report, a month-end reconciliation — reads as novel the first
# time it appears after the baseline closes. Those are HIGH signals against
# entirely normal behaviour, and a handful of them is all it takes for the
# family to be filtered out.
#
# The threshold is a session count rather than a calendar duration so it works
# at both ends of the range: an agent serving five sessions a day earns it in
# twenty days, one serving five hundred earns it in hours.
MIN_SESSIONS_FOR_BASELINE = 100

DEFAULT_BASELINE_PERIOD_DAYS = 30


@dataclass
class SourceProfile:
    """What is known about one source this agent has read."""
    source_uri: str = ""
    access_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    classifications_seen: List[str] = field(default_factory=list)
    avg_content_length: int = 0
    # Mean accesses per session, not per day: it stays meaningful when traffic
    # volume changes, which a per-day rate does not.
    access_frequency: float = 0.0
    sessions_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentBehavioralBaseline:
    """Observed scope for one agent, built from shipped telemetry."""

    agent_id: str = ""
    baseline_period_days: int = DEFAULT_BASELINE_PERIOD_DAYS
    observation_start: str = ""
    last_updated: str = ""

    observed_sources: Dict[str, SourceProfile] = field(default_factory=dict)
    observed_tools: Dict[str, int] = field(default_factory=dict)
    typical_turn_count: Tuple[float, float] = (0.0, 0.0)
    typical_session_secs: Tuple[float, float] = (0.0, 0.0)
    common_tool_sequences: List[List[str]] = field(default_factory=list)

    sessions_observed: int = 0
    baseline_ready: bool = False
    # The threshold this baseline was built against, carried on the record.
    #
    # Without it the SDK would re-apply its own compiled-in default and could
    # silently disagree with the builder — a baseline built at 60 would be
    # treated as not-ready by an SDK defaulting to 100, and the soft boundary
    # would go quiet for reasons nobody could see. The number that decided
    # readiness has to travel with the decision.
    min_sessions: int = MIN_SESSIONS_FOR_BASELINE

    # ------------------------------------------------------------------

    def has_seen_source(self, uri: str) -> bool:
        """Whether this agent has read *uri* during the baseline window.

        Containment-aware for the same reason the manifest check is: the
        baseline records what the interceptor reported, which is object-level
        (``bucket/reports/q3.pdf``), while a later read is a *different* object
        in the same bucket. Exact matching would make every new object key in a
        thoroughly-familiar bucket a HIGH "source never seen before" — turning
        normal traffic into a stream of alerts within a day of activation.

        The comparison runs both ways: a previously-seen object implies its
        container is familiar, and a previously-seen container covers a new
        object inside it.
        """
        if not uri:
            return False
        if uri in self.observed_sources:
            return True
        from ai_telemetry.manifest import _covers
        container = source_container(uri)
        for seen in self.observed_sources:
            if _covers(seen, uri) or _covers(uri, seen):
                return True
            # Siblings: two different objects in the same bucket, two chunks of
            # the same index. Neither covers the other, but the agent is plainly
            # familiar with the place they came from.
            if container and source_container(seen) == container:
                return True
        return False

    def has_seen_tool(self, tool_name: str) -> bool:
        return bool(tool_name) and tool_name in self.observed_tools

    def has_seen_sequence(self, sequence: List[str]) -> bool:
        """Whether this tool sequence appears in the baseline as a contiguous run."""
        if not sequence:
            return True
        for known in self.common_tool_sequences:
            if _contains_run(known, sequence):
                return True
        return False

    def source_profile(self, uri: str) -> Optional[SourceProfile]:
        return self.observed_sources.get(uri)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["observed_sources"] = {k: v.to_dict() if hasattr(v, "to_dict") else v
                                 for k, v in self.observed_sources.items()}
        d["typical_turn_count"] = list(self.typical_turn_count)
        d["typical_session_secs"] = list(self.typical_session_secs)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentBehavioralBaseline":
        b = cls()
        for k, v in (d or {}).items():
            if k == "observed_sources":
                b.observed_sources = {
                    uri: (prof if isinstance(prof, SourceProfile)
                          else SourceProfile(**prof))
                    for uri, prof in (v or {}).items()
                }
            elif k in ("typical_turn_count", "typical_session_secs"):
                setattr(b, k, tuple(v or (0.0, 0.0)))
            elif hasattr(b, k):
                setattr(b, k, v)
        # Never trust a stored ready flag over the count that justifies it. A
        # baseline hand-edited or truncated in transit could otherwise activate
        # HIGH signals on a handful of sessions. The threshold is taken from the
        # record so the builder's choice is honoured, but the comparison is
        # always recomputed here.
        b.baseline_ready = b.sessions_observed >= max(1, int(b.min_sessions or
                                                            MIN_SESSIONS_FOR_BASELINE))
        return b


def source_container(uri: str) -> str:
    """The thing a source lives *in* — its bucket, table, index or host.

    Needed because the baseline records what the interceptor observed, which is
    leaf-level: ``bucket/reports/q3.pdf``. The next read is ``q4.pdf``, a
    different leaf in the same container. Comparing leaves would call that a
    source never seen before, and an agent working normally through a familiar
    bucket would generate a HIGH per new object key.

    Resource-type aware rather than "strip the last path segment", because the
    generic rule is wrong in both directions: it would reduce every DynamoDB ARN
    to ``…:table`` and make *any* table look familiar once one had been read,
    while leaving a deeply-nested S3 prefix as its own container so siblings
    still missed.
    """
    if not uri:
        return ""
    low = uri.lower()

    if low.startswith("arn:aws:s3:::"):
        rest = uri[len("arn:aws:s3:::"):]
        return "arn:aws:s3:::" + rest.split("/", 1)[0]

    if low.startswith("arn:aws:dynamodb:"):
        # …:table/Name[/index/Idx] — the table is the container, not ":table".
        marker = ":table/"
        i = uri.find(marker)
        if i != -1:
            tail = uri[i + len(marker):]
            return uri[: i + len(marker)] + tail.split("/", 1)[0]
        return uri

    if "://" in uri:
        scheme, rest = uri.split("://", 1)
        return f"{scheme}://{rest.split('/', 1)[0]}"

    # Secrets, parameters, knowledge bases and anything unrecognised are their
    # own container: there is no sibling relationship worth inferring, and
    # guessing one would make an unfamiliar secret look familiar.
    return uri


def _contains_run(haystack: List[str], needle: List[str]) -> bool:
    n, h = len(needle), len(haystack)
    if n > h:
        return False
    return any(haystack[i:i + n] == needle for i in range(h - n + 1))


class BaselineStore:
    """Read-only baselines, keyed by agent id.

    Handed to the client at init and consulted by the scope checker. Holding it
    read-only is the point: an SDK that updated the baseline from live traffic
    would teach itself to accept whatever it was currently seeing, which is
    precisely the weakness the manifest exists to cover for.
    """

    def __init__(self, baselines: Optional[Dict[str, AgentBehavioralBaseline]] = None):
        self._baselines = dict(baselines or {})
        self._lock = threading.Lock()

    def get(self, agent_id: str) -> Optional[AgentBehavioralBaseline]:
        with self._lock:
            return self._baselines.get(agent_id)

    def put(self, baseline: AgentBehavioralBaseline) -> None:
        with self._lock:
            self._baselines[baseline.agent_id] = baseline

    def __len__(self) -> int:
        return len(self._baselines)

    @property
    def ready_agents(self) -> List[str]:
        return sorted(a for a, b in self._baselines.items() if b.baseline_ready)

    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str) -> "BaselineStore":
        """Load baselines written by ``baseline_builder``.

        Never raises on a missing or malformed file: a baseline that cannot be
        loaded leaves the store empty, which suppresses every soft-boundary
        signal. Failing closed here would mean an unreadable file turns into a
        stream of HIGH alerts claiming the agent has never seen anything.
        """
        store = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        store.put(AgentBehavioralBaseline.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        except OSError:
            return store
        return store

    def to_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for b in self._baselines.values():
                fh.write(json.dumps(b.to_dict()) + "\n")
        return len(self._baselines)


__all__ = [
    "SourceProfile", "AgentBehavioralBaseline", "BaselineStore",
    "MIN_SESSIONS_FOR_BASELINE", "DEFAULT_BASELINE_PERIOD_DAYS",
]
