"""Build behavioural baselines from shipped telemetry (spec §7).

Runs as a job, not in the request path — the ZASP shape: periodic, external,
producing a versioned record something else consumes. It reads what the SDK
already ships and writes one ``AgentBehavioralBaseline`` per agent.

    python -m ai_telemetry.baseline_builder /var/log/zendata/ -o baselines.jsonl

────────────────────────────────────────────────────────────────────────────────
SESSIONS HAVE TO BE REBUILT, BECAUSE THE SDK DOES NOT SHIP THEM

The spec's ``BaselineBuilder.build()`` takes ``List[Session]`` and walks
``session.turns[].source_provenance_tags``. No such object exists anywhere in
this SDK. What ships is a flat stream of events, one per LLM or tool call, so
the first job here is to put the shape back together: group by ``session_id``,
order by timestamp, and split into turns on each LLM call — a turn being one
model call plus the tool calls that follow it before the next.

``session_id`` is the load-bearing field and it was not on the event until this
work added it. Grouping by ``trace_id`` instead is the tempting substitute and
it is wrong by one to two orders of magnitude: a trace is a single interaction,
so a hundred-turn conversation would present as a hundred sessions and cross
``MIN_SESSIONS_FOR_BASELINE`` after about a day. The baseline would then activate
on data far too thin to tell "this agent does not read that bucket" from "this
agent has not read that bucket yet", and every soft-boundary signal would fire
against it.

────────────────────────────────────────────────────────────────────────────────
WHAT COUNTS AS AN AGENT

Baselines are keyed on ``agent_id`` when the fingerprinting module resolved one,
and ``app_id`` otherwise. Both are stable across sessions, which is the only
property that matters here — a baseline keyed on anything per-session would
never accumulate.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai_telemetry.baseline import (
    AgentBehavioralBaseline, BaselineStore, SourceProfile,
    MIN_SESSIONS_FOR_BASELINE, DEFAULT_BASELINE_PERIOD_DAYS,
)

logger = logging.getLogger(__name__)

# Longest tool sequence worth remembering, and how many to keep. A sequence
# longer than this is almost always a whole session rather than a pattern, and
# storing whole sessions would make SC_TOOL_SEQUENCE_ANOMALY fire on any run
# that merely differs in length.
MAX_SEQUENCE_LEN = 6
TOP_SEQUENCES = 10


@dataclass
class Turn:
    """One model call plus the tool calls that followed it."""
    started_ms: int = 0
    ended_ms: int = 0
    tool_calls: List[str] = field(default_factory=list)
    source_tags: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Session:
    session_id: str = ""
    agent_id: str = ""
    turns: List[Turn] = field(default_factory=list)
    started_ms: int = 0
    ended_ms: int = 0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_ms - self.started_ms) / 1000.0)


# ---------------------------------------------------------------------------
# Reading telemetry
# ---------------------------------------------------------------------------

def read_events(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Every event under *path*, from JSONL or CSV.

    JSONL is preferred and CSV is a fallback: the provenance chain is a list of
    objects, and the CSV sink cannot carry it, so a CSV-only deployment sees
    only the first source of a multi-source tool call.
    """
    p = Path(path)
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.suffix in (".jsonl", ".csv"):
                yield from read_events(child)
        return
    try:
        if p.suffix == ".csv":
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    yield dict(row)
        else:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
    except OSError as e:
        logger.warning("cannot read %s: %s", p, e)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _tags(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    chain = event.get("source_provenance")
    if isinstance(chain, str):
        try:
            chain = json.loads(chain)
        except json.JSONDecodeError:
            chain = None
    if isinstance(chain, list):
        return [t for t in chain if isinstance(t, dict) and t.get("source_uri")]
    uri = event.get("source_uri")
    if uri:
        return [{
            "source_uri": uri,
            "classification": event.get("source_classification") or "",
            "content_length": _int(event.get("content_length")),
        }]

    # The pre-existing pipe-delimited field. Telemetry written before source
    # provenance existed carries only this, and the observed-source index in the
    # scanner already reads it — the two disagreeing would mean a baseline that
    # sees no sources for data the over-privilege report happily analyses.
    declared = event.get("data_sources")
    if isinstance(declared, str) and declared:
        return [{"source_uri": part.strip(),
                 "classification": event.get("zendata_classification") or "",
                 "content_length": 0}
                for part in declared.split("|") if part.strip()]
    return []


def _agent_key(event: Dict[str, Any], agent_key: Optional[str] = None) -> str:
    """Which agent this event belongs to.

    ``agent_id`` when fingerprinting resolved one, then ``app_id``. ``agent_key``
    overrides both, for telemetry where the principal is carried under another
    name — an identity-centric export keyed on ``user_id``, for instance, where
    agent_id is populated for only the handful of callers the fingerprinter
    managed to cluster and every other event would otherwise be dropped.
    """
    if agent_key:
        value = event.get(agent_key)
        if value:
            return str(value)
    return event.get("agent_id") or event.get("app_id") or ""


# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------

def reconstruct_sessions(
    events: Iterable[Dict[str, Any]],
    session_key: Optional[str] = None,
    agent_key: Optional[str] = None,
) -> Dict[str, List[Session]]:
    """Group a flat event stream back into per-agent sessions.

    ``session_key`` names a field to use when ``session_id`` is absent, for
    telemetry written before it existed. ``user_id`` is the honest choice there
    and not an arbitrary one: it is exactly what ``capture()`` falls back to
    when the host application sets no session, so it reproduces the grouping the
    SDK itself would have recorded.

    ``trace_id`` is deliberately NOT accepted, however tempting it looks. A
    trace is one interaction, so every turn becomes its own session and the
    hundred-session threshold falls after a day of traffic — activating the soft
    boundary on data far too thin to trust. Passing it is a silent way to make
    the threshold meaningless, so it is refused outright.
    """
    if session_key == "trace_id":
        raise ValueError(
            "trace_id cannot stand in for session_id: a trace is one "
            "interaction, so every turn would count as a session and "
            "MIN_SESSIONS_FOR_BASELINE would be met roughly 100x early. "
            "Use user_id, or set session_id on the events.")

    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    skipped_no_session = 0
    fallback_used = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        agent = _agent_key(ev, agent_key)
        session_id = ev.get("session_id")
        if not session_id and session_key:
            session_id = ev.get(session_key)
            if session_id:
                fallback_used += 1
        if not agent or not session_id:
            # No session identifier and no usable fallback. Counting it under a
            # synthetic id would inflate the session count and bring the
            # baseline ready early on exactly the data that is least
            # trustworthy, so it is dropped and reported instead.
            skipped_no_session += 1
            continue
        grouped[(agent, session_id)].append(ev)

    if fallback_used:
        logger.info(
            "%d event(s) had no session_id; grouped by %s instead",
            fallback_used, session_key)
    if skipped_no_session:
        logger.warning(
            "%d event(s) carried no session identifier and were skipped — "
            "telemetry written before session_id was added to the event schema "
            "needs --session-key to contribute to a session count",
            skipped_no_session)

    by_agent: Dict[str, List[Session]] = defaultdict(list)
    for (agent, session_id), evs in grouped.items():
        evs.sort(key=lambda e: _int(e.get("timestamp_ms")))
        by_agent[agent].append(_build_session(agent, session_id, evs))
    return dict(by_agent)


def _build_session(agent: str, session_id: str,
                   events: List[Dict[str, Any]]) -> Session:
    session = Session(session_id=session_id, agent_id=agent)
    session.started_ms = _int(events[0].get("timestamp_ms"))
    session.ended_ms = _int(events[-1].get("timestamp_ms"))

    current: Optional[Turn] = None
    for ev in events:
        ts = _int(ev.get("timestamp_ms"))
        kind = ev.get("event_type")

        if kind in ("llm_call", "user_message") or current is None:
            current = Turn(started_ms=ts, ended_ms=ts)
            session.turns.append(current)

        current.ended_ms = max(current.ended_ms, ts)
        if kind == "tool_call" and ev.get("tool_name"):
            current.tool_calls.append(ev["tool_name"])
        current.source_tags.extend(_tags(ev))

    return session


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build(agent_id: str, sessions: List[Session],
          period_days: int = DEFAULT_BASELINE_PERIOD_DAYS,
          min_sessions: int = MIN_SESSIONS_FOR_BASELINE) -> AgentBehavioralBaseline:
    """One agent's baseline from its sessions (spec §7).

    ``min_sessions`` lowers the readiness threshold. Doing so trades precision
    for earlier activation and the trade is real — see MIN_SESSIONS_FOR_BASELINE
    — so the chosen value is written onto the record rather than left implicit,
    and travels with it to whatever evaluates against it.
    """
    sources: Dict[str, SourceProfile] = {}
    lengths: Dict[str, List[int]] = defaultdict(list)
    tools: Counter = Counter()
    sequences: Counter = Counter()
    turn_counts: List[int] = []
    durations: List[float] = []

    for session in sessions:
        turn_counts.append(len(session.turns))
        durations.append(session.duration_seconds)
        seen_this_session: set = set()

        for turn in session.turns:
            for name in turn.tool_calls:
                tools[name] += 1
            for run in _subsequences(turn.tool_calls):
                sequences[run] += 1

            for tag in turn.source_tags:
                uri = tag.get("source_uri")
                if not uri:
                    continue
                profile = sources.get(uri)
                if profile is None:
                    profile = SourceProfile(
                        source_uri=uri,
                        first_seen=tag.get("retrieved_at") or "",
                    )
                    sources[uri] = profile
                profile.access_count += 1
                profile.last_seen = tag.get("retrieved_at") or profile.last_seen
                cls = tag.get("classification")
                if cls and cls not in profile.classifications_seen:
                    profile.classifications_seen.append(cls)
                length = _int(tag.get("content_length"))
                if length:
                    lengths[uri].append(length)
                if uri not in seen_this_session:
                    profile.sessions_seen += 1
                    seen_this_session.add(uri)

    session_count = len(sessions)
    for uri, profile in sources.items():
        if lengths[uri]:
            profile.avg_content_length = int(statistics.fmean(lengths[uri]))
        profile.access_frequency = round(
            profile.access_count / session_count, 4) if session_count else 0.0
        profile.classifications_seen.sort()

    baseline = AgentBehavioralBaseline(
        agent_id=agent_id,
        baseline_period_days=period_days,
        observed_sources=sources,
        observed_tools=dict(tools),
        typical_turn_count=_mean_std(turn_counts),
        typical_session_secs=_mean_std(durations),
        common_tool_sequences=[list(seq) for seq, _ in sequences.most_common(TOP_SEQUENCES)],
        sessions_observed=session_count,
        # Never set directly — MIN_SESSIONS_FOR_BASELINE is the whole gate, and
        # a builder that could override it would make the threshold advisory.
        min_sessions=min_sessions,
        baseline_ready=session_count >= max(1, min_sessions),
    )
    if sessions:
        baseline.observation_start = _iso(min(s.started_ms for s in sessions))
        baseline.last_updated = _iso(max(s.ended_ms for s in sessions))
    return baseline


def _subsequences(tool_calls: List[str]) -> Iterable[tuple]:
    """Contiguous runs of 2..MAX_SEQUENCE_LEN tool calls."""
    n = len(tool_calls)
    for size in range(2, min(MAX_SEQUENCE_LEN, n) + 1):
        for i in range(n - size + 1):
            yield tuple(tool_calls[i:i + size])


def _mean_std(values: List[float]) -> tuple:
    if not values:
        return (0.0, 0.0)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (round(mean, 3), round(std, 3))


def _iso(timestamp_ms: int) -> str:
    from datetime import datetime, timezone
    if not timestamp_ms:
        return ""
    return datetime.fromtimestamp(
        timestamp_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_all(path: str | Path,
              period_days: int = DEFAULT_BASELINE_PERIOD_DAYS,
              session_key: Optional[str] = None,
              agent_key: Optional[str] = None,
              min_sessions: int = MIN_SESSIONS_FOR_BASELINE) -> BaselineStore:
    """Read telemetry under *path* and build one baseline per agent."""
    by_agent = reconstruct_sessions(read_events(path), session_key=session_key,
                                    agent_key=agent_key)
    store = BaselineStore()
    for agent_id, sessions in by_agent.items():
        baseline = build(agent_id, sessions, period_days=period_days,
                         min_sessions=min_sessions)
        store.put(baseline)
        logger.info(
            "%s: %d session(s), %d source(s), ready=%s",
            agent_id, baseline.sessions_observed,
            len(baseline.observed_sources), baseline.baseline_ready)
    return store


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build agent behavioural baselines from SDK telemetry.")
    parser.add_argument("telemetry", help="JSONL/CSV file or directory")
    parser.add_argument("-o", "--out", default="baselines.jsonl")
    parser.add_argument("--period-days", type=int, default=DEFAULT_BASELINE_PERIOD_DAYS)
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS_FOR_BASELINE,
                        help="Sessions before the soft boundary activates "
                             f"(default {MIN_SESSIONS_FOR_BASELINE}). Lowering it "
                             "trades precision for earlier activation.")
    parser.add_argument("--agent-key", default=None,
                        help="Field naming the principal when events carry no "
                             "agent_id or app_id (e.g. user_id).")
    parser.add_argument("--session-key", default=None,
                        help="Field to group by when events carry no session_id "
                             "(e.g. user_id). trace_id is refused — a trace is "
                             "one interaction, so every turn would count as a "
                             "session.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s")

    store = build_all(args.telemetry, period_days=args.period_days,
                      session_key=args.session_key, agent_key=args.agent_key,
                      min_sessions=args.min_sessions)
    written = store.to_jsonl(args.out)
    ready = store.ready_agents
    print(f"{written} baseline(s) -> {args.out}")
    print(f"{len(ready)} of {written} ready "
          f"(>= {args.min_sessions} sessions); "
          f"soft-boundary signals stay suppressed for the rest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
