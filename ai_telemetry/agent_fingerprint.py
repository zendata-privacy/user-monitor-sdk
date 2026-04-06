"""
Agent Fingerprinting — Module 3

Builds a persistent behavioral fingerprint for each non-human caller using
six observable API-layer components.  The fingerprint is stable across
sessions and requires no IAM credentials; when IAM is later connected,
existing records are enriched and behavioral history carries forward.

Components
----------
  1. API key cluster ID       — SHA-256 of API key; groups all calls from
                                the same service account into one agent identity.
  2. System prompt version    — Hash of system prompt; version changes are
                                recorded as agent lineage entries.
  3. Tool call fingerprint    — Union of distinct tools ever invoked; the
                                combination is highly identifying per agent type.
  4. Temporal rhythm          — 24-bucket hour-of-day call distribution;
                                distinguishes nightly batch from real-time agents.
  5. Orchestration position   — ORCHESTRATOR / SUBAGENT / STANDALONE, inferred
                                from prompt structure analysis (M1 rule).
  6. Prompt template hash     — Structural skeleton of the user prompt; same
                                skeleton across sessions → same agent type even
                                across different service accounts.

Agent record fields (per spec §Module 3)
-----------------------------------------
  agent_id                pseudonymous stable ID
  display_name            inferred from system prompt ("You are X")
  agent_type              NIGHTLY_BATCH | REALTIME | HIGH_FREQUENCY_PIPELINE |
                          BUSINESS_HOURS_AGENT | UNKNOWN
  orchestration_role      ORCHESTRATOR | SUBAGENT | STANDALONE
  tool_scope              list of distinct tool names ever invoked
  first_seen / last_seen  epoch ms
  system_prompt_versions  list of {hash, first_seen_ms, last_seen_ms, call_count}
  iam_identity            null until IAM is connected
  inferred_role           coarse role label (FINANCE_AGENT, CODE_AGENT, etc.)
  role_confidence         0.0 – 1.0
  drift_score             0.0 – 1.0, cumulative deviation from baseline behaviour
  drift_alerts            list of named drift events
"""

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryEvent

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Component 2 — extract agent display name from system prompt
_RE_AGENT_NAME = re.compile(
    r"you are (?:a |an |the )?([A-Za-z][A-Za-z0-9 \-]{2,50}?)(?:[.,\n]|$)",
    re.IGNORECASE,
)

# Component 3 — financial / sensitive tool name keywords (used for role inference)
_FINANCE_TOOLS = frozenset({
    "get_stock_price", "get_portfolio", "execute_trade", "place_order",
    "check_balance", "transfer_funds", "get_market_data", "calculate_risk",
    "fetch_financial_data", "predict_price",
})
_CODE_TOOLS = frozenset({
    "code_interpreter", "bash", "python_repl", "execute_code", "run_script",
})
_SEARCH_TOOLS = frozenset({
    "web_search", "search", "tavily_search", "bing_search", "google_search",
    "retrieve_documents", "vector_search",
})

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SystemPromptVersion:
    """One observed version of an agent's system prompt."""
    prompt_hash: str
    first_seen_ms: int
    last_seen_ms: int
    call_count: int = 1


@dataclass
class AgentRecord:
    """Persistent behavioral fingerprint for one agent (service account cluster)."""
    agent_id: str                                           # "agt_" + api_key_hash[:20]
    api_key_cluster: str                                    # api_key_hash[:16] — display label

    display_name: Optional[str] = None                     # inferred from system prompt
    agent_type: str = "UNKNOWN"                            # temporal rhythm classification
    orchestration_role: str = "STANDALONE"                 # ORCHESTRATOR | SUBAGENT | STANDALONE

    # Component 3 — tool fingerprint
    tool_scope: List[str] = field(default_factory=list)

    # Timeline
    first_seen_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    last_seen_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    call_count: int = 0

    # Component 2 — system prompt versioning / lineage
    system_prompt_versions: List[SystemPromptVersion] = field(default_factory=list)

    # Component 6 — prompt template fingerprint (distinct structural hashes)
    prompt_template_hashes: List[str] = field(default_factory=list)

    # Component 4 — temporal rhythm: 24 buckets (index = UTC hour, value = call count)
    temporal_rhythm: List[int] = field(default_factory=lambda: [0] * 24)

    # IAM enrichment (null until connected)
    iam_identity: Optional[str] = None

    # Module 3 — coarse role from tool scope
    inferred_role: Optional[str] = None
    role_confidence: float = 0.0

    # Module 4 — agent role taxonomy (system prompt classification + tool validation)
    agent_role: Optional[str] = None           # CS_AGENT | FINANCE_AGENT | ORCHESTRATOR | …
    risk_level: Optional[str] = None           # CRITICAL | HIGH | MEDIUM | LOW-MED | LOW
    primary_risk: Optional[str] = None         # human-readable risk description
    role_scope_gap: bool = False               # True when tool scope diverges from stated role
    # Embedding of the most recent system prompt — used for benchmark calibration
    system_prompt_embedding: Optional[List[float]] = None

    # Drift
    drift_score: float = 0.0
    drift_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "api_key_cluster": self.api_key_cluster,
            "display_name": self.display_name,
            "agent_type": self.agent_type,
            "orchestration_role": self.orchestration_role,
            "tool_scope": list(self.tool_scope),
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
            "call_count": self.call_count,
            "system_prompt_version_count": len(self.system_prompt_versions),
            "system_prompt_versions": [
                {
                    "prompt_hash": v.prompt_hash,
                    "first_seen_ms": v.first_seen_ms,
                    "last_seen_ms": v.last_seen_ms,
                    "call_count": v.call_count,
                }
                for v in self.system_prompt_versions
            ],
            "prompt_template_hashes": list(self.prompt_template_hashes),
            "temporal_rhythm": list(self.temporal_rhythm),
            "iam_identity": self.iam_identity,
            "inferred_role": self.inferred_role,
            "role_confidence": self.role_confidence,
            # Module 4 additions
            "agent_role": self.agent_role,
            "risk_level": self.risk_level,
            "primary_risk": self.primary_risk,
            "role_scope_gap": self.role_scope_gap,
            "drift_score": round(self.drift_score, 4),
            "drift_alerts": list(self.drift_alerts),
        }


# ---------------------------------------------------------------------------
# Fingerprint store
# ---------------------------------------------------------------------------


class AgentFingerprintStore:
    """Thread-safe in-memory store for agent fingerprint records.

    One record per service account (api_key cluster).  Records survive for the
    lifetime of the process; a future persistence layer (Redis / Postgres) can
    be swapped in by replacing get / update.
    """

    def __init__(self) -> None:
        self._records: Dict[str, AgentRecord] = {}
        self._lock = threading.Lock()

    def get_or_create(self, agent_id: str, api_key_cluster: str) -> AgentRecord:
        with self._lock:
            if agent_id not in self._records:
                self._records[agent_id] = AgentRecord(
                    agent_id=agent_id,
                    api_key_cluster=api_key_cluster,
                )
            return self._records[agent_id]

    def update(self, record: AgentRecord) -> None:
        with self._lock:
            self._records[record.agent_id] = record

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        with self._lock:
            return self._records.get(agent_id)

    def all_records(self) -> List[AgentRecord]:
        with self._lock:
            return list(self._records.values())


# Module-level singleton
_global_store = AgentFingerprintStore()


def get_fingerprint_store() -> AgentFingerprintStore:
    return _global_store


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _derive_agent_id(api_key_hash: str) -> str:
    """Stable pseudonymous ID: 'agt_' + first 20 hex chars of api_key_hash."""
    return "agt_" + api_key_hash[:20]


def _extract_system_prompt(event: "TelemetryEvent") -> Optional[str]:
    """Pull system prompt text from event.input_messages (if prompts were captured)."""
    messages = event.input_messages or []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    return None


def _extract_user_prompt(event: "TelemetryEvent") -> Optional[str]:
    """Pull the last user-role message text from event.input_messages."""
    messages = event.input_messages or []
    user_text = None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            user_text = content if isinstance(content, str) else str(content)
    return user_text


def _extract_tool_names(event: "TelemetryEvent") -> List[str]:
    """Extract distinct tool names from event.output_message.tool_calls."""
    names: List[str] = []
    if not event.output_message:
        return names
    tool_calls = event.output_message.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        # OpenAI format: {"function": {"name": ...}}
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name and name not in names:
            names.append(name)
    return names


def _compute_prompt_template_hash(user_prompt: str) -> str:
    """Structural skeleton hash of the first 200 chars — identifies template patterns."""
    from ai_telemetry.client import _sha256
    # Reuse the skeleton normaliser from entity_classifier without importing the module
    # (avoids circular import) — inline the same logic.
    s = re.sub(r'"[^"]*"', '"__V__"', user_prompt)
    s = re.sub(r"'[^']*'", "'__V__'", s)
    s = re.sub(r"\b\d[\d,.\s]*\b", "__N__", s)
    s = re.sub(r"\b[A-Z]{2,6}\b", "__S__", s)
    skeleton = s.strip()[:200]
    return _sha256(skeleton)[:16]


def _infer_display_name(system_prompt: str) -> Optional[str]:
    """Extract agent name/purpose from 'You are [X]' pattern in system prompt."""
    m = _RE_AGENT_NAME.search(system_prompt)
    if m:
        return m.group(1).strip()[:60]
    return None


def _update_system_prompt_version(record: AgentRecord, prompt_hash: str, now_ms: int) -> None:
    """Track system prompt lineage.  Version changes increment drift_score."""
    for v in record.system_prompt_versions:
        if v.prompt_hash == prompt_hash:
            v.last_seen_ms = now_ms
            v.call_count += 1
            return
    # New version observed
    record.system_prompt_versions.append(
        SystemPromptVersion(
            prompt_hash=prompt_hash,
            first_seen_ms=now_ms,
            last_seen_ms=now_ms,
        )
    )
    if len(record.system_prompt_versions) > 1:
        # System prompt changed — lineage event; penalise drift
        record.drift_score = min(1.0, record.drift_score + 0.20)
        if "SYSTEM_PROMPT_VERSION_CHANGE" not in record.drift_alerts:
            record.drift_alerts.append("SYSTEM_PROMPT_VERSION_CHANGE")


def _update_tool_scope(record: AgentRecord, tool_names: List[str]) -> None:
    """Expand tool_scope with newly observed tools.

    Tool scope expansion after an agent is well-established (10+ calls) is a
    drift signal — the agent may have been re-prompted with a wider capability set.
    """
    for name in tool_names:
        if name not in record.tool_scope:
            record.tool_scope.append(name)
            if record.call_count > 10:
                record.drift_score = min(1.0, record.drift_score + 0.10)
                if "TOOL_SCOPE_EXPANSION" not in record.drift_alerts:
                    record.drift_alerts.append("TOOL_SCOPE_EXPANSION")


def _update_temporal_rhythm(record: AgentRecord, timestamp_ms: int) -> None:
    """Increment the appropriate UTC hour bucket."""
    hour = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).hour
    record.temporal_rhythm[hour] += 1


def _infer_agent_type(record: AgentRecord) -> str:
    """Classify agent by call-time distribution (Component 4).

    Requires at least 5 calls for a stable signal.
    """
    if record.call_count < 5:
        return "UNKNOWN"
    total = sum(record.temporal_rhythm)
    if total == 0:
        return "UNKNOWN"

    # Off-hours: 22:00–06:00 UTC
    off_hours = sum(record.temporal_rhythm[h] for h in list(range(22, 24)) + list(range(0, 6)))
    # Business hours: 08:00–19:00 UTC Mon–Fri (we track by hour only, not weekday)
    biz_hours = sum(record.temporal_rhythm[h] for h in range(8, 19))

    off_ratio = off_hours / total
    biz_ratio = biz_hours / total

    if off_ratio >= 0.65:
        return "NIGHTLY_BATCH"
    if record.call_count >= 1000:
        return "HIGH_FREQUENCY_PIPELINE"
    if biz_ratio >= 0.70:
        return "BUSINESS_HOURS_AGENT"
    return "REALTIME"


def _infer_orchestration_role(event: "TelemetryEvent", record: AgentRecord) -> str:
    """Derive orchestration position from M1 entity_subtype (Component 5)."""
    subtype = event.entity_subtype or ""
    if subtype == "SUBAGENT_RECEIVER":
        return "SUBAGENT"
    if subtype == "ORCHESTRATOR_SENDER":
        return "ORCHESTRATOR"
    # Preserve prior role if already established
    if record.orchestration_role != "STANDALONE":
        return record.orchestration_role
    return "STANDALONE"


def _infer_role_from_tools(tool_scope: List[str]) -> tuple:
    """Return (inferred_role, confidence) from the agent's tool scope."""
    tool_set = set(tool_scope)
    if not tool_set:
        return None, 0.0

    finance_hits = len(tool_set & _FINANCE_TOOLS)
    code_hits = len(tool_set & _CODE_TOOLS)
    search_hits = len(tool_set & _SEARCH_TOOLS)

    max_hits = max(finance_hits, code_hits, search_hits)
    if max_hits == 0:
        return "GENERIC_AGENT", 0.4

    total_known = len(_FINANCE_TOOLS | _CODE_TOOLS | _SEARCH_TOOLS)
    confidence = min(0.95, 0.4 + 0.55 * (max_hits / max(len(tool_set), 1)))

    if finance_hits >= code_hits and finance_hits >= search_hits:
        return "FINANCE_AGENT", confidence
    if code_hits >= search_hits:
        return "CODE_AGENT", confidence
    return "RESEARCH_AGENT", confidence


def _compute_role_confidence(record: AgentRecord) -> float:
    """Confidence rises with call count (stabilises after 20 calls)."""
    base = min(1.0, record.call_count / 20.0)
    # Penalise drift
    return max(0.0, base - record.drift_score * 0.3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fingerprint_agent(event: "TelemetryEvent") -> Optional[str]:
    """Build or update the behavioral fingerprint for the agent that produced *event*.

    Called from TelemetryClient._send() for every llm_call event whose
    entity_type is AUTOMATED, AGENT, or AMBIGUOUS.

    Returns the agent_id string (also written onto event.agent_id).
    Returns None if there is insufficient information to fingerprint.
    """
    api_key_hash = event.api_key_hash
    if not api_key_hash:
        return None

    store = get_fingerprint_store()
    agent_id = _derive_agent_id(api_key_hash)
    api_key_cluster = api_key_hash[:16]
    record = store.get_or_create(agent_id, api_key_cluster)

    now_ms = event.timestamp_ms

    # ---- Component 1: call count + timeline --------------------------------
    record.call_count += 1
    record.last_seen_ms = now_ms

    # ---- Component 2: system prompt version / lineage ----------------------
    prompt_hash = event.system_prompt_hash
    if prompt_hash:
        _update_system_prompt_version(record, prompt_hash, now_ms)
        # Try to infer display name from raw text (only possible when prompts captured)
        if record.display_name is None:
            sys_text = _extract_system_prompt(event)
            if sys_text:
                record.display_name = _infer_display_name(sys_text)

    # ---- Component 3: tool call fingerprint --------------------------------
    tool_names = _extract_tool_names(event)
    _update_tool_scope(record, tool_names)

    # ---- Component 4: temporal rhythm --------------------------------------
    _update_temporal_rhythm(record, now_ms)
    record.agent_type = _infer_agent_type(record)

    # ---- Component 5: orchestration position -------------------------------
    record.orchestration_role = _infer_orchestration_role(event, record)

    # ---- Component 6: prompt template hash ---------------------------------
    user_text = _extract_user_prompt(event)
    if user_text:
        tmpl_hash = _compute_prompt_template_hash(user_text)
        if tmpl_hash not in record.prompt_template_hashes:
            record.prompt_template_hashes.append(tmpl_hash)
            # Many distinct templates → less template-driven (more conversational)

    # ---- Component 3 (Module 3) coarse role from tools ---------------------
    inferred_role, _ = _infer_role_from_tools(record.tool_scope)
    if inferred_role:
        record.inferred_role = inferred_role

    # ---- Module 4: agent role taxonomy from system prompt + tool scope ------
    from ai_telemetry.role_inference import (
        infer_agent_role, detect_scope_gap, benchmark_calibrate,
    )
    sys_text = _extract_system_prompt(event)
    if sys_text:
        agent_role, role_conf, risk_level, primary_risk = infer_agent_role(
            sys_text, record.tool_scope
        )
        # Benchmark calibration — inherit role from similar known agents when
        # the record is new and a system prompt embedding is available
        embedding = event.system_prompt_embedding
        if embedding:
            record.system_prompt_embedding = embedding  # cache latest embedding
            cal_role, cal_conf = benchmark_calibrate(
                embedding, record.call_count, store.all_records()
            )
            if cal_role and cal_conf > role_conf:
                # Calibrated confidence beats direct inference — use it
                agent_role, role_conf = cal_role, cal_conf

        record.agent_role = agent_role
        record.risk_level = risk_level
        record.primary_risk = primary_risk

        # Scope gap detection — tool scope diverges from stated role
        gap = detect_scope_gap(agent_role, record.tool_scope)
        if gap and not record.role_scope_gap:
            record.role_scope_gap = True
            record.drift_score = min(1.0, record.drift_score + 0.15)
            if "ROLE_SCOPE_GAP" not in record.drift_alerts:
                record.drift_alerts.append("ROLE_SCOPE_GAP")

        record.role_confidence = role_conf
    else:
        record.role_confidence = _compute_role_confidence(record)

    store.update(record)

    # ---- Annotate the event ------------------------------------------------
    event.agent_id = agent_id
    event.agent_display_name = record.display_name
    event.agent_inferred_type = record.agent_type
    event.orchestration_role = record.orchestration_role
    event.prompt_template_hash = (
        record.prompt_template_hashes[-1] if record.prompt_template_hashes else None
    )
    event.system_prompt_version_count = len(record.system_prompt_versions)
    # Module 4 fields
    event.agent_role = record.agent_role
    event.agent_risk_level = record.risk_level
    event.agent_primary_risk = record.primary_risk
    event.role_scope_gap = record.role_scope_gap if record.role_scope_gap else None

    return agent_id
