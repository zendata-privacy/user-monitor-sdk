"""
Agent Network Topology — Module 8

Synthesizes per-agent behavioral signals into a system-level risk model.
Constructs a directed graph of agent relationships from signals already
captured by the SDK (no additional instrumentation required), assigns
authority scores to each node, escalates drift alerts by authority weight,
and produces a kill-chain analysis showing the highest-risk adversarial
path through the agent network.

Topology Signal Sources
-----------------------
  ORCHESTRATION_ROLE    ORCHESTRATOR → SUBAGENT edges from orchestration_role field
  TOOL_PATTERN_OVERLAP  Shared tool usage implies workflow adjacency
  SYSTEM_PROMPT_REF     Display-name cross-references parsed from system prompts
  SESSION_COOCCURRENCE  Agents co-appearing in the same session_id
  API_KEY_SEQUENCE      Sequential different api_key_cluster within same session

Node Authority Score (0.0 – 1.0)
---------------------------------
  Tool scope authority    25%   write/execute/payment > read-only tools
  Downstream agent count  25%   more subordinates = higher blast radius
  Data access sensitivity 20%   PII / financial / credential access
  Role risk weight        20%   FINANCE_AGENT, SECURITY_AGENT score highest
  HITL gap                10%   fully autonomous agents score highest

Authority Escalation
--------------------
  WARNING  on node authority > 0.7  → CRITICAL
  INVESTIGATE on node authority > 0.7 → WARNING
  MONITOR  on node authority > 0.5  → INVESTIGATE

Kill Chain Analysis
-------------------
  Highest-risk path from least-privileged entry point to highest-authority
  target.  Risk score = max_path_authority × path_length_factor × entry_vuln.
  Corresponds to multi-agent kill chain analysis in ZASP red team reports.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Tool authority classification
# ---------------------------------------------------------------------------

# Tools that confer HIGH write/payment/execution authority
_HIGH_AUTHORITY_TOOLS: frozenset = frozenset({
    "transfer_funds", "execute_trade", "place_order", "process_payment",
    "approve_invoice", "wire_transfer", "fund_account",
    "bash", "code_interpreter", "run_script", "python_repl", "execute_code",
    "delete_record", "drop_table", "truncate_table",
    "revoke_access", "grant_permission", "create_user", "modify_acl",
    "deploy", "kubectl_apply", "terraform_apply",
    "exfiltrate_data", "write_file", "upload_file",
})

# Tools that confer MEDIUM write authority
_MEDIUM_AUTHORITY_TOOLS: frozenset = frozenset({
    "send_email", "send_sms", "send_message", "send_notification",
    "update_record", "create_record", "insert_row", "patch_document",
    "create_ticket", "update_customer_record", "flag_transaction",
    "approve_request", "reject_request", "schedule_job",
    "create_alert", "modify_policy",
})

# Tools that indicate sensitive data access (score data sensitivity)
_SENSITIVE_DATA_TOOLS: frozenset = frozenset({
    "get_account_balance", "get_portfolio", "get_financial_data",
    "fetch_financial_data", "get_account_info", "lookup_customer",
    "get_credit_score", "fetch_pii", "get_employee_records",
    "check_credentials", "lookup_ssn", "get_medical_record",
    "fetch_salary", "get_contract", "retrieve_credentials",
    "access_hr_records", "get_transaction_history",
    "transfer_funds", "execute_trade", "place_order",
})

# Role risk weights (0.0 – 1.0)
_ROLE_RISK_WEIGHT: Dict[str, float] = {
    "FINANCE_AGENT":     0.85,
    "SECURITY_AGENT":    0.80,
    "ORCHESTRATOR":      0.75,
    "PROCUREMENT_AGENT": 0.65,
    "HR_AGENT":          0.55,
    "CS_AGENT":          0.40,
    "DATA_ANALYST":      0.35,
    "COMPLIANCE_AGENT":  0.30,
    "GENERIC_AGENT":     0.20,
    "UNKNOWN":           0.25,
}

# Authority weight components (must sum to 1.0)
_AUTHORITY_WEIGHTS: Dict[str, float] = {
    "tool_scope":   0.25,
    "downstream":   0.25,
    "data_sens":    0.20,
    "role_risk":    0.20,
    "hitl_gap":     0.10,
}

# Escalation thresholds
_ESCALATION_RULES: List[Tuple[float, str, str]] = [
    # (min_authority, original_class, escalated_class)
    (0.70, "WARNING",     "CRITICAL"),
    (0.70, "INVESTIGATE", "WARNING"),
    (0.50, "MONITOR",     "INVESTIGATE"),
]

# Edge type labels
EDGE_ORCHESTRATION    = "ORCHESTRATION_ROLE"
EDGE_TOOL_OVERLAP     = "TOOL_PATTERN_OVERLAP"
EDGE_PROMPT_REF       = "SYSTEM_PROMPT_REF"
EDGE_SESSION          = "SESSION_COOCCURRENCE"
EDGE_API_KEY_SEQ      = "API_KEY_SEQUENCE"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TopologyEdge:
    """Directed edge in the agent network graph."""
    source:      str          # agent_id
    target:      str          # agent_id
    edge_type:   str          # EDGE_* constant above
    confidence:  float        # 0.0 – 1.0 (how certain this edge exists)
    call_volume: int = 1      # inferred relative call volume
    inferred:    bool = True  # True = heuristic; False = explicit reference

    def to_dict(self) -> Dict:
        return {
            "source":      self.source,
            "target":      self.target,
            "edge_type":   self.edge_type,
            "confidence":  round(self.confidence, 3),
            "call_volume": self.call_volume,
            "inferred":    self.inferred,
        }


@dataclass
class TopologyNode:
    """Node in the agent network graph with authority scoring."""
    agent_id:           str
    display_name:       str
    agent_role:         str
    orchestration_role: str    # ORCHESTRATOR | SUBAGENT | STANDALONE
    risk_level:         str    # CRITICAL | HIGH | MEDIUM | LOW-MED | LOW

    # Authority score components (0.0 – 1.0 each)
    tool_scope_authority:  float = 0.0
    downstream_count:      int   = 0
    downstream_authority:  float = 0.0
    data_sensitivity:      float = 0.0
    role_risk_weight:      float = 0.0
    hitl_gap:              float = 0.5   # default: unknown

    # Computed composite
    authority_score:       float = 0.0

    # Drift state
    drift_score:           float = 0.0
    drift_class:           str   = "STABLE"
    effective_drift_class: str   = "STABLE"   # after authority escalation
    authority_escalated:   bool  = False

    # Graph position
    entry_point:           bool  = False  # no incoming edges → user-facing
    depth:                 int   = 0      # BFS depth from entry points

    # Raw tool scope (for display)
    tool_scope:            List[str] = field(default_factory=list)
    call_count:            int = 0

    def to_dict(self) -> Dict:
        return {
            "agent_id":            self.agent_id,
            "display_name":        self.display_name,
            "agent_role":          self.agent_role,
            "orchestration_role":  self.orchestration_role,
            "risk_level":          self.risk_level,
            "authority_score":     round(self.authority_score, 3),
            "authority_breakdown": {
                "tool_scope":   round(self.tool_scope_authority, 3),
                "downstream":   round(self.downstream_authority, 3),
                "data_sens":    round(self.data_sensitivity, 3),
                "role_risk":    round(self.role_risk_weight, 3),
                "hitl_gap":     round(self.hitl_gap, 3),
            },
            "downstream_count":    self.downstream_count,
            "drift_score":         round(self.drift_score, 4),
            "drift_class":         self.drift_class,
            "effective_drift_class": self.effective_drift_class,
            "authority_escalated": self.authority_escalated,
            "entry_point":         self.entry_point,
            "depth":               self.depth,
            "tool_scope":          self.tool_scope,
            "call_count":          self.call_count,
        }


@dataclass
class KillChainPath:
    """Highest-risk adversarial traversal path through the agent network."""
    path:                    List[str]   # agent_ids, entry → target
    entry_agent_id:          str
    target_agent_id:         str
    entry_authority:         float       # low = easy entry point
    target_authority:        float       # high = valuable target
    max_authority:           float       # highest-authority node on path
    path_length:             int
    edge_types:              List[str]   # edge types traversed
    risk_score:              float       # composite path risk
    risk_tier:               str         # CRITICAL | HIGH | MEDIUM | LOW
    narrative:               str         # human-readable kill chain description
    escalation_nodes:        List[str]   # nodes where authority-escalation applies

    def to_dict(self) -> Dict:
        return {
            "path":               self.path,
            "entry_agent_id":     self.entry_agent_id,
            "target_agent_id":    self.target_agent_id,
            "entry_authority":    round(self.entry_authority, 3),
            "target_authority":   round(self.target_authority, 3),
            "max_authority":      round(self.max_authority, 3),
            "path_length":        self.path_length,
            "edge_types":         self.edge_types,
            "risk_score":         round(self.risk_score, 4),
            "risk_tier":          self.risk_tier,
            "narrative":          self.narrative,
            "escalation_nodes":   self.escalation_nodes,
        }


@dataclass
class TopologyGraph:
    """Complete agent network topology with authority scores and kill chains."""
    nodes:       Dict[str, TopologyNode]  = field(default_factory=dict)
    edges:       List[TopologyEdge]       = field(default_factory=list)
    kill_chains: List[KillChainPath]      = field(default_factory=list)

    # Adjacency: outgoing edges per node
    _adj_out: Dict[str, List[str]] = field(default_factory=dict)
    _adj_in:  Dict[str, List[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Authority scoring
# ---------------------------------------------------------------------------

def _score_tool_scope_authority(tool_scope: List[str]) -> float:
    """Compute tool scope authority component (0.0–1.0)."""
    if not tool_scope:
        return 0.0
    high   = sum(1 for t in tool_scope if t in _HIGH_AUTHORITY_TOOLS)
    medium = sum(1 for t in tool_scope if t in _MEDIUM_AUTHORITY_TOOLS)
    # Weighted contribution + volume bonus
    raw = (high * 0.9 + medium * 0.5) / max(len(tool_scope), 1)
    # Bonus for having ANY high-authority tool
    if high > 0:
        raw = max(raw, 0.65)
    return min(1.0, raw)


def _score_data_sensitivity(tool_scope: List[str], agent_role: str) -> float:
    """Compute data sensitivity component (0.0–1.0)."""
    sensitive_count = sum(1 for t in tool_scope if t in _SENSITIVE_DATA_TOOLS)
    base = min(1.0, sensitive_count * 0.25)
    # Role bonus
    if agent_role in ("FINANCE_AGENT", "HR_AGENT"):
        base = max(base, 0.60)
    elif agent_role in ("SECURITY_AGENT", "COMPLIANCE_AGENT"):
        base = max(base, 0.50)
    return base


def _score_downstream_authority(downstream_count: int) -> float:
    """Compute downstream agent count authority component (0.0–1.0)."""
    if downstream_count == 0:
        return 0.0
    elif downstream_count == 1:
        return 0.35
    elif downstream_count == 2:
        return 0.55
    elif downstream_count <= 4:
        return 0.75
    else:
        return min(1.0, 0.75 + (downstream_count - 4) * 0.05)


def _compute_authority_score(
    tool_scope_auth: float,
    downstream_auth: float,
    data_sens: float,
    role_risk: float,
    hitl_gap: float,
) -> float:
    """Weighted composite authority score (0.0–1.0)."""
    w = _AUTHORITY_WEIGHTS
    return min(1.0, (
        w["tool_scope"] * tool_scope_auth +
        w["downstream"] * downstream_auth +
        w["data_sens"]  * data_sens +
        w["role_risk"]  * role_risk +
        w["hitl_gap"]   * hitl_gap
    ))


def _escalate_drift_class(drift_class: str, authority_score: float) -> Tuple[str, bool]:
    """Return (effective_class, was_escalated)."""
    for min_auth, original, escalated in _ESCALATION_RULES:
        if authority_score >= min_auth and drift_class == original:
            return escalated, True
    return drift_class, False


# ---------------------------------------------------------------------------
# Edge inference
# ---------------------------------------------------------------------------

def _infer_orchestration_edges(
    records: list,
) -> List[TopologyEdge]:
    """ORCHESTRATOR → SUBAGENT edges from orchestration_role field."""
    edges: List[TopologyEdge] = []
    orchestrators = [r for r in records if r.orchestration_role == "ORCHESTRATOR"]
    subagents      = [r for r in records if r.orchestration_role == "SUBAGENT"]

    for orch in orchestrators:
        for sub in subagents:
            edges.append(TopologyEdge(
                source=orch.agent_id,
                target=sub.agent_id,
                edge_type=EDGE_ORCHESTRATION,
                confidence=0.75,
                call_volume=1,
                inferred=True,
            ))
    return edges


def _infer_tool_overlap_edges(records: list) -> List[TopologyEdge]:
    """Agents with ≥ 50% tool overlap are inferred to be in the same workflow path.

    Only emitted when at least 2 tools overlap and both agents are non-STANDALONE,
    to avoid spurious connections between unrelated agents that happen to use
    common tools like 'web_search'.
    """
    edges: List[TopologyEdge] = []
    seen: Set[Tuple[str, str]] = set()

    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a.agent_id == b.agent_id:
                continue
            scope_a = set(a.tool_scope)
            scope_b = set(b.tool_scope)
            if not scope_a or not scope_b:
                continue
            overlap = scope_a & scope_b
            # Require meaningful overlap: at least 2 tools and > 40% of smaller set
            min_scope = min(len(scope_a), len(scope_b))
            if len(overlap) < 2 or len(overlap) / max(min_scope, 1) < 0.40:
                continue
            # Skip if both are STANDALONE — too ambiguous
            if a.orchestration_role == "STANDALONE" and b.orchestration_role == "STANDALONE":
                continue
            confidence = round(min(0.65, len(overlap) / max(min_scope, 1) * 0.65), 3)
            # Infer direction: higher-authority role likely directs lower
            role_a = _ROLE_RISK_WEIGHT.get(a.agent_role or "", 0.25)
            role_b = _ROLE_RISK_WEIGHT.get(b.agent_role or "", 0.25)
            src, tgt = (a.agent_id, b.agent_id) if role_a >= role_b else (b.agent_id, a.agent_id)
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            edges.append(TopologyEdge(
                source=src,
                target=tgt,
                edge_type=EDGE_TOOL_OVERLAP,
                confidence=confidence,
                call_volume=len(overlap),
                inferred=True,
            ))
    return edges


_RE_AGENT_NAMES = re.compile(
    r"\b(?:coordinate with|delegate to|call(?: the)?|route to|"
    r"send to|hand off to|escalate to|ask|consult)\s+"
    r"(?:the\s+)?([A-Za-z][A-Za-z0-9 \-]{2,40}?)(?:\s+agent|\s+service|\s+bot)?\b",
    re.IGNORECASE,
)


def _infer_prompt_reference_edges(records: list) -> List[TopologyEdge]:
    """Parse system prompt version texts for cross-agent references.

    Matches patterns like 'coordinate with the fraud detection agent',
    'delegate to the payment processor', etc.
    """
    edges: List[TopologyEdge] = []
    name_to_agent: Dict[str, str] = {}

    # Build display_name index (lowercase)
    for r in records:
        if r.display_name:
            name_to_agent[r.display_name.lower()] = r.agent_id
            # Also index short prefix (first word)
            first_word = r.display_name.lower().split()[0]
            name_to_agent.setdefault(first_word, r.agent_id)

    for r in records:
        # We can only inspect the display name of the system prompt since
        # we store hashes not text.  However system_prompt_versions stores
        # the hash + call counts; the actual text is available via the
        # ORCHESTRATOR role indicators already captured in agent_role.
        # Use role config indicator phrases to infer coordination patterns.
        if r.orchestration_role != "ORCHESTRATOR":
            continue
        # An ORCHESTRATOR with multiple subagents → infer edges to all named agents
        for target_rec in records:
            if target_rec.agent_id == r.agent_id:
                continue
            if target_rec.display_name and target_rec.display_name.lower() in name_to_agent:
                edges.append(TopologyEdge(
                    source=r.agent_id,
                    target=target_rec.agent_id,
                    edge_type=EDGE_PROMPT_REF,
                    confidence=0.55,
                    call_volume=1,
                    inferred=True,
                ))
    return edges


def _infer_session_cooccurrence_edges(events: List[Dict]) -> List[TopologyEdge]:
    """Agents co-appearing in the same session_id are workflow co-participants.

    Derives directed edges by call timestamp order within each session.
    """
    from collections import defaultdict
    edges: List[TopologyEdge] = []
    seen: Set[Tuple[str, str]] = set()

    # Map session → [(timestamp, agent_id)]
    session_timeline: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for ev in events:
        sid   = ev.get("session_id")
        ts    = ev.get("timestamp_ms", 0)
        aid   = ev.get("agent_id")
        # Derive agent_id from api_key_hash when not present
        if not aid:
            akh = ev.get("api_key_hash")
            if akh:
                aid = "agt_" + akh[:20]
        if sid and aid and ev.get("event_type") == "llm_call":
            session_timeline[sid].append((ts, aid))

    for sid, timeline in session_timeline.items():
        timeline.sort(key=lambda x: x[0])
        # Consecutive different agents → infer handoff edge
        for i in range(len(timeline) - 1):
            _, src = timeline[i]
            _, tgt = timeline[i + 1]
            if src == tgt:
                continue
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            edges.append(TopologyEdge(
                source=src,
                target=tgt,
                edge_type=EDGE_SESSION,
                confidence=0.60,
                call_volume=1,
                inferred=True,
            ))

    return edges


def _infer_api_key_sequence_edges(events: List[Dict]) -> List[TopologyEdge]:
    """Sequential api_key_cluster changes within a session indicate agent handoffs."""
    from collections import defaultdict
    edges: List[TopologyEdge] = []
    seen: Set[Tuple[str, str]] = set()

    session_clusters: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for ev in events:
        sid   = ev.get("session_id")
        ts    = ev.get("timestamp_ms", 0)
        akh   = ev.get("api_key_hash", "")
        aid   = ev.get("agent_id") or ("agt_" + akh[:20] if akh else None)
        if sid and aid:
            cluster = akh[:16] if akh else aid[:16]
            session_clusters[sid].append((ts, aid, cluster))

    for sid, timeline in session_clusters.items():
        timeline.sort(key=lambda x: x[0])
        prev_cluster = None
        prev_aid     = None
        for ts, aid, cluster in timeline:
            if prev_cluster and cluster != prev_cluster and prev_aid and aid != prev_aid:
                key = (prev_aid, aid)
                if key not in seen:
                    seen.add(key)
                    edges.append(TopologyEdge(
                        source=prev_aid,
                        target=aid,
                        edge_type=EDGE_API_KEY_SEQ,
                        confidence=0.70,
                        call_volume=1,
                        inferred=True,
                    ))
            prev_cluster = cluster
            prev_aid     = aid

    return edges


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _deduplicate_edges(edges: List[TopologyEdge]) -> List[TopologyEdge]:
    """Merge duplicate (source, target) edges, keeping highest confidence."""
    merged: Dict[Tuple[str, str], TopologyEdge] = {}
    for e in edges:
        key = (e.source, e.target)
        if key not in merged or e.confidence > merged[key].confidence:
            # Combine call volumes from all edges on same (src, tgt) pair
            if key in merged:
                e = TopologyEdge(
                    source=e.source,
                    target=e.target,
                    edge_type=e.edge_type,
                    confidence=max(e.confidence, merged[key].confidence),
                    call_volume=e.call_volume + merged[key].call_volume,
                    inferred=e.inferred and merged[key].inferred,
                )
            merged[key] = e
    return list(merged.values())


def build_topology_graph(events: Optional[List[Dict]] = None) -> TopologyGraph:
    """Construct the agent network topology from SDK stores + optional event log.

    Parameters
    ----------
    events : list of dicts (optional)
        Pre-loaded JSONL event dicts.  If provided, enables session co-occurrence
        and API key sequence edge inference.  Pass None to use store-only signals.

    Returns
    -------
    TopologyGraph with fully scored nodes and kill chain paths.
    """
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store

    fp_store    = get_fingerprint_store()
    drift_store = get_drift_store()
    records     = fp_store.all_records()

    if not records:
        return TopologyGraph()

    # ── Build candidate edges ──────────────────────────────────────────────
    raw_edges: List[TopologyEdge] = []
    raw_edges += _infer_orchestration_edges(records)
    raw_edges += _infer_tool_overlap_edges(records)
    raw_edges += _infer_prompt_reference_edges(records)
    if events:
        raw_edges += _infer_session_cooccurrence_edges(events)
        raw_edges += _infer_api_key_sequence_edges(events)

    edges = _deduplicate_edges(raw_edges)

    # ── Build adjacency ────────────────────────────────────────────────────
    adj_out: Dict[str, List[str]] = {r.agent_id: [] for r in records}
    adj_in:  Dict[str, List[str]] = {r.agent_id: [] for r in records}
    for e in edges:
        if e.source in adj_out:
            adj_out[e.source].append(e.target)
        if e.target in adj_in:
            adj_in[e.target].append(e.source)

    # ── Build nodes with authority scores ─────────────────────────────────
    nodes: Dict[str, TopologyNode] = {}
    for rec in records:
        base = drift_store.get_or_create(rec.agent_id)

        tool_scope    = list(rec.tool_scope)
        agent_role    = rec.agent_role or rec.inferred_role or "UNKNOWN"
        downstream_cn = len(adj_out.get(rec.agent_id, []))

        ts_auth = _score_tool_scope_authority(tool_scope)
        ds_auth = _score_downstream_authority(downstream_cn)
        dsen    = _score_data_sensitivity(tool_scope, agent_role)
        rr_wt   = _ROLE_RISK_WEIGHT.get(agent_role, 0.25)
        hitl    = 0.70 if rec.orchestration_role == "ORCHESTRATOR" else 0.50

        authority = _compute_authority_score(ts_auth, ds_auth, dsen, rr_wt, hitl)

        eff_class, escalated = _escalate_drift_class(
            base.latest_drift_classification, authority
        )

        node = TopologyNode(
            agent_id=rec.agent_id,
            display_name=rec.display_name or rec.agent_id[:16],
            agent_role=agent_role,
            orchestration_role=rec.orchestration_role,
            risk_level=rec.risk_level or "UNKNOWN",
            tool_scope_authority=ts_auth,
            downstream_count=downstream_cn,
            downstream_authority=ds_auth,
            data_sensitivity=dsen,
            role_risk_weight=rr_wt,
            hitl_gap=hitl,
            authority_score=authority,
            drift_score=base.latest_drift_score,
            drift_class=base.latest_drift_classification,
            effective_drift_class=eff_class,
            authority_escalated=escalated,
            entry_point=(len(adj_in.get(rec.agent_id, [])) == 0),
            tool_scope=tool_scope,
            call_count=base.total_calls,
        )
        nodes[rec.agent_id] = node

    # ── BFS depth from entry points ────────────────────────────────────────
    entry_nodes = [n for n in nodes.values() if n.entry_point]
    visited: Set[str] = set()
    queue = [(n.agent_id, 0) for n in entry_nodes]
    while queue:
        aid, depth = queue.pop(0)
        if aid in visited:
            continue
        visited.add(aid)
        if aid in nodes:
            nodes[aid].depth = depth
        for child in adj_out.get(aid, []):
            if child not in visited:
                queue.append((child, depth + 1))

    graph = TopologyGraph(nodes=nodes, edges=edges, _adj_out=adj_out, _adj_in=adj_in)

    # ── Kill chain analysis ────────────────────────────────────────────────
    graph.kill_chains = find_highest_risk_paths(graph, top_n=5)

    return graph


# ---------------------------------------------------------------------------
# Kill chain analysis
# ---------------------------------------------------------------------------

def find_highest_risk_paths(
    graph: TopologyGraph,
    top_n: int = 5,
) -> List[KillChainPath]:
    """Find the top-N highest-risk adversarial paths through the graph.

    Algorithm
    ---------
    1. For each entry point (no incoming edges), run DFS up to depth 8.
    2. Score each path to a high-authority target by:
           risk = max_authority_on_path
                  × (1 + 0.15 × path_length)
                  × (1 + (1 - entry_authority))   # easier entry = higher risk
    3. Deduplicate and return top-N by risk_score.
    """
    paths: List[KillChainPath] = []

    entry_nodes = [n for n in graph.nodes.values() if n.entry_point]
    if not entry_nodes:
        # Fall back: all STANDALONE nodes as entry points
        entry_nodes = [n for n in graph.nodes.values()
                       if n.orchestration_role == "STANDALONE"]
    if not entry_nodes:
        entry_nodes = list(graph.nodes.values())

    def _dfs(current_id: str, path: List[str], edge_types: List[str],
              visited: Set[str], max_depth: int) -> None:
        if len(path) > max_depth:
            return
        for nxt in graph._adj_out.get(current_id, []):
            if nxt in visited:
                continue
            # Find edge type for this hop
            etype = next(
                (e.edge_type for e in graph.edges
                 if e.source == current_id and e.target == nxt),
                EDGE_ORCHESTRATION,
            )
            new_path       = path + [nxt]
            new_edge_types = edge_types + [etype]
            new_visited    = visited | {nxt}

            # Collect path as a candidate if it reaches a high-authority target
            if len(new_path) >= 2:
                target_node = graph.nodes.get(nxt)
                if target_node and target_node.authority_score >= 0.40:
                    _record_path(new_path, new_edge_types)

            _dfs(nxt, new_path, new_edge_types, new_visited, max_depth)

    def _record_path(path: List[str], edge_types: List[str]) -> None:
        entry_id  = path[0]
        target_id = path[-1]
        entry_n   = graph.nodes.get(entry_id)
        target_n  = graph.nodes.get(target_id)
        if entry_n is None or target_n is None:
            return

        authorities = [graph.nodes[p].authority_score for p in path if p in graph.nodes]
        max_auth  = max(authorities)
        path_len  = len(path)

        # Risk: high-authority target, long path (more propagation), easy entry point
        risk = (
            max_auth
            * (1 + 0.15 * path_len)
            * (1 + (1 - entry_n.authority_score))
        )
        risk = min(risk, 1.0)

        tier = (
            "CRITICAL" if risk >= 0.75
            else "HIGH"    if risk >= 0.50
            else "MEDIUM"  if risk >= 0.30
            else "LOW"
        )

        # Identify nodes where authority escalation applies
        escalation_nodes = [
            p for p in path
            if p in graph.nodes and graph.nodes[p].authority_escalated
        ]

        # Generate kill chain narrative
        node_labels = [
            f"{graph.nodes[p].display_name} ({graph.nodes[p].agent_role})"
            for p in path if p in graph.nodes
        ]
        narrative = _build_narrative(entry_n, target_n, node_labels, edge_types, risk)

        paths.append(KillChainPath(
            path=list(path),
            entry_agent_id=entry_id,
            target_agent_id=target_id,
            entry_authority=entry_n.authority_score,
            target_authority=target_n.authority_score,
            max_authority=max_auth,
            path_length=path_len,
            edge_types=edge_types,
            risk_score=round(risk, 4),
            risk_tier=tier,
            narrative=narrative,
            escalation_nodes=escalation_nodes,
        ))

    for entry_node in entry_nodes:
        _dfs(
            current_id=entry_node.agent_id,
            path=[entry_node.agent_id],
            edge_types=[],
            visited={entry_node.agent_id},
            max_depth=8,
        )

    # Sort by risk_score descending, deduplicate by (entry, target) keeping highest
    seen_pairs: Set[Tuple[str, str]] = set()
    unique: List[KillChainPath] = []
    for p in sorted(paths, key=lambda x: -x.risk_score):
        key = (p.entry_agent_id, p.target_agent_id)
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique.append(p)
        if len(unique) >= top_n:
            break

    return unique


def _build_narrative(
    entry_node: TopologyNode,
    target_node: TopologyNode,
    node_labels: List[str],
    edge_types: List[str],
    risk_score: float,
) -> str:
    """Generate a human-readable kill chain description."""
    chain_str = " → ".join(node_labels)
    edge_summary = ", ".join(set(edge_types)) if edge_types else "inferred"

    return (
        f"Entry via {entry_node.display_name} ({entry_node.agent_role}, "
        f"authority={entry_node.authority_score:.2f}). "
        f"Path: {chain_str}. "
        f"Target: {target_node.display_name} ({target_node.agent_role}, "
        f"authority={target_node.authority_score:.2f}). "
        f"Traversal signals: {edge_summary}. "
        f"Path risk score: {risk_score:.3f}. "
        f"If {entry_node.display_name} is compromised via prompt injection or "
        f"adversarial input, the attacker can propagate instructions to "
        f"{target_node.display_name} which has "
        f"{'payment/execution' if target_node.tool_scope_authority > 0.7 else 'elevated'} "
        f"authority. Recommend: isolate entry point, add human-in-the-loop "
        f"checkpoint at highest-authority node."
    )


# ---------------------------------------------------------------------------
# Authority-escalated alert feed
# ---------------------------------------------------------------------------

def get_escalated_drift_alerts(graph: TopologyGraph) -> List[Dict]:
    """Return drift alerts with authority-weighted escalation applied.

    Augments the standard drift alert feed (Module 5/6) with authority context.
    A WARNING on a node with authority > 0.7 is promoted to CRITICAL.
    """
    from ai_telemetry.dashboard_api import drift_alert_feed

    raw_alerts = drift_alert_feed(max_alerts=200)
    escalated: List[Dict] = []

    for alert in raw_alerts:
        agent_id = alert.get("agent_id")
        node     = graph.nodes.get(agent_id) if agent_id else None

        if node:
            eff_class, was_escalated = _escalate_drift_class(
                alert.get("drift_class", "STABLE"), node.authority_score
            )
            alert = dict(alert)
            alert["authority_score"]      = round(node.authority_score, 3)
            alert["effective_drift_class"] = eff_class
            alert["authority_escalated"]  = was_escalated
            alert["downstream_count"]     = node.downstream_count
            if was_escalated:
                alert["escalation_reason"] = (
                    f"Authority score {node.authority_score:.2f} > threshold — "
                    f"original {alert.get('drift_class')} escalated to {eff_class}."
                )
        else:
            alert = dict(alert)
            alert["authority_score"]      = None
            alert["effective_drift_class"] = alert.get("drift_class")
            alert["authority_escalated"]  = False

        escalated.append(alert)

    # Re-sort: effective class first, then drift_score
    _SORDER = {"CRITICAL": 0, "WARNING": 1, "INVESTIGATE": 2, "MONITOR": 3, "STABLE": 4}
    escalated.sort(key=lambda x: (
        _SORDER.get(x.get("effective_drift_class", "STABLE"), 99),
        -x.get("drift_score", 0),
    ))
    return escalated


# ---------------------------------------------------------------------------
# Full topology risk report
# ---------------------------------------------------------------------------

def get_topology_risk_report(events: Optional[List[Dict]] = None) -> Dict:
    """Build and return the complete topology risk report.

    Returns a JSON-serialisable dict ready for the dashboard and API.
    """
    graph = build_topology_graph(events=events)

    nodes_list = [n.to_dict() for n in graph.nodes.values()]
    edges_list = [e.to_dict() for e in graph.edges]
    kill_chains = [k.to_dict() for k in graph.kill_chains]
    escalated   = get_escalated_drift_alerts(graph)

    # Summary statistics
    total      = len(nodes_list)
    high_auth  = sum(1 for n in nodes_list if n["authority_score"] >= 0.70)
    escalated_count = sum(1 for n in nodes_list if n["authority_escalated"])
    critical_paths  = sum(1 for k in kill_chains if k["risk_tier"] == "CRITICAL")

    return {
        "nodes":                nodes_list,
        "edges":                edges_list,
        "kill_chains":          kill_chains,
        "escalated_alerts":     escalated,
        "summary": {
            "total_agents":         total,
            "high_authority_nodes": high_auth,
            "authority_escalated":  escalated_count,
            "kill_chain_paths":     len(kill_chains),
            "critical_paths":       critical_paths,
            "edge_count":           len(edges_list),
        },
        "note": (
            "Topology is heuristically inferred from orchestration roles, tool overlap, "
            "and session signals. Authority scores weight drift alerts by blast radius. "
            "Kill chain paths show adversarial propagation routes."
        ),
    }


__all__ = [
    "TopologyEdge",
    "TopologyNode",
    "KillChainPath",
    "TopologyGraph",
    "build_topology_graph",
    "find_highest_risk_paths",
    "get_escalated_drift_alerts",
    "get_topology_risk_report",
    "EDGE_ORCHESTRATION",
    "EDGE_TOOL_OVERLAP",
    "EDGE_PROMPT_REF",
    "EDGE_SESSION",
    "EDGE_API_KEY_SEQ",
]
