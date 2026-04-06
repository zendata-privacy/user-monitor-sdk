"""
Dynamic Role Inference — Module 4

Infers agent and human roles from behavioral signals without IAM.

Human Role Inference
--------------------
Accumulates content signal hits from prompt text across sessions per entity.
Maps to a 7-role taxonomy once session and confidence thresholds are met.

  ENGINEER          SQL, APIs, deployment, debugging
  FINANCE           Margins, P&L, deal structuring, budgets
  HR                Headcount, compensation, performance, hiring
  COMPLIANCE        Regulatory citations, audit, risk scoring
  CUSTOMER_SERVICE  Account inquiries, escalation, CRM, returns
  SECURITY          Threat indicators, vulnerability, access control
  EXECUTIVE         Strategic synthesis, cross-domain, competitive

Agent Role Inference
--------------------
Classifies the system prompt against a 9-role enterprise taxonomy, then
validates the classification against the observed tool scope.  A mismatch
(scope gap) is a primary drift signal.

  CS_AGENT           Customer-facing account / inquiry handling
  FINANCE_AGENT      Payments, wire transfers, reconciliation
  ORCHESTRATOR       Coordinates downstream agents, task decomposition
  DATA_ANALYST       Reporting, querying, read-only aggregation
  COMPLIANCE_AGENT   Policy checking, regulatory audit documentation
  PROCUREMENT_AGENT  Vendor selection, purchase orders, invoices
  HR_AGENT           Employee records, compensation, hiring
  SECURITY_AGENT     Threat detection, access control, incident response
  GENERIC_AGENT      Fallback when no strong signal matches

Benchmark-Driven Role Calibration
----------------------------------
When a system prompt embedding is available, the SDK searches all known
AgentRecord entries for cosine similarity > 0.85.  Confirmed roles from
similar agents are inherited with confidence proportional to similarity.
After 10 calls the candidate role is confirmed or re-evaluated.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Human role taxonomy — content signals
# ---------------------------------------------------------------------------

# Each value is a list of phrases / keywords that strongly indicate the role.
# Multi-word phrases are checked with `in` on lowercase prompt text.
_HUMAN_ROLE_SIGNALS: Dict[str, List[str]] = {
    "ENGINEER": [
        "sql", "api rate limit", "deployment pipeline", "error code", "stack trace",
        "debug", "exception", "latency", "throughput", "endpoint", "database schema",
        "migration", "query plan", "index", "dockerfile", "kubernetes", "ci/cd",
        "unit test", "integration test", "dependency", "library", "framework",
        "git", "pull request", "branch", "build failure", "memory leak",
    ],
    "FINANCE": [
        "margin analysis", "deal structuring", "p&l", "profit and loss", "budget",
        "revenue", "churn", "valuation", "ebitda", "cash flow", "financial model",
        "forecast", "variance analysis", "working capital", "irr", "npv", "roi",
        "accounts receivable", "accounts payable", "reconciliation", "ledger",
        "amortization", "depreciation", "earnings", "operating margin",
    ],
    "HR": [
        "headcount", "performance review", "org chart", "compensation", "benefits",
        "hiring", "onboarding", "offboarding", "employee", "talent", "attrition",
        "salary", "equity", "bonus", "job description", "interview", "termination",
        "pto", "promotion", "succession", "culture", "engagement survey",
        "workforce planning", "hris",
    ],
    "COMPLIANCE": [
        "regulatory", "compliance", "audit trail", "risk scoring", "policy",
        "violation", "gdpr", "ccpa", "sox", "pci dss", "hipaa", "aml", "kyc",
        "sec", "finra", "ctr", "sar", "suspicious activity", "sanctions", "ofac",
        "iso 27001", "attestation", "control framework", "remediation", "finding",
        "material weakness", "internal control",
    ],
    "CUSTOMER_SERVICE": [
        "account inquiry", "return policy", "escalation procedure", "crm",
        "support ticket", "customer complaint", "refund", "order status",
        "billing issue", "subscription", "service level", "sla", "case resolution",
        "customer satisfaction", "call center", "chat support", "knowledge base",
        "retention", "churn prevention",
    ],
    "SECURITY": [
        "threat indicator", "vulnerability", "cve", "exploit", "malware",
        "intrusion detection", "incident response", "access control", "firewall",
        "siem", "indicator of compromise", "ioc", "penetration test", "phishing",
        "ransomware", "zero-day", "patch management", "privilege escalation",
        "lateral movement", "threat intelligence", "blue team", "red team",
    ],
    "EXECUTIVE": [
        "strategic", "competitive landscape", "market share", "board of directors",
        "shareholder", "cross-functional", "product roadmap", "company vision",
        "mission statement", "portfolio strategy", "acquisition", "partnership",
        "kpi", "okr", "quarterly review", "annual plan", "executive briefing",
        "go-to-market", "competitive positioning",
    ],
}

# Per-role confidence thresholds and minimum session counts (spec §Module 4)
_HUMAN_ROLE_CONFIG: Dict[str, Dict] = {
    "ENGINEER":         {"threshold": 0.75, "min_sessions": 5},
    "FINANCE":          {"threshold": 0.75, "min_sessions": 5},
    "HR":               {"threshold": 0.70, "min_sessions": 5},
    "COMPLIANCE":       {"threshold": 0.80, "min_sessions": 3},
    "CUSTOMER_SERVICE": {"threshold": 0.70, "min_sessions": 5},
    "SECURITY":         {"threshold": 0.80, "min_sessions": 3},
    "EXECUTIVE":        {"threshold": 0.65, "min_sessions": 8},
}


# ---------------------------------------------------------------------------
# Agent role taxonomy — system prompt indicators + tool validation
# ---------------------------------------------------------------------------

_AGENT_ROLE_CONFIGS: Dict[str, Dict] = {
    "CS_AGENT": {
        "prompt_indicators": [
            "customer service", "account access", "inquiry handling", "support agent",
            "return policy", "crm", "customer-facing", "ticket resolution",
        ],
        "expected_tools": {
            "get_account_info", "lookup_order", "create_ticket",
            "search_knowledge_base", "update_customer_record",
        },
        "forbidden_tools": {
            "transfer_funds", "execute_trade", "place_order",
            "bash", "code_interpreter", "run_script",
        },
        "risk_level": "MEDIUM",
        "primary_risk": "Data access, social engineering",
    },
    "FINANCE_AGENT": {
        "prompt_indicators": [
            "payment", "wire transfer", "reconciliation", "ledger", "transaction",
            "fund transfer", "invoice processing", "financial reconciliation",
            "accounts payable", "accounts receivable",
        ],
        "expected_tools": {
            "transfer_funds", "execute_trade", "place_order", "get_account_balance",
            "process_payment", "approve_invoice",
        },
        "forbidden_tools": {
            "bash", "code_interpreter", "web_search",
        },
        "risk_level": "HIGH",
        "primary_risk": "Transaction authority, AML",
    },
    "ORCHESTRATOR": {
        "prompt_indicators": [
            "coordinate with", "delegate to", "subagent", "task decomposition",
            "sub-agent", "orchestrate", "assign tasks", "workflow coordinator",
            "call the following agents", "route to",
        ],
        "expected_tools": set(),  # orchestrators use any tools
        "forbidden_tools": set(),
        "risk_level": "HIGH",
        "primary_risk": "Privilege escalation, tool chain injection",
    },
    "DATA_ANALYST": {
        "prompt_indicators": [
            "reporting", "querying", "aggregation", "read-only", "analytics",
            "dashboard", "data pipeline", "sql query", "data warehouse",
            "business intelligence", "metrics", "kpi reporting",
        ],
        "expected_tools": {
            "run_sql_query", "get_report", "fetch_data", "read_table",
            "generate_chart", "export_csv",
        },
        "forbidden_tools": {
            "transfer_funds", "execute_trade", "place_order",
            "delete_record", "approve_invoice",
        },
        "risk_level": "MEDIUM",
        "primary_risk": "Bulk data extraction",
    },
    "COMPLIANCE_AGENT": {
        "prompt_indicators": [
            "policy checking", "regulatory language", "audit documentation",
            "compliance review", "risk assessment", "regulatory reporting",
            "aml check", "kyc verification", "sanction screening",
        ],
        "expected_tools": {
            "check_sanctions", "verify_kyc", "generate_audit_report",
            "flag_transaction", "lookup_regulation",
        },
        "forbidden_tools": {
            "transfer_funds", "execute_trade", "bash",
        },
        "risk_level": "LOW-MED",
        "primary_risk": "Regulatory bypass via false premise",
    },
    "PROCUREMENT_AGENT": {
        "prompt_indicators": [
            "vendor selection", "purchase order", "invoice", "approval workflow",
            "procurement", "sourcing", "supplier", "contract management",
            "budget approval", "spend management",
        ],
        "expected_tools": {
            "create_purchase_order", "approve_invoice", "search_vendors",
            "submit_requisition", "get_budget_status",
        },
        "forbidden_tools": {
            "transfer_funds", "bash", "code_interpreter",
        },
        "risk_level": "HIGH",
        "primary_risk": "Financial fraud, invoice injection",
    },
    "HR_AGENT": {
        "prompt_indicators": [
            "employee records", "compensation", "performance review", "hiring",
            "payroll", "benefits administration", "hris", "onboarding",
            "termination process", "org chart management",
        ],
        "expected_tools": {
            "get_employee_record", "update_compensation", "create_job_requisition",
            "run_payroll", "manage_benefits",
        },
        "forbidden_tools": {
            "transfer_funds", "execute_trade", "bash",
        },
        "risk_level": "HIGH",
        "primary_risk": "Sensitive PII, employment discrimination",
    },
    "SECURITY_AGENT": {
        "prompt_indicators": [
            "threat detection", "access control", "incident response",
            "security logs", "vulnerability scanning", "siem", "firewall rules",
            "intrusion detection", "threat intelligence", "forensic analysis",
        ],
        "expected_tools": {
            "query_siem", "block_ip", "revoke_access", "run_scan",
            "fetch_threat_intel", "isolate_host",
        },
        "forbidden_tools": set(),  # security agents legitimately need wide access
        "risk_level": "CRITICAL",
        "primary_risk": "Privilege scope, lateral movement",
    },
    "GENERIC_AGENT": {
        "prompt_indicators": [],
        "expected_tools": set(),
        "forbidden_tools": set(),
        "risk_level": "LOW",
        "primary_risk": "Unknown",
    },
}

_SIMILARITY_THRESHOLD = 0.85   # cosine similarity for benchmark calibration
_CONFIRMED_CONFIDENCE = 0.80   # minimum source confidence for calibration donor
_CALIBRATION_CALLS = 10        # calls before candidate role is re-evaluated


# ---------------------------------------------------------------------------
# Human role store
# ---------------------------------------------------------------------------


@dataclass
class HumanRoleRecord:
    """Per-entity accumulated role evidence across all sessions."""
    entity_key: str
    session_count: int = 0
    prompt_count: int = 0
    # role → total signal hits across all prompts
    role_hits: Dict[str, int] = field(default_factory=lambda: {r: 0 for r in _HUMAN_ROLE_CONFIG})
    inferred_role: Optional[str] = None
    role_confidence: float = 0.0
    role_locked: bool = False   # True once the role is confidently confirmed


class HumanRoleStore:
    """Thread-safe, in-memory store for per-entity human role evidence."""

    def __init__(self) -> None:
        self._records: Dict[str, HumanRoleRecord] = {}
        self._sessions: Dict[str, Set[str]] = {}  # entity_key → set of session_ids
        self._lock = threading.Lock()

    def get_or_create(self, entity_key: str) -> HumanRoleRecord:
        with self._lock:
            if entity_key not in self._records:
                self._records[entity_key] = HumanRoleRecord(entity_key=entity_key)
            return self._records[entity_key]

    def record_session(self, entity_key: str, session_id: str) -> int:
        """Mark session_id as seen for this entity.  Returns updated session count."""
        with self._lock:
            if entity_key not in self._sessions:
                self._sessions[entity_key] = set()
            self._sessions[entity_key].add(session_id)
            count = len(self._sessions[entity_key])
            if entity_key in self._records:
                self._records[entity_key].session_count = count
            return count

    def update(self, record: HumanRoleRecord) -> None:
        with self._lock:
            self._records[record.entity_key] = record


# Module-level singleton
_human_store = HumanRoleStore()


def get_human_role_store() -> HumanRoleStore:
    return _human_store


# ---------------------------------------------------------------------------
# Human role inference logic
# ---------------------------------------------------------------------------


def _count_role_signals(prompt_text: str) -> Dict[str, int]:
    """Count signal hits for every human role in one prompt (case-insensitive)."""
    text_lower = prompt_text.lower()
    return {
        role: sum(1 for sig in signals if sig in text_lower)
        for role, signals in _HUMAN_ROLE_SIGNALS.items()
    }


def _compute_human_role_confidence(record: HumanRoleRecord) -> Tuple[Optional[str], float]:
    """Return (best_role, confidence) from accumulated hits.

    confidence = (hits for best role) / (total hits across all roles)
    Only roles that individually exceed their threshold are considered.
    """
    total = sum(record.role_hits.values())
    if total == 0:
        return None, 0.0

    proportions = {
        role: record.role_hits[role] / total
        for role in _HUMAN_ROLE_CONFIG
    }
    best_role = max(proportions, key=proportions.get)
    best_conf = proportions[best_role]
    return best_role, best_conf


def update_human_role(
    entity_key: str,
    session_id: str,
    prompt_text: str,
) -> Tuple[Optional[str], float]:
    """Accumulate evidence for *entity_key* from one prompt.

    Updates the HumanRoleRecord in the store and returns the current
    (inferred_role, confidence).  Role is only locked once both the
    confidence threshold and minimum session count are met.
    """
    store = _human_store
    session_count = store.record_session(entity_key, session_id)
    record = store.get_or_create(entity_key)

    if record.role_locked:
        return record.inferred_role, record.role_confidence

    # Accumulate signal hits from this prompt
    hits = _count_role_signals(prompt_text)
    record.prompt_count += 1
    for role, count in hits.items():
        record.role_hits[role] += count

    # Recompute confidence
    best_role, best_conf = _compute_human_role_confidence(record)

    if best_role is not None:
        cfg = _HUMAN_ROLE_CONFIG[best_role]
        if best_conf >= cfg["threshold"] and record.session_count >= cfg["min_sessions"]:
            record.inferred_role = best_role
            record.role_confidence = best_conf
            record.role_locked = True
        else:
            # Store provisional result even if not yet locked
            record.inferred_role = best_role if best_conf > 0.30 else None
            record.role_confidence = best_conf

    store.update(record)
    return record.inferred_role, record.role_confidence


def get_human_role(entity_key: str) -> Tuple[Optional[str], float]:
    """Return the current (inferred_role, confidence) for an entity."""
    record = _human_store.get_or_create(entity_key)
    return record.inferred_role, record.role_confidence


# ---------------------------------------------------------------------------
# Agent role inference logic
# ---------------------------------------------------------------------------


def _score_system_prompt(system_prompt: str) -> Dict[str, float]:
    """Score the system prompt against each agent role's indicators."""
    text_lower = system_prompt.lower()
    scores: Dict[str, float] = {}
    for role, cfg in _AGENT_ROLE_CONFIGS.items():
        indicators = cfg["prompt_indicators"]
        if not indicators:
            scores[role] = 0.0
            continue
        hits = sum(1 for ind in indicators if ind in text_lower)
        scores[role] = hits / len(indicators)
    return scores


def _score_tool_scope(tool_scope: List[str], role: str) -> float:
    """Return 0–1 coverage of a role's expected tools in the observed scope.

    When *tool_scope* is empty there is no tool evidence for any role, so all
    roles return 0.0.  The neutral 0.5 is only used when the role does not
    constrain tools (empty expected_tools) but tools *have* been observed —
    meaning the role is compatible with whatever tools are present.
    """
    if not tool_scope:
        return 0.0  # no tool evidence — don't inflate any role
    expected = _AGENT_ROLE_CONFIGS[role]["expected_tools"]
    if not expected:
        return 0.5  # role doesn't constrain tools; neutral when tools are observed
    scope_set = set(tool_scope)
    overlap = len(scope_set & expected)
    return overlap / len(expected)


def detect_scope_gap(agent_role: str, tool_scope: List[str]) -> bool:
    """Return True if any forbidden tool for the declared role appears in scope."""
    forbidden = _AGENT_ROLE_CONFIGS.get(agent_role, {}).get("forbidden_tools", set())
    if not forbidden:
        return False
    return bool(set(tool_scope) & forbidden)


def infer_agent_role(
    system_prompt: str,
    tool_scope: List[str],
) -> Tuple[str, float, str, str]:
    """Infer the agent's functional role from system prompt + tool scope.

    Returns (role, confidence, risk_level, primary_risk).

    Algorithm
    ---------
    combined_score[role] = 0.7 * prompt_score[role] + 0.3 * tool_score[role]
    confidence = combined_score[best] / max_possible (normalised 0–1)
    """
    prompt_scores = _score_system_prompt(system_prompt)

    combined: Dict[str, float] = {}
    for role in _AGENT_ROLE_CONFIGS:
        if role == "GENERIC_AGENT":
            combined[role] = 0.0
            continue
        p_score = prompt_scores.get(role, 0.0)
        t_score = _score_tool_scope(tool_scope, role)
        combined[role] = 0.70 * p_score + 0.30 * t_score

    best_role = max(combined, key=combined.get)
    best_score = combined[best_role]

    if best_score < 0.05:
        best_role = "GENERIC_AGENT"
        confidence = 0.30
    else:
        # Normalize: confidence relative to theoretical max (all indicators hit)
        confidence = min(0.95, best_score * 2.5)

    cfg = _AGENT_ROLE_CONFIGS[best_role]
    return best_role, round(confidence, 4), cfg["risk_level"], cfg["primary_risk"]


# ---------------------------------------------------------------------------
# Benchmark-driven calibration (cosine similarity)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity; returns 0.0 on length mismatch or zero norms."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def benchmark_calibrate(
    new_embedding: List[float],
    new_call_count: int,
    all_records: list,          # List[AgentRecord] — avoid circular import with string hint
) -> Tuple[Optional[str], float]:
    """Search known agent records for embedding similarity > threshold.

    Returns (candidate_role, confidence) or (None, 0.0) if no match found.

    On the first call the search considers all records; after ``_CALIBRATION_CALLS``
    the new agent's own behavioral data outweighs the inherited role.

    Confidence degrades linearly to zero influence once call_count >= CALIBRATION_CALLS
    so the agent's observed behaviour takes over.
    """
    if not new_embedding or new_call_count >= _CALIBRATION_CALLS:
        return None, 0.0

    candidates: List[Tuple[float, str, float]] = []  # (similarity, role, donor_confidence)
    for rec in all_records:
        if not getattr(rec, "system_prompt_embedding", None):
            continue
        donor_role = getattr(rec, "agent_role", None)
        donor_conf = getattr(rec, "role_confidence", 0.0)
        if not donor_role or donor_conf < _CONFIRMED_CONFIDENCE:
            continue
        sim = _cosine_similarity(new_embedding, rec.system_prompt_embedding)
        if sim >= _SIMILARITY_THRESHOLD:
            candidates.append((sim, donor_role, donor_conf))

    if not candidates:
        return None, 0.0

    # Vote: weight each candidate by similarity * donor_confidence
    vote_weights: Dict[str, float] = {}
    for sim, role, donor_conf in candidates:
        vote_weights[role] = vote_weights.get(role, 0.0) + sim * donor_conf

    best_role = max(vote_weights, key=vote_weights.get)
    # Scale confidence by how far we are from the calibration call limit
    # (fades from full borrowed confidence → 0 as call_count approaches threshold)
    decay = 1.0 - (new_call_count / _CALIBRATION_CALLS)
    raw_conf = min(0.90, vote_weights[best_role] / max(len(candidates), 1))
    return best_role, round(raw_conf * decay, 4)
