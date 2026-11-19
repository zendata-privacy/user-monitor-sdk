"""Source provenance — where retrieved data came from, travelling with the data.

Not to be confused with ``ai_telemetry/provenance.py``, which is **build**
provenance: which repository and commit the running artifact was built from. The
two answer different questions and must never share a module or a field name.
Build provenance says *this code came from that commit*; source provenance says
*this sentence came from that bucket*. The first binds a deployment to a repo,
the second binds a claim to a document.

────────────────────────────────────────────────────────────────────────────────
WHY THE TAG TRAVELS INSTEAD OF BEING LOGGED AT RETRIEVAL

If provenance were only recorded where the retrieval happened, inter-agent
collusion would be structurally invisible. Agent A reads a source it is entitled
to read — no violation. Agent A passes the content to Agent B — no violation,
because a message is not a retrieval. Agent B grounds an answer in data it was
never permitted to see, and no single logged event is anomalous. The violation
exists only in the join, and the join only exists if the provenance is attached
to the payload rather than to A's log line.

Embedding the tag in the message payload is the same pattern as a trace ID in an
HTTP header: the context survives the hop because it is carried, not looked up.

────────────────────────────────────────────────────────────────────────────────
WHY content_hash IS NOT A SIGNATURE

It is not there to prove authorship, and nothing about it is secret. It exists
to answer one question cheaply: did the bytes behind this URI change between two
retrievals in the same session? A vector index chunk whose content shifts
mid-session is either a data integrity problem or someone editing the retrieval
corpus underneath the agent — prompt injection planted in a document the agent
will fetch later. No conversation-layer signal can see that, because every
individual turn looks well-formed; only comparing the same source against itself
over time exposes it.

The comparison is only valid between hashes of the same scope, which is why
``hash_scope`` is recorded. Content is hashed up to a cap — a multi-megabyte S3
object would otherwise dominate the SDK's latency budget — and comparing a
truncated hash against a full one would report a manipulation that never
happened. A signal whose job is to be trusted cannot afford a false positive
manufactured by its own bookkeeping.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ai_telemetry.classification import UNKNOWN

# Hash at most this much content by default. SHA-256 runs around a gigabyte a
# second, so a 1 MB cap costs on the order of a millisecond — inside the 4ms
# budget for the whole scope layer with room for the manifest lookups. Raise it
# only with a measurement in hand.
DEFAULT_MAX_HASH_BYTES = 1_048_576

HASH_SCOPE_FULL = "full"
HASH_SCOPE_TRUNCATED = "truncated"


def hash_content(content: str, max_bytes: int = DEFAULT_MAX_HASH_BYTES) -> tuple:
    """Return ``(sha256_hex, hash_scope, content_length)`` for *content*.

    ``hash_scope`` is what makes a later comparison honest — see the module
    docstring. Two hashes may only be compared when their scopes agree.
    """
    raw = content.encode("utf-8", errors="replace") if isinstance(content, str) else bytes(content or b"")
    total = len(raw)
    if total > max_bytes:
        return hashlib.sha256(raw[:max_bytes]).hexdigest(), HASH_SCOPE_TRUNCATED, total
    return hashlib.sha256(raw).hexdigest(), HASH_SCOPE_FULL, total


@dataclass
class SourceProvenanceTag:
    """Attached to every piece of data retrieved by any agent."""

    source_id: str = ""                 # stable hash of the URI
    source_type: str = ""               # s3_bucket | database | mcp_tool | web | vector_index | memory
    source_uri: str = ""                # the actual URI, ARN or endpoint
    retrieved_by: str = ""              # agent_id that originally retrieved this
    retrieved_at: str = ""              # ISO 8601
    in_manifest: bool = False           # was it in the retrieving agent's manifest
    in_baseline: bool = False           # has it appeared in that agent's baseline
    first_seen: bool = False            # first time this agent has touched it
    classification: str = UNKNOWN       # the dominant class — what the tag reads as

    content_hash: str = ""
    content_length: int = 0
    hash_scope: str = HASH_SCOPE_FULL
    manipulation_delta: bool = False

    # ── Beyond the spec's dataclass, additive ────────────────────────────
    # Every class detected, not only the dominant one. The spec's tag carries a
    # single ``classification`` and the checker would then only ever test that
    # one — so content holding both PHI and PII, reaching an agent cleared for
    # PHI but not PII, would pass. Carrying the full set costs a list and closes
    # that hole; ``classification`` still reads as the single dominant label.
    classifications_detected: List[str] = field(default_factory=list)
    # Which permitted pattern admitted this source, for the audit trail. A
    # violation report that cannot name the rule it matched is not reviewable.
    matched_rule: Optional[str] = None
    tool_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceProvenanceTag":
        t = cls()
        for k, v in (d or {}).items():
            if hasattr(t, k):
                setattr(t, k, v)
        return t

    def classes(self) -> List[str]:
        """Every classification to test, dominant included."""
        found = list(self.classifications_detected)
        if self.classification and self.classification != UNKNOWN and self.classification not in found:
            found.append(self.classification)
        return found


def make_source_id(uri: str) -> str:
    return hashlib.sha256((uri or "").encode("utf-8")).hexdigest()[:16]


# Ordered most specific first: a DynamoDB ARN also contains "arn:aws", so the
# generic patterns have to lose to the specific ones.
_SOURCE_TYPE_RULES = (
    ("arn:aws:s3:", "s3_bucket"),
    ("arn:aws:dynamodb:", "database"),
    ("arn:aws:rds:", "database"),
    ("arn:aws:secretsmanager:", "secret_store"),
    ("arn:aws:bedrock:", "vector_index"),
    ("arn:aws:kendra:", "vector_index"),
    ("vector-index:", "vector_index"),
    ("mcp:", "mcp_tool"),
    ("memory:", "memory"),
    ("redis:", "memory"),
    ("postgres:", "database"),
    ("mysql:", "database"),
    ("mongodb:", "database"),
    ("https:", "web"),
    ("http:", "web"),
    ("file:", "file"),
)


def infer_source_type(uri: str) -> str:
    """Best-effort ``source_type`` from the URI shape."""
    low = (uri or "").lower()
    for prefix, kind in _SOURCE_TYPE_RULES:
        if low.startswith(prefix):
            return kind
    return "unknown"


@dataclass
class InterAgentMessage:
    """Wrapper for agent-to-agent communication.

    ``source_provenance`` is the field the whole feature turns on: it carries
    every upstream tag so the receiving agent's manifest can be evaluated
    against data it did not retrieve itself.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    content: str = ""
    source_provenance: List[SourceProvenanceTag] = field(default_factory=list)
    tool_call_chain: List[str] = field(default_factory=list)
    session_id: str = ""
    turn_id: str = ""
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # Hops this content has already made. Distinct from len(source_provenance):
    # one hop can carry many tags, and one tag can survive many hops.
    hop_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_provenance"] = [t.to_dict() if hasattr(t, "to_dict") else t
                                  for t in self.source_provenance]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterAgentMessage":
        m = cls()
        for k, v in (d or {}).items():
            if k == "source_provenance":
                m.source_provenance = [SourceProvenanceTag.from_dict(t) for t in (v or [])]
            elif hasattr(m, k):
                setattr(m, k, v)
        return m


__all__ = [
    "SourceProvenanceTag",
    "InterAgentMessage",
    "hash_content",
    "make_source_id",
    "infer_source_type",
    "DEFAULT_MAX_HASH_BYTES",
    "HASH_SCOPE_FULL",
    "HASH_SCOPE_TRUNCATED",
]
