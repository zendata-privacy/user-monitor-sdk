"""Build capability manifests from an identity/entitlement export.

The AWS scanner derives a manifest from an IAM policy. Most enterprises already
hold the same statement somewhere else entirely — in the IGA system that granted
the access in the first place. SailPoint, Okta, Entra and the identity exports
they produce all carry the same shape: a principal, and the entitlements it was
granted.

That is a capability manifest in everything but name, and it is arguably a
better one. An IAM policy says what the credential *can* reach; an entitlement
record says what a human *approved* it to reach, and carries the approver. The
gap between the two is itself worth reporting.

────────────────────────────────────────────────────────────────────────────────
ENTITLEMENTS NAME AN ACTION; A MANIFEST NAMES A SOURCE

Entitlements are written ``payroll_db.read``, ``fx_platform.write`` — resource
plus action. The scope boundary is about which *sources* an agent may read, so
the action suffix is stripped and read/write is kept separately rather than
folded in. Keeping it matters: an agent entitled only to ``reporting.read``
that is observed writing is a finding this vocabulary can express and an ARN
list cannot.

Resource names here are logical (``crm``, ``payroll_db``), not ARNs. Nothing in
the matcher requires ARNs — ``_covers`` is a string-boundary rule — so a
manifest in this vocabulary works exactly as one built from a scan, provided
the telemetry names its sources the same way. When both an identity export and
an AWS scan describe the same agent, the scan wins for sources (it is what the
credential can actually reach) and the export supplies the approver.
"""
from __future__ import annotations

import csv
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ai_telemetry.manifest import AgentCapabilityManifest

logger = logging.getLogger(__name__)

# Actions that mean "may read this source". Anything else is retained on the
# record but does not widen read scope.
_READ_ACTIONS = frozenset({"read", "get", "list", "select", "query", "view"})

# Column names seen across the identity exports this has to accept. Ordered by
# preference within each role.
_FIELDS = {
    "id": ("id", "identity_id", "principal_id", "user_id", "sub"),
    "entitlements": ("entitlements", "permissions", "grants", "scopes"),
    "roles": ("roles", "role"),
    "display_name": ("displayName", "display_name", "name"),
    "email": ("email", "mail", "userPrincipalName"),
    "identity_type": ("identity_type", "type", "principal_type"),
    "agent_type": ("agent_type", "agentType"),
    "description": ("description", "purpose", "notes"),
    "approved_by": ("approved_by", "approver", "manager", "owner"),
    "department": ("department",),
    # The CERTIFIED system prompt hash. Without it a roster-derived manifest has
    # nothing for check_manifest_drift to compare against, so SC_MANIFEST_DRIFT
    # cannot fire on this path at all — the boundary would silently cover
    # sources and classifications but not the prompt, which is the one thing an
    # observed baseline can never catch on its own.
    "system_prompt_hash": ("system_prompt_hash", "systemPromptHash",
                           "certified_prompt_hash"),
}


def _pick(row: Dict[str, Any], role: str) -> str:
    for name in _FIELDS[role]:
        value = row.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def split_entitlement(entitlement: str) -> Tuple[str, str]:
    """``payroll_db.read`` -> ``("payroll_db", "read")``.

    An entitlement with no action is treated as read: an export that names a
    bare resource is granting access to it, and assuming otherwise would
    silently drop it from permitted_sources and turn every legitimate read of
    it into a CRITICAL.
    """
    entitlement = (entitlement or "").strip()
    if not entitlement:
        return "", ""
    resource, _, action = entitlement.rpartition(".")
    if not resource:
        return entitlement, "read"
    return resource, action.lower()


def parse_entitlements(raw: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """``(entitled_resources, {resource: [actions]})`` from a delimited field."""
    by_resource: Dict[str, List[str]] = {}
    for part in (raw or "").replace(",", "|").split("|"):
        resource, action = split_entitlement(part)
        if not resource:
            continue
        actions = by_resource.setdefault(resource, [])
        if action and action not in actions:
            actions.append(action)
    # Every entitled resource, whatever the action — NOT just the readable ones.
    #
    # This looked wrong at first and is not. The manifest is compared against
    # what the telemetry reports, and the observed vocabulary (`source_uri`, and
    # the older `data_sources`) names a source that was *touched*; it carries no
    # action. Comparing a source-touch against a read-only list compares two
    # different things, and the result is a CRITICAL every time an agent reads
    # something it holds a write grant on — which on a real entitlement export
    # is most of them. Measured on the demo fixture, restricting to `.read`
    # fired SC_OUT_OF_SCOPE_SOURCE for 274 identities whose scenario is
    # "normal", against 4 that genuinely drifted.
    #
    # The read/write distinction is real and worth keeping, so it is returned
    # separately rather than thrown away — but it belongs to a signal about
    # *what an agent did* to a source, which this vocabulary cannot express and
    # which nothing in the spec asks for.
    return sorted(by_resource), by_resource


def build_manifest(
    row: Dict[str, Any],
    permitted_classifications: Optional[Iterable[str]] = None,
    manifest_version: int = 1,
) -> AgentCapabilityManifest:
    """One manifest from one identity row."""
    agent_id = _pick(row, "id")
    entitled, by_resource = parse_entitlements(_pick(row, "entitlements"))

    manifest = AgentCapabilityManifest(
        agent_id=agent_id,
        manifest_version=manifest_version,
        stated_purpose=_pick(row, "description"),
        permitted_sources=entitled,
        permitted_classifications=sorted(permitted_classifications or []),
        approved_by=_pick(row, "approved_by"),
        system_prompt_hash=_pick(row, "system_prompt_hash"),
        source_documents=[f"identity_export:{agent_id}"],
    )
    manifest.derived_from = {
        "permitted_sources": "identity_entitlements",
        "system_prompt_hash": ("identity_export" if manifest.system_prompt_hash
                               else "not_declared"),
        "permitted_classifications": (
            "supplied" if permitted_classifications else "not_declared"),
    }
    # Write-capable grants, kept as an onboarding observation rather than a
    # runtime signal. An agent entitled to write where its stated purpose is to
    # summarise is worth a conversation, and this is the only place both facts
    # are in hand.
    writable = sorted(r for r, actions in by_resource.items()
                      if any(a not in _READ_ACTIONS for a in actions))
    if writable:
        manifest.scope_gaps = {"write_capable_sources": writable}
    return manifest


def load_csv(
    path: str,
    permitted_classifications: Optional[Iterable[str]] = None,
    only_machine: bool = False,
) -> Dict[str, AgentCapabilityManifest]:
    """Manifests keyed by identity id, from an identity export CSV.

    ``only_machine`` restricts to service accounts and agents. Off by default:
    a human's entitlements bound what an assistant acting on their behalf may
    reach, and dropping them would leave that traffic unscoped.
    """
    manifests: Dict[str, AgentCapabilityManifest] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if only_machine and _pick(row, "identity_type").lower() not in (
                    "machine", "service", "agent", "serviceaccount"):
                continue
            manifest = build_manifest(
                row, permitted_classifications=permitted_classifications)
            if manifest.agent_id and manifest.permitted_sources:
                manifests[manifest.agent_id] = manifest
    logger.info("Loaded %d manifest(s) from %s", len(manifests), path)
    return manifests


class IdentityManifestStore:
    """Manifests for many principals, for a process serving more than one.

    ``TelemetryClient`` holds a single manifest because one process is normally
    one agent. A gateway or a replay evaluating traffic for hundreds of
    identities needs to look the manifest up per event instead, which is what
    this is for.
    """

    def __init__(self, manifests: Optional[Dict[str, AgentCapabilityManifest]] = None):
        self._manifests = dict(manifests or {})

    @classmethod
    def from_csv(cls, path: str, **kw) -> "IdentityManifestStore":
        return cls(load_csv(path, **kw))

    def get(self, agent_id: str) -> Optional[AgentCapabilityManifest]:
        return self._manifests.get(agent_id)

    def __len__(self) -> int:
        return len(self._manifests)

    @property
    def agent_ids(self) -> List[str]:
        return sorted(self._manifests)


__all__ = [
    "build_manifest", "load_csv", "parse_entitlements",
    "split_entitlement", "IdentityManifestStore",
]
