"""Agent Capability Manifest — the hard boundary.

The declared scope of an agent: which sources it may read, which classifications
it may handle, and the hash of the system prompt that was certified. Built once
at ``init()`` and compared against on every retrieval. Unlike the behavioural
baseline it is available on the first call of the first session, and unlike the
baseline it cannot be taught to accept drift — which is the whole point of
having it.

That last property is worth stating plainly, because the SDK already contains
the alternative. ``drift_detector.py`` builds a prompt baseline by *observing*:
for the first hundred calls it adds every ``system_prompt_hash`` it sees to
``certified_prompt_hashes`` and alerts on anything new afterwards. If an
unauthorised prompt edit is live during that window, the baseline certifies it
as legitimate and never fires again. A manifest hash is declared by a human who
signed off on it, so the same edit is a violation on the first turn regardless
of when it was made.

────────────────────────────────────────────────────────────────────────────────
WHY SOURCE MATCHING IS NOT ``uri in permitted_sources``

The specification defines the check as exact set membership::

    def permits_source(self, uri: str) -> bool:
        if uri in self.prohibited_sources: return False
        return uri in self.permitted_sources

Every real deployment breaks on this, in both directions.

*Permitted sources are containers; observations are members.* The spec's own
example manifest declares ``arn:aws:s3:::customer-data-bucket``. At runtime the
boto3 interceptor reports what was actually read —
``arn:aws:s3:::customer-data-bucket/reports/q3.pdf``. Exact membership fails,
``SC_OUT_OF_SCOPE_SOURCE`` fires CRITICAL, and it fires on *every object read
from an approved bucket*. That directly contradicts the spec's own acceptance
criterion that approved sources never fire it. Building the manifest from the
scan rather than from a requirements document makes this worse rather than
better, because IAM resources are patterns by nature: policies are written
``arn:aws:s3:::bucket/*``, almost never as a literal object key.

*Prohibited sources have the mirror problem.* An exact-membership deny list
containing ``arn:aws:s3:::hr-records`` does not stop a read of
``arn:aws:s3:::hr-records/salaries.csv``, so the one list whose failure is
unsafe fails open.

So matching is containment-aware, with three rules and one AWS quirk:

1. **Glob.** ``*`` and ``?`` in a declared pattern match as shell globs, because
   that is what an IAM ``Resource`` already is.
2. **Boundary containment.** A declared prefix covers anything beneath it, but
   only at a path or ARN-field boundary — ``/`` or ``:``. The boundary test is
   the security-relevant half: without it ``https://api.example.com`` would
   cover ``https://api.example.com.evil.com/steal``, and a permit list that can
   be widened by registering a lookalike domain is not a boundary at all.
3. **Deny wins, and is evaluated with the same containment rules**, so a deny
   list entry cannot be stepped around by reaching one level deeper.
4. **Secrets Manager suffixes.** AWS appends a random ``-AbCdEf`` to every
   secret ARN. A policy naming ``…:secret:db-creds`` and a runtime read of
   ``…:secret:db-creds-AbCdEf`` are the same secret; the ``-`` is not a boundary
   character, so without a rule for it every secret read is a false CRITICAL.

Matching is case-sensitive. Both sides of the comparison originate from AWS
APIs — the declared side from an IAM policy document, the observed side from a
boto3 event hook — and those agree on case, so normalising would only widen the
match for no gain.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ai_telemetry.classification import (
    CLASSIFICATIONS, CREDENTIALS, INTERNAL, PHI, PII, PUBLIC,
)

# Characters that end a path segment or an ARN field. A declared prefix covers a
# longer URI only when the next character is one of these.
_BOUNDARY = frozenset("/:")

# AWS appends "-" plus six alphanumerics to every Secrets Manager ARN.
_SECRET_SUFFIX = re.compile(r"^-[A-Za-z0-9]{6}$")

_GLOB_CHARS = frozenset("*?[")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _covers(pattern: str, uri: str) -> bool:
    """True when a declared *pattern* covers an observed *uri*.

    See the module docstring for why this is not string equality.
    """
    if not pattern or not uri:
        return False
    if pattern == uri:
        return True

    # A trailing "/*" is the IAM idiom for "everything inside". Treat the
    # container itself as covered too: an agent permitted every object in a
    # bucket is not out of scope for naming the bucket.
    base = pattern[:-2] if pattern.endswith("/*") else pattern
    if base == uri:
        return True

    if _GLOB_CHARS & set(pattern):
        if fnmatch.fnmatchcase(uri, pattern):
            return True
        # Fall through: a pattern may glob in its middle and still be a prefix
        # of the observation ("secret:db-*" vs "secret:db-creds-AbCdEf").

    if uri.startswith(base):
        rest = uri[len(base):]
        if rest and rest[0] in _BOUNDARY:
            return True
        # Secrets Manager's random suffix — see module docstring.
        if ":secret:" in base and _SECRET_SUFFIX.match(rest):
            return True

    return False


def _covered_by_any(patterns: Iterable[str], uri: str) -> Optional[str]:
    """The first pattern covering *uri*, or None. Returns the pattern so the
    violation record can name which declaration matched."""
    for p in patterns or ():
        if _covers(p, uri):
            return p
    return None


@dataclass
class AgentCapabilityManifest:
    """Declared scope for one agent. Versioned; produced at onboarding."""

    agent_id: str = ""
    manifest_version: int = 1
    stated_purpose: str = ""

    permitted_sources: List[str] = field(default_factory=list)
    permitted_source_types: List[str] = field(default_factory=list)
    permitted_tools: List[str] = field(default_factory=list)
    permitted_classifications: List[str] = field(default_factory=list)
    permitted_output_types: List[str] = field(default_factory=list)

    prohibited_sources: List[str] = field(default_factory=list)
    prohibited_outputs: List[str] = field(default_factory=list)

    system_prompt_hash: str = ""
    source_documents: List[str] = field(default_factory=list)
    created_at: str = ""
    approved_by: str = ""

    # ── Beyond the spec's dataclass, additive ────────────────────────────
    # Where each field came from. A manifest assembled from an IAM scan and one
    # typed by hand at init() are not equally trustworthy, and a violation
    # report that cannot say which it was leaves the reader unable to judge it.
    derived_from: Dict[str, str] = field(default_factory=dict)
    # Sources the requirements document claims but the scan did not grant, and
    # the reverse. Not a runtime signal — an onboarding finding. An agent whose
    # PRD promises one bucket and whose role grants forty is worth a
    # conversation before any of this ships.
    scope_gaps: Dict[str, List[str]] = field(default_factory=dict)

    # Which scanned asset this manifest describes. Stamped onto every event so
    # runtime observations can be joined back to the scan record that granted
    # the scope — the join that finally populates ``data_sources_observed`` and
    # makes ``over_privilege_gap`` mean something.
    #
    # Carried explicitly rather than matched on ``app_id == record.name``,
    # for the same reason ``provenance.py`` refuses to guess which repo an
    # artifact came from: a naming convention is an inference, and an inference
    # that silently mismatches produces an agent whose observed sources are
    # attributed to a different asset. The application states it instead.
    asset_arn: str = ""
    account_id: str = ""

    # ------------------------------------------------------------------
    # The two checks the spec defines
    # ------------------------------------------------------------------

    def permits_source(self, uri: str) -> bool:
        """True when *uri* is within declared scope. Deny list wins."""
        if _covered_by_any(self.prohibited_sources, uri):
            return False
        return _covered_by_any(self.permitted_sources, uri) is not None

    def permits_classification(self, classification: str) -> bool:
        """Set membership, exactly as specified.

        Deliberately not a ranked comparison: clearances are categories, not a
        ladder. An agent cleared for PHI is not thereby cleared for payment
        credentials, and any ordering that made it so would be wrong in a way
        that only shows up on the day it matters.
        """
        return classification in set(self.permitted_classifications)

    # ------------------------------------------------------------------
    # Supporting detail for violation records
    # ------------------------------------------------------------------

    def matching_rule(self, uri: str) -> Optional[str]:
        """Which permitted pattern covers *uri*, for the audit trail."""
        return _covered_by_any(self.permitted_sources, uri)

    def prohibiting_rule(self, uri: str) -> Optional[str]:
        """Which deny pattern covers *uri*, if any."""
        return _covered_by_any(self.prohibited_sources, uri)

    def is_unbounded(self) -> bool:
        """True when the manifest permits everything.

        An IAM role with ``Resource: "*"`` produces this, and it means the hard
        boundary cannot fire — every source is in scope. Silently treating that
        as a clean manifest would report an unmonitored agent as a compliant
        one, so it is surfaced rather than hidden.
        """
        return any(p.strip() == "*" for p in self.permitted_sources)

    def declares_classifications(self) -> bool:
        """Whether this manifest states a data clearance at all.

        An empty ``permitted_classifications`` is "nobody said", not "cleared
        for nothing" — and the difference is the whole signal. ``exceeds()``
        compares against a set, so against an empty one EVERY class exceeds:
        an identity roster with no clearance column produces a manifest that
        fires SC_CLASSIFICATION_ESCALATION CRITICAL on the first event of every
        principal. Measured on a real roster of 411 identities, that was 411
        CRITICALs and 2,341 violations, none of them real.

        This is the same failure ``is_enforceable()`` already guards for
        sources, applied to the axis it was never applied to. Absence of a
        declaration is not evidence of a breach.
        """
        return bool(self.permitted_classifications)

    def is_enforceable(self) -> bool:
        """True when this manifest can actually produce a source violation."""
        return bool(self.permitted_sources) and not self.is_unbounded()

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentCapabilityManifest":
        m = cls()
        for k, v in (d or {}).items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m

    def fingerprint(self) -> str:
        """Stable hash of the declared scope, ignoring bookkeeping fields.

        Two manifests with the same fingerprint impose the same boundary. Used
        to decide whether a rebuild is genuinely a new version or the same scope
        re-derived, so ``manifest_version`` counts real changes rather than
        restarts.
        """
        core = {
            "permitted_sources": sorted(self.permitted_sources),
            "permitted_classifications": sorted(self.permitted_classifications),
            "permitted_tools": sorted(self.permitted_tools),
            "prohibited_sources": sorted(self.prohibited_sources),
            "system_prompt_hash": self.system_prompt_hash,
        }
        return _sha256(json.dumps(core, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

# Spec §6. Kept verbatim: the conservatism is the point — a requirements
# document is a statement of intent, and a manifest that infers scope the
# document never granted produces violations nobody can trace to a decision.
MANIFEST_EXTRACTION_PROMPT = """\
You are analyzing a requirements document for an AI agent.
Extract the following as JSON:
{
"stated_purpose": "one sentence describing what the agent does",
"permitted_data_sources": ["list of data source names or URIs mentioned"],
"permitted_tools": ["list of tools or APIs the agent is expected to use"],
"permitted_data_types": ["list of data types the agent may handle"],
"prohibited_outputs": ["list of output types explicitly prohibited"],
"scope_notes": "any scope restrictions mentioned in free text"
}
If a field is not mentioned in the document, return an empty list.
Do not infer or assume. Only extract what is explicitly stated.
"""

# The scan classifies each data source on a sensitivity tier; the manifest needs
# clearance categories. A tier cannot name a category, so this maps only what it
# honestly can: a HIGH-tier source is treated as clearing the agent for personal
# data, not for health data or credentials, both of which have to be stated. The
# effect is a manifest that is narrower than the scan — which is the safe
# direction, since a missing clearance produces an alert a human resolves, while
# a spurious one produces silence.
_TIER_TO_CLASSES: Dict[str, List[str]] = {
    "HIGH":    [PII, INTERNAL, PUBLIC],
    "MEDIUM":  [INTERNAL, PUBLIC],
    "LOW":     [PUBLIC],
    "UNKNOWN": [PUBLIC],
}


class ManifestBuilder:
    """Assembles an ``AgentCapabilityManifest`` from the available inputs.

    Precedence, highest first:

    1. **Explicit ``permitted_sources`` at init.** The spec calls this an
       override and it is treated as one — a human naming the boundary outranks
       anything derived.
    2. **The AWS asset scan.** IAM is enforceable ground truth and it speaks
       ARNs, the same vocabulary the boto3 interceptor reports at runtime, so
       both sides of the comparison need no normalisation layer between them.
    3. **The requirements document**, via LLM extraction.

    When both a scan and a document are present the document does not lose —
    the disagreement is recorded in ``scope_gaps``. A PRD promising one bucket
    against a role granting forty is a finding in its own right, and it is only
    visible at the moment both are in hand.
    """

    def __init__(self, agent_id: str, debug: bool = False) -> None:
        self.agent_id = agent_id
        self.debug = debug

    # ------------------------------------------------------------------

    def build(
        self,
        system_prompt: Optional[str] = None,
        asset_record: Optional[Dict[str, Any]] = None,
        requirements_doc: Optional[bytes] = None,
        extract_fn: Optional[Any] = None,
        permitted_sources: Optional[Sequence[str]] = None,
        permitted_classifications: Optional[Sequence[str]] = None,
        permitted_tools: Optional[Sequence[str]] = None,
        prohibited_sources: Optional[Sequence[str]] = None,
        approved_by: str = "",
        manifest_version: int = 1,
    ) -> AgentCapabilityManifest:
        m = AgentCapabilityManifest(
            agent_id=self.agent_id,
            manifest_version=manifest_version,
            approved_by=approved_by,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        scan = self._from_scan(asset_record) if asset_record else {}
        doc = self._from_document(requirements_doc, extract_fn) if requirements_doc else {}

        # ── system_prompt_hash ───────────────────────────────────────────
        # An explicit prompt at init() wins: it is what the process is actually
        # running, where the scan reports what was deployed. When they differ
        # that IS the drift, and the manifest has to hold the certified value
        # for the check to have anything to compare against.
        if system_prompt:
            m.system_prompt_hash = _sha256(system_prompt)
            m.stated_purpose = _infer_purpose(system_prompt)
            m.derived_from["system_prompt_hash"] = "init"
        elif scan.get("system_prompt_hash"):
            m.system_prompt_hash = scan["system_prompt_hash"]
            m.derived_from["system_prompt_hash"] = "aws_scan"
            if scan.get("system_prompt"):
                m.stated_purpose = _infer_purpose(scan["system_prompt"])

        # ── permitted_sources ────────────────────────────────────────────
        if permitted_sources is not None:
            m.permitted_sources = _clean(permitted_sources)
            m.derived_from["permitted_sources"] = "init_override"
        elif scan.get("sources"):
            m.permitted_sources = _clean(scan["sources"])
            m.derived_from["permitted_sources"] = "aws_scan_iam"
        elif doc.get("sources"):
            m.permitted_sources = _clean(doc["sources"])
            m.derived_from["permitted_sources"] = "requirements_doc"

        # ── permitted_classifications ────────────────────────────────────
        if permitted_classifications is not None:
            m.permitted_classifications = _clean_classes(permitted_classifications)
            m.derived_from["permitted_classifications"] = "init_override"
        elif scan.get("classifications"):
            m.permitted_classifications = _clean_classes(scan["classifications"])
            m.derived_from["permitted_classifications"] = "aws_scan_tier"
        elif doc.get("classifications"):
            m.permitted_classifications = _clean_classes(doc["classifications"])
            m.derived_from["permitted_classifications"] = "requirements_doc"

        # ── tools, prohibitions, purpose ─────────────────────────────────
        m.permitted_tools = _clean(
            permitted_tools if permitted_tools is not None else doc.get("tools", []))
        m.prohibited_sources = _clean(prohibited_sources or [])
        m.prohibited_outputs = _clean(doc.get("prohibited_outputs", []))
        if not m.stated_purpose and doc.get("stated_purpose"):
            m.stated_purpose = doc["stated_purpose"]

        if scan.get("arn"):
            m.source_documents.append(f"aws_scan:{scan['arn']}")
            m.asset_arn = scan["arn"]
        if scan.get("account_id"):
            m.account_id = scan["account_id"]
        if requirements_doc:
            m.source_documents.append(
                f"requirements_doc:sha256:{hashlib.sha256(requirements_doc).hexdigest()[:16]}")

        # ── the cross-check ──────────────────────────────────────────────
        if scan.get("sources") and doc.get("sources"):
            m.scope_gaps = _diff_scope(scan["sources"], doc["sources"])

        if self.debug and not m.is_enforceable():
            reason = ("permits every source (IAM Resource '*')" if m.is_unbounded()
                      else "declares no permitted sources")
            print(f"[AI Telemetry] Manifest for {self.agent_id} {reason} — "
                  f"SC_OUT_OF_SCOPE_SOURCE cannot fire.")
        return m

    # ------------------------------------------------------------------

    @staticmethod
    def _from_scan(record: Dict[str, Any]) -> Dict[str, Any]:
        """Pull manifest inputs out of a scanner ``AssetRecord`` dict.

        Accepts the JSONL shape the scanner already emits, so the platform can
        hand a scan record straight to the SDK with no adapter in between.
        """
        tier = str(record.get("pii_classification") or "UNKNOWN").upper()
        classes = list(_TIER_TO_CLASSES.get(tier, [PUBLIC]))

        # The scan may have resolved real categories by sampling content through
        # the PII API. Those are stated, not inferred from a tier, so they win.
        stated = record.get("data_classifications") or []
        if stated:
            classes = [c for c in stated if c in CLASSIFICATIONS]

        # The scan already extracted the system prompt and hashed it. Ignoring
        # that would leave a scan-derived manifest with no certified hash, and
        # SC_MANIFEST_DRIFT cannot fire without one — silently disabling the
        # drift check on precisely the path we chose as primary. It would also
        # make the manifest fingerprint disagree with the scanner's, so every
        # re-scan would look like a scope change.
        prompt = record.get("system_prompt")
        prompt_hash = record.get("system_prompt_hash") or (
            _sha256(prompt) if prompt else "")

        return {
            "arn": record.get("arn"),
            "account_id": record.get("account_id"),
            "sources": list(record.get("data_sources_permitted") or []),
            "classifications": classes,
            "system_prompt_hash": prompt_hash,
            "system_prompt": prompt,
        }

    def _from_document(
        self, doc: bytes, extract_fn: Optional[Any]
    ) -> Dict[str, Any]:
        """Run the spec §6 extraction over a requirements document.

        ``extract_fn`` is supplied by the host — ``(prompt, document) -> str`` —
        because the SDK has no LLM client and no hard dependencies, and is not
        going to grow one to read a PDF at startup. Absent it, the document is
        recorded as a source but contributes no scope.
        """
        if extract_fn is None:
            if self.debug:
                print("[AI Telemetry] requirements_doc supplied without "
                      "extract_fn — document recorded, no scope extracted.")
            return {}
        try:
            raw = extract_fn(MANIFEST_EXTRACTION_PROMPT, doc)
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception as exc:                        # noqa: BLE001
            if self.debug:
                print(f"[AI Telemetry] Manifest extraction failed: {exc}")
            return {}
        return {
            "stated_purpose": parsed.get("stated_purpose") or "",
            "sources": parsed.get("permitted_data_sources") or [],
            "tools": parsed.get("permitted_tools") or [],
            "classifications": parsed.get("permitted_data_types") or [],
            "prohibited_outputs": parsed.get("prohibited_outputs") or [],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PURPOSE_RE = re.compile(r"you are\s+(?:an?\s+)?([^.\n]{3,120})", re.I)


def _infer_purpose(system_prompt: str) -> str:
    """One line describing the agent, from its own system prompt.

    Mirrors ``agent_fingerprint._infer_display_name``'s "You are X" convention
    rather than inventing a second one, so the two agree on what an agent calls
    itself.
    """
    m = _PURPOSE_RE.search(system_prompt or "")
    return m.group(1).strip() if m else ""


def _clean(values: Iterable[str]) -> List[str]:
    """De-duplicate, drop blanks, preserve a stable order."""
    seen: Dict[str, None] = {}
    for v in values or ():
        s = str(v).strip()
        if s:
            seen.setdefault(s, None)
    return list(seen)


def _clean_classes(values: Iterable[str]) -> List[str]:
    """Keep only recognised clearance classes, upper-cased.

    An unrecognised class in ``permitted_classifications`` can never match a
    label this SDK produces, so it silently narrows the clearance and turns
    legitimate traffic into CRITICALs. Dropping it is the honest failure.
    """
    return [c for c in _clean(str(v).upper() for v in values) if c in CLASSIFICATIONS]


def _diff_scope(scan_sources: Sequence[str], doc_sources: Sequence[str]) -> Dict[str, List[str]]:
    """Where the IAM grant and the requirements document disagree."""
    granted_not_documented = [
        s for s in scan_sources if not _covered_by_any(doc_sources, s)
    ]
    documented_not_granted = [
        s for s in doc_sources if not _covered_by_any(scan_sources, s)
    ]
    gaps: Dict[str, List[str]] = {}
    if granted_not_documented:
        gaps["granted_not_documented"] = sorted(granted_not_documented)
    if documented_not_granted:
        gaps["documented_not_granted"] = sorted(documented_not_granted)
    return gaps


__all__ = [
    "AgentCapabilityManifest",
    "ManifestBuilder",
    "MANIFEST_EXTRACTION_PROMPT",
]
