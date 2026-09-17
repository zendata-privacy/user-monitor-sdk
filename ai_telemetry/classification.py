"""Data classification vocabulary for scope boundary checks.

The manifest declares which classifications an agent is cleared to handle; a
retrieved source carries the classification of what came back. When the second
is not in the first, ``SC_CLASSIFICATION_ESCALATION`` fires. That check is only
as good as the label, so this module is where the label is decided.

────────────────────────────────────────────────────────────────────────────────
THE FIVE CLASSES ARE A CLEARANCE VOCABULARY, NOT A SEVERITY SCALE

The scanner already has a classification enum — ``PIIClassification`` with
HIGH / MEDIUM / LOW / UNKNOWN — and it is tempting to reuse it. It answers a
different question. HIGH/MEDIUM/LOW is a *sensitivity tier*: how much should we
worry. PII/PHI/CREDENTIALS/INTERNAL/PUBLIC is a *data category*: what kind of
thing is this. A clearance check needs the category, because clearances are not
nested — an agent cleared for PHI is not thereby cleared for payment
credentials, and a tier scale cannot express that. The two live side by side:
the tier keeps driving ZASP routing and report severity, the category drives
scope.

Because the categories are a set and not a ladder, ``permits_classification``
is set membership, exactly as the spec defines it. The rank below exists only
to choose which single label to put on a tag when content matches several, and
to phrase the violation message. It is never used to decide whether access is
permitted.

────────────────────────────────────────────────────────────────────────────────
WHERE A CLASSIFICATION COMES FROM, IN PRECEDENCE ORDER

1. **Declared on the source.** The scan already classified every data source it
   found, from resource tags and content sampling. This is free, it is stable
   across retrievals, and it is the only source of truth available before any
   content has been read. It wins.

2. **Detected in the content**, by the Zendata PII API. Accurate, and the only
   thing that catches PHI arriving from a source nobody tagged as clinical. It
   is a network call, so it NEVER runs on the caller's thread — see the note on
   ``PiiApiClassifier`` below.

3. **Detected in the content by the built-in regex classifier** in
   ``detection.py``. Free, inline, and already running on every event. It knows
   twelve categories rather than forty-seven, so it under-reports; it is a
   floor, not a substitute.

A source with no classification from any of the three is ``UNKNOWN``, and
UNKNOWN never fires an escalation. Reporting "we could not tell" as a CRITICAL
violation would train operators to ignore the signal, which costs more than the
detections it would buy.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Set

# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

PII = "PII"
PHI = "PHI"
CREDENTIALS = "CREDENTIALS"
INTERNAL = "INTERNAL"
PUBLIC = "PUBLIC"
UNKNOWN = "UNKNOWN"

CLASSIFICATIONS: tuple = (PII, PHI, CREDENTIALS, INTERNAL, PUBLIC)

# Used ONLY to pick a single winning label when content matches several classes,
# and to word the violation detail. Never consulted for a permit decision.
_RANK: Dict[str, int] = {
    CREDENTIALS: 4,
    PHI: 3,
    PII: 2,
    INTERNAL: 1,
    PUBLIC: 0,
    UNKNOWN: -1,
}


# ---------------------------------------------------------------------------
# Zendata PII API entity types → clearance classes
# ---------------------------------------------------------------------------
# Every one of the 47 types in pii-api/src/constants.py PII_TYPES appears here
# exactly once. A type absent from this table classifies as UNKNOWN, which
# suppresses the signal — so a new entity type added to the PII API without a
# line here fails silently open. _assert_total_coverage() at the bottom of this
# module is what stops that happening unnoticed.

_ENTITY_CLASS: Dict[str, str] = {
    # ── CREDENTIALS ──────────────────────────────────────────────────────
    # Secrets and payment instruments. Separated from PII because possession
    # confers the ability to *act*, not merely to identify. An agent cleared to
    # read customer names is not thereby cleared to read their card numbers.
    "PASSWORD": CREDENTIALS,
    "CVV": CREDENTIALS,
    "CREDIT_CARD": CREDENTIALS,
    "CREDIT_CARD_EXPIRATION": CREDENTIALS,
    "BANK_ACCOUNT": CREDENTIALS,
    "ACCOUNT_NUMBER": CREDENTIALS,

    # ── PHI ──────────────────────────────────────────────────────────────
    # Health data, including the facility name: "patient of <oncology clinic>"
    # is a diagnosis even when no condition is stated, which is why
    # ORGANIZATION_MEDICAL_FACILITY is here and not under INTERNAL.
    "BLOOD_TYPE": PHI,
    "CONDITION": PHI,
    "DOSE": PHI,
    "DRUG": PHI,
    "INJURY": PHI,
    "MEDICAL_PROCESS": PHI,
    "HEALTHCARE_NUMBER": PHI,
    "ORGANIZATION_MEDICAL_FACILITY": PHI,

    # ── PII ──────────────────────────────────────────────────────────────
    # Identifies a natural person, directly or in combination. Includes the
    # GDPR Article 9 special categories (religion, politics, sexuality, origin):
    # the five-class vocabulary has no separate slot for them, so they take the
    # strictest class available to them rather than falling to INTERNAL.
    "NAME": PII,
    "EMAIL_ADDRESS": PII,
    "PHONE_NUMBER": PII,
    "SSN": PII,
    "DOB": PII,
    "AGE": PII,
    "DRIVER_LICENSE": PII,
    "PASSPORT_NUMBER": PII,
    "VEHICLE_ID": PII,
    "IP_ADDRESS": PII,
    "USERNAME": PII,
    "LOCATION_ADDRESS": PII,
    "NUMERICAL_PII": PII,
    "GENDER_SEXUALITY": PII,
    "MARITAL_STATUS": PII,
    "RELIGION": PII,
    "POLITICAL_AFFILIATION": PII,
    "ORIGIN": PII,
    "PHYSICAL_ATTRIBUTE": PII,
    "ZODIAC_SIGN": PII,
    "OCCUPATION": PII,

    # ── INTERNAL ─────────────────────────────────────────────────────────
    # Business content that is not public and does not identify a person.
    "MONEY": INTERNAL,
    "STATISTICS": INTERNAL,
    "ORGANIZATION": INTERNAL,
    "FILENAME": INTERNAL,
    "EVENT": INTERNAL,

    # ── PUBLIC ───────────────────────────────────────────────────────────
    # Carries no sensitivity on its own. Note LOCATION (a city, a country) is
    # public where LOCATION_ADDRESS (a street address) is PII — the PII API
    # draws that distinction and it is worth preserving.
    "DATE": PUBLIC,
    "DATE_INTERVAL": PUBLIC,
    "DURATION": PUBLIC,
    "TIME": PUBLIC,
    "LANGUAGE": PUBLIC,
    "LOCATION": PUBLIC,
    "URL": PUBLIC,
}

# The built-in regex classifier in detection.py speaks a smaller vocabulary of
# its own. Mapped here so the fallback path produces the same five classes.
_BUILTIN_CLASS: Dict[str, str] = {
    "SSN": PII,
    "EMAIL": PII,
    "PHONE": PII,
    "DOB": PII,
    "IP_ADDRESS": PII,
    "EMPLOYEE_ID": PII,
    "CREDIT_CARD": CREDENTIALS,
    "IBAN": CREDENTIALS,
    "BANK_ACCOUNT": CREDENTIALS,
    "ACCOUNT_NUMBER": CREDENTIALS,
    "SALARY": INTERNAL,
    "BANK_BALANCE": INTERNAL,
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def classes_for_entity_types(entity_types: Iterable[str]) -> Set[str]:
    """Clearance classes implied by a set of PII API entity type names."""
    found: Set[str] = set()
    for t in entity_types or ():
        cls = _ENTITY_CLASS.get(str(t).upper())
        if cls:
            found.add(cls)
    return found


def classes_for_builtin_types(pii_types: Iterable[str]) -> Set[str]:
    """Clearance classes implied by ``detection.py`` labels."""
    found: Set[str] = set()
    for t in pii_types or ():
        cls = _BUILTIN_CLASS.get(str(t).upper())
        if cls:
            found.add(cls)
    return found


def dominant(classes: Iterable[str]) -> str:
    """The single label to put on a tag when content matches several classes.

    A tag carries one ``classification`` because the spec's data model does, but
    the checker is handed the full set — see ``ScopeBoundaryChecker`` — so this
    choice never loses a detection. It only decides what the tag reads as.
    """
    best = UNKNOWN
    for c in classes or ():
        if _RANK.get(c, -1) > _RANK.get(best, -1):
            best = c
    return best


def exceeds(classification: str, permitted: Iterable[str]) -> bool:
    """True when *classification* is outside the permitted set.

    UNKNOWN never exceeds. A label we could not determine is not evidence of a
    violation, and raising CRITICAL on absence of evidence is how a signal gets
    turned off by the people it is meant to serve.
    """
    if not classification or classification == UNKNOWN:
        return False
    return classification not in set(permitted or ())


def describe_escalation(classification: str, permitted: Iterable[str]) -> str:
    """Human-readable clause for the violation detail."""
    allowed = sorted(permitted or (), key=lambda c: -_RANK.get(c, -1))
    return (
        f"{classification} is outside the cleared set "
        f"{{{', '.join(allowed) or 'none declared'}}}"
    )


# ---------------------------------------------------------------------------
# Zendata PII API client
# ---------------------------------------------------------------------------

class PiiApiClassifier:
    """Classifies retrieved content by calling the Zendata PII API.

    ────────────────────────────────────────────────────────────────────────
    THIS MUST NOT RUN ON THE CALLER'S THREAD

    It is an HTTPS round trip. The SDK's stated latency budget for the whole
    scope layer is 4ms p99; a network call is two orders of magnitude over that
    on its own. It is safe here only because Phase 1 observes rather than
    enforces, so every scope check runs on the background flush thread inside
    ``TelemetryClient._send()`` — the same thread that already calls the Zendata
    evaluation engine and the drift detector. Moving any of this onto
    ``capture()`` would put a network call on the host application's request
    path, which is the one thing the SDK's contract promises never to do.

    Results are cached by content hash. Retrieval-augmented agents fetch the
    same chunk repeatedly within a session, and the classification of identical
    bytes cannot change — so the cache is not merely an optimisation, it is what
    keeps a chatty agent from generating one API call per turn per source.
    """

    _ENDPOINT = "https://apis.zendata.xyz/pii"

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        timeout_s: float = 5.0,
        max_chars: int = 20_000,
        cache_size: int = 2048,
        min_score: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint or self._ENDPOINT
        self.timeout_s = timeout_s
        # An additional floor on top of the per-entity thresholds pii-api
        # already applies at emission time. Zero means "trust pii-api's own
        # thresholds", which is the right default — but a marginal NAME match
        # in a log line should not be what raises a CRITICAL classification
        # escalation against an agent, so deployments that would rather trade
        # recall for precision can raise it. drae/profiler/pii_detector.py
        # applies the same second floor for the same reason.
        self.min_score = min_score
        # Bound what is sent. A retrieved S3 object can be megabytes; the point
        # is to learn WHICH classes are present, and a prefix answers that for
        # the overwhelming majority of documents at a fraction of the cost. The
        # truncation is recorded by the caller on the tag so nobody later reads
        # a partial scan as a complete one.
        self.max_chars = max_chars
        self.debug = debug

        self._cache: Dict[str, Set[str]] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def classify(self, text: str, content_hash: str) -> Set[str]:
        """Clearance classes present in *text*. Empty set on any failure.

        Never raises. A classification failure must degrade to "we did not
        look", which ``exceeds()`` treats as no violation — the SDK is not
        entitled to report a CRITICAL because its own dependency was down.
        """
        if not text or not isinstance(text, str):
            return set()

        with self._lock:
            hit = self._cache.get(content_hash)
        if hit is not None:
            return set(hit)

        try:
            entity_types = self._call(text[: self.max_chars])
        except Exception as exc:                       # noqa: BLE001 — see docstring
            if self.debug:
                print(f"[AI Telemetry] PII API classification skipped: {exc}")
            return set()

        classes = classes_for_entity_types(entity_types)
        self._remember(content_hash, classes)
        return classes

    # ------------------------------------------------------------------

    def _call(self, text: str) -> Set[str]:
        payload = json.dumps({
            "texts": [text],
            # The scope layer needs labels, not content. Asking for neither the
            # original nor the masked text keeps the response small and means no
            # copy of the retrieved data is held in the SDK any longer than the
            # call itself.
            "includeAnonymized": False,
            "includeOriginalText": False,
            "includeMaskedText": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "x-api-key": self.api_key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        return _entity_types_from_response(body, self.min_score)

    def _remember(self, content_hash: str, classes: Set[str]) -> None:
        with self._lock:
            if content_hash in self._cache:
                return
            self._cache[content_hash] = set(classes)
            self._cache_order.append(content_hash)
            while len(self._cache_order) > self._cache_size:
                self._cache.pop(self._cache_order.pop(0), None)


def _entity_types_from_response(body: Any, min_score: float = 0.0) -> Set[str]:
    """Entity type names out of a PII API response.

    Two response shapes are accepted. The deployed Lambda
    (``pii-api/src/api/lambda_function.py``) takes a ``texts`` array and returns
    a list of per-text results; the FastAPI handler in ``main.py`` returns the
    bare object. Supporting both means the SDK works against either deployment
    with no version negotiation.

    ``score`` is GLiNER's sigmoid similarity, and pii-api has already dropped
    everything under its own per-entity threshold before this sees it. An entity
    without a score is kept: absence means an older response shape, not low
    confidence, and silently discarding it would turn a schema change into a
    signal that quietly stops firing.
    """
    results = body if isinstance(body, list) else [body]
    types: Set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        # classify() reports failures in-band rather than raising.
        if result.get("error"):
            continue
        for entity in result.get("entities") or ():
            if not isinstance(entity, dict) or not entity.get("type"):
                continue
            if min_score:
                try:
                    if float(entity.get("score", 1.0)) < min_score:
                        continue
                except (TypeError, ValueError):
                    pass
            types.add(str(entity["type"]).upper())
    return types


# ---------------------------------------------------------------------------

def _assert_total_coverage() -> None:
    """Fail loudly if the PII API grows a type this table does not map.

    An unmapped type classifies as UNKNOWN and UNKNOWN suppresses the signal, so
    the failure mode of forgetting a line above is a CRITICAL that quietly stops
    firing. That is precisely the kind of silent regression a scope layer cannot
    afford, so it is checked rather than trusted. Import-time cost is one set
    comparison over 47 strings.
    """
    try:
        from ai_telemetry._pii_types import PII_TYPES
    except Exception:
        return  # vendored copy absent — nothing to check against
    missing = set(PII_TYPES) - set(_ENTITY_CLASS)
    if missing:
        raise RuntimeError(
            "ai_telemetry.classification._ENTITY_CLASS does not map these "
            f"Zendata PII API entity types: {sorted(missing)}. Unmapped types "
            "classify as UNKNOWN and silently suppress "
            "SC_CLASSIFICATION_ESCALATION."
        )


_assert_total_coverage()


__all__ = [
    "PII", "PHI", "CREDENTIALS", "INTERNAL", "PUBLIC", "UNKNOWN",
    "CLASSIFICATIONS",
    "classes_for_entity_types", "classes_for_builtin_types",
    "dominant", "exceeds", "describe_escalation",
    "PiiApiClassifier",
]
