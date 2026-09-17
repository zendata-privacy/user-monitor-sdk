"""Built-in PII classification — labels events, never masks them.

The SDK has always been able to populate ``pii_detected`` / ``pii_types``, but
only through a user-supplied ``pii_redactor``. Without one those fields stay
empty, and an empty field is indistinguishable from "we looked and found
nothing" — so every downstream consumer read clean traffic where nobody had
looked at all. That is the difference between a correlation that can prove a
code risk is live and one that reports LATENT for everything.

This module is the fallback: dependency-free regex classification that runs when
no redactor is configured. It is deliberately NOT a redactor.

  * It **labels**, it does not mask. A redactor's job is to remove PII from what
    gets stored; this only records which categories were present. Pretending to
    redact with regexes would be worse than not redacting — it would look like
    the text had been sanitised.
  * It runs BEFORE prompts are dropped, so ``capture_prompts=False`` still yields
    the PII signal. You get to know that SSNs are flowing through the assistant
    without storing a single one.
  * A real redactor always wins. Pass ``pii_redactor=`` and this never runs.

The labels match the vocabulary the correlator's ``vocab.py`` bridges onto the
code scanner's ``sensitive_data_type``. Changing one without the other silently
breaks the join, which is why they are named here explicitly rather than
inferred.
"""
from __future__ import annotations

import re
from typing import Any

# Ordered most-specific first: a string matching several patterns is labelled
# with every category it matches, but the specific ones carry the signal.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("BANK_ACCOUNT", re.compile(r"\b(?:acct|account)[^\n]{0,12}?\b\d{8,17}\b", re.I)),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?!\d)")),
    ("DOB", re.compile(r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("EMPLOYEE_ID", re.compile(r"\bEMP\d{4,10}\b", re.I)),
    ("SALARY", re.compile(r"\b(?:salary|compensation|payroll|annual pay)\b[^\n]{0,24}?[$€£]?\s?\d[\d,]{3,}", re.I)),
    ("BANK_BALANCE", re.compile(r"\b(?:balance|available funds)\b[^\n]{0,24}?[$€£]?\s?\d[\d,]{2,}", re.I)),
    ("ACCOUNT_NUMBER", re.compile(r"\b(?:account|acct)\s*(?:number|no\.?|#)\s*:?\s*\w{6,}", re.I)),
]

# One combined alternation instead of twelve separate passes. capture() is on
# the caller's request path — the SDK's contract is that it never blocks — so
# scanning each prompt twelve times taxed every LLM call in the host app.
# Named groups keep the label attached to whichever branch matched.
_COMBINED = re.compile(
    "|".join(f"(?P<{label}>{pattern.pattern})" for label, pattern in _PATTERNS),
    re.I)

# Every pattern above needs a digit or an "@" to match. Prose has neither, and
# prose is most of what an assistant handles, so this one cheap scan skips the
# regex engine entirely for the common case.
_TRIGGER = re.compile(r"[\d@]")

# Cardholder-shaped digits are far too easy to match by accident — a 16-digit
# order id, a concatenated timestamp. Luhn is cheap and removes almost all of it.
_CARD_RE = re.compile(r"\d")


def _luhn_ok(value: str) -> bool:
    digits = [int(d) for d in _CARD_RE.findall(value)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def classify_text(text: str) -> set[str]:
    """PII categories present in ``text``. Empty set means none matched."""
    if not text or not isinstance(text, str):
        return set()
    if not _TRIGGER.search(text):
        return set()

    found: set[str] = set()
    for match in _COMBINED.finditer(text):
        label = match.lastgroup
        if label is None or label in found:
            continue
        if label == "CREDIT_CARD" and not _luhn_ok(match.group(0)):
            # A 16-digit run that fails Luhn is not a card number. Reporting it
            # as PCI would put a false regulatory finding in a security report.
            continue
        found.add(label)
    return found


def _walk(value: Any, found: set[str]) -> None:
    if isinstance(value, str):
        found |= classify_text(value)
    elif isinstance(value, dict):
        for v in value.values():
            _walk(v, found)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk(v, found)


def classify_event(event) -> set[str]:
    """Categories present across an event's free-text fields."""
    found: set[str] = set()
    for attr in ("input_messages", "output_message", "tool_input", "tool_output"):
        _walk(getattr(event, attr, None), found)
    return found


def apply_classification_to_event(event) -> None:
    """Populate ``pii_detected`` / ``pii_types`` in place, without masking.

    Merges with anything already set, so a redactor that ran first is never
    overwritten — only supplemented.
    """
    found = classify_event(event)
    if not found:
        # Say "we looked and found nothing" rather than leaving it unknown. An
        # empty string and a False are different claims, and only the second is
        # the one this function is entitled to make.
        if event.pii_detected is None:
            event.pii_detected = False
        return

    existing = {t for t in (event.pii_types or "").split("|") if t}
    event.pii_types = "|".join(sorted(existing | found))
    event.pii_detected = True


__all__ = ["classify_text", "classify_event", "apply_classification_to_event"]
