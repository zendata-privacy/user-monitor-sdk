"""Tests for the built-in PII classifier.

It exists because an unset ``pii_types`` is indistinguishable from "we looked and
found nothing", so every downstream consumer read clean traffic where nobody had
looked at all. Two properties matter most and are asserted hardest here: it must
not invent PII, and it must not block ``capture()``.
"""
import csv
import os
import tempfile
import time

import pytest

from ai_telemetry.client import TelemetryClient, TelemetryEvent
from ai_telemetry.detection import (apply_classification_to_event, classify_text,
                                    classify_event)


@pytest.mark.parametrize("text,expected", [
    ("SSN 123-45-6789 on file",              "SSN"),
    ("write to ada.lovelace@acme.io",        "EMAIL"),
    ("card 4111 1111 1111 1111",             "CREDIT_CARD"),
    ("call me on (415) 555-0132",            "PHONE"),
    ("employee EMP10601 started monday",     "EMPLOYEE_ID"),
    ("annual salary $184,500 approved",      "SALARY"),
    ("host 192.168.14.22 unreachable",       "IP_ADDRESS"),
    ("born 1987-03-14",                      "DOB"),
])
def test_detects(text, expected):
    assert expected in classify_text(text)


@pytest.mark.parametrize("text", [
    "Analyze customer churn patterns in retail banking this quarter",
    "List open incidents from the past 7 days",
    "Generate a payroll reconciliation summary for March",
    "",
])
def test_does_not_invent_pii(text):
    """A false PII finding in a security report is worse than a missed one — it
    sends someone to investigate a leak that never happened."""
    assert classify_text(text) == set()


@pytest.mark.parametrize("text", [
    "order 1234567890123456 shipped",     # 16 digits, fails Luhn
    "card 1234 5678 9012 3456",           # fails Luhn
])
def test_luhn_rejects_card_shaped_numbers(text):
    """Any 16-digit run looks like a card. Reporting one as PCI would put a
    false regulatory finding in front of a compliance team."""
    assert "CREDIT_CARD" not in classify_text(text)


def test_classifies_nested_event_fields():
    ev = TelemetryEvent(event_type="llm_call",
                        input_messages=[{"role": "user", "content": "ssn 123-45-6789"}],
                        tool_output={"rows": ["ada@acme.io"]})
    assert classify_event(ev) == {"SSN", "EMAIL"}


def test_labels_without_masking():
    """It is a classifier, not a redactor. Pretending to redact with regexes
    would be worse than not redacting — the text would look sanitised."""
    original = [{"role": "user", "content": "ssn 123-45-6789"}]
    ev = TelemetryEvent(event_type="llm_call", input_messages=list(original))
    apply_classification_to_event(ev)
    assert ev.input_messages == original
    assert ev.pii_types == "SSN" and ev.pii_detected is True


def test_clean_event_records_false_not_unknown():
    ev = TelemetryEvent(event_type="llm_call",
                        input_messages=[{"role": "user", "content": "hello"}])
    apply_classification_to_event(ev)
    assert ev.pii_detected is False
    assert not ev.pii_types


def test_merges_with_an_existing_redactor_result():
    ev = TelemetryEvent(event_type="llm_call", pii_types="MEDICAL_RECORD",
                        input_messages=[{"role": "user", "content": "ssn 123-45-6789"}])
    apply_classification_to_event(ev)
    assert ev.pii_types == "MEDICAL_RECORD|SSN"


def test_survives_capture_prompts_false():
    """The signal must outlive the prompt: you learn SSNs are flowing through the
    assistant without storing a single one."""
    d = tempfile.mkdtemp()
    log = os.path.join(d, "events.csv")
    c = TelemetryClient(app_id="t", api_key="k", enable_local_log=True,
                        log_path=log, capture_prompts=False)
    c.capture(TelemetryEvent(event_type="llm_call",
              input_messages=[{"role": "user", "content": "ssn 123-45-6789"}]))
    c.flush(); c.shutdown()
    row = list(csv.DictReader(open(log, encoding="utf-8-sig")))[0]
    assert row["pii_types"] == "SSN"


def test_a_supplied_redactor_wins():
    class _Redactor:
        def redact(self, text):
            return "[MASKED]", ["CUSTOM_TYPE"]

    c = TelemetryClient(app_id="t", api_key="k", pii_redactor=_Redactor())
    ev = TelemetryEvent(event_type="llm_call",
                        input_messages=[{"role": "user", "content": "ssn 123-45-6789"}])
    c.capture(ev)
    assert "CUSTOM_TYPE" in (ev.pii_types or "")
    assert "MASKED" in str(ev.input_messages)


def test_can_be_disabled():
    c = TelemetryClient(app_id="t", api_key="k", classify_pii=False)
    ev = TelemetryEvent(event_type="llm_call",
                        input_messages=[{"role": "user", "content": "ssn 123-45-6789"}])
    c.capture(ev)
    assert not ev.pii_types


def test_classification_does_not_block_capture():
    """capture() runs on the host application's request path. The classifier
    must not put its module import or regex compilation there."""
    c = TelemetryClient(app_id="t", api_key="k")
    payload = "Summarise account activity for ada@acme.io " * 20
    c.capture(TelemetryEvent(event_type="llm_call",
              input_messages=[{"role": "user", "content": payload}]))   # warm
    start = time.perf_counter()
    c.capture(TelemetryEvent(event_type="llm_call",
              input_messages=[{"role": "user", "content": payload}]))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 1.0, f"capture() took {elapsed_ms:.2f}ms"


def test_prose_skips_the_regex_engine():
    prose = "Analyze customer churn patterns in retail banking " * 200
    start = time.perf_counter()
    for _ in range(200):
        classify_text(prose)
    per_call_ms = (time.perf_counter() - start) * 1000 / 200
    assert per_call_ms < 0.5, f"{per_call_ms:.3f}ms per call on plain prose"
