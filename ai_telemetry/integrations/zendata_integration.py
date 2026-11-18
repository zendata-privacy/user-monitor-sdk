import os
import re
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import TYPE_CHECKING

_SENSITIVE_PII = frozenset({
    "SSN", "EMAIL", "EMAIL_ADDRESS", "ACCOUNT_NUMBER", "BANK_ACCOUNT", "IBAN",
    "CREDIT_CARD", "CVV", "PASSPORT_NUMBER", "DRIVER_LICENSE", "DOB", "PASSWORD",
    "HEALTHCARE_NUMBER",
})
_EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryEvent

def evaluate_event(event: "TelemetryEvent", permitted_classifications=None) -> None:
    """
    Evaluates the LLM interaction against the Zendata DLP & Security Engine.
    Enriches the event object with the results.
    """
    engine_url = os.environ.get("ZENDATA_ENGINE_URL")
    api_key = os.environ.get("ZENDATA_API_KEY", "")
    mode = os.environ.get("ZENDATA_EVALUATION_MODE", "async")

    if not engine_url:
        # Silently skip if Zendata is not configured
        return

    payload = {
        "user_id": event.user_id,
        "sailpoint_risk": event.sailpoint_risk,
        "prompt": str(event.input_messages) if event.input_messages else "",
        "completion": str(event.output_message) if event.output_message else "",
        "tools_called": event.tools_called or "",
        "data_sources": event.data_sources or "",
    }
    
    # In a real environment, we'd make an HTTP request to Zendata engine
    # For now, we simulate a mock Zendata integration if the URL is "mock" or we catch a connection error.
    try:
        if engine_url == "mock":
            raise urllib.error.URLError("Mock URL")
            
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            engine_url, 
            data=data, 
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode())
            event.pii_detected = result.get("pii_detected", False)
            event.pii_types = result.get("pii_types", "")
            event.trust_boundary_crossed = result.get("trust_boundary_crossed", False)
            event.risk_signal = result.get("risk_signal", "LOW")
            event.zendata_composite_risk = result.get("zendata_composite_risk", event.sailpoint_risk)
            event.zendata_classification = result.get("zendata_classification", "Internal")
    except Exception as e:
        # Fallback Mock logic for demonstration/testing
        _mock_evaluate(event, permitted_classifications)


def _sensitive_classes(labels) -> set:
    """Clearance classes (PII, CREDENTIALS, ...) of sensitive labels, whichever
    labeller produced them (built-in regex names or PII API entity types)."""
    from ai_telemetry.classification import classes_for_builtin_types, classes_for_entity_types
    labels = list(labels)
    return classes_for_builtin_types(labels) | classes_for_entity_types(labels)


def _mock_evaluate(event: "TelemetryEvent", permitted_classifications=None) -> None:
    """Mock implementation for demonstration."""
    content = ""
    if event.input_messages:
        content += str(event.input_messages)
    if event.output_message:
        content += str(event.output_message)
    content = content.lower()
        
    pii_types = []
    if "ssn" in content or "social security" in content:
        pii_types.append("SSN")
    if _EMAIL_PATTERN.search(content):
        pii_types.append("Email")

    # Merge with labels already on the event (the inline regex labeller and the
    # Zendata PII service). This used to OVERWRITE them, so a model-detected
    # SSN or account number vanished from pii_types the moment the mock ran.
    existing = [t for t in (event.pii_types or "").split("|") if t]
    merged = sorted(set(existing) | set(pii_types))
    event.pii_detected = bool(merged)
    event.pii_types = "|".join(merged)

    # The +30 composite bump is for SENSITIVE IDENTIFIERS — the class the mock
    # always meant (SSN, email) — extended to the equivalent labels the regex
    # labeller and PII service emit. Generic entities (NAME, DATE, ORGANIZATION,
    # MONEY ...) are labelled but do not move risk: nearly every banking answer
    # contains one, and bumping on them would make every conversation HIGH.
    sensitive = {t.upper() for t in merged} & _SENSITIVE_PII

    # Trust boundary. When the host declared where data went
    # (context.record_egress), capture() already set it from that — e.g. an
    # email to a colleague inside the organisation is not a crossing. Only for
    # hosts that declare nothing does the old heuristic apply: the word
    # "external" in the conversation, or send_email among the tools.
    if event.trust_boundary_crossed is None:
        event.trust_boundary_crossed = False
        if "external" in content or event.tools_called and "send_email" in event.tools_called:
            event.trust_boundary_crossed = True
        
    # Calculate composite risk
    base_risk = 0
    try:
        base_risk = int(event.sailpoint_risk) if event.sailpoint_risk else 10
    except ValueError:
        pass
        
    composite_risk = base_risk
    # Sensitive PII scores by CLEARANCE when the caller's is known: +30 when
    # it is of a class the caller is not cleared for, +10 when it is (an HR
    # officer reading an SSN in HR records is doing their job, but it is still
    # sensitive data in flight). No declared clearance: +30, as before.
    if sensitive:
        if permitted_classifications is None:
            composite_risk += 30
        elif _sensitive_classes(sensitive) - set(permitted_classifications):
            composite_risk += 30
        else:
            composite_risk += 10
    if event.trust_boundary_crossed:
        composite_risk += 40
        
    event.zendata_composite_risk = str(min(100, composite_risk))
    
    if composite_risk >= 80:
        event.risk_signal = "CRITICAL"
        event.zendata_classification = "Highly Confidential"
    elif composite_risk >= 50:
        event.risk_signal = "HIGH"
        event.zendata_classification = "Confidential"
    else:
        event.risk_signal = "LOW"
        event.zendata_classification = "Internal"
