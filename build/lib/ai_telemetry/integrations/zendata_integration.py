import os
import re
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import TYPE_CHECKING

_EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryEvent

def evaluate_event(event: "TelemetryEvent") -> None:
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
        _mock_evaluate(event)


def _mock_evaluate(event: "TelemetryEvent") -> None:
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
        
    event.pii_detected = len(pii_types) > 0
    event.pii_types = "|".join(pii_types)
    
    # Determine trust boundary crossed
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
    if event.pii_detected:
        composite_risk += 30
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
