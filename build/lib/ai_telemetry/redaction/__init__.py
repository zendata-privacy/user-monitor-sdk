"""
PII redaction plug-ins for the AI Telemetry SDK.

Currently provides:
    * ComprehendRedactor — masks PII via Amazon Comprehend DetectPiiEntities
"""

from ai_telemetry.redaction.comprehend import (
    ComprehendRedactor,
    apply_redaction_to_event,
)

__all__ = ["ComprehendRedactor", "apply_redaction_to_event"]
