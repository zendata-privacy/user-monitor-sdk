"""
PII redaction plug-ins for the AI Telemetry SDK.

The SDK does not ship a built-in redactor. Pass any object exposing
``.redact(text) -> (masked_text, [types])`` (or
``.redact_batch(list[str]) -> list[(masked_text, [types])]``) to
``TelemetryClient(pii_redactor=...)`` and ``apply_redaction_to_event``
will scrub all free-text fields on each event before it is enqueued.

The reference redactor is the Zendata PII API; see
``guide_to_deploy.md`` §7 for setup.
"""

from typing import Any


def apply_redaction_to_event(event, redactor: Any) -> None:
    """Mutate *event* in place, masking PII across all free-text fields.

    Populates ``event.pii_detected`` and ``event.pii_types`` whenever PII
    is found, so downstream risk scoring and dashboards can count hits
    without needing access to the raw text.
    """
    types_seen: set = set()

    def _redact(val: Any) -> Any:
        if isinstance(val, str) and val:
            masked, types = redactor.redact(val)
            if types:
                types_seen.update(types)
            return masked
        if isinstance(val, dict):
            return {k: _redact(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_redact(v) for v in val]
        return val

    if event.input_messages:
        event.input_messages = _redact(event.input_messages)
    if event.output_message:
        event.output_message = _redact(event.output_message)
    if event.tool_input is not None:
        event.tool_input = _redact(event.tool_input)
    if event.tool_output is not None:
        event.tool_output = _redact(event.tool_output)

    if types_seen:
        event.pii_detected = True
        existing = {t for t in (event.pii_types or "").split("|") if t}
        event.pii_types = "|".join(sorted(existing | types_seen))


__all__ = ["apply_redaction_to_event"]
