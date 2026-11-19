"""
AI Telemetry SDK - Anthropic Integration
Monkey-patches the Anthropic client to automatically capture telemetry.

Usage:
    from ai_telemetry.integrations.anthropic_integration import patch_anthropic
    patch_anthropic(telemetry_client)
"""

import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryClient


def _capture_system_prompt_fields(event, kwargs, telemetry) -> None:
    """Populate system_prompt_hash and system_prompt_embedding on *event*.

    Anthropic passes the system prompt as a top-level ``system`` kwarg (string).
    Some callers also embed a role="system" entry in the messages list; both
    patterns are checked so no context is missed.
    """
    from ai_telemetry.client import _sha256
    # Anthropic-native: system= kwarg
    system_text = kwargs.get("system") or ""
    # Fallback: OpenAI-style role="system" in messages list
    if not system_text:
        messages = kwargs.get("messages", [])
        system_text = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"),
            "",
        )
    if not system_text:
        return
    event.system_prompt_hash = _sha256(system_text)
    if telemetry.embed_fn:
        try:
            event.system_prompt_embedding = telemetry.embed_fn(system_text)
        except Exception:
            pass  # best-effort; never block telemetry


def patch_anthropic(telemetry: "TelemetryClient"):
    """
    Patches anthropic.resources.messages.Messages.create so every
    messages.create call is automatically traced.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package is not installed. Run: pip install anthropic")

    from ai_telemetry.client import TelemetryEvent
    from ai_telemetry.transport import protocol_of_client

    _original_create = anthropic.resources.messages.Messages.create

    def _patched_create(self_anthropic, *args, **kwargs):
        start = time.time()
        error_info = {}
        response = None
        try:
            response = _original_create(self_anthropic, *args, **kwargs)
            return response
        except Exception as exc:
            error_info = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")

            event = TelemetryEvent(
                event_type="llm_call" if not error_info else "error",
                provider="anthropic",
                model=model,
                latency_ms=latency_ms,
                input_messages=list(messages),
                **error_info,
            )

            _capture_system_prompt_fields(event, kwargs, telemetry)
            event.call_protocol = protocol_of_client(self_anthropic)

            if response is not None:
                # Extract text and tool_use blocks
                text_blocks = [b for b in response.content if b.type == "text"]
                tool_blocks = [b for b in response.content if b.type == "tool_use"]

                event.output_message = {
                    "role": "assistant",
                    "content": text_blocks[0].text if text_blocks else None,
                    "tool_calls": [
                        {"id": tb.id, "name": tb.name, "input": tb.input}
                        for tb in tool_blocks
                    ] or None,
                }
                if response.usage:
                    event.prompt_tokens = response.usage.input_tokens
                    event.completion_tokens = response.usage.output_tokens
                    event.total_tokens = (
                        response.usage.input_tokens + response.usage.output_tokens
                    )

            telemetry.capture(event)

    anthropic.resources.messages.Messages.create = _patched_create

    # ------------------------------------------------------------------
    # Async patch — AsyncAnthropic / AsyncMessages.create
    # ------------------------------------------------------------------
    _original_async_create = anthropic.resources.messages.AsyncMessages.create

    async def _patched_async_create(self_anthropic, *args, **kwargs):
        start = time.time()
        error_info = {}
        response = None
        try:
            response = await _original_async_create(self_anthropic, *args, **kwargs)
            return response
        except Exception as exc:
            error_info = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")

            event = TelemetryEvent(
                event_type="llm_call" if not error_info else "error",
                provider="anthropic",
                model=model,
                latency_ms=latency_ms,
                input_messages=list(messages),
                **error_info,
            )

            _capture_system_prompt_fields(event, kwargs, telemetry)
            event.call_protocol = protocol_of_client(self_anthropic)

            if response is not None:
                text_blocks = [b for b in response.content if b.type == "text"]
                tool_blocks = [b for b in response.content if b.type == "tool_use"]

                event.output_message = {
                    "role": "assistant",
                    "content": text_blocks[0].text if text_blocks else None,
                    "tool_calls": [
                        {"id": tb.id, "name": tb.name, "input": tb.input}
                        for tb in tool_blocks
                    ] or None,
                }
                if response.usage:
                    event.prompt_tokens = response.usage.input_tokens
                    event.completion_tokens = response.usage.output_tokens
                    event.total_tokens = (
                        response.usage.input_tokens + response.usage.output_tokens
                    )

            telemetry.capture(event)

    anthropic.resources.messages.AsyncMessages.create = _patched_async_create
    return anthropic
