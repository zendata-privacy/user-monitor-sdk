"""
AI Telemetry SDK - OpenAI Integration
Monkey-patches the OpenAI client to automatically capture telemetry.

Usage:
    from ai_telemetry.integrations.openai import patch_openai
    patch_openai(telemetry_client)
"""

import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryClient


def _capture_system_prompt_fields(event, messages, telemetry) -> None:
    """Populate system_prompt_hash and system_prompt_embedding on *event*.

    Iterates the OpenAI-style messages list looking for a role="system" entry.
    Safe to call even when messages is empty or None.
    """
    from ai_telemetry.client import _sha256
    system_text = next(
        (m.get("content", "") for m in (messages or []) if m.get("role") == "system"),
        None,
    )
    if not system_text:
        return
    event.system_prompt_hash = _sha256(system_text)
    if telemetry.embed_fn:
        try:
            event.system_prompt_embedding = telemetry.embed_fn(system_text)
        except Exception:
            pass  # embedding is best-effort; never block telemetry


def patch_openai(telemetry: "TelemetryClient"):
    """
    Patches openai.OpenAI and openai.AsyncOpenAI so every
    chat.completions.create call is automatically traced.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package is not installed. Run: pip install openai")

    from ai_telemetry.client import TelemetryEvent
    from ai_telemetry.transport import protocol_of_client

    _original_create = openai.resources.chat.completions.Completions.create

    def _patched_create(self_openai, *args, **kwargs):
        start = time.time()
        error_info = {}
        response = None
        try:
            response = _original_create(self_openai, *args, **kwargs)
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
                provider="openai",
                model=model,
                latency_ms=latency_ms,
                input_messages=list(messages),
                **error_info,
            )

            _capture_system_prompt_fields(event, messages, telemetry)
            event.call_protocol = protocol_of_client(self_openai)

            if response is not None:
                choice = response.choices[0] if response.choices else None
                if choice:
                    event.output_message = {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (choice.message.tool_calls or [])
                        ] or None,
                    }
                if response.usage:
                    event.prompt_tokens = response.usage.prompt_tokens
                    event.completion_tokens = response.usage.completion_tokens
                    event.total_tokens = response.usage.total_tokens

            telemetry.capture(event)

    openai.resources.chat.completions.Completions.create = _patched_create

    # ------------------------------------------------------------------
    # Async patch — AsyncOpenAI / AsyncCompletions.create
    # ------------------------------------------------------------------
    _original_async_create = openai.resources.chat.completions.AsyncCompletions.create

    async def _patched_async_create(self_openai, *args, **kwargs):
        start = time.time()
        error_info = {}
        response = None
        try:
            response = await _original_async_create(self_openai, *args, **kwargs)
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
                provider="openai",
                model=model,
                latency_ms=latency_ms,
                input_messages=list(messages),
                **error_info,
            )

            _capture_system_prompt_fields(event, messages, telemetry)
            event.call_protocol = protocol_of_client(self_openai)

            if response is not None:
                choice = response.choices[0] if response.choices else None
                if choice:
                    event.output_message = {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (choice.message.tool_calls or [])
                        ] or None,
                    }
                if response.usage:
                    event.prompt_tokens = response.usage.prompt_tokens
                    event.completion_tokens = response.usage.completion_tokens
                    event.total_tokens = response.usage.total_tokens

            telemetry.capture(event)

    openai.resources.chat.completions.AsyncCompletions.create = _patched_async_create
    return openai
