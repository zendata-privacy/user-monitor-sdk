"""
AI Telemetry SDK - Decorators
Generic decorators for manual instrumentation of any AI function.

Usage:
    from ai_telemetry.decorators import trace_llm, trace_tool
"""

import time
import functools
import traceback
from typing import Optional, TYPE_CHECKING
from ai_telemetry.context import current_user_context

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryClient


def trace_llm(
    telemetry: "TelemetryClient",
    model: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Decorator: wraps any function that calls an LLM and captures telemetry.

    @trace_llm(client, model="gpt-4o", provider="openai")
    def my_chat_function(prompt):
        ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from ai_telemetry.client import TelemetryEvent
            start = time.time()
            error_info = {}
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                error_info = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(),
                }
                raise
            finally:
                latency_ms = int((time.time() - start) * 1000)
                user_ctx = current_user_context.get()
                
                event = TelemetryEvent(
                    event_type="llm_call" if not error_info else "error",
                    model=model or "unknown",
                    provider=provider or "generic",
                    latency_ms=latency_ms,
                    metadata={"function": fn.__name__},
                    user_id=user_ctx.get("user_id"),
                    email=user_ctx.get("email"),
                    display_name=user_ctx.get("display_name"),
                    department=user_ctx.get("department"),
                    job_title=user_ctx.get("job_title"),
                    sailpoint_risk=user_ctx.get("sailpoint_risk"),
                    data_sources=user_ctx.get("data_sources"),
                    tools_called=user_ctx.get("tools_called"),
                    **error_info,
                )
                if result is not None:
                    event.output_message = {
                        "role": "assistant",
                        "content": str(result),
                    }
                telemetry.capture(event)
        return wrapper
    return decorator


def trace_tool(telemetry: "TelemetryClient", tool_name: Optional[str] = None):
    """
    Decorator: wraps any tool/function call and captures telemetry.

    @trace_tool(client, tool_name="web_search")
    def web_search(query: str) -> str:
        ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from ai_telemetry.client import TelemetryEvent
            start = time.time()
            error_info = {}
            result = None
            name = tool_name or fn.__name__
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                error_info = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(),
                }
                raise
            finally:
                latency_ms = int((time.time() - start) * 1000)
                user_ctx = current_user_context.get()
                
                # Append tool to current context so that llm_call can see what tools were called
                tools_called = user_ctx.get("tools_called", "")
                if tools_called:
                    if name not in tools_called.split("|"):
                        user_ctx["tools_called"] = tools_called + "|" + name
                else:
                    user_ctx["tools_called"] = name
                
                event = TelemetryEvent(
                    event_type="tool_call" if not error_info else "error",
                    tool_name=name,
                    tool_input={"args": args, "kwargs": kwargs},
                    tool_output=result,
                    latency_ms=latency_ms,
                    user_id=user_ctx.get("user_id"),
                    email=user_ctx.get("email"),
                    display_name=user_ctx.get("display_name"),
                    department=user_ctx.get("department"),
                    job_title=user_ctx.get("job_title"),
                    sailpoint_risk=user_ctx.get("sailpoint_risk"),
                    **error_info,
                )
                telemetry.capture(event)
        return wrapper
    return decorator
