"""
AI Telemetry SDK - Decorators
Generic decorators for manual instrumentation of any AI function.

Usage:
    from ai_telemetry.decorators import trace_llm, trace_tool
"""

import inspect
import time
import functools
import traceback
from typing import Optional, TYPE_CHECKING
from ai_telemetry.context import current_user_context

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryClient


def _safe_serialize(obj, max_depth: int = 2):
    """Recursively convert obj to a JSON-safe structure, falling back to str()."""
    if max_depth <= 0:
        return str(obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, max_depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, max_depth - 1) for v in obj]
    return str(obj)


def trace_llm(
    telemetry: "TelemetryClient",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    call_protocol: Optional[str] = None,
):
    """
    Decorator: wraps any function that calls an LLM and captures telemetry.

    @trace_llm(client, model="gpt-4o", provider="openai")
    def my_chat_function(prompt):
        ...
    """
    def _capture_llm_event(fn, start, result, error_info):
        from ai_telemetry.client import TelemetryEvent
        latency_ms = int((time.time() - start) * 1000)
        user_ctx = current_user_context.get()
        event = TelemetryEvent(
            event_type="llm_call" if not error_info else "error",
            model=model or "unknown",
            provider=provider or "generic",
            call_protocol=call_protocol,
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

    def decorator(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                error_info = {}
                result = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error_info = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "error_traceback": traceback.format_exc(),
                    }
                    raise
                finally:
                    _capture_llm_event(fn, start, result, error_info)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
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
                    _capture_llm_event(fn, start, result, error_info)
            return sync_wrapper
    return decorator


def trace_tool(
    telemetry: "TelemetryClient",
    tool_name: Optional[str] = None,
    source_uri: Optional[str] = None,
    classification: Optional[str] = None,
    call_protocol: Optional[str] = None,
):
    """
    Decorator: wraps any tool/function call and captures telemetry.

    @trace_tool(client, tool_name="web_search")
    def web_search(query: str) -> str:
        ...

    ``source_uri`` declares where this tool reads from, which is what the scope
    boundary check needs and what the SDK cannot work out for itself — all it
    sees here is a function name, its arguments and its return value::

        @trace_tool(client, tool_name="lookup_account",
                    source_uri="arn:aws:dynamodb:us-east-1:...:table/CustomerAccounts")
        def lookup_account(customer_id: str) -> dict:
            ...

    Use it for tools that always read the same place. For sources only known at
    runtime use ``source_context()``; for boto3 calls, the interceptor reports
    the real ARN with no annotation at all.

    ``classification`` optionally pins what this tool returns (PII, PHI,
    CREDENTIALS, INTERNAL, PUBLIC) when the operator knows better than content
    inspection does — a table of pseudonymous ids that content scanning would
    read as clean, for instance.

    ``call_protocol`` declares how the tool reaches its source (https, http,
    grpc, …). When omitted it is taken from the scheme of ``source_uri``
    (``https://…`` → https, ``postgres://…`` → postgres); an ARN or a plain
    function leaves it empty.
    """
    from ai_telemetry.transport import protocol_from_url
    tool_protocol = call_protocol or protocol_from_url(source_uri)
    def _capture_tool_event(name, start, args, kwargs, result, error_info):
        from ai_telemetry.client import TelemetryEvent
        latency_ms = int((time.time() - start) * 1000)
        user_ctx = current_user_context.get()

        # Append tool to current context so that llm_call can see what tools were called
        updated_ctx = {**user_ctx}
        tools_called = updated_ctx.get("tools_called", "")
        if tools_called:
            if name not in tools_called.split("|"):
                updated_ctx["tools_called"] = tools_called + "|" + name
        else:
            updated_ctx["tools_called"] = name
        current_user_context.set(updated_ctx)

        event = TelemetryEvent(
            event_type="tool_call" if not error_info else "error",
            tool_name=name,
            call_protocol=tool_protocol,
            tool_input=_safe_serialize({"args": args, "kwargs": kwargs}),
            tool_output=_safe_serialize(result),
            latency_ms=latency_ms,
            user_id=updated_ctx.get("user_id"),
            email=updated_ctx.get("email"),
            display_name=updated_ctx.get("display_name"),
            department=updated_ctx.get("department"),
            job_title=updated_ctx.get("job_title"),
            sailpoint_risk=updated_ctx.get("sailpoint_risk"),
            # The accumulated list was being written back to the CONTEXT so a
            # later llm_call could see it, but never onto the tool_call event
            # itself — so every tool_call shipped with tools_called empty, and
            # in an agent loop the llm_call that decides which tool to use runs
            # BEFORE any tool, so it saw an empty list too. Net effect: the
            # column existed and was never populated by anything.
            tools_called=updated_ctx.get("tools_called"),
            data_sources=updated_ctx.get("data_sources"),
            **error_info,
        )
        # Declared source identity — producer 1 of the four described in
        # ai_telemetry/scope_runtime.py. Set on the event rather than resolved
        # here: hashing and classifying the content is background-thread work,
        # and this decorator runs on the caller's.
        if source_uri:
            event.source_uri = source_uri
        if classification:
            event.source_classification = classification
        telemetry.capture(event)

    def decorator(fn):
        name = tool_name or fn.__name__
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                error_info = {}
                result = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error_info = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "error_traceback": traceback.format_exc(),
                    }
                    raise
                finally:
                    _capture_tool_event(name, start, args, kwargs, result, error_info)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
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
                    _capture_tool_event(name, start, args, kwargs, result, error_info)
            return sync_wrapper
    return decorator
