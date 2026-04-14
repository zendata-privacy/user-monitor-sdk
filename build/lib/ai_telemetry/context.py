import uuid
import contextvars
from typing import Dict, Any

# User identity context — populated by the host application
current_user_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "current_user_context", default={}
)

# Trace context — propagated automatically through async/threaded execution
_current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_current_span_id: contextvars.ContextVar[str] = contextvars.ContextVar("span_id", default="")


class TraceContext:
    """
    Manage distributed trace context for correlating events across an interaction.

    Usage::

        from ai_telemetry.context import TraceContext

        trace_id = TraceContext.start_trace()   # once per user interaction
        # ... all captures within this context share trace_id automatically

        span_id, parent = TraceContext.new_span()  # once per logical operation
        # ... captures after new_span() report span_id as their parent
    """

    @staticmethod
    def start_trace() -> str:
        """Start a new trace. Call once at the start of each user interaction.

        Returns the new trace_id.
        """
        trace_id = str(uuid.uuid4())
        _current_trace_id.set(trace_id)
        _current_span_id.set("")
        return trace_id

    @staticmethod
    def current_trace_id() -> str:
        """Return the active trace ID, or empty string if no trace is active."""
        return _current_trace_id.get()

    @staticmethod
    def new_span() -> tuple[str, str]:
        """Advance the span context. Returns (span_id, parent_span_id).

        Call at the start of each logical operation (tool call, LLM call, etc.)
        to establish this operation as the parent for nested captures.
        """
        parent_span_id = _current_span_id.get()
        span_id = str(uuid.uuid4())
        _current_span_id.set(span_id)
        return span_id, parent_span_id
