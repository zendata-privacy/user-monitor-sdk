import contextlib
import threading
import uuid
import contextvars
from typing import Any, Dict, Iterator, Optional

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


# ---------------------------------------------------------------------------
# Per-request accumulator
# ---------------------------------------------------------------------------
#
# trace_tool used to record "this tool ran" by SETTING current_user_context
# again with tools_called extended. That only works when the tool runs in the
# same context as the LLM call that follows it. Agent frameworks run tools on
# worker threads or in copied contexts (LangChain: run_in_executor), and a
# ContextVar set inside a copy is invisible to the parent — so the llm_call that
# decides the next step never saw which tools had run, and tool scope stayed
# empty for every LangChain agent.
#
# The fix is one MUTABLE object per request, placed in the context once. A copy
# of the context holds a reference to the same object, so a tool appending to
# it on any thread is seen by every later event of the same request.

_ACC_KEY = "_request_acc"


def start_request(**fields: Any) -> Dict[str, Any]:
    """Begin one request: set the caller context with a fresh accumulator.

    Call once per inbound request (the user message, or the machine task) with
    the caller's identity fields. Returns the context dict that was set.
    """
    ctx = dict(fields)
    ctx[_ACC_KEY] = {"tools": [], "sources": [], "lock": threading.Lock()}
    current_user_context.set(ctx)
    return ctx


def _acc() -> Optional[Dict[str, Any]]:
    return (current_user_context.get() or {}).get(_ACC_KEY)


def record_tool(name: str) -> None:
    acc = _acc()
    if acc is None or not name:
        return
    with acc["lock"]:
        if name not in acc["tools"]:
            acc["tools"].append(name)


def record_sources(uris) -> None:
    acc = _acc()
    if acc is None:
        return
    with acc["lock"]:
        for uri in uris or ():
            if uri and uri not in acc["sources"]:
                acc["sources"].append(uri)


def request_tools() -> Optional[str]:
    """Pipe-joined tools run so far in this request, or None without an accumulator."""
    acc = _acc()
    if acc is None:
        return None
    with acc["lock"]:
        return "|".join(acc["tools"]) or None


def request_sources() -> Optional[str]:
    acc = _acc()
    if acc is None:
        return None
    with acc["lock"]:
        return "|".join(acc["sources"]) or None


@contextlib.contextmanager
def source_context(source_uri: str, **extra: Any) -> Iterator[None]:
    """Declare which source the retrievals inside this block came from.

    The SDK cannot infer a source identity. ``trace_tool`` sees a function name,
    its arguments and its return value; none of that says the result came from a
    particular DynamoDB table. Three producers supply it, and this is the one
    for sources only known at runtime — a vector index chunk id, a database row
    key, an object path assembled from user input::

        for hit in retriever.search(query):
            with source_context(f"vector-index://kb/chunk_id:{hit.id}"):
                context.append(fetch(hit.id))

    For a tool that always reads the same place, declaring it once on the
    decorator is simpler::

        @trace_tool(client, source_uri="arn:aws:dynamodb:...:table/CustomerAccounts")

    Neither is needed where the boto3 interceptor is active: it reports the real
    ARN from an ``after-call`` hook with no annotation at all.

    Restores the previous context on exit, including on exception, so a failed
    retrieval cannot leave a stale source attributed to whatever runs next.
    """
    ctx = current_user_context.get() or {}
    merged = {**ctx, "source_uri": source_uri}
    merged.update(extra)
    token = current_user_context.set(merged)
    try:
        yield
    finally:
        current_user_context.reset(token)
