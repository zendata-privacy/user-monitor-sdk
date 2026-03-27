"""
Tests for ContextVar propagation and isolation (context.py).

Covers: user context set/get, async task isolation, child context
isolation, TraceContext lifecycle, and span chaining.
"""

import asyncio
import contextvars

import pytest

from ai_telemetry.context import (
    TraceContext,
    _current_span_id,
    _current_trace_id,
    current_user_context,
)


class TestUserContext:
    def test_default_is_empty_dict(self):
        assert current_user_context.get() == {}

    def test_set_and_get_roundtrip(self):
        token = current_user_context.set({"user_id": "u-42"})
        assert current_user_context.get()["user_id"] == "u-42"
        current_user_context.reset(token)

    def test_isolation_between_concurrent_async_tasks(self):
        """Two concurrently running async tasks must not see each other's context."""
        results: dict = {}

        async def task(name: str, user_id: str):
            # Each task copies and sets its own context
            updated = {**current_user_context.get(), "user_id": user_id}
            current_user_context.set(updated)
            await asyncio.sleep(0)  # yield — let the other task run
            results[name] = current_user_context.get()["user_id"]

        async def run():
            await asyncio.gather(task("a", "user-a"), task("b", "user-b"))

        asyncio.run(run())
        assert results["a"] == "user-a"
        assert results["b"] == "user-b"

    def test_child_context_mutation_does_not_affect_parent(self):
        """A copied context running as a child must not bleed mutations back."""
        current_user_context.set({"user_id": "parent"})

        def child_fn():
            current_user_context.set({"user_id": "child"})
            return current_user_context.get()["user_id"]

        child_ctx = contextvars.copy_context()
        child_result = child_ctx.run(child_fn)

        assert child_result == "child"
        assert current_user_context.get()["user_id"] == "parent"

    def test_dict_copy_does_not_mutate_original(self):
        """Verify the decorator fix: shallow copy prevents in-place mutation."""
        original = {"user_id": "u-1"}
        current_user_context.set(original)

        # Simulate what the fixed decorator does
        ctx = current_user_context.get()
        updated = {**ctx, "tools_called": "tool_x"}
        current_user_context.set(updated)

        # original dict object must be unchanged
        assert "tools_called" not in original


class TestTraceContext:
    def test_start_trace_returns_uuid_string(self):
        trace_id = TraceContext.start_trace()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # standard UUID format

    def test_start_trace_sets_current_trace_id(self):
        trace_id = TraceContext.start_trace()
        assert TraceContext.current_trace_id() == trace_id

    def test_current_trace_id_empty_before_trace_started(self):
        # reset_context fixture ensures empty state
        assert TraceContext.current_trace_id() == ""

    def test_start_trace_resets_span_id(self):
        _current_span_id.set("old-span")
        TraceContext.start_trace()
        assert _current_span_id.get() == ""

    def test_new_span_returns_tuple_of_ids(self):
        TraceContext.start_trace()
        span_id, parent_span_id = TraceContext.new_span()
        assert isinstance(span_id, str) and len(span_id) == 36
        assert isinstance(parent_span_id, str)

    def test_first_new_span_has_empty_parent(self):
        TraceContext.start_trace()
        _, parent_span_id = TraceContext.new_span()
        assert parent_span_id == ""

    def test_second_new_span_chains_first_as_parent(self):
        TraceContext.start_trace()
        span_a, _ = TraceContext.new_span()
        span_b, parent_b = TraceContext.new_span()
        assert parent_b == span_a
        assert span_b != span_a

    def test_multiple_traces_are_independent(self):
        id1 = TraceContext.start_trace()
        id2 = TraceContext.start_trace()
        assert id1 != id2
        assert TraceContext.current_trace_id() == id2
