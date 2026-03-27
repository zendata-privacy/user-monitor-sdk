"""
Tests for trace_llm and trace_tool decorators (decorators.py).

Covers: sync/async wrapping, event fields, error capture,
latency, user context propagation, tools_called accumulation.
"""

import asyncio
import time

import pytest

from ai_telemetry.context import current_user_context
from ai_telemetry.decorators import trace_llm, trace_tool


# ---------------------------------------------------------------------------
# trace_llm
# ---------------------------------------------------------------------------

class TestTraceLlm:
    def test_sync_returns_original_value(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def chat():
            return "hello"

        assert chat() == "hello"

    def test_sync_captures_llm_call_event(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def chat():
            return "ok"

        chat()
        assert len(mock_telemetry.captured) == 1
        e = mock_telemetry.captured[0]
        assert e.event_type == "llm_call"
        assert e.model == "gpt-4o"
        assert e.provider == "openai"

    def test_sync_captures_output_message(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def chat():
            return "the answer"

        chat()
        assert mock_telemetry.captured[0].output_message == {
            "role": "assistant",
            "content": "the answer",
        }

    def test_sync_captures_latency(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def slow():
            time.sleep(0.05)
            return "ok"

        slow()
        assert mock_telemetry.captured[0].latency_ms >= 50

    def test_sync_captures_error(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.error_type == "ValueError"
        assert e.error_message == "boom"
        assert e.error_traceback is not None

    def test_sync_re_raises_exception(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def bad():
            raise RuntimeError("must propagate")

        with pytest.raises(RuntimeError, match="must propagate"):
            bad()

    def test_sync_reads_user_context(self, mock_telemetry):
        current_user_context.set({
            "user_id": "u-1",
            "email": "test@example.com",
            "department": "eng",
        })

        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        def chat():
            return "ok"

        chat()
        e = mock_telemetry.captured[0]
        assert e.user_id == "u-1"
        assert e.email == "test@example.com"
        assert e.department == "eng"

    def test_async_returns_original_value(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        async def chat():
            return "async hello"

        assert asyncio.run(chat()) == "async hello"

    def test_async_captures_llm_call_event(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="claude-3", provider="anthropic")
        async def chat():
            return "ok"

        asyncio.run(chat())
        e = mock_telemetry.captured[0]
        assert e.event_type == "llm_call"
        assert e.model == "claude-3"
        assert e.provider == "anthropic"

    def test_async_captures_error(self, mock_telemetry):
        @trace_llm(mock_telemetry, model="gpt-4o", provider="openai")
        async def bad():
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError):
            asyncio.run(bad())

        assert mock_telemetry.captured[0].event_type == "error"
        assert mock_telemetry.captured[0].error_type == "RuntimeError"

    def test_defaults_used_when_model_and_provider_omitted(self, mock_telemetry):
        @trace_llm(mock_telemetry)
        def chat():
            return "ok"

        chat()
        e = mock_telemetry.captured[0]
        assert e.model == "unknown"
        assert e.provider == "generic"


# ---------------------------------------------------------------------------
# trace_tool
# ---------------------------------------------------------------------------

class TestTraceTool:
    def test_sync_returns_original_value(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="search")
        def search(q):
            return "results"

        assert search("python") == "results"

    def test_sync_captures_tool_call_event(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="web_search")
        def search(q):
            return "r"

        search("q")
        e = mock_telemetry.captured[0]
        assert e.event_type == "tool_call"
        assert e.tool_name == "web_search"

    def test_uses_function_name_as_default_tool_name(self, mock_telemetry):
        @trace_tool(mock_telemetry)
        def my_special_tool():
            return "ok"

        my_special_tool()
        assert mock_telemetry.captured[0].tool_name == "my_special_tool"

    def test_appends_to_tools_called_in_context(self, mock_telemetry):
        current_user_context.set({})

        @trace_tool(mock_telemetry, tool_name="tool_a")
        def tool_a():
            return "a"

        @trace_tool(mock_telemetry, tool_name="tool_b")
        def tool_b():
            return "b"

        tool_a()
        tool_b()
        assert current_user_context.get()["tools_called"] == "tool_a|tool_b"

    def test_deduplicates_tools_called(self, mock_telemetry):
        current_user_context.set({})

        @trace_tool(mock_telemetry, tool_name="search")
        def search():
            return "r"

        search()
        search()
        assert current_user_context.get()["tools_called"] == "search"

    def test_sets_initial_tools_called_when_context_empty(self, mock_telemetry):
        current_user_context.set({})

        @trace_tool(mock_telemetry, tool_name="first_tool")
        def first():
            return "ok"

        first()
        assert current_user_context.get()["tools_called"] == "first_tool"

    def test_sync_captures_latency(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="slow_tool")
        def slow():
            time.sleep(0.05)
            return "r"

        slow()
        assert mock_telemetry.captured[0].latency_ms >= 50

    def test_sync_captures_error(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="bad_tool")
        def bad():
            raise KeyError("missing")

        with pytest.raises(KeyError):
            bad()

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.error_type == "KeyError"

    def test_sync_re_raises_exception(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="t")
        def bad():
            raise ValueError("must propagate")

        with pytest.raises(ValueError, match="must propagate"):
            bad()

    def test_async_returns_original_value(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="async_tool")
        async def atool():
            return "async result"

        assert asyncio.run(atool()) == "async result"

    def test_async_captures_tool_call_event(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="async_tool")
        async def atool():
            return "ok"

        asyncio.run(atool())
        assert mock_telemetry.captured[0].event_type == "tool_call"
        assert mock_telemetry.captured[0].tool_name == "async_tool"

    def test_async_captures_error(self, mock_telemetry):
        @trace_tool(mock_telemetry, tool_name="async_bad")
        async def bad():
            raise TypeError("async error")

        with pytest.raises(TypeError):
            asyncio.run(bad())

        assert mock_telemetry.captured[0].event_type == "error"
        assert mock_telemetry.captured[0].error_type == "TypeError"

    def test_context_not_mutated_in_place(self, mock_telemetry):
        """tools_called update must use a new dict (context isolation fix)."""
        original_ctx = {"user_id": "u-1"}
        current_user_context.set(original_ctx)

        @trace_tool(mock_telemetry, tool_name="tool_x")
        def tool_x():
            return "ok"

        tool_x()
        # original_ctx must not have been mutated
        assert "tools_called" not in original_ctx
        # context var must have been updated
        assert current_user_context.get()["tools_called"] == "tool_x"
