"""
Tests for the LangChain callback handler (integrations/langchain_integration.py).

Covers: LLM start/end, chat model start, LLM error, tool start/end,
tool error, stale entry cleanup, and token extraction.
"""

import sys
import time
import types
import uuid
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inject a minimal langchain_core stub so we can import the handler
# without the real package installed.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_langchain_core(monkeypatch):
    base_mod = types.ModuleType("langchain_core.callbacks.base")
    base_mod.BaseCallbackHandler = object  # simplest valid base

    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    callbacks_mod.base = base_mod

    lc_mod = types.ModuleType("langchain_core")
    lc_mod.callbacks = callbacks_mod

    monkeypatch.setitem(sys.modules, "langchain_core", lc_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.base", base_mod)
    # ensure fresh import each test
    sys.modules.pop("ai_telemetry.integrations.langchain_integration", None)
    yield


def _make_llm_response(text="output text", model_name="gpt-4o",
                       prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.llm_output = {
        "model_name": model_name,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    gen = MagicMock()
    gen.text = text
    response.generations = [[gen]]
    return response


class TestLLMHooks:
    def test_on_llm_end_captures_llm_call(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_end(_make_llm_response(), run_id=run_id)

        assert len(mock_telemetry.captured) == 1
        e = mock_telemetry.captured[0]
        assert e.event_type == "llm_call"
        assert e.provider == "langchain"
        assert e.model == "gpt-4o"

    def test_on_llm_end_extracts_token_usage(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_end(
            _make_llm_response(prompt_tokens=20, completion_tokens=8),
            run_id=run_id,
        )

        e = mock_telemetry.captured[0]
        assert e.prompt_tokens == 20
        assert e.completion_tokens == 8
        assert e.total_tokens == 28

    def test_on_llm_end_captures_output_text(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_end(_make_llm_response(text="The sky is blue"), run_id=run_id)

        e = mock_telemetry.captured[0]
        assert e.output_message["content"] == "The sky is blue"

    def test_on_chat_model_start_then_on_llm_end(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_chat_model_start({}, [[MagicMock()]], run_id=run_id)
        handler.on_llm_end(_make_llm_response(), run_id=run_id)

        assert mock_telemetry.captured[0].event_type == "llm_call"

    def test_on_llm_end_measures_latency(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        time.sleep(0.05)
        handler.on_llm_end(_make_llm_response(), run_id=run_id)

        assert mock_telemetry.captured[0].latency_ms >= 50

    def test_on_llm_error_captures_error_event(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_error(ValueError("context window exceeded"), run_id=run_id)

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.provider == "langchain"
        assert e.error_type == "ValueError"
        assert "context window" in e.error_message

    def test_on_llm_error_cleans_up_start_time(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)

        assert str(run_id) not in handler._start_times


class TestToolHooks:
    def test_on_tool_end_captures_tool_call(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_tool_start({}, "query string", run_id=run_id)
        handler.on_tool_end("tool output", run_id=run_id)

        e = mock_telemetry.captured[0]
        assert e.event_type == "tool_call"
        assert e.tool_output == "tool output"

    def test_on_tool_end_measures_latency(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_tool_start({}, "input", run_id=run_id)
        time.sleep(0.05)
        handler.on_tool_end("output", run_id=run_id)

        assert mock_telemetry.captured[0].latency_ms >= 50

    def test_on_tool_end_cleans_up_start_time(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_tool_start({}, "input", run_id=run_id)
        handler.on_tool_end("output", run_id=run_id)

        assert f"tool_{run_id}" not in handler._start_times

    def test_on_tool_error_captures_error_event(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_tool_start({}, "input", run_id=run_id)
        handler.on_tool_error(KeyError("missing key"), run_id=run_id)

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.error_type == "KeyError"

    def test_on_tool_error_cleans_up_start_time(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        run_id = uuid.uuid4()
        handler.on_tool_start({}, "input", run_id=run_id)
        handler.on_tool_error(RuntimeError("fail"), run_id=run_id)

        assert f"tool_{run_id}" not in handler._start_times


class TestStaleCleanup:
    def test_cleanup_removes_entries_older_than_300s(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        # Inject a stale entry
        old_key = "stale-run-id"
        handler._start_times[old_key] = time.time() - 400

        # Trigger cleanup via any hook
        handler.on_llm_start({}, ["prompt"], run_id=uuid.uuid4())

        assert old_key not in handler._start_times

    def test_cleanup_keeps_recent_entries(self, mock_telemetry):
        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        handler = create_langchain_handler(mock_telemetry)

        recent_key = "recent-run-id"
        handler._start_times[recent_key] = time.time() - 10

        handler.on_llm_start({}, ["prompt"], run_id=uuid.uuid4())

        assert recent_key in handler._start_times


class TestImportError:
    def test_raises_when_langchain_core_missing(self, mock_telemetry, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_langchain(name, *args, **kwargs):
            if name == "langchain_core" or name.startswith("langchain_core."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_langchain)
        sys.modules.pop("ai_telemetry.integrations.langchain_integration", None)

        from ai_telemetry.integrations.langchain_integration import create_langchain_handler
        with pytest.raises(ImportError, match="langchain-core"):
            create_langchain_handler(mock_telemetry)
