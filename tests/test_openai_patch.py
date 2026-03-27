"""
Tests for OpenAI monkey-patch (integrations/openai_integration.py).

Covers: sync and async patching, event field capture, token extraction,
return value preservation, and error capture.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_response(content="Hello world", prompt_tokens=10,
                   completion_tokens=5, tool_calls=None):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.role = "assistant"
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = tool_calls
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


@pytest.fixture
def fake_openai(monkeypatch):
    """Injects a minimal mock openai module into sys.modules."""
    resp = _make_response()
    mod = MagicMock()
    mod.resources.chat.completions.Completions.create.return_value = resp

    async def _async_original(self, *args, **kwargs):
        return resp

    mod.resources.chat.completions.AsyncCompletions.create = _async_original

    monkeypatch.setitem(sys.modules, "openai", mod)
    yield mod, resp
    # remove cached integration module so next import is fresh
    sys.modules.pop("ai_telemetry.integrations.openai_integration", None)


class TestOpenAIPatchSync:
    def test_captures_llm_call_event(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        mod.resources.chat.completions.Completions.create(
            MagicMock(), model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(mock_telemetry.captured) == 1
        e = mock_telemetry.captured[0]
        assert e.event_type == "llm_call"
        assert e.provider == "openai"
        assert e.model == "gpt-4o"

    def test_extracts_token_usage(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        mod.resources.chat.completions.Completions.create(
            MagicMock(), model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        e = mock_telemetry.captured[0]
        assert e.prompt_tokens == 10
        assert e.completion_tokens == 5
        assert e.total_tokens == 15

    def test_extracts_output_message(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        mod.resources.chat.completions.Completions.create(
            MagicMock(), model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        e = mock_telemetry.captured[0]
        assert e.output_message["role"] == "assistant"
        assert e.output_message["content"] == "Hello world"

    def test_preserves_return_value(self, fake_openai, mock_telemetry):
        mod, resp = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        result = mod.resources.chat.completions.Completions.create(
            MagicMock(), model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is resp

    def test_captures_error_event(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai

        # Override to raise
        mod.resources.chat.completions.Completions.create.side_effect = (
            RuntimeError("rate limit")
        )
        patch_openai(mock_telemetry)

        with pytest.raises(RuntimeError):
            mod.resources.chat.completions.Completions.create(
                MagicMock(), model="gpt-4o", messages=[],
            )

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.error_type == "RuntimeError"
        assert "rate limit" in e.error_message

    def test_captures_input_messages(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        msgs = [{"role": "user", "content": "tell me a joke"}]
        mod.resources.chat.completions.Completions.create(
            MagicMock(), model="gpt-4o", messages=msgs,
        )

        assert mock_telemetry.captured[0].input_messages == msgs


class TestOpenAIPatchAsync:
    def test_async_captures_llm_call_event(self, fake_openai, mock_telemetry):
        mod, _ = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        async def run():
            await mod.resources.chat.completions.AsyncCompletions.create(
                MagicMock(), model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        asyncio.run(run())
        assert mock_telemetry.captured[0].event_type == "llm_call"
        assert mock_telemetry.captured[0].provider == "openai"

    def test_async_preserves_return_value(self, fake_openai, mock_telemetry):
        mod, resp = fake_openai
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(mock_telemetry)

        async def run():
            return await mod.resources.chat.completions.AsyncCompletions.create(
                MagicMock(), model="gpt-4o", messages=[],
            )

        result = asyncio.run(run())
        assert result is resp
