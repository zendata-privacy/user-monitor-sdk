"""
Tests for Anthropic monkey-patch (integrations/anthropic_integration.py).

Covers: sync and async patching, text/tool block extraction,
token usage (input_tokens/output_tokens), return value preservation,
and error capture.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest


def _make_text_block(text="Hello from Claude"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_block(id="tool_1", name="get_weather", input=None):
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input or {"location": "NYC"}
    return block


def _make_response(content=None, input_tokens=8, output_tokens=4):
    resp = MagicMock()
    resp.content = content if content is not None else [_make_text_block()]
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Injects a minimal mock anthropic module into sys.modules."""
    resp = _make_response()
    mod = MagicMock()
    mod.resources.messages.Messages.create.return_value = resp

    async def _async_original(self, *args, **kwargs):
        return resp

    mod.resources.messages.AsyncMessages.create = _async_original

    monkeypatch.setitem(sys.modules, "anthropic", mod)
    yield mod, resp
    sys.modules.pop("ai_telemetry.integrations.anthropic_integration", None)


class TestAnthropicPatchSync:
    def test_captures_llm_call_event(self, fake_anthropic, mock_telemetry):
        mod, _ = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        mod.resources.messages.Messages.create(
            MagicMock(), model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )

        e = mock_telemetry.captured[0]
        assert e.event_type == "llm_call"
        assert e.provider == "anthropic"
        assert e.model == "claude-3-5-sonnet-20241022"

    def test_extracts_token_usage(self, fake_anthropic, mock_telemetry):
        mod, _ = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        mod.resources.messages.Messages.create(
            MagicMock(), model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hi"}],
        )

        e = mock_telemetry.captured[0]
        assert e.prompt_tokens == 8
        assert e.completion_tokens == 4
        assert e.total_tokens == 12

    def test_extracts_text_block(self, fake_anthropic, mock_telemetry):
        mod, _ = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        mod.resources.messages.Messages.create(
            MagicMock(), model="claude-3-5-sonnet-20241022", messages=[],
        )

        e = mock_telemetry.captured[0]
        assert e.output_message["role"] == "assistant"
        assert e.output_message["content"] == "Hello from Claude"

    def test_extracts_tool_use_block(self, fake_anthropic, mock_telemetry):
        tool_resp = _make_response(
            content=[_make_tool_block(id="t1", name="search")]
        )
        mod, _ = fake_anthropic
        mod.resources.messages.Messages.create.return_value = tool_resp

        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        mod.resources.messages.Messages.create(
            MagicMock(), model="claude-3-5-sonnet-20241022", messages=[],
        )

        e = mock_telemetry.captured[0]
        tool_calls = e.output_message["tool_calls"]
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["id"] == "t1"

    def test_preserves_return_value(self, fake_anthropic, mock_telemetry):
        mod, resp = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        result = mod.resources.messages.Messages.create(
            MagicMock(), model="claude-3-5-sonnet-20241022", messages=[],
        )

        assert result is resp

    def test_captures_error_event(self, fake_anthropic, mock_telemetry):
        mod, _ = fake_anthropic
        mod.resources.messages.Messages.create.side_effect = ConnectionError("timeout")

        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        with pytest.raises(ConnectionError):
            mod.resources.messages.Messages.create(
                MagicMock(), model="claude-3-5-sonnet-20241022", messages=[],
            )

        e = mock_telemetry.captured[0]
        assert e.event_type == "error"
        assert e.error_type == "ConnectionError"


class TestAnthropicPatchAsync:
    def test_async_captures_llm_call_event(self, fake_anthropic, mock_telemetry):
        mod, _ = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        async def run():
            await mod.resources.messages.AsyncMessages.create(
                MagicMock(), model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
            )

        asyncio.run(run())
        assert mock_telemetry.captured[0].event_type == "llm_call"
        assert mock_telemetry.captured[0].provider == "anthropic"

    def test_async_preserves_return_value(self, fake_anthropic, mock_telemetry):
        mod, resp = fake_anthropic
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(mock_telemetry)

        async def run():
            return await mod.resources.messages.AsyncMessages.create(
                MagicMock(), model="claude-3-5-sonnet-20241022", messages=[],
            )

        assert asyncio.run(run()) is resp
