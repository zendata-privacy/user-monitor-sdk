"""
AI Telemetry SDK - LangChain Integration
A LangChain callback handler that captures telemetry for every LLM and tool call.

Usage:
    from ai_telemetry.integrations.langchain_integration import TelemetryCallbackHandler
    handler = TelemetryCallbackHandler(telemetry_client)
    chain.invoke(input, config={"callbacks": [handler]})
"""

import time
import traceback
from typing import Any, Dict, List, Union, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from ai_telemetry.client import TelemetryClient


def create_langchain_handler(telemetry: "TelemetryClient"):
    """
    Factory function that lazily imports langchain-core and returns a
    TelemetryCallbackHandler instance. Raises ImportError only when called,
    not at module import time.
    """
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain-core is not installed. Run: pip install langchain-core"
        )

    _MAX_START_TIME_AGE = 300  # seconds

    class TelemetryCallbackHandler(BaseCallbackHandler):
        """
        LangChain callback handler that forwards LLM and tool events to your
        AI Telemetry platform.
        """

        def __init__(self):
            super().__init__()
            self.telemetry = telemetry
            self._start_times: Dict[str, float] = {}

        def _cleanup_start_times(self) -> None:
            now = time.time()
            stale = [k for k, v in self._start_times.items() if now - v > _MAX_START_TIME_AGE]
            for k in stale:
                del self._start_times[k]

        # ------------------------------------------------------------------
        # LLM hooks
        # ------------------------------------------------------------------

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            **kwargs,
        ):
            self._cleanup_start_times()
            self._start_times[str(run_id)] = time.time()

        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List],
            *,
            run_id: UUID,
            **kwargs,
        ):
            self._cleanup_start_times()
            self._start_times[str(run_id)] = time.time()

        def on_llm_end(
            self,
            response,
            *,
            run_id: UUID,
            **kwargs,
        ):
            from ai_telemetry.client import TelemetryEvent
            from ai_telemetry.context import current_user_context

            start = self._start_times.pop(str(run_id), time.time())
            latency_ms = int((time.time() - start) * 1000)

            llm_output = response.llm_output or {}

            # Flatten generations
            all_generations = [g for row in response.generations for g in row]
            output_text = all_generations[0].text if all_generations else None

            # Extract token usage — try multiple locations across LangChain versions:
            # 1. llm_output["token_usage"]  (older langchain-openai style)
            # 2. llm_output["usage"]         (some providers)
            # 3. generation.message.usage_metadata (langchain-openai >=0.2 / core >=0.2)
            # 4. generation.message.response_metadata["token_usage"]
            prompt_tokens = completion_tokens = total_tokens = None

            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")

            if total_tokens is None and all_generations:
                gen = all_generations[0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    # usage_metadata: {"input_tokens": N, "output_tokens": N, "total_tokens": N}
                    um = getattr(msg, "usage_metadata", None) or {}
                    if um:
                        prompt_tokens = um.get("input_tokens")
                        completion_tokens = um.get("output_tokens")
                        total_tokens = um.get("total_tokens")

                    if total_tokens is None:
                        # response_metadata may hold token_usage dict
                        rm = getattr(msg, "response_metadata", None) or {}
                        tu = rm.get("token_usage") or rm.get("usage") or {}
                        if tu:
                            prompt_tokens = tu.get("prompt_tokens") or tu.get("input_tokens")
                            completion_tokens = tu.get("completion_tokens") or tu.get("output_tokens")
                            total_tokens = tu.get("total_tokens")

            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            # Extract model name — try llm_output first, then generation response_metadata
            model_name = llm_output.get("model_name") or llm_output.get("model")
            if not model_name and all_generations:
                gen = all_generations[0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    rm = getattr(msg, "response_metadata", None) or {}
                    model_name = rm.get("model_name") or rm.get("model")
            model_name = model_name or "unknown"

            # Pull user context so entity classification and user fields are populated
            user_ctx = current_user_context.get()

            event = TelemetryEvent(
                event_type="llm_call",
                # ZD-W3-07 · diligence. This was hardcoded "openai", which
                # mislabels every non-OpenAI model reached through LangChain.
                # The error hook 20 lines below already reports "langchain",
                # and the TypeScript implementation of this same SDK reports
                # "langchain" from both hooks. This branch was the outlier.
                provider="langchain",
                model=model_name,
                latency_ms=latency_ms,
                output_message={"role": "assistant", "content": output_text},
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                # User context — required for entity classification
                user_id=user_ctx.get("user_id"),
                email=user_ctx.get("email"),
                display_name=user_ctx.get("display_name"),
                department=user_ctx.get("department"),
                job_title=user_ctx.get("job_title"),
                sailpoint_risk=user_ctx.get("sailpoint_risk"),
                data_sources=user_ctx.get("data_sources"),
                tools_called=user_ctx.get("tools_called"),
                prompt_submission_time_ms=user_ctx.get("prompt_submission_time_ms"),
            )
            self.telemetry.capture(event)

        def on_llm_error(
            self,
            error: Union[Exception, KeyboardInterrupt],
            *,
            run_id: UUID,
            **kwargs,
        ):
            from ai_telemetry.client import TelemetryEvent

            self._start_times.pop(str(run_id), None)
            event = TelemetryEvent(
                event_type="error",
                provider="langchain",
                error_type=type(error).__name__,
                error_message=str(error),
                error_traceback=traceback.format_exc(),
            )
            self.telemetry.capture(event)

        # ------------------------------------------------------------------
        # Tool hooks
        # ------------------------------------------------------------------

        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs,
        ):
            self._cleanup_start_times()
            self._start_times[f"tool_{run_id}"] = time.time()

        def on_tool_end(
            self,
            output: str,
            *,
            run_id: UUID,
            **kwargs,
        ):
            from ai_telemetry.client import TelemetryEvent

            start = self._start_times.pop(f"tool_{run_id}", time.time())
            latency_ms = int((time.time() - start) * 1000)

            event = TelemetryEvent(
                event_type="tool_call",
                latency_ms=latency_ms,
                tool_output=output,
            )
            self.telemetry.capture(event)

        def on_tool_error(
            self,
            error: Union[Exception, KeyboardInterrupt],
            *,
            run_id: UUID,
            **kwargs,
        ):
            from ai_telemetry.client import TelemetryEvent

            self._start_times.pop(f"tool_{run_id}", None)
            event = TelemetryEvent(
                event_type="error",
                error_type=type(error).__name__,
                error_message=str(error),
                error_traceback=traceback.format_exc(),
            )
            self.telemetry.capture(event)

    return TelemetryCallbackHandler()


# Convenience alias so existing code using TelemetryCallbackHandler(telemetry)
# continues to work without changes.
TelemetryCallbackHandler = create_langchain_handler
