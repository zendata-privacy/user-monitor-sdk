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

            start = self._start_times.pop(str(run_id), time.time())
            latency_ms = int((time.time() - start) * 1000)

            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage", {})

            # Flatten generations
            all_generations = [g for row in response.generations for g in row]
            output_text = all_generations[0].text if all_generations else None

            event = TelemetryEvent(
                event_type="llm_call",
                provider="langchain",
                model=llm_output.get("model_name", "unknown"),
                latency_ms=latency_ms,
                output_message={"role": "assistant", "content": output_text},
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
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
