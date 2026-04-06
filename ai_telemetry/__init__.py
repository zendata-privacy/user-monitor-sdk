"""
AI Telemetry SDK for Python
"""

from importlib.metadata import version as _metadata_version, PackageNotFoundError
from typing import Callable, List, Optional

try:
    __version__ = _metadata_version("ai-telemetry-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0"

from ai_telemetry.client import TelemetryClient, TelemetryEvent
from ai_telemetry.decorators import trace_llm, trace_tool
from ai_telemetry.context import current_user_context, TraceContext
from ai_telemetry.agent_fingerprint import get_fingerprint_store, AgentRecord

_default_client: TelemetryClient | None = None


def init(
    app_id: str,
    api_key: str,
    endpoint: str = "https://ingest.zendata.dev/v1/events",
    capture_prompts: bool = True,
    auto_patch_openai: bool = False,
    auto_patch_anthropic: bool = False,
    debug: bool = False,
    enable_local_log: bool = False,
    log_path: Optional[str] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> TelemetryClient:
    """
    Initialize the AI Telemetry SDK.

    Args:
        app_id:              Your registered application ID.
        api_key:             Your platform API key.
        endpoint:            Ingest endpoint URL.
        capture_prompts:     Set False to redact prompt/response content.
        auto_patch_openai:   Automatically instrument the OpenAI SDK.
        auto_patch_anthropic: Automatically instrument the Anthropic SDK.
        debug:               Print debug logs to stdout.
        embed_fn:            Optional callable ``(text: str) -> List[float]`` that
                             returns a 768-dim embedding vector.  When provided,
                             the SDK will populate ``system_prompt_embedding`` on
                             every LLM call event for agent similarity clustering
                             and version-drift detection (Module 1).

    Returns:
        TelemetryClient instance (also set as the module-level default).

    Example::

        import ai_telemetry as telemetry
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")

        telemetry.init(
            app_id="my-ai-app",
            api_key="sk-telem-...",
            auto_patch_openai=True,
            embed_fn=lambda text: model.encode(text).tolist(),
        )
    """
    global _default_client

    client = TelemetryClient(
        app_id=app_id,
        api_key=api_key,
        endpoint=endpoint,
        capture_prompts=capture_prompts,
        debug=debug,
        enable_local_log=enable_local_log,
        log_path=log_path,
        embed_fn=embed_fn,
    )
    _default_client = client

    if auto_patch_openai:
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(client)

    if auto_patch_anthropic:
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(client)

    return client


def get_client() -> TelemetryClient:
    """Return the default initialized client."""
    if _default_client is None:
        raise RuntimeError(
            "ai_telemetry.init() has not been called. "
            "Call init(app_id=..., api_key=...) at startup."
        )
    return _default_client


__all__ = [
    "__version__",
    "init",
    "get_client",
    "TelemetryClient",
    "TelemetryEvent",
    "TraceContext",
    "trace_llm",
    "trace_tool",
    "get_fingerprint_store",
    "AgentRecord",
]
