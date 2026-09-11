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
from ai_telemetry.context import current_user_context, TraceContext, source_context
from ai_telemetry.agent_fingerprint import get_fingerprint_store, AgentRecord
from ai_telemetry.manifest import AgentCapabilityManifest, ManifestBuilder
from ai_telemetry.source_provenance import SourceProvenanceTag, InterAgentMessage
from ai_telemetry.scope_checker import ScopeBoundaryChecker, ScopeViolation
from ai_telemetry.baseline import AgentBehavioralBaseline, BaselineStore

_default_client: TelemetryClient | None = None


def init(
    app_id: str,
    api_key: str,
    endpoint: str = "https://ingest.zendata.dev/v1/events",
    capture_prompts: bool = True,
    auto_patch_openai: bool = False,
    auto_patch_anthropic: bool = False,
    auto_patch_boto3: bool = False,
    debug: bool = False,
    enable_local_log: bool = False,
    log_path: Optional[str] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    pii_redactor: Optional[object] = None,
    classify_pii: bool = True,
    # ── Module 9 — capability manifest capture ───────────────────────────
    system_prompt: Optional[str] = None,
    requirements_doc: Optional[bytes] = None,
    asset_record: Optional[dict] = None,
    approved_by: str = "",
    permitted_sources: Optional[List[str]] = None,
    permitted_classifications: Optional[List[str]] = None,
    permitted_tools: Optional[List[str]] = None,
    prohibited_sources: Optional[List[str]] = None,
    manifest_extract_fn: Optional[Callable] = None,
    manifest_version: int = 1,
    pii_api_key: Optional[str] = None,
    pii_api_endpoint: Optional[str] = None,
    max_hash_bytes: Optional[int] = None,
    baseline_path: Optional[str] = None,
    baseline_store: Optional[object] = None,
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
        auto_patch_boto3:    Report the bucket, table, secret or knowledge base
                             each AWS read actually touched, with no changes to
                             application code. This is what makes the capability
                             manifest enforceable in an account nobody is going
                             to hand-annotate — and it emits ARNs, the same
                             vocabulary the manifest holds when it is built from
                             an IAM scan.
        debug:               Print debug logs to stdout.
        embed_fn:            Optional callable ``(text: str) -> List[float]`` that
                             returns a 768-dim embedding vector.  When provided,
                             the SDK will populate ``system_prompt_embedding`` on
                             every LLM call event for agent similarity clustering
                             and version-drift detection (Module 1).
        classify_pii:        Label pii_detected/pii_types with the built-in
                             regex classifier when no redactor is supplied
                             (default True). It labels only, never masks, and
                             runs before prompts are dropped so the signal
                             survives capture_prompts=False.
        pii_redactor:        Optional bring-your-own redactor exposing
                             ``.redact(text) -> (masked, [types])``. The reference
                             implementation is the Zendata PII API client; see
                             ``guide_to_deploy.md`` §7.

        system_prompt:       The agent's system prompt. Hashed into the capability
                             manifest; a later runtime prompt that does not match
                             fires ``SC_MANIFEST_DRIFT``. This is the check no
                             observed baseline can make, because a baseline
                             certifies whatever it is shown during its learning
                             window — including an unauthorised edit that was
                             live at the time.
        asset_record:        An ``AssetRecord`` dict from the AWS asset scanner.
                             The preferred source of scope: its
                             ``data_sources_permitted`` come from the agent's IAM
                             policy, which is enforceable ground truth and speaks
                             the same ARN vocabulary the runtime reports.
        requirements_doc:    Raw bytes of a requirements/PRD document. Parsed via
                             ``manifest_extract_fn`` into declared scope. Where a
                             scan is also present the document does not override
                             it — the disagreement is recorded in
                             ``manifest.scope_gaps``, because a PRD promising one
                             bucket against a role granting forty is a finding.
        manifest_extract_fn: ``(prompt, document) -> str`` returning JSON. Supplied
                             by the host because the SDK has no LLM client and no
                             hard dependencies.
        permitted_sources:   Explicit override of the derived source list.
        baseline_path:       Path to baselines.jsonl from
                             ``python -m ai_telemetry.baseline_builder``. Activates
                             the soft boundary for any agent with at least 100
                             observed sessions; agents below that stay suppressed.
                             Loaded, never accumulated — nothing in this SDK
                             survives a restart or is shared between replicas, so
                             a baseline learned in process would see a fraction
                             of traffic and call familiar sources unseen.
        pii_api_key:         Enables Zendata PII API classification of retrieved
                             content, which is what populates a provenance tag's
                             ``classification`` accurately. Called only on the
                             background flush thread, never on the caller's.

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

    # Build the capability manifest before the client exists, so the hard
    # boundary is in place for the very first event rather than for the first
    # event after some later registration step.
    manifest = None
    if any((system_prompt, asset_record, requirements_doc, permitted_sources,
            permitted_classifications)):
        from ai_telemetry.manifest import ManifestBuilder
        manifest = ManifestBuilder(agent_id=app_id, debug=debug).build(
            system_prompt=system_prompt,
            asset_record=asset_record,
            requirements_doc=requirements_doc,
            extract_fn=manifest_extract_fn,
            permitted_sources=permitted_sources,
            permitted_classifications=permitted_classifications,
            permitted_tools=permitted_tools,
            prohibited_sources=prohibited_sources,
            approved_by=approved_by,
            manifest_version=manifest_version,
        )

    if baseline_store is None and baseline_path:
        from ai_telemetry.baseline import BaselineStore
        baseline_store = BaselineStore.from_jsonl(baseline_path)
        if debug:
            ready = baseline_store.ready_agents
            print(f"[AI Telemetry] Loaded {len(baseline_store)} baseline(s); "
                  f"{len(ready)} ready for soft-boundary signals")

    client = TelemetryClient(
        app_id=app_id,
        api_key=api_key,
        endpoint=endpoint,
        capture_prompts=capture_prompts,
        debug=debug,
        enable_local_log=enable_local_log,
        log_path=log_path,
        embed_fn=embed_fn,
        pii_redactor=pii_redactor,
        classify_pii=classify_pii,
        manifest=manifest,
        pii_api_key=pii_api_key,
        pii_api_endpoint=pii_api_endpoint,
        max_hash_bytes=max_hash_bytes,
        baseline_store=baseline_store,
    )
    _default_client = client

    if auto_patch_openai:
        from ai_telemetry.integrations.openai_integration import patch_openai
        patch_openai(client)

    if auto_patch_anthropic:
        from ai_telemetry.integrations.anthropic_integration import patch_anthropic
        patch_anthropic(client)

    if auto_patch_boto3:
        from ai_telemetry.integrations.boto3_integration import patch_boto3
        patch_boto3(client, account_id=(asset_record or {}).get("account_id"))

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
    "source_context",
    "AgentCapabilityManifest",
    "ManifestBuilder",
    "SourceProvenanceTag",
    "InterAgentMessage",
    "ScopeBoundaryChecker",
    "ScopeViolation",
    "AgentBehavioralBaseline",
    "BaselineStore",
    "get_fingerprint_store",
    "AgentRecord",
]
