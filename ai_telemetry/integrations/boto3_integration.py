"""boto3 auto-instrumentation — source identity with no application changes.

The other three source producers all require somebody to have annotated the
call: a ``source_uri=`` on the decorator, a ``source_context()`` block, or a
hand-maintained ``data_sources`` string. They are fine for a demo and they do
not survive contact with a customer account holding four hundred tool functions
nobody is going to go back and label.

This one requires nothing. It registers on botocore's event system and reports
the bucket, table, secret or knowledge base that was actually read, straight
from the request the application already made.

The vocabulary lines up by construction. The manifest's ``permitted_sources``
come from the agent's IAM policy — real ARNs — and what this reports is the ARN
of the resource the SDK call touched. Both sides of the comparison are the same
kind of string, produced by the same system, with no normalisation layer in
between to disagree about. That is the argument for building the manifest from
the scan rather than from a requirements document, and it only pays off if the
runtime side speaks ARNs too.

────────────────────────────────────────────────────────────────────────────────
WHY TWO EVENTS RATHER THAN ONE

``after-call`` carries the response but not the request parameters, so it cannot
tell you which table was queried. ``before-parameter-build`` carries the
parameters but fires before the call has succeeded.

Using only the first is impossible; using only the second would record every
*attempt* as a retrieval, including calls that failed with AccessDenied — and an
AccessDenied is the one case where the agent demonstrably did not receive the
data. So the URI is derived at ``before-parameter-build`` and stashed on
botocore's per-request ``context`` dict, then promoted to a recorded access at
``after-call``. The context object is created per request and passed to both
hooks, which makes this correct under concurrency in a way that any module-level
"last seen parameters" variable would not be.

────────────────────────────────────────────────────────────────────────────────
READS ONLY

Only operations that return data are mapped. ``PutObject`` is not a retrieval,
and recording it as one would put a source on the provenance chain that never
contributed a byte to the model's context — inflating every downstream check
with data the agent wrote rather than read.

The deliberate consequence: shared-memory collusion where agent A *writes* to a
bucket that agent B later reads is caught on B's read, not on A's write. That is
the right half to catch, since B's read is where out-of-scope data actually
enters a context window.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

# Per-request key on botocore's context dict. Namespaced because that dict is
# shared with botocore's own internals and with any other library doing this.
_CTX_KEY = "zendata_source_uri"

_lock = threading.Lock()
_patched_sessions: List[int] = []
_account_id: Optional[str] = None
_account_resolved = False


# ---------------------------------------------------------------------------
# Operation → resource mapping
# ---------------------------------------------------------------------------
# Only operations whose response carries data the agent can ground an answer in.
# Keyed by botocore service_id (lowercased), then operation name.

def _s3(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    bucket = p.get("Bucket")
    if not bucket:
        return None
    # S3 ARNs carry neither region nor account — this is the one service where
    # the ARN is genuinely global, so nothing has to be resolved.
    if op in ("GetObject", "HeadObject", "GetObjectAttributes", "SelectObjectContent"):
        key = p.get("Key")
        return f"arn:aws:s3:::{bucket}/{key}" if key else f"arn:aws:s3:::{bucket}"
    if op in ("ListObjectsV2", "ListObjects", "ListObjectVersions"):
        prefix = p.get("Prefix")
        return f"arn:aws:s3:::{bucket}/{prefix}" if prefix else f"arn:aws:s3:::{bucket}"
    return None


def _dynamodb(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    table = p.get("TableName")
    if not table:
        # BatchGetItem addresses several tables at once. Reporting only the
        # first would silently drop the others from the provenance chain, so it
        # is handled by the multi-source path instead.
        return None
    base = f"arn:aws:dynamodb:{region}:{account}:table/{table}"
    index = p.get("IndexName")
    return f"{base}/index/{index}" if index else base


def _dynamodb_multi(op: str, p: Dict[str, Any], region: str, account: str) -> List[str]:
    if op != "BatchGetItem":
        return []
    return [
        f"arn:aws:dynamodb:{region}:{account}:table/{t}"
        for t in (p.get("RequestItems") or {})
    ]


def _secretsmanager(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    secret = p.get("SecretId")
    if not secret:
        return None
    # SecretId accepts either a bare name or a full ARN. Passing a full ARN
    # through untouched matters: AWS appends a random six-character suffix to
    # every secret ARN, and reconstructing one from the name would produce a
    # string that never matches what IAM granted.
    return secret if secret.startswith("arn:") else \
        f"arn:aws:secretsmanager:{region}:{account}:secret:{secret}"


def _ssm(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    name = p.get("Name")
    if not name:
        return None
    return f"arn:aws:ssm:{region}:{account}:parameter{name if name.startswith('/') else '/' + name}"


def _bedrock_agent_runtime(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    kb = p.get("knowledgeBaseId")
    if kb:
        return f"arn:aws:bedrock:{region}:{account}:knowledge-base/{kb}"
    agent = p.get("agentId")
    if agent:
        return f"arn:aws:bedrock:{region}:{account}:agent/{agent}"
    return None


def _rds_data(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    # The RDS Data API is given the cluster ARN directly by the caller.
    return p.get("resourceArn")


def _kendra(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    index = p.get("IndexId")
    return f"arn:aws:kendra:{region}:{account}:index/{index}" if index else None


def _athena(op: str, p: Dict[str, Any], region: str, account: str) -> Optional[str]:
    db = (p.get("QueryExecutionContext") or {}).get("Database")
    return f"arn:aws:athena:{region}:{account}:database/{db}" if db else None


_READ_OPS: Dict[str, Dict[str, Callable]] = {
    "s3": {op: _s3 for op in (
        "GetObject", "HeadObject", "GetObjectAttributes", "SelectObjectContent",
        "ListObjectsV2", "ListObjects", "ListObjectVersions")},
    "dynamodb": {op: _dynamodb for op in (
        "GetItem", "Query", "Scan", "BatchGetItem", "ExecuteStatement", "PartiQLSelect")},
    "secrets manager": {"GetSecretValue": _secretsmanager},
    "ssm": {op: _ssm for op in ("GetParameter", "GetParameters", "GetParameterHistory")},
    "bedrock agent runtime": {op: _bedrock_agent_runtime for op in (
        "Retrieve", "RetrieveAndGenerate", "InvokeAgent")},
    "rds data": {op: _rds_data for op in ("ExecuteStatement", "BatchExecuteStatement")},
    "kendra": {op: _kendra for op in ("Query", "Retrieve")},
    "athena": {"StartQueryExecution": _athena},
}

_MULTI_OPS: Dict[str, Dict[str, Callable]] = {
    "dynamodb": {"BatchGetItem": _dynamodb_multi},
}

# Services this must never instrument. sts is called by the account resolver
# below, and instrumenting it would recurse; the rest are control-plane calls
# that retrieve no data an agent could ground an answer in.
_SKIP_SERVICES = frozenset({"sts", "iam", "cloudtrail", "macie2", "sso", "sso-oidc"})


# ---------------------------------------------------------------------------
# Account resolution
# ---------------------------------------------------------------------------

def _resolve_account(session) -> str:
    """The caller's AWS account id, resolved once per process.

    ARNs in an IAM policy carry a real account id, so an ARN built here without
    one would never match the manifest and every read would look out of scope.
    One STS call at first use is the cheapest correct answer; ``sts`` is in
    ``_SKIP_SERVICES`` so this cannot recurse into the interceptor.

    Failure degrades to an empty account rather than raising. An unmatched
    source is a false CRITICAL, which is bad — but taking down the host
    application's first S3 call to avoid it would be worse.
    """
    global _account_id, _account_resolved
    if _account_resolved:
        return _account_id or ""
    with _lock:
        if _account_resolved:
            return _account_id or ""
        _account_resolved = True
        try:
            _account_id = session.client("sts").get_caller_identity()["Account"]
        except Exception:
            _account_id = None
    return _account_id or ""


def set_account_id(account_id: str) -> None:
    """Pin the account id instead of resolving it via STS.

    Worth using when the scan record is already in hand: it holds the account,
    and pinning it removes a network call and a set of STS permissions from the
    deployment.
    """
    global _account_id, _account_resolved
    with _lock:
        _account_id = account_id
        _account_resolved = True


# ---------------------------------------------------------------------------
# The hooks
# ---------------------------------------------------------------------------

def _derive(service_id: str, op: str, params: Dict[str, Any],
            region: str, account: str) -> List[str]:
    multi = _MULTI_OPS.get(service_id, {}).get(op)
    if multi:
        found = multi(op, params, region, account)
        if found:
            return found
    single = _READ_OPS.get(service_id, {}).get(op)
    if single:
        uri = single(op, params, region, account)
        return [uri] if uri else []
    return []


def _make_before_hook(session, debug: bool):
    def _before_parameter_build(params=None, model=None, context=None, **kwargs):
        """Derive the source URI and stash it on the per-request context.

        Never raises. A hook that throws here would break the host
        application's AWS call, which is an unacceptable price for telemetry.
        """
        try:
            if context is None or model is None:
                return
            service_id = str(
                model.service_model.service_id).lower()      # e.g. "dynamodb"
            if service_id in _SKIP_SERVICES:
                return
            op = model.name
            if op not in _READ_OPS.get(service_id, {}) and \
               op not in _MULTI_OPS.get(service_id, {}):
                return

            region = getattr(session, "region_name", None) or ""
            account = _resolve_account(session)
            uris = _derive(service_id, op, params or {}, region, account)
            if uris:
                context[_CTX_KEY] = uris
        except Exception as exc:                            # noqa: BLE001
            if debug:
                print(f"[AI Telemetry] boto3 source derivation skipped: {exc}")
    return _before_parameter_build


def _make_after_hook(debug: bool):
    def _after_call(http_response=None, parsed=None, context=None, **kwargs):
        """Promote a stashed URI to a recorded access, on success only.

        An AccessDenied is the one case where the agent demonstrably did not
        receive the data, so recording it as a retrieval would attribute
        content to a source that returned none.
        """
        try:
            if not context:
                return
            uris = context.pop(_CTX_KEY, None)
            if not uris:
                return
            status = getattr(http_response, "status_code", 200)
            if status is not None and status >= 400:
                return
            if isinstance(parsed, dict) and parsed.get("Error"):
                return
            record_sources(uris)
        except Exception as exc:                            # noqa: BLE001
            if debug:
                print(f"[AI Telemetry] boto3 source recording skipped: {exc}")
    return _after_call


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def record_sources(uris: List[str]) -> None:
    """Add sources to the ones touched during the current tool call.

    Accumulated on the context var rather than emitted directly, because a
    single tool function commonly reads several places — a DynamoDB lookup then
    an S3 fetch — and all of them belong on that one tool call's provenance
    chain. ``scope_runtime`` drains this when the tool call is captured.

    Mirrors how ``trace_tool`` already accumulates ``tools_called``, so the two
    behave the same way inside one tool invocation.
    """
    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}
    existing = list(ctx.get("boto3_sources") or [])
    for uri in uris:
        if uri and uri not in existing:
            existing.append(uri)
    current_user_context.set({**ctx, "boto3_sources": existing})


def drain_sources() -> List[str]:
    """Take and clear the accumulated sources for the current context."""
    from ai_telemetry.context import current_user_context
    ctx = current_user_context.get() or {}
    found = list(ctx.get("boto3_sources") or [])
    if found:
        current_user_context.set({**ctx, "boto3_sources": []})
    return found


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def patch_boto3(telemetry_client=None, session=None, account_id: Optional[str] = None) -> bool:
    """Instrument boto3 so retrievals report the resource they actually read.

    Registers on the *session's* event system, not by monkey-patching client
    classes: this is botocore's own extension point, it survives client
    creation and recreation, and it leaves no wrapper for another library to
    trip over.

    Returns True when the hooks were registered.

        import ai_telemetry
        from ai_telemetry.integrations.boto3_integration import patch_boto3

        client = ai_telemetry.init(app_id="cs-agent", api_key=..., asset_record=rec)
        patch_boto3(client)
    """
    debug = bool(getattr(telemetry_client, "debug", False))
    try:
        import boto3
    except ImportError:
        if debug:
            print("[AI Telemetry] boto3 not installed — source interception disabled")
        return False

    if account_id:
        set_account_id(account_id)

    sess = session or boto3.DEFAULT_SESSION or boto3.setup_default_session() or boto3.DEFAULT_SESSION
    if sess is None:
        sess = boto3.Session()

    with _lock:
        if id(sess) in _patched_sessions:
            return True
        _patched_sessions.append(id(sess))

    events = sess.events
    events.register("before-parameter-build.*.*", _make_before_hook(sess, debug),
                    unique_id="zendata-source-derive")
    events.register("after-call.*.*", _make_after_hook(debug),
                    unique_id="zendata-source-record")

    if debug:
        print("[AI Telemetry] boto3 source interception active")
    return True


__all__ = ["patch_boto3", "record_sources", "drain_sources", "set_account_id"]
