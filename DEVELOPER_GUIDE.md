# AI Telemetry SDK: Developer & Contributor Guide

This guide covers the SDK's architecture, how to integrate it into an application, and how
to extend it. It describes the code in `ai_telemetry/` as of November 2026.

> **Integrating a client application?** The short, step-by-step version is
> [`docs/CLIENT_INTEGRATION.md`](docs/CLIENT_INTEGRATION.md). This guide goes deeper.

---

## 1. Architectural Overview

The SDK is a **zero-dependency, non-blocking** observation layer for LLM applications.

### Core Design Principles
1. **Non-blocking**: `capture()` only does cheap, in-memory work on the caller's thread
   and enqueues the event. Network calls (Zendata evaluation, PII API, ingest) run on a
   background flush thread.
2. **Context-aware**: the SDK uses Python `contextvars` plus a per-request accumulator
   (`start_request()`) to attach *who* made a request, *which tools* ran and *which
   sources* they read, even when an agent framework runs tools on worker threads.
3. **Declared, not guessed**: data sources (`source_uri`), egress (`record_egress`) and
   entitlements (identity roster) are declared by the host application. The SDK never
   infers a source from a tool's name.

### Data Flow

```text
[ Request ] --> start_request(user_id=…, prompt_text=…, …)      (per-request context)
     │
     ▼
[ @trace_tool / @trace_llm / LangChain handler / auto-patchers ]
     │   tools and sources recorded in the request accumulator;
     │   record_egress() declares data leaving the organisation
     ▼
[ TelemetryClient.capture(event) ]  — caller's thread, no network
     │   1. api_key_hash, build provenance, trace/span ids, session_id
     │   2. ip_hash / http_headers_digest (raw values never kept)
     │   3. trust_boundary_crossed from declared egress
     │   4. entity classification (llm_call, user_message); tool_call inherits
     │   5. PII labelling (inline classifier or your pii_redactor)
     │   6. strip content if capture_prompts=False, then enqueue
     ▼
[ Background flush thread ]  — every flush_interval_s (5 s) or batch_size (20) events
     │   1. Zendata PII API labelling (if pii_api_key is set)
     │   2. scope evaluation: sources vs the identity's manifest (every event type)
     │   3. llm_call only: Zendata risk evaluation, agent fingerprint,
     │      behavioural drift + ZASP score (non-human callers)
     ▼
     ├──> HTTP POST {"events": [...]} to `endpoint`, header X-API-Key   (primary)
     └──> Local CSV + JSONL (only when enable_local_log=True)          (optional)
```

---

## 2. Core Components

| Component | File | Role |
|---|---|---|
| `init()` | `ai_telemetry/__init__.py` | Builds the client, the manifest store (identity roster) and optional auto-patches |
| `TelemetryEvent` | `client.py` | Dataclass for one event: metrics, payload, caller context, security signals |
| `TelemetryClient` | `client.py` | Queue, `capture()`, background flush (`_send`), HTTP and local-log writers |
| `current_user_context`, `start_request`, `record_egress`, `source_context`, `TraceContext` | `context.py` | Per-request identity, accumulator, declared egress and sources, trace ids |
| `trace_tool`, `trace_llm` | `decorators.py` | Wrap tools and model calls |
| `create_langchain_handler` | `integrations/langchain_integration.py` | LangChain callback handler |
| `patch_openai` / `patch_anthropic` / `patch_boto3` | `integrations/` | Auto-instrumentation (`auto_patch_*` in `init()`) |
| `evaluate_event` | `integrations/zendata_integration.py` | Zendata risk evaluation (PII, trust boundary, composite risk) |
| Identity manifests | `identity_manifest.py`, `manifest.py`, `scope_runtime.py`, `scope_checker.py` | Per-identity scope from the roster; hard boundary checks |
| Entity classifier | `entity_classifier.py`, `session_state.py` | HUMAN / AGENT / API CALL / AMBIGUOUS |
| Agent fingerprint, drift, roles | `agent_fingerprint.py`, `drift_detector.py`, `role_inference.py`, `baseline.py` | Learned behaviour per agent and per human role |
| Persistence | `persistence.py` | Save / restore learned state as JSON |
| Export | `export.py` | `write_events_csv()`: events back to the local CSV format |

---

## 3. Implementation Guide for Developers

### Step 0: Install

The SDK is not on PyPI. Install it from this repository's folder:

```bash
pip install /path/to/user-monitor-sdk               # core
pip install "/path/to/user-monitor-sdk[langchain]"  # + extras: openai, anthropic, langchain, all
```

### Step 1: Initialization

Initialise once at startup, before any request is served:

```python
import os
import ai_telemetry as telemetry

telemetry.init(
    app_id="my-app",
    api_key=os.environ["ZENDATA_API_KEY"],
    endpoint="https://dashboard.zendata.dev/api/telemetry/ingest",
    capture_prompts=True,                       # False strips content (PII labels survive)
    identity_roster="config/identities.csv",    # one scope per identity (see §3e)
    pii_api_key=os.environ.get("PII_API_KEY"),
    pii_api_endpoint=os.environ.get("PII_API_URL"),
    debug=False,
)
```

Main parameters (all are documented in the `init()` docstring):

| Parameter | Default | Purpose |
|---|---|---|
| `app_id`, `api_key` | required | Application id and ingest key |
| `endpoint` | `https://ingest.zendata.dev/v1/events` | Ingest URL |
| `capture_prompts` | `True` | `False` removes prompt, response and tool payloads |
| `identity_roster` | `None` | Identity/entitlement CSV: one manifest per identity |
| `system_prompt`, `permitted_sources`, `permitted_tools`, `permitted_classifications`, `prohibited_sources` | `None` | Declare a single agent's scope directly (no roster) |
| `pii_api_key`, `pii_api_endpoint` | `None` | Zendata PII API labelling on the flush thread |
| `max_queue_size` | `1000` | Events held for the background flush. When full, new events are dropped (counted and reported at shutdown) |
| `pii_label_llm_input` | `True` | `False` sends only the model's answer of an `llm_call` to the PII API, when user messages and tool results are captured as their own events |
| `pii_concurrency` | `1` | PII API calls made at once per flush batch. The PII service serves one request per task, so keep this at or below its task count |
| `pii_redactor` | `None` | Your own redactor (masks and labels); otherwise the built-in classifier labels only (`classify_pii=True`) |
| `auto_patch_openai`, `auto_patch_anthropic`, `auto_patch_boto3` | `False` | Auto-instrumentation; boto3 reports the real bucket/table/secret ARN as the source |
| `baseline_path` / `baseline_store` | `None` | Offline baselines (`python -m ai_telemetry.baseline_builder`) for the soft scope boundary |
| `enable_local_log`, `log_path` | `False`, `None` | Local CSV + JSONL; default path `$ZENDATA_LOG_DIR/events.csv` (or the temp dir) |
| `debug` | `False` | Print SDK diagnostics |

### Step 2: Context Injection (once per request)

Open each request with `start_request()` **before** any LLM or tool call:

```python
import time
from ai_telemetry.context import TraceContext, start_request

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    start_request(
        user_id=req.user_id,                                  # roster id
        email="…", display_name="…", department="…",
        job_title="…", sailpoint_risk="…",
        session_id=req.session_id,                            # conversation id (optional)
        prompt_text=req.question,                             # raw user input
        prompt_submission_time_ms=int(time.time() * 1000),    # when it arrived
        ip_address=request.client.host if request.client else None,
        http_headers=dict(request.headers),                   # hashed, never stored raw
    )
    TraceContext.start_trace()
    ...
```

`start_request()` sets `current_user_context` **and** creates the request accumulator. Tools
record themselves and their sources into it from any thread, so every later event of the
request carries `tools_called` and the sources read. Setting `current_user_context`
directly still works, but tools run on other threads (LangChain's executor, for example)
are then invisible to later events.

Optional context keys the SDK reads:

| Key | Effect |
|---|---|
| `session_id` | Groups turns into one session (default: `user_id`) |
| `prompt_text`, `prompt_submission_time_ms` | Required by the entity classifier |
| `ip_address`, `http_headers` | Hashed into `ip_hash`, `http_headers_digest` |
| `caller_api_key` / `caller_api_key_hash` | Identifies a machine caller's credential (gives each agent its own `agent_id`) |
| `force_entity_type` | Forces the label on `llm_call` and `tool_call` events (see §3c) |

### Step 3: Decorating Functions

```python
from ai_telemetry.decorators import trace_llm, trace_tool
import ai_telemetry as telemetry

t = telemetry.get_client()

@trace_tool(t, tool_name="lookup_account",
            source_uri="accounts_db/customers", call_protocol="postgres")
def lookup_account(customer_id: str) -> dict: ...

def _source(args, kwargs, result):              # source known only at call time
    return f"{kwargs['resource']}/records"

@trace_tool(t, tool_name="search_records", source_uri=_source)
def search_records(resource: str, query: str) -> list: ...

@trace_llm(t, model="gpt-4o", provider="openai")
def generate_response(prompt: str): ...
```

`trace_tool(telemetry, tool_name=None, source_uri=None, classification=None, call_protocol=None)`:

- `source_uri`: a string, or a callable `(args, kwargs, result) -> str | list | None`.
  This is what the scope check compares against the identity's entitlements.
- `classification`: pins what the tool returns (`PII`, `PHI`, `CREDENTIALS`, `INTERNAL`, `PUBLIC`).
- `call_protocol`: taken from the `source_uri` scheme when omitted.

For sources known only inside the function, wrap the read:
`with source_context("vector-index://kb/chunk_id:42"): ...`

For data that leaves the organisation, declare it inside the tool:
`record_egress("mailto:a@other.com", external=True)`. The SDK sets
`trust_boundary_crossed` from these declarations instead of guessing.

### Step 4: Graceful Shutdown

```python
telemetry.get_client().shutdown()     # flushes the queue
```

For short-lived processes, carry learned state between runs (see §3f).

### Delivery counters (`stats()`)

`telemetry.get_client().stats()` returns cheap, thread-safe counters since the
client was created — useful for a host's own `/stats` or health endpoint:
`captured`, `dropped` (queue full), `queued`, `queue_max`, `sent_events` and
`sent_batches` (2xx POSTs), `send_failures` (batches not delivered: 4xx,
retries exhausted, or circuit breaker open), `pii_calls` / `pii_failures`
(PII API requests, cache misses only), and `last_send_ms` / `last_capture_ms`
(epoch ms, `None` until first seen).

---

## 3b. Entity Classification (Human vs. Agent Detection)

The classifier tags each `llm_call` and `user_message` event as **HUMAN**, **AGENT**,
**API CALL** (automated, non-delegating caller), **AMBIGUOUS** or **UNKNOWN**. `tool_call`
events inherit the session's label. Results appear as fields on `TelemetryEvent`
(`entity_type`, `entity_subtype`, `classification_signals`, `behavioral_alerts`,
`session_turn_index`, `inter_turn_latency_ms`).

### What the classifier needs from your application

| Key | Type | Description |
|---|---|---|
| `prompt_text` | `str` | The raw prompt the user submitted (not the full messages array) |
| `prompt_submission_time_ms` | `int` | Epoch milliseconds when the request was received |

Both go into the request context (Step 2) **before** any LLM call. Without `prompt_text`
the classifier does not run.

> **Why not infer these inside the SDK?** By the time a patched LLM call fires, the
> submission time is gone and `input_messages` may contain injected system prompts.
> Only your application sees the clean user input in time.

### How classification works

```text
TelemetryClient.capture(event)
       │
       ├── reads current_user_context: prompt_text, prompt_submission_time_ms, user_id, session_id
       ├── resolves SessionState (keyed on session_id, default user_id)
       ├── runs the rules (T1–T3, T5, T6, P1, P4, S1, S2, S4, A1, A2, B1, M1, D1–D5)
       ├── per-turn score = 0.5 + Σ signal weights; session score = EMA of turns
       │     3+ strong automation signals on one turn → score forced to 0.20
       ├── score < 0.25 → AGENT if M1 (delegation language) fired, else API CALL
       │   score > 0.75 → HUMAN;  otherwise AMBIGUOUS
       └── writes the result onto the event and updates SessionState
```

### Session state and persistence

Session state lives in memory (`ai_telemetry.session_state`, via `get_session_store()`).
Learned state (drift baselines, agent fingerprints, human role evidence, per-identity risk
baselines) can be saved and restored with `ai_telemetry.persistence` (§3f). Individual
conversations are deliberately not persisted.

### Testing the classifier locally

```python
import time
from ai_telemetry.session_state import SessionStore
from ai_telemetry.entity_classifier import classify_event
from ai_telemetry.client import TelemetryEvent

store = SessionStore()

# An automated caller: fast, templated, imperative prompts in one session.
for i in range(3):
    e = TelemetryEvent(event_type="llm_call", trace_id=f"b{i}")
    classify_event(e, store, session_id="batch", entity_key="svc_001",
                   prompt_text=f'Return only JSON. Do not include any explanation. Format as {{"id": {i}}}',
                   submission_ms=int(time.time() * 1000) - 50)
    print(e.entity_type)       # AMBIGUOUS, AMBIGUOUS, API CALL

# A delegating agent: M1 delegation language.
e2 = TelemetryEvent(event_type="llm_call", trace_id="t2")
classify_event(e2, store, session_id="t2", entity_key="agent_007",
               prompt_text="Your task is to extract all account numbers from the tool result below.",
               submission_ms=int(time.time() * 1000) - 80)
print(e2.entity_type, e2.entity_subtype)   # AGENT SUBAGENT_RECEIVER
```

A single templated turn usually stays AMBIGUOUS; the label settles as the session
accumulates turns.

### Extending the classifier

Each rule in `ai_telemetry/entity_classifier.py` is a standalone `_check_XX(...)` function
returning a list of signal strings. To add one:

1. Write `_check_XX(...)` returning `["MY_SIGNAL"]` or `[]`.
2. Give the signal a weight in `_SIGNAL_WEIGHTS` (negative = automation, positive = human),
   and add it to `_ALERT_MAP` if it should raise a behavioural alert.
3. Call it inside `classify_event()` and append its result to the signal list.

---

## 3c. Full Integration Checklist for Application Developers

### 1. Install the SDK

`pip install /path/to/user-monitor-sdk` (Step 0). The older `sys.path.insert(...)`
approach still works for local experiments but is not recommended for deployments.

### 2. Add `force_entity_type` to your request model (optional)

When your chatbot's own LLM calls should always be labelled AGENT, accept a field that
defaults to `"AGENT"` and lets test scripts opt out with `null`:

```python
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = "usr_default"
    force_entity_type: Optional[str] = "AGENT"
```

`force_entity_type` applies to `llm_call` and `tool_call` events only. `user_message`
events are always classified from behaviour.

### 3. Capture a `user_message` event at the start of each request

```python
@app.post("/chat")
async def chat(req: QueryRequest, request: Request):
    # Phase 1: open the request (no force_entity_type yet)
    ctx = start_request(
        user_id=req.user_id, email="…", display_name="…", department="…",
        job_title="…", sailpoint_risk="0",
        prompt_text=req.question,
        prompt_submission_time_ms=int(time.time() * 1000),
    )
    TraceContext.start_trace()

    # Phase 2: the human turn — classified from behaviour
    tokens = max(1, len(req.question) // 4)
    telemetry.get_client().capture(TelemetryEvent(
        event_type="user_message",
        user_id=req.user_id,
        input_messages=[{"role": "user", "content": req.question}],
        prompt_tokens=tokens, total_tokens=tokens,
    ))

    # Phase 3: label the agent pipeline (skipped when the caller sends null)
    if req.force_entity_type:
        ctx["force_entity_type"] = req.force_entity_type

    # Phase 4: run the agent
    return await your_agent.run(req.question)
```

`start_request()` returns the context dict it set, so adding a key to it (Phase 3) is
seen by every later event of the request.

| Event type | Entity | Explanation |
|---|---|---|
| `user_message` | from behaviour | The user's turn |
| `llm_call` | `AGENT` when forced, else from behaviour | The agent's model calls |
| `tool_call` | forced value, else the session's label | Tool calls |

### 4. Wire up the LangChain callback handler

Create it **after** `telemetry.init()` and pass it per invocation. When tools already use
`@trace_tool`, turn off the handler's tool capture so each call is recorded once:

```python
from ai_telemetry.integrations.langchain_integration import create_langchain_handler

handler = create_langchain_handler(telemetry.get_client(),
                                   capture_tools=False, agent_name="my-agent")

result = await agent_executor.ainvoke({"input": user_message},
                                      config={"callbacks": [handler]})
```

With LangChain `@tool`, put `@trace_tool` underneath it:

```python
@tool
@trace_tool(t, tool_name="search_records", source_uri=_source)
def search_records(resource: str, query: str) -> list:
    """Search a resource."""
```

### 5. Write synthetic test scripts correctly

Test scripts send `force_entity_type: null` so their traffic is classified from behaviour:

```python
payload = {"question": question, "user_id": user_id, "force_entity_type": None}
r = requests.post(f"{base_url}/chat", json=payload, timeout=120)
```

### 6. Graceful shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    telemetry.get_client().shutdown()
```

---

## 3d. Connecting your chatbot to a Zendata Autonomous Workflow scan

A Zendata **Autonomous Workflow** scan can score *your own* chatbot. Zendata sends it test
traffic; your SDK ships the resulting telemetry to Zendata; the scan collects it.

**1. Point the SDK at Zendata.** Copy the *Ingest endpoint* and *Project key* from step 4 of
the Autonomous Workflow wizard (the key is per user and can only send telemetry):

```python
import os
import ai_telemetry as telemetry

os.environ.setdefault("ZENDATA_ENGINE_URL", "mock")   # built-in risk evaluator — without it
                                                       # pii_detected / composite risk stay empty
telemetry.init(
    app_id="claims-bot",                               # = step 4 "Application identity"
    api_key=os.environ["ZENDATA_API_KEY"],             # the project key (zdi_…)
    endpoint="https://dashboard.zendata.dev/api/telemetry/ingest",
)
```

The SDK flushes every 5 s or every 20 events (`{"events": [...]}` with `X-API-Key`). Send
your chatbot one message, wait ~10 s, then press **Run Connectivity Test** in the wizard.

> If the wizard shows "Unavailable (NetworkError …)" for the endpoint and key, a browser
> content blocker is blocking URLs containing `telemetry`. Allow the dashboard's domain.

**Before relying on it, read how the key and endpoint behave**
([`docs/CLIENT_INTEGRATION.md` §9](docs/CLIENT_INTEGRATION.md#how-the-key-and-endpoint-behave)):
the key is per user, not per project, and a scan collects everything received under it in
its window; `app_id` must match the scan's Application identity or code findings stay
LATENT; only batches received before the scan's 20 s post-traffic wait are collected; a
POST is limited to 1,000 events / 5 MB (larger batches are refused and not retried); a
single key cannot be rotated on its own.

**2. Accept Zendata's test calls.** During the scan Zendata (from IP **3.84.255.108**) calls:

| Call | Request | Your reply |
|---|---|---|
| Health check | `GET {url}/` | any status below 500 |
| Each question | `POST {url}/chat` · `{"question": "...", "user_id": "<roster id>", "force_entity_type": null}` · optional auth header you configure | JSON `{"answer": "..."}` within 120 s |

Up to 10 calls run in parallel; *calls per identity* (1–10) × the roster, capped at 500.

**3. Attribute every call to its user.** Use the request's `user_id` in `start_request()`
(Step 2) before any LLM call. The IDs come from the scan's identity roster, so events join
to identities. Keep the §3c pattern: capture the `user_message` event first, then honour
`force_entity_type` (Zendata sends `null`, so the behavioural classifier runs).

`ai_telemetry.export.write_events_csv()` is what the scan uses to turn the received events
back into `events.csv` with exactly the columns this SDK writes locally.

---

## 3e. Identity Roster and Scope

`init(identity_roster="identities.csv")` builds one capability manifest per identity from
an identity/entitlement export (SailPoint, Entra ID, Okta, …). Column names are matched
flexibly (`identity_manifest.py`):

| Field | Accepted columns |
|---|---|
| id | `id`, `identity_id`, `principal_id`, `user_id`, `sub` |
| entitlements | `entitlements`, `permissions`, `grants`, `scopes` (`resource.action`, separated by `\|` or `,`) |
| type | `identity_type`, `type`, `principal_type` (`human` / `machine`) |
| clearance | `permitted_classifications`, `clearance`, `data_classifications` |
| certified prompt | `system_prompt_hash`, `certified_prompt_hash` |
| others | `displayName`, `email`, `department`, `roles`, `agent_type`, `description`, `approved_by` |

An entitlement with no action counts as read. A source is in scope when an entitlement's
resource equals it or is a prefix of it followed by `/` or `:` (`payroll_db` covers
`payroll_db/employees`, not `payroll_db_archive`). A trailing `/*` and glob patterns are
also supported (`manifest.py::_covers`). Name your `source_uri`s with the roster's
resource names.

---

## 3f. Persisting Learned State

Several detectors learn over many calls (drift baselines activate after 100 calls). A
process that restarts, such as a daily batch job, starts from zero unless it carries the
state:

```python
from ai_telemetry import persistence

persistence.load_state("/data/sdk_state.json")   # before the first event
...
persistence.save_state("/data/sdk_state.json")   # at shutdown (atomic write)
```

The file is JSON (never pickle), so loading it cannot execute code.

---

## 4. Extending the SDK (For Contributors)

### Adding a New Auto-Patcher
1. Create `ai_telemetry/integrations/yourprovider_integration.py`.
2. Implement `patch_yourprovider(client)` that wraps the library's generation method,
   times it and captures a `TelemetryEvent`.
3. Add an `auto_patch_yourprovider=False` flag to `init()` in `ai_telemetry/__init__.py`.

### Custom Exporters
`TelemetryClient._send()` runs the flush-thread enrichment, then `_send_http()` (primary)
and, when `enable_local_log=True`, `_write_csv()` and `_write_jsonl()`. A new destination
(Kafka, S3, Datadog) belongs next to `_send_http()`.

### Modifying the Security Engine Integration
`evaluate_event` in `integrations/zendata_integration.py` uses only `urllib`. It runs on the
background flush thread for `llm_call` events, never on the caller's thread.
`ZENDATA_ENGINE_URL=mock` uses the built-in evaluator.

---

## 5. Local Testing

Test without a server by writing events to a local file:

```python
import os
import ai_telemetry as telemetry
from ai_telemetry.client import TelemetryEvent

os.environ["ZENDATA_ENGINE_URL"] = "mock"

client = telemetry.init("test", "test", endpoint="http://127.0.0.1:9/disabled",
                        enable_local_log=True, log_path="./events.csv", debug=True)
client.capture(TelemetryEvent(
    event_type="llm_call",
    input_messages=[{"role": "user", "content": "My SSN is 078-05-1120"}],
))
client.shutdown()      # flushes
```

`./events.csv` (and `./events.jsonl`) will show `pii_detected=True`, `pii_types=SSN`. The
HTTP send to the disabled endpoint fails silently; local logs are written regardless.

Unit tests: `python -m pytest tests/` (Python) and `npm test` (TypeScript).

---

## 6. Where Output Goes

| Output | When | Location |
|---|---|---|
| Ingest (HTTP) | always | `endpoint`, batches of `{"events": [...]}` with `X-API-Key` |
| Local CSV | `enable_local_log=True` | `log_path`, default `$ZENDATA_LOG_DIR/events.csv` (temp dir if unset) |
| Local JSONL | `enable_local_log=True` | same path with `.jsonl` |
| Diagnostics | `debug=True` | stdout, prefixed `[AI Telemetry]` |
