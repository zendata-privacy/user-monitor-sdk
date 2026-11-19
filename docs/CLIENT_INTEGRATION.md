# Client Integration Guide (Python)

How to add the Zendata AI Telemetry SDK to your AI application so a Zendata
scan can see **who** called your assistant, **which tools** ran, **which data
sources** they read, and **where data left** your organisation.

Every step below is taken from a working reference integration (the Zendata
virtual bank, a FastAPI + LangChain assistant). Function names and arguments
match the SDK code in `ai_telemetry/`.

---

## 1. Install

The SDK is distributed by Zendata as a source package (it is **not** on PyPI).
Install it from the folder or archive Zendata gives you:

```bash
pip install /path/to/user-monitor-sdk                 # core, zero hard dependencies
pip install "/path/to/user-monitor-sdk[langchain]"    # + LangChain callback handler
pip install "/path/to/user-monitor-sdk[openai]"       # + OpenAI auto-patch
pip install "/path/to/user-monitor-sdk[anthropic]"    # + Anthropic auto-patch
pip install "/path/to/user-monitor-sdk[all]"          # everything
```

Python 3.10 or newer. In a Docker image:

```dockerfile
COPY user-monitor-sdk /opt/sdk
RUN pip install --no-cache-dir /opt/sdk
```

The package installs as `ai-telemetry-sdk` and is imported as `ai_telemetry`.

---

## 2. Initialise once, at startup

Call `init()` before any request is served and before anything that uses the
client (tools, LangChain handler).

```python
import os
import ai_telemetry as telemetry

# Built-in risk evaluator: fills pii_detected, zendata_composite_risk and
# zendata_classification. Without it those risk columns stay empty.
os.environ.setdefault("ZENDATA_ENGINE_URL", "mock")

telemetry.init(
    app_id="claims-assistant",                       # must match the scan's app_id (§9)
    api_key=os.environ["ZENDATA_API_KEY"],           # project key from the scan wizard (zdi_…)
    endpoint="https://dashboard.zendata.dev/api/telemetry/ingest",
    capture_prompts=True,                            # False strips prompt/response text
    identity_roster="config/identities.csv",         # see §3 — one scope per identity
    pii_api_key=os.environ.get("PII_API_KEY"),       # optional — Zendata PII API
    pii_api_endpoint=os.environ.get("PII_API_URL"),  #   classification of retrieved data
    enable_local_log=False,                          # True + log_path= keeps a local CSV copy
)
```

The SDK queues events and sends them from a background thread (every 5 s or
every 20 events) to `endpoint` with header `X-API-Key: <api_key>`. It never
blocks your request.

### Most-used `init()` parameters

| Parameter | Default | Purpose |
|---|---|---|
| `app_id` | required | Your application's id |
| `api_key` | required | Zendata project key |
| `endpoint` | `https://ingest.zendata.dev/v1/events` | Where events are sent. For a scan, use the *Ingest endpoint* shown in the wizard |
| `capture_prompts` | `True` | `False` removes prompt and response content from events |
| `identity_roster` | `None` | Path to the identity/entitlement CSV (§3) |
| `pii_api_key` / `pii_api_endpoint` | `None` | Enable Zendata PII API classification of data your tools return |
| `system_prompt` | `None` | Your agent's system prompt; hashed so prompt changes are detected |
| `permitted_sources` / `permitted_tools` / `permitted_classifications` | `None` | Declare one agent's scope directly when you have no roster |
| `auto_patch_openai` / `auto_patch_anthropic` | `False` | Instrument those client libraries automatically |
| `enable_local_log` / `log_path` | `False` / `None` | Also write events to a local CSV |
| `debug` | `False` | Print SDK debug output |

The full list, with descriptions, is in the `init()` docstring in
`ai_telemetry/__init__.py`.

---

## 3. The identity roster

The roster tells the SDK what each identity (person, service account, AI agent)
is **entitled** to. Out-of-scope reads and scope drift are judged against it.
It is a CSV export from your identity system (SailPoint, Entra ID, Okta, …).
Column names are matched flexibly; the important ones are:

| Column (accepted names) | Example |
|---|---|
| `id` (`user_id`, `principal_id`, `sub`) | `agt_risk_summarizer` |
| `entitlements` (`permissions`, `grants`, `scopes`) | `risk_model_db.read\|audit_db.read\|reporting.write` |
| `displayName`, `email`, `department` | |
| `identity_type` (`type`) | `human` or `machine` |
| `agent_type` | `batch_automation` |
| `description` (`purpose`) | the agent's stated purpose |
| `permitted_classifications` (`clearance`) | `PII\|INTERNAL` |
| `system_prompt_hash` | certified prompt hash, for machine identities |

Entitlements are `<resource>.<action>`, separated by `|`. The `<resource>` names
are the vocabulary your tools must use when they report a source (§5).

---

## 4. Start every request with the caller's identity

At the start of each inbound request, before any model or tool call, open the
request context with `start_request()` and record the user's message:

```python
import time
from ai_telemetry.client import TelemetryEvent
from ai_telemetry.context import TraceContext, start_request

def begin_request(identity: dict, text: str, session_id: str, request) -> None:
    start_request(
        user_id=identity["id"],                    # must match the roster id
        email=identity.get("email", ""),
        display_name=identity.get("displayName", ""),
        department=identity.get("department", ""),
        job_title=identity.get("jobTitle", ""),
        sailpoint_risk=identity.get("riskScore", ""),
        session_id=session_id,
        prompt_text=text,                                   # raw user input
        prompt_submission_time_ms=int(time.time() * 1000),  # when it arrived
        ip_address=request.client.host if request.client else None,
    )
    TraceContext.start_trace()                     # one trace per interaction
    telemetry.get_client().capture(TelemetryEvent(
        event_type="user_message",
        user_id=identity["id"],
        session_id=session_id,
        input_messages=[{"role": "user", "content": text}],
    ))
```

Use `start_request()` rather than setting `current_user_context` yourself:
it also creates the per-request record that collects tools, sources and egress
across threads. Agent frameworks run tools on worker threads, and without it
the tools used would not appear on the request's events.

Inside your own code, `current_user_context.get()` returns the caller's fields
(`(current_user_context.get() or {}).get("user_id")`).

---

## 5. Decorate your tools with `@trace_tool`

Every function your agent can call should be wrapped. Declare **where it reads
from** with `source_uri`; that is what scope checks use.

```python
import ai_telemetry as telemetry
from ai_telemetry.decorators import trace_tool

t = telemetry.get_client()

# A tool that always reads the same place: a fixed string.
@trace_tool(t, tool_name="get_market_data",
            source_uri="market_data/latest", call_protocol="sqlite")
def get_market_data(symbol: str) -> dict: ...

# A tool whose source depends on its arguments: a callable
# (args, kwargs, result) -> str | list | None.
def _read_source(args, kwargs, result):
    resource = kwargs.get("resource") or (args[0] if args else None)
    return f"{resource}/records" if resource else None

@trace_tool(t, tool_name="search_records",
            source_uri=_read_source, call_protocol="sqlite")
def search_records(resource: str, query: str) -> list: ...

# A tool that reads no data source: name only.
@trace_tool(t, tool_name="transfer_funds")
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict: ...
```

`@trace_tool` arguments:

| Argument | Purpose |
|---|---|
| `tool_name` | Name shown in findings (defaults to the function name) |
| `source_uri` | String or callable: the source the call read |
| `call_protocol` | How the source is reached (`https`, `postgres`, `sqlite`, …). Taken from the `source_uri` scheme when omitted |
| `classification` | Pin what the tool returns (`PII`, `PHI`, `CREDENTIALS`, `INTERNAL`, `PUBLIC`) when content inspection would get it wrong |

**Source naming rule.** A source is in scope when an entitlement's resource
equals it or is a prefix of it followed by `/` or `:`. With entitlement
`payroll_db.read`, the sources `payroll_db`, `payroll_db/employees` and
`payroll_db:2026` are covered; `payroll_db_archive` is not. Start your source
URIs with the roster's resource names.

For a source known only inside the function (a vector-index chunk, an object
path built from input), wrap the read instead:

```python
from ai_telemetry.context import source_context

with source_context(f"vector-index://kb/chunk_id:{hit.id}"):
    chunk = fetch(hit.id)
```

With LangChain, put `@trace_tool` **under** `@tool`:

```python
from langchain_core.tools import tool

@tool
@trace_tool(t, tool_name="search_records", source_uri=_read_source)
def search_records(resource: str, query: str) -> list:
    """Search a resource."""
```

---

## 6. Declare data that leaves your organisation

When a tool sends data somewhere (email, webhook, third-party API), declare the
destination and whether it is outside your trust boundary. The SDK decides
trust-boundary crossings from these declarations, not from tool names.

```python
from ai_telemetry.context import record_egress

ORG_DOMAIN = "example.com"

@trace_tool(t, tool_name="send_email")
def send_email(to: str, subject: str, body: str) -> dict:
    for r in [x.strip() for x in to.split(",") if x.strip()]:
        record_egress(f"mailto:{r}", external=not r.lower().endswith("@" + ORG_DOMAIN))
    ...
```

---

## 7. LangChain

Create the handler **after** `init()` and pass it on each invocation. When your
tools already use `@trace_tool`, set `capture_tools=False` so each tool call is
recorded once:

```python
from ai_telemetry.integrations.langchain_integration import create_langchain_handler

handler = create_langchain_handler(telemetry.get_client(),
                                   capture_tools=False, agent_name="claims-agent")

result = await agent_executor.ainvoke({"input": text}, config={"callbacks": [handler]})
```

The handler records the model calls (prompts, completions, tokens, latency).
For OpenAI or Anthropic clients without LangChain, use `auto_patch_openai=True`
/ `auto_patch_anthropic=True` in `init()`, or wrap your own call with
`@trace_llm(t, model="…", provider="…")` from `ai_telemetry.decorators`.

---

## 8. Shut down cleanly

Flush queued events when the process exits:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    yield
    telemetry.get_client().shutdown()
```

### Short-lived processes (batch jobs)

Drift baselines, agent fingerprints and risk baselines are learned in memory.
A process that restarts often (a daily batch job) can carry them between runs:

```python
from ai_telemetry import persistence

persistence.load_state("/data/sdk_state.json")   # at startup, before the first event
...
persistence.save_state("/data/sdk_state.json")   # at shutdown
```

A long-running service does not need this.

---

## 9. Connect to a Zendata scan

1. In the Autonomous Workflow wizard, copy the **Ingest endpoint** and
   **Project key** into `endpoint=` and `api_key=` (§2).
2. Your chatbot must answer:
   - `GET {url}/`: any status below 500 (health check)
   - `POST {url}/chat` with `{"question": "...", "user_id": "<roster id>", "force_entity_type": null}`:
     reply `{"answer": "..."}` within 120 s. Up to 10 calls run in parallel.
3. Use the request's `user_id` as the identity in `start_request()` (§4), so
   events join to the scan's roster.
4. Send one message, wait about 10 s, then press **Run Connectivity Test**.

Scan traffic comes from IP **3.84.255.108**; allow it if your endpoint is
restricted.

### How the key and endpoint behave

- **The key is personal, not per project.** Despite the label "Project key",
  each Zendata user has one key, and everything sent with it goes to that
  user's ingest area. A scan collects **every** batch received in its time
  window under the key, whatever `app_id` the events carry. Use the key in one
  chatbot at a time, and don't run two scans of different chatbots at once
  with the same account, or their events mix.
- **`app_id` must match the scan.** Set `app_id=` to the value chosen under
  **Application identity** on step 4 (it is read from your source by the
  discovery scan). With any other value, code findings cannot be CONFIRMED by
  runtime telemetry; they stay LATENT.
- **Only telemetry received during the scan counts.** The scan collects batches
  received after its job starts (or up to the **Collection window** hours
  before, if set) and stops collecting **20 s** after its last call to your
  chatbot. Events your SDK sends later are not part of that scan, so keep
  flushes prompt: the SDK flushes every 5 s, but with `pii_api_key` set each
  event is labelled before it is sent, which can delay a flush under load.
- **Batch limits.** The endpoint accepts at most **1,000 events** and **5 MB**
  per POST. A larger batch is refused (HTTP 413) and the SDK does not retry
  client errors, so that batch is lost. Keep `capture_prompts=False` or lower
  traffic bursts if your events are large.
- **The key can only send.** It cannot read anything. Requests from a user
  who has been deactivated by their organisation are refused. Keys are derived
  from a Zendata server secret, so a single key cannot be rotated on its own:
  if a key leaks, contact Zendata, which rotates the secret (this changes every
  user's key).
- If the wizard shows "Unavailable (NetworkError …)" instead of the endpoint
  and key, a browser content blocker is blocking URLs containing `telemetry`.
  Allow the dashboard's domain.

---

## Checklist

- [ ] SDK installed (`import ai_telemetry` works)
- [ ] `ZENDATA_ENGINE_URL=mock` set, then `telemetry.init()` at startup with the Zendata endpoint and key
- [ ] `app_id` equals the scan's **Application identity** (step 4)
- [ ] Identity roster CSV passed as `identity_roster=`
- [ ] `start_request()` + `user_message` event at the start of every request
- [ ] Every tool wrapped with `@trace_tool`, with `source_uri` for tools that read data
- [ ] Source URIs start with the roster's resource names
- [ ] `record_egress()` in every tool that sends data out
- [ ] LangChain handler with `capture_tools=False` (if using LangChain)
- [ ] `shutdown()` on exit
