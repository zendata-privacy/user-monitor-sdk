# AI Telemetry SDK: Developer & Contributor Guide

Welcome to the AI Telemetry SDK Developer Guide. This document provides a deep dive into the architecture, design patterns, and extensibility of the SDK. It is intended for developers integrating the SDK into their enterprise applications, as well as open-source contributors looking to extend its capabilities.

---

## 1. Architectural Overview

The AI Telemetry SDK is designed to be a **zero-dependency, non-blocking, and highly extensible** observation layer for LLM applications.

### Core Design Principles
1. **Asynchronous Dispatch**: Telemetry tracking should never block the main application thread. Events are queued and flushed in a background thread.
2. **Context-Awareness**: Telemetry isn't just about latency; it's about *who* is doing *what*. The SDK uses Python's `contextvars` to seamlessly weave user identity and access risk into low-level LLM calls.
3. **Security-First (Zendata Integration)**: Native support for real-time payload inspection (DLP) and risk classification before logs are finalized.

### Component Diagram

```text
[ FastAPI / Flask Request ] --> Sets `current_user_context`
       │
       ▼
[ @trace_tool / @trace_llm ] --> Executes target function, captures latency/tokens
       │
       ▼
[ TelemetryClient.capture() ] --> 1. Intercepts payload
       │                          2. Calls Zendata DLP Engine (if configured)
       │                          3. Enqueues enriched TelemetryEvent
       ▼
[ Background Flush Thread ] ----> Periodically flushes Queue
       │
       ├──> Local CSV Writer (`logs/events.csv`)
       └──> HTTP POST to Remote Dashboard Endpoint
```

---

## 2. Core Components

### `TelemetryEvent` (`ai_telemetry/client.py`)
A `dataclass` representing a single unit of work. It captures:
* **Execution Metrics:** Latency, tokens, model, provider.
* **Payload Data:** Prompts, responses, tool inputs/outputs (unless `capture_prompts=False`).
* **User Context:** Populated automatically from the context store (e-mail, department, sailpoint_risk).
* **Security Analytics:** Populated via Zendata (PII types, trust boundary, composite risk).

### `TelemetryClient` (`ai_telemetry/client.py`)
The engine of the SDK. It manages a thread-safe `queue.Queue` and a daemon thread that wakes up every `flush_interval_s` (or when the application shuts down) to batch-write logs to disk and dispatch them to a remote HTTP endpoint.

### `current_user_context` (`ai_telemetry/context.py`)
Uses `contextvars.ContextVar`. In asynchronous frameworks like FastAPI, standard global variables leak across concurrent requests. `ContextVar` ensures that deep within the call stack, the SDK retrieves the exact user identity associated with the current HTTP request.

### `evaluate_event` (`ai_telemetry/integrations/zendata_integration.py`)
An interceptor that takes a pending `TelemetryEvent` of type `llm_call`, sends the prompt and completion to the Zendata Security Engine, and enriches the event with PII and risk classifications.

---

## 3. Implementation Guide for Developers

### Step 1: Initialization
Initialize the client as early as possible in your application lifecycle (e.g., FastAPI `@asynccontextmanager lifespan`).

```python
import ai_telemetry as telemetry

telemetry.init(
    app_id="my-app",
    api_key="sk-...",          # For remote ingest
    debug=True,                # Enable console logging for SDK activity
    capture_prompts=True       # Disable to strip PII before it even reaches Zendata
)
```

### Step 2: Context Injection (Middleware / Routers)
Whenever a request enters your system, populate the context variable. The decorators will pick this up automatically.

```python
import time
from ai_telemetry.context import current_user_context

@app.post("/chat")
async def chat_endpoint(request: Request):
    user_data = {
        "user_id": "usr_999",
        "email": "dev@bank.com",
        "department": "Engineering",
        "sailpoint_risk": "10",
        "tools_called": "",     # Initialize empty state for this request

        # Required for entity classification — must be set at the request boundary,
        # before any LLM call is made.
        "prompt_text":               request.question,          # raw user input
        "prompt_submission_time_ms": int(time.time() * 1000),   # epoch ms at receipt
    }
    # Set context safely for this specific async task
    current_user_context.set(user_data)

    # ... execute LLM logic ...
```

### Step 3: Decorating Functions
Use `@trace_llm` for model calls and `@trace_tool` for functional tools (like RAG retrievers, DB queries, or API calls).

```python
from ai_telemetry.decorators import trace_llm, trace_tool
import ai_telemetry as telemetry

t = telemetry.get_client()

@trace_tool(t, tool_name="fetch_customer_data")
def get_customer(id: str):
    # This automatically appends "fetch_customer_data" to the
    # 'tools_called' string in the current_user_context!
    pass

@trace_llm(t, model="gpt-4", provider="openai")
def generate_response(prompt: str):
    # When this completes, it captures latency, reads 'tools_called',
    # and invokes Zendata DLP before queuing the event.
    pass
```

### Step 4: Graceful Shutdown
Always flush the queue on app termination so you don't lose the last few events in memory.

```python
telemetry.get_client().shutdown()
```

---


---

## 3b. Entity Classification (Human vs. Agent Detection)

The SDK ships a built-in classifier that analyses every `llm_call` event and tags it as
**HUMAN**, **AUTOMATED**, **AGENT**, or **UNKNOWN** using 15 behavioral rules.  
Results appear as new fields on `TelemetryEvent` and in `logs/events.csv`.

### What the classifier needs from your application

The classifier needs two pieces of information that only your application layer can provide:

| Key | Type | Description |
|---|---|---|
| `prompt_text` | `str` | The raw prompt the user submitted (not the full messages array) |
| `prompt_submission_time_ms` | `int` | Epoch milliseconds at the moment the request was received |

Both must be set inside `current_user_context` **before** any LLM call executes.  
See **Step 2** above for the exact placement.

> **Why not infer these inside the SDK?**  
> The SDK patches LLM clients at the network level. By the time a patch fires, the LLM  
> call has already started — the submission timestamp is gone, and `input_messages` may  
> contain injected system prompts that would distort prompt-structure analysis.  
> Your application is the only place that sees the clean, original user input in time.

### How classification works

```text
TelemetryClient.capture(event)          ← called by auto-patcher
       │
       ├── reads current_user_context
       │     └── prompt_text, prompt_submission_time_ms, user_id
       │
       ├── resolves SessionState (keyed on trace_id)
       │     └── turn history, established entity type, probe scores
       │
       ├── runs all 15 rules (T1–T3, P1/P4, S1/S4, A1/A2, M1, D1–D5)
       │     └── each returns a list of signal strings
       │
       ├── votes on entity_type from signals
       │     AGENT wins if M1 fires; otherwise AUTOMATED vs HUMAN by vote count
       │
       ├── writes entity_type, classification_signals, behavioral_alerts
       │   session_turn_index, inter_turn_latency_ms onto the event
       │
       └── updates SessionState (turns, baselines, probe scores)
```

### Session state

State is maintained in memory by `ai_telemetry.session_state.SessionStore` (a module-level
singleton). Each session is keyed on `trace_id`; entity-level call-rate history is keyed
on `user_id`.

State is **not persisted** across process restarts. If you need cross-restart baselines
(e.g. for D1 drift detection), seed the store at startup by replaying recent risk scores
from your own datastore into `store.update_baseline(entity_key, score)`.

### Testing the classifier locally

```python
import time, sys
sys.path.insert(0, "sdk_v1")

from ai_telemetry.session_state import SessionStore
from ai_telemetry.entity_classifier import classify_event
from ai_telemetry.client import TelemetryEvent

store = SessionStore()

# Simulate an automated agent prompt
event = TelemetryEvent(event_type="llm_call", sailpoint_risk="10", trace_id="t1")
classify_event(
    event, store,
    session_id="t1",
    entity_key="usr_001",
    prompt_text="Return only JSON. Do not include any explanation. Format as {\"result\": ...}",
    submission_ms=int(time.time() * 1000) - 50,   # 50 ms — below human threshold
)
print(event.entity_type)             # AUTOMATED
print(event.classification_signals)  # T1_SUBMISSION_VELOCITY|P4_IMPERATIVE_FRAMING|...

# Simulate a multi-agent delegation prompt
event2 = TelemetryEvent(event_type="llm_call", sailpoint_risk="10", trace_id="t2")
classify_event(
    event2, store,
    session_id="t2",
    entity_key="agent_007",
    prompt_text="Your task is to extract all account numbers from the tool result below.",
    submission_ms=int(time.time() * 1000) - 80,
)
print(event2.entity_type)      # AGENT
print(event2.entity_subtype)   # SUBAGENT_RECEIVER
```

### Extending the classifier

The classifier lives in `ai_telemetry/entity_classifier.py`. Each rule is a standalone
function (`_check_T1`, `_check_P4`, etc.) that takes a `SessionState` and relevant inputs
and returns a list of signal strings. To add a new rule:

1. Write a `_check_XX(...)` function that returns `["MY_SIGNAL"]` or `[]`.
2. Add `"MY_SIGNAL"` to `_AUTOMATED_SIGNALS`, `_HUMAN_SIGNALS`, or `_ALERT_MAP`
   depending on what it indicates.
3. Call your function inside `classify_event()` and append its result to `all_signals`.

---

## 3c. Full Integration Checklist for Application Developers

This section consolidates every change your application code must make for the SDK to work correctly end-to-end — correct entity classification, human turn capture, and proper separation between real user traffic and synthetic test traffic.

### 1. Make the SDK importable

The SDK is a local package (`sdk_v1/`). Add it to `sys.path` **at the very top of your entry point**, before any other import, so it works regardless of how the server is launched (uvicorn, gunicorn, pytest, etc.):

```python
# app/main.py  — must be the first lines of the file
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk_v1"))

# All other imports follow
import ai_telemetry as telemetry
from ai_telemetry.client import TelemetryEvent
from ai_telemetry.context import current_user_context
```

> **Why not `pip install -e`?**
> Editable installs require `setuptools >= 64` with `editable-mode = "compat"` and write
> access to site-packages. The `sys.path.insert` approach works universally, survives
> folder renames, and requires no installation step.

---

### 2. Add `force_entity_type` to your request model

Add an optional field that defaults to `"AGENT"` for real chatbot traffic, but lets synthetic test scripts opt out by passing `null`:

```python
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = "usr_default"
    force_entity_type: Optional[str] = "AGENT"
    # ^ Default "AGENT": all LangChain internal calls (llm_call, tool_call) in a real
    #   chatbot session are agent-pipeline events and should be labelled accordingly.
    # ^ Pass null from synthetic test scripts to let the behavioral classifier run
    #   naturally and produce HUMAN / AUTOMATED classifications instead.
```

---

### 3. Capture a `user_message` event before setting `force_entity_type`

This is the critical two-phase pattern. The `user_message` event represents the human's turn and **must be captured before** `force_entity_type` is placed in the context — otherwise the override would suppress natural behavioural classification for the human turn too.

```python
@app.post("/chat")
async def chat(request: QueryRequest, http_request: Request):
    # ── Phase 1: build context WITHOUT force_entity_type ──────────────────
    user_data = {
        "user_id":                   request.user_id,
        "email":                     "...",
        "display_name":              "...",
        "department":                "...",
        "job_title":                 "...",
        "sailpoint_risk":            "0",
        "tools_called":              "",
        "data_sources":              "crm_db|docs_store",

        # Required by the entity classifier — must reflect the raw user input
        # and the exact moment the request arrived, before any LLM work starts.
        "prompt_text":               request.question,
        "prompt_submission_time_ms": int(time.time() * 1000),
    }
    current_user_context.set(user_data)

    # ── Phase 2: capture the human turn ───────────────────────────────────
    # No force_entity_type in context yet → the behavioural classifier runs
    # naturally and labels this event HUMAN (or AUTOMATED for bots/scripts).
    # Token count is estimated at ~4 characters per token.
    _estimated_tokens = max(1, len(request.question) // 4)
    telemetry.get_client().capture(TelemetryEvent(
        event_type="user_message",
        user_id=user_data.get("user_id"),
        email=user_data.get("email"),
        display_name=user_data.get("display_name"),
        department=user_data.get("department"),
        job_title=user_data.get("job_title"),
        sailpoint_risk=user_data.get("sailpoint_risk"),
        data_sources=user_data.get("data_sources"),
        prompt_tokens=_estimated_tokens,
        total_tokens=_estimated_tokens,
    ))

    # ── Phase 3: arm force_entity_type for the agent pipeline ─────────────
    # All subsequent llm_call and tool_call events (generated by LangChain)
    # are internal agent-pipeline events → force them to AGENT.
    # Synthetic test scripts pass force_entity_type=null to skip this.
    if request.force_entity_type:
        user_data["force_entity_type"] = request.force_entity_type
        current_user_context.set(user_data)

    # ── Phase 4: run the agent ─────────────────────────────────────────────
    response = await your_agent.run(request.question)
    return response
```

**What you see in the dashboard per user turn:**

| Event type | Entity | Explanation |
|---|---|---|
| `user_message` | `HUMAN` | The user's question — classified by behavioural signals |
| `llm_call` | `AGENT` | LangChain internal LLM call — forced |
| `tool_call` | `AGENT` | Tool call — inherits AGENT from session state |

---

### 4. Wire up the LangChain callback handler

Create the handler **after** `telemetry.init()` and pass it per-invocation via `config=`. If you also use `@trace_tool` decorators on your tool functions, disable the handler's tool hooks to avoid double-counting:

```python
from ai_telemetry.integrations.langchain_integration import create_langchain_handler

_lc_handler = create_langchain_handler(telemetry.get_client())

# Disable if tool calls are already captured by @trace_tool decorators
_lc_handler.on_tool_start = lambda *a, **kw: None
_lc_handler.on_tool_end   = lambda *a, **kw: None
_lc_handler.on_tool_error = lambda *a, **kw: None

# Pass per invocation so LCEL propagates it through the full chain
result = await agent_executor.ainvoke(
    {"input": user_message},
    config={"callbacks": [_lc_handler]},
)
```

> **Do not** create the handler before `telemetry.init()` — the client does not exist yet.

---

### 5. Write synthetic test scripts correctly

Any script that sends programmatic requests for testing or load generation must pass `force_entity_type=null` so its traffic is classified naturally (HUMAN or AUTOMATED) and does not pollute the AGENT bucket:

```python
import requests

def send_question(base_url: str, question: str, user_id: str) -> dict:
    payload = {
        "question":          question,
        "user_id":           user_id,
        "force_entity_type": None,   # ← lets the behavioural classifier run
    }
    r = requests.post(f"{base_url}/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()
```

The behavioural classifier will then fire timing and structure rules:
- Fast, templated, imperative prompts → `AUTOMATED`
- Conversational prompts with human-like pacing → `HUMAN`

---

### 6. Graceful shutdown

Always call `shutdown()` when the process exits so in-flight events are flushed before the background thread is killed:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    telemetry.get_client().shutdown()
```

---

## 4. Extending the SDK (For Open Source Contributors)

We welcome contributions to expand the ecosystem! Here is how you can contribute:

### Adding a New Auto-Patcher (e.g., OpenAI GPT-4o, Cohere)
If you want to add auto-instrumentation for a new LLM provider library:
1. Create a new file: `sdk_v1/ai_telemetry/integrations/yourprovider_integration.py`.
2. Implement a `patch_yourprovider(client: TelemetryClient)` function.
3. Monkey-patch the library's base generation method to wrap it in timing logic and construct a `TelemetryEvent`.
4. Add the flag to `ai_telemetry/__init__.py`'s `init()` parameters (e.g., `auto_patch_yourprovider=False`).

### Custom Log Exporters
Currently, `_send()` in `client.py` writes to a local CSV (`logs/events.csv`) and does an HTTP POST.
* **To contribute:** You could abstract `_send` into an `Exporter` interface, allowing developers to configure `KafkaExporter`, `S3Exporter`, or `DatadogExporter`.

### Modifying the Security Engine Integration
The Zendata integration in `zendata_integration.py` uses standard `urllib` to avoid heavy dependencies (like `requests` or `aiohttp`).
* Note that `evaluate_event` currently runs synchronously during `capture()`.
* **Future Enhancement:** If latency becomes a concern for large payloads, consider contributing an asynchronous evaluation pipeline using `asyncio` or a separate worker thread specifically for Zendata classification.

---

## 5. Local Testing

To test telemetry modifications locally without needing a real server:

1. Set `ZENDATA_ENGINE_URL=mock` in your environment.
2. Set `debug=True` in `telemetry.init()`.
3. Run a quick python script:

```python
import ai_telemetry as telemetry
from ai_telemetry.client import TelemetryEvent

client = telemetry.init("test", "test", debug=True)

# Directly capture a mock event
client.capture(TelemetryEvent(
    event_type="llm_call",
    input_messages=[{"role": "user", "content": "My SSN is 000-00-0000"}]
))

client.shutdown() # Forces flush
```
Check `logs/events.csv` to ensure the mock Zendata engine caught the SSN and flagged it as `pii_detected=True`.

## Step 1: Setup Environment and Dependencies

Assuming you're at the root of the project (`/home/t8000/Downloads/CODE_Other_Models/Zendata/Financial_SDK/financial_rag_project`):
```bash
# Activate your existing virtual environment
source venv/bin/activate

# Install the standard requirements
pip install -r requirements.txt

# Install your local telemetry SDK in editable mode
pip install -e ./sdk_v1
```

## Step 2: Configure Environment Variables

Ensure you have a `.env` file (you can copy `.env-example` if you haven't already) containing at least your Google Cloud credentials, Zendata configuration, and Telemetry settings:
```env
# Example essential config
GOOGLE_PROJECT_ID="your-gcp-project-id"
GOOGLE_LOCATION="us-central1"
TELEMETRY_ENABLED=true
TELEMETRY_STORAGE_PATH=./logs/events.csv
```

## Step 3: Run Data Pipelines (Optional if already fetched)

If you haven't fetched the data and trained the predictive model yet, run these sequentially:
```bash
python3 companies.py
python3 fetch_data.py
python3 app/models/train_predictor.py
```

## Step 4: Start the Server

Launch the FastAPI backend that serves the agents and the UI:
```bash
uvicorn app.main:app --reload --port 8080
```

---

## 3. Where is the Output Saved & Shown?

Once the system is running and you start interacting with it, outputs and logs are distributed in three main places:

### 1. User Interface (Frontend)
- **Where to find it:** Open your browser and go to `http://127.0.0.1:8080`.
- **What it shows:** You will see the web UI chat interface to interact with your Root/Sub-Agents.

### 2. Real-Time Telemetry Logs
- **Where to find it:** The `logs/events.csv` file.
- **What it shows:** Every time an agent processes a request or uses a tool, a tab-delimited record is appended here synchronously. It includes the execution metrics (latency, tokens used), the user's identity context, and security classifications (like PII types and Zendata composite risk scores).

### 3. Backend Console Logs
- **Where to find it:** Your terminal window running `uvicorn` (or outputted to `uvicorn.log` in the root).
- **What it shows:** Standard FastAPI routing logs, application errors, and general telemetry debug data.

> **Note:** You also have an offline batch processing script (`calculate_usage.py`) which, when run, will output aggregate analytics to your `logs_example/` directory (such as `identity_summary.csv` or `zendata_summary.json`).