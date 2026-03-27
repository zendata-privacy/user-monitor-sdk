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

Assuming you're at the root of the project (`/home/t8000/Downloads/CODE_Other_Models/Zendata/Financial_ADK/financial_rag_project`):
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