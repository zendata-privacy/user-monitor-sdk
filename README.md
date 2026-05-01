# AI Telemetry SDK

Zero-dependency telemetry SDK for AI applications. Captures LLM prompts & responses, tool/function calls, token usage, latency, and errors — then batches and ships them to your platform.

Designed like **Sentry** (easy install + API key) with the instrumentation depth of **Datadog**.

---

## How It Works

```
Your App → [AI Telemetry SDK] → (batched HTTPS) → Your Ingest Endpoint → Dashboard
```

1. **Register your app** on the platform → get an `app_id` and `api_key`
2. **Install the SDK** and call `init()`
3. The SDK **auto-patches** OpenAI / Anthropic clients (or use a LangChain callback)
4. Events are **buffered** and flushed every 5 seconds (or when the batch hits 20 events)
5. Your dashboard shows traces, token spend, error rates, and latency

---

## Python SDK

### Install

```bash
pip install ai-telemetry-sdk                    # core only — zero hard dependencies
pip install "ai-telemetry-sdk[openai]"          # + OpenAI
pip install "ai-telemetry-sdk[anthropic]"       # + Anthropic
pip install "ai-telemetry-sdk[langchain]"       # + LangChain
pip install "ai-telemetry-sdk[all]"             # everything
```

The SDK itself is **cloud-agnostic** — there is no AWS or Azure SDK in
its dependency graph. PII masking is delegated to a user-supplied
redactor (the project root provides a Zendata `pii-api` integration);
log shipping is handled by the project-level `upload_blob_bucket.py`, which
selects between Amazon S3 and Azure Blob Storage at runtime via a
`CLOUD_PROVIDER` env var. See `../guide_to_deploy.md` §8 and §10b for
the multi-cloud deployment recipe (Azure Container Instances and
Azure Container Apps, with the AWS-hosted `pii-api` reached over
HTTPS).

### Quickstart — OpenAI auto-patch

```python
import ai_telemetry as telemetry
import openai

# 1. Initialize once at app startup
telemetry.init(
    app_id="my-ai-app",
    api_key="sk-telem-...",
    auto_patch_openai=True,     # patches openai globally
    capture_prompts=True,       # set False to redact content
)

# 2. Use OpenAI exactly as normal — telemetry is automatic
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Quickstart — Anthropic auto-patch

```python
import ai_telemetry as telemetry
import anthropic

telemetry.init(
    app_id="my-ai-app",
    api_key="sk-telem-...",
    auto_patch_anthropic=True,
)

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Quickstart — LangChain

```python
import ai_telemetry as telemetry
from ai_telemetry.integrations.langchain_integration import TelemetryCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

t = telemetry.init(app_id="my-app", api_key="sk-telem-...")
handler = TelemetryCallbackHandler(t)

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm

result = chain.invoke({"topic": "AI observability"}, config={"callbacks": [handler]})
```

### Manual Instrumentation (any HTTP / custom model)

```python
import ai_telemetry as telemetry
from ai_telemetry.decorators import trace_llm, trace_tool

t = telemetry.init(app_id="my-app", api_key="sk-telem-...")

@trace_llm(t, model="llama-3", provider="ollama")
def call_local_model(prompt: str) -> str:
    # ... your HTTP call ...
    return "response text"

@trace_tool(t, tool_name="web_search")
def web_search(query: str) -> str:
    # ... your tool logic ...
    return "results"
```

### Manual Event Capture

```python
from ai_telemetry.client import TelemetryEvent

t.capture(TelemetryEvent(
    event_type="llm_call",
    model="gpt-4o",
    provider="openai",
    prompt_tokens=150,
    completion_tokens=45,
    total_tokens=195,
    latency_ms=823,
    metadata={"user_id": "u_123", "session": "sess_abc"},
))
```

### Zendata DLP & Security Integration (Python)

This SDK natively supports real-time Data Loss Prevention (DLP) and Security evaluation via the **Zendata Engine**.

#### Context Injection
Use `current_user_context` from `ai_telemetry.context` to inject identity and risk data into your LLM traces.

```python
from ai_telemetry.context import current_user_context

# Set this context per request
current_user_context.set({
    "user_id": "usr_123",
    "email": "user@domain.com",
    "department": "Finance",
    "sailpoint_risk": "35",
})
```

#### Evaluation and Logging
When `llm_call` events are captured, the SDK intercepts the payload and queries the Zendata API to evaluate PII and security risks. 
To enable this, configure your environment with:
```env
ZENDATA_ENGINE_URL=http://localhost:8080/evaluate
ZENDATA_API_KEY=your_zendata_api_key
ZENDATA_EVALUATION_MODE=blocking 
```

The output is written automatically to a flat, enriched CSV format at `logs/events.csv` containing merged metrics, user context, and Zendata risk signals (`pii_detected`, `trust_boundary_crossed`, `zendata_composite_risk`, etc.).

### Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `app_id` | required | Your registered app ID |
| `api_key` | required | Your platform API key |
| `endpoint` | `https://ingest.yourplatform.com/v1/events` | Ingest URL |
| `capture_prompts` | `True` | Set `False` to redact all prompt/response content |
| `batch_size` | `20` | Max events per HTTP request |
| `flush_interval_s` | `5.0` | Background flush interval |
| `auto_patch_openai` | `False` | Auto-instrument openai globally |
| `auto_patch_anthropic` | `False` | Auto-instrument anthropic globally |
| `debug` | `False` | Print SDK debug logs |

---

## JavaScript / TypeScript SDK

### Install

```bash
npm install ai-telemetry-sdk
```

### Quickstart — OpenAI

```typescript
import OpenAI from "openai";
import { init } from "ai-telemetry-sdk";
import { wrapOpenAI } from "ai-telemetry-sdk/integrations/openai";

// 1. Initialize
const telemetry = init({
  appId: "my-ai-app",
  apiKey: "sk-telem-...",
  capturePrompts: true,
});

// 2. Wrap your client
const openai = wrapOpenAI(new OpenAI(), telemetry);

// 3. Use normally
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});
```

### Quickstart — Anthropic

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { init } from "ai-telemetry-sdk";
import { wrapAnthropic } from "ai-telemetry-sdk/integrations/anthropic";

const telemetry = init({ appId: "my-app", apiKey: "sk-telem-..." });
const anthropic = wrapAnthropic(new Anthropic(), telemetry);

const response = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello!" }],
});
```

### Quickstart — LangChain

```typescript
import { init } from "ai-telemetry-sdk";
import { TelemetryCallbackHandler } from "ai-telemetry-sdk/integrations/langchain";
import { ChatOpenAI } from "@langchain/openai";

const telemetry = init({ appId: "my-app", apiKey: "sk-telem-..." });
const handler = new TelemetryCallbackHandler(telemetry);

const llm = new ChatOpenAI({ model: "gpt-4o" });
const result = await llm.invoke("Hello!", { callbacks: [handler] });
```

### Manual Instrumentation

```typescript
import { init, traceLLM, traceTool } from "ai-telemetry-sdk";

const telemetry = init({ appId: "my-app", apiKey: "sk-telem-..." });

const myLLM = traceLLM(
  async (prompt: string) => { /* your HTTP call */ return "response"; },
  telemetry,
  { model: "llama-3", provider: "ollama" }
);

const myTool = traceTool(
  async (query: string) => { /* tool logic */ return "result"; },
  telemetry,
  { toolName: "web_search" }
);
```

### Graceful Shutdown (Node.js)

```typescript
process.on("SIGTERM", async () => {
  await telemetry.shutdown();  // flushes remaining events
  process.exit(0);
});
```

### Configuration Reference

| Option | Default | Description |
|---|---|---|
| `appId` | required | Your registered app ID |
| `apiKey` | required | Your platform API key |
| `endpoint` | `https://ingest.yourplatform.com/v1/events` | Ingest URL |
| `capturePrompts` | `true` | Set `false` to redact content |
| `batchSize` | `20` | Max events per HTTP request |
| `flushIntervalMs` | `5000` | Background flush interval |
| `debug` | `false` | Print SDK debug logs |

---

## Event Schema

Every event sent to your platform follows this schema:

```json
{
  "event_id": "uuid-v4",
  "app_id": "my-ai-app",
  "event_type": "llm_call | tool_call | error",
  "timestamp_ms": 1741200000000,

  "provider": "openai",
  "model": "gpt-4o",
  "latency_ms": 823,
  "prompt_tokens": 150,
  "completion_tokens": 45,
  "total_tokens": 195,

  "input_messages": [{"role": "user", "content": "Hello!"}],
  "output_message": {"role": "assistant", "content": "Hi there!"},

  "tool_name": "web_search",
  "tool_input": {"query": "latest AI news"},
  "tool_output": "...",

  "error_type": "RateLimitError",
  "error_message": "...",

  "metadata": {"user_id": "u_123"}
}
```

---

## Privacy

Set `capture_prompts: false` (Python) or `capturePrompts: false` (JS) to strip all prompt, response, and tool I/O content before events leave the process. Only structural metadata (model, tokens, latency, errors) is sent.

---

## License

MIT

## Framework Agnostic Design

This telemetry SDK is completely framework-agnostic and works across LangGraph, CrewAI, ADK, and custom-built agent systems. Because the SDK operates at the lowest levels (network client patching, callbacks, and context variables), it integrates seamlessly regardless of your high-level agent framework:

*   **Google ADK:** Use the `@trace_tool` and `@trace_llm` decorators on the specific tools and models the ADK orchestrates.
*   **LangGraph & LangChain:** The SDK includes a native `TelemetryCallbackHandler`. Because LangGraph is built on LangChain, you simply pass this handler into the graph's execution config, and it automatically traces all nodes, edges, LLMs, and tools.
*   **CrewAI:** CrewAI delegates to standard LLM clients (like `openai` or `anthropic`) under the hood. By initializing the SDK with `auto_patch_openai=True`, every time a CrewAI agent reaches out to an LLM, the SDK intercepts it, applies the Zendata DLP checks, and logs it—without needing to change a single line of CrewAI code.

Because `current_user_context` is a standard Python `ContextVar`, it automatically propagates down the execution stack, meaning context like the user's identity and `sailpoint_risk` remains intact whether LangGraph, CrewAI, or ADK is managing the specific thread or coroutine.



---

## Entity Classification

The SDK automatically classifies every `llm_call` event as **HUMAN**, **AUTOMATED**, **AGENT**, or **UNKNOWN** using 15 behavioral rules across 6 categories (timing, prompt structure, session behavior, API patterns, multi-agent topology, and behavioral drift).

Classification results are written onto each `TelemetryEvent` as additional fields — no extra code is needed beyond the two lines below.

### Required: inject prompt context at the request boundary

Before calling any LLM (i.e., when you set `current_user_context`), add the raw user input and the current timestamp to the context dictionary:

```python
import time
from ai_telemetry.context import current_user_context

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_data = {
        "user_id":    "usr_123",
        "email":      "user@domain.com",
        "department": "Finance",
        "sailpoint_risk": "35",
        "tools_called":   "",

        # Required for entity classification:
        "prompt_text":               request.question,          # the raw user prompt
        "prompt_submission_time_ms": int(time.time() * 1000),   # epoch ms at request receipt
    }
    current_user_context.set(user_data)

    # ... run your LLM / agent as normal ...
```

> **Why at the request boundary?**  
> The SDK needs the prompt text and submission timestamp *before* the LLM call starts so it can measure inter-turn latency (T1/T2/T3) and analyse prompt content (P1/P4/S1/M1/D3–D5). These values are not recoverable from inside the LLM call itself.

### New event fields

| Field | Type | Description |
|---|---|---|
| `entity_type` | `str` | `HUMAN` · `AUTOMATED` · `AGENT` · `UNKNOWN` |
| `entity_subtype` | `str` | `SUBAGENT_RECEIVER` · `ORCHESTRATOR_SENDER` (AGENT only) |
| `classification_signals` | `str` | Pipe-delimited rule IDs that fired (e.g. `T1_SUBMISSION_VELOCITY\|P4_IMPERATIVE_FRAMING`) |
| `behavioral_alerts` | `str` | Pipe-delimited alert names (e.g. `DRIFT_ALERT\|REGULATORY_PROBE`) |
| `session_turn_index` | `int` | 0-based turn number within the session |
| `inter_turn_latency_ms` | `int` | ms between the previous response and this submission |
| `prompt_submission_time_ms` | `int` | Epoch ms when the prompt was received by the application |

### Classification signals reference

| Rule | Signal | Indicates |
|---|---|---|
| T1 | `T1_SUBMISSION_VELOCITY` | Prompt submitted < 500 ms after session start, or < 200 ms after prior response |
| T2 | `T2_LOW_LATENCY_VARIANCE` | Inter-turn gap stddev < 200 ms across 5+ turns |
| T2 | `T2_HIGH_LATENCY_VARIANCE` | Inter-turn gap stddev > 3 000 ms (human indicator) |
| T3 | `T3_TYPING_SPEED` | > 15 characters/second (human avg is 4–6) |
| P1 | `P1_TEMPLATE_STRUCTURE` | 3+ turns share an identical structural skeleton |
| P4 | `P4_IMPERATIVE_FRAMING` | 2+ imperative command phrases in one prompt |
| S1 | `S1_WARMUP_ABSENT` | First prompt is complex with no orienting language |
| S4 | `S4_SIMULTANEOUS_SESSIONS` | Same entity active in 2+ concurrent sessions |
| A1 | `A1_HIGH_CALL_VOLUME` | > 1 000 calls from this key in the last 24 h |
| A1 | `A1_LOW_CALL_VOLUME` | < 20 calls (human indicator) |
| A2 | `A2_BURST_CALLS` | > 5 calls in any 10-second window |
| M1 | `M1_DELEGATION_LANGUAGE` | Delegation/orchestration language detected |

### Behavioral alerts reference

| Alert | Trigger |
|---|---|
| `DRIFT_ALERT` | Current risk score > 1.5 SD from entity's rolling baseline (D1) |
| `TAKEOVER_ALERT` | AUTOMATED/AGENT session suddenly shows human timing patterns (D2) |
| `ROLE_PLAY_INJECTION` | Prompt instructs the AI to adopt a different persona or identity (D3) |
| `PROGRESSIVE_PROBE` | 3 consecutive turns with escalating sensitivity toward a sensitive topic (D4) |
| `REGULATORY_PROBE` | Structuring, threshold-avoidance, or AML regulatory language detected (D5) |


---

## Log Viewer & Dashboard

The SDK writes a second log file alongside `events.csv` — a **newline-delimited JSON** file (`events.jsonl`) where each line is a complete, self-contained event object including all classification and alert fields. No extra configuration is needed; it is written automatically whenever `enable_local_log=True`.

### JSONL log location

The JSONL file is placed next to your CSV log, with the same base name:

```
logs/events.csv    ← existing CSV log
logs/events.jsonl  ← new, written automatically
```

### Add a log API endpoint to your application

Add this endpoint to your FastAPI app to expose the JSONL log over HTTP with optional filtering:

```python
import json, os
from fastapi import Query
from fastapi.responses import JSONResponse
from typing import Optional

# Derive the JSONL path from your existing log path
_JSONL_PATH = "logs/events.jsonl"   # adjust to match your log_path setting

@app.get("/api/events")
async def get_events(
    entity_type: Optional[str] = Query(None, description="HUMAN | AUTOMATED | AGENT | UNKNOWN"),
    alert:       Optional[str] = Query(None, description="e.g. REGULATORY_PROBE"),
    event_type:  Optional[str] = Query(None, description="llm_call | tool_call | error"),
    limit:       int           = Query(200, ge=1, le=2000),
):
    events = []
    if os.path.exists(_JSONL_PATH):
        with open(_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if entity_type:
        events = [e for e in events if e.get("entity_type") == entity_type]
    if alert:
        events = [e for e in events if alert in (e.get("behavioral_alerts") or "")]
    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]

    events = list(reversed(events))[:limit]   # most recent first
    return JSONResponse({"events": events, "total": len(events)})
```

### Add a dashboard endpoint

```python
from starlette.responses import FileResponse

@app.get("/dashboard")
async def get_dashboard():
    return FileResponse("templates/dashboard.html")
```

### Add the dashboard HTML

Copy `dashboard.html` from this repository's `app/templates/` directory into your application's templates folder. It is a self-contained, zero-dependency single-page app — no build step, no npm, no CDN required.

The dashboard features:
- **Live event table** — auto-refreshes every 5 seconds
- **Entity badges** — color-coded HUMAN (green) / AUTOMATED (red) / AGENT (amber)
- **Alert pills** — REGULATORY_PROBE and PROGRESSIVE_PROBE highlighted in red, others in amber
- **Signal pills** — shows every classification rule that fired per event
- **Filter bar** — filter by entity type, event type, alert name, or free-text search
- **Row detail panel** — click any row to expand and see the full raw JSON event
- **Summary counters** — live totals for events, humans, automated, agents, and alert count

### How to run

```bash
# 1. Install and activate your environment
source venv/bin/activate

# 2. (First time only) fetch data and train the predictor
python3 companies.py
python3 fetch_data.py
python3 app/models/train_predictor.py

# 3. Start the server
uvicorn app.main:app --reload --port 8080
```

| URL | What you get |
|---|---|
| `http://localhost:8080/` | Chatbot — send messages to generate telemetry events |
| `http://localhost:8080/dashboard` | Live event dashboard |
| `http://localhost:8080/api/events` | Raw JSON events (filterable) |
| `http://localhost:8080/api/events?entity_type=AUTOMATED` | Automated sessions only |
| `http://localhost:8080/api/events?alert=REGULATORY_PROBE` | Events with regulatory alert |
