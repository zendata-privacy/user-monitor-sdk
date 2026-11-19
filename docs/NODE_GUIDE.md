# Zendata AI Telemetry SDK — Node.js / TypeScript

**ZD-W5-06 · diligence** · Verified against `src/` on 2026-09-16

The Python half of this SDK has 1,048 lines of documentation. The Node half had
**one** `npm install` mention — for a complete TypeScript implementation with 70
passing tests at 94% line coverage. This closes that gap.

An evaluator assessing a dual-language SDK will try the Node path. It works; it
simply was not written down.

---

## Install

```bash
npm install ai-telemetry-sdk
```

The package is **proprietary** — `package.json` declares `"license": "UNLICENSED"`
and `"private": true`. It is distributed under agreement, not from the public npm
registry.

**Requirements:** Node 18+. The package is ESM (`"type": "module"`); see
[Running the tests](#running-the-tests) for the one consequence that matters.

---

## Quick start

```ts
import { init } from "ai-telemetry-sdk";

const telemetry = init({
  appId: "my-app",
  apiKey: process.env.ZENDATA_API_KEY!,
});

telemetry.capture({
  event_type: "llm_call",
  model: "gpt-4o",
  provider: "openai",
  input_messages: [{ role: "user", content: "Summarise this contract" }],
  output_message: { role: "assistant", content: "..." },
  prompt_tokens: 120,
  completion_tokens: 35,
});

await telemetry.shutdown();   // flushes anything still queued
```

`capture()` is the interceptor entry point — the same name as the Python
`TelemetryClient.capture()`. **There is no `intercept()` method** in either
language, despite what the diligence checklist's benchmark snippet suggests.

---

## Configuration

`init()` and `new TelemetryClient()` take the same options.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `appId` | `string` | — **required** | Identifies the application |
| `apiKey` | `string` | — **required** | Ingest credential |
| `endpoint` | `string` | `https://ingest.zendata.dev/v1/events` | Ingest URL |
| `batchSize` | `number` | `20` | Events per flush |
| `flushIntervalMs` | `number` | `5000` | Background flush cadence |
| `maxQueueSize` | `number` | `1000` | Queue cap; beyond this, events are dropped rather than growing memory without bound |
| `capturePrompts` | `boolean` | `true` | Set `false` to omit prompt and completion text |
| `debug` | `boolean` | `false` | Log SDK activity to console |

Two behaviours worth knowing before you deploy:

- **`capture()` is non-blocking.** It enqueues and returns; it never awaits the
  network. That is what makes the p99 latency figure meaningful, and it means a
  slow or unreachable ingest endpoint cannot stall your request path.
- **There is a circuit breaker.** After 5 consecutive flush failures the client
  stops attempting delivery for a cooldown period. A telemetry outage degrades
  to dropped events, not to a degraded application.

### Suppressing prompt content

```ts
const telemetry = init({
  appId: "my-app",
  apiKey: process.env.ZENDATA_API_KEY!,
  capturePrompts: false,     // metadata and token counts only
});
```

Use this where prompts may carry regulated data and you want telemetry without
the payload.

---

## Automatic instrumentation

Wrap a provider client and calls are captured without further code:

```ts
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { init } from "ai-telemetry-sdk";
import { wrapOpenAI } from "ai-telemetry-sdk/integrations/openai";
import { wrapAnthropic } from "ai-telemetry-sdk/integrations/anthropic";

const telemetry = init({ appId: "my-app", apiKey: process.env.ZENDATA_API_KEY! });

const openai = wrapOpenAI(new OpenAI(), telemetry);
const anthropic = wrapAnthropic(new Anthropic(), telemetry);
```

`init()` also accepts `autoWrapOpenAI` and `autoWrapAnthropic` if you would
rather not wrap explicitly.

The package declares subpath exports for `./integrations/openai`,
`./integrations/anthropic`, `./integrations/langchain` and `./trace`, so these
imports resolve without reaching into `dist/`. `traceLLM` and `traceTool` are
also re-exported from the package root; the `wrap*` helpers are not, and must be
imported from their subpath.

For LangChain, import the callback handler from its own subpath:

```ts
import { TelemetryCallbackHandler } from "ai-telemetry-sdk/integrations/langchain";

const handler = new TelemetryCallbackHandler(telemetry);
await chain.invoke(input, { callbacks: [handler] });
```

It reports `provider: "langchain"` from both the completion and the error hook —
the Python side was corrected to match in ZD-W3-07, where `on_llm_end` had
hardcoded `"openai"` and mislabelled every non-OpenAI model reached through a
chain.

---

## Manual tracing

`traceLLM` and `traceTool` wrap an async function and emit an event per call,
linking events into a trace via `trace_id` / `span_id` / `parent_span_id`:

```ts
import { traceLLM, traceTool } from "ai-telemetry-sdk";

const answer   = await traceLLM(callModel,  telemetry)(prompt);
const rowCount = await traceTool(queryDatabase, telemetry)(sql);
```

---

## Event shape

`TelemetryEvent` has three `event_type` values — `"llm_call"`, `"tool_call"` and
`"error"` — and all other fields are optional.

| Group | Fields |
|---|---|
| Identity | `event_id`, `app_id`, `timestamp_ms` |
| LLM | `model`, `provider`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `input_messages`, `output_message`, `latency_ms` |
| Tool | `tool_name`, `tool_input`, `tool_output` |
| Error | `error_type`, `error_message`, `error_stack` |
| Trace | `trace_id`, `span_id`, `parent_span_id` |
| Transport | `call_protocol` — scheme of the endpoint the call went to (`https`, `http`, …), from the client's `baseURL`; undefined when unknown |
| Custom | `metadata` |

---

## Shutdown

```ts
await telemetry.shutdown();
```

Flushes the queue and clears the background timer. Call it on `SIGTERM` in a
long-running service, and before process exit in a short-lived one — otherwise
queued events are lost.

---

## Running the tests

```bash
npm ci        # not `npm install` — node_modules is no longer vendored
npm test      # 70 tests, 6 suites
```

**Use `npm test`, not `npx jest`.** The package sets `"type": "module"`, so jest
needs `--experimental-vm-modules`, which the `test` script already supplies. A
bare `npx jest` fails on ESM with an unhelpful error.

```bash
npm run build     # tsc -> dist/
npm run dev       # tsc --watch
```

---

## Parity with the Python SDK

| | Python | TypeScript |
|---|---|---|
| Entry point | `TelemetryClient.capture(event)` | `telemetry.capture(event)` |
| Non-blocking capture | yes | yes |
| Circuit breaker | yes | yes |
| OpenAI / Anthropic / LangChain | yes | yes |
| Scope Reference Model (policy layer) | **yes** | not implemented |
| Tests | 382 passing, 47% coverage | 70 passing, 94% coverage |

The one real asymmetry is the **Scope Reference Model** — `scope_checker.py` and
`scope_runtime.py` exist only in Python. A Node application gets telemetry and
tracing, not runtime scope enforcement. That is a genuine gap, stated here rather
than left to be discovered.
