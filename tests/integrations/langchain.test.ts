/**
 * Tests for TelemetryCallbackHandler (src/integrations/langchain.ts).
 *
 * Covers: LLM start/end, chat model start, LLM error,
 * tool start/end, tool error, latency, and stale entry cleanup.
 */

import { jest } from "@jest/globals";
import { type TelemetryClient, type TelemetryEvent } from "../../src/client.js";
import { TelemetryCallbackHandler } from "../../src/integrations/langchain.js";

function makeMockClient() {
  const captured: TelemetryEvent[] = [];
  return {
    capture: jest.fn((e: TelemetryEvent) => captured.push(e)),
    captured,
  } as unknown as TelemetryClient & { captured: TelemetryEvent[] };
}

function makeLLMResult(overrides: {
  text?: string;
  modelName?: string;
  promptTokens?: number;
  completionTokens?: number;
} = {}) {
  return {
    llm_output: {
      model_name: overrides.modelName ?? "gpt-4o",
      token_usage: {
        prompt_tokens: overrides.promptTokens ?? 10,
        completion_tokens: overrides.completionTokens ?? 5,
        total_tokens: (overrides.promptTokens ?? 10) + (overrides.completionTokens ?? 5),
      },
    },
    generations: [[{ text: overrides.text ?? "output text" }]],
  };
}

// ---------------------------------------------------------------------------
// LLM hooks
// ---------------------------------------------------------------------------

describe("LLM hooks", () => {
  it("handleLLMEnd captures llm_call event", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-1";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMEnd(makeLLMResult(), runId);

    const e = client.captured[0];
    expect(e.event_type).toBe("llm_call");
    expect(e.provider).toBe("langchain");
    expect(e.model).toBe("gpt-4o");
  });

  it("handleLLMEnd extracts token usage", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-2";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMEnd(
      makeLLMResult({ promptTokens: 20, completionTokens: 8 }),
      runId,
    );

    const e = client.captured[0];
    expect(e.prompt_tokens).toBe(20);
    expect(e.completion_tokens).toBe(8);
    expect(e.total_tokens).toBe(28);
  });

  it("handleLLMEnd captures output text", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-3";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMEnd(makeLLMResult({ text: "The sky is blue" }), runId);

    expect(client.captured[0].output_message?.content).toBe("The sky is blue");
  });

  it("handleChatModelStart + handleLLMEnd captures llm_call", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-4";

    handler.handleChatModelStart({}, [[]], runId);
    handler.handleLLMEnd(makeLLMResult(), runId);

    expect(client.captured[0].event_type).toBe("llm_call");
  });

  it("handleLLMEnd cleans up start time entry", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-5";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMEnd(makeLLMResult(), runId);

    expect(handler["startTimes"].has(runId)).toBe(false);
  });

  it("handleLLMError captures error event", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-6";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMError(new RangeError("context window exceeded"), runId);

    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.provider).toBe("langchain");
    expect(e.error_type).toBe("RangeError");
    expect(e.error_message).toContain("context window");
  });

  it("handleLLMError cleans up start time entry", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "run-7";

    handler.handleLLMStart({}, ["prompt"], runId);
    handler.handleLLMError(new Error("fail"), runId);

    expect(handler["startTimes"].has(runId)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Tool hooks
// ---------------------------------------------------------------------------

describe("Tool hooks", () => {
  it("handleToolEnd captures tool_call event", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "tool-run-1";

    handler.handleToolStart({}, "query input", runId, undefined, undefined, undefined, "web_search");
    handler.handleToolEnd("search results", runId);

    const e = client.captured[0];
    expect(e.event_type).toBe("tool_call");
    expect(e.tool_name).toBe("web_search");
    expect(e.tool_output).toBe("search results");
  });

  it("handleToolEnd captures tool_input", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "tool-run-2";

    handler.handleToolStart({}, "my input", runId, undefined, undefined, undefined, "search");
    handler.handleToolEnd("output", runId);

    expect(client.captured[0].tool_input).toEqual({ input: "my input" });
  });

  it("handleToolEnd cleans up entries", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "tool-run-3";

    handler.handleToolStart({}, "input", runId);
    handler.handleToolEnd("output", runId);

    expect(handler["startTimes"].has(`tool_${runId}`)).toBe(false);
    expect(handler["toolInputs"].has(`tool_${runId}`)).toBe(false);
  });

  it("handleToolError captures error event", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "tool-run-4";

    handler.handleToolStart({}, "input", runId);
    handler.handleToolError(new TypeError("bad input"), runId);

    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.error_type).toBe("TypeError");
  });

  it("handleToolError cleans up start time entry", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);
    const runId = "tool-run-5";

    handler.handleToolStart({}, "input", runId);
    handler.handleToolError(new Error("fail"), runId);

    expect(handler["startTimes"].has(`tool_${runId}`)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Stale entry cleanup
// ---------------------------------------------------------------------------

describe("Stale entry cleanup", () => {
  it("removes entries older than 5 minutes during handleLLMStart", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);

    const staleKey = "stale-key";
    handler["startTimes"].set(staleKey, Date.now() - 6 * 60 * 1000); // 6 min ago

    handler.handleLLMStart({}, ["prompt"], "new-run");

    expect(handler["startTimes"].has(staleKey)).toBe(false);
  });

  it("keeps entries younger than 5 minutes", () => {
    const client = makeMockClient();
    const handler = new TelemetryCallbackHandler(client);

    const recentKey = "recent-key";
    handler["startTimes"].set(recentKey, Date.now() - 30_000); // 30s ago

    handler.handleLLMStart({}, ["prompt"], "new-run");

    expect(handler["startTimes"].has(recentKey)).toBe(true);
  });
});
