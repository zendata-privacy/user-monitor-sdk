/**
 * Tests for traceLLM and traceTool (src/trace.ts).
 *
 * Covers: return value preservation, event field capture,
 * error capture, latency, and function name defaulting.
 */

import { jest } from "@jest/globals";
import { TelemetryClient, type TelemetryEvent } from "../src/client.js";
import { traceLLM, traceTool } from "../src/trace.js";

// ---------------------------------------------------------------------------
// Shared mock client
// ---------------------------------------------------------------------------

function makeMockClient() {
  const captured: TelemetryEvent[] = [];
  const client = {
    capture: jest.fn((e: TelemetryEvent) => captured.push(e)),
    captured,
  } as unknown as TelemetryClient & { captured: TelemetryEvent[] };
  return client;
}

// ---------------------------------------------------------------------------
// traceLLM
// ---------------------------------------------------------------------------

describe("traceLLM()", () => {
  it("preserves the original return value", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => "hello", client);
    expect(await fn()).toBe("hello");
  });

  it("captures llm_call event", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => "ok", client, { model: "gpt-4o", provider: "openai" });
    await fn();
    expect(client.captured[0].event_type).toBe("llm_call");
    expect(client.captured[0].model).toBe("gpt-4o");
    expect(client.captured[0].provider).toBe("openai");
  });

  it("captures output_message with result", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => "the answer", client);
    await fn();
    expect(client.captured[0].output_message).toEqual({
      role: "assistant",
      content: "the answer",
    });
  });

  it("measures latency_ms", async () => {
    const client = makeMockClient();
    const fn = traceLLM(
      async () => { await new Promise(r => setTimeout(r, 30)); return "ok"; },
      client,
    );
    await fn();
    // setTimeout(30) can fire a little early and Date.now() has 1 ms
    // resolution, so 29 ms is a correct measurement of a 30 ms sleep.
    expect(client.captured[0].latency_ms).toBeGreaterThanOrEqual(25);
  });

  it("captures error event and re-throws", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => { throw new RangeError("out of range"); }, client);
    await expect(fn()).rejects.toThrow("out of range");
    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.error_type).toBe("RangeError");
    expect(e.error_message).toBe("out of range");
  });

  it("defaults model to 'unknown' and provider to 'generic'", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => "ok", client);
    await fn();
    expect(client.captured[0].model).toBe("unknown");
    expect(client.captured[0].provider).toBe("generic");
  });

  it("always fires capture even when function throws", async () => {
    const client = makeMockClient();
    const fn = traceLLM(async () => { throw new Error("fail"); }, client);
    await fn().catch(() => {});
    expect(client.captured).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// traceTool
// ---------------------------------------------------------------------------

describe("traceTool()", () => {
  it("preserves the original return value", async () => {
    const client = makeMockClient();
    const fn = traceTool(async () => "result", client);
    expect(await fn()).toBe("result");
  });

  it("captures tool_call event", async () => {
    const client = makeMockClient();
    const fn = traceTool(async () => "r", client, { toolName: "web_search" });
    await fn();
    expect(client.captured[0].event_type).toBe("tool_call");
    expect(client.captured[0].tool_name).toBe("web_search");
  });

  it("uses function name when toolName not provided", async () => {
    const client = makeMockClient();
    async function mySpecialTool() { return "ok"; }
    const fn = traceTool(mySpecialTool, client);
    await fn();
    expect(client.captured[0].tool_name).toBe("mySpecialTool");
  });

  it("captures tool_output", async () => {
    const client = makeMockClient();
    const fn = traceTool(async () => ({ data: 42 }), client, { toolName: "compute" });
    await fn();
    expect(client.captured[0].tool_output).toEqual({ data: 42 });
  });

  it("measures latency_ms", async () => {
    const client = makeMockClient();
    const fn = traceTool(
      async () => { await new Promise(r => setTimeout(r, 30)); return "r"; },
      client,
    );
    await fn();
    // setTimeout(30) can fire a little early and Date.now() has 1 ms
    // resolution, so 29 ms is a correct measurement of a 30 ms sleep.
    expect(client.captured[0].latency_ms).toBeGreaterThanOrEqual(25);
  });

  it("captures error event and re-throws", async () => {
    const client = makeMockClient();
    const fn = traceTool(
      async () => { throw new TypeError("bad input"); },
      client,
      { toolName: "bad_tool" },
    );
    await expect(fn()).rejects.toThrow("bad input");
    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.error_type).toBe("TypeError");
  });

  it("always fires capture even when function throws", async () => {
    const client = makeMockClient();
    const fn = traceTool(async () => { throw new Error("fail"); }, client);
    await fn().catch(() => {});
    expect(client.captured).toHaveLength(1);
  });
});
