/**
 * Tests for TelemetryClient (src/client.ts).
 *
 * Covers: capture, flush, queue overflow, prompt redaction,
 * trace context auto-population, HTTP retry / circuit breaker, shutdown.
 */

import { jest } from "@jest/globals";
import { TelemetryClient, type TelemetryEvent } from "../src/client.js";
import { _traceStorage } from "../src/context.js";

// ---------------------------------------------------------------------------
// Mock global fetch so no real HTTP calls are made
// ---------------------------------------------------------------------------

const mockFetch = jest.fn();
(global as unknown as { fetch: typeof mockFetch }).fetch = mockFetch;

function okResponse(): Response {
  return { ok: true, status: 200 } as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
  mockFetch.mockResolvedValue(okResponse());
});

function makeClient(overrides: Partial<ConstructorParameters<typeof TelemetryClient>[0]> = {}) {
  return new TelemetryClient({
    appId: "test-app",
    apiKey: "sk-test",
    endpoint: "http://localhost:9999/v1/events",
    flushIntervalMs: 999_999, // disable auto-flush
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// capture()
// ---------------------------------------------------------------------------

describe("capture()", () => {
  it("enriches event with event_id, app_id, timestamp_ms", async () => {
    const client = makeClient();
    const event: TelemetryEvent = { event_type: "llm_call" };
    client.capture(event);

    // peek at queue via flush
    mockFetch.mockResolvedValue(okResponse());
    await client.flush();

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    const sent = body.events[0] as TelemetryEvent;
    expect(sent.event_id).toBeDefined();
    expect(sent.app_id).toBe("test-app");
    expect(sent.timestamp_ms).toBeGreaterThan(0);
    await client.shutdown();
  });

  it("auto-populates trace_id from AsyncLocalStorage", async () => {
    const client = makeClient();
    let captured: TelemetryEvent | undefined;

    _traceStorage.run({ traceId: "trace-abc", spanId: "" }, () => {
      client.capture({ event_type: "llm_call" });
    });

    await client.flush();
    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    captured = body.events[0] as TelemetryEvent;
    expect(captured.trace_id).toBe("trace-abc");
    await client.shutdown();
  });

  it("does not overwrite explicit trace_id", async () => {
    const client = makeClient();
    _traceStorage.run({ traceId: "from-storage", spanId: "" }, () => {
      client.capture({ event_type: "llm_call", trace_id: "pre-set" });
    });

    await client.flush();
    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.events[0].trace_id).toBe("pre-set");
    await client.shutdown();
  });

  it("assigns a span_id", async () => {
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.flush();
    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.events[0].span_id).toBeTruthy();
    await client.shutdown();
  });

  it("strips content fields when capturePrompts=false", async () => {
    const client = makeClient({ capturePrompts: false });
    client.capture({
      event_type: "llm_call",
      input_messages: [{ role: "user", content: "secret" }],
      output_message: { role: "assistant", content: "answer" },
    });
    await client.flush();
    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    const sent = body.events[0] as TelemetryEvent;
    expect(sent.input_messages).toBeUndefined();
    expect(sent.output_message).toBeUndefined();
    await client.shutdown();
  });
});

// ---------------------------------------------------------------------------
// Queue overflow
// ---------------------------------------------------------------------------

describe("Queue overflow", () => {
  it("drops oldest event when queue is full (shift behavior)", async () => {
    const client = makeClient({ maxQueueSize: 3 });
    for (let i = 0; i < 5; i++) {
      client.capture({ event_type: "llm_call", metadata: { i } });
    }
    // Queue should hold at most maxQueueSize items
    expect((client as unknown as { queue: TelemetryEvent[] }).queue.length).toBeLessThanOrEqual(3);
    await client.shutdown();
  });

  it("does not throw when overflow occurs", async () => {
    const client = makeClient({ maxQueueSize: 2 });
    expect(() => {
      for (let i = 0; i < 10; i++) {
        client.capture({ event_type: "llm_call" });
      }
    }).not.toThrow();
    await client.shutdown();
  });
});

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------

describe("flush()", () => {
  it("sends all queued events in one request", async () => {
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    client.capture({ event_type: "tool_call" });
    await client.flush();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.events).toHaveLength(2);
    await client.shutdown();
  });

  it("is a no-op when queue is empty", async () => {
    const client = makeClient();
    await client.flush();
    expect(mockFetch).not.toHaveBeenCalled();
    await client.shutdown();
  });

  it("clears the queue after flushing", async () => {
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect((client as unknown as { queue: TelemetryEvent[] }).queue).toHaveLength(0);
    await client.shutdown();
  });
});

// ---------------------------------------------------------------------------
// HTTP error handling / circuit breaker
// ---------------------------------------------------------------------------

describe("HTTP error handling", () => {
  it("4xx does not increment consecutiveFailures", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 401 } as Response);
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect((client as unknown as { consecutiveFailures: number }).consecutiveFailures).toBe(0);
    await client.shutdown();
  });

  it("5xx increments consecutiveFailures", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 503 } as Response);
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect(
      (client as unknown as { consecutiveFailures: number }).consecutiveFailures
    ).toBeGreaterThan(0);
    await client.shutdown();
  });

  it("network error increments consecutiveFailures", async () => {
    mockFetch.mockRejectedValue(new Error("ECONNREFUSED"));
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect(
      (client as unknown as { consecutiveFailures: number }).consecutiveFailures
    ).toBeGreaterThan(0);
    await client.shutdown();
  });

  it("circuit opens after failure threshold", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 503 } as Response);
    const client = makeClient();
    (client as unknown as { consecutiveFailures: number }).consecutiveFailures = 4;
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect(
      (client as unknown as { circuitOpenUntil: number }).circuitOpenUntil
    ).toBeGreaterThan(Date.now());
    await client.shutdown();
  });

  it("skips send when circuit is open", async () => {
    const client = makeClient();
    (client as unknown as { circuitOpenUntil: number }).circuitOpenUntil = Date.now() + 60_000;
    client.capture({ event_type: "llm_call" });
    await client.flush();
    expect(mockFetch).not.toHaveBeenCalled();
    await client.shutdown();
  });
});

// ---------------------------------------------------------------------------
// shutdown()
// ---------------------------------------------------------------------------

describe("shutdown()", () => {
  it("flushes pending events before stopping", async () => {
    const client = makeClient();
    client.capture({ event_type: "llm_call" });
    await client.shutdown();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("clears the flush timer", async () => {
    const client = makeClient();
    await client.shutdown();
    expect((client as unknown as { flushTimer: NodeJS.Timeout | null }).flushTimer).toBeNull();
  });
});
