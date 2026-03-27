/**
 * Tests for wrapOpenAI (src/integrations/openai.ts).
 *
 * Covers: llm_call capture, token extraction, output_message,
 * return value preservation, and error capture.
 */

import { jest } from "@jest/globals";
import { type TelemetryClient, type TelemetryEvent } from "../../src/client.js";
import { wrapOpenAI } from "../../src/integrations/openai.js";

function makeMockClient() {
  const captured: TelemetryEvent[] = [];
  return {
    capture: jest.fn((e: TelemetryEvent) => captured.push(e)),
    captured,
  } as unknown as TelemetryClient & { captured: TelemetryEvent[] };
}

function makeOpenAIClient(overrides: Partial<{
  response: Record<string, unknown>;
  shouldThrow: boolean;
}> = {}) {
  const defaultResponse = {
    choices: [{
      message: { role: "assistant", content: "Hello from GPT", tool_calls: undefined },
    }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };

  const response = { ...defaultResponse, ...overrides.response };

  return {
    chat: {
      completions: {
        create: jest.fn(async (..._args: unknown[]) => {
          if (overrides.shouldThrow) throw new Error("rate limited");
          return response;
        }),
      },
    },
  };
}

describe("wrapOpenAI()", () => {
  it("captures llm_call event", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient(), client);

    await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });

    const e = client.captured[0];
    expect(e.event_type).toBe("llm_call");
    expect(e.provider).toBe("openai");
    expect(e.model).toBe("gpt-4o");
  });

  it("extracts token usage", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient(), client);

    await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });

    const e = client.captured[0];
    expect(e.prompt_tokens).toBe(10);
    expect(e.completion_tokens).toBe(5);
    expect(e.total_tokens).toBe(15);
  });

  it("captures input_messages", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient(), client);
    const msgs = [{ role: "user", content: "tell me a joke" }];

    await openai.chat.completions.create({ model: "gpt-4o", messages: msgs });

    expect(client.captured[0].input_messages).toEqual(msgs);
  });

  it("captures output_message from first choice", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient(), client);

    await openai.chat.completions.create({ model: "gpt-4o", messages: [] });

    const msg = client.captured[0].output_message;
    expect(msg?.role).toBe("assistant");
    expect(msg?.content).toBe("Hello from GPT");
  });

  it("preserves return value unchanged", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient(), client);

    const result = await openai.chat.completions.create({ model: "gpt-4o", messages: [] });
    expect((result as Record<string, unknown>).choices).toBeDefined();
  });

  it("captures error event and re-throws", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient({ shouldThrow: true }), client);

    await expect(
      openai.chat.completions.create({ model: "gpt-4o", messages: [] })
    ).rejects.toThrow("rate limited");

    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.error_message).toBe("rate limited");
  });

  it("always fires capture even on error", async () => {
    const client = makeMockClient();
    const openai = wrapOpenAI(makeOpenAIClient({ shouldThrow: true }), client);
    await openai.chat.completions.create({}).catch(() => {});
    expect(client.captured).toHaveLength(1);
  });

  it("measures latency_ms", async () => {
    const client = makeMockClient();
    const slowOpenAI = {
      chat: {
        completions: {
          create: jest.fn(async (..._args: unknown[]) => {
            await new Promise(r => setTimeout(r, 30));
            return makeOpenAIClient().chat.completions.create();
          }),
        },
      },
    };
    const openai = wrapOpenAI(slowOpenAI, client);
    await openai.chat.completions.create({ model: "gpt-4o", messages: [] });
    expect(client.captured[0].latency_ms).toBeGreaterThanOrEqual(30);
  });
});
