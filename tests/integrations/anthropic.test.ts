/**
 * Tests for wrapAnthropic (src/integrations/anthropic.ts).
 *
 * Covers: llm_call capture, text/tool block extraction,
 * input_tokens/output_tokens, return value preservation, error capture.
 */

import { jest } from "@jest/globals";
import { type TelemetryClient, type TelemetryEvent } from "../../src/client.js";
import { wrapAnthropic } from "../../src/integrations/anthropic.js";

function makeMockClient() {
  const captured: TelemetryEvent[] = [];
  return {
    capture: jest.fn((e: TelemetryEvent) => captured.push(e)),
    captured,
  } as unknown as TelemetryClient & { captured: TelemetryEvent[] };
}

function makeTextBlock(text = "Hello from Claude") {
  return { type: "text", text };
}

function makeToolBlock(id = "tool_1", name = "get_weather", input = { city: "NYC" }) {
  return { type: "tool_use", id, name, input };
}

function makeAnthropicClient(overrides: {
  content?: Array<Record<string, unknown>>;
  usage?: { input_tokens: number; output_tokens: number };
  shouldThrow?: boolean;
} = {}) {
  const response = {
    content: overrides.content ?? [makeTextBlock()],
    usage: overrides.usage ?? { input_tokens: 8, output_tokens: 4 },
  };

  return {
    messages: {
      create: jest.fn(async () => {
        if (overrides.shouldThrow) throw new Error("overloaded");
        return response;
      }),
    },
  };
}

describe("wrapAnthropic()", () => {
  it("captures llm_call event", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(makeAnthropicClient(), client);

    await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      messages: [{ role: "user", content: "hi" }],
    });

    const e = client.captured[0];
    expect(e.event_type).toBe("llm_call");
    expect(e.provider).toBe("anthropic");
    expect(e.model).toBe("claude-3-5-sonnet-20241022");
  });

  it("extracts input_tokens and output_tokens", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(
      makeAnthropicClient({ usage: { input_tokens: 12, output_tokens: 6 } }),
      client,
    );

    await anthropic.messages.create({ model: "claude-3-5-sonnet-20241022", messages: [] });

    const e = client.captured[0];
    expect(e.prompt_tokens).toBe(12);
    expect(e.completion_tokens).toBe(6);
    expect(e.total_tokens).toBe(18);
  });

  it("extracts text block content", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(
      makeAnthropicClient({ content: [makeTextBlock("Paris is beautiful")] }),
      client,
    );

    await anthropic.messages.create({ model: "claude-3-5-sonnet-20241022", messages: [] });

    expect(client.captured[0].output_message?.content).toBe("Paris is beautiful");
  });

  it("extracts tool_use blocks", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(
      makeAnthropicClient({
        content: [makeToolBlock("t1", "search", { query: "test" })],
      }),
      client,
    );

    await anthropic.messages.create({ model: "claude-3-5-sonnet-20241022", messages: [] });

    const toolCalls = client.captured[0].output_message?.tool_calls as Array<Record<string, unknown>> | undefined;
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls![0].name).toBe("search");
    expect(toolCalls![0].id).toBe("t1");
  });

  it("captures input_messages", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(makeAnthropicClient(), client);
    const msgs = [{ role: "user", content: "what is 2+2?" }];

    await anthropic.messages.create({ model: "claude-3-5-sonnet-20241022", messages: msgs });

    expect(client.captured[0].input_messages).toEqual(msgs);
  });

  it("preserves return value unchanged", async () => {
    const client = makeMockClient();
    const rawClient = makeAnthropicClient();
    const anthropic = wrapAnthropic(rawClient, client);

    const result = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      messages: [],
    });
    expect((result as Record<string, unknown>).content).toBeDefined();
  });

  it("captures error event and re-throws", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(makeAnthropicClient({ shouldThrow: true }), client);

    await expect(
      anthropic.messages.create({ model: "claude-3-5-sonnet-20241022", messages: [] })
    ).rejects.toThrow("overloaded");

    const e = client.captured[0];
    expect(e.event_type).toBe("error");
    expect(e.error_message).toBe("overloaded");
  });

  it("always fires capture even on error", async () => {
    const client = makeMockClient();
    const anthropic = wrapAnthropic(makeAnthropicClient({ shouldThrow: true }), client);
    await anthropic.messages.create({}).catch(() => {});
    expect(client.captured).toHaveLength(1);
  });
});
