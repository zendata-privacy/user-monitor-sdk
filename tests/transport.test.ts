/**
 * call_protocol: transport protocol (call type) of captured calls.
 */
import { jest } from "@jest/globals";
import { type TelemetryClient, type TelemetryEvent } from "../src/client.js";
import { wrapOpenAI } from "../src/integrations/openai.js";
import { protocolFromUrl, protocolOfClient } from "../src/transport.js";

function mockTelemetry() {
  const captured: TelemetryEvent[] = [];
  return {
    capture: jest.fn((e: TelemetryEvent) => captured.push(e)),
    captured,
  } as unknown as TelemetryClient & { captured: TelemetryEvent[] };
}

function fakeOpenAI(baseURL?: string) {
  return {
    baseURL,
    chat: { completions: { create: async () => ({ choices: [], usage: undefined }) } },
  };
}

describe("call_protocol", () => {
  test("scheme of a URL", () => {
    expect(protocolFromUrl("https://api.openai.com/v1")).toBe("https");
    expect(protocolFromUrl("http://localhost:11434")).toBe("http");
    expect(protocolFromUrl("wss://x/y")).toBe("wss");
    expect(protocolFromUrl("arn:aws:dynamodb:us-east-1:1:table/T")).toBeUndefined();
    expect(protocolFromUrl(undefined)).toBeUndefined();
  });

  test("client and LangChain-style configuration", () => {
    expect(protocolOfClient({ baseURL: "https://api.anthropic.com" })).toBe("https");
    expect(protocolOfClient({ configuration: { baseURL: "http://proxy:4000" } })).toBe("http");
    expect(protocolOfClient({})).toBeUndefined();
  });

  test("wrapOpenAI records the client's protocol", async () => {
    const t = mockTelemetry();
    const client = wrapOpenAI(fakeOpenAI("http://localhost:11434/v1"), t);
    await client.chat.completions.create({ model: "m", messages: [] });
    expect(t.captured[0].call_protocol).toBe("http");
  });

  test("no endpoint, no call_protocol", async () => {
    const t = mockTelemetry();
    const client = wrapOpenAI(fakeOpenAI(), t);
    await client.chat.completions.create({ model: "m", messages: [] });
    expect(t.captured[0].call_protocol).toBeUndefined();
  });
});
