/**
 * AI Telemetry SDK — Anthropic Integration (TypeScript)
 *
 * Wraps an Anthropic client instance so every messages.create call
 * is automatically traced.
 *
 * Usage:
 *   import Anthropic from "@anthropic-ai/sdk";
 *   import { wrapAnthropic } from "ai-telemetry-sdk/integrations/anthropic";
 *
 *   const anthropic = wrapAnthropic(new Anthropic(), telemetryClient);
 */

import type { TelemetryClient } from "../client.js";

type AnyAnthropic = {
  messages: {
    create: (...args: unknown[]) => Promise<unknown>;
  };
};

export function wrapAnthropic<T extends AnyAnthropic>(
  anthropic: T,
  telemetry: TelemetryClient
): T {
  const originalCreate = anthropic.messages.create.bind(anthropic.messages);

  anthropic.messages.create = async function (...args: unknown[]) {
    const [params] = args as [Record<string, unknown>];
    const start = Date.now();
    let response: Record<string, unknown> | undefined;
    let errorInfo: Record<string, string> = {};

    try {
      response = (await originalCreate(...args)) as Record<string, unknown>;
      return response;
    } catch (err: unknown) {
      const e = err as Error;
      errorInfo = {
        error_type: e.constructor?.name ?? "Error",
        error_message: e.message,
        error_stack: e.stack ?? "",
      };
      throw err;
    } finally {
      const latency_ms = Date.now() - start;
      const messages = (params?.messages ?? []) as Array<{
        role: string;
        content: string;
      }>;
      const model = (params?.model as string) ?? "unknown";

      const event = {
        event_type: Object.keys(errorInfo).length ? ("error" as const) : ("llm_call" as const),
        provider: "anthropic",
        model,
        latency_ms,
        input_messages: messages,
        ...errorInfo,
      };

      if (response) {
        const r = response as {
          content?: Array<{ type: string; text?: string; id?: string; name?: string; input?: unknown }>;
          usage?: { input_tokens: number; output_tokens: number };
        };

        const textBlock = r.content?.find((b) => b.type === "text");
        const toolBlocks = r.content?.filter((b) => b.type === "tool_use") ?? [];

        Object.assign(event, {
          output_message: {
            role: "assistant",
            content: textBlock?.text ?? null,
            tool_calls: toolBlocks.length
              ? toolBlocks.map((tb) => ({
                  id: tb.id,
                  name: tb.name,
                  input: tb.input,
                }))
              : undefined,
          },
        });

        if (r.usage) {
          Object.assign(event, {
            prompt_tokens: r.usage.input_tokens,
            completion_tokens: r.usage.output_tokens,
            total_tokens: r.usage.input_tokens + r.usage.output_tokens,
          });
        }
      }

      telemetry.capture(event);
    }
  } as typeof originalCreate;

  return anthropic;
}
