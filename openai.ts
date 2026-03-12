/**
 * AI Telemetry SDK — OpenAI Integration (TypeScript)
 *
 * Wraps an OpenAI client instance so every chat.completions.create call
 * is automatically traced.
 *
 * Usage:
 *   import OpenAI from "openai";
 *   import { wrapOpenAI } from "ai-telemetry-sdk/integrations/openai";
 *
 *   const openai = wrapOpenAI(new OpenAI(), telemetryClient);
 */

import type { TelemetryClient } from "../client.js";

type AnyOpenAI = {
  chat: {
    completions: {
      create: (...args: unknown[]) => Promise<unknown>;
    };
  };
};

export function wrapOpenAI<T extends AnyOpenAI>(
  openai: T,
  telemetry: TelemetryClient
): T {
  const originalCreate = openai.chat.completions.create.bind(
    openai.chat.completions
  );

  openai.chat.completions.create = async function (...args: unknown[]) {
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
        provider: "openai",
        model,
        latency_ms,
        input_messages: messages,
        ...errorInfo,
      };

      if (response) {
        const r = response as {
          choices?: Array<{
            message: {
              role: string;
              content: string;
              tool_calls?: unknown[];
            };
          }>;
          usage?: {
            prompt_tokens: number;
            completion_tokens: number;
            total_tokens: number;
          };
        };

        const choice = r.choices?.[0];
        if (choice) {
          Object.assign(event, {
            output_message: {
              role: choice.message.role,
              content: choice.message.content,
              tool_calls: choice.message.tool_calls,
            },
          });
        }
        if (r.usage) {
          Object.assign(event, {
            prompt_tokens: r.usage.prompt_tokens,
            completion_tokens: r.usage.completion_tokens,
            total_tokens: r.usage.total_tokens,
          });
        }
      }

      telemetry.capture(event);
    }
  } as typeof originalCreate;

  return openai;
}
