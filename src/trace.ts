/**
 * AI Telemetry SDK — Generic Tracing Utilities (TypeScript)
 *
 * Use these when you want to manually instrument any async function.
 *
 * Usage:
 *   import { traceLLM, traceTool } from "ai-telemetry-sdk/trace";
 */

import type { TelemetryClient } from "./client.js";

interface TraceLLMOptions {
  model?: string;
  provider?: string;
}

/** Wrap any async LLM-calling function and capture telemetry. */
export function traceLLM<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  telemetry: TelemetryClient,
  opts: TraceLLMOptions = {}
): T {
  return (async function (...args: unknown[]) {
    const start = Date.now();
    let result: unknown;
    let errorInfo: Record<string, string> = {};

    try {
      result = await fn(...args);
      return result;
    } catch (err: unknown) {
      const e = err as Error;
      errorInfo = {
        error_type: e.constructor?.name ?? "Error",
        error_message: e.message,
        error_stack: e.stack ?? "",
      };
      throw err;
    } finally {
      telemetry.capture({
        event_type: Object.keys(errorInfo).length ? "error" : "llm_call",
        model: opts.model ?? "unknown",
        provider: opts.provider ?? "generic",
        latency_ms: Date.now() - start,
        output_message: result
          ? { role: "assistant", content: String(result) }
          : undefined,
        metadata: { function: fn.name },
        ...errorInfo,
      });
    }
  }) as T;
}

interface TraceToolOptions {
  toolName?: string;
}

/** Wrap any async tool/function call and capture telemetry. */
export function traceTool<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  telemetry: TelemetryClient,
  opts: TraceToolOptions = {}
): T {
  return (async function (...args: unknown[]) {
    const name = opts.toolName ?? fn.name ?? "unknown";
    const start = Date.now();
    let result: unknown;
    let errorInfo: Record<string, string> = {};

    try {
      result = await fn(...args);
      return result;
    } catch (err: unknown) {
      const e = err as Error;
      errorInfo = {
        error_type: e.constructor?.name ?? "Error",
        error_message: e.message,
        error_stack: e.stack ?? "",
      };
      throw err;
    } finally {
      telemetry.capture({
        event_type: Object.keys(errorInfo).length ? "error" : "tool_call",
        tool_name: name,
        tool_input: { args },
        tool_output: result,
        latency_ms: Date.now() - start,
        ...errorInfo,
      });
    }
  }) as T;
}
