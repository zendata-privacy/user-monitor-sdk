/**
 * AI Telemetry SDK — Trace Context
 *
 * Uses AsyncLocalStorage so trace/span IDs propagate automatically
 * through async call chains without manual threading.
 *
 * Usage:
 *   import { TraceContext } from "ai-telemetry-sdk";
 *
 *   // At the start of each user interaction:
 *   TraceContext.startTrace(() => {
 *     // all captures inside here share the same trace_id
 *     await handleUserRequest();
 *   });
 */

import { randomUUID } from "node:crypto";
import { AsyncLocalStorage } from "node:async_hooks";

interface TraceStore {
  traceId: string;
  spanId: string;
}

export const _traceStorage = new AsyncLocalStorage<TraceStore>();

export class TraceContext {
  /**
   * Run fn() inside a new trace scope. All captures within fn() share
   * the same trace_id automatically.
   */
  static startTrace<T>(fn: () => T): T {
    return _traceStorage.run({ traceId: randomUUID(), spanId: "" }, fn);
  }

  /** Return the active trace ID, or undefined if outside a trace scope. */
  static currentTraceId(): string | undefined {
    return _traceStorage.getStore()?.traceId;
  }

  /**
   * Advance the span context. Returns { spanId, parentSpanId }.
   *
   * Call at the start of each logical operation (tool call, LLM call, etc.)
   * so that nested captures report this span as their parent.
   */
  static newSpan(): { spanId: string; parentSpanId: string } {
    const store = _traceStorage.getStore();
    const parentSpanId = store?.spanId ?? "";
    const spanId = randomUUID();
    if (store) {
      store.spanId = spanId;
    }
    return { spanId, parentSpanId };
  }
}
