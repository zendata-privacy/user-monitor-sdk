/**
 * AI Telemetry SDK — LangChain Integration (TypeScript)
 *
 * Usage:
 *   import { TelemetryCallbackHandler } from "ai-telemetry-sdk/integrations/langchain";
 *
 *   const handler = new TelemetryCallbackHandler(telemetryClient);
 *   await chain.invoke(input, { callbacks: [handler] });
 */

import type { TelemetryClient } from "../client.js";
import { protocolOfClient } from "../transport.js";

// Minimal interface — avoids a hard dep on @langchain/core
interface LLMResult {
  llm_output?: {
    model_name?: string;
    token_usage?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      total_tokens?: number;
    };
  };
  generations?: Array<Array<{ text: string }>>;
}

interface ToolInfo {
  input: string;
  name: string;
}

const MAX_START_TIME_AGE_MS = 300_000; // 5 minutes

export class TelemetryCallbackHandler {
  private telemetry: TelemetryClient;
  private startTimes = new Map<string, number>();
  private protocols = new Map<string, string>();
  private toolInputs = new Map<string, ToolInfo>();

  constructor(telemetry: TelemetryClient) {
    this.telemetry = telemetry;
  }

  private cleanupStale(): void {
    const cutoff = Date.now() - MAX_START_TIME_AGE_MS;
    for (const [key, ts] of this.startTimes) {
      if (ts < cutoff) {
        this.startTimes.delete(key);
        this.protocols.delete(key);
        this.toolInputs.delete(key); // no-op for LLM keys
      }
    }
  }

  // LangChain callback interface methods -----------------------------------

  handleLLMStart(llm: unknown, _prompts: string[], runId: string) {
    this.cleanupStale();
    this.startTimes.set(runId, Date.now());
    this.rememberProtocol(runId, llm);
  }

  private rememberProtocol(runId: string, llm: unknown) {
    const p = protocolOfClient(llm);
    if (p) this.protocols.set(runId, p);
  }

  private takeProtocol(runId: string): { call_protocol?: string } {
    const p = this.protocols.get(runId);
    this.protocols.delete(runId);
    return p ? { call_protocol: p } : {};
  }

  handleChatModelStart(
    llm: unknown,
    _messages: unknown[][],
    runId: string
  ) {
    this.cleanupStale();
    this.startTimes.set(runId, Date.now());
    this.rememberProtocol(runId, llm);
  }

  handleLLMEnd(output: LLMResult, runId: string) {
    const start = this.startTimes.get(runId) ?? Date.now();
    this.startTimes.delete(runId);
    const latency_ms = Date.now() - start;

    const usage = output.llm_output?.token_usage ?? {};
    const text = output.generations?.[0]?.[0]?.text;

    this.telemetry.capture({
      event_type: "llm_call",
      provider: "langchain",
      model: output.llm_output?.model_name ?? "unknown",
      latency_ms,
      output_message: { role: "assistant", content: text },
      prompt_tokens: usage.prompt_tokens,
      completion_tokens: usage.completion_tokens,
      total_tokens: usage.total_tokens,
      ...this.takeProtocol(runId),
    });
  }

  handleLLMError(error: Error, runId: string) {
    this.startTimes.delete(runId);
    this.telemetry.capture({
      event_type: "error",
      provider: "langchain",
      error_type: error.constructor?.name ?? "Error",
      error_message: error.message,
      error_stack: error.stack,
      ...this.takeProtocol(runId),
    });
  }

  handleToolStart(
    _tool: unknown,
    input: string,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _metadata?: unknown,
    name?: string
  ) {
    this.cleanupStale();
    this.startTimes.set(`tool_${runId}`, Date.now());
    this.toolInputs.set(`tool_${runId}`, { input, name: name ?? "unknown" });
  }

  handleToolEnd(output: string, runId: string) {
    const start = this.startTimes.get(`tool_${runId}`) ?? Date.now();
    this.startTimes.delete(`tool_${runId}`);
    const toolInfo = this.toolInputs.get(`tool_${runId}`);
    this.toolInputs.delete(`tool_${runId}`);

    this.telemetry.capture({
      event_type: "tool_call",
      tool_name: toolInfo?.name ?? "unknown",
      tool_input: toolInfo ? { input: toolInfo.input } : undefined,
      tool_output: output,
      latency_ms: Date.now() - start,
    });
  }

  handleToolError(error: Error, runId: string) {
    this.startTimes.delete(`tool_${runId}`);
    this.telemetry.capture({
      event_type: "error",
      error_type: error.constructor?.name ?? "Error",
      error_message: error.message,
      error_stack: error.stack,
    });
  }
}
