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

export class TelemetryCallbackHandler {
  private telemetry: TelemetryClient;
  private startTimes = new Map<string, number>();

  constructor(telemetry: TelemetryClient) {
    this.telemetry = telemetry;
  }

  // LangChain callback interface methods -----------------------------------

  handleLLMStart(_llm: unknown, _prompts: string[], runId: string) {
    this.startTimes.set(runId, Date.now());
  }

  handleChatModelStart(
    _llm: unknown,
    _messages: unknown[][],
    runId: string
  ) {
    this.startTimes.set(runId, Date.now());
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
    this.startTimes.set(`tool_${runId}`, Date.now());
    this.telemetry.capture({
      event_type: "tool_call",
      tool_name: name ?? "unknown",
      tool_input: { input },
    });
  }

  handleToolEnd(output: string, runId: string) {
    const start = this.startTimes.get(`tool_${runId}`) ?? Date.now();
    this.startTimes.delete(`tool_${runId}`);

    this.telemetry.capture({
      event_type: "tool_call",
      latency_ms: Date.now() - start,
      tool_output: output,
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
