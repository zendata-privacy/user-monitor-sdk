/**
 * AI Telemetry SDK — Core Client (TypeScript)
 */

import { randomUUID } from "node:crypto";

export interface TelemetryEvent {
  event_id?: string;
  app_id?: string;
  event_type: "llm_call" | "tool_call" | "error";
  timestamp_ms?: number;

  // LLM fields
  model?: string;
  provider?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  input_messages?: Array<{ role: string; content: string }>;
  output_message?: { role: string; content?: string; tool_calls?: unknown[] };
  latency_ms?: number;

  // Tool / function call fields
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: unknown;

  // Error fields
  error_type?: string;
  error_message?: string;
  error_stack?: string;

  // Custom metadata
  metadata?: Record<string, unknown>;
}

export interface TelemetryClientOptions {
  appId: string;
  apiKey: string;
  endpoint?: string;
  batchSize?: number;
  flushIntervalMs?: number;
  maxQueueSize?: number;
  capturePrompts?: boolean;
  debug?: boolean;
}

export class TelemetryClient {
  private readonly appId: string;
  private readonly apiKey: string;
  private readonly endpoint: string;
  private readonly batchSize: number;
  private readonly maxQueueSize: number;
  private readonly capturePrompts: boolean;
  private readonly debug: boolean;

  private queue: TelemetryEvent[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private flushing = false;

  // Circuit breaker state
  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;
  private readonly circuitFailureThreshold = 5;
  private readonly circuitCooldownMs = 30_000;
  private readonly maxRetries = 3;

  constructor(opts: TelemetryClientOptions) {
    this.appId = opts.appId;
    this.apiKey = opts.apiKey;
    this.endpoint = opts.endpoint ?? "https://ingest.yourplatform.com/v1/events";
    this.batchSize = opts.batchSize ?? 20;
    this.maxQueueSize = opts.maxQueueSize ?? 1000;
    this.capturePrompts = opts.capturePrompts ?? true;
    this.debug = opts.debug ?? false;

    const intervalMs = opts.flushIntervalMs ?? 5_000;
    this.flushTimer = setInterval(() => this.flush(), intervalMs);
    // Don't block process exit
    this.flushTimer.unref();
  }

  /** Enqueue a telemetry event (non-blocking). */
  capture(event: TelemetryEvent): void {
    const enriched: TelemetryEvent = {
      event_id: randomUUID(),
      timestamp_ms: Date.now(),
      app_id: this.appId,
      ...event,
    };

    if (!this.capturePrompts) {
      delete enriched.input_messages;
      delete enriched.output_message;
      delete enriched.tool_input;
      delete enriched.tool_output;
    }

    if (this.queue.length >= this.maxQueueSize) {
      this.queue.shift();
      if (this.debug) console.warn("[AI Telemetry] Queue full — dropping oldest event");
    }
    this.queue.push(enriched);

    if (this.queue.length >= this.batchSize) {
      this.flush();
    }
  }

  /** Flush all pending events (returns a Promise). */
  async flush(): Promise<void> {
    if (this.flushing || this.queue.length === 0) return;
    this.flushing = true;
    try {
      const batch = this.queue.splice(0, this.queue.length);
      await this._send(batch);
    } finally {
      this.flushing = false;
    }
  }

  /** Flush remaining events and stop the background timer. */
  async shutdown(): Promise<void> {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    await this.flush();
  }

  private async _send(events: TelemetryEvent[]): Promise<void> {
    try {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const os = await import("node:os");
      const logDir = process.env.ZENDATA_LOG_DIR ?? path.join(os.tmpdir(), "zendata");
      await fs.mkdir(logDir, { recursive: true });
      const logFile = path.join(logDir, "events.csv");

      const headers = [
        "event_id", "timestamp_iso", "event_type", "model", "prompt_tokens",
        "completion_tokens", "total_tokens", "latency_ms", "user_id", "email",
        "display_name", "department", "job_title", "sailpoint_risk", "pii_detected",
        "pii_types", "trust_boundary_crossed", "risk_signal", "zendata_composite_risk",
        "zendata_classification", "data_sources", "tools_called",
      ];

      let fileExists = true;
      try {
        await fs.access(logFile);
      } catch {
        fileExists = false;
      }

      let csvContent = "";
      if (!fileExists) {
        csvContent += headers.join(",") + "\n";
      }

      for (const e of events) {
        const tsIso = e.timestamp_ms
          ? new Date(e.timestamp_ms).toISOString()
          : new Date().toISOString();
        const row: (string | number)[] = [
          e.event_id ?? "",
          tsIso,
          e.event_type ?? "",
          e.model ?? "",
          e.prompt_tokens ?? "",
          e.completion_tokens ?? "",
          e.total_tokens ?? "",
          e.latency_ms ?? "",
        ];

        const meta = e.metadata ?? {};
        for (let i = 8; i < headers.length; i++) {
          const val = meta[headers[i]];
          row.push(typeof val === "string" || typeof val === "number" ? val : "");
        }

        csvContent += row.map(v => typeof v === "string" && v.includes(",") ? `"${v}"` : v).join(",") + "\n";
      }

      await fs.appendFile(logFile, csvContent, "utf-8");
    } catch (err) {
      if (this.debug) {
        console.error("[AI Telemetry] Failed to write to local log file:", err);
      }
    }

    await this._sendHttp(events);
  }

  private async _sendHttp(events: TelemetryEvent[]): Promise<void> {
    // Circuit breaker: skip if endpoint is known-down
    if (Date.now() < this.circuitOpenUntil) {
      if (this.debug) {
        console.warn("[AI Telemetry] Circuit open — skipping send");
      }
      return;
    }

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const res = await fetch(this.endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
            "User-Agent": "ai-telemetry-sdk-js/1.0.0",
          },
          body: JSON.stringify({ events }),
        });

        if (res.ok) {
          this.consecutiveFailures = 0;
          if (this.debug) {
            console.log(`[AI Telemetry] Flushed ${events.length} events → ${res.status}`);
          }
          return;
        }

        if (res.status < 500) {
          // 4xx — client error, don't retry
          this.consecutiveFailures = 0;
          if (this.debug) {
            console.warn(`[AI Telemetry] Send failed (client error ${res.status})`);
          }
          return;
        }

        // 5xx — fall through to retry
        if (this.debug) {
          console.warn(`[AI Telemetry] Server error ${res.status}, attempt ${attempt + 1}/${this.maxRetries}`);
        }
      } catch (err) {
        // Network error — retry
        if (this.debug) {
          console.warn(`[AI Telemetry] Network error, attempt ${attempt + 1}/${this.maxRetries}:`, err);
        }
      }

      if (attempt < this.maxRetries - 1) {
        await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
      }
    }

    // All retries exhausted
    this.consecutiveFailures++;
    if (this.consecutiveFailures >= this.circuitFailureThreshold) {
      this.circuitOpenUntil = Date.now() + this.circuitCooldownMs;
      if (this.debug) {
        console.warn(`[AI Telemetry] Circuit opened for ${this.circuitCooldownMs / 1000}s after ${this.consecutiveFailures} consecutive failures`);
      }
    }
    if (this.debug) {
      console.error(`[AI Telemetry] Send failed after ${this.maxRetries} retries`);
    }
  }
}
