/**
 * AI Telemetry SDK — Main Entry Point
 */

export { TelemetryClient } from "./client.js";
export type { TelemetryEvent, TelemetryClientOptions } from "./client.js";
export { traceLLM, traceTool } from "./trace.js";
export { TraceContext } from "./context.js";

// Convenience factory ---------------------------------------------------------
import { TelemetryClient, TelemetryClientOptions } from "./client.js";

let _defaultClient: TelemetryClient | null = null;

export interface InitOptions extends TelemetryClientOptions {
  autoWrapOpenAI?: boolean;
  autoWrapAnthropic?: boolean;
}

/**
 * Initialize the AI Telemetry SDK.
 *
 * @example
 * ```ts
 * import { init } from "ai-telemetry-sdk";
 * const telemetry = init({ appId: "my-app", apiKey: "sk-telem-..." });
 * ```
 */
export function init(opts: InitOptions): TelemetryClient {
  _defaultClient = new TelemetryClient(opts);
  return _defaultClient;
}

/** Return the default initialized client. */
export function getClient(): TelemetryClient {
  if (!_defaultClient) {
    throw new Error(
      "ai-telemetry-sdk: init() has not been called. Call init({ appId, apiKey }) at startup."
    );
  }
  return _defaultClient;
}
