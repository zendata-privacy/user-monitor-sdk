/**
 * Call type (transport protocol) of a captured call: `call_protocol`.
 *
 * The scheme of the endpoint the call actually went to — "https", "http",
 * "grpc", "wss", … — read from the client's configured base URL, so a custom or
 * proxied endpoint is reported as it is. Nothing is guessed: unknown → undefined.
 * Only the scheme is kept; host, path, query and credentials are never recorded.
 * Mirrors ai_telemetry/transport.py.
 */

const URL_KEYS = [
  "baseURL", "baseUrl", "base_url", "apiBase", "openAIApiBase",
  "anthropicApiUrl", "azureOpenAIEndpoint", "endpoint",
];

export function protocolFromUrl(url: unknown): string | undefined {
  if (typeof url !== "string" && !(url instanceof URL)) return undefined;
  const text = String(url).trim();
  const m = /^([a-zA-Z][a-zA-Z0-9+.-]*):\/\//.exec(text);
  if (!m) return undefined;
  const scheme = m[1].toLowerCase();
  return scheme.startsWith("http+unix") ? "unix" : scheme;
}

// LangChain model classes whose endpoint is fixed by the provider library when no
// URL is configured; their protocol is that documented default. Mirrors transport.py.
const DEFAULT_HTTPS_CLASSES = new Set([
  "ChatOpenAI", "OpenAI", "OpenAIEmbeddings", "ChatAnthropic", "ChatBedrockConverse",
  "BedrockChat", "ChatMistralAI", "ChatGoogleGenerativeAI", "ChatVertexAI", "ChatCohere", "ChatGroq",
]);

/** LangChain serialized model (`{ id: [..., "ChatOpenAI"], kwargs }`) → its default protocol. */
export function defaultProtocolOfSerialized(obj: unknown): string | undefined {
  if (!obj || typeof obj !== "object") return undefined;
  const id = (obj as { id?: unknown }).id;
  const cls = Array.isArray(id) && id.length ? String(id[id.length - 1]) : undefined;
  return cls && DEFAULT_HTTPS_CLASSES.has(cls) ? "https" : undefined;
}

/** Protocol of the endpoint a client object (OpenAI, Anthropic, …) is bound to. */
export function protocolOfClient(obj: unknown, depth = 0): string | undefined {
  if (!obj || typeof obj !== "object" || depth > 2) return undefined;
  const rec = obj as Record<string, unknown>;
  for (const key of URL_KEYS) {
    const p = protocolFromUrl(rec[key]);
    if (p) return p;
  }
  // LangChain models keep client options under `clientConfig` / `configuration`.
  for (const nested of ["clientConfig", "configuration", "_client", "client", "kwargs"]) {
    const p = protocolOfClient(rec[nested], depth + 1);
    if (p) return p;
  }
  return undefined;
}
