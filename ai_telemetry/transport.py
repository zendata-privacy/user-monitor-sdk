"""Call type (transport protocol) of a captured call: ``call_protocol``.

The value is the scheme of the endpoint the call actually went to — ``https``,
``http``, ``grpc``, ``grpcs``, ``wss``, ``ws``, ``unix`` … — read from the
client's configured base URL, so a custom or proxied endpoint is reported as it
is (an LLM reached over plain ``http`` through a local proxy shows ``http``).

Nothing is guessed: when the endpoint is not known the field stays empty.
Only the scheme is kept; host, path, query and credentials are never recorded.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit

# Attribute names under which SDK clients / LangChain models keep their endpoint.
_URL_ATTRS = (
    "base_url", "_base_url", "endpoint_url", "api_base", "openai_api_base",
    "anthropic_api_url", "azure_endpoint", "endpoint",
)


def protocol_from_url(url: Any) -> Optional[str]:
    """The lower-case scheme of a URL (``https://x/y`` → ``https``), else None.

    Strings without a scheme (an ARN, a bare host, a file path) return None.
    """
    if url is None:
        return None
    text = str(url).strip()
    if "://" not in text:
        return None
    scheme = urlsplit(text).scheme.lower()
    if scheme.startswith("http+unix"):       # e.g. docker-style unix sockets
        return "unix"
    return scheme or None


def protocol_of_client(obj: Any) -> Optional[str]:
    """Protocol of the endpoint an SDK resource / client / model is bound to.

    Looks at the object, its ``_client`` (OpenAI / Anthropic resources) and its
    botocore ``meta`` (boto3 clients). Returns None when no endpoint is found.
    """
    seen = 0
    while obj is not None and seen < 3:
        for attr in _URL_ATTRS:
            proto = protocol_from_url(getattr(obj, attr, None))
            if proto:
                return proto
        meta = getattr(obj, "meta", None)
        proto = protocol_from_url(getattr(meta, "endpoint_url", None))
        if proto:
            return proto
        obj = getattr(obj, "_client", None)
        seen += 1
    return None


# LangChain model classes whose endpoint is fixed by the provider library when
# no URL is configured (ChatOpenAI → https://api.openai.com, ChatAnthropic →
# https://api.anthropic.com, Bedrock → https://bedrock-runtime.<region>…). For
# these the protocol is the library's documented default, not a guess.
_DEFAULT_HTTPS_CLASSES = {
    "ChatOpenAI", "OpenAI", "OpenAIEmbeddings", "ChatAnthropic",
    "ChatBedrock", "ChatBedrockConverse", "BedrockLLM", "ChatMistralAI",
    "ChatGoogleGenerativeAI", "ChatVertexAI", "ChatCohere", "ChatGroq",
}


def protocol_from_serialized(serialized: Any) -> Optional[str]:
    """Protocol from a LangChain ``serialized`` model description.

    A configured URL in its kwargs wins; otherwise a known provider class with a
    fixed default endpoint reports that endpoint's protocol (https).
    """
    if not isinstance(serialized, dict):
        return None
    kwargs = serialized.get("kwargs") or {}
    for attr in _URL_ATTRS:
        proto = protocol_from_url(kwargs.get(attr))
        if proto:
            return proto
    ident = serialized.get("id") or []
    cls = ident[-1] if isinstance(ident, list) and ident else serialized.get("name")
    if cls in _DEFAULT_HTTPS_CLASSES:
        return "https"
    return None
