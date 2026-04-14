"""
Amazon Comprehend PII redactor.

Scrubs PII from telemetry events before they are persisted locally or sent
to the ingest endpoint. Fails open: any AWS error leaves the original text
unchanged so Comprehend can never break the host application.

Required IAM permission:
    comprehend:DetectPiiEntities

Supported languages (see Comprehend docs): "en", "es".
"""

import os
import threading
from typing import Any, Iterable, List, Optional, Tuple


class ComprehendRedactor:
    """Thread-safe wrapper around Amazon Comprehend DetectPiiEntities.

    Parameters
    ----------
    language_code : str
        ISO 639-1 code. Default "en".
    entity_types : Optional[Iterable[str]]
        If set, only these Comprehend entity types are masked
        (e.g. {"EMAIL", "SSN", "CREDIT_DEBIT_NUMBER"}). Default: mask every
        type Comprehend returns.
    mask_format : str
        Replacement template; ``{type}`` is substituted with the matched
        Comprehend entity type. Default ``"[{type}]"`` -> ``"[EMAIL]"``.
    aws_region : Optional[str]
        AWS region for the Comprehend client. Falls back to the
        ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` env vars.
    max_bytes : int
        Max UTF-8 bytes per Comprehend call. The service hard limit is
        100,000 bytes; longer strings are chunked automatically.
    """

    _COMPREHEND_HARD_LIMIT = 100_000

    def __init__(
        self,
        language_code: str = "en",
        entity_types: Optional[Iterable[str]] = None,
        mask_format: str = "[{type}]",
        aws_region: Optional[str] = None,
        max_bytes: int = 90_000,
    ):
        self.language_code = language_code
        self.entity_types = {t.upper() for t in entity_types} if entity_types else None
        self.mask_format = mask_format
        self.aws_region = (
            aws_region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        self.max_bytes = min(max_bytes, self._COMPREHEND_HARD_LIMIT - 1_000)
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        import boto3
                    except ImportError as exc:
                        raise RuntimeError(
                            "ComprehendRedactor requires boto3. "
                            "Install with: pip install 'ai-telemetry-sdk[aws]' "
                            "or pip install boto3."
                        ) from exc
                    kwargs = {"region_name": self.aws_region} if self.aws_region else {}
                    self._client = boto3.client("comprehend", **kwargs)
        return self._client

    def redact(self, text: str) -> Tuple[str, List[str]]:
        """Return ``(masked_text, list_of_pii_types_found)``.

        Returns the original text and an empty list when no PII is found
        or on transient AWS errors.
        """
        if not text or not isinstance(text, str):
            return text, []

        chunks = self._chunk_by_bytes(text, self.max_bytes)
        out_parts: List[str] = []
        types_found: set = set()
        for chunk in chunks:
            masked, types = self._redact_single(chunk)
            out_parts.append(masked)
            types_found.update(types)
        return "".join(out_parts), sorted(types_found)

    def _redact_single(self, text: str) -> Tuple[str, List[str]]:
        try:
            client = self._get_client()
            resp = client.detect_pii_entities(
                Text=text, LanguageCode=self.language_code
            )
        except Exception:
            return text, []

        entities = resp.get("Entities") or []
        if not entities:
            return text, []

        # Apply replacements right-to-left so offsets stay valid.
        entities = sorted(entities, key=lambda e: e["BeginOffset"], reverse=True)
        out = text
        types_found: List[str] = []
        for ent in entities:
            etype = str(ent.get("Type", "PII")).upper()
            if self.entity_types and etype not in self.entity_types:
                continue
            replacement = self.mask_format.format(type=etype)
            out = out[: ent["BeginOffset"]] + replacement + out[ent["EndOffset"]:]
            types_found.append(etype)
        return out, types_found

    @staticmethod
    def _chunk_by_bytes(text: str, max_bytes: int) -> List[str]:
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return [text]
        chunks: List[str] = []
        i = 0
        while i < len(data):
            end = min(i + max_bytes, len(data))
            while end < len(data) and (data[end] & 0xC0) == 0x80:
                end -= 1
            chunks.append(data[i:end].decode("utf-8", errors="replace"))
            i = end
        return chunks


def apply_redaction_to_event(event, redactor: ComprehendRedactor) -> None:
    """Mutate *event* in place, masking PII across all free-text fields.

    Populates ``event.pii_detected`` and ``event.pii_types`` whenever PII
    is found, so downstream risk scoring and dashboards can count hits
    without needing access to the raw text.
    """
    types_seen: set = set()

    def _redact(val: Any) -> Any:
        if isinstance(val, str) and val:
            masked, types = redactor.redact(val)
            if types:
                types_seen.update(types)
            return masked
        if isinstance(val, dict):
            return {k: _redact(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_redact(v) for v in val]
        return val

    if event.input_messages:
        event.input_messages = _redact(event.input_messages)
    if event.output_message:
        event.output_message = _redact(event.output_message)
    if event.tool_input is not None:
        event.tool_input = _redact(event.tool_input)
    if event.tool_output is not None:
        event.tool_output = _redact(event.tool_output)

    if types_seen:
        event.pii_detected = True
        existing = {t for t in (event.pii_types or "").split("|") if t}
        event.pii_types = "|".join(sorted(existing | types_seen))
