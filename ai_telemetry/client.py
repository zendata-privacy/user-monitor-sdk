"""
AI Telemetry SDK - Core Client
Sends telemetry events to your configured endpoint.
"""

import atexit
import time
import uuid
import threading
import queue
import json
import urllib.request
import urllib.error
import os
import csv
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class TelemetryEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    app_id: str = ""
    event_type: str = ""          # "llm_call", "tool_call", "error"
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # LLM fields
    model: Optional[str] = None
    provider: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_messages: Optional[List[Dict]] = None
    output_message: Optional[Dict] = None
    latency_ms: Optional[int] = None

    # Tool / function call fields
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[Any] = None

    # Error fields
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # New Context and Analytics Fields
    user_id: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None
    sailpoint_risk: Optional[str] = None
    pii_detected: Optional[bool] = None
    pii_types: Optional[str] = None
    trust_boundary_crossed: Optional[bool] = None
    risk_signal: Optional[str] = None
    zendata_composite_risk: Optional[str] = None
    zendata_classification: Optional[str] = None
    data_sources: Optional[str] = None
    tools_called: Optional[str] = None

    # Trace context — links events into an execution chain
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Entity classification (populated by entity_classifier)
    entity_type: Optional[str] = None              # HUMAN | AUTOMATED | AGENT | UNKNOWN
    entity_subtype: Optional[str] = None           # SUBAGENT_RECEIVER | ORCHESTRATOR_SENDER
    classification_signals: Optional[str] = None   # pipe-delimited rule IDs that fired
    behavioral_alerts: Optional[str] = None        # pipe-delimited alert names
    session_turn_index: Optional[int] = None       # 0-based turn number within the session
    inter_turn_latency_ms: Optional[int] = None    # ms between last response and this submission
    prompt_submission_time_ms: Optional[int] = None  # epoch ms when the prompt was submitted

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class TelemetryClient:
    """
    Core telemetry client. Buffers events and flushes them in a background thread.

    Usage:
        client = TelemetryClient(app_id="my-app", api_key="sk-...")
    """

    def __init__(
        self,
        app_id: str,
        api_key: str,
        endpoint: str = "https://ingest.zendata.dev/v1/events",
        batch_size: int = 20,
        flush_interval_s: float = 5.0,
        max_queue_size: int = 1000,
        capture_prompts: bool = True,
        debug: bool = False,
        enable_local_log: bool = False,
        log_path: Optional[str] = None,
    ):
        self.app_id = app_id
        self.api_key = api_key
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.capture_prompts = capture_prompts
        self.debug = debug
        self.enable_local_log = enable_local_log
        self.log_path = log_path or os.path.join(
            os.environ.get("ZENDATA_LOG_DIR", os.path.join(tempfile.gettempdir(), "zendata")),
            "events.csv",
        )

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, event: TelemetryEvent):
        """Enqueue a telemetry event (non-blocking)."""
        event.app_id = self.app_id

        # Auto-populate trace context from context vars if not explicitly set on the event
        from ai_telemetry.context import _current_trace_id, _current_span_id, current_user_context
        if not event.trace_id:
            tid = _current_trace_id.get()
            event.trace_id = tid or None
        if not event.span_id:
            event.span_id = str(uuid.uuid4())
        if not event.parent_span_id:
            pid = _current_span_id.get()
            event.parent_span_id = pid or None

        # Entity classification
        user_ctx = current_user_context.get()
        entity_key = user_ctx.get("user_id") or "unknown"
        session_id = event.trace_id or entity_key

        if event.event_type == "llm_call":
            # Full classification — runs rules and updates session state
            prompt_text = user_ctx.get("prompt_text", "")
            if prompt_text:
                from ai_telemetry.entity_classifier import classify_event
                from ai_telemetry.session_state import get_session_store
                submission_ms = user_ctx.get("prompt_submission_time_ms") or event.timestamp_ms
                classify_event(
                    event,
                    get_session_store(),
                    session_id,
                    entity_key,
                    prompt_text,
                    submission_ms,
                )

        elif event.event_type == "tool_call":
            # Inherit entity type from the session established by the preceding LLM call
            from ai_telemetry.session_state import get_session_store
            state = get_session_store().get_or_create_session(session_id, entity_key)
            if state.entity_type != "UNKNOWN":
                event.entity_type = state.entity_type
                event.entity_subtype = state.entity_subtype

        if not self.capture_prompts:
            event.input_messages = None
            event.output_message = None
            event.tool_input = None
            event.tool_output = None
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            if self.debug:
                print("[AI Telemetry] Queue full — dropping event")

    def flush(self):
        """Flush all pending events synchronously."""
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if events:
            self._send(events)

    def shutdown(self):
        """Flush remaining events and stop the background thread."""
        self._shutdown.set()
        self.flush()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_loop(self):
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self.flush_interval_s)
            self.flush()

    def _send(self, events: List[TelemetryEvent]):
        # Zendata evaluation runs here, on the background thread (never on the caller's thread)
        from ai_telemetry.integrations.zendata_integration import evaluate_event
        for event in events:
            if event.event_type == "llm_call":
                evaluate_event(event)

        # HTTP first — primary delivery path
        self._send_http(events)

        # CSV + JSONL — optional, best-effort; never allowed to block or raise
        if self.enable_local_log:
            try:
                self._write_csv(events)
            except Exception as ex:
                if self.debug:
                    print(f"[AI Telemetry] CSV write failed (non-fatal): {ex}")
            try:
                self._write_jsonl(events)
            except Exception as ex:
                if self.debug:
                    print(f"[AI Telemetry] JSONL write failed (non-fatal): {ex}")

    def _send_http(self, events: List[TelemetryEvent]):
        _MAX_RETRIES = 3
        _BACKOFF_BASE = 1.0
        _CIRCUIT_THRESHOLD = 5
        _CIRCUIT_COOLDOWN_S = 30.0

        # Circuit breaker: skip entirely if endpoint is known-down
        if self._circuit_open_until > time.time():
            if self.debug:
                print("[AI Telemetry] Circuit open — skipping send")
            return

        serializable = []
        for e in events:
            try:
                json.dumps(e.to_dict())  # validate before batching
                serializable.append(e.to_dict())
            except (TypeError, ValueError) as ex:
                if self.debug:
                    print(f"[AI Telemetry] Skipping non-serializable event {e.event_id}: {ex}")
        if not serializable:
            return
        payload = json.dumps({"events": serializable}).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "User-Agent": "ai-telemetry-sdk-python/1.0.0",
            },
            method="POST",
        )

        last_err: Exception = Exception("unknown")
        for attempt in range(_MAX_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if self.debug:
                        print(f"[AI Telemetry] Flushed {len(events)} events → {resp.status}")
                    self._consecutive_failures = 0
                    return  # success
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    # 4xx — client error, don't retry
                    self._consecutive_failures = 0
                    if self.debug:
                        print(f"[AI Telemetry] Send failed (client error {e.code}): {e}")
                    return
                last_err = e  # 5xx — fall through to retry
            except urllib.error.URLError as e:
                last_err = e  # network error — retry

            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                if self.debug:
                    print(f"[AI Telemetry] Retry {attempt + 1}/{_MAX_RETRIES} in {wait}s: {last_err}")
                time.sleep(wait)

        # All retries exhausted
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_THRESHOLD:
            self._circuit_open_until = time.time() + _CIRCUIT_COOLDOWN_S
            if self.debug:
                print(f"[AI Telemetry] Circuit opened for {_CIRCUIT_COOLDOWN_S}s after {self._consecutive_failures} consecutive failures")
        if self.debug:
            print(f"[AI Telemetry] Send failed after {_MAX_RETRIES} retries: {last_err}")

    def _write_csv(self, events: List[TelemetryEvent]):
        headers = [
            "event_id", "timestamp_iso", "event_type", "trace_id", "span_id", "parent_span_id",
            "model", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms",
            "user_id", "email", "display_name", "department", "job_title", "sailpoint_risk",
            "pii_detected", "pii_types", "trust_boundary_crossed", "risk_signal",
            "zendata_composite_risk", "zendata_classification", "data_sources", "tools_called",
            "entity_type", "entity_subtype", "classification_signals", "behavioral_alerts",
            "session_turn_index", "inter_turn_latency_ms",
        ]

        log_file = self.log_path
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=',')
            if not file_exists:
                writer.writeheader()

            for e in events:
                ts_iso = datetime.fromtimestamp(e.timestamp_ms / 1000.0, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                row = {
                    "event_id": e.event_id,
                    "timestamp_iso": ts_iso,
                    "event_type": e.event_type or "",
                    "trace_id": e.trace_id or "",
                    "span_id": e.span_id or "",
                    "parent_span_id": e.parent_span_id or "",
                    "model": e.model or "",
                    "prompt_tokens": e.prompt_tokens if e.prompt_tokens is not None else "",
                    "completion_tokens": e.completion_tokens if e.completion_tokens is not None else "",
                    "total_tokens": e.total_tokens if e.total_tokens is not None else "",
                    "latency_ms": e.latency_ms if e.latency_ms is not None else "",
                    "user_id": e.user_id or "",
                    "email": e.email or "",
                    "display_name": e.display_name or "",
                    "department": e.department or "",
                    "job_title": e.job_title or "",
                    "sailpoint_risk": e.sailpoint_risk if e.sailpoint_risk is not None else "",
                    "pii_detected": e.pii_detected if e.pii_detected is not None else "",
                    "pii_types": e.pii_types or "",
                    "trust_boundary_crossed": e.trust_boundary_crossed if e.trust_boundary_crossed is not None else "",
                    "risk_signal": e.risk_signal or "",
                    "zendata_composite_risk": e.zendata_composite_risk if e.zendata_composite_risk is not None else "",
                    "zendata_classification": e.zendata_classification or "",
                    "data_sources": e.data_sources or "",
                    "tools_called": e.tools_called or "",
                    "entity_type": e.entity_type or "",
                    "entity_subtype": e.entity_subtype or "",
                    "classification_signals": e.classification_signals or "",
                    "behavioral_alerts": e.behavioral_alerts or "",
                    "session_turn_index": e.session_turn_index if e.session_turn_index is not None else "",
                    "inter_turn_latency_ms": e.inter_turn_latency_ms if e.inter_turn_latency_ms is not None else "",
                }
                meta = e.metadata or {}
                for k, v in meta.items():
                    if k in headers and not row.get(k):
                        row[k] = v
                writer.writerow(row)

    def _write_jsonl(self, events: List[TelemetryEvent]):
        """Append events to a newline-delimited JSON file (one object per line).

        The file path is derived from ``log_path`` by replacing the extension
        with ``.jsonl``.  Each line is a complete, self-contained JSON object
        including a human-readable ``timestamp_iso`` field.
        """
        jsonl_path = os.path.splitext(self.log_path)[0] + ".jsonl"
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, mode="a", encoding="utf-8") as f:
            for e in events:
                try:
                    d = e.to_dict()
                    d["timestamp_iso"] = datetime.fromtimestamp(
                        e.timestamp_ms / 1000.0, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    f.write(json.dumps(d) + "\n")
                except (TypeError, ValueError) as ex:
                    if self.debug:
                        print(f"[AI Telemetry] Skipping non-serializable event {e.event_id}: {ex}")