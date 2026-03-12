"""
AI Telemetry SDK - Core Client
Sends telemetry events to your configured endpoint.
"""

import time
import uuid
import threading
import queue
import json
import urllib.request
import urllib.error
import os
import csv
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

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class TelemetryClient:
    """
    Core telemetry client. Buffers events and flushes them in a background thread.

    Usage:
        client = TelemetryClient(app_id="my-app", api_key="sk-...", endpoint="https://your-platform/ingest")
    """

    def __init__(
        self,
        app_id: str,
        api_key: str,
        endpoint: str = "https://ingest.yourplatform.com/v1/events",
        batch_size: int = 20,
        flush_interval_s: float = 5.0,
        max_queue_size: int = 1000,
        capture_prompts: bool = True,
        debug: bool = False,
    ):
        self.app_id = app_id
        self.api_key = api_key
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.capture_prompts = capture_prompts
        self.debug = debug

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, event: TelemetryEvent):
        """Enqueue a telemetry event (non-blocking)."""
        event.app_id = self.app_id

        # Zendata Engine Real-Time Evaluation
        if event.event_type == "llm_call":
            from ai_telemetry.integrations.zendata_integration import evaluate_event
            evaluate_event(event)

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
        # Local logging implementation
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "events.csv")
        
        headers = [
            "event_id", "timestamp_iso", "event_type", "model", "prompt_tokens", 
            "completion_tokens", "total_tokens", "latency_ms", "user_id", "email", 
            "display_name", "department", "job_title", "sailpoint_risk", "pii_detected", 
            "pii_types", "trust_boundary_crossed", "risk_signal", "zendata_composite_risk", 
            "zendata_classification", "data_sources", "tools_called"
        ]
        
        file_exists = os.path.isfile(log_file)
        try:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
                if not file_exists:
                    writer.writeheader()
                
                for e in events:
                    ts_iso = datetime.fromtimestamp(e.timestamp_ms / 1000.0, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    row = {
                        "event_id": e.event_id,
                        "timestamp_iso": ts_iso,
                        "event_type": e.event_type or "",
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
                    }
                    # Merge existing metadata fields just in case
                    meta = e.metadata or {}
                    for k, v in meta.items():
                        if k in headers and not row.get(k):
                            row[k] = v
                    writer.writerow(row)
        except Exception as ex:
            if self.debug:
                print(f"[AI Telemetry] Failed to write to local log file: {ex}")

        # HTTP submission (unchanged)
        payload = json.dumps({"events": [e.to_dict() for e in events]}).encode()
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
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if self.debug:
                    print(f"[AI Telemetry] Flushed {len(events)} events → {resp.status}")
        except urllib.error.URLError as e:
            if self.debug:
                print(f"[AI Telemetry] Send failed: {e}")
            pass # Keep submission, but pass if it fails