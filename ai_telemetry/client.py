"""
AI Telemetry SDK - Core Client
Sends telemetry events to your configured endpoint.
"""

import atexit
import hashlib
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
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

# Imported at module load, NOT inside capture(). Deferring it put the module
# import and the regex compilation on the first captured event — ~8ms on the
# caller's request path, in a method whose entire contract is that it does not
# block. A lazy import inside a hot path only moves the cost, it does not remove
# it, and it moves it somewhere unmeasured.
try:
    from ai_telemetry.detection import apply_classification_to_event as _classify_event
except Exception:                                   # pragma: no cover
    _classify_event = None


# ---------------------------------------------------------------------------
# Module-level helpers (used by integrations too)
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Return the lowercase hex SHA-256 digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SENSITIVE_HEADERS = frozenset({
    "authorization", "cookie", "set-cookie", "x-api-key",
    "x-auth-token", "proxy-authorization", "www-authenticate",
})


def _digest_http_headers(headers: Dict[str, str]) -> str:
    """Return a SHA-256 digest of non-sensitive HTTP headers for client fingerprinting.

    Sensitive credential/session headers are excluded before hashing so the
    digest can safely be stored and compared without exposing secrets.
    """
    filtered = {
        k.lower(): v
        for k, v in headers.items()
        if k.lower() not in _SENSITIVE_HEADERS
    }
    canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical)


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

    # Module 2 — Human vs. Non-Human Classification
    caller_type_score: Optional[float] = None  # 0.0 (definite agent) → 1.0 (definite human); resolves in 3-5 calls

    # Module 5 — Behavioral Drift Detection
    drift_score: Optional[float] = None              # aggregate 0.0–1.0
    drift_classification: Optional[str] = None       # STABLE | MONITOR | INVESTIGATE | WARNING | CRITICAL
    drift_signals: Optional[str] = None              # pipe-delimited alert names from all components
    zasp_score: Optional[float] = None               # derived ZASP safety score (0–10, higher = safer)

    # Module 4 — Dynamic Role Inference
    human_inferred_role: Optional[str] = None       # ENGINEER | FINANCE | HR | COMPLIANCE | CUSTOMER_SERVICE | SECURITY | EXECUTIVE
    human_role_confidence: Optional[float] = None   # 0.0–1.0; emitted only for HUMAN entity_type
    agent_role: Optional[str] = None                # CS_AGENT | FINANCE_AGENT | ORCHESTRATOR | DATA_ANALYST | … (9-role taxonomy)
    agent_risk_level: Optional[str] = None          # CRITICAL | HIGH | MEDIUM | LOW-MED | LOW
    agent_primary_risk: Optional[str] = None        # human-readable risk description
    role_scope_gap: Optional[bool] = None           # True when observed tool scope diverges from stated agent role

    # Module 3 — Agent Fingerprinting
    agent_id: Optional[str] = None                      # pseudonymous stable ID (agt_<api_key_hash[:20]>)
    agent_display_name: Optional[str] = None            # inferred from "You are X" in system prompt
    agent_inferred_type: Optional[str] = None           # NIGHTLY_BATCH | REALTIME | HIGH_FREQUENCY_PIPELINE | BUSINESS_HOURS_AGENT
    orchestration_role: Optional[str] = None            # ORCHESTRATOR | SUBAGENT | STANDALONE
    prompt_template_hash: Optional[str] = None          # 16-char structural hash for cross-session template matching
    system_prompt_version_count: Optional[int] = None   # number of distinct system prompt versions observed

    # Module 1 — Signal Capture (fields added to complete the spec)
    api_key_hash: Optional[str] = None             # SHA-256 of caller's API key; enables clustering without credential exposure
    system_prompt_hash: Optional[str] = None       # SHA-256 of system prompt; enables agent versioning and lineage tracking
    system_prompt_embedding: Optional[List[float]] = None  # 768-dim embedding for agent similarity clustering / version drift
    ip_hash: Optional[str] = None                  # SHA-256 of source IP; enables geographic clustering without raw storage
    http_headers_digest: Optional[str] = None      # SHA-256 digest of non-sensitive HTTP headers for client fingerprinting

    # Entity classification (populated by entity_classifier)
    entity_type: Optional[str] = None              # HUMAN | AUTOMATED | AGENT | UNKNOWN
    entity_subtype: Optional[str] = None           # SUBAGENT_RECEIVER | ORCHESTRATOR_SENDER
    classification_signals: Optional[str] = None   # pipe-delimited rule IDs that fired
    behavioral_alerts: Optional[str] = None        # pipe-delimited alert names
    session_turn_index: Optional[int] = None       # 0-based turn number within the session
    inter_turn_latency_ms: Optional[int] = None    # ms between last response and this submission
    prompt_submission_time_ms: Optional[int] = None  # epoch ms when the prompt was submitted

    # Build provenance — which repository and commit this running artifact was
    # built from. Stamped at build time (ZENDATA_REPO / ZENDATA_COMMIT); see
    # ai_telemetry/provenance.py. This is what lets the correlator bind a repo to
    # a live app on the application's own report rather than on an operator's
    # assertion, and what lets it detect that the code someone SCANNED is not the
    # code that is RUNNING.
    repo: Optional[str] = None
    commit_sha: Optional[str] = None       # may carry a "-dirty" suffix
    build_id: Optional[str] = None

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
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        pii_redactor: Optional[Any] = None,
        classify_pii: bool = True,
    ):
        self.app_id = app_id
        self.api_key = api_key
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.capture_prompts = capture_prompts
        self.debug = debug
        self.enable_local_log = enable_local_log
        self.embed_fn = embed_fn  # optional callable: str -> List[float] (e.g. 768-dim sentence embedding)
        self.pii_redactor = pii_redactor  # any object with .redact(text) -> (masked, [types])
        # Built-in, dependency-free PII labelling used when no redactor is
        # supplied. Set False to record nothing about PII at all — which means
        # pii_types stays empty and no consumer can tell that from "clean".
        self.classify_pii = classify_pii
        # Pre-compute the API-key hash once so every event gets it for free
        self._api_key_hash: Optional[str] = _sha256(api_key) if api_key else None
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

        # --- Module 1: always-on signal fields --------------------------------
        # api_key_hash: attach unless the caller has already set it
        if not event.api_key_hash:
            event.api_key_hash = self._api_key_hash

        # build provenance: same contract as api_key_hash — attach unless the
        # caller set it. current_provenance() is lru_cached, so this is a tuple
        # lookup after the first event, and it is deliberately silent when the
        # build was not stamped: absence is reported as absence, never guessed.
        if not event.repo:
            from ai_telemetry.provenance import current_provenance
            event.repo, event.commit_sha, event.build_id = current_provenance()

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
        # Session state must persist ACROSS calls from the same caller so the
        # behavioral classifier can accumulate timing/structure signals over
        # multiple turns. trace_id is fresh per request and would isolate every
        # call into its own single-turn session — leaving every event stuck at
        # the 0.5 baseline (AMBIGUOUS). Default to entity_key (per-user session)
        # and let the host app override via user_ctx["session_id"] when it has
        # a more meaningful conversation id.
        session_id = user_ctx.get("session_id") or entity_key

        # --- Module 1: ip_hash and http_headers_digest from request context ----
        # The host application sets "ip_address" and/or "http_headers" in
        # current_user_context (e.g. from a FastAPI/Flask middleware).  The SDK
        # hashes them here so raw values are never persisted.
        if not event.ip_hash:
            ip = user_ctx.get("ip_address")
            if ip:
                event.ip_hash = _sha256(str(ip))

        if not event.http_headers_digest:
            headers = user_ctx.get("http_headers")
            if isinstance(headers, dict) and headers:
                event.http_headers_digest = _digest_http_headers(headers)

        forced_entity = user_ctx.get("force_entity_type")

        if event.event_type in ("llm_call", "user_message"):
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
            # force_entity_type applies only to llm_call (agent pipeline calls).
            # user_message events always use natural behavioral classification so
            # the human turn is correctly identified as HUMAN/AUTOMATED/AGENT.
            if event.event_type == "llm_call" and forced_entity:
                event.entity_type = forced_entity
                from ai_telemetry.session_state import get_session_store
                store = get_session_store()
                state = store.get_or_create_session(session_id, entity_key)
                state.entity_type = forced_entity
                store.update_session(state)

        elif event.event_type == "tool_call":
            # Inherit entity type from the session established by the preceding LLM call
            from ai_telemetry.session_state import get_session_store
            state = get_session_store().get_or_create_session(session_id, entity_key)
            if forced_entity:
                event.entity_type = forced_entity
            elif state.entity_type != "UNKNOWN":
                event.entity_type = state.entity_type
                event.entity_subtype = state.entity_subtype

        # PII classification runs BEFORE prompts are dropped, so the signal
        # survives capture_prompts=False: you learn that SSNs are flowing through
        # the assistant without storing a single one.
        #
        # A supplied redactor wins — it both masks and labels. The built-in
        # classifier only labels, and exists because an unset pii_types is
        # indistinguishable from "we looked and found nothing", so every
        # downstream consumer read clean traffic where nobody had looked at all.
        if self.pii_redactor is not None and self.capture_prompts:
            try:
                from ai_telemetry.redaction import apply_redaction_to_event
                apply_redaction_to_event(event, self.pii_redactor)
            except Exception as exc:
                if self.debug:
                    print(f"[AI Telemetry] PII redaction skipped: {exc}")
        elif self.classify_pii and _classify_event is not None:
            try:
                _classify_event(event)
            except Exception as exc:
                if self.debug:
                    print(f"[AI Telemetry] PII classification skipped: {exc}")

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
        from ai_telemetry.agent_fingerprint import fingerprint_agent, get_fingerprint_store
        from ai_telemetry.drift_detector import analyze_drift, derive_zasp_score
        for event in events:
            if event.event_type == "llm_call":
                evaluate_event(event)
                # Module 3 — build/update agent fingerprint for non-human callers
                if event.entity_type in ("API CALL", "AGENT", "AMBIGUOUS"):
                    try:
                        fingerprint_agent(event)
                    except Exception as ex:
                        if self.debug:
                            print(f"[AI Telemetry] Agent fingerprinting error (non-fatal): {ex}")

                    # Module 5 — behavioral drift detection
                    if event.agent_id:
                        try:
                            # Derive ZASP safety score and attach to event
                            zasp = derive_zasp_score(event)
                            event.zasp_score = zasp

                            # Gather cluster means from agents with the same role
                            fp_store = get_fingerprint_store()
                            from ai_telemetry.drift_detector import get_drift_store
                            drift_store = get_drift_store()
                            cluster_means = []
                            if event.agent_role:
                                for rec in fp_store.all_records():
                                    if rec.agent_id == event.agent_id:
                                        continue
                                    if rec.agent_role != event.agent_role:
                                        continue
                                    peer_b = drift_store.get_or_create(rec.agent_id)
                                    if peer_b.recent_safety_scores:
                                        cluster_means.append(
                                            sum(peer_b.recent_safety_scores) / len(peer_b.recent_safety_scores)
                                        )

                            tool_scope = list(
                                fp_store.get(event.agent_id).tool_scope
                                if fp_store.get(event.agent_id) else []
                            )
                            drift_score, drift_class, drift_alerts = analyze_drift(
                                event, event.agent_id, tool_scope, cluster_means
                            )
                            event.drift_score = drift_score
                            event.drift_classification = drift_class
                            event.drift_signals = "|".join(drift_alerts) if drift_alerts else None
                        except Exception as ex:
                            if self.debug:
                                print(f"[AI Telemetry] Drift detection error (non-fatal): {ex}")

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

    @staticmethod
    def _existing_csv_header(path: str) -> List[str]:
        """The column names already written to ``path``, or [] if unreadable.

        Reads one line. Never raises: a header we cannot read is treated the same
        as no header at all, and the caller falls back to the current shape.
        """
        try:
            with open(path, newline='', encoding='utf-8') as fh:
                first = fh.readline()
            if not first.strip():
                return []
            # ﻿: a BOM on column 1 would make "event_id" compare unequal.
            return [c.strip().lstrip('﻿') for c in next(csv.reader([first]))]
        except (OSError, UnicodeDecodeError, StopIteration, csv.Error):
            return []

    def _write_csv(self, events: List[TelemetryEvent]):
        headers = [
            "event_id", "timestamp_iso", "event_type", "trace_id", "span_id", "parent_span_id",
            "model", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms",
            "user_id", "email", "display_name", "department", "job_title", "sailpoint_risk",
            "pii_detected", "pii_types", "trust_boundary_crossed", "risk_signal",
            "zendata_composite_risk", "zendata_classification", "data_sources", "tools_called",
            "entity_type", "entity_subtype", "classification_signals", "behavioral_alerts",
            "session_turn_index", "inter_turn_latency_ms",
            # Module 1 — Signal Capture additions
            "api_key_hash", "system_prompt_hash", "ip_hash", "http_headers_digest",
            # Module 2 — Human vs. Non-Human Classification
            "caller_type_score",
            # Module 3 — Agent Fingerprinting
            "agent_id", "agent_display_name", "agent_inferred_type",
            "orchestration_role", "prompt_template_hash", "system_prompt_version_count",
            # Module 4 — Dynamic Role Inference
            "human_inferred_role", "human_role_confidence",
            "agent_role", "agent_risk_level", "agent_primary_risk", "role_scope_gap",
            # Module 5 — Behavioral Drift Detection
            "zasp_score", "drift_score", "drift_classification", "drift_signals",
            # Build provenance — APPENDED AT THE END, deliberately. Every reader
            # of this file is positional once the header is parsed, so inserting
            # a column mid-list shifts every column after it for anyone holding
            # an older header. Growing only at the tail is the one change that
            # leaves existing columns where they were.
            "repo", "commit_sha", "build_id",
        ]

        log_file = self.log_path
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Write the header when the file carries NO DATA — not merely when it is
        # absent. An existing-but-empty file still needs one, and that case is
        # not hypothetical: ai_tester.py clears this exact path with
        # open(p, "w") before a run, which leaves a 0-byte file behind. Keying
        # off os.path.isfile() therefore suppressed the header for every
        # container run, and every consumer downstream reads a CSV header
        # positionally — csv.DictReader in the Elastic indexer, and the
        # gateway dashboard's _parseCSV — so they silently took the FIRST EVENT
        # as their column names and every row came back with user_id undefined.
        # A headerless telemetry file is indistinguishable from a corrupt one;
        # size, not existence, is the question worth asking.
        try:
            needs_header = os.path.getsize(log_file) == 0
        except OSError:
            needs_header = True   # missing file — created in append mode below

        # Reconcile against a header this file ALREADY has. csv.DictWriter emits
        # positionally from `fieldnames` and knows nothing about the file it is
        # appending to: given a 51-column header on disk and a 54-name fieldnames
        # list, it does not raise — it writes 54-field rows under a 51-field
        # header, and csv.DictReader then buries the extra values in row[None].
        # Every consumer doing row.get("repo") silently gets "", which is exactly
        # the invisible binding failure this whole feature exists to eliminate.
        # So: when the file on disk disagrees, write in ITS shape and say so.
        if not needs_header:
            existing = self._existing_csv_header(log_file)
            if existing and existing != headers:
                if self.debug:
                    missing = [h for h in headers if h not in existing]
                    print(f"[AI Telemetry] {log_file} was written with a different "
                          f"column set; appending in the file's shape. Columns that "
                          f"cannot be recorded: {missing or 'none'}")
                headers = existing

        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            # extrasaction="ignore": a row may legitimately carry keys the FILE's
            # header lacks (see reconciliation above). Dropping them keeps the
            # rows aligned, which matters more than recording a late column.
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=',',
                                    extrasaction='ignore')
            if needs_header:
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
                    # Module 1 additions
                    "api_key_hash": e.api_key_hash or "",
                    "system_prompt_hash": e.system_prompt_hash or "",
                    "ip_hash": e.ip_hash or "",
                    "http_headers_digest": e.http_headers_digest or "",
                    # Module 2 addition
                    "caller_type_score": e.caller_type_score if e.caller_type_score is not None else "",
                    # Module 3 additions
                    "agent_id": e.agent_id or "",
                    "agent_display_name": e.agent_display_name or "",
                    "agent_inferred_type": e.agent_inferred_type or "",
                    "orchestration_role": e.orchestration_role or "",
                    "prompt_template_hash": e.prompt_template_hash or "",
                    "system_prompt_version_count": e.system_prompt_version_count if e.system_prompt_version_count is not None else "",
                    # Module 5 additions
                    "zasp_score": e.zasp_score if e.zasp_score is not None else "",
                    "drift_score": e.drift_score if e.drift_score is not None else "",
                    "drift_classification": e.drift_classification or "",
                    "drift_signals": e.drift_signals or "",
                    # Module 4 additions
                    "human_inferred_role": e.human_inferred_role or "",
                    "human_role_confidence": e.human_role_confidence if e.human_role_confidence is not None else "",
                    "agent_role": e.agent_role or "",
                    "agent_risk_level": e.agent_risk_level or "",
                    "agent_primary_risk": e.agent_primary_risk or "",
                    "role_scope_gap": e.role_scope_gap if e.role_scope_gap is not None else "",
                    # Build provenance
                    "repo": e.repo or "",
                    "commit_sha": e.commit_sha or "",
                    "build_id": e.build_id or "",
                    # system_prompt_embedding is omitted from CSV (float[] — stored in JSONL only)
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