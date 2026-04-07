"""
Alert Dispatcher — Module 6

Delivers drift and behavioral alerts through multiple channels:
  - Dashboard inline feed (always active — no config required)
  - Webhook POST        JSON payload to any HTTP endpoint
  - SIEM CEF            ArcSight Common Event Format (Splunk, Sentinel, Chronicle)
  - SIEM LEEF           IBM QRadar Log Event Extended Format
  - Email digest        SMTP (plain-text summary, batched by default)
  - PagerDuty           Events API v2 (CRITICAL tier only by default)
  - OpsGenie            Alerts API v2 (CRITICAL tier only by default)

Usage::

    from ai_telemetry.alert_dispatcher import AlertDispatcher, AlertConfig

    dispatcher = AlertDispatcher(AlertConfig(
        webhook_url="https://hooks.example.com/zendata",
        siem_format="CEF",
        siem_host="splunk.corp.internal",
        siem_port=514,
        email_to=["security@corp.com"],
        email_smtp_host="smtp.corp.com",
        pagerduty_routing_key="abc123...",
        opsgenie_api_key="xyz...",
        min_severity_webhook="WARNING",
        min_severity_pagerduty="CRITICAL",
    ))

    # Dispatch a single alert dict (from drift_alert_feed() or directly)
    dispatcher.dispatch(alert_payload)
"""

from __future__ import annotations

import json
import logging
import smtplib
import socket
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: Dict[str, int] = {
    "CRITICAL":   0,
    "WARNING":    1,
    "INVESTIGATE": 2,
    "MONITOR":    3,
    "STABLE":     4,
}

_CEF_SEVERITY_MAP: Dict[str, int] = {
    # ArcSight CEF severity 0-10
    "CRITICAL":    10,
    "WARNING":     7,
    "INVESTIGATE": 5,
    "MONITOR":     3,
    "STABLE":      0,
}

_PAGERDUTY_SEVERITY_MAP: Dict[str, str] = {
    "CRITICAL":    "critical",
    "WARNING":     "warning",
    "INVESTIGATE": "warning",
    "MONITOR":     "info",
    "STABLE":      "info",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AlertConfig:
    """Configuration for all alert channels.

    All fields are optional — only channels with credentials / URLs
    configured will fire.  Channels with no config are silently skipped.
    """

    # ---- Webhook ----
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_timeout_s: int = 5
    min_severity_webhook: str = "MONITOR"      # MONITOR | INVESTIGATE | WARNING | CRITICAL

    # ---- SIEM (UDP syslog) ----
    siem_format: Optional[str] = None          # "CEF" | "LEEF" | None
    siem_host: Optional[str] = None
    siem_port: int = 514
    siem_timeout_s: int = 3

    # ---- Email ----
    email_to: List[str] = field(default_factory=list)
    email_from: str = "zendata-alerts@noreply.local"
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_smtp_user: Optional[str] = None
    email_smtp_password: Optional[str] = None
    email_use_tls: bool = True
    email_batch: bool = True                   # batch into digest (flush every N minutes)
    email_batch_interval_s: int = 300          # 5-minute digest window
    min_severity_email: str = "WARNING"

    # ---- PagerDuty ----
    pagerduty_routing_key: Optional[str] = None
    pagerduty_url: str = "https://events.pagerduty.com/v2/enqueue"
    min_severity_pagerduty: str = "CRITICAL"

    # ---- OpsGenie ----
    opsgenie_api_key: Optional[str] = None
    opsgenie_url: str = "https://api.opsgenie.com/v2/alerts"
    opsgenie_team: Optional[str] = None
    min_severity_opsgenie: str = "CRITICAL"

    # ---- IAM mode ----
    iam_mode: str = "NO-IAM"                   # NO-IAM | IAM-CONNECTED | IAM-PRIMARY

    def severity_passes(self, drift_class: str, min_severity: str) -> bool:
        """Return True when drift_class is at least as severe as min_severity."""
        return _SEVERITY_ORDER.get(drift_class, 99) <= _SEVERITY_ORDER.get(min_severity, 99)


# ---------------------------------------------------------------------------
# Alert Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """Thread-safe alert routing to all configured channels."""

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        self.config = config or AlertConfig()
        self._lock = threading.Lock()
        # Email digest buffer
        self._email_buffer: List[Dict] = []
        self._email_timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, alert: Dict) -> None:
        """Route a single alert to all enabled channels.

        ``alert`` is expected to contain at minimum:
          agent_id, drift_class (or drift_classification), drift_score,
          primary_signal, last_seen_iso, recommended_action.
        """
        cfg = self.config
        severity = alert.get("drift_class") or alert.get("drift_classification", "STABLE")

        threads = []

        # Webhook
        if cfg.webhook_url and cfg.severity_passes(severity, cfg.min_severity_webhook):
            threads.append(threading.Thread(
                target=self._webhook, args=(alert,), daemon=True))

        # SIEM
        if cfg.siem_format and cfg.siem_host and cfg.severity_passes(severity, "MONITOR"):
            threads.append(threading.Thread(
                target=self._siem, args=(alert, cfg.siem_format), daemon=True))

        # Email (batched)
        if cfg.email_to and cfg.email_smtp_host and cfg.severity_passes(severity, cfg.min_severity_email):
            self._buffer_email(alert)

        # PagerDuty
        if cfg.pagerduty_routing_key and cfg.severity_passes(severity, cfg.min_severity_pagerduty):
            threads.append(threading.Thread(
                target=self._pagerduty, args=(alert,), daemon=True))

        # OpsGenie
        if cfg.opsgenie_api_key and cfg.severity_passes(severity, cfg.min_severity_opsgenie):
            threads.append(threading.Thread(
                target=self._opsgenie, args=(alert,), daemon=True))

        for t in threads:
            t.start()

    def flush_email_digest(self) -> None:
        """Force-flush any buffered email alerts immediately."""
        with self._lock:
            if self._email_timer:
                self._email_timer.cancel()
                self._email_timer = None
            buf = list(self._email_buffer)
            self._email_buffer.clear()
        if buf:
            self._send_email_digest(buf)

    # ------------------------------------------------------------------
    # Channel: Webhook
    # ------------------------------------------------------------------

    def _webhook(self, alert: Dict) -> None:
        payload = json.dumps(alert, default=str).encode("utf-8")
        req = urllib.request.Request(
            url=self.config.webhook_url,
            data=payload,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("X-ZenData-Alert", "1")
        for k, v in self.config.webhook_headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.config.webhook_timeout_s):
                pass
            log.info("[AlertDispatcher] Webhook delivered: %s", alert.get("agent_id"))
        except urllib.error.URLError as exc:
            log.warning("[AlertDispatcher] Webhook failed: %s", exc)

    # ------------------------------------------------------------------
    # Channel: SIEM (CEF / LEEF over UDP syslog)
    # ------------------------------------------------------------------

    def _siem(self, alert: Dict, fmt: str) -> None:
        if fmt.upper() == "CEF":
            msg = self._format_cef(alert)
        elif fmt.upper() == "LEEF":
            msg = self._format_leef(alert)
        else:
            log.warning("[AlertDispatcher] Unknown SIEM format: %s", fmt)
            return

        self._udp_syslog(msg)

    def _format_cef(self, alert: Dict) -> str:
        """ArcSight Common Event Format (CEF:0)."""
        severity = alert.get("drift_class") or alert.get("drift_classification", "STABLE")
        cef_sev = _CEF_SEVERITY_MAP.get(severity, 3)
        agent_id = _cef_escape(str(alert.get("agent_id", "unknown")))
        signal = _cef_escape(str(alert.get("primary_signal", "DRIFT")))
        score = alert.get("drift_score", 0.0)
        name = _cef_escape(f"AgentDrift:{signal}")
        ext = (
            f"agentId={agent_id} "
            f"driftScore={score:.4f} "
            f"driftClass={severity} "
            f"signal={signal} "
            f"recommendedAction={_cef_escape(str(alert.get('recommended_action', '')))} "
            f"iamIdentity={_cef_escape(str(alert.get('iam_identity') or 'pseudonymous'))} "
            f"rt={int(time.time() * 1000)}"
        )
        # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        return (
            f"CEF:0|ZenData|BehavioralTelemetry|1.0|{signal}|{name}|{cef_sev}|{ext}"
        )

    def _format_leef(self, alert: Dict) -> str:
        """IBM QRadar Log Event Extended Format (LEEF:1.0)."""
        severity = alert.get("drift_class") or alert.get("drift_classification", "STABLE")
        ts = datetime.now(timezone.utc).strftime("%b %d %Y %H:%M:%S")
        agent_id = alert.get("agent_id", "unknown")
        signal = alert.get("primary_signal", "DRIFT")
        score = alert.get("drift_score", 0.0)
        fields = (
            f"devTime={ts}\t"
            f"agentId={agent_id}\t"
            f"driftScore={score:.4f}\t"
            f"driftClass={severity}\t"
            f"signal={signal}\t"
            f"severity={_SEVERITY_ORDER.get(severity, 99)}\t"
            f"iamIdentity={alert.get('iam_identity') or 'pseudonymous'}\t"
            f"recommendedAction={alert.get('recommended_action', '')}"
        )
        # LEEF:Version|Vendor|Product|Version|EventID|fields
        return f"LEEF:1.0|ZenData|BehavioralTelemetry|1.0|{signal}|{fields}"

    def _udp_syslog(self, message: str) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.siem_timeout_s)
            # RFC 3164 priority: facility=16 (local0), severity=5 (notice)
            pri = (16 * 8) + 5
            syslog_msg = f"<{pri}>{message}"
            sock.sendto(syslog_msg.encode("utf-8"),
                        (self.config.siem_host, self.config.siem_port))
            log.info("[AlertDispatcher] SIEM message sent to %s:%s",
                     self.config.siem_host, self.config.siem_port)
        except Exception as exc:
            log.warning("[AlertDispatcher] SIEM UDP failed: %s", exc)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Channel: Email digest
    # ------------------------------------------------------------------

    def _buffer_email(self, alert: Dict) -> None:
        with self._lock:
            self._email_buffer.append(alert)
            if not self.config.email_batch:
                buf = list(self._email_buffer)
                self._email_buffer.clear()
                threading.Thread(target=self._send_email_digest,
                                 args=(buf,), daemon=True).start()
                return
            # Schedule a flush if one isn't pending
            if self._email_timer is None:
                self._email_timer = threading.Timer(
                    self.config.email_batch_interval_s,
                    self._flush_email_timer,
                )
                self._email_timer.daemon = True
                self._email_timer.start()

    def _flush_email_timer(self) -> None:
        with self._lock:
            buf = list(self._email_buffer)
            self._email_buffer.clear()
            self._email_timer = None
        if buf:
            self._send_email_digest(buf)

    def _send_email_digest(self, alerts: List[Dict]) -> None:
        cfg = self.config
        if not cfg.email_to or not cfg.email_smtp_host:
            return

        critical = [a for a in alerts if a.get("drift_class") in ("CRITICAL", "WARNING")]
        body_lines = [
            "ZenData Behavioral Telemetry — Alert Digest",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Total alerts: {len(alerts)}  |  CRITICAL/WARNING: {len(critical)}",
            "",
        ]
        for a in alerts:
            body_lines += [
                f"  Agent    : {a.get('agent_id', '—')} ({a.get('display_name', '—')})",
                f"  Severity : {a.get('drift_class', '—')}  score={a.get('drift_score', 0):.4f}",
                f"  Signal   : {a.get('primary_signal', '—')}",
                f"  Action   : {a.get('recommended_action', '—')}",
                f"  Seen     : {a.get('last_seen_iso', '—')}",
                "",
            ]

        msg = MIMEText("\n".join(body_lines))
        subject_tier = "CRITICAL" if any(
            a.get("drift_class") == "CRITICAL" for a in alerts) else "WARNING"
        msg["Subject"] = (
            f"[ZenData {subject_tier}] {len(alerts)} Agent Drift Alerts"
        )
        msg["From"] = cfg.email_from
        msg["To"] = ", ".join(cfg.email_to)

        try:
            if cfg.email_use_tls:
                server = smtplib.SMTP(cfg.email_smtp_host, cfg.email_smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(cfg.email_smtp_host, cfg.email_smtp_port)
            if cfg.email_smtp_user and cfg.email_smtp_password:
                server.login(cfg.email_smtp_user, cfg.email_smtp_password)
            server.sendmail(cfg.email_from, cfg.email_to, msg.as_string())
            server.quit()
            log.info("[AlertDispatcher] Email digest sent to %s", cfg.email_to)
        except Exception as exc:
            log.warning("[AlertDispatcher] Email failed: %s", exc)

    # ------------------------------------------------------------------
    # Channel: PagerDuty Events API v2
    # ------------------------------------------------------------------

    def _pagerduty(self, alert: Dict) -> None:
        cfg = self.config
        severity = alert.get("drift_class") or alert.get("drift_classification", "STABLE")
        payload = {
            "routing_key": cfg.pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": f"zendata-{alert.get('agent_id', 'unknown')}-{alert.get('primary_signal', 'DRIFT')}",
            "payload": {
                "summary": (
                    f"[ZenData {severity}] Agent {alert.get('agent_id', '?')} "
                    f"— {alert.get('primary_signal', 'DRIFT')}"
                ),
                "source": "ZenData BehavioralTelemetry",
                "severity": _PAGERDUTY_SEVERITY_MAP.get(severity, "warning"),
                "timestamp": alert.get("last_seen_iso") or datetime.now(timezone.utc).isoformat(),
                "custom_details": {
                    "agent_id":          alert.get("agent_id"),
                    "display_name":      alert.get("display_name"),
                    "drift_score":       alert.get("drift_score"),
                    "drift_class":       severity,
                    "primary_signal":    alert.get("primary_signal"),
                    "recommended_action": alert.get("recommended_action"),
                    "iam_identity":      alert.get("iam_identity") or "pseudonymous",
                },
            },
        }
        self._http_post(cfg.pagerduty_url, payload, headers={}, label="PagerDuty")

    # ------------------------------------------------------------------
    # Channel: OpsGenie Alerts API v2
    # ------------------------------------------------------------------

    def _opsgenie(self, alert: Dict) -> None:
        cfg = self.config
        severity = alert.get("drift_class") or alert.get("drift_classification", "STABLE")
        payload: Dict = {
            "message": (
                f"[ZenData {severity}] Agent {alert.get('agent_id', '?')} "
                f"— {alert.get('primary_signal', 'DRIFT')}"
            ),
            "alias": f"zendata-{alert.get('agent_id', 'unknown')}-{alert.get('primary_signal', 'DRIFT')}",
            "description": alert.get("recommended_action", ""),
            "priority": "P1" if severity == "CRITICAL" else "P2",
            "details": {
                "agent_id":    str(alert.get("agent_id", "")),
                "drift_score": str(alert.get("drift_score", 0)),
                "drift_class": severity,
                "signal":      str(alert.get("primary_signal", "")),
            },
        }
        if cfg.opsgenie_team:
            payload["responders"] = [{"name": cfg.opsgenie_team, "type": "team"}]

        self._http_post(
            cfg.opsgenie_url,
            payload,
            headers={"Authorization": f"GenieKey {cfg.opsgenie_api_key}"},
            label="OpsGenie",
        )

    # ------------------------------------------------------------------
    # Shared HTTP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _http_post(url: str, body: Dict, headers: Dict, label: str,
                   timeout: int = 5) -> None:
        data = json.dumps(body, default=str).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                log.info("[AlertDispatcher] %s → HTTP %s", label, resp.status)
        except urllib.error.URLError as exc:
            log.warning("[AlertDispatcher] %s failed: %s", label, exc)


# ---------------------------------------------------------------------------
# Module-level singleton (configured lazily)
# ---------------------------------------------------------------------------

_global_dispatcher: Optional[AlertDispatcher] = None
_dispatcher_lock = threading.Lock()


def configure_dispatcher(config: AlertConfig) -> AlertDispatcher:
    """Set the module-level dispatcher.  Call once at startup."""
    global _global_dispatcher
    with _dispatcher_lock:
        _global_dispatcher = AlertDispatcher(config)
    return _global_dispatcher


def get_dispatcher() -> Optional[AlertDispatcher]:
    """Return the global dispatcher, or None if not configured."""
    return _global_dispatcher


def dispatch_alert(alert: Dict) -> None:
    """Dispatch via the global dispatcher if configured; log otherwise."""
    d = get_dispatcher()
    if d:
        d.dispatch(alert)
    else:
        log.debug("[AlertDispatcher] No dispatcher configured — alert: %s", alert)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cef_escape(s: str) -> str:
    """Escape CEF extension field values."""
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("=", "\\=").replace("\n", " ")


__all__ = [
    "AlertConfig",
    "AlertDispatcher",
    "configure_dispatcher",
    "get_dispatcher",
    "dispatch_alert",
]
