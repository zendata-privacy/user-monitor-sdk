"""Turn events received over the HTTP transport back into the SDK's events.csv.

The Zendata gateway's /api/telemetry/ingest stores each SDK flush exactly as
sent — ``TelemetryEvent.to_dict()`` objects. Everything downstream (the
Autonomous Workflow dashboard, the correlation job, the AIBOM evidence export)
reads events.csv, so the scan job has to rebuild that file. It does so with the
SDK's OWN writer, ``TelemetryClient._write_csv``, rather than a second copy of
the column list: two writers would drift, and the dashboard reads columns
positionally once the header is parsed.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Iterable, List

from ai_telemetry.client import TelemetryClient, TelemetryEvent

_FIELDS = {f.name for f in dataclasses.fields(TelemetryEvent)}


def event_from_dict(d: dict) -> TelemetryEvent:
    """Rebuild a TelemetryEvent from its to_dict() form. Unknown keys (e.g. a
    newer SDK's fields, or the Node SDK's app_id) are dropped, not fatal."""
    kwargs = {k: v for k, v in d.items() if k in _FIELDS}
    ts = kwargs.get("timestamp_ms")
    if ts is not None:
        try:
            kwargs["timestamp_ms"] = int(float(ts))
        except (TypeError, ValueError):
            kwargs.pop("timestamp_ms")
    return TelemetryEvent(**kwargs)


def write_events_csv(events: Iterable[dict], path: str, debug: bool = False) -> int:
    """Append ``events`` (to_dict() objects) to ``path`` in the SDK's CSV
    format, writing the header if the file is empty or absent. Returns the
    number of rows written."""
    rebuilt: List[TelemetryEvent] = [event_from_dict(e) for e in events if isinstance(e, dict)]
    if not rebuilt:
        return 0
    rebuilt.sort(key=lambda e: e.timestamp_ms or 0)
    writer = SimpleNamespace(log_path=path, debug=debug,
                             _existing_csv_header=TelemetryClient._existing_csv_header)
    TelemetryClient._write_csv(writer, rebuilt)
    return len(rebuilt)
