"""ai_telemetry.export — HTTP-transport events back to the SDK's events.csv.

The guarantee that matters: an event that travelled SDK -> HTTP JSON -> Zendata
ingest -> write_events_csv produces the SAME csv bytes the SDK would have
written locally. The dashboard reads that file positionally.
"""
import csv
import json
from types import SimpleNamespace

from ai_telemetry.client import TelemetryClient, TelemetryEvent
from ai_telemetry.export import event_from_dict, write_events_csv


def _events():
    return [
        TelemetryEvent(event_id="e-1", event_type="user_message", user_id="u1", timestamp_ms=1_700_000_000_000,
                       prompt_tokens=4, total_tokens=4, department="Risk"),
        TelemetryEvent(event_id="e-2", event_type="llm_call", user_id="u1", model="gpt-4o", timestamp_ms=1_700_000_001_000,
                       pii_detected=True, pii_types="EMAIL", zendata_composite_risk=72.5,
                       metadata={"note": "x"}, entity_type="AGENT"),
    ]


def _direct_csv(events, path):
    shim = SimpleNamespace(log_path=str(path), debug=False,
                           _existing_csv_header=TelemetryClient._existing_csv_header)
    TelemetryClient._write_csv(shim, events)
    return path.read_bytes()


def test_round_trip_is_byte_identical(tmp_path):
    direct = _direct_csv(_events(), tmp_path / "direct.csv")
    wire = [json.loads(json.dumps(e.to_dict())) for e in _events()]   # what /ingest stores
    n = write_events_csv(wire, str(tmp_path / "rebuilt.csv"))
    assert n == 2
    assert (tmp_path / "rebuilt.csv").read_bytes() == direct


def test_header_written_once_and_sorted_by_time(tmp_path):
    out = tmp_path / "e.csv"
    late, early = [e.to_dict() for e in _events()][::-1], None
    write_events_csv(late, str(out))
    write_events_csv([_events()[0].to_dict()], str(out))
    rows = list(csv.DictReader(out.open()))
    assert [r["event_type"] for r in rows] == ["user_message", "llm_call", "user_message"]
    assert out.read_text().count("event_id,timestamp_iso") == 1


def test_unknown_keys_and_bad_values_are_tolerated(tmp_path):
    e = event_from_dict({"event_type": "llm_call", "user_id": "u", "app_id": "node-only",
                         "timestamp_ms": "1700000000000.0", "brand_new_field": 1})
    assert e.timestamp_ms == 1_700_000_000_000 and e.user_id == "u"
    assert write_events_csv([{"event_type": "x"}, "not-a-dict"], str(tmp_path / "e.csv")) == 1
    assert write_events_csv([], str(tmp_path / "empty.csv")) == 0
