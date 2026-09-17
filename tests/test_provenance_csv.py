"""Build provenance on the event, and the CSV sink that carries it.

`_write_csv` had no test coverage at all before this file, which mattered once it
started carrying provenance: the CSV is the transport the correlator actually
reads (events.csv -> S3 -> csv.DictReader), and its failure mode is silent.

`csv.DictWriter` emits positionally from `fieldnames` and knows nothing about the
file it is appending to. Given a short header on disk and a longer `fieldnames`,
it does not raise — it writes long rows under a short header, and `DictReader`
then buries the surplus in `row[None]`. Every consumer doing `row.get("repo")`
gets "", the binding silently fails, and the report shows an unstamped build. The
tests below pin the reconciliation that prevents it.
"""
import csv
import os
import tempfile

import pytest

from ai_telemetry.client import TelemetryClient, TelemetryEvent
from ai_telemetry import provenance

REPO = "zendata-privacy/financial-chatbot-scan-sdk-pii"
COMMIT = "31f2730a732b9e230329d2afc32f8eecd4997e3f"


@pytest.fixture(autouse=True)
def _clean_provenance_env(monkeypatch):
    """Provenance is lru_cached and env-driven; isolate every test from both."""
    for var in ("ZENDATA_REPO", "ZENDATA_COMMIT", "ZENDATA_BUILD_ID",
                "ZENDATA_PROVENANCE_GIT", "ZENDATA_APP_ROOT",
                "GITHUB_REPOSITORY", "GITHUB_SHA"):
        monkeypatch.delenv(var, raising=False)
    provenance.current_provenance.cache_clear()
    provenance._own_repo.cache_clear()
    yield
    provenance.current_provenance.cache_clear()
    provenance._own_repo.cache_clear()


@pytest.fixture
def csv_client():
    made = []

    def _make(**kw):
        path = os.path.join(tempfile.mkdtemp(), "events.csv")
        c = TelemetryClient(app_id="test-app", api_key="sk-test",
                            endpoint="http://127.0.0.1:1", flush_interval_s=999,
                            enable_local_log=True, log_path=path, **kw)
        made.append(c)
        return c, path

    yield _make
    for c in made:
        c._shutdown.set()


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── stamping ────────────────────────────────────────────────────────────────

def test_capture_stamps_provenance_from_the_build_env(monkeypatch, csv_client):
    monkeypatch.setenv("ZENDATA_REPO", REPO)
    monkeypatch.setenv("ZENDATA_COMMIT", COMMIT)
    monkeypatch.setenv("ZENDATA_BUILD_ID", "42")
    provenance.current_provenance.cache_clear()

    client, _ = csv_client()
    ev = TelemetryEvent(event_type="tool_call", tool_name="query_financials")
    client.capture(ev)

    assert ev.repo == REPO
    assert ev.commit_sha == COMMIT
    assert ev.build_id == "42"


def test_an_unstamped_build_reports_nothing_rather_than_guessing(csv_client):
    """Absence must stay absence: the correlator treats a missing repo as
    "not stamped" and falls back to weaker binding sources. A guess here would
    become `sdk_provenance`, which outranks everything but a human override."""
    client, _ = csv_client()
    ev = TelemetryEvent(event_type="llm_call")
    client.capture(ev)
    assert ev.repo is None and ev.commit_sha is None


def test_an_explicitly_set_repo_is_not_overwritten(monkeypatch, csv_client):
    monkeypatch.setenv("ZENDATA_REPO", REPO)
    provenance.current_provenance.cache_clear()
    client, _ = csv_client()
    ev = TelemetryEvent(event_type="llm_call", repo="caller/override")
    client.capture(ev)
    assert ev.repo == "caller/override"


def test_provenance_reaches_to_dict_for_the_http_and_jsonl_sinks(monkeypatch, csv_client):
    """Both non-CSV transports serialize to_dict(), so they need no changes —
    this pins that they actually carry the fields."""
    monkeypatch.setenv("ZENDATA_REPO", REPO)
    monkeypatch.setenv("ZENDATA_COMMIT", COMMIT)
    provenance.current_provenance.cache_clear()
    client, _ = csv_client()
    ev = TelemetryEvent(event_type="llm_call")
    client.capture(ev)
    d = ev.to_dict()
    assert d["repo"] == REPO and d["commit_sha"] == COMMIT


# ── the CSV sink ────────────────────────────────────────────────────────────

def test_csv_carries_provenance_columns_aligned(monkeypatch, csv_client):
    monkeypatch.setenv("ZENDATA_REPO", REPO)
    monkeypatch.setenv("ZENDATA_COMMIT", COMMIT)
    provenance.current_provenance.cache_clear()

    client, path = csv_client()
    ev = TelemetryEvent(event_type="tool_call", user_id="usr_1")
    client.capture(ev)
    client._write_csv([ev])

    rows = _rows(path)
    assert len(rows) == 1
    # A None key is csv.DictReader's restkey: it means the row had MORE fields
    # than the header, i.e. the file is ragged.
    assert None not in rows[0], "row has more fields than the header — misaligned"
    assert rows[0]["repo"] == REPO
    assert rows[0]["commit_sha"] == COMMIT
    assert rows[0]["user_id"] == "usr_1"


def test_provenance_columns_are_appended_at_the_end(csv_client):
    """Position is a compatibility contract: every reader is positional once the
    header is parsed, so a column inserted mid-list shifts everything after it
    for anyone holding an older header."""
    client, path = csv_client()
    ev = TelemetryEvent(event_type="llm_call")
    client.capture(ev)
    client._write_csv([ev])
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[-3:] == ["repo", "commit_sha", "build_id"]


def test_header_is_written_once_not_per_flush(csv_client):
    client, path = csv_client()
    for _ in range(3):
        ev = TelemetryEvent(event_type="llm_call")
        client.capture(ev)
        client._write_csv([ev])
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 4                       # 1 header + 3 events
    assert lines[0].startswith("event_id,")


def test_an_empty_file_still_gets_a_header(csv_client):
    """ai_tester.py truncates this path with open(p, "w") before a run, leaving a
    0-byte file. Keying off existence rather than size suppressed the header for
    every container run, and every downstream reader then took the first EVENT as
    its column names."""
    client, path = csv_client()
    open(path, "w").close()
    ev = TelemetryEvent(event_type="llm_call")
    client.capture(ev)
    client._write_csv([ev])
    assert _rows(path)[0]["event_type"] == "llm_call"


def test_appending_to_an_old_short_header_never_produces_ragged_rows(csv_client):
    """THE regression this reconciliation exists for.

    A file written by a previous SDK version has fewer columns. Without
    reconciliation DictWriter appends longer rows under it, and every consumer
    silently reads "" for the provenance it can see in the file.
    """
    client, path = csv_client()
    old = ["event_id", "timestamp_iso", "event_type", "user_id"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=old)
        w.writeheader()
        w.writerow({"event_id": "old1", "timestamp_iso": "t",
                    "event_type": "llm_call", "user_id": "u0"})

    ev = TelemetryEvent(event_type="tool_call", user_id="u1")
    client.capture(ev)
    client._write_csv([ev])

    rows = _rows(path)
    assert len(rows) == 2
    for row in rows:
        assert None not in row, "ragged row — DictWriter wrote past the header"
        assert len(row) == len(old)
    assert rows[1]["user_id"] == "u1"


def test_a_header_we_cannot_read_falls_back_to_the_current_shape(csv_client):
    client, path = csv_client()
    with open(path, "wb") as fh:
        fh.write(b"\xff\xfe not valid utf-8 \xff\n")
    assert client._existing_csv_header(path) == []


def test_a_bom_does_not_defeat_header_comparison(csv_client):
    """A UTF-8 BOM on column 1 would make "event_id" compare unequal, and the
    writer would silently switch to the file's shape for no reason."""
    client, path = csv_client()
    ev = TelemetryEvent(event_type="llm_call")
    client.capture(ev)
    client._write_csv([ev])
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("﻿" + body)
    assert client._existing_csv_header(path)[0] == "event_id"
