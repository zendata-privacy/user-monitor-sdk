"""Save and restore the SDK's learned in-process state.

Several detectors learn over many calls: per-agent drift baselines (100 calls to
activate), agent fingerprints (tool scope, prompt versions, temporal rhythm),
human role evidence, and per-identity risk baselines. All of it lives in
process memory, so a process that is restarted — a redeploy, or a batch job
that runs once a day — starts every detector from zero and the signals that
need history can never fire.

    from ai_telemetry import persistence
    persistence.load_state("/data/sdk_state.json")   # before the first event
    ...
    persistence.save_state("/data/sdk_state.json")   # at shutdown

JSON, not pickle: a state file is data and can never execute code when loaded.
Sessions (one conversation's turns) are deliberately not persisted — a
conversation does not survive a restart, and neither should its state.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import typing
from typing import Any, Dict

FORMAT_VERSION = 1


# ── generic dataclass <-> JSON ──────────────────────────────────────────────

def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (set, frozenset)):
        return sorted(_encode(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    return obj


def _decode(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:                       # Optional[X]
        inner = [a for a in args if a is not type(None)]
        return _decode(inner[0], value) if inner else value
    if dataclasses.is_dataclass(tp):
        return _build(tp, value)
    if origin in (list, typing.List):
        return [_decode(args[0] if args else Any, v) for v in value]
    if origin in (set, typing.Set, frozenset):
        return {_decode(args[0] if args else Any, v) for v in value}
    if origin in (dict, typing.Dict):
        vt = args[1] if len(args) == 2 else Any
        return {k: _decode(vt, v) for k, v in value.items()}
    return value


def _build(cls, data: Dict[str, Any]):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _decode(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)                             # unknown keys from newer versions are ignored


# ── the stores ──────────────────────────────────────────────────────────────

def snapshot() -> Dict[str, Any]:
    """The learned state, as a JSON-ready dict."""
    from ai_telemetry.agent_fingerprint import get_fingerprint_store
    from ai_telemetry.drift_detector import get_drift_store
    from ai_telemetry.role_inference import get_human_role_store
    from ai_telemetry.session_state import get_session_store

    drift, fp, roles, sessions = (get_drift_store(), get_fingerprint_store(),
                                  get_human_role_store(), get_session_store())
    with drift._lock:
        drift_items = list(drift._baselines.values())
    with fp._lock:
        fp_items = list(fp._records.values())
    with roles._lock:
        role_items = list(roles._records.values())
        role_sessions = {k: sorted(v) for k, v in roles._sessions.items()}
    with sessions._lock:
        baselines = list(sessions._baselines.values())
    return {
        "format": FORMAT_VERSION,
        "drift_baselines": [_encode(b) for b in drift_items],
        "agent_fingerprints": [_encode(r) for r in fp_items],
        "human_roles": [_encode(r) for r in role_items],
        "human_role_sessions": role_sessions,
        "entity_risk_baselines": [_encode(b) for b in baselines],
    }


def restore(state: Dict[str, Any]) -> Dict[str, int]:
    """Load a snapshot into the stores. Returns counts restored per store."""
    from ai_telemetry.agent_fingerprint import AgentRecord, get_fingerprint_store
    from ai_telemetry.drift_detector import AgentDriftBaseline, get_drift_store
    from ai_telemetry.role_inference import HumanRoleRecord, get_human_role_store
    from ai_telemetry.session_state import EntityBaseline, get_session_store

    if not state or state.get("format") != FORMAT_VERSION:
        return {}
    drift, fp, roles, sessions = (get_drift_store(), get_fingerprint_store(),
                                  get_human_role_store(), get_session_store())
    counts = {"drift_baselines": 0, "agent_fingerprints": 0, "human_roles": 0,
              "entity_risk_baselines": 0}
    for d in state.get("drift_baselines", []):
        b = _build(AgentDriftBaseline, d)
        drift.update(b)
        counts["drift_baselines"] += 1
    for d in state.get("agent_fingerprints", []):
        r = _build(AgentRecord, d)
        with fp._lock:
            fp._records[r.agent_id] = r
        counts["agent_fingerprints"] += 1
    for d in state.get("human_roles", []):
        r = _build(HumanRoleRecord, d)
        roles.update(r)
        counts["human_roles"] += 1
    with roles._lock:
        for k, v in (state.get("human_role_sessions") or {}).items():
            roles._sessions[k] = set(v)
    for d in state.get("entity_risk_baselines", []):
        b = _build(EntityBaseline, d)
        with sessions._lock:
            sessions._baselines[b.entity_key] = b
        counts["entity_risk_baselines"] += 1
    return counts


def save_state(path: str) -> None:
    """Write the snapshot atomically (a crash mid-write leaves the old file)."""
    data = json.dumps(snapshot())
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".sdk_state.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(data)
    os.replace(tmp, path)


def load_state(path: str) -> Dict[str, int]:
    """Restore from a file written by save_state. Missing/invalid file: nothing restored."""
    try:
        with open(path, encoding="utf-8") as fh:
            return restore(json.load(fh))
    except (OSError, ValueError):
        return {}


__all__ = ["snapshot", "restore", "save_state", "load_state"]
