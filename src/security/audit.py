"""Privacy-minimizing JSONL audit events (protect the log at rest)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .canonical import canonical_json_bytes


def append_audit_event(path: str | Path, operation: str, outcome: str, *, actor_id: str,
                       source_sha256: str | None = None, output_sha256: str | None = None,
                       details: dict | None = None) -> dict:
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "operation": operation,
        "outcome": outcome,
        "actor_id": actor_id,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "details": details or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab") as stream:
        stream.write(canonical_json_bytes(event) + b"\n")
        stream.flush()
    return event
