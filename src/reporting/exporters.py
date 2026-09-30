"""
LibraryGuard - Export events to CSV / JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .events import EventReport

from ..behavior.engine import BehaviorEvent

CSV_COLUMNS = [
    "timestamp",
    "timestamp_end",
    "event",
    "person_id",
    "confidence",
    "label",
    "duration",
    "frame_start",
    "frame_end",
]


def events_to_dataframe(events: List[BehaviorEvent]):
    """Convert events to a pandas DataFrame (pandas is a project dependency)."""
    import pandas as pd

    rows = [e.to_dict() for e in events]
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def export_csv(events: List[BehaviorEvent], path: Union[str, Path]) -> Path:
    """Write events to a CSV file. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for event in events:
            row = event.to_dict()
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    return out


def export_json(report: "EventReport", path: Union[str, Path], indent: int = 2) -> Path:
    """Write a full :class:`EventReport` to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=indent, ensure_ascii=False)
    return out


def export_events_json(events: List[BehaviorEvent], path: Union[str, Path], indent: int = 2) -> Path:
    """Write just the event list to JSON (schema-compatible with docs)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_count": len(events),
        "events": [e.to_dict() for e in events],
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
    return out