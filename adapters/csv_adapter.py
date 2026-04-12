"""
adapters/csv_adapter.py — CSV / Excel export-import adapter.

Converts between the internal LabelRecord / SavedAnnotation types and
flat CSV / Excel files suitable for downstream analysis or re-import.

Public API
----------
    export_label_records(records, path)   → writes CSV
    export_annotations(annotations, path) → writes CSV (app-level SavedAnnotation)
    import_label_records(path)            → list[dict]  (raw, for audit)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import csv
import io


# ---------------------------------------------------------------------------
# Export: LabelRecord (pipeline output)
# ---------------------------------------------------------------------------

LABEL_RECORD_COLUMNS = [
    "record_id", "task_id", "asset_id",
    "segment_start", "segment_end",
    "deviation_type", "final_label", "was_override",
    "correction_reason", "rule_label", "winning_rule",
    "conflict", "abstain_reason",
    "confirmed_at", "run_id", "status",
    # local features
    "power_mean", "power_std", "power_p10", "power_p90",
    "zero_fraction", "transition_sharpness", "segment_duration_h",
]


def export_label_records(
    records: list,  # list[LabelRecord]
    path: str | Path,
    *,
    fmt: str = "csv",
) -> Path:
    """Write *records* to *path* as CSV (or xlsx if fmt='xlsx').

    Returns the resolved Path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = [_label_record_to_row(r) for r in records]

    if fmt == "xlsx":
        _write_xlsx(rows, LABEL_RECORD_COLUMNS, target)
    else:
        _write_csv(rows, LABEL_RECORD_COLUMNS, target)

    return target


def _label_record_to_row(rec) -> dict[str, Any]:
    f = rec.local_features
    rt = rec.rule_result.rule_trace
    return {
        "record_id": rec.record_id,
        "task_id": rec.task_id,
        "asset_id": rec.asset_id,
        "segment_start": _iso(rec.segment.start),
        "segment_end": _iso(rec.segment.end),
        "deviation_type": rec.deviation_type,
        "final_label": rec.final_label,
        "was_override": rec.was_override,
        "correction_reason": rec.correction_reason or "",
        "rule_label": rec.rule_result.label,
        "winning_rule": rt.winning_rule or "",
        "conflict": rt.conflict,
        "abstain_reason": rec.rule_result.abstain_reason or "",
        "confirmed_at": _iso(rec.confirmed_at),
        "run_id": rec.run_id or "",
        "status": rec.status,
        "power_mean": _fmt(f.power_mean if f else None),
        "power_std": _fmt(f.power_std if f else None),
        "power_p10": _fmt(f.power_p10 if f else None),
        "power_p90": _fmt(f.power_p90 if f else None),
        "zero_fraction": _fmt(f.zero_fraction if f else None),
        "transition_sharpness": _fmt(f.transition_sharpness if f else None),
        "segment_duration_h": _fmt(f.segment_duration_h if f else None),
    }


# ---------------------------------------------------------------------------
# Export: SavedAnnotation (app-level)
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "annotation_id", "filename", "well_column", "well_value",
    "recommendation_mode", "x", "x_end", "y", "trace_name",
    "series", "window_size", "date_from", "date_to",
    "label", "correction_reason", "created_at",
]


def export_annotations(
    annotations: list,  # list[SavedAnnotation]
    path: str | Path,
    *,
    fmt: str = "csv",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [_annotation_to_row(a) for a in annotations]
    if fmt == "xlsx":
        _write_xlsx(rows, ANNOTATION_COLUMNS, target)
    else:
        _write_csv(rows, ANNOTATION_COLUMNS, target)
    return target


def _annotation_to_row(ann) -> dict[str, Any]:
    return {
        "annotation_id": ann.annotation_id,
        "filename": ann.filename or "",
        "well_column": ann.well_column or "",
        "well_value": ann.well_value or "",
        "recommendation_mode": ann.recommendation_mode,
        "x": ann.x or "",
        "x_end": ann.x_end or "",
        "y": _fmt(ann.y),
        "trace_name": ann.trace_name or "",
        "series": "|".join(ann.series) if ann.series else "",
        "window_size": ann.window_size or "",
        "date_from": ann.date_from or "",
        "date_to": ann.date_to or "",
        "label": ann.label or "",
        "correction_reason": ann.correction_reason or "",
        "created_at": ann.created_at or "",
    }


# ---------------------------------------------------------------------------
# Import (round-trip / audit)
# ---------------------------------------------------------------------------

def import_label_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV exported by export_label_records() and return raw dicts."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    with target.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


# ---------------------------------------------------------------------------
# Low-level writers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx(rows: list[dict], columns: list[str], path: Path) -> None:
    try:
        import openpyxl  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("openpyxl is required for xlsx export") from exc

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(columns)
    for row in rows:
        ws.append([row.get(c, "") for c in columns])
    wb.save(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt) -> str:
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val:.6g}"
