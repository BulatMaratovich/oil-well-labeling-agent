from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data_utils import load_tabular_file, parse_datetime_series
from app.models import new_id
from core.canonical_schema import MaintenanceDocument

_TEXT_HINTS = (
    "raw_text",
    "text",
    "report",
    "description",
    "comment",
    "comments",
    "notes",
    "maintenance_text",
    "summary",
    "details",
    "event_text",
    "опис",
    "текст",
    "коммент",
    "примеч",
    "отчет",
    "отчёт",
)
_ASSET_HINTS = (
    "asset_id",
    "asset",
    "well_id",
    "well",
    "well_name",
    "скваж",
    "object",
)
_DATE_HINTS = (
    "event_date",
    "date",
    "datetime",
    "event_time",
    "timestamp",
    "дата",
    "время",
)
_SOURCE_HINTS = (
    "source",
    "origin",
    "document_type",
    "doc_type",
    "sheet",
)
_EVENT_HINTS = ("event_type", "type", "event", "тип")
_ACTION_HINTS = ("action_summary", "action", "operation", "work", "работ")
_PARTS_HINTS = ("parts_replaced", "part", "parts", "component", "детал", "узел")
_TEXT_EXTENSIONS = {".txt", ".md", ".log"}
_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_maintenance_documents(
    filename: str,
    payload: bytes,
    *,
    fallback_asset_id: Optional[str] = None,
) -> list[MaintenanceDocument]:
    resolved_name = filename or "maintenance.txt"
    suffix = Path(resolved_name).suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("Файл технического контекста пуст.")
        return [
            MaintenanceDocument(
                doc_id=new_id(),
                asset_id=fallback_asset_id or "unknown",
                event_date=datetime.now(tz=timezone.utc),
                raw_text=text,
                source=f"upload:{resolved_name}",
            )
        ]

    if suffix in _TABULAR_EXTENSIONS:
        frame = load_tabular_file(resolved_name, payload)
        documents = _documents_from_frame(frame, resolved_name, fallback_asset_id=fallback_asset_id)
        if documents:
            return documents
        raise ValueError(
            "Не удалось собрать maintenance-документы из файла. "
            "Нужен хотя бы один непустой текст/описание или заполняемые строки."
        )

    raise ValueError("Поддерживаются TXT/MD/LOG/CSV/Excel файлы для maintenance-контекста.")


def serialize_maintenance_document(doc: MaintenanceDocument) -> dict[str, object]:
    return {
        "doc_id": doc.doc_id,
        "asset_id": doc.asset_id,
        "event_date": doc.event_date.isoformat(),
        "raw_text": doc.raw_text,
        "source": doc.source,
    }


def _documents_from_frame(
    frame: pd.DataFrame,
    filename: str,
    *,
    fallback_asset_id: Optional[str] = None,
) -> list[MaintenanceDocument]:
    prepared = frame.copy()
    prepared.columns = [str(column) for column in prepared.columns]
    visible_columns = [column for column in prepared.columns if not column.startswith("__")]
    if not visible_columns:
        return []

    text_column = _pick_column(visible_columns, _TEXT_HINTS)
    asset_column = _pick_column(visible_columns, _ASSET_HINTS)
    date_column = _pick_column(visible_columns, _DATE_HINTS)
    source_column = _pick_column(visible_columns, _SOURCE_HINTS)
    event_column = _pick_column(visible_columns, _EVENT_HINTS)
    action_column = _pick_column(visible_columns, _ACTION_HINTS)
    parts_column = _pick_column(visible_columns, _PARTS_HINTS)

    parsed_dates = (
        parse_datetime_series(prepared[date_column])
        if date_column and date_column in prepared.columns
        else pd.Series([pd.NaT] * len(prepared), index=prepared.index)
    )

    documents: list[MaintenanceDocument] = []
    for index, row in prepared.iterrows():
        raw_text = _row_to_text(
            row,
            visible_columns,
            text_column=text_column,
            extra_columns=[
                asset_column,
                date_column,
                event_column,
                action_column,
                parts_column,
            ],
        )
        if not raw_text:
            continue

        raw_asset_id = _string_value(row.get(asset_column)) if asset_column else None
        asset_id = raw_asset_id or fallback_asset_id or "unknown"

        parsed_date = parsed_dates.loc[index] if index in parsed_dates.index else pd.NaT
        if pd.isna(parsed_date):
            event_date = datetime.now(tz=timezone.utc)
        else:
            event_date = parsed_date.to_pydatetime()

        source = (
            _string_value(row.get(source_column))
            if source_column
            else None
        ) or _string_value(row.get("__sheet_name")) or f"upload:{filename}"

        documents.append(
            MaintenanceDocument(
                doc_id=new_id(),
                asset_id=asset_id,
                event_date=event_date,
                raw_text=raw_text,
                source=source,
            )
        )

    return documents


def _pick_column(columns: list[str], hints: tuple[str, ...]) -> Optional[str]:
    lowered = {column: column.lower() for column in columns}
    for hint in hints:
        for column, normalized in lowered.items():
            if normalized == hint or hint in normalized:
                return column
    return None


def _row_to_text(
    row: pd.Series,
    visible_columns: list[str],
    *,
    text_column: Optional[str],
    extra_columns: list[Optional[str]],
) -> str:
    parts: list[str] = []

    if text_column:
        primary_text = _string_value(row.get(text_column))
        if primary_text:
            parts.append(primary_text)

    for column in extra_columns:
        if not column or column == text_column:
            continue
        value = _string_value(row.get(column))
        if value:
            parts.append(f"{column}: {value}")

    if not parts:
        for column in visible_columns:
            value = _string_value(row.get(column))
            if value:
                parts.append(f"{column}: {value}")

    return "\n".join(parts).strip()


def _string_value(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
