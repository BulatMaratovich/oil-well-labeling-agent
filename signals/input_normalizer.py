"""
signals/input_normalizer.py — Stage 1: Input Normalizer.

Loads a raw tabular file (CSV / Excel), applies schema normalization
(time parsing, column selection, well filtering), and emits one
CanonicalTimeSeries per selected signal from the TaskSpec.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from app.data_utils import load_tabular_file, parse_datetime_series
from core.canonical_schema import CanonicalTimeSeries
from core.task_manager import TaskSpec, SignalSpec


FileInput = Union[Path, str, bytes]


class NormalizationError(ValueError):
    """Raised when the input file cannot be normalized."""


def normalize(
    file_input: FileInput,
    task_spec: TaskSpec,
    *,
    filename: str = "data.csv",
    asset_id: Optional[str] = None,
) -> list[CanonicalTimeSeries]:
    """Load *file_input* and return one :class:`CanonicalTimeSeries` per
    selected signal in *task_spec*.

    Parameters
    ----------
    file_input:
        Raw bytes, a file path (``str`` or :class:`~pathlib.Path`), or the
        contents already loaded as ``bytes``.
    task_spec:
        Describes which columns to use and which signals are selected.
    filename:
        Original file name — used only for format detection when
        *file_input* is ``bytes``.
    asset_id:
        Identifier to embed in the result.  Falls back to
        ``task_spec.well_column`` value after filtering, or ``"unknown"``.
    """
    # ------------------------------------------------------------------
    # 1. Load raw DataFrame
    # ------------------------------------------------------------------
    df = _load(file_input, filename)

    time_col = task_spec.time_column
    well_col = task_spec.well_column

    if not time_col or time_col not in df.columns:
        available = ", ".join(df.columns.tolist()[:10])
        raise NormalizationError(
            f"Time column '{time_col}' not found in file. "
            f"Available columns: {available}"
        )

    # ------------------------------------------------------------------
    # 2. Filter to a single asset (well) when well_column is set
    # ------------------------------------------------------------------
    if well_col and well_col in df.columns and asset_id:
        df = df[df[well_col].astype(str) == str(asset_id)].copy()
        if df.empty:
            raise NormalizationError(
                f"No rows found for asset '{asset_id}' in column '{well_col}'."
            )
    elif well_col and well_col in df.columns:
        # Infer asset_id from the first (or only) unique value
        unique_wells = df[well_col].dropna().astype(str).unique().tolist()
        if len(unique_wells) == 1:
            asset_id = unique_wells[0]
        elif not asset_id:
            # Multi-well file — caller must supply asset_id
            raise NormalizationError(
                f"Multiple assets found in column '{well_col}': "
                f"{unique_wells[:5]}... Supply 'asset_id' to filter."
            )

    effective_asset_id = asset_id or "unknown"

    # ------------------------------------------------------------------
    # 3. Parse & sort timestamps; drop rows with unparseable time
    # ------------------------------------------------------------------
    df = df.copy()
    df[time_col] = parse_datetime_series(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if df.empty:
        raise NormalizationError("All rows were dropped after datetime parsing.")

    # ------------------------------------------------------------------
    # 4. Build one CanonicalTimeSeries per selected signal
    # ------------------------------------------------------------------
    selected_signals = _selected_signals(task_spec)
    if not selected_signals:
        raise NormalizationError(
            "No signals are marked selected_for_review in the TaskSpec."
        )

    results: list[CanonicalTimeSeries] = []
    for sig in selected_signals:
        if sig.name not in df.columns:
            # Skip missing columns with a warning rather than hard-failing
            continue
        col_df = df[[time_col, sig.name]].copy()
        col_df[sig.name] = pd.to_numeric(col_df[sig.name], errors="coerce")
        results.append(
            CanonicalTimeSeries(
                asset_id=effective_asset_id,
                timestamp_col=time_col,
                signal_col=sig.name,
                unit=sig.unit,
                values=col_df,
            )
        )

    if not results:
        missing = [s.name for s in selected_signals if s.name not in df.columns]
        raise NormalizationError(
            f"None of the selected signals were found in the file. "
            f"Missing: {missing}"
        )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(file_input: FileInput, filename: str) -> pd.DataFrame:
    if isinstance(file_input, bytes):
        return load_tabular_file(filename, file_input)
    path = Path(file_input)
    if not path.exists():
        raise NormalizationError(f"File not found: {path}")
    payload = path.read_bytes()
    return load_tabular_file(path.name, payload)


def _selected_signals(task_spec: TaskSpec) -> list[SignalSpec]:
    return [s for s in task_spec.signal_schema if s.selected_for_review]
