"""
Interactive CSV labeling session agent.

Manages the multi-turn conversation for labeling a CSV file window by window:
  1. Parse CSV, detect columns (asks Claude if ambiguous)
  2. Extract per-device time series
  3. Run automatic window detection
  4. For each window: propose label, accept user feedback, refine, confirm
  5. Export labeled CSV

The agent maintains a full conversation history per session, allowing it to
refine its reasoning based on user corrections and domain context.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic

from src.models.schemas import ScenarioConfig
from src.orchestrator import LabelingPipeline, PipelineResult, load_scenario
from src.tools.window_detector import AnomalousWindow, detect_windows, extract_series, parse_csv

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """You are an expert industrial equipment diagnostics specialist helping \
to label anomalous windows in power consumption time-series data from electric motors.

You are assisting a user through an interactive labeling session. For each anomalous window:
1. Explain your proposed label clearly and concisely
2. Mention the key evidence (level ratio, amplitude ratio, transition pattern)
3. Acknowledge competing explanations if any
4. If the user provides additional context or corrections, update your reasoning

Be conversational but precise. Use bullet points for evidence. Keep explanations under 150 words.
When you need to revise a label based on user feedback, clearly state what changed and why.

Always end your proposal with a structured summary line:
LABEL: <label> | CONFIDENCE: <0.00-1.00> | ROUTING: <AUTO_LABEL|REVIEW|MANDATORY_REVIEW>"""


@dataclass
class WindowLabel:
    """Label result for one anomalous window."""
    window_idx: int
    label: str
    confidence: float
    routing: str
    explanation: str
    confirmed: bool = False
    user_overridden: bool = False


@dataclass
class CsvLabelingSession:
    """
    Manages the full interactive labeling session for a CSV file.

    State flow:
        init → csv_parsed → windows_detected → labeling → done
    """
    session_id: str
    scenario_path: str
    db_path: str = "data/profiles.db"

    # Set during csv_parsed phase
    csv_rows: list[dict] = field(default_factory=list)
    csv_columns: list[str] = field(default_factory=list)
    timestamp_col: Optional[str] = None
    value_col: Optional[str] = None
    device_col: Optional[str] = None

    # Set during windows_detected phase
    device_series: dict = field(default_factory=dict)  # {device_id: (ts_list, val_list)}
    all_windows: list[AnomalousWindow] = field(default_factory=list)

    # Labeling progress
    current_window_idx: int = 0
    labels: dict[int, WindowLabel] = field(default_factory=dict)
    pipeline_results: dict[int, PipelineResult] = field(default_factory=dict)

    # Chat history per window (window_idx → messages)
    window_chat: dict[int, list[dict]] = field(default_factory=dict)

    # Global chat messages (column detection phase, etc.)
    global_chat: list[dict] = field(default_factory=list)

    # Status
    status: str = "init"  # init|csv_loaded|windows_detected|labeling|done
    error: Optional[str] = None

    # Scenario (loaded on demand)
    _scenario: Optional[ScenarioConfig] = field(default=None, repr=False)
    _pipeline: Optional[LabelingPipeline] = field(default=None, repr=False)
    _client: Optional[anthropic.Anthropic] = field(default=None, repr=False)

    def __post_init__(self):
        self._client = anthropic.Anthropic()

    @property
    def scenario(self) -> ScenarioConfig:
        if self._scenario is None:
            self._scenario = load_scenario(self.scenario_path)
        return self._scenario

    @property
    def pipeline(self) -> LabelingPipeline:
        if self._pipeline is None:
            self._pipeline = LabelingPipeline(db_path=self.db_path)
        return self._pipeline

    # ── Phase 1: CSV column detection ─────────────────────────────────────

    def load_csv(self, file_path: str) -> dict:
        """
        Parse CSV and use Claude to identify timestamp and value columns.
        Returns a chat message to show the user.
        """
        columns, rows = parse_csv(file_path)
        self.csv_rows = rows
        self.csv_columns = columns

        # Show Claude the first few rows to identify columns
        sample_rows = rows[:5]
        sample_text = "\n".join(
            str(dict(row)) for row in sample_rows
        )

        prompt = f"""I have loaded a CSV file with these columns: {columns}

Sample rows (first 5):
{sample_text}

Identify which columns contain:
1. Timestamps (datetime values)
2. Power/value readings (numeric, the signal we're analyzing)
3. Device/equipment identifier (optional — if multiple devices are in the file)

Respond in JSON:
{{
  "timestamp_col": "<column name>",
  "value_col": "<column name>",
  "device_col": "<column name or null>",
  "explanation": "<brief explanation of your choice>"
}}

If you cannot identify the columns clearly, set the column to null and explain."""

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system="You are a data analyst. Respond only with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        try:
            col_info = json.loads(text)
        except json.JSONDecodeError:
            col_info = {
                "timestamp_col": None,
                "value_col": None,
                "device_col": None,
                "explanation": "Could not parse column identification. Please specify manually.",
            }

        self.timestamp_col = col_info.get("timestamp_col")
        self.value_col = col_info.get("value_col")
        self.device_col = col_info.get("device_col")

        explanation = col_info.get("explanation", "")

        msg = f"CSV loaded with {len(rows)} rows and {len(columns)} columns: {', '.join(columns)}.\n\n"

        if self.timestamp_col and self.value_col:
            msg += f"Detected columns:\n"
            msg += f"• **Timestamps**: `{self.timestamp_col}`\n"
            msg += f"• **Values**: `{self.value_col}`\n"
            if self.device_col:
                msg += f"• **Device ID**: `{self.device_col}`\n"
            msg += f"\n{explanation}\n\nIs this correct? Reply **yes** to continue or specify the correct columns."
            self.status = "csv_loaded"
        else:
            msg += f"Could not automatically identify columns. {explanation}\n\n"
            msg += f"Please specify: which column contains timestamps and which contains the power values?"

        self.global_chat.append({"role": "assistant", "content": msg})
        return {"message": msg, "columns_detected": self.timestamp_col is not None and self.value_col is not None}

    def set_columns(self, timestamp_col: str, value_col: str, device_col: Optional[str] = None) -> str:
        """Manually set column mapping."""
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.device_col = device_col
        self.status = "csv_loaded"
        msg = f"Columns set: timestamps=`{timestamp_col}`, values=`{value_col}`" + (
            f", device=`{device_col}`" if device_col else ""
        ) + ". Ready to detect anomalous windows."
        self.global_chat.append({"role": "assistant", "content": msg})
        return msg

    # ── Phase 2: Window detection ──────────────────────────────────────────

    def detect_windows_all(self) -> dict:
        """
        Extract time series and detect anomalous windows across all devices.
        Returns summary dict.
        """
        if not self.timestamp_col or not self.value_col:
            raise ValueError("Columns not set. Call load_csv() or set_columns() first.")

        self.device_series = extract_series(
            rows=self.csv_rows,
            timestamp_col=self.timestamp_col,
            value_col=self.value_col,
            device_col=self.device_col,
        )

        self.all_windows = []
        device_summaries = []

        for device_id, (timestamps, values) in self.device_series.items():
            windows = detect_windows(timestamps, values, device_id)
            self.all_windows.extend(windows)
            device_summaries.append(f"• **{device_id}**: {len(timestamps)} samples, {len(windows)} anomalous windows detected")

            # Build device profile from the non-anomalous portions
            if len(values) > 30:
                try:
                    self.pipeline.store.update_profile_samples(
                        device_id=device_id,
                        new_samples=values,
                    )
                except Exception as e:
                    logger.warning("Could not update profile for %s: %s", device_id, e)

        # Sort windows by time
        self.all_windows.sort(key=lambda w: w.start_time)

        total = len(self.all_windows)
        summary = f"**Window detection complete.**\n\n" + "\n".join(device_summaries)
        summary += f"\n\n**Total: {total} anomalous windows** to label."

        if total > 0:
            self.status = "labeling"
            self.current_window_idx = 0
            summary += f"\n\nStarting with window 1 of {total}..."
        else:
            self.status = "done"
            summary += "\n\nNo anomalous windows found. The signal appears normal throughout."

        self.global_chat.append({"role": "assistant", "content": summary})
        return {
            "message": summary,
            "total_windows": total,
            "devices": list(self.device_series.keys()),
        }

    # ── Phase 3: Interactive window labeling ───────────────────────────────

    def get_window_proposal(self, window_idx: int) -> dict:
        """
        Run the labeling pipeline for the given window and generate a chat proposal.

        Returns dict with message, label, confidence, routing.
        """
        if window_idx >= len(self.all_windows):
            return {"message": "All windows labeled.", "done": True}

        win = self.all_windows[window_idx]
        device_id = win.device_id
        timestamps, values = self.device_series[device_id]

        # Extract the window's time series
        win_ts = timestamps[win.start_idx:win.end_idx + 1]
        win_vals = values[win.start_idx:win.end_idx + 1]
        pre_ts = timestamps[win.context_start_idx:win.start_idx]
        pre_vals = values[win.context_start_idx:win.start_idx]

        from src.models.schemas import TimeSeriesWindow

        window_obj = TimeSeriesWindow(
            device_id=device_id,
            timestamps=win_ts,
            values=win_vals,
            scenario_id=self.scenario.scenario_id,
            pre_timestamps=pre_ts,
            pre_values=pre_vals,
        )

        try:
            result = self.pipeline.run(window=window_obj, scenario=self.scenario)
            self.pipeline_results[window_idx] = result
        except Exception as e:
            logger.error("Pipeline failed for window %d: %s", window_idx, e)
            result = None

        # Build conversational proposal message
        total = len(self.all_windows)
        time_fmt = "%Y-%m-%d %H:%M"
        start_str = win.start_time.strftime(time_fmt)
        end_str = win.end_time.strftime(time_fmt)

        msg = f"### Window {window_idx + 1} of {total}: {device_id}\n"
        msg += f"**Time range**: {start_str} → {end_str}\n"
        msg += f"**Type detected**: `{win.anomaly_type}` (score: {win.anomaly_score:.2f})\n\n"

        if result:
            p = result.proposal
            msg += f"**My analysis:**\n"
            msg += f"{p.explanation}\n\n"
            msg += f"**Evidence:**\n"
            for ev in p.evidence:
                msg += f"• {ev}\n"
            msg += f"\n**Physical plausibility:** {p.physical_plausibility}\n\n"
            if result.similar_devices:
                msg += f"*Based on {len(result.similar_devices)} similar devices in the knowledge base.*\n\n"

            routing_str = p.routing.value
            msg += f"LABEL: {p.label} | CONFIDENCE: {p.confidence:.2f} | ROUTING: {routing_str}"

            # Store preliminary label
            self.labels[window_idx] = WindowLabel(
                window_idx=window_idx,
                label=p.label,
                confidence=p.confidence,
                routing=routing_str,
                explanation=p.explanation,
            )

            if window_idx not in self.window_chat:
                self.window_chat[window_idx] = []
            self.window_chat[window_idx].append({"role": "assistant", "content": msg})

            return {
                "message": msg,
                "label": p.label,
                "confidence": p.confidence,
                "routing": routing_str,
                "window_idx": window_idx,
            }
        else:
            msg += "⚠ Pipeline failed for this window. Please provide a label manually or skip."
            if window_idx not in self.window_chat:
                self.window_chat[window_idx] = []
            self.window_chat[window_idx].append({"role": "assistant", "content": msg})
            return {"message": msg, "label": "uncertain", "confidence": 0.0, "routing": "MANDATORY_REVIEW", "window_idx": window_idx}

    def process_user_message(self, window_idx: int, user_message: str) -> dict:
        """
        Process a user message for the current window and return agent response.

        Handles:
        - Confirmations ("yes", "confirm", "ok")
        - Corrections ("actually this is X", "I think it's Y because Z")
        - Questions ("why do you think...", "what does level_ratio mean")
        - Skip ("skip", "next")
        """
        if window_idx not in self.window_chat:
            self.window_chat[window_idx] = []

        self.window_chat[window_idx].append({"role": "user", "content": user_message})

        # Check for simple confirmation
        lower = user_message.lower().strip()
        if lower in ("yes", "confirm", "ok", "correct", "agree", "да", "подтвердить", "верно"):
            return self._confirm_current_label(window_idx)

        # Check for skip
        if lower in ("skip", "next", "пропустить", "следующее"):
            return self._skip_window(window_idx)

        # Otherwise: have a conversation with Claude about this window
        return self._chat_about_window(window_idx, user_message)

    def _confirm_current_label(self, window_idx: int) -> dict:
        """Confirm the current label and advance to next window."""
        if window_idx in self.labels:
            self.labels[window_idx].confirmed = True

        next_idx = window_idx + 1
        total = len(self.all_windows)

        if next_idx >= total:
            msg = "✓ Label confirmed. All windows have been labeled! You can now export the results."
            self.status = "done"
        else:
            msg = f"✓ Label confirmed. Moving to window {next_idx + 1} of {total}..."
            self.current_window_idx = next_idx

        self.window_chat[window_idx].append({"role": "assistant", "content": msg})
        return {
            "message": msg,
            "confirmed": True,
            "next_window_idx": next_idx if next_idx < total else None,
            "done": next_idx >= total,
        }

    def _skip_window(self, window_idx: int) -> dict:
        """Skip this window (mark as uncertain) and advance."""
        self.labels[window_idx] = WindowLabel(
            window_idx=window_idx,
            label="uncertain",
            confidence=0.0,
            routing="MANDATORY_REVIEW",
            explanation="Skipped by user.",
            confirmed=True,
        )

        next_idx = window_idx + 1
        total = len(self.all_windows)

        if next_idx >= total:
            msg = "Window skipped. All windows processed! You can now export the results."
            self.status = "done"
        else:
            msg = f"Window skipped. Moving to window {next_idx + 1} of {total}..."
            self.current_window_idx = next_idx

        self.window_chat[window_idx].append({"role": "assistant", "content": msg})
        return {
            "message": msg,
            "confirmed": True,
            "next_window_idx": next_idx if next_idx < total else None,
            "done": next_idx >= total,
        }

    def _chat_about_window(self, window_idx: int, user_message: str) -> dict:
        """
        Multi-turn conversation about the current window using Claude.
        May revise the label based on user feedback.
        """
        win = self.all_windows[window_idx] if window_idx < len(self.all_windows) else None
        result = self.pipeline_results.get(window_idx)

        # Build context message for Claude
        context_parts = [
            f"You are helping label anomalous window {window_idx + 1} of {len(self.all_windows)}.",
        ]

        if win:
            context_parts.append(
                f"Window: device={win.device_id}, "
                f"time={win.start_time.strftime('%Y-%m-%d %H:%M')} → {win.end_time.strftime('%Y-%m-%d %H:%M')}, "
                f"type={win.anomaly_type}, level_ratio={win.level_ratio:.3f}, amplitude_ratio={win.amplitude_ratio:.3f}"
            )

        if result:
            p = result.proposal
            context_parts.append(
                f"Current label proposal: '{p.label}' (confidence={p.confidence:.2f})\n"
                f"Explanation: {p.explanation}"
            )

        if self.scenario:
            context_parts.append(
                f"Scenario: {self.scenario.name}\n"
                f"Target label: {self.scenario.target_label}\n"
                f"Competing labels: {', '.join(cl.label for cl in self.scenario.competing_labels)}"
            )

        system_with_context = _SYSTEM_PROMPT + "\n\nCurrent window context:\n" + "\n".join(context_parts)

        # Include full conversation history for this window
        messages = list(self.window_chat[window_idx])

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=system_with_context,
            messages=messages,
        )

        reply = response.content[0].text

        # Try to parse revised label from the structured summary line
        revised_label = None
        revised_confidence = None
        revised_routing = None

        for line in reply.split("\n"):
            if line.startswith("LABEL:"):
                parts = line.split("|")
                try:
                    revised_label = parts[0].replace("LABEL:", "").strip()
                    revised_confidence = float(parts[1].replace("CONFIDENCE:", "").strip())
                    revised_routing = parts[2].replace("ROUTING:", "").strip()
                except (IndexError, ValueError):
                    pass

        # Update label if Claude revised it
        if revised_label and window_idx in self.labels:
            old_label = self.labels[window_idx].label
            self.labels[window_idx].label = revised_label
            self.labels[window_idx].confidence = revised_confidence or self.labels[window_idx].confidence
            self.labels[window_idx].routing = revised_routing or self.labels[window_idx].routing
            self.labels[window_idx].explanation = reply
            if revised_label != old_label:
                self.labels[window_idx].user_overridden = True

        self.window_chat[window_idx].append({"role": "assistant", "content": reply})

        return {
            "message": reply,
            "label_revised": revised_label is not None,
            "label": revised_label or (self.labels[window_idx].label if window_idx in self.labels else None),
            "window_idx": window_idx,
        }

    # ── Phase 4: Export ────────────────────────────────────────────────────

    def export_labeled_csv(self) -> str:
        """
        Generate labeled CSV content.

        Returns the CSV as a string with added columns:
        label, label_confidence, label_routing, label_explanation
        """
        if not self.csv_rows:
            raise ValueError("No CSV data to export")

        # Build a lookup: for each row index, which window does it belong to?
        row_labels: dict[int, WindowLabel] = {}

        if self.timestamp_col and self.value_col:
            # Re-parse to get row-to-index mapping
            from dateutil import parser as dateparser
            device_row_indices: dict[str, list[int]] = {}  # device_id → list of global row indices

            for row_idx, row in enumerate(self.csv_rows):
                device_id = row.get(self.device_col, "device_001") if self.device_col else "device_001"
                if device_id not in device_row_indices:
                    device_row_indices[device_id] = []
                device_row_indices[device_id].append(row_idx)

            # Map window indices to row indices
            for win_idx, win in enumerate(self.all_windows):
                if win_idx not in self.labels:
                    continue
                wlabel = self.labels[win_idx]
                device_id = win.device_id
                row_indices = device_row_indices.get(device_id, [])

                # Mark all rows within [start_idx, end_idx] of this device's series
                for local_idx in range(win.start_idx, win.end_idx + 1):
                    if local_idx < len(row_indices):
                        global_idx = row_indices[local_idx]
                        row_labels[global_idx] = wlabel

        # Build output CSV
        output = io.StringIO()
        fieldnames = list(self.csv_columns) + ["label", "label_confidence", "label_routing", "label_explanation"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(self.csv_rows):
            out_row = dict(row)
            wlabel = row_labels.get(row_idx)
            if wlabel:
                out_row["label"] = wlabel.label
                out_row["label_confidence"] = f"{wlabel.confidence:.3f}"
                out_row["label_routing"] = wlabel.routing
                # Truncate explanation to avoid very long cells
                explanation = wlabel.explanation.replace("\n", " ")[:200]
                out_row["label_explanation"] = explanation
            else:
                out_row["label"] = ""
                out_row["label_confidence"] = ""
                out_row["label_routing"] = ""
                out_row["label_explanation"] = ""
            writer.writerow(out_row)

        return output.getvalue()

    def get_state_snapshot(self) -> dict:
        """
        Return a full state snapshot for the browser to render.

        Used by the polling endpoint to keep the browser in sync.
        """
        windows_summary = []
        for idx, win in enumerate(self.all_windows):
            wlabel = self.labels.get(idx)
            windows_summary.append({
                "idx": idx,
                "device_id": win.device_id,
                "start_time": win.start_time.isoformat(),
                "end_time": win.end_time.isoformat(),
                "start_idx": win.start_idx,
                "end_idx": win.end_idx,
                "context_start_idx": win.context_start_idx,
                "context_end_idx": win.context_end_idx,
                "anomaly_type": win.anomaly_type,
                "anomaly_score": win.anomaly_score,
                "level_ratio": win.level_ratio,
                "amplitude_ratio": win.amplitude_ratio,
                "label": wlabel.label if wlabel else None,
                "confidence": wlabel.confidence if wlabel else None,
                "routing": wlabel.routing if wlabel else None,
                "confirmed": wlabel.confirmed if wlabel else False,
            })

        # Serialize device series for chart (downsample if large)
        series_data = {}
        for device_id, (timestamps, values) in self.device_series.items():
            ts_strs = [t.isoformat() for t in timestamps]
            series_data[device_id] = {"timestamps": ts_strs, "values": values}

        chat_all = list(self.global_chat)
        if self.current_window_idx in self.window_chat:
            chat_all.extend(self.window_chat[self.current_window_idx])

        return {
            "status": self.status,
            "current_window_idx": self.current_window_idx,
            "total_windows": len(self.all_windows),
            "windows": windows_summary,
            "series": series_data,
            "chat": chat_all,
            "columns": {
                "timestamp": self.timestamp_col,
                "value": self.value_col,
                "device": self.device_col,
            },
            "labeled_count": sum(1 for w in windows_summary if w["confirmed"]),
            "error": self.error,
        }

    def close(self) -> None:
        if self._pipeline:
            self._pipeline.close()
