"""
core/state_manager.py — Run State checkpoint / resume.

Implements the RunState schema from agent-orchestrator.md:
  - Checkpoints each completed stage to data/tasks/<task_id>/runs/<run_id>/state.json
  - Allows resumption from the last completed stage
  - Stores lightweight stage summaries (not full DataFrames)

RunState fields:
  run_id, task_id, asset_id, filename, started_at, updated_at,
  completed_stages, last_stage, status, stage_summaries, errors
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from observability.logger import get_logger

log = get_logger(__name__)

TASKS_DIR = Path("data/tasks")

ALL_STAGES = [
    "input_normalizer",
    "signal_sanitizer",
    "global_series_profiler",
    "historical_profile_builder",
    "candidate_event_detector",
    "local_segment_analyzer",
    "context_fact_extractor",
    "rule_engine",
]


# ---------------------------------------------------------------------------
# RunState dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    run_id: str
    task_id: str
    asset_id: str
    filename: str
    started_at: str
    updated_at: str
    status: str = "running"          # "running" | "complete" | "failed" | "paused"
    last_stage: Optional[str] = None
    completed_stages: list[str] = field(default_factory=list)
    stage_summaries: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def next_stage(self) -> Optional[str]:
        """Return the first stage not yet completed, or None if all done."""
        for stage in ALL_STAGES:
            if stage not in self.completed_stages:
                return stage
        return None

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def resumable(self) -> bool:
        return self.status in ("running", "paused") and bool(self.completed_stages)


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------

class StateManager:
    """Checkpoint / resume manager for a single run."""

    def __init__(self, task_id: str, run_id: str) -> None:
        self.task_id = task_id
        self.run_id = run_id
        self._path = TASKS_DIR / task_id / "runs" / run_id / "state.json"
        self._state: Optional[RunState] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new_run(
        cls,
        task_id: str,
        run_id: str,
        asset_id: str,
        filename: str,
    ) -> "StateManager":
        """Create a new StateManager and initialise an empty RunState."""
        mgr = cls(task_id, run_id)
        now = _now()
        mgr._state = RunState(
            run_id=run_id,
            task_id=task_id,
            asset_id=asset_id,
            filename=filename,
            started_at=now,
            updated_at=now,
        )
        mgr._save()
        log.info("run_started", task_id=task_id, run_id=run_id, asset_id=asset_id)
        return mgr

    @classmethod
    def resume(cls, task_id: str, run_id: str) -> Optional["StateManager"]:
        """Load an existing RunState. Returns None if not found."""
        mgr = cls(task_id, run_id)
        if not mgr._path.exists():
            return None
        mgr._load()
        if mgr._state and mgr._state.resumable:
            log.info("run_resumed", task_id=task_id, run_id=run_id,
                     last_stage=mgr._state.last_stage)
            return mgr
        return None

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def mark_stage_complete(
        self,
        stage: str,
        summary: Optional[dict] = None,
    ) -> None:
        """Record that *stage* completed successfully."""
        state = self._require_state()
        if stage not in state.completed_stages:
            state.completed_stages.append(stage)
        state.last_stage = stage
        if summary:
            state.stage_summaries[stage] = summary
        state.updated_at = _now()
        self._save()
        log.info("stage_checkpointed", task_id=self.task_id,
                 run_id=self.run_id, stage=stage)

    def mark_stage_failed(self, stage: str, error: str) -> None:
        state = self._require_state()
        state.errors.append(f"{stage}: {error}")
        state.status = "failed"
        state.updated_at = _now()
        self._save()
        log.error("stage_failed_checkpoint", task_id=self.task_id,
                  run_id=self.run_id, stage=stage, error=error)

    def add_warning(self, message: str) -> None:
        self._require_state().warnings.append(message)

    def complete(self) -> None:
        state = self._require_state()
        state.status = "complete"
        state.updated_at = _now()
        self._save()
        log.info("run_complete", task_id=self.task_id, run_id=self.run_id)

    def pause(self) -> None:
        state = self._require_state()
        state.status = "paused"
        state.updated_at = _now()
        self._save()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[RunState]:
        return self._state

    def stage_done(self, stage: str) -> bool:
        return self._state is not None and stage in self._state.completed_stages

    def get_summary(self, stage: str) -> Optional[dict]:
        if self._state is None:
            return None
        return self._state.stage_summaries.get(stage)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(self._state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._state = RunState(**{
                k: v for k, v in data.items()
                if k in RunState.__dataclass_fields__
            })
        except Exception as exc:
            log.warning("state_load_failed", path=str(self._path), error=str(exc))
            self._state = None

    def _require_state(self) -> RunState:
        if self._state is None:
            raise RuntimeError("RunState not initialised. Call new_run() first.")
        return self._state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def list_runs(task_id: str) -> list[dict]:
    """List all run state files for *task_id*, newest first."""
    runs_dir = TASKS_DIR / task_id / "runs"
    if not runs_dir.exists():
        return []
    result = []
    for state_file in sorted(runs_dir.glob("*/state.json"), reverse=True):
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            result.append({
                "run_id": data.get("run_id"),
                "status": data.get("status"),
                "last_stage": data.get("last_stage"),
                "updated_at": data.get("updated_at"),
                "asset_id": data.get("asset_id"),
                "n_errors": len(data.get("errors", [])),
            })
        except Exception:
            pass
    return result
