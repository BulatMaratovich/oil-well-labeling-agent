"""
FastAPI server for the interactive time-series labeling tool.

Endpoints:
  POST /sessions                      Create a new labeling session
  POST /sessions/{id}/upload          Upload CSV file
  POST /sessions/{id}/columns         Set/confirm column mapping
  POST /sessions/{id}/detect          Run window detection
  POST /sessions/{id}/windows/{idx}/propose  Get label proposal for window
  POST /sessions/{id}/chat            Send user message, get agent response
  POST /sessions/{id}/windows/{idx}/confirm  Confirm label for window
  GET  /sessions/{id}/state           Get full session state (for polling)
  GET  /sessions/{id}/export          Download labeled CSV

Run:
    python server.py
    # or
    uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.session_agent import CsvLabelingSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Time-Series Labeling Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# In-memory session store
_sessions: dict[str, CsvLabelingSession] = {}
_session_lock = threading.Lock()

DB_PATH = "data/profiles.db"
Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

DEFAULT_SCENARIO = "scenarios/belt_break.yaml"


# ── Request / Response models ─────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    scenario_path: str = DEFAULT_SCENARIO


class ColumnRequest(BaseModel):
    timestamp_col: str
    value_col: str
    device_col: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    window_idx: Optional[int] = None


class ConfirmRequest(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None


# ── Session helpers ───────────────────────────────────────────────────────────

def _get_session(session_id: str) -> CsvLabelingSession:
    with _session_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main labeling UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Static files not found. Run the full setup first.</h1>")


@app.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new labeling session."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    session_id = str(uuid.uuid4())[:8]
    session = CsvLabelingSession(
        session_id=session_id,
        scenario_path=request.scenario_path,
        db_path=DB_PATH,
    )

    with _session_lock:
        _sessions[session_id] = session

    logger.info("Created session %s with scenario %s", session_id, request.scenario_path)
    return {"session_id": session_id, "scenario": request.scenario_path}


@app.post("/sessions/{session_id}/upload")
async def upload_csv(
    session_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a CSV file and auto-detect columns."""
    session = _get_session(session_id)

    # Save uploaded file
    upload_path = Path("uploads") / f"{session_id}_{file.filename}"
    content = await file.read()
    upload_path.write_bytes(content)

    logger.info("Session %s: uploaded %s (%d bytes)", session_id, file.filename, len(content))

    try:
        result = session.load_csv(str(upload_path))
        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(session.csv_rows),
            "columns": session.csv_columns,
            "message": result["message"],
            "columns_detected": result["columns_detected"],
            "timestamp_col": session.timestamp_col,
            "value_col": session.value_col,
            "device_col": session.device_col,
        }
    except Exception as e:
        logger.error("Session %s: CSV load failed: %s", session_id, e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sessions/{session_id}/columns")
async def set_columns(session_id: str, request: ColumnRequest):
    """Manually set or confirm column mapping."""
    session = _get_session(session_id)
    msg = session.set_columns(
        timestamp_col=request.timestamp_col,
        value_col=request.value_col,
        device_col=request.device_col,
    )
    return {"message": msg, "status": session.status}


@app.post("/sessions/{session_id}/detect")
async def detect_windows(session_id: str, background_tasks: BackgroundTasks):
    """
    Trigger window detection (runs in background).
    Poll /state to get updates.
    """
    session = _get_session(session_id)

    if session.status not in ("csv_loaded",):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot detect windows in status '{session.status}'. Upload CSV first."
        )

    def _run_detection():
        try:
            session.status = "detecting"
            result = session.detect_windows_all()
            logger.info("Session %s: detection complete, %d windows", session_id, result["total_windows"])
        except Exception as e:
            logger.error("Session %s: detection failed: %s", session_id, e)
            session.error = str(e)
            session.status = "error"

    background_tasks.add_task(_run_detection)
    return {"message": "Detection started. Poll /state for updates.", "session_id": session_id}


@app.post("/sessions/{session_id}/windows/{window_idx}/propose")
async def propose_label(session_id: str, window_idx: int, background_tasks: BackgroundTasks):
    """
    Get a label proposal for the specified window.
    Runs the full pipeline (may take a few seconds).
    """
    session = _get_session(session_id)

    if window_idx >= len(session.all_windows):
        raise HTTPException(status_code=404, detail=f"Window {window_idx} not found")

    def _run_proposal():
        try:
            result = session.get_window_proposal(window_idx)
            logger.info(
                "Session %s window %d: label=%s confidence=%.2f",
                session_id, window_idx,
                result.get("label"), result.get("confidence", 0),
            )
        except Exception as e:
            logger.error("Session %s window %d: proposal failed: %s", session_id, window_idx, e)
            session.error = str(e)

    background_tasks.add_task(_run_proposal)
    return {"message": "Proposal started. Poll /state for updates.", "window_idx": window_idx}


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatRequest):
    """
    Send a user message for the current (or specified) window.
    Returns the agent's response.
    """
    session = _get_session(session_id)

    window_idx = request.window_idx if request.window_idx is not None else session.current_window_idx

    try:
        result = session.process_user_message(
            window_idx=window_idx,
            user_message=request.message,
        )
        return {
            "session_id": session_id,
            "window_idx": window_idx,
            "message": result["message"],
            "confirmed": result.get("confirmed", False),
            "next_window_idx": result.get("next_window_idx"),
            "done": result.get("done", False),
            "label": result.get("label"),
        }
    except Exception as e:
        logger.error("Session %s chat failed: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/windows/{window_idx}/confirm")
async def confirm_window(session_id: str, window_idx: int, request: ConfirmRequest):
    """Confirm (and optionally override) the label for a window."""
    session = _get_session(session_id)

    if window_idx >= len(session.all_windows):
        raise HTTPException(status_code=404, detail=f"Window {window_idx} not found")

    # Override label if provided
    if request.label and window_idx in session.labels:
        session.labels[window_idx].label = request.label
        if request.confidence is not None:
            session.labels[window_idx].confidence = request.confidence
        session.labels[window_idx].user_overridden = True

    result = session.process_user_message(window_idx=window_idx, user_message="confirm")
    return {
        "session_id": session_id,
        "window_idx": window_idx,
        "confirmed": True,
        "next_window_idx": result.get("next_window_idx"),
        "done": result.get("done", False),
    }


@app.get("/sessions/{session_id}/state")
async def get_state(session_id: str):
    """
    Get the full session state snapshot for browser rendering.
    Call this on a polling interval (every 2-3 seconds).
    """
    session = _get_session(session_id)
    return session.get_state_snapshot()


@app.get("/sessions/{session_id}/export")
async def export_csv(session_id: str):
    """Download the labeled CSV."""
    session = _get_session(session_id)

    try:
        csv_content = session.export_labeled_csv()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=labeled_{session_id}.csv"},
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session."""
    with _session_lock:
        session = _sessions.pop(session_id, None)

    if session:
        try:
            session.close()
        except Exception:
            pass
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/scenarios")
async def list_scenarios():
    """List available scenario YAML files."""
    scenarios_dir = Path("scenarios")
    if not scenarios_dir.exists():
        return {"scenarios": []}
    files = [str(f) for f in scenarios_dir.glob("*.yaml")]
    return {"scenarios": files}


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
