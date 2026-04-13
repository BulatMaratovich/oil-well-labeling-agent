from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.session_store as session_store_module
import core.task_manager as core_task_manager
from app.session_store import FileSessionStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    labels_dir = tmp_path / "review_labels"
    tasks_dir = tmp_path / "tasks"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(session_store_module, "SESSIONS_DIR", sessions_dir, raising=False)
    monkeypatch.setattr(main, "SESSION_STORE", FileSessionStore(sessions_dir))
    monkeypatch.setattr(main, "LABELS_DIR", labels_dir)
    monkeypatch.setattr(core_task_manager, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(main.settings, "mistral_api_key", "")

    return TestClient(main.app)


def test_upload_chat_plot_and_annotate_persist_across_store_reload(client, tmp_path, monkeypatch):
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=72, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "well": ["A-1"] * len(timestamps),
            "power": [120.0] * 24 + [0.0] * 24 + [118.0] * 24,
        }
    )
    source = frame.to_csv(index=False).encode("utf-8")

    upload = client.post("/api/upload", files={"file": ("synthetic.csv", BytesIO(source), "text/csv")})
    assert upload.status_code == 200
    upload_body = upload.json()
    session_id = upload_body["session_id"]

    state_path = tmp_path / "sessions" / session_id / "state.json"
    assert state_path.exists()
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload.get("dataframe_path")
    assert state_payload.get("dataframe_json") is None

    monkeypatch.setattr(main, "SESSION_STORE", FileSessionStore(tmp_path / "sessions"))

    chat = client.post(
        f"/api/chat/{session_id}",
        json={
            "message": "ищем planned stop, работаем интервалами",
            "window_size": 12,
            "recommendation_mode": "interval",
        },
    )
    assert chat.status_code == 200
    assert chat.json()["state"]["recommendation_mode"] == "interval"
    assert chat.json()["state"]["window_size"] == 12

    maintenance = client.post(
        f"/api/maintenance/{session_id}",
        files={
            "file": (
                "maintenance.txt",
                BytesIO(b"2026-01-01 planned stop for well A-1 due to scheduled maintenance."),
                "text/plain",
            )
        },
    )
    assert maintenance.status_code == 200
    assert maintenance.json()["maintenance_context"]["document_count"] == 1

    monkeypatch.setattr(main, "SESSION_STORE", FileSessionStore(tmp_path / "sessions"))

    plot = client.get(
        f"/api/plot/{session_id}",
        params={
            "time_column": "timestamp",
            "well_column": "well",
            "well_value": "A-1",
            "series": "power",
            "detect_candidates": "true",
            "use_scope_dates": "true",
        },
    )
    assert plot.status_code == 200
    plot_body = plot.json()
    assert plot_body["maintenance_context"]["used_in_last_search"] is True
    assert plot_body["maintenance_context"]["matched_candidate_count"] >= 1
    assert plot_body["candidate_intervals"]

    first_candidate = plot_body["candidate_intervals"][0]
    assert first_candidate["proposed_label"] == "planned_stop"
    assert len(first_candidate["maintenance_facts"]) == 1

    annotate = client.post(
        f"/api/recommendation/{session_id}",
        json={
            "mode": "interval",
            "x": first_candidate["start"],
            "x_end": first_candidate["end"],
            "y": None,
            "trace_name": first_candidate.get("series_name"),
            "candidate_id": first_candidate["candidate_id"],
            "locked": True,
            "review_action": "accept",
        },
    )
    assert annotate.status_code == 200
    saved_annotation = annotate.json()["saved_annotation"]
    assert saved_annotation["label"] == "planned_stop"

    monkeypatch.setattr(main, "SESSION_STORE", FileSessionStore(tmp_path / "sessions"))

    annotations = client.get(f"/api/annotations/{session_id}", params={"well_value": "A-1"})
    assert annotations.status_code == 200
    annotations_body = annotations.json()
    assert len(annotations_body["annotations"]) == 1
    assert annotations_body["annotations"][0]["label"] == "planned_stop"
