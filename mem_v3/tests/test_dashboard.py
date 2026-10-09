from __future__ import annotations

import json
from pathlib import Path
import application.dashboard as dashboard
from application.dashboard import get_latest_evidence_dir, get_live_state, get_milestones


def test_dashboard_get_live_state():
    state = get_live_state()
    assert "timestamp" in state
    assert "evidence_dir" in state
    assert "progress" in state
    assert "api_telemetry" in state


def test_dashboard_get_milestones():
    milestones = get_milestones(limit=10)
    assert isinstance(milestones, list)


def test_dashboard_clean_environment_no_evidence(monkeypatch, tmp_path):
    # Simulate a clean environment where evidence and checkpoints directories do not exist
    clean_root = tmp_path / "clean_repo"
    clean_root.mkdir()
    monkeypatch.setattr(dashboard, "_root", clean_root)
    dashboard._milestones_cache = {"time": 0.0, "data": []}

    latest_dir = get_latest_evidence_dir()
    assert latest_dir is None

    state = get_live_state()
    assert state["evidence_dir"] == "None"
    assert state["progress"] == {}
    assert state["controller"] == {}
    assert state["api_telemetry"] == {}
    assert state["latest_checkpoint"] == ""
    assert state["events"] == []

    milestones = get_milestones(limit=10)
    assert milestones == []


def test_dashboard_populated_environment(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    ev_root = repo_root / "evidence" / "run_abc"
    ev_root.mkdir(parents=True)
    ckpt_dir = repo_root / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    prog_data = {"step": 42, "loss": 1.23}
    (ev_root / "runtime_progress_latest.json").write_text(json.dumps(prog_data), encoding="utf-8")
    (ev_root / "v89_sustained_control_events.jsonl").write_text(
        json.dumps({"event": "lane_switch"}) + "\n",
        encoding="utf-8",
    )
    (ev_root / "runtime_milestones.jsonl").write_text(
        json.dumps({"step": 10, "loss": 2.0}) + "\n" + json.dumps({"step": 20, "loss": 1.5}) + "\n",
        encoding="utf-8",
    )
    (ckpt_dir / "model_latest.txt").write_text("checkpoints/model_step_42.pt", encoding="utf-8")

    monkeypatch.setattr(dashboard, "_root", repo_root)
    dashboard._milestones_cache = {"time": 0.0, "data": []}

    latest_dir = get_latest_evidence_dir()
    assert latest_dir is not None
    assert latest_dir.name == "run_abc"

    state = get_live_state()
    assert "run_abc" in state["evidence_dir"]
    assert state["progress"] == prog_data
    assert state["latest_checkpoint"] == "checkpoints/model_step_42.pt"
    assert len(state["events"]) == 1
    assert state["events"][0]["event"] == "lane_switch"

    milestones = get_milestones(limit=10)
    assert len(milestones) == 2
    assert milestones[0]["step"] == 10
    assert milestones[1]["step"] == 20
