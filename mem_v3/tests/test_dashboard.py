from __future__ import annotations

from application.dashboard import get_live_state, get_milestones


def test_dashboard_get_live_state():
    state = get_live_state()
    assert "timestamp" in state
    assert "evidence_dir" in state
    assert "progress" in state
    assert "api_telemetry" in state


def test_dashboard_get_milestones():
    milestones = get_milestones(limit=10)
    assert isinstance(milestones, list)
