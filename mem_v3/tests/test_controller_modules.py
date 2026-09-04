from __future__ import annotations

import pytest
from pathlib import Path

from runtime.controller.lane_manager import LaneManager, LaneDefinition
from runtime.controller.degradation_detector import DegradationDetector, DegradationMetrics
from runtime.controller.supervisor import LaneSupervisor


def test_lane_manager_standard_lanes():
    lm = LaneManager()
    assert lm.is_valid_lane("fast_seq256_zero0_gacc4")
    assert lm.is_valid_lane("aggressive_seq256_zero0_gacc4")
    assert lm.is_valid_lane("safe_seq256")
    assert not lm.is_valid_lane("unknown_lane")

    lane = lm.get_lane("fast_seq256_zero0_gacc4")
    cli_args = lane.to_cli_args()
    assert "--deepspeed-batch-size" in cli_args
    assert "--sequence-length" in cli_args


def test_lane_manager_recovery_and_promotion():
    lm = LaneManager()
    recovery = lm.next_recovery_lane("aggressive_seq256_zero0_gacc4")
    assert recovery.name == "safe_seq256"

    assert lm.can_promote("safe_seq256", tokens_per_second=18000.0, optimizer_ratio=0.35) is True
    assert lm.can_promote("safe_seq256", tokens_per_second=15000.0, optimizer_ratio=0.35) is False
    assert lm.can_promote("safe_seq256", tokens_per_second=18000.0, optimizer_ratio=0.48) is False


def test_degradation_detector_health_tracking():
    detector = DegradationDetector(min_degrade_step=100, drop_ratio=0.60, required_bad_windows=2)

    # Initial healthy state
    m1 = DegradationMetrics(step=50, tokens_per_second=30000.0, steps_per_second=10.0, optimizer_ratio=0.20, gpu_utilization=85.0)
    res1 = detector.update(m1)
    assert res1["health"] == "OPTIMAL"
    assert res1["degraded"] is False
    assert res1["should_switch"] is False

    # First degraded sample
    m2 = DegradationMetrics(step=120, tokens_per_second=15000.0, steps_per_second=5.0, optimizer_ratio=0.45, gpu_utilization=60.0)
    res2 = detector.update(m2)
    assert res2["degraded"] is True
    assert res2["bad_windows"] == 1
    assert res2["should_switch"] is False

    # Second degraded sample triggers switch
    m3 = DegradationMetrics(step=140, tokens_per_second=14000.0, steps_per_second=4.5, optimizer_ratio=0.48, gpu_utilization=55.0)
    res3 = detector.update(m3)
    assert res3["degraded"] is True
    assert res3["bad_windows"] == 2
    assert res3["should_switch"] is True
    assert res3["recommended_action"] == "lane_switch"


def test_supervisor_command_building():
    sup = LaneSupervisor()
    lane_args = ["--deepspeed-batch-size", "8", "--deepspeed-zero-stage", "0"]
    cmd = sup.build_command_args(
        lane_args=lane_args,
        max_steps=50000,
        checkpoint_path="checkpoints/v89_live_01/mem_model_optimizer.pt",
        llm_enabled=True,
        api_executive_moderate=True,
    )
    assert "--deepspeed-wsl-accelerated" in cmd
    assert "--llm" in cmd
    assert "--api-executive-mode" in cmd
    assert "--deepspeed-load-checkpoint" in cmd
    assert "checkpoints/v89_live_01/mem_model_optimizer.pt" in cmd
