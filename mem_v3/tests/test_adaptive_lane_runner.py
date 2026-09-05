from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from runtime.adaptive_lane_runner import AdaptiveLaneRunner, LaneConfig, get_lanes_for_model
from runtime.checkpoint_manager import CheckpointManager
from runtime.lm_model import build_tiny_causal_lm, TinyCausalTransformer


def test_build_tiny_causal_lm_presets():
    # Test 75M preset
    m75 = build_tiny_causal_lm(50257, 256, preset="medium_75m")
    assert isinstance(m75, TinyCausalTransformer)
    assert m75.token_embedding.embedding_dim == 640

    # Test 250M preset
    m250 = build_tiny_causal_lm(50257, 256, preset="xlarge_250m")
    assert isinstance(m250, TinyCausalTransformer)
    assert m250.token_embedding.embedding_dim == 1024
    params = sum(p.numel() for p in m250.parameters())
    assert params > 250_000_000, f"Expected >250M parameters, got {params}"


def test_get_lanes_for_model_configurations():
    lanes_75m = get_lanes_for_model("medium_75m")
    assert "aggressive_seq256_zero0_gacc4" in lanes_75m
    assert lanes_75m["aggressive_seq256_zero0_gacc4"].batch_size == 16
    assert lanes_75m["aggressive_seq256_zero0_gacc4"].min_tokens_floor == 18000.0

    lanes_130m = get_lanes_for_model("large_130m")
    assert "ultra_peak_seq256" in lanes_130m
    assert lanes_130m["ultra_peak_seq256"].batch_size == 28
    assert "aggressive_seq256_zero0_gacc4" in lanes_130m
    assert lanes_130m["aggressive_seq256_zero0_gacc4"].batch_size == 20
    assert lanes_130m["aggressive_seq256_zero0_gacc4"].min_tokens_floor == 24000.0

    lanes_250m = get_lanes_for_model("xlarge_250m")
    assert "aggressive_seq256_zero0_gacc4" in lanes_250m
    assert lanes_250m["aggressive_seq256_zero0_gacc4"].batch_size == 8
    assert lanes_250m["aggressive_seq256_zero0_gacc4"].min_tokens_floor == 6000.0


def test_adaptive_lane_runner_transitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        ev_dir = Path(tmpdir) / "evidence"
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_mgr = CheckpointManager(root=ckpt_dir)

        runner = AdaptiveLaneRunner(
            checkpoint_manager=ckpt_mgr,
            evidence_dir=ev_dir,
            initial_lane="aggressive_seq256_zero0_gacc4",
            model_preset="medium_75m",
        )
        assert runner.current_lane.name == "aggressive_seq256_zero0_gacc4"

        # Demotion evaluation: window throughput below floor (18k) for 3 bad windows
        new_lane, reason, bad_cnt, _ = runner.evaluate_lane_transition(
            step=120,
            window_tokens_sec=14000.0,
            window_loss=5.0,
            optimizer_ratio=0.2,
            data_wait_ratio=0.1,
            bad_windows=2,
            stable_windows=0,
        )
        assert new_lane is not None
        assert new_lane.name == "fast_seq256_zero0_gacc4"
        assert "Demoting" in reason

        # Promotion evaluation: in safe lane with sustained stable throughput and low overhead
        runner.current_lane = runner.lanes["safe_seq256"]
        runner.last_switch_step = 0
        new_lane, reason, _, _ = runner.evaluate_lane_transition(
            step=200,
            window_tokens_sec=19000.0,
            window_loss=4.5,
            optimizer_ratio=0.25,
            data_wait_ratio=0.1,
            bad_windows=0,
            stable_windows=3,
        )
        assert new_lane is not None
        assert new_lane.name == "fast_seq256_zero0_gacc4"
        assert "Promoting" in reason