from __future__ import annotations

from pathlib import Path
import pytest
import torch
from runtime.checkpoint_manager import CheckpointManager
from runtime.torch_native_runner import PyTorchNativeRunner


def test_torch_native_runner_synthetic_run(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    metrics, checkpoint = runner.run(
        steps=5,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 1},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence")},
    )

    assert metrics["execution_performed"] is True
    assert metrics["micro_train_steps_completed"] == 5
    assert metrics["loss_finite"] is True
    assert metrics["parameter_delta_abs_sum_positive"] is True
    assert checkpoint["checkpoint_written"] is True


def test_torch_native_runner_gradient_accumulation_and_resume(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # 1. Run 4 steps with grad accum = 2
    metrics1, ckpt1 = runner.run(
        steps=4,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 2},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence1")},
    )

    assert metrics1["execution_performed"] is True
    assert metrics1["optimizer_step_count"] == 2
    assert ckpt1["checkpoint_written"] is True
    saved_ckpt_path = ckpt1["checkpoint_path"]

    # 2. Resume using saved checkpoint
    metrics2, ckpt2 = runner.run(
        steps=3,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=False,
        load_checkpoint=saved_ckpt_path,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 2},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence2")},
    )

    assert metrics2["execution_performed"] is True
    resume_info = metrics2["sustained_control"]["checkpoint_resume"]
    assert resume_info["requested"] is True
    assert resume_info["success"] is True
    assert resume_info["loaded_model"] is True
    # Verify global step resumption: started from step 4, executed 3 steps -> global step 7
    assert resume_info["resumed_step"] == 4
    assert resume_info["loaded_rng"] is True
    # At step 3 with grad accum = 2: steps 1, 2 (boundary opt step), step 3 (final boundary flush opt step) -> total 2 optimizer steps
    assert metrics2["optimizer_step_count"] == 2


def test_torch_native_runner_incomplete_accumulation_scaling(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # 10 microbatches with gradient_accumulation_steps=4
    # Full groups: step 4 (4 batches), step 8 (4 batches)
    # Incomplete group: steps 9 and 10 (2 batches, correctly scaled by 1/2) -> 3 optimizer steps
    metrics, _ = runner.run(
        steps=10,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=False,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 4},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence")},
    )

    assert metrics["execution_performed"] is True
    assert metrics["micro_train_steps_completed"] == 10
    assert metrics["optimizer_step_count"] == 3


def test_torch_native_runner_adaptive_memory_suggestions(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # Run 20 steps with adaptive memory suggestions enabled
    metrics, _ = runner.run(
        steps=20,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=False,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 2},
        dataset_settings={
            "real_dataset": False,
            "evidence_dir": str(tmp_path / "evidence"),
            "adaptive_memory_apply_suggestions": True,
        },
    )

    assert metrics["execution_performed"] is True
    assert metrics["adaptive_memory"]["adaptive_memory_enabled"] is True
    assert metrics["adaptive_memory"]["decision_effects_recorded"] >= 2


def test_torch_native_runner_strict_checkpoint_rejection_on_mismatch(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # Save a checkpoint with a completely different model architecture (mismatched keys)
    mismatched_model = torch.nn.Linear(32, 8)
    ckpt_res = ckpt_mgr.save_live_checkpoint(
        model=mismatched_model,
        metadata={"step": 5},
    )
    mismatched_ckpt_path = ckpt_res["checkpoint_path"]

    # Attempting to resume without allowing partial mismatch must fail loudly
    with pytest.raises(RuntimeError) as exc_info:
        runner.run(
            steps=2,
            batch_size=2,
            load_checkpoint=mismatched_ckpt_path,
            dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence")},
        )

    assert "checkpoint_resume_failed" in str(exc_info.value)
    assert "checkpoint_model_keys_mismatch" in str(exc_info.value)


def test_torch_native_runner_partial_checkpoint_allowed_when_opted_in(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # Save a checkpoint with partial matching keys (only first layer "0.weight" and "0.bias")
    partial_state = {
        "0.weight": torch.randn(128, 16),
        "0.bias": torch.randn(128),
    }
    ckpt_res = ckpt_mgr.save_live_checkpoint(
        model_state_dict=partial_state,
        metadata={"step": 10},
    )
    partial_ckpt_path = ckpt_res["checkpoint_path"]

    # Default strict mode rejects partial keys
    with pytest.raises(RuntimeError) as exc_info:
        runner.run(
            steps=2,
            batch_size=2,
            load_checkpoint=partial_ckpt_path,
            dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence1")},
        )
    assert "checkpoint_model_keys_mismatch" in str(exc_info.value)

    # Explicit opt-in with allow_partial_checkpoint=True succeeds
    metrics, _ = runner.run(
        steps=2,
        batch_size=2,
        load_checkpoint=partial_ckpt_path,
        dataset_settings={
            "real_dataset": False,
            "evidence_dir": str(tmp_path / "evidence2"),
            "allow_partial_checkpoint": True,
        },
    )

    resume_info = metrics["sustained_control"]["checkpoint_resume"]
    assert resume_info["requested"] is True
    assert resume_info["success"] is True
    assert resume_info["loaded_model"] is True
    assert resume_info.get("partial_transfer") is True

