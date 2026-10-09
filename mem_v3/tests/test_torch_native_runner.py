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


def test_torch_native_runner_faithful_resumption_cumulative_tokens_and_steps(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # 1. First session: 5 steps, batch_size=2. Synthetic tokens per step = 2 * 16 = 32 -> 160 tokens
    metrics1, ckpt1 = runner.run(
        steps=5,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 1},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence1")},
    )

    assert metrics1["execution_performed"] is True
    assert metrics1["micro_train_steps_completed"] == 5
    assert metrics1["tokens_processed"] == 160
    assert ckpt1["checkpoint_written"] is True
    saved_path = ckpt1["checkpoint_path"]

    # 2. Resumed session: 5 additional steps starting from step 5.
    metrics2, ckpt2 = runner.run(
        steps=5,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=False,
        load_checkpoint=saved_path,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 1},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence2")},
    )

    resume_info = metrics2["sustained_control"]["checkpoint_resume"]
    assert resume_info["requested"] is True
    assert resume_info["success"] is True
    assert resume_info["resumed_step"] == 5
    assert resume_info["resumed_tokens"] == 160

    # Verify cumulative token continuity: 160 resumed + 160 newly processed = 320 tokens total
    assert metrics2["tokens_processed"] == 320
    assert metrics2["micro_train_steps_completed"] == 5
    # Throughput must be computed on the 160 new session tokens, not the cumulative 320
    expected_approx_tps = 160.0 / metrics2["total_seconds"]
    assert abs(metrics2["tokens_per_second"] - expected_approx_tps) < 1.0


def test_torch_native_runner_interrupted_gradient_accumulation(tmp_path):
    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    runner = PyTorchNativeRunner(checkpoint_manager=ckpt_mgr)

    # 1. Run 3 microbatches with grad_accum=4 (interrupted mid-accumulation)
    # Loop ends at step 3: flush remaining 3 unapplied microbatches scaled by 1/3 -> 1 optimizer step
    metrics1, ckpt1 = runner.run(
        steps=3,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 4},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence1")},
    )

    assert metrics1["execution_performed"] is True
    assert metrics1["optimizer_step_count"] == 1
    assert ckpt1["checkpoint_written"] is True
    saved_path = ckpt1["checkpoint_path"]

    # 2. Resume from step 3 for 5 more steps (steps 4..8) with grad_accum=4:
    # - Step 4 hits global boundary (4 % 4 == 0) -> 1 microbatch scaled by 1 -> opt step 1
    # - Steps 5, 6, 7, 8 accumulate 4 microbatches, hits boundary (8 % 4 == 0) -> scaled by 1/4 -> opt step 2
    # Total optimizer steps in resumed session = 2
    metrics2, _ = runner.run(
        steps=5,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=False,
        load_checkpoint=saved_path,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 4},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence2")},
    )

    assert metrics2["execution_performed"] is True
    resume_info = metrics2["sustained_control"]["checkpoint_resume"]
    assert resume_info["success"] is True
    assert resume_info["resumed_step"] == 3
    assert metrics2["optimizer_step_count"] == 2


def test_torch_native_runner_dataset_state_roundtrip(tmp_path):
    from runtime.real_dataset import RealDatasetBatcher, DatasetRuntimeInfo

    # Create mock batcher with state
    batcher = object.__new__(RealDatasetBatcher)
    batcher.cache_read_pos = 1024
    batcher.buffer = [10, 20, 30]
    batcher._mix_index = 3
    batcher.iterator_restarts = 2
    batcher.dataset_exhaustions = 1
    batcher.empty_rows_seen = 4
    batcher.info = DatasetRuntimeInfo("mock", "cfg", "mock", "cfg", "train", True, False, "", "gpt2", 50257, 128)
    batcher.info.samples_seen = 50
    batcher.info.tokens_emitted = 4096

    ckpt_mgr = CheckpointManager(root=tmp_path / "checkpoints")
    model = torch.nn.Sequential(torch.nn.Linear(16, 128), torch.nn.GELU(), torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Save checkpoint including batcher state
    ckpt_res = ckpt_mgr.save_post_train(
        model=model,
        optimizer=opt,
        batcher=batcher,
        metadata={"step": 10, "tokens_processed": 4096},
    )
    assert ckpt_res["checkpoint_written"] is True
    ckpt_path = ckpt_res["checkpoint_path"]

    # Verify payload contains dataset_state_dict
    payload = ckpt_mgr.load_torch_checkpoint(ckpt_path)
    assert "dataset_state_dict" in payload
    assert payload["dataset_state_dict"]["cache_read_pos"] == 1024
    assert payload["dataset_state_dict"]["samples_seen"] == 50
    assert payload["dataset_state_dict"]["buffer"] == [10, 20, 30]

    # Test load_state_dict restores batcher exactly
    batcher2 = object.__new__(RealDatasetBatcher)
    batcher2.info = DatasetRuntimeInfo("mock", "cfg", "mock", "cfg", "train", True, False, "", "gpt2", 50257, 128)
    batcher2._iterator_factory = None
    batcher2.load_state_dict(payload["dataset_state_dict"])
    assert batcher2.state_dict() == batcher.state_dict()


