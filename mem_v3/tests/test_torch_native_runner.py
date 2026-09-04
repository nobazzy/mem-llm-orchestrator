from __future__ import annotations

from pathlib import Path
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
