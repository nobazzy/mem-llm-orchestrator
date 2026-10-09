from pathlib import Path

import torch

from runtime.checkpoint_manager import CheckpointManager


def test_corrupt_new_checkpoint_does_not_replace_latest(monkeypatch, tmp_path):
    manager = CheckpointManager(root=tmp_path / "checkpoints", keep_last=3)

    good = manager.save_live_checkpoint(
        label="unit",
        model_state_dict={"w": torch.tensor([1.0])},
        optimizer_state_dict={"step": 1},
        metadata={"global_step": 1},
    )

    assert good["checkpoint_written"] is True
    good_path = Path(good["checkpoint_path"])
    assert good_path.exists()

    latest_path = tmp_path / "checkpoints" / "unit_latest.txt"
    previous_latest = latest_path.read_text(encoding="utf-8").strip()

    def fake_torch_save(payload, path):
        Path(path).write_bytes(b"corrupted checkpoint bytes")

    monkeypatch.setattr(torch, "save", fake_torch_save)

    bad = manager.save_live_checkpoint(
        label="unit",
        model_state_dict={"w": torch.tensor([2.0])},
        optimizer_state_dict={"step": 2},
        metadata={"global_step": 2},
    )

    assert bad["checkpoint_written"] is False
    assert bad["checkpoint_validation_failed"] is True
    assert latest_path.read_text(encoding="utf-8").strip() == previous_latest

    loaded = manager.load_torch_checkpoint(good_path, map_location="cpu")
    assert "model_state_dict" in loaded
    assert torch.equal(loaded["model_state_dict"]["w"], torch.tensor([1.0]))


def test_post_publish_validation_failure_restores_backup(monkeypatch, tmp_path):
    manager = CheckpointManager(root=tmp_path / "checkpoints")

    # 1. Save initial live checkpoint
    good1 = manager.save_live_checkpoint(
        label="v89",
        model_state_dict={"w": torch.tensor([10.0])},
        optimizer_state_dict={"step": 1000},
        metadata={"global_step": 1000},
    )
    assert good1["checkpoint_written"] is True
    slot_file = Path(good1["checkpoint_path"])
    assert slot_file.exists()

    # 2. Simulate post-publish check failure (e.g. hash mismatch or validation error after publish)
    orig_validate = manager._publish_dir_atomically
    
    # We patch _validate_checkpoint_file so that when called on the final published path during step 6, it raises
    real_val = getattr(manager, "_validate_checkpoint_file", None)

    import runtime.checkpoint_manager as ckm
    orig_val_fn = ckm._validate_checkpoint_file
    call_count = [0]

    def patched_val(path, map_location="cpu", allow_unsafe=True):
        call_count[0] += 1
        # Call 1: temp dir validation (succeeds)
        # Call 2: post-publish final dir validation (fails!)
        if call_count[0] == 2:
            raise RuntimeError("simulated_post_publish_corruption")
        return orig_val_fn(path, map_location=map_location, allow_unsafe=allow_unsafe)

    monkeypatch.setattr(ckm, "_validate_checkpoint_file", patched_val)

    bad = manager.save_live_checkpoint(
        label="v89",
        model_state_dict={"w": torch.tensor([20.0])},
        optimizer_state_dict={"step": 1000}, # same slot 0
        metadata={"global_step": 1000},
    )

    assert bad["checkpoint_written"] is False
    assert bad["checkpoint_validation_failed"] is True

    # 3. Verify slot was restored to good1!
    loaded = manager.load_torch_checkpoint(slot_file, map_location="cpu")
    assert torch.equal(loaded["model_state_dict"]["w"], torch.tensor([10.0]))
