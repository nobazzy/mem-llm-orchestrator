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
