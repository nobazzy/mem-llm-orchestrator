#!/usr/bin/env bash
set -euo pipefail

echo "== MEM v3 durable checkpoint patch =="

if [ ! -f runtime/checkpoint_manager.py ]; then
  echo "ERRO: rode este script na raiz do mem_v3"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
cp runtime/checkpoint_manager.py "runtime/checkpoint_manager.py.bak_durable_${TS}"

cat > runtime/checkpoint_manager.py <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def __init__(self, root: str | Path = "checkpoints", keep_last: int = 3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(2, int(keep_last))

    def _fsync_file(self, path: Path) -> None:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_dir(self, path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # Some filesystems/platforms do not support directory fsync.
            # Best-effort fallback keeps compatibility with WSL/Windows mounts.
            pass

    def _atomic_replace_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self._fsync_dir(path.parent)

    def _atomic_replace_text(self, path: Path, text: str) -> None:
        self._atomic_replace_bytes(path, text.encode("utf-8"))

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_sha256(self, checkpoint_path: Path) -> str:
        digest = self._sha256_file(checkpoint_path)
        self._atomic_replace_text(checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256"), digest + "\n")
        return digest

    def _verify_sha256_if_present(self, checkpoint_path: Path) -> None:
        sha_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        if not sha_path.exists():
            return
        expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
        actual = self._sha256_file(checkpoint_path)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint_sha256_mismatch:{checkpoint_path}:expected={expected}:actual={actual}"
            )

    def _validate_checkpoint(self, path: Path) -> Dict[str, Any]:
        self._verify_sha256_if_present(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint_validation_failed:{path}:payload_not_dict")
        if "model_state_dict" not in payload:
            raise RuntimeError(f"checkpoint_validation_failed:{path}:missing_model_state_dict")
        return payload

    def _safe_torch_save_validated(self, payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")

        try:
            torch.save(payload, tmp)

            # Durability: force checkpoint bytes before rename.
            self._fsync_file(tmp)

            os.replace(tmp, path)

            # Durability: force directory entry for rename.
            self._fsync_dir(path.parent)

            # Validate final checkpoint before publishing metadata/latest.
            validated_payload = self._validate_checkpoint(path)

            # Optional checksum written only after a valid checkpoint exists.
            digest = self._write_sha256(path)

            return {
                "ok": True,
                "path": str(path),
                "sha256": digest,
                "payload": validated_payload,
            }

        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return {
                "ok": False,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _latest_file(self, label: str) -> Path:
        return self.root / f"{label}_latest.txt"

    def _read_latest_path(self, label: str) -> Optional[Path]:
        latest = self._latest_file(label)
        if not latest.exists():
            return None
        raw = latest.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    def _choose_next_live_dir(self, label: str) -> Path:
        previous = self._read_latest_path(label)
        previous_name = previous.parent.name if previous else ""

        slots = [self.root / f"{label}_live_{i:02d}" for i in range(self.keep_last)]

        used = -1
        for i, d in enumerate(slots):
            if d.name == previous_name:
                used = i
                break

        # Rotate to next slot. Never overwrite latest slot directly.
        next_idx = (used + 1) % len(slots)
        return slots[next_idx]

    def _publish_metadata_and_latest(
        self,
        *,
        label: str,
        ckpt_dir: Path,
        ckpt_path: Path,
        metadata: Dict[str, Any],
        sha256: str,
    ) -> None:
        meta = dict(metadata or {})
        meta.update(
            {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_sha256": sha256,
                "checkpoint_validated": True,
                "checkpoint_published_at": time.time(),
                "checkpoint_rotation_keep_last": self.keep_last,
            }
        )

        self._atomic_replace_text(
            ckpt_dir / "metadata.json",
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
        )

        # latest is published only after checkpoint and metadata are valid/durable.
        rel = os.path.relpath(ckpt_path, Path.cwd())
        self._atomic_replace_text(self._latest_file(label), rel + "\n")

    def save_live_checkpoint(
        self,
        *,
        label: str,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ckpt_dir = self._choose_next_live_dir(label)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "mem_model_optimizer.pt"

        payload: Dict[str, Any] = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "metadata": metadata or {},
        }

        result = self._safe_torch_save_validated(payload, ckpt_path)

        if not result.get("ok"):
            previous = self._read_latest_path(label)
            return {
                "checkpoint_written": False,
                "checkpoint_mode": "validation_failed",
                "checkpoint_validation_failed": True,
                "error": result.get("error", ""),
                "attempted_path": str(ckpt_path),
                "previous_latest": str(previous) if previous else "",
                "event": "checkpoint_validation_failed",
            }

        self._publish_metadata_and_latest(
            label=label,
            ckpt_dir=ckpt_dir,
            ckpt_path=ckpt_path,
            metadata=metadata or {},
            sha256=str(result["sha256"]),
        )

        return {
            "checkpoint_written": True,
            "checkpoint_mode": "live_rotating_validated",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_sha256": str(result["sha256"]),
            "checkpoint_validated": True,
            "latest_file": str(self._latest_file(label)),
        }

    def save_post_train(
        self,
        *,
        label: str,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ckpt_dir = self.root / f"{label}_post_train_{int(time.time())}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "mem_model_optimizer.pt"

        payload: Dict[str, Any] = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "metadata": metadata or {},
        }

        result = self._safe_torch_save_validated(payload, ckpt_path)

        if not result.get("ok"):
            return {
                "checkpoint_written": False,
                "checkpoint_mode": "post_train_validation_failed",
                "checkpoint_validation_failed": True,
                "error": result.get("error", ""),
                "attempted_path": str(ckpt_path),
                "event": "checkpoint_validation_failed",
            }

        self._publish_metadata_and_latest(
            label=label,
            ckpt_dir=ckpt_dir,
            ckpt_path=ckpt_path,
            metadata=metadata or {},
            sha256=str(result["sha256"]),
        )

        return {
            "checkpoint_written": True,
            "checkpoint_mode": "post_train_validated",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_sha256": str(result["sha256"]),
            "checkpoint_validated": True,
            "latest_file": str(self._latest_file(label)),
        }

    def load_torch_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        self._verify_sha256_if_present(p)

        payload = torch.load(p, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint_load_failed:{p}:payload_not_dict")
        if "model_state_dict" not in payload:
            raise RuntimeError(f"checkpoint_load_failed:{p}:missing_model_state_dict")
        return payload
PY

mkdir -p tests

cat > tests/test_checkpoint_manager_durable.py <<'PY'
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
PY

python -m py_compile runtime/checkpoint_manager.py
source .venv312/bin/activate 2>/dev/null || true
python scripts/v89_static_validation.py
python -m pytest -q tests/test_checkpoint_manager_durable.py

echo "PATCH_DURABLE_CHECKPOINT_OK"
