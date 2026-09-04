from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch


CHECKPOINT_FILENAME = "mem_model_optimizer.pt"
SHA_FILENAME = "mem_model_optimizer.pt.sha256"
METADATA_FILENAME = "metadata.json"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _fsync_file(path: Path) -> None:
    try:
        with path.open("r+b") as f:
            os.fsync(f.fileno())
    except Exception:
        try:
            with path.open("rb") as f:
                os.fsync(f.fileno())
        except Exception:
            pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # DrvFS/WSL/Windows mounts may not support directory fsync reliably.
        pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write_text(path, text + "\n")


def _extract_state_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def _step_from_metadata(metadata: Dict[str, Any]) -> int:
    for key in (
        "global_step",
        "runtime_step",
        "micro_train_steps_completed",
        "step",
        "steps",
    ):
        value = metadata.get(key)
        try:
            if value is not None:
                return int(value)
        except Exception:
            pass
    return 0


def _slot_from_step(step: int) -> int:
    if step <= 0:
        return 0
    # step 1000 -> live_00, 2000 -> live_01, 3000 -> live_02, 4000 -> live_00
    return max(0, (step // 1000) - 1) % 3


def _validate_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint_payload_not_dict")

    if "model_state_dict" not in payload:
        raise RuntimeError("checkpoint_missing_model_state_dict")

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError("checkpoint_invalid_model_state_dict")

    if "metadata" not in payload:
        raise RuntimeError("checkpoint_missing_metadata")


def _validate_checkpoint_file(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    _validate_payload(payload)
    return payload


class CheckpointManager:
    """
    Checkpoint manager v89 final atomic contract.

    Business rule:
    - validation_failed never mutates a previously published live slot.
    - checkpoint is first written into a temporary directory.
    - temp checkpoint is loaded and validated before publish.
    - only a fully valid temp directory replaces live_00/live_01/live_02.
    - latest.txt is updated only after publish.
    """

    def __init__(self, root: str | Path = "checkpoints", *args: Any, **kwargs: Any) -> None:
        root = kwargs.pop("checkpoint_root", root)
        root = kwargs.pop("checkpoint_dir", root)
        root = kwargs.pop("root_dir", root)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _normalise_args(
        self,
        model: Any = None,
        optimizer: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label: str = "v89",
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        if "label" in kwargs and kwargs["label"] is not None:
            label = str(kwargs["label"])

        if "metadata" in kwargs and kwargs["metadata"] is not None:
            metadata = kwargs["metadata"]

        metadata = dict(metadata or {})

        model_state = kwargs.get("model_state_dict")
        optimizer_state = kwargs.get("optimizer_state_dict")

        if model_state is None:
            model_state = _extract_state_dict(model)
        if optimizer_state is None:
            optimizer_state = _extract_state_dict(optimizer)

        if not isinstance(model_state, dict) or not model_state:
            raise RuntimeError("checkpoint_model_state_dict_empty")

        if optimizer_state is None:
            optimizer_state = {}

        metadata.setdefault("checkpoint_label", label)
        metadata.setdefault("checkpoint_created_at", time.time())

        payload = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metadata": metadata,
        }

        return label, payload, metadata

    def _publish_dir_atomically(self, tmp_dir: Path, final_dir: Path) -> None:
        backup_dir = final_dir.with_name(final_dir.name + f".bak.{os.getpid()}.{uuid.uuid4().hex}")

        final_parent = final_dir.parent
        final_parent.mkdir(parents=True, exist_ok=True)

        try:
            if final_dir.exists():
                os.replace(final_dir, backup_dir)

            os.replace(tmp_dir, final_dir)
            _fsync_dir(final_parent)

            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

        except Exception:
            # Rollback: if publish failed, restore previous published slot.
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            if backup_dir.exists():
                os.replace(backup_dir, final_dir)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            _fsync_dir(final_parent)
            raise

    def _save_checkpoint_dir(
        self,
        *,
        label: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        final_dir: Path,
        latest_file: Optional[Path],
        mode: str,
    ) -> Dict[str, Any]:
        tmp_dir = final_dir.with_name(final_dir.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")

        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=False)

            tmp_pt = tmp_dir / CHECKPOINT_FILENAME
            tmp_sha = tmp_dir / SHA_FILENAME
            tmp_meta = tmp_dir / METADATA_FILENAME

            # 1. Write checkpoint into temp directory only.
            torch.save(payload, tmp_pt)
            _fsync_file(tmp_pt)

            # 2. Validate temp checkpoint before touching published slot.
            _validate_checkpoint_file(tmp_pt, map_location="cpu")

            # 3. Compute sha from validated temp file.
            sha = _sha256_file(tmp_pt)

            final_metadata = dict(metadata)
            final_metadata.update(
                {
                    "checkpoint_label": label,
                    "checkpoint_path": str(final_dir / CHECKPOINT_FILENAME),
                    "checkpoint_dir": str(final_dir),
                    "checkpoint_mode": mode,
                    "checkpoint_validated": True,
                    "checkpoint_sha256": sha,
                    "checkpoint_written": True,
                    "checkpoint_published_at": time.time(),
                }
            )

            # 4. Complete temp directory.
            tmp_sha.write_text(sha + "\n", encoding="utf-8")
            _fsync_file(tmp_sha)
            _atomic_write_json(tmp_meta, final_metadata)
            _fsync_dir(tmp_dir)

            # 5. Only now publish slot.
            self._publish_dir_atomically(tmp_dir, final_dir)

            final_pt = final_dir / CHECKPOINT_FILENAME

            # 6. Revalidate published file and sha.
            actual_sha = _sha256_file(final_pt)
            if actual_sha != sha:
                raise RuntimeError(
                    f"checkpoint_post_publish_sha256_mismatch: expected={sha} actual={actual_sha}"
                )
            _validate_checkpoint_file(final_pt, map_location="cpu")

            # 7. Update latest only after fully valid publish.
            if latest_file is not None:
                _atomic_write_text(latest_file, str(final_pt) + "\n")

            return {
                "checkpoint_written": True,
                "checkpoint_path": str(final_pt),
                "checkpoint_dir": str(final_dir),
                "checkpoint_mode": mode,
                "checkpoint_validated": True,
                "checkpoint_sha256": sha,
                "latest_file": str(latest_file) if latest_file else "",
            }

        except Exception as exc:
            # Critical business rule:
            # validation_failed must not publish tmp and must not touch final slot.
            error_payload = {
                "ts": time.time(),
                "label": label,
                "final_dir": str(final_dir),
                "tmp_dir": str(tmp_dir),
                "tmp_exists_before_cleanup": tmp_dir.exists(),
                "tmp_checkpoint_exists_before_cleanup": (tmp_dir / CHECKPOINT_FILENAME).exists(),
                "tmp_checkpoint_size_bytes": ((tmp_dir / CHECKPOINT_FILENAME).stat().st_size if (tmp_dir / CHECKPOINT_FILENAME).exists() else 0),
                "checkpoint_mode": "validation_failed",
                "checkpoint_error_type": type(exc).__name__,
                "checkpoint_error": repr(exc),
            }

            try:
                debug_dir = Path("evidence_packets")
                debug_dir.mkdir(parents=True, exist_ok=True)
                with (debug_dir / "checkpoint_failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_json_safe(error_payload), ensure_ascii=False, sort_keys=True) + "\n")
            except Exception:
                pass

            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return {
                "checkpoint_written": False,
                "checkpoint_path": "",
                "checkpoint_dir": str(final_dir),
                "checkpoint_mode": "validation_failed",
                "checkpoint_validated": False,
                "checkpoint_validation_failed": True,
                "checkpoint_error": repr(exc),
            }

    def save_live_checkpoint(
        self,
        model: Any = None,
        optimizer: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label: str = "v89",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        label, payload, metadata = self._normalise_args(
            model=model,
            optimizer=optimizer,
            metadata=metadata,
            label=label,
            **kwargs,
        )

        step = _step_from_metadata(metadata)
        slot = _slot_from_step(step)

        final_dir = self.root / f"{label}_live_{slot:02d}"
        latest_file = self.root / f"{label}_latest.txt"

        return self._save_checkpoint_dir(
            label=label,
            payload=payload,
            metadata=metadata,
            final_dir=final_dir,
            latest_file=latest_file,
            mode="live_rotating_validated_model_optimizer_state",
        )

    def save_post_train(
        self,
        model: Any = None,
        optimizer: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label: str = "v89",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        label, payload, metadata = self._normalise_args(
            model=model,
            optimizer=optimizer,
            metadata=metadata,
            label=label,
            **kwargs,
        )

        final_dir = self.root / f"{label}_post_train"
        latest_file = self.root / f"{label}_latest.txt"

        return self._save_checkpoint_dir(
            label=label,
            payload=payload,
            metadata=metadata,
            final_dir=final_dir,
            latest_file=latest_file,
            mode="post_train_validated_model_optimizer_state",
        )

    def latest_checkpoint_path(self, label: str = "v89") -> Optional[str]:
        # Try requested label first, then generic v89 fallback.
        # This fixes runner labels like v89_slimpajama_75m_... while
        # checkpoints are published as v89_live_00/01/02.
        labels = []
        for candidate_label in (label, "v89"):
            if candidate_label and candidate_label not in labels:
                labels.append(candidate_label)

        for candidate_label in labels:
            latest_file = self.root / f"{candidate_label}_latest.txt"
            if latest_file.exists():
                path = Path(_read_text(latest_file))
                if not path.is_absolute():
                    path = Path.cwd() / path
                if path.exists() and ".tmp." not in str(path):
                    try:
                        self.load_torch_checkpoint(path, map_location="cpu")
                        return str(path)
                    except Exception:
                        pass

            candidates = []
            for slot in range(3):
                path = self.root / f"{candidate_label}_live_{slot:02d}" / CHECKPOINT_FILENAME
                if path.exists() and ".tmp." not in str(path):
                    candidates.append(path)

            candidates = sorted(
                candidates,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for path in candidates:
                try:
                    self.load_torch_checkpoint(path, map_location="cpu")
                    return str(path)
                except Exception:
                    continue

        return None

    def load_torch_checkpoint(self, path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        sha_path = path.with_name(SHA_FILENAME)
        if sha_path.exists():
            expected = _read_text(sha_path)
            actual = _sha256_file(path)
            if expected != actual:
                raise RuntimeError(
                    f"checkpoint_sha256_mismatch: {path}: expected={expected} actual={actual}"
                )

        return _validate_checkpoint_file(path, map_location=map_location)


# Backward-compatible aliases.
DurableCheckpointManager = CheckpointManager
RuntimeCheckpointManager = CheckpointManager


def latest_checkpoint_path(label: str = "v89", root: str | Path = "checkpoints") -> Optional[str]:
    return CheckpointManager(root=root).latest_checkpoint_path(label=label)


def load_torch_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return CheckpointManager().load_torch_checkpoint(path, map_location=map_location)


def save_live_checkpoint(
    model: Any = None,
    optimizer: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    label: str = "v89",
    **kwargs: Any,
) -> Dict[str, Any]:
    return CheckpointManager().save_live_checkpoint(
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        label=label,
        **kwargs,
    )


def save_post_train(
    model: Any = None,
    optimizer: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    label: str = "v89",
    **kwargs: Any,
) -> Dict[str, Any]:
    return CheckpointManager().save_post_train(
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        label=label,
        **kwargs,
    )
