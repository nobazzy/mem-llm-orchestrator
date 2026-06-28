from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class StateManager:
    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.evidence_root = self.base_dir / "evidence"
        self.checkpoints_root = self.base_dir / "checkpoints"
        for d in [self.reports_dir, self.evidence_root, self.checkpoints_root]:
            d.mkdir(parents=True, exist_ok=True)

    def new_evidence_dir(self, version_tag: str = "v89") -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.evidence_root / f"{version_tag}_wsl_deepspeed_{stamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path
