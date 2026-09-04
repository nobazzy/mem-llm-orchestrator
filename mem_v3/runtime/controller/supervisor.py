from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class LaneSupervisor:
    """Manages process lifecycle, graceful termination, and rotating checkpoint resumption."""

    def __init__(self, checkpoints_root: str | Path = "checkpoints") -> None:
        self.checkpoints_root = Path(checkpoints_root)

    def find_latest_checkpoint(self, label: str = "v89") -> Optional[Path]:
        latest_ptr = self.checkpoints_root / f"{label}_latest.txt"
        if latest_ptr.exists():
            path_str = latest_ptr.read_text(encoding="utf-8").strip()
            p = Path(path_str)
            if p.exists():
                return p
        # Scan slots
        for slot in (0, 1, 2):
            slot_file = self.checkpoints_root / f"{label}_live_{slot:02d}" / "mem_model_optimizer.pt"
            if slot_file.exists():
                return slot_file
        return None

    def build_command_args(
        self,
        *,
        python_exe: str = sys.executable,
        main_script: str = "main.py",
        lane_args: List[str],
        max_steps: int,
        checkpoint_path: Optional[str] = None,
        llm_enabled: bool = True,
        api_executive_moderate: bool = True,
        confirmation_token: str = "I_UNDERSTAND_V89_RECOVERY_CONTROL",
    ) -> List[str]:
        cmd = [
            python_exe,
            main_script,
            "--deepspeed-wsl-accelerated",
            "--operator",
            "--deepspeed-real-micro-train",
            "--real-dataset",
            "--deepspeed-persistent-checkpoint",
            "--deepspeed-max-steps", str(max_steps),
            "--confirm-deepspeed-accelerated", confirmation_token,
        ]
        if llm_enabled:
            cmd.append("--llm")
        if api_executive_moderate:
            cmd.append("--api-executive-mode")
        if checkpoint_path:
            cmd.extend(["--deepspeed-load-checkpoint", str(checkpoint_path)])
        cmd.extend(lane_args)
        return cmd
