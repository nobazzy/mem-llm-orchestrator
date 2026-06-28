from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class RealChaosSample:
    ts: float
    cpu_percent: float
    ram_percent: float
    loadavg_1m: float
    gpu_util_percent: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_temperature_c: float
    gpu_power_w: float
    chaos_profile: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealChaosProbe:
    """Records real environmental pressure, not simulated faults."""

    def __init__(self, chaos_profile: str = "real_streaming_mix") -> None:
        self.chaos_profile = chaos_profile

    def score(self, sample: Dict[str, Any]) -> float:
        """Compute a single chaos pressure score from a sample dict.

        Weights:
          cpu_percent       * 0.15  — CPU contention contribution (0-100 scale -> 0-15)
          ram_percent       * 0.05  — RAM pressure (lighter weight; RAM is more elastic)
          gpu_vram_ratio    * 50.0  — VRAM saturation (0-1 ratio scaled to 0-50, dominant signal)
          gpu_util_percent  * 0.30  — GPU compute utilisation (0-100 scale -> 0-30)

        Maximum theoretical score: 15 + 5 + 50 + 30 = 100.
        """
        gpu_vram_ratio = sample.get("gpu_memory_used_mb", 0.0) / max(sample.get("gpu_memory_total_mb", 1.0), 1.0)
        raw = (
            sample.get("cpu_percent", 0.0)    * 0.15 +
            sample.get("ram_percent", 0.0)    * 0.05 +
            gpu_vram_ratio                    * 50.0 +
            sample.get("gpu_util_percent", 0.0) * 0.30
        )
        return round(raw, 3)

    def sample(self) -> Dict[str, Any]:
        cpu_percent = 0.0
        ram_percent = 0.0
        loadavg_1m = 0.0
        try:
            import psutil  # type: ignore
            cpu_percent = float(psutil.cpu_percent(interval=None))
            ram_percent = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        try:
            loadavg_1m = float(os.getloadavg()[0])
        except Exception:
            pass
        gpu_util = gpu_used = gpu_total = gpu_temp = gpu_power = 0.0
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ], text=True, stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()[0]
            vals = [x.strip() for x in out.split(',')]
            gpu_util, gpu_used, gpu_total, gpu_temp, gpu_power = [float(v) for v in vals[:5]]
        except Exception:
            pass
        return RealChaosSample(
            ts=round(time.time(), 3),
            cpu_percent=round(cpu_percent, 3),
            ram_percent=round(ram_percent, 3),
            loadavg_1m=round(loadavg_1m, 3),
            gpu_util_percent=round(gpu_util, 3),
            gpu_memory_used_mb=round(gpu_used, 3),
            gpu_memory_total_mb=round(gpu_total, 3),
            gpu_temperature_c=round(gpu_temp, 3),
            gpu_power_w=round(gpu_power, 3),
            chaos_profile=self.chaos_profile,
        ).to_dict()
