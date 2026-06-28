from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RuntimeProfiler:
    evidence_dir: Optional[Path] = None
    data_fetch_seconds: List[float] = field(default_factory=list)
    forward_loss_seconds: List[float] = field(default_factory=list)
    backward_seconds: List[float] = field(default_factory=list)
    optimizer_seconds: List[float] = field(default_factory=list)
    guardrail_seconds: List[float] = field(default_factory=list)
    total_step_seconds: List[float] = field(default_factory=list)

    def add(self, *, data_fetch: float, forward_loss: float, backward: float, optimizer: float, guardrail: float, total_step: float) -> None:
        self.data_fetch_seconds.append(float(data_fetch))
        self.forward_loss_seconds.append(float(forward_loss))
        self.backward_seconds.append(float(backward))
        self.optimizer_seconds.append(float(optimizer))
        self.guardrail_seconds.append(float(guardrail))
        self.total_step_seconds.append(float(total_step))

    @staticmethod
    def _sum(xs: List[float]) -> float:
        return float(sum(xs))

    @staticmethod
    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def summary(self, *, gpu_utilization_hint: float = 0.0) -> Dict[str, float | str | int | bool]:
        total = max(self._sum(self.total_step_seconds), 1e-9)
        data = self._sum(self.data_fetch_seconds)
        fwd = self._sum(self.forward_loss_seconds)
        bwd = self._sum(self.backward_seconds)
        opt = self._sum(self.optimizer_seconds)
        grd = self._sum(self.guardrail_seconds)
        ratios = {
            "data_wait_ratio": data / total,
            "forward_loss_ratio": fwd / total,
            "backward_ratio": bwd / total,
            "optimizer_ratio": opt / total,
            "guardrail_ratio": grd / total,
        }
        bottleneck = "balanced_or_gpu_compute"
        if ratios["data_wait_ratio"] > 0.35:
            bottleneck = "data_or_tokenizer_bound"
        elif ratios["forward_loss_ratio"] + ratios["backward_ratio"] > 0.55:
            bottleneck = "model_compute_bound"
        elif ratios["optimizer_ratio"] > 0.25:
            bottleneck = "optimizer_or_sync_bound"
        return {
            "profiler_enabled": True,
            "samples": len(self.total_step_seconds),
            "total_profiled_seconds": round(total, 6),
            "avg_step_seconds_profiled": round(self._avg(self.total_step_seconds), 9),
            "avg_data_fetch_seconds": round(self._avg(self.data_fetch_seconds), 9),
            "avg_forward_loss_seconds": round(self._avg(self.forward_loss_seconds), 9),
            "avg_backward_seconds": round(self._avg(self.backward_seconds), 9),
            "avg_optimizer_seconds": round(self._avg(self.optimizer_seconds), 9),
            "avg_guardrail_seconds": round(self._avg(self.guardrail_seconds), 9),
            **{k: round(v, 6) for k, v in ratios.items()},
            "bottleneck_classification": bottleneck,
            "gpu_utilization_hint": float(gpu_utilization_hint),
        }

    def write(self) -> None:
        if not self.evidence_dir:
            return
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        (self.evidence_dir / "profiler_report.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
