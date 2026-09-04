from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DegradationMetrics:
    step: int
    tokens_per_second: float
    steps_per_second: float
    optimizer_ratio: float
    gpu_utilization: float
    bottleneck: str = "unknown"
    loss: Optional[float] = None
    nan_or_inf: bool = False


class DegradationDetector:
    """Evaluates telemetry stream to detect stall, memory pressure, and throughput drop."""

    def __init__(
        self,
        *,
        min_degrade_step: int = 3000,
        drop_ratio: float = 0.68,
        required_bad_windows: int = 4,
        health_good_tokens: float = 19000.0,
        health_acceptable_tokens: float = 17500.0,
        health_attention_tokens: float = 16000.0,
    ) -> None:
        self.min_degrade_step = min_degrade_step
        self.drop_ratio = drop_ratio
        self.required_bad_windows = required_bad_windows
        self.health_good_tokens = health_good_tokens
        self.health_acceptable_tokens = health_acceptable_tokens
        self.health_attention_tokens = health_attention_tokens

        self.best_tokens_per_second: float = 0.0
        self.bad_window_count: int = 0
        self.history: List[DegradationMetrics] = []

    def update(self, metrics: DegradationMetrics) -> Dict[str, Any]:
        self.history.append(metrics)
        self.best_tokens_per_second = max(self.best_tokens_per_second, metrics.tokens_per_second)

        degraded = False
        if (
            metrics.step >= self.min_degrade_step
            and self.best_tokens_per_second > 0
            and metrics.tokens_per_second < (self.best_tokens_per_second * self.drop_ratio)
            and metrics.optimizer_ratio >= 0.40
        ):
            degraded = True
            self.bad_window_count += 1
        else:
            self.bad_window_count = max(0, self.bad_window_count - 1)

        health = "HEALTHY"
        if metrics.tokens_per_second < self.health_attention_tokens or metrics.optimizer_ratio > 0.50:
            health = "ATTENTION"
        elif metrics.tokens_per_second < self.health_acceptable_tokens:
            health = "ACCEPTABLE"
        elif metrics.tokens_per_second >= self.health_good_tokens:
            health = "OPTIMAL"

        should_switch = self.bad_window_count >= self.required_bad_windows or metrics.nan_or_inf

        return {
            "health": health,
            "degraded": degraded,
            "bad_windows": self.bad_window_count,
            "best_tokens_per_second": round(self.best_tokens_per_second, 2),
            "should_switch": should_switch,
            "recommended_action": "lane_switch" if should_switch else "observe",
        }
