from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LaneDefinition:
    name: str
    batch_size: int
    zero_stage: int
    precision: str
    gradient_accumulation_steps: int
    sequence_length: int
    model_preset: str = "tiny_decoder"
    min_tokens_floor: float = 12000.0
    expected_peak_tokens: float = 33000.0
    notes: str = ""

    def to_cli_args(self) -> List[str]:
        return [
            "--deepspeed-batch-size", str(self.batch_size),
            "--deepspeed-zero-stage", str(self.zero_stage),
            "--deepspeed-precision", self.precision,
            "--deepspeed-gradient-accumulation-steps", str(self.gradient_accumulation_steps),
            "--sequence-length", str(self.sequence_length),
            "--model-preset", self.model_preset,
        ]


class LaneManager:
    """Manages training execution lanes and transition rules."""

    STANDARD_LANES: Dict[str, LaneDefinition] = {
        "fast_seq256_zero0_gacc4": LaneDefinition(
            name="fast_seq256_zero0_gacc4",
            batch_size=8,
            zero_stage=0,
            precision="fp16",
            gradient_accumulation_steps=4,
            sequence_length=256,
            min_tokens_floor=17500.0,
            expected_peak_tokens=35000.0,
            notes="Primary high-throughput lane",
        ),
        "aggressive_seq256_zero0_gacc4": LaneDefinition(
            name="aggressive_seq256_zero0_gacc4",
            batch_size=8,
            zero_stage=0,
            precision="fp16",
            gradient_accumulation_steps=4,
            sequence_length=256,
            min_tokens_floor=18000.0,
            expected_peak_tokens=42000.0,
            notes="High-capacity throughput lane",
        ),
        "safe_seq256": LaneDefinition(
            name="safe_seq256",
            batch_size=4,
            zero_stage=1,
            precision="fp16",
            gradient_accumulation_steps=4,
            sequence_length=256,
            min_tokens_floor=12000.0,
            expected_peak_tokens=22000.0,
            notes="Conservative recovery lane with ZeRO-1",
        ),
    }

    def __init__(self, custom_lanes: Optional[Dict[str, LaneDefinition]] = None) -> None:
        self.lanes = dict(custom_lanes or self.STANDARD_LANES)

    def get_lane(self, name: str) -> LaneDefinition:
        if name not in self.lanes:
            raise KeyError(f"Unknown lane '{name}'. Supported lanes: {list(self.lanes.keys())}")
        return self.lanes[name]

    def is_valid_lane(self, name: str) -> bool:
        return name in self.lanes

    def next_recovery_lane(self, current_lane: str) -> LaneDefinition:
        if current_lane in {"fast_seq256_zero0_gacc4", "aggressive_seq256_zero0_gacc4"}:
            return self.lanes["safe_seq256"]
        return self.lanes.get("safe_seq256", list(self.lanes.values())[0])

    def can_promote(self, current_lane: str, tokens_per_second: float, optimizer_ratio: float) -> bool:
        if current_lane == "safe_seq256":
            return tokens_per_second >= 17000.0 and optimizer_ratio <= 0.42
        return False
