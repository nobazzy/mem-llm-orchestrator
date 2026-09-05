from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from runtime.checkpoint_manager import CheckpointManager
from runtime.lm_model import build_tiny_causal_lm
from runtime.real_dataset import RealDatasetBatcher


@dataclass
class LaneConfig:
    name: str
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    min_tokens_floor: float
    expected_peak_tokens: float
    precision: str = "fp16"
    notes: str = ""


def get_lanes_for_model(model_preset: str = "medium_75m") -> Dict[str, LaneConfig]:
    if model_preset in {"xlarge_250m", "decoder_250m", "250m_decoder", "250m", "huge_350m"}:
        return {
            "aggressive_seq256_zero0_gacc4": LaneConfig(
                name="aggressive_seq256_zero0_gacc4",
                batch_size=8,
                sequence_length=256,
                gradient_accumulation_steps=4,
                min_tokens_floor=6000.0,
                expected_peak_tokens=22000.0,
                precision="fp16",
                notes="Primary 250M high-capacity lane",
            ),
            "fast_seq256_zero0_gacc4": LaneConfig(
                name="fast_seq256_zero0_gacc4",
                batch_size=6,
                sequence_length=256,
                gradient_accumulation_steps=4,
                min_tokens_floor=4500.0,
                expected_peak_tokens=16000.0,
                precision="fp16",
                notes="Fast fallback 250M lane",
            ),
            "safe_seq256": LaneConfig(
                name="safe_seq256",
                batch_size=4,
                sequence_length=256,
                gradient_accumulation_steps=2,
                min_tokens_floor=3000.0,
                expected_peak_tokens=10000.0,
                precision="fp16",
                notes="Conservative recovery 250M lane",
            ),
        }

    if model_preset in {"large_130m", "decoder_130m", "130m_decoder", "130m", "medium_100m"}:
        return {
            "aggressive_seq256_zero0_gacc4": LaneConfig(
                name="aggressive_seq256_zero0_gacc4",
                batch_size=24,
                sequence_length=256,
                gradient_accumulation_steps=4,
                min_tokens_floor=18000.0,
                expected_peak_tokens=45000.0,
                precision="fp16",
                notes="Primary 130M high-throughput large-batch lane",
            ),
            "fast_seq256_zero0_gacc4": LaneConfig(
                name="fast_seq256_zero0_gacc4",
                batch_size=16,
                sequence_length=256,
                gradient_accumulation_steps=4,
                min_tokens_floor=12000.0,
                expected_peak_tokens=32000.0,
                precision="fp16",
                notes="Fast fallback 130M lane",
            ),
            "safe_seq256": LaneConfig(
                name="safe_seq256",
                batch_size=8,
                sequence_length=256,
                gradient_accumulation_steps=2,
                min_tokens_floor=6000.0,
                expected_peak_tokens=18000.0,
                precision="fp16",
                notes="Conservative recovery 130M lane",
            ),
        }

    return {
        "aggressive_seq256_zero0_gacc4": LaneConfig(
            name="aggressive_seq256_zero0_gacc4",
            batch_size=16,
            sequence_length=256,
            gradient_accumulation_steps=4,
            min_tokens_floor=18000.0,
            expected_peak_tokens=42000.0,
            precision="fp16",
            notes="Primary high-throughput lane",
        ),
        "fast_seq256_zero0_gacc4": LaneConfig(
            name="fast_seq256_zero0_gacc4",
            batch_size=10,
            sequence_length=256,
            gradient_accumulation_steps=4,
            min_tokens_floor=16000.0,
            expected_peak_tokens=32000.0,
            precision="fp16",
            notes="Stable fast fallback lane",
        ),
        "safe_seq256": LaneConfig(
            name="safe_seq256",
            batch_size=6,
            sequence_length=256,
            gradient_accumulation_steps=2,
            min_tokens_floor=12000.0,
            expected_peak_tokens=22000.0,
            precision="fp16",
            notes="Conservative recovery lane",
        ),
    }


STANDARD_LANES: Dict[str, LaneConfig] = get_lanes_for_model("medium_75m")


class AdaptiveLaneRunner:
    """Autonomous Lane-Switching Training Controller with Real-Time Degradation Watch."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        evidence_dir: Path,
        initial_lane: str = "aggressive_seq256_zero0_gacc4",
        lanes: Optional[Dict[str, LaneConfig]] = None,
        model_preset: str = "medium_75m",
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.model_preset = model_preset
        self.lanes = dict(lanes or get_lanes_for_model(model_preset))
        self.current_lane = self.lanes.get(initial_lane, list(self.lanes.values())[0])
        self.events_file = self.evidence_dir / "v89_sustained_control_events.jsonl"
        self.progress_file = self.evidence_dir / "runtime_progress_latest.json"
        self.milestones_file = self.evidence_dir / "runtime_milestones.jsonl"
        self.controller_status_file = self.evidence_dir.parent / "v89_controller_status_latest.json"
        self.lane_history: List[Dict[str, Any]] = []
        self.last_switch_step = 0
        self.min_switch_cooldown_steps = 100

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            "event": event_type,
            "lane": self.current_lane.name,
            **payload,
        }
        self.lane_history.append(entry)
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _update_controller_status(
        self,
        *,
        step: int,
        target_steps: int,
        tokens_per_sec: float,
        loss: float,
        state: str = "RUNNING_LANE",
        reason: str = "active_training",
        efficiency_score: float = 1.0,
    ) -> None:
        gpu_info = {}
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
            gpu_info = {
                "vram_allocated_mb": round(vram_alloc, 1),
                "vram_reserved_mb": round(vram_res, 1),
                "device_name": torch.cuda.get_device_name(0),
            }

        status = {
            "version": "v89.0.0",
            "timestamp": time.time(),
            "state": state,
            "lane": self.current_lane.name,
            "batch_size": self.current_lane.batch_size,
            "sequence_length": self.current_lane.sequence_length,
            "gradient_accumulation_steps": self.current_lane.gradient_accumulation_steps,
            "min_tokens_floor": self.current_lane.min_tokens_floor,
            "expected_peak_tokens": self.current_lane.expected_peak_tokens,
            "current_tokens_per_second": round(tokens_per_sec, 1),
            "efficiency_score": round(efficiency_score, 4),
            "global_step": step,
            "target_steps": target_steps,
            "loss": round(loss, 4),
            "reason": reason,
            "lane_switches_count": len(self.lane_history),
            "recent_lane_events": self.lane_history[-10:],
            "gpu": gpu_info,
        }
        try:
            self.controller_status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def evaluate_lane_transition(
        self,
        *,
        step: int,
        window_tokens_sec: float,
        window_loss: float,
        optimizer_ratio: float,
        data_wait_ratio: float,
        bad_windows: int,
        stable_windows: int = 0,
    ) -> Tuple[Optional[LaneConfig], str, int, int]:
        """Evaluates whether to switch lane (Promotion or Demotion) with anti-flapping hysteresis."""
        current = self.current_lane

        # Cooldown guard: prevent flapping within cooldown window
        if (step - self.last_switch_step) < self.min_switch_cooldown_steps:
            return None, "cooldown_active", bad_windows, stable_windows

        reasons = []
        if window_tokens_sec < current.min_tokens_floor:
            reasons.append("below_lane_min_tokens")
        if optimizer_ratio > 0.45:
            reasons.append("optimizer_ratio_high")
        if data_wait_ratio > 0.35:
            reasons.append("data_wait_severe")

        # Demotion logic (requires 3 consecutive bad windows)
        if "below_lane_min_tokens" in reasons:
            bad_windows += 1
            stable_windows = 0
            if bad_windows >= 3:
                if current.name == "aggressive_seq256_zero0_gacc4":
                    self.last_switch_step = step
                    return self.lanes["fast_seq256_zero0_gacc4"], f"Demoting to fast fallback: throughput {window_tokens_sec:.0f} < floor {current.min_tokens_floor:.0f}", 0, 0
                elif current.name == "fast_seq256_zero0_gacc4":
                    self.last_switch_step = step
                    return self.lanes["safe_seq256"], f"Demoting to safe recovery: throughput {window_tokens_sec:.0f} < floor {current.min_tokens_floor:.0f}", 0, 0
        else:
            bad_windows = max(0, bad_windows - 1)
            stable_windows += 1

        # Promotion logic (requires 3 consecutive stable windows above threshold)
        if stable_windows >= 3 and optimizer_ratio <= 0.32:
            if current.name == "safe_seq256" and "fast_seq256_zero0_gacc4" in self.lanes:
                fast_floor = self.lanes["fast_seq256_zero0_gacc4"].min_tokens_floor
                if window_tokens_sec >= (fast_floor * 0.9):
                    self.last_switch_step = step
                    return self.lanes["fast_seq256_zero0_gacc4"], f"Promoting to fast lane: sustained throughput {window_tokens_sec:.0f} tok/s", 0, 0
            elif current.name == "fast_seq256_zero0_gacc4" and "aggressive_seq256_zero0_gacc4" in self.lanes:
                agg_floor = self.lanes["aggressive_seq256_zero0_gacc4"].min_tokens_floor
                if window_tokens_sec >= (agg_floor * 0.9):
                    self.last_switch_step = step
                    return self.lanes["aggressive_seq256_zero0_gacc4"], f"Promoting to aggressive lane: sustained throughput {window_tokens_sec:.0f} tok/s", 0, 0

        return None, "keep_current_lane", bad_windows, stable_windows

    def train_loop(
        self,
        *,
        total_steps: int,
        dataset_name: str,
        dataset_config: str = "",
        fallback_name: str = "roneneldan/TinyStories",
        model_preset: str = "medium_75m",
        checkpoint_interval: int = 500,
        eval_window_steps: int = 40,
    ) -> Dict[str, Any]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        use_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        batcher = RealDatasetBatcher(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            fallback_name=fallback_name,
            split="train",
            streaming=True,
            tokenizer_name="gpt2",
            sequence_length=self.current_lane.sequence_length,
            batch_size=self.current_lane.batch_size,
            device=device,
        )

        model = build_tiny_causal_lm(
            vocab_size=batcher.vocab_size,
            seq_len=self.current_lane.sequence_length,
            preset=model_preset,
        ).to(device=device, dtype=torch.float32)

        base_lr = 2e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, eps=1e-6, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        total_tokens_processed = 0
        start_time = time.perf_counter()
        loss_first = None
        loss_last = None

        self._log_event("training_started", {
            "lane": self.current_lane.name,
            "batch_size": self.current_lane.batch_size,
            "target_steps": total_steps,
            "model_preset": model_preset,
            "dataset": dataset_name,
        })

        window_start_time = time.perf_counter()
        window_tokens = 0
        window_steps = 0
        window_losses: List[float] = []
        data_fetch_durations: List[float] = []
        optimizer_durations: List[float] = []
        step_durations: List[float] = []
        bad_windows = 0
        stable_windows = 0

        for step in range(1, total_steps + 1):
            t_step_start = time.perf_counter()

            t_data_start = time.perf_counter()
            input_ids, labels = batcher.next_batch()
            data_fetch_time = time.perf_counter() - t_data_start
            data_fetch_durations.append(data_fetch_time)

            t_fwd_start = time.perf_counter()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            else:
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            loss_val = float(loss.detach().item())
            if loss_first is None:
                loss_first = loss_val
            loss_last = loss_val
            window_losses.append(loss_val)

            loss.backward()

            t_opt_start = time.perf_counter()
            if step % self.current_lane.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            opt_time = time.perf_counter() - t_opt_start
            optimizer_durations.append(opt_time)

            num_tokens = int(input_ids.numel())
            total_tokens_processed += num_tokens
            window_tokens += num_tokens
            window_steps += 1

            total_step_time = time.perf_counter() - t_step_start
            step_durations.append(total_step_time)

            if step % eval_window_steps == 0 or step == total_steps:
                window_elapsed = max(time.perf_counter() - window_start_time, 1e-6)
                win_tok_sec = window_tokens / window_elapsed
                win_step_sec = window_steps / window_elapsed
                avg_win_loss = sum(window_losses) / max(1, len(window_losses))
                avg_data_ratio = sum(data_fetch_durations) / max(1e-6, sum(step_durations))
                avg_opt_ratio = sum(optimizer_durations) / max(1e-6, sum(step_durations))

                overall_elapsed = max(time.perf_counter() - start_time, 1e-6)
                cum_tok_sec = total_tokens_processed / overall_elapsed
                cum_step_sec = step / overall_elapsed

                efficiency_score = min(win_tok_sec / self.current_lane.expected_peak_tokens, 1.25)

                prog_payload = {
                    "step": step,
                    "target_steps": total_steps,
                    "tokens_processed": total_tokens_processed,
                    "tokens_per_second": round(win_tok_sec, 2),
                    "cumulative_tokens_per_second": round(cum_tok_sec, 2),
                    "steps_per_second": round(win_step_sec, 2),
                    "loss": round(loss_val, 4),
                    "loss_first": round(loss_first, 4) if loss_first else None,
                    "loss_last": round(loss_last, 4),
                    "phase": "training",
                    "bottleneck": "gpu_compute",
                    "lane": self.current_lane.name,
                    "elapsed_seconds": round(overall_elapsed, 2),
                    "timestamp": time.time(),
                }
                try:
                    self.progress_file.write_text(json.dumps(prog_payload, indent=2), encoding="utf-8")

                    milestone_entry = {
                        "step": step,
                        "loss": round(loss_val, 4),
                        "tokens_per_second": round(win_tok_sec, 2),
                        "steps_per_second": round(win_step_sec, 2),
                        "tokens_processed": total_tokens_processed,
                        "lane": self.current_lane.name,
                        "elapsed_seconds": round(overall_elapsed, 2),
                        "timestamp": time.time(),
                    }
                    with open(self.milestones_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(milestone_entry) + "\n")
                except Exception:
                    pass

                self._update_controller_status(
                    step=step,
                    target_steps=total_steps,
                    tokens_per_sec=win_tok_sec,
                    loss=loss_val,
                    state="RUNNING_LANE",
                    reason=f"Window throughput: {win_tok_sec:.0f} tok/s",
                    efficiency_score=efficiency_score,
                )

                new_lane, trans_reason, bad_windows, stable_windows = self.evaluate_lane_transition(
                    step=step,
                    window_tokens_sec=win_tok_sec,
                    window_loss=avg_win_loss,
                    optimizer_ratio=avg_opt_ratio,
                    data_wait_ratio=avg_data_ratio,
                    bad_windows=bad_windows,
                    stable_windows=stable_windows,
                )

                if new_lane is not None and new_lane.name != self.current_lane.name:
                    old_lane_name = self.current_lane.name
                    self.current_lane = new_lane
                    batcher.batch_size = new_lane.batch_size
                    self._log_event("lane_switched", {
                        "from_lane": old_lane_name,
                        "to_lane": new_lane.name,
                        "step": step,
                        "reason": trans_reason,
                        "trigger_tokens_sec": round(win_tok_sec, 1),
                        "new_batch_size": new_lane.batch_size,
                        "new_gradient_accumulation": new_lane.gradient_accumulation_steps,
                    })
                    self._update_controller_status(
                        step=step,
                        target_steps=total_steps,
                        tokens_per_sec=win_tok_sec,
                        loss=loss_val,
                        state="LANE_SWITCHED",
                        reason=trans_reason,
                        efficiency_score=efficiency_score,
                    )

                window_start_time = time.perf_counter()
                window_tokens = 0
                window_steps = 0
                window_losses = []
                data_fetch_durations = []
                optimizer_durations = []
                step_durations = []

            if step % checkpoint_interval == 0:
                try:
                    self.checkpoint_manager.save_live_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        metadata={
                            "version": "v89.0.0",
                            "step": step,
                            "tokens_processed": total_tokens_processed,
                            "loss": loss_val,
                            "lane": self.current_lane.name,
                        },
                        label="v89",
                    )
                except Exception:
                    pass

        return {
            "steps_completed": total_steps,
            "tokens_processed": total_tokens_processed,
            "loss_first": loss_first,
            "loss_last": loss_last,
            "lane_history": self.lane_history,
        }