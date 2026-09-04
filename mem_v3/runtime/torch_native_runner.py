from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from runtime.deepspeed_runner import DeepSpeedRunMetrics, _percentile, _safe_int
from runtime.profiler import RuntimeProfiler
from runtime.adaptive_memory import AdaptiveRuntimeMemory
from runtime.real_chaos import RealChaosProbe


class PyTorchNativeRunner:
    """Portable, pure PyTorch execution runner for environments without DeepSpeed."""

    def __init__(self, checkpoint_manager: Any) -> None:
        self.checkpoint_manager = checkpoint_manager

    def run(
        self,
        *,
        steps: int,
        batch_size: int,
        zero_stage: int = 0,
        precision: str = "fp32",
        persistent_checkpoint: bool = False,
        load_checkpoint: str = "",
        gradient_accumulation_steps: int = 1,
        applied_hyperparams: Dict[str, Any] | None = None,
        executive_directives: Dict[str, Any] | None = None,
        dataset_settings: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        applied = dict(applied_hyperparams or {})
        directives = dict(executive_directives or {})
        data_cfg = dict(dataset_settings or {})

        evidence_path = Path(str(data_cfg.get("evidence_dir", ""))) if str(data_cfg.get("evidence_dir", "")) else None
        progress_path = (evidence_path / "runtime_progress_latest.json") if evidence_path else None
        milestones_path = (evidence_path / "runtime_milestones.jsonl") if evidence_path else None

        profiler = RuntimeProfiler(evidence_path)
        adaptive_memory = AdaptiveRuntimeMemory(evidence_path)
        chaos_profile = str(data_cfg.get("chaos_profile", "clean"))
        chaos_probe = RealChaosProbe(chaos_profile)

        effective_batch = _safe_int(applied.get("batch_size", batch_size), batch_size, low=1, high=16)
        effective_precision = str(applied.get("precision", precision)).lower()
        if effective_precision not in {"fp32", "fp16", "bf16"}:
            effective_precision = "fp32"
        effective_grad_accum = _safe_int(applied.get("gradient_accumulation_steps", gradient_accumulation_steps), gradient_accumulation_steps, low=1, high=16)
        effective_steps = max(1, int(steps))
        seq_len = _safe_int(data_cfg.get("sequence_length", 128), 128, low=32, high=512)

        api_lr_multiplier = float(directives.get("lr_multiplier", 1.0)) if directives.get("enabled") else 1.0
        api_lr_multiplier = max(0.20, min(1.0, api_lr_multiplier))
        base_lr = 2e-4 if effective_precision == "fp16" else 5e-4
        optimizer_lr = base_lr * api_lr_multiplier

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.float32

        use_amp = (device.type == "cuda") and (effective_precision in {"fp16", "bf16"})
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        real_dataset = bool(data_cfg.get("real_dataset", False))
        dataset_info: Dict[str, Any] = {"enabled": real_dataset, "task": "causal_language_modeling"}

        if real_dataset:
            from runtime.lm_model import build_tiny_causal_lm
            from runtime.real_dataset import RealDatasetBatcher
            batcher = RealDatasetBatcher(
                dataset_name=str(data_cfg.get("dataset_name", "HuggingFaceFW/fineweb-edu")),
                dataset_config=str(data_cfg.get("dataset_config", "sample-10BT")),
                fallback_name=str(data_cfg.get("dataset_fallback_name", "roneneldan/TinyStories")),
                split=str(data_cfg.get("dataset_split", "train")),
                streaming=bool(data_cfg.get("dataset_streaming", True)),
                tokenizer_name=str(data_cfg.get("tokenizer_name", "gpt2")),
                sequence_length=seq_len,
                batch_size=effective_batch,
                device=device,
                chaos_profile=chaos_profile,
                dataset_mix=str(data_cfg.get("dataset_mix", "")),
            )
            model_preset = str(data_cfg.get("model_preset", "tiny_decoder"))
            model = build_tiny_causal_lm(batcher.vocab_size, seq_len, model_preset).to(device=device, dtype=model_dtype)
            dataset_info = batcher.info.to_dict()
            criterion = nn.CrossEntropyLoss()
        else:
            batcher = None
            model = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).to(device=device, dtype=model_dtype)
            criterion = None

        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr, eps=1e-6)

        checkpoint: Dict[str, Any] = {"checkpoint_written": False, "checkpoint_mode": "not_attempted"}
        metrics = DeepSpeedRunMetrics(
            micro_train_step_target=effective_steps,
            effective_batch_size=effective_batch,
            effective_zero_stage=zero_stage,
            effective_precision=effective_precision,
            effective_gradient_accumulation_steps=effective_grad_accum,
            train_micro_batch_size_per_gpu=effective_batch,
            train_batch_size=effective_batch * effective_grad_accum,
            sequence_length=seq_len,
            model_preset=str(data_cfg.get("model_preset", "tiny_decoder")),
            applied_hyperparams_consumed=bool(applied),
            optimizer_lr=optimizer_lr,
            gradient_clip_norm=float(directives.get("gradient_clip_norm", 1.0)) if directives.get("enabled") else 1.0,
            api_executive_enabled=bool(directives.get("enabled")),
            api_executive_action=str(directives.get("action", "")),
            api_lr_multiplier=api_lr_multiplier,
            dataset=dataset_info,
            workload="real_dataset_causal_lm" if real_dataset else "synthetic_benchmark",
            benchmark_mode="mem_native_pytorch",
            chaos_profile=chaos_profile,
            chaos_environment=chaos_probe.sample(),
        )

        step_times: List[float] = []
        start_time = time.perf_counter()
        first_param = next(model.parameters()).detach().clone().float()

        for step in range(1, effective_steps + 1):
            step_start = time.perf_counter()
            data_start = time.perf_counter()

            if real_dataset and batcher is not None:
                input_ids, labels = batcher.next_batch()
                data_fetch_seconds = time.perf_counter() - data_start
                forward_start = time.perf_counter()
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(input_ids)
                        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                else:
                    logits = model(input_ids)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                metrics.tokens_processed += int(input_ids.numel())
            else:
                x = torch.randn(effective_batch, 16, device=device, dtype=model_dtype)
                y = (0.01 * torch.tanh(x.float()[:, :1])).to(device=device, dtype=torch.float32)
                data_fetch_seconds = time.perf_counter() - data_start
                forward_start = time.perf_counter()
                out = model(x)
                loss = ((out.float() - y) ** 2).mean()

            forward_loss_seconds = time.perf_counter() - forward_start
            loss_val = float(loss.detach().float().item())
            if metrics.loss_first is None:
                metrics.loss_first = loss_val
            metrics.loss_last = loss_val

            if not math.isfinite(loss_val):
                metrics.loss_finite = False
                metrics.nan_or_inf_detected = True
                break

            backward_start = time.perf_counter()
            loss.backward()
            backward_seconds = time.perf_counter() - backward_start
            metrics.backward_count += 1

            # Guardrails
            guardrail_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=metrics.gradient_clip_norm)
            guardrail_seconds = time.perf_counter() - guardrail_start

            optimizer_start = time.perf_counter()
            if step % effective_grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                metrics.optimizer_step_count += 1
            optimizer_seconds = time.perf_counter() - optimizer_start

            metrics.forward_count += 1
            metrics.micro_train_steps_completed = step
            total_step_seconds = time.perf_counter() - step_start
            step_times.append(total_step_seconds)

            profiler.add(
                data_fetch=data_fetch_seconds,
                forward_loss=forward_loss_seconds,
                backward=backward_seconds,
                optimizer=optimizer_seconds,
                guardrail=guardrail_seconds,
                total_step=total_step_seconds,
            )

            heartbeat_interval = _safe_int(data_cfg.get("progress_heartbeat_interval", 5), 5, low=1, high=100)
            if evidence_path and (step % heartbeat_interval == 0 or step == effective_steps):
                now_elapsed = max(time.perf_counter() - start_time, 1e-9)
                cur_tokens_sec = round(metrics.tokens_processed / now_elapsed, 2)
                cur_steps_sec = round(step / now_elapsed, 2)
                prog_payload = {
                    "step": step,
                    "target_steps": data_cfg.get("target_steps", effective_steps),
                    "tokens_processed": metrics.tokens_processed,
                    "tokens_per_second": cur_tokens_sec,
                    "steps_per_second": cur_steps_sec,
                    "loss": round(loss_val, 4),
                    "loss_first": round(metrics.loss_first, 4) if metrics.loss_first is not None else None,
                    "loss_last": round(loss_val, 4),
                    "phase": "training",
                    "bottleneck": "gpu_compute" if torch.cuda.is_available() else "cpu_compute",
                    "elapsed_seconds": round(now_elapsed, 2),
                    "timestamp": time.time(),
                }
                try:
                    evidence_path.mkdir(parents=True, exist_ok=True)
                    if progress_path:
                        tmp_prog = progress_path.with_suffix(".tmp")
                        tmp_prog.write_text(json.dumps(prog_payload, indent=2), encoding="utf-8")
                        tmp_prog.replace(progress_path)
                    if milestones_path:
                        milestone_entry = {
                            "step": step,
                            "loss": round(loss_val, 4),
                            "tokens_per_second": cur_tokens_sec,
                            "steps_per_second": cur_steps_sec,
                            "tokens_processed": metrics.tokens_processed,
                            "elapsed_seconds": round(now_elapsed, 2),
                            "timestamp": time.time(),
                        }
                        with open(milestones_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(milestone_entry) + "\n")
                except Exception:
                    pass

            checkpoint_interval = _safe_int(data_cfg.get("checkpoint_interval", 500), 500, low=50, high=100000)
            if persistent_checkpoint and self.checkpoint_manager and step % checkpoint_interval == 0:
                try:
                    self.checkpoint_manager.save_live_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        metadata={
                            "version": "v89.0.0",
                            "step": step,
                            "micro_train_steps_completed": step,
                            "tokens_processed": metrics.tokens_processed,
                            "loss": loss_val,
                            "tokens_per_second": cur_tokens_sec,
                            "steps_per_second": cur_steps_sec,
                        },
                        label="v89",
                    )
                except Exception:
                    pass

        total_elapsed = max(time.perf_counter() - start_time, 1e-9)
        metrics.total_seconds = round(total_elapsed, 6)
        metrics.avg_step_seconds = round(total_elapsed / max(1, metrics.micro_train_steps_completed), 9)
        metrics.steps_per_second = round(metrics.micro_train_steps_completed / total_elapsed, 3)
        metrics.tokens_per_second = round(metrics.tokens_processed / total_elapsed, 3)
        metrics.step_seconds_p95 = round(_percentile(step_times, 95), 9)
        metrics.step_seconds_p99 = round(_percentile(step_times, 99), 9)
        metrics.parameter_delta_abs_sum_positive = bool(
            (next(model.parameters()).detach().float() - first_param).abs().sum().float().item() > 0
        )
        metrics.safe_teardown_completed = True
        metrics.process_group_destroyed_or_not_initialized = True
        metrics.profiler = profiler.summary()
        metrics.adaptive_memory = adaptive_memory.summary()

        metrics.execution_performed = bool(
            metrics.micro_train_steps_completed == effective_steps
            and metrics.loss_finite
            and not metrics.nan_or_inf_detected
            and metrics.parameter_delta_abs_sum_positive
        )

        if persistent_checkpoint and metrics.execution_performed and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.save_post_train(
                model, optimizer,
                {
                    "version": "v89.0.0",
                    "workload": metrics.workload,
                    "tokens_processed": metrics.tokens_processed,
                    "micro_train_steps_completed": metrics.micro_train_steps_completed,
                    "batch_size": effective_batch,
                    "precision": effective_precision,
                    "steps_per_second": metrics.steps_per_second,
                    "tokens_per_second": metrics.tokens_per_second,
                },
                label="v89",
            )

        return metrics.to_dict(), checkpoint
