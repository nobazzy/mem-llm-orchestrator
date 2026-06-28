from __future__ import annotations

# MEM v89/v5: defensive effective batch override.
# Keeps micro-batch unchanged and increases only gradient accumulation.
def _mem_v89_apply_gacc_override(args):
    try:
        import os as _mem_v89_gacc_os
        _raw = (
            _mem_v89_gacc_os.environ.get("MEM_V89_FORCE_GRAD_ACCUM")
            or _mem_v89_gacc_os.environ.get("MEM_V89_GRADIENT_ACCUMULATION_STEPS")
        )
        if not _raw:
            return args
        _gacc = max(1, int(float(_raw)))
        for _attr in (
            "gradient_accumulation_steps",
            "gradient_accumulation",
            "grad_accum_steps",
            "grad_accum",
            "gradient_accumulation_step",
        ):
            if hasattr(args, _attr):
                setattr(args, _attr, _gacc)
        return args
    except Exception:
        return args


import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import time as _mem_v89_time
import json as _mem_v89_json
import os as _mem_v89_os
import math as _mem_v89_math
def _mem_v89_scheduled_lr(base_lr, local_step):
    """
    MEM v89 LR scheduler - restart safe.

    IMPORTANT:
    LR is calculated from effective GLOBAL step:
      global_step = MEM_V89_GLOBAL_STEP_START + local_step

    Fallback:
      if MEM_V89_GLOBAL_STEP_START is missing, read checkpoints/v89_latest.txt
      and its metadata.json step/global_step.

    This prevents LR warmup/decay from resetting after lane restart/checkpoint resume.
    """
    import os as _os
    import math as _math
    import json as _json
    from pathlib import Path as _Path

    def _int(x, default=0):
        try:
            return int(float(x))
        except Exception:
            return default

    def _float(x, default):
        try:
            return float(x)
        except Exception:
            return default

    if not hasattr(_mem_v89_scheduled_lr, "_start_step"):
        start = _int(_os.environ.get("MEM_V89_GLOBAL_STEP_START"), 0)

        if start <= 0:
            try:
                latest = _Path("checkpoints/v89_latest.txt")
                if latest.exists():
                    ckpt = _Path(latest.read_text(encoding="utf-8", errors="ignore").strip())
                    meta = ckpt.parent / "metadata.json"
                    if meta.exists():
                        d = _json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
                        start = _int(d.get("global_step") or d.get("step"), 0)
            except Exception:
                start = 0

        _mem_v89_scheduled_lr._start_step = max(0, start)

    start_step = int(getattr(_mem_v89_scheduled_lr, "_start_step", 0))
    local_step = max(0, _int(local_step, 0))
    global_step = max(1, start_step + local_step)

    base = _float(_os.environ.get("MEM_V89_BASE_LR_REAL"), _float(base_lr, 2.0e-5))
    peak_cap = _float(_os.environ.get("MEM_V89_LR_PEAK_CAP"), 1.8e-5)
    warmup = max(0, _int(_os.environ.get("MEM_V89_LR_WARMUP_STEPS"), 10000))
    decay_steps = max(1, _int(_os.environ.get("MEM_V89_LR_DECAY_STEPS"), 200000))
    min_mult = max(0.0, min(1.0, _float(_os.environ.get("MEM_V89_LR_MIN_MULT"), 0.05)))

    if warmup > 0 and global_step <= warmup:
        mult = max(0.05, global_step / max(1, warmup))
    else:
        denom = max(1, decay_steps - warmup)
        progress = min(1.0, max(0.0, (global_step - warmup) / denom))
        mult = min_mult + 0.5 * (1.0 - min_mult) * (1.0 + _math.cos(_math.pi * progress))

    lr = float(base) * float(mult)
    return min(float(lr), float(peak_cap))

def _mem_v89_patch_ds_load_checkpoint_for_cross_zero_resume(engine):
    """Patch DeepSpeedEngine.load_checkpoint to tolerate zero0->zero1 optimizer incompatibility."""
    if engine is None:
        return engine

    if getattr(engine, "_mem_v89_cross_zero_resume_patch", False):
        return engine

    original_load_checkpoint = engine.load_checkpoint

    def _patched_load_checkpoint(*args, **kwargs):
        try:
            return original_load_checkpoint(*args, **kwargs)
        except AssertionError as exc:
            msg = str(exc)

            if "Empty ds_version in checkpoint" not in msg:
                raise

            # Only fallback for the known DeepSpeed optimizer-state incompatibility.
            kwargs2 = dict(kwargs)
            kwargs2["load_optimizer_states"] = False
            kwargs2["load_lr_scheduler_states"] = False

            import json
            import time
            from pathlib import Path as _Path

            event = {
                "event": "optimizer_state_reinitialized",
                "reason": "empty_ds_version_cross_zero_stage_resume",
                "message": msg,
                "loaded_model_expected": True,
                "loaded_optimizer": False,
                "ts": time.time(),
            }

            try:
                ev_dir = _Path("evidence_packets")
                ev_dir.mkdir(parents=True, exist_ok=True)
                with (ev_dir / "v89_optimizer_reinit_events.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                pass

            print(
                "[MEM_V89_RECOVERY] Empty ds_version in checkpoint; "
                "reloading model without optimizer/lr scheduler state. "
                "Optimizer will be reinitialized.",
                flush=True,
            )

            return original_load_checkpoint(*args, **kwargs2)

    engine.load_checkpoint = _patched_load_checkpoint
    engine._mem_v89_cross_zero_resume_patch = True
    return engine




@dataclass
class DeepSpeedRunMetrics:
    execution_performed: bool = False
    engine_initialized: bool = False
    engine_class: str = ""
    workload: str = "real_dataset_causal_lm"
    forward_count: int = 0
    backward_count: int = 0
    optimizer_step_count: int = 0
    micro_train_steps_completed: int = 0
    micro_train_step_target: int = 0
    total_seconds: float = 0.0
    avg_step_seconds: float = 0.0
    steps_per_second: float = 0.0
    tokens_per_second: float = 0.0
    tokens_processed: int = 0
    step_seconds_p95: float = 0.0
    step_seconds_p99: float = 0.0
    loss_first: float | None = None
    loss_last: float | None = None
    loss_finite: bool = True
    nan_or_inf_detected: bool = False
    gradients_observed: bool = False
    grad_finite: bool = True
    grad_nonfinite_detected: bool = False
    parameter_delta_abs_sum_positive: bool = False
    forward_output_device: str = ""
    loss_device: str = ""
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0
    cuda_max_allocated_mb: float = 0.0
    cuda_max_reserved_mb: float = 0.0
    optimizer_path: str = "torch_adamw_external_no_fused_adam"
    effective_batch_size: int = 1
    effective_zero_stage: int = 0
    effective_precision: str = "fp32"
    effective_gradient_accumulation_steps: int = 1
    train_micro_batch_size_per_gpu: int = 1
    train_batch_size: int = 1
    sequence_length: int = 128
    model_preset: str = "tiny_decoder"
    applied_hyperparams_consumed: bool = False
    input_dtype: str = "torch.long"
    param_dtype: str = ""
    safe_teardown_completed: bool = False
    process_group_destroyed_or_not_initialized: bool = False
    numerical_guardrails_enabled: bool = True
    numerical_recovery_events: int = 0
    grad_sanitized_events: int = 0
    loss_sanitized_events: int = 0
    first_nonfinite_step: int = 0
    first_nonfinite_kind: str = ""
    gradient_clip_norm: float = 1.0
    max_grad_abs_after_guardrail: float = 0.0
    optimizer_lr: float = 0.0
    api_executive_enabled: bool = False
    api_executive_action: str = ""
    api_lr_multiplier: float = 1.0
    api_numerical_recovery_budget: int = 1000
    api_checkpoint_milestones: List[int] = field(default_factory=list)
    api_event_triggers: List[str] = field(default_factory=list)
    api_runtime_changes_count: int = 0
    api_directives_applied: int = 0
    api_directives_rejected_by_local_policy: int = 0
    dataset: Dict[str, Any] = field(default_factory=dict)
    profiler: Dict[str, Any] = field(default_factory=dict)
    adaptive_memory: Dict[str, Any] = field(default_factory=dict)
    benchmark_mode: str = "mem_adaptive"
    bottleneck_classification: str = "unknown"
    chaos_profile: str = "clean"
    chaos_environment: Dict[str, Any] = field(default_factory=dict)
    real_chaos_score: float = 0.0
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    progress_write_failures: int = 0
    gradient_audit: Dict[str, Any] = field(default_factory=dict)
    guardrail_mode: str = "sampled"
    guardrail_sample_interval: int = 8
    guardrail_full_checks: int = 0
    guardrail_sampled_skips: int = 0
    grad_clip_applied_events: int = 0
    grad_clamp_events: int = 0
    grad_norm_before_clip_last: float = 0.0
    grad_norm_after_guardrail_last: float = 0.0
    grad_norm_before_clip_max: float = 0.0
    grad_norm_after_guardrail_max: float = 0.0
    adaptive_memory_suggestions_applied: int = 0
    heartbeat_writes: int = 0
    sustained_control: Dict[str, Any] = field(default_factory=dict)
    early_stop_reason: str = ""
    adaptive_memory_suggestions_observed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil((p / 100.0) * len(values))) - 1))
    return float(values[idx])


def _safe_int(value: Any, default: int, *, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


class DeepSpeedRunner:
    def __init__(self, checkpoint_manager: Any) -> None:
        self.checkpoint_manager = checkpoint_manager

    # ------------------------------------------------------------------
    # Phase 1: resolve effective hyperparameters from applied + directives
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_config(
        steps: int,
        batch_size: int,
        zero_stage: int,
        precision: str,
        gradient_accumulation_steps: int,
        applied: Dict[str, Any],
        directives: Dict[str, Any],
        data_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        effective_batch = _safe_int(applied.get("batch_size", batch_size), batch_size, low=1, high=16)
        effective_zero = _safe_int(applied.get("zero_stage", zero_stage), zero_stage, low=0, high=1)
        effective_precision = str(applied.get("precision", precision)).lower()
        if effective_precision not in {"fp32", "fp16", "bf16"}:
            effective_precision = "fp32"
        effective_grad_accum = _safe_int(applied.get("gradient_accumulation_steps", gradient_accumulation_steps), gradient_accumulation_steps, low=1, high=16)
        effective_steps = max(1, int(steps))
        seq_len = _safe_int(data_cfg.get("sequence_length", 128), 128, low=32, high=512)

        api_lr_multiplier = float(directives.get("lr_multiplier", 1.0)) if directives.get("enabled") else 1.0
        api_lr_multiplier = max(0.20, min(1.0, api_lr_multiplier))
        base_lr = float(_mem_v89_os.environ.get("MEM_V89_BASE_LR_REAL", "1.2e-5")) if data_cfg.get("real_dataset") else (2e-4 if effective_precision == "fp16" else 5e-4)
        optimizer_lr = base_lr * api_lr_multiplier

        train_micro_batch = max(1, effective_batch)
        train_batch = train_micro_batch * max(1, effective_grad_accum)
        ds_config = {
            "train_batch_size": train_batch,
            "train_micro_batch_size_per_gpu": train_micro_batch,
            "gradient_accumulation_steps": max(1, effective_grad_accum),
            "steps_per_print": max(1, min(1000, max(1, effective_steps // 20))),
            "zero_optimization": {"stage": int(effective_zero)},
            "zero_allow_untested_optimizer": True,
            "gradient_clipping": float(directives.get("gradient_clip_norm", 1.0)) if directives.get("enabled") else 1.0,
            "fp16": {"enabled": effective_precision == "fp16", "loss_scale": 0, "initial_scale_power": int(directives.get("loss_scale_initial_power", 8)) if directives.get("enabled") else 8, "loss_scale_window": 500, "hysteresis": 2, "min_loss_scale": 1},
            "bf16": {"enabled": effective_precision == "bf16"},
        }
        return {
            "effective_batch": effective_batch,
            "effective_zero": effective_zero,
            "effective_precision": effective_precision,
            "effective_grad_accum": effective_grad_accum,
            "effective_steps": effective_steps,
            "seq_len": seq_len,
            "train_micro_batch": train_micro_batch,
            "train_batch": train_batch,
            "optimizer_lr": optimizer_lr,
            "api_lr_multiplier": api_lr_multiplier,
            "ds_config": ds_config,
        }

    # ------------------------------------------------------------------
    # Phase 2: build model and optimizer (before DeepSpeed init)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_model_and_optimizer(
        cfg: Dict[str, Any],
        directives: Dict[str, Any],
        data_cfg: Dict[str, Any],
        device: Any,
        model_dtype: Any,
    ) -> tuple:
        """Returns (model, optimizer, batcher_or_None, criterion_or_None, dataset_info)."""
        import torch
        import torch.nn as nn

        real_dataset = bool(data_cfg.get("real_dataset", False))
        seq_len = cfg["seq_len"]
        effective_batch = cfg["effective_batch"]
        chaos_profile = str(data_cfg.get("chaos_profile", "clean"))
        dataset_mix = str(data_cfg.get("dataset_mix", ""))

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
                dataset_mix=dataset_mix,
            )
            prewarm_info = {}
            if bool(data_cfg.get("dataset_prewarm", True)):
                try:
                    prewarm_batches = int(os.environ.get("MEM_DATASET_PREWARM_BATCHES", os.environ.get("MEM_DATASET_PREFETCH_BATCHES", "16")))
                    prewarm_info = batcher.prewarm(prewarm_batches)
                except Exception as exc:
                    prewarm_info = {"prewarm_enabled": True, "prewarm_error": f"{type(exc).__name__}: {str(exc)[:240]}"}
            model_preset = str(data_cfg.get("model_preset", "tiny_decoder"))
            model = build_tiny_causal_lm(batcher.vocab_size, seq_len, model_preset).to(device=device, dtype=model_dtype)
            dataset_info = batcher.info.to_dict()
            dataset_info["prewarm"] = prewarm_info
            criterion = nn.CrossEntropyLoss()
        else:
            batcher = None
            model = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).to(device=device, dtype=model_dtype)
            criterion = None

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer_lr"], eps=1e-6)
        return model, optimizer, batcher, criterion, dataset_info

    # ------------------------------------------------------------------
    # Phase 3: training loop
    # ------------------------------------------------------------------
    @staticmethod
    def _train_loop(
        engine: Any,
        batcher: Any,
        criterion: Any,
        cfg: Dict[str, Any],
        directives: Dict[str, Any],
        data_cfg: Dict[str, Any],
        metrics: "DeepSpeedRunMetrics",
        profiler: Any,
        adaptive_memory: Any,
        chaos_probe: Any,
        write_progress: Any,
        checkpoint_manager: Any = None,
        persistent_checkpoint: bool = False,
    ) -> List[float]:
        """Run the training loop, updating metrics in place. Returns step_times list."""
        import torch
        import math

        device = torch.device("cuda:0")
        effective_steps = cfg["effective_steps"]
        effective_batch = cfg["effective_batch"]
        real_dataset = bool(data_cfg.get("real_dataset", False))

        step_times: List[float] = []
        milestone_set = {1, effective_steps}
        for frac in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
            milestone_set.add(max(1, min(effective_steps, int(effective_steps * frac))))
        for m in metrics.api_checkpoint_milestones:
            milestone_set.add(max(1, min(effective_steps, int(m))))
        heartbeat_interval = max(1, int(data_cfg.get("progress_heartbeat_interval", 250) or 250))
        # P0.1: rotating live checkpoint for lane-switch continuity.
        # The controller may terminate a lane before the full target is reached,
        # so end-of-run checkpointing is not enough. This keeps a latest checkpoint
        # available without filling disk with many large checkpoint directories.
        live_checkpoint_every_steps = max(1, int(os.environ.get("MEM_LIVE_CHECKPOINT_EVERY_STEPS", str(data_cfg.get("live_checkpoint_every_steps", 1000) or 1000))))
        last_live_checkpoint_step = 0
        sustained_best_tokens = 0.0
        sustained_bad_windows = 0
        sustained_min_step = max(1000, int(data_cfg.get("sustained_control_min_step", 5000) or 5000))
        sustained_drop_ratio = float(data_cfg.get("sustained_control_drop_ratio", 0.55) or 0.55)
        sustained_stop_enabled = bool(data_cfg.get("sustained_control_internal_stop", False))

        start = time.perf_counter()
        for step in range(1, effective_steps + 1):
            step_start = time.perf_counter()
            data_start = time.perf_counter()
            if real_dataset:
                input_ids, labels = batcher.next_batch()
                data_fetch_seconds = time.perf_counter() - data_start
                forward_start = time.perf_counter()
                logits = engine(input_ids)
                metrics.forward_output_device = str(logits.device) if not metrics.forward_output_device else metrics.forward_output_device
                loss = criterion(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
                metrics.tokens_processed += int(input_ids.numel())
                forward_loss_seconds = time.perf_counter() - forward_start
            else:
                x = torch.randn(effective_batch, 16, device=device, dtype=next(engine.module.parameters()).dtype)
                y = (0.01 * torch.tanh(x.float()[:, :1])).to(device=device, dtype=torch.float32)
                data_fetch_seconds = time.perf_counter() - data_start
                forward_start = time.perf_counter()
                out = engine(x)
                metrics.forward_output_device = str(out.device) if not metrics.forward_output_device else metrics.forward_output_device
                loss = ((torch.nan_to_num(out.float(), nan=0.0, posinf=1.0, neginf=-1.0) - y) ** 2).mean()
                forward_loss_seconds = time.perf_counter() - forward_start

            reg = torch.zeros((), device=device, dtype=torch.float32)
            for rp in engine.module.parameters():
                reg = reg + rp.float().pow(2).mean()
            loss = loss + (1e-8 * reg)
            metrics.loss_device = str(loss.device) if not metrics.loss_device else metrics.loss_device
            loss_val = float(loss.detach().float().item())
            if metrics.loss_first is None:
                metrics.loss_first = loss_val
            metrics.loss_last = loss_val
            if not math.isfinite(loss_val):
                metrics.loss_sanitized_events += 1
                metrics.numerical_recovery_events += 1
                metrics.first_nonfinite_step = metrics.first_nonfinite_step or step
                metrics.first_nonfinite_kind = metrics.first_nonfinite_kind or "loss"
                metrics.loss_finite = False
                metrics.nan_or_inf_detected = True
                write_progress(metrics, phase="review_stop", step=step, loss=loss_val)
                break

            backward_start = time.perf_counter()
            engine.backward(loss)
            backward_seconds = time.perf_counter() - backward_start
            metrics.backward_count += 1

            guardrail_start = time.perf_counter()
            guardrail_mode = str(data_cfg.get("guardrail_mode", "sampled")).lower()
            guardrail_interval = max(1, int(data_cfg.get("guardrail_sample_interval", 8) or 8))
            gradient_audit_enabled = bool(data_cfg.get("gradient_audit", True))
            force_full_guardrail = (
                guardrail_mode == "full"
                or step in milestone_set
                or step == 1
                or (guardrail_mode == "sampled" and step % guardrail_interval == 0)
            )
            has_grad, still_nonfinite, max_grad_abs = False, False, 0.0
            grad_norm_sq_before, grad_norm_sq_after = 0.0, 0.0
            if force_full_guardrail and guardrail_mode != "minimal":
                metrics.guardrail_full_checks += 1
                for p in engine.module.parameters():
                    grad = getattr(p, "grad", None)
                    if grad is None:
                        continue
                    has_grad = True
                    gd = grad.detach()
                    if gradient_audit_enabled:
                        try:
                            grad_norm_sq_before += float(gd.float().pow(2).sum().item())
                        except Exception:
                            pass
                    if not torch.isfinite(gd).all().item():
                        metrics.grad_sanitized_events += 1
                        metrics.numerical_recovery_events += 1
                        metrics.first_nonfinite_step = metrics.first_nonfinite_step or step
                        metrics.first_nonfinite_kind = metrics.first_nonfinite_kind or "gradient"
                        grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    try:
                        local_max = float(grad.detach().abs().max().float().item())
                    except Exception:
                        local_max = 0.0
                    if local_max > 100.0:
                        grad.data.clamp_(min=-100.0, max=100.0)
                        metrics.grad_clamp_events += 1
                    if not torch.isfinite(grad.detach()).all().item():
                        still_nonfinite = True
                    try:
                        max_grad_abs = max(max_grad_abs, float(grad.detach().abs().max().float().item()))
                        if gradient_audit_enabled:
                            grad_norm_sq_after += float(grad.detach().float().pow(2).sum().item())
                    except Exception:
                        pass
                if gradient_audit_enabled:
                    before = math.sqrt(max(0.0, grad_norm_sq_before))
                    after = math.sqrt(max(0.0, grad_norm_sq_after))
                    metrics.grad_norm_before_clip_last = round(before, 6)
                    metrics.grad_norm_after_guardrail_last = round(after, 6)
                    metrics.grad_norm_before_clip_max = max(metrics.grad_norm_before_clip_max, metrics.grad_norm_before_clip_last)
                    metrics.grad_norm_after_guardrail_max = max(metrics.grad_norm_after_guardrail_max, metrics.grad_norm_after_guardrail_last)
                    if before > float(metrics.gradient_clip_norm or 0.0) > 0.0:
                        metrics.grad_clip_applied_events += 1
            else:
                metrics.guardrail_sampled_skips += 1
            metrics.gradients_observed = metrics.gradients_observed or has_grad
            metrics.max_grad_abs_after_guardrail = max(metrics.max_grad_abs_after_guardrail, max_grad_abs)
            guardrail_seconds = time.perf_counter() - guardrail_start
            if still_nonfinite:
                metrics.grad_finite = False
                metrics.grad_nonfinite_detected = True
                metrics.nan_or_inf_detected = True
                write_progress(metrics, phase="review_stop", step=step, loss=loss_val)
                break

            optimizer_start = time.perf_counter()
            engine.step()
            optimizer_seconds = time.perf_counter() - optimizer_start
            metrics.optimizer_step_count += 1
            if str(_mem_v89_os.environ.get("MEM_V89_LR_SCHEDULE", "1")).lower() not in {"0", "false", "no", "off"}:
                _mem_v89_next_lr = _mem_v89_scheduled_lr(cfg["optimizer_lr"], step)
                _mem_v89_opt = getattr(engine, "optimizer", None) or getattr(engine, "client_optimizer", None)
                _mem_v89_param_groups = getattr(_mem_v89_opt, "param_groups", None)
                if _mem_v89_param_groups:
                    for _mem_v89_pg in _mem_v89_param_groups:
                        _mem_v89_pg["lr"] = _mem_v89_next_lr
                metrics.optimizer_lr = _mem_v89_next_lr
            metrics.forward_count += 1
            metrics.micro_train_steps_completed = step
            total_step_seconds = time.perf_counter() - step_start
            step_times.append(total_step_seconds)
            profiler.add(
                data_fetch=data_fetch_seconds, forward_loss=forward_loss_seconds,
                backward=backward_seconds, optimizer=optimizer_seconds,
                guardrail=guardrail_seconds, total_step=total_step_seconds,
            )

            if step in milestone_set or step % heartbeat_interval == 0:
                live_elapsed = max(time.perf_counter() - start, 1e-9)
                metrics.steps_per_second = round(metrics.micro_train_steps_completed / live_elapsed, 3)
                metrics.tokens_per_second = round(metrics.tokens_processed / live_elapsed, 3)
                if real_dataset and batcher is not None:
                    metrics.dataset = batcher.info.to_dict()

                mem_effect = adaptive_memory.record(
                    step=step, directive=metrics.api_executive_action, lr=metrics.optimizer_lr,
                    gradient_clip_norm=metrics.gradient_clip_norm, loss=loss_val,
                    tokens_processed=metrics.tokens_processed, steps_per_second=metrics.steps_per_second,
                    tokens_per_second=metrics.tokens_per_second,
                )
                metrics.adaptive_memory = adaptive_memory.summary()

                # --- v89: observe adaptive memory suggestions by default; apply only when explicitly enabled. ---
                suggested = adaptive_memory.suggest_action()
                if suggested != "none":
                    metrics.adaptive_memory_suggestions_observed += 1
                if bool(data_cfg.get("adaptive_memory_apply_suggestions", False)):
                    if suggested == "reduce_lr":
                        new_lr = max(metrics.optimizer_lr * 0.97, cfg["optimizer_lr"] * 0.85)
                        if new_lr < metrics.optimizer_lr:
                            for pg in engine.optimizer.param_groups:
                                pg["lr"] = new_lr
                            metrics.optimizer_lr = new_lr
                            metrics.adaptive_memory_suggestions_applied += 1
                    elif suggested == "increase_clip":
                        new_clip = max(0.95, metrics.gradient_clip_norm * 0.98)
                        if new_clip < metrics.gradient_clip_norm:
                            metrics.gradient_clip_norm = new_clip
                            metrics.adaptive_memory_suggestions_applied += 1
                # (suggested == 'none' → no action)

                metrics.profiler = profiler.summary()
                metrics.bottleneck_classification = str(metrics.profiler.get("bottleneck_classification", "unknown"))
                metrics.chaos_environment = chaos_probe.sample()
                metrics.real_chaos_score = chaos_probe.score(metrics.chaos_environment)
                if step % heartbeat_interval == 0 and step not in milestone_set:
                    metrics.heartbeat_writes += 1
                sustained_best_tokens = max(sustained_best_tokens, float(metrics.tokens_per_second or 0.0))
                degraded = bool(
                    step >= sustained_min_step
                    and sustained_best_tokens > 0
                    and float(metrics.tokens_per_second or 0.0) < sustained_best_tokens * sustained_drop_ratio
                    and metrics.bottleneck_classification == "optimizer_or_sync_bound"
                )
                sustained_bad_windows = sustained_bad_windows + 1 if degraded else 0
                metrics.sustained_control = {
                    "enabled": True,
                    "internal_stop_enabled": sustained_stop_enabled,
                    "best_tokens_per_second": round(sustained_best_tokens, 3),
                    "drop_ratio": sustained_drop_ratio,
                    "bad_windows": sustained_bad_windows,
                    "degraded": degraded,
                    "recommended_action": "lane_switch_or_restart" if sustained_bad_windows >= 3 else "observe",
                }

                if (
                    persistent_checkpoint
                    and checkpoint_manager is not None
                    and step > 0
                    and step - last_live_checkpoint_step >= live_checkpoint_every_steps
                    and metrics.loss_finite
                    and not metrics.nan_or_inf_detected
                ):
                    try:
                        live_ckpt = checkpoint_manager.save_live_checkpoint(
                            engine.module, getattr(engine, "optimizer", None),
                            {
                                "version": "v89.0.0",
                                "checkpoint_kind": "live_lane_switch",
                                "workload": metrics.workload,
                                "benchmark_mode": metrics.benchmark_mode,
                                "chaos_profile": metrics.chaos_profile,
                                "real_chaos_score": metrics.real_chaos_score,
                                "dataset": metrics.dataset,
                                "profiler": metrics.profiler,
                                "adaptive_memory": metrics.adaptive_memory,
                                "tokens_processed": metrics.tokens_processed,
                                "micro_train_steps_completed": metrics.micro_train_steps_completed,
                                "batch_size": cfg["effective_batch"],
                                "zero_stage": cfg["effective_zero"],
                                "precision": cfg["effective_precision"],
                                "sequence_length": cfg["seq_len"],
                                "model_preset": str(data_cfg.get("model_preset", "tiny_decoder")),
                                "steps_per_second": metrics.steps_per_second,
                                "tokens_per_second": metrics.tokens_per_second,
                                "api_executive_enabled": metrics.api_executive_enabled,
                                "api_runtime_changes_count": metrics.api_runtime_changes_count,
                                "api_directives_applied": metrics.api_directives_applied,
                            },
                            label="v89",
                        )
                        last_live_checkpoint_step = step
                        metrics.sustained_control["checkpoint_written"] = True
                        metrics.sustained_control["checkpoint_path"] = live_ckpt.get("checkpoint_path", "")
                        metrics.sustained_control["checkpoint_mode"] = live_ckpt.get("checkpoint_mode", "")
                    except Exception as exc:
                        metrics.sustained_control["checkpoint_written"] = False
                        metrics.sustained_control["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
                        metrics.progress_write_failures += 1

                profiler.write()
                metrics.milestones.append({
                    "step": step, "loss": loss_val, "tokens_processed": metrics.tokens_processed,
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 3),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 3),
                    "dataset": metrics.dataset, "optimizer_lr": metrics.optimizer_lr,
                    "api_executive_enabled": metrics.api_executive_enabled,
                    "api_executive_action": metrics.api_executive_action,
                    "adaptive_memory_effect": mem_effect,
                    "adaptive_memory_suggested_action": suggested,
                    "bottleneck_classification": metrics.bottleneck_classification,
                    "chaos_environment": metrics.chaos_environment,
                    "real_chaos_score": metrics.real_chaos_score,
                    "gradient_audit": metrics.gradient_audit,
                })
                write_progress(metrics, phase="milestone" if step in milestone_set else "heartbeat", step=step, loss=loss_val)
                if sustained_stop_enabled and sustained_bad_windows >= 3:
                    metrics.early_stop_reason = "sustained_degradation_detected"
                    write_progress(metrics, phase="review_stop", step=step, loss=loss_val)
                    break

        return step_times

    # ------------------------------------------------------------------
    # Phase 4: teardown — destroy process group and empty CUDA cache
    # ------------------------------------------------------------------
    @staticmethod
    def _teardown(engine: Any, metrics: "DeepSpeedRunMetrics") -> None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            metrics.process_group_destroyed_or_not_initialized = True
        except Exception:
            metrics.process_group_destroyed_or_not_initialized = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        metrics.safe_teardown_completed = True

    # ------------------------------------------------------------------
    # Public entrypoint — orchestrates all phases
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        steps: int,
        batch_size: int,
        zero_stage: int,
        precision: str,
        persistent_checkpoint: bool,
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

        from runtime.profiler import RuntimeProfiler
        from runtime.adaptive_memory import AdaptiveRuntimeMemory
        from runtime.real_chaos import RealChaosProbe

        profiler = RuntimeProfiler(evidence_path)
        adaptive_memory = AdaptiveRuntimeMemory(evidence_path)
        chaos_profile = str(data_cfg.get("chaos_profile", "clean"))
        chaos_probe = RealChaosProbe(chaos_profile)

        # --- Fix #5: write_progress tracks I/O failures instead of silently dropping them ---
        def _write_progress(metrics_obj: "DeepSpeedRunMetrics", *, phase: str, step: int = 0, loss: float | None = None) -> None:
            if not progress_path:
                return
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": "v89.0.0",
                    "phase": phase,
                    "step": int(step or metrics_obj.micro_train_steps_completed),
                    "target_steps": int(metrics_obj.micro_train_step_target),
                    "loss": loss,
                    "tokens_processed": int(metrics_obj.tokens_processed),
                    "tokens_per_second": float(metrics_obj.tokens_per_second or 0.0),
                    "steps_per_second": float(metrics_obj.steps_per_second or 0.0),
                    "gpu_utilization_hint": float(metrics_obj.chaos_environment.get("gpu_utilization_percent", 0.0) or 0.0) if isinstance(metrics_obj.chaos_environment, dict) else 0.0,
                    "heartbeat_writes": int(metrics_obj.heartbeat_writes),
                    "bottleneck": metrics_obj.bottleneck_classification,
                    "sustained_control": metrics_obj.sustained_control,
                    "early_stop_reason": metrics_obj.early_stop_reason,
                    "api_executive_enabled": bool(metrics_obj.api_executive_enabled),
                    "api_runtime_changes_count": int(metrics_obj.api_runtime_changes_count),
                    "api_directives_applied": int(metrics_obj.api_directives_applied),
                    "profiler": metrics_obj.profiler,
                    "adaptive_memory": metrics_obj.adaptive_memory,
                    "bottleneck_classification": metrics_obj.bottleneck_classification,
                    "chaos_profile": metrics_obj.chaos_profile,
                    "chaos_environment": metrics_obj.chaos_environment,
                    "real_chaos_score": metrics_obj.real_chaos_score,
                    "dataset": metrics_obj.dataset,
                    "nan_or_inf_detected": bool(metrics_obj.nan_or_inf_detected),
                    "grad_nonfinite_detected": bool(metrics_obj.grad_nonfinite_detected),
                    "safe_teardown_completed": bool(metrics_obj.safe_teardown_completed),
                    "progress_write_failures": int(metrics_obj.progress_write_failures),
                    "gradient_audit": metrics_obj.gradient_audit,
                    "guardrail_mode": metrics_obj.guardrail_mode,
                    "guardrail_sample_interval": metrics_obj.guardrail_sample_interval,
                    "ts": round(_mem_v89_time.time(), 3),
                }
                tmp_progress_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
                tmp_progress_path.write_text(_mem_v89_json.dumps(payload, indent=2, default=str), encoding="utf-8")
                os.replace(tmp_progress_path, progress_path)
                if milestones_path and phase in {"milestone", "final", "review_stop"}:
                    with milestones_path.open("a", encoding="utf-8") as fh:
                        fh.write(_mem_v89_json.dumps(payload, default=str) + "\n")
            except Exception as exc:
                metrics_obj.progress_write_failures += 1
                try:
                    print("WRITE_PROGRESS_FAIL:", repr(exc), flush=True)
                except Exception:
                    pass

        # Phase 1: resolve config
        cfg = self._resolve_config(
            steps, batch_size, zero_stage, precision, gradient_accumulation_steps,
            applied, directives, data_cfg,
        )

        os.environ.setdefault("DS_BUILD_OPS", "0")
        os.environ.setdefault("DS_BUILD_FUSED_ADAM", "0")
        os.environ.setdefault("DS_BUILD_CPU_ADAM", "0")
        os.environ.setdefault("DS_BUILD_AIO", "0")
        os.environ.setdefault("DS_BUILD_UTILS", "0")

        # WSL2 single-GPU fix: NCCL cannot enumerate CUDA devices under WSL2 because
        # it uses a different libcuda path than PyTorch.  For a single-GPU run there is
        # no need for NCCL at all — switch the distributed backend to gloo and set the
        # required process-group env vars so DeepSpeed can initialize without torchrun.
        import platform as _platform
        _is_wsl = "microsoft" in _platform.release().lower() or os.path.exists("/usr/lib/wsl/lib/libcuda.so.1")
        if _is_wsl:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            # Tell DeepSpeed to use gloo instead of nccl for the single-process group
            os.environ["NCCL_SOCKET_IFNAME"] = "lo"
            os.environ["DS_ACCELERATOR"] = "cuda"
            # Inject comm backend override into ds_config before deepspeed.initialize
            cfg["ds_config"]["communication_data_type"] = "fp16"
            cfg["ds_config"]["comms_logger"] = {"enabled": False}
            # Force gloo backend via env — DeepSpeed respects TORCH_DISTRIBUTED_DEFAULT_BACKEND
            os.environ["TORCH_DISTRIBUTED_DEFAULT_BACKEND"] = "gloo"

        import torch
        import deepspeed

        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        device = torch.device("cuda:0")
        if cfg["effective_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            cfg["effective_precision"] = "fp16"
            cfg["ds_config"]["bf16"]["enabled"] = False
            cfg["ds_config"]["fp16"]["enabled"] = True
        model_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["effective_precision"], torch.float32)

        # Phase 2: build model, optimizer, dataset
        model, optimizer, batcher, criterion, dataset_info = self._build_model_and_optimizer(
            cfg, directives, data_cfg, device, model_dtype,
        )

        # P0 checkpoint/resume continuity fix.
        # First session is allowed to start from scratch when load_checkpoint is empty.
        # Later lane/restart sessions may pass a checkpoint path selected by the controller.
        # If a checkpoint path is explicitly provided, model/optimizer state must load
        # or the session fails loudly instead of silently restarting from scratch.
        pending_optimizer_state: Dict[str, Any] = {}
        checkpoint_resume: Dict[str, Any] = {
            "requested": bool(str(load_checkpoint or "").strip()),
            "path": str(load_checkpoint or "").strip(),
            "success": False,
            "loaded_model": False,
            "loaded_optimizer": False,
            "error": "",
        }
        if checkpoint_resume["requested"]:
            try:
                resume_path = checkpoint_resume["path"]

                try:
                    payload = self.checkpoint_manager.load_torch_checkpoint(
                        resume_path,
                        map_location=device,
                    )
                except Exception as resume_exc:
                    # Resume robusto:
                    # Se o controller entregar um slot antigo/corrompido por mtime,
                    # o runner consulta o latest.txt validado pelo CheckpointManager.
                    # Se não houver latest válido, mantém a falha original.
                    checkpoint_label = (
                        checkpoint_resume.get("label")
                        or checkpoint_resume.get("checkpoint_label")
                        or checkpoint_resume.get("metadata", {}).get("checkpoint_label")
                        or "v89"
                    )

                    latest_path = self.checkpoint_manager.latest_checkpoint_path(checkpoint_label)

                    if latest_path and str(latest_path) != str(resume_path):
                        try:
                            payload = self.checkpoint_manager.load_torch_checkpoint(
                                latest_path,
                                map_location=device,
                            )
                            checkpoint_resume["original_failed_path"] = str(resume_path)
                            checkpoint_resume["path"] = str(latest_path)
                            checkpoint_resume["resume_recovered_from_invalid_path"] = True
                            checkpoint_resume["resume_recovery_reason"] = (
                                f"{type(resume_exc).__name__}: {resume_exc}"
                            )
                        except Exception:
                            raise resume_exc
                    else:
                        raise resume_exc
                model_state = payload.get("model_state_dict") or {}
                if not model_state:
                    raise RuntimeError("checkpoint_missing_model_state_dict")

                # Robust model resume: torch.compile can save keys under
                # `_orig_mod.` and some wrappers can save under `module.`.
                # Never accept a silent strict=False no-op: require that a
                # meaningful fraction of model tensors actually load.
                def _strip_prefix_from_state_dict(state, prefix):
                    if not isinstance(state, dict):
                        return state
                    if not any(str(k).startswith(prefix) for k in state.keys()):
                        return state
                    return {
                        (str(k)[len(prefix):] if str(k).startswith(prefix) else k): v
                        for k, v in state.items()
                    }

                def _normalize_model_state_variants(state):
                    variants = []
                    seen = set()
                    candidates = [state]
                    candidates.append(_strip_prefix_from_state_dict(state, "_orig_mod."))
                    candidates.append(_strip_prefix_from_state_dict(state, "module."))
                    candidates.append(_strip_prefix_from_state_dict(_strip_prefix_from_state_dict(state, "module."), "_orig_mod."))
                    candidates.append(_strip_prefix_from_state_dict(_strip_prefix_from_state_dict(state, "_orig_mod."), "module."))
                    for cand in candidates:
                        if not isinstance(cand, dict):
                            continue
                        sig = tuple(list(cand.keys())[:20])
                        if sig in seen:
                            continue
                        seen.add(sig)
                        variants.append(cand)
                    return variants

                expected_keys = set(model.state_dict().keys())
                best_result = None
                best_state = None
                best_loaded = -1
                load_attempts = []
                for candidate_state in _normalize_model_state_variants(model_state):
                    candidate_keys = set(candidate_state.keys())
                    loaded_keys = len(expected_keys.intersection(candidate_keys))
                    try:
                        result = model.load_state_dict(candidate_state, strict=False)
                        load_attempts.append({
                            "loaded_key_count": loaded_keys,
                            "missing_count": len(getattr(result, "missing_keys", []) or []),
                            "unexpected_count": len(getattr(result, "unexpected_keys", []) or []),
                        })
                        if loaded_keys > best_loaded:
                            best_loaded = loaded_keys
                            best_result = result
                            best_state = candidate_state
                    except Exception as load_exc:
                        load_attempts.append({"error": f"{type(load_exc).__name__}: {load_exc}"})

                if best_state is None or best_loaded <= 0:
                    checkpoint_resume["model_load_attempts"] = load_attempts
                    raise RuntimeError("checkpoint_model_state_dict_incompatible:no_matching_keys")

                # Re-load the best variant. This is intentionally strict-ish: allow
                # non-critical wrapper differences, but reject near-empty loads.
                result = model.load_state_dict(best_state, strict=False)
                loaded_ratio = best_loaded / max(1, len(expected_keys))
                checkpoint_resume["loaded_model_key_count"] = best_loaded
                checkpoint_resume["expected_model_key_count"] = len(expected_keys)
                checkpoint_resume["model_loaded_ratio"] = round(loaded_ratio, 4)
                checkpoint_resume["model_missing_key_count"] = len(getattr(result, "missing_keys", []) or [])
                checkpoint_resume["model_unexpected_key_count"] = len(getattr(result, "unexpected_keys", []) or [])
                checkpoint_resume["model_load_attempts"] = load_attempts
                if loaded_ratio < 0.90:
                    raise RuntimeError(f"checkpoint_model_state_dict_incomplete:loaded_ratio={loaded_ratio:.4f}")
                checkpoint_resume["loaded_model"] = True

                opt_state = payload.get("optimizer_state_dict") or {}
                if opt_state:
                    pending_optimizer_state = opt_state

                checkpoint_resume["metadata"] = payload.get("metadata", {})
                checkpoint_resume["success"] = True
            except Exception as exc:
                checkpoint_resume["error"] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(
                    f"checkpoint_resume_failed:{checkpoint_resume['path']}:{checkpoint_resume['error']}"
                )

        # v89 experimental torch.compile support. It is opt-in because DeepSpeed +
        # WSL + dynamic data can vary by install. Any failure automatically falls
        # back to the eager model so sustained runs are not blocked.
        torch_compile_status: Dict[str, Any] = {"requested": False, "enabled": False, "fallback": False, "reason": "disabled", "attempted_modes": []}
        if os.environ.get("MEM_TORCH_COMPILE_EXPERIMENTAL", "0") == "1":
            torch_compile_status["requested"] = True
            requested_mode = os.environ.get("MEM_TORCH_COMPILE_MODE", "max-autotune")
            modes = [requested_mode]
            if requested_mode != "reduce-overhead":
                modes.append("reduce-overhead")
            last_error = ""
            for mode in modes:
                torch_compile_status["attempted_modes"].append(mode)
                try:
                    model = torch.compile(model, mode=mode, fullgraph=False)  # type: ignore[attr-defined]
                    torch_compile_status.update({"enabled": True, "fallback": mode != requested_mode, "reason": f"compiled:{mode}", "mode": mode})
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if not torch_compile_status["enabled"]:
                torch_compile_status.update({"enabled": False, "fallback": True, "reason": last_error or "compile_failed"})

        checkpoint: Dict[str, Any] = {"checkpoint_written": False, "checkpoint_mode": "not_attempted"}
        metrics = DeepSpeedRunMetrics(
            micro_train_step_target=cfg["effective_steps"],
            effective_batch_size=cfg["effective_batch"], effective_zero_stage=cfg["effective_zero"],
            effective_precision=cfg["effective_precision"], effective_gradient_accumulation_steps=cfg["effective_grad_accum"],
            train_micro_batch_size_per_gpu=cfg["train_micro_batch"], train_batch_size=cfg["train_batch"],
            sequence_length=cfg["seq_len"], model_preset=str(data_cfg.get("model_preset", "tiny_decoder")),
            applied_hyperparams_consumed=bool(applied), optimizer_lr=cfg["optimizer_lr"],
            gradient_clip_norm=float(directives.get("gradient_clip_norm", 1.0)) if directives.get("enabled") else 1.0,
            api_executive_enabled=bool(directives.get("enabled")), api_executive_action=str(directives.get("action", "")),
            api_lr_multiplier=cfg["api_lr_multiplier"],
            api_numerical_recovery_budget=int(directives.get("numerical_recovery_budget", 1000)),
            api_checkpoint_milestones=[int(x) for x in directives.get("checkpoint_milestones", [])],
            api_event_triggers=list(directives.get("event_triggers", [])),
            api_runtime_changes_count=int(directives.get("api_runtime_changes_count", 0)),
            api_directives_applied=int(directives.get("api_directives_applied", 0)),
            api_directives_rejected_by_local_policy=int(directives.get("api_directives_rejected_by_local_policy", 0)),
            dataset=dataset_info,
            workload="real_dataset_causal_lm" if data_cfg.get("real_dataset") else "real_dataset_required",
            benchmark_mode=str(data_cfg.get("benchmark_mode", "mem_real_chaos")),
            chaos_profile=chaos_profile,
            chaos_environment=chaos_probe.sample(),
            guardrail_mode=str(data_cfg.get("guardrail_mode", "sampled")),
            guardrail_sample_interval=max(1, int(data_cfg.get("guardrail_sample_interval", 8) or 8)),
        )
        metrics.real_chaos_score = chaos_probe.score(metrics.chaos_environment)
        metrics.sustained_control["torch_compile"] = torch_compile_status
        metrics.sustained_control["checkpoint_resume"] = checkpoint_resume
        _write_progress(metrics, phase="initialized", step=0, loss=None)

        engine = None
        try:
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=cfg["ds_config"], model_parameters=model.parameters(),
            )
            engine = _mem_v89_patch_ds_load_checkpoint_for_cross_zero_resume(engine)
            metrics.engine_initialized = True
            metrics.engine_class = engine.__class__.__name__
            if pending_optimizer_state:
                try:
                    target_optimizer = getattr(engine, "optimizer", None) or optimizer
                    if target_optimizer is not None:
                        optimizer_load_attempts = []

                        # DeepSpeed ZeRO stage 1/2 load_state_dict expects a per-DP-rank
                        # list/dict-like checkpoint in some versions, while our live rotating
                        # checkpoint stores the single-rank optimizer state as a plain dict.
                        # First try the state as saved; if DeepSpeed raises KeyError: 0, retry
                        # with [state] for rank-0 continuity. This keeps optimizer resume strict
                        # without entering an infinite restart loop.
                        candidate_states = [pending_optimizer_state]
                        if isinstance(pending_optimizer_state, dict):
                            candidate_states.append([pending_optimizer_state])

                        last_exc = None
                        for idx, candidate_state in enumerate(candidate_states):
                            try:
                                target_optimizer.load_state_dict(candidate_state)
                                checkpoint_resume["loaded_optimizer"] = True
                                checkpoint_resume["optimizer_state_format"] = "single_rank_list_wrapped" if idx == 1 else "native"
                                last_exc = None
                                break

                            except AssertionError as opt_exc:
                                msg = str(opt_exc)

                                if "Empty ds_version in checkpoint" in msg:
                                    # MEM v89/v3 recovery:
                                    # safe_seq256 usa zero_stage=1 e pode receber checkpoint salvo
                                    # por lanes zero_stage=0. Nesse caso, o modelo já foi carregado,
                                    # mas o optimizer state pode ser incompatível no DeepSpeed.
                                    # Não abortar a lane: mantém pesos e reinicializa optimizer.
                                    loaded_optimizer = False
                                    checkpoint_resume["loaded_optimizer"] = False
                                    checkpoint_resume["optimizer_state_format"] = "reinitialized_empty_ds_version"
                                    checkpoint_resume["optimizer_reinitialized"] = True
                                    checkpoint_resume["optimizer_reinit_reason"] = "empty_ds_version_cross_zero_stage_resume"
                                    checkpoint_resume["optimizer_load_attempts"] = optimizer_load_attempts + [f"AssertionError: {msg}"]

                                    try:
                                        import json, time
                                        from pathlib import Path as _Path

                                        ev_dir = _Path("evidence_packets")
                                        ev_dir.mkdir(parents=True, exist_ok=True)

                                        with (ev_dir / "v89_optimizer_reinit_events.jsonl").open("a", encoding="utf-8") as f:
                                            f.write(json.dumps({
                                                "event": "optimizer_state_reinitialized",
                                                "reason": "empty_ds_version_cross_zero_stage_resume",
                                                "message": msg,
                                                "candidate_format": "single_rank_list_wrapped" if idx == 1 else "native",
                                                "loaded_model": bool(checkpoint_resume.get("loaded_model")),
                                                "loaded_optimizer": False,
                                                "ts": time.time(),
                                            }, ensure_ascii=False) + "\\n")
                                    except Exception:
                                        pass

                                    print(
                                        "[MEM_V89_RECOVERY] Empty ds_version while loading optimizer state; "
                                        "keeping loaded model weights and reinitializing optimizer.",
                                        flush=True,
                                    )

                                    last_exc = None
                                    break

                                last_exc = opt_exc
                                optimizer_load_attempts.append(f"{type(opt_exc).__name__}: {opt_exc}")

                            except Exception as opt_exc:
                                last_exc = opt_exc
                                optimizer_load_attempts.append(f"{type(opt_exc).__name__}: {opt_exc}")

                        if last_exc is not None:
                            checkpoint_resume["optimizer_load_attempts"] = optimizer_load_attempts
                            raise last_exc
                except Exception as exc:
                    checkpoint_resume["optimizer_load_error"] = f"{type(exc).__name__}: {exc}"
                    raise RuntimeError(
                        f"checkpoint_resume_failed:{checkpoint_resume['path']}:optimizer:{checkpoint_resume['optimizer_load_error']}"
                    )
            metrics.sustained_control["checkpoint_resume"] = checkpoint_resume
            _write_progress(metrics, phase="engine_initialized", step=0, loss=None)

            first_param = next(engine.module.parameters()).detach().clone().float()
            metrics.param_dtype = str(next(engine.module.parameters()).dtype)

            # Phase 3: training loop
            step_times = self._train_loop(
                engine=engine, batcher=batcher, criterion=criterion, cfg=cfg,
                directives=directives, data_cfg=data_cfg, metrics=metrics,
                profiler=profiler, adaptive_memory=adaptive_memory,
                chaos_probe=chaos_probe, write_progress=_write_progress,
                checkpoint_manager=self.checkpoint_manager, persistent_checkpoint=persistent_checkpoint,
            )

            # Post-loop final metrics
            total = sum(step_times)
            metrics.total_seconds = round(total, 6)
            metrics.avg_step_seconds = round(total / max(1, metrics.micro_train_steps_completed), 9)
            metrics.steps_per_second = round(metrics.micro_train_steps_completed / max(total, 1e-9), 3)
            metrics.tokens_per_second = round(metrics.tokens_processed / max(total, 1e-9), 3)
            metrics.step_seconds_p95 = round(_percentile(step_times, 95), 9)
            metrics.step_seconds_p99 = round(_percentile(step_times, 99), 9)
            metrics.parameter_delta_abs_sum_positive = bool(
                (next(engine.module.parameters()).detach().float() - first_param).abs().sum().float().item() > 0
            )
            metrics.cuda_allocated_mb = round(torch.cuda.memory_allocated(device) / 1024 / 1024, 3)
            metrics.cuda_reserved_mb = round(torch.cuda.memory_reserved(device) / 1024 / 1024, 3)
            metrics.cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3)
            metrics.cuda_max_reserved_mb = round(torch.cuda.max_memory_reserved(device) / 1024 / 1024, 3)
            if data_cfg.get("real_dataset") and batcher is not None:
                metrics.dataset = batcher.info.to_dict()
            metrics.profiler = profiler.summary()
            metrics.bottleneck_classification = str(metrics.profiler.get("bottleneck_classification", "unknown"))
            metrics.chaos_environment = chaos_probe.sample()
            metrics.real_chaos_score = chaos_probe.score(metrics.chaos_environment)
            metrics.adaptive_memory = adaptive_memory.summary()
            metrics.sustained_control["torch_compile"] = torch_compile_status
            metrics.progress_write_failures += adaptive_memory._write_failures
            metrics.gradient_audit = {
                "enabled": bool(data_cfg.get("gradient_audit", True)),
                "guardrail_mode": metrics.guardrail_mode,
                "guardrail_sample_interval": metrics.guardrail_sample_interval,
                "guardrail_full_checks": metrics.guardrail_full_checks,
                "guardrail_sampled_skips": metrics.guardrail_sampled_skips,
                "grad_clip_applied_events": metrics.grad_clip_applied_events,
                "grad_clamp_events": metrics.grad_clamp_events,
                "grad_sanitized_events": metrics.grad_sanitized_events,
                "numerical_recovery_events": metrics.numerical_recovery_events,
                "first_nonfinite_step": metrics.first_nonfinite_step,
                "first_nonfinite_kind": metrics.first_nonfinite_kind,
                "grad_norm_before_clip_last": metrics.grad_norm_before_clip_last,
                "grad_norm_after_guardrail_last": metrics.grad_norm_after_guardrail_last,
                "grad_norm_before_clip_max": metrics.grad_norm_before_clip_max,
                "grad_norm_after_guardrail_max": metrics.grad_norm_after_guardrail_max,
                "adaptive_memory_suggestions_observed": metrics.adaptive_memory_suggestions_observed,
                "adaptive_memory_suggestions_applied": metrics.adaptive_memory_suggestions_applied,
            }
            profiler.write()

            metrics.execution_performed = bool(
                metrics.micro_train_steps_completed == cfg["effective_steps"]
                and metrics.loss_finite
                and not metrics.nan_or_inf_detected
                and metrics.parameter_delta_abs_sum_positive
            )
            if persistent_checkpoint and metrics.execution_performed:
                checkpoint = self.checkpoint_manager.save_post_train(
                    engine.module, optimizer,
                    {
                        "version": "v89.0.0", "workload": metrics.workload,
                        "benchmark_mode": metrics.benchmark_mode, "chaos_profile": metrics.chaos_profile,
                        "real_chaos_score": metrics.real_chaos_score, "dataset": metrics.dataset,
                        "profiler": metrics.profiler, "adaptive_memory": metrics.adaptive_memory,
                        "tokens_processed": metrics.tokens_processed,
                        "micro_train_steps_completed": metrics.micro_train_steps_completed,
                        "batch_size": cfg["effective_batch"], "zero_stage": cfg["effective_zero"],
                        "precision": cfg["effective_precision"], "sequence_length": cfg["seq_len"],
                        "model_preset": str(data_cfg.get("model_preset", "tiny_decoder")),
                        "steps_per_second": metrics.steps_per_second, "tokens_per_second": metrics.tokens_per_second,
                        "api_executive_enabled": metrics.api_executive_enabled,
                        "api_runtime_changes_count": metrics.api_runtime_changes_count,
                        "api_directives_applied": metrics.api_directives_applied,
                        "gradient_audit": metrics.gradient_audit,
                    },
                    label="v89",
                )
        finally:
            # Phase 4: teardown
            self._teardown(engine, metrics)

        if metrics.execution_performed:
            metrics.execution_performed = bool(
                metrics.safe_teardown_completed
                and metrics.process_group_destroyed_or_not_initialized
                and (not persistent_checkpoint or checkpoint.get("checkpoint_written") is True)
            )
        _write_progress(metrics, phase="final", step=metrics.micro_train_steps_completed, loss=metrics.loss_last)
        return metrics.to_dict(), checkpoint
