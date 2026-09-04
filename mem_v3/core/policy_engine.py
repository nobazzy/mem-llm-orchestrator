from __future__ import annotations
import os as _mem_v89_policy_os

from typing import Any, Dict, List

from domain.models import CONFIRMATION_TOKEN, CandidatePlan, EnvironmentReport, ExecutiveDirective, PolicyDecision, RuntimeRequest


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


class LocalPolicyEngine:
    def evaluate(self, req: RuntimeRequest, plan: CandidatePlan, env: EnvironmentReport, directive: ExecutiveDirective | None = None) -> PolicyDecision:
        if req.confirmation != CONFIRMATION_TOKEN:
            return PolicyDecision(False, "no_go", "invalid_confirmation_token", 0, False)
        if not req.operator:
            return PolicyDecision(False, "no_go", "operator_mode_required", 0, False)
        if not req.real_micro_train:
            return PolicyDecision(False, "no_go", "real_micro_train_not_requested", 0, False)
        if not plan.should_run_micro_train:
            return PolicyDecision(False, "no_go", "candidate_declined_micro_train", 0, False)
        is_native = req.benchmark_mode in {"mem_native_pytorch", "torch_native"}
        if is_native:
            if not env.torch.get("import_ok"):
                return PolicyDecision(False, "no_go", "torch_unavailable", 0, False)
        else:
            if not env.torch.get("cuda_available"):
                return PolicyDecision(False, "no_go", "cuda_unavailable", 0, False)
            if not env.deepspeed.get("import_ok"):
                return PolicyDecision(False, "no_go", "deepspeed_not_importable", 0, False)
        if req.max_steps > 10_000_000:
            return PolicyDecision(False, "no_go", "requested_steps_exceed_hard_cap", 0, False)
        if not req.real_dataset:
            return PolicyDecision(False, "no_go", "real_dataset_required", 0, False)

        batch = int(req.batch_size)
        zero = int(req.zero_stage)
        precision = str(req.precision).lower()
        grad_accum = int(req.gradient_accumulation_steps)
        lane = "v89_pytorch_native_mem_lane" if is_native else "v89_real_chaos_mem_lane"
        reason = (
            "v89 LocalPolicy allowed: portable native PyTorch execution with real dataset streaming"
            if is_native
            else "v89 LocalPolicy allowed: real chaos benchmark with real dataset mix, profiler, checkpoint pressure and guarded adaptation"
        )
        rejections: List[str] = []
        real_limited_apply = False
        if req.real_limited_apply:
            real_limited_apply = True
            batch = max(1, min(max(batch, 4), 8))
            zero = max(0, min(zero, 1))
            if precision == "fp32":
                precision = "fp16"
            grad_accum = max(1, min(grad_accum, 4))

        executive_runtime_directives: Dict[str, Any] = {
            "enabled": False,
            "source": "disabled",
            "authority_level": "none",
            "action": "observe_only",
            "lr_multiplier": 1.0,
            "gradient_clip_norm": 1.0,
            "loss_scale_initial_power": 8,
            "numerical_recovery_budget": 1000,
            "checkpoint_milestones": [m for m in range(50000, int(req.max_steps) + 1, 50000)],
            "event_triggers": [],
            "dataset_directive": "none",
            "validated_by_local_policy": False,
            "rejected_fields": [],
            "api_runtime_changes_count": 0,
            "api_directives_generated": 0,
            "api_directives_applied": 0,
            "api_directives_rejected_by_local_policy": 0,
        }
        api_exec = False
        if req.api_executive_moderate:
            d = directive or ExecutiveDirective.disabled("missing_directive")
            if not d.enabled:
                rejections.append("api_executive_requested_but_directive_disabled")
            elif d.authority_level not in {"moderate", "moderate_local_fallback"}:
                rejections.append("api_executive_authority_not_moderate")
            else:
                milestones = []
                for raw in d.checkpoint_milestones:
                    m = _clamp_int(raw, 0, 1, req.max_steps)
                    if m not in milestones:
                        milestones.append(m)
                lr_multiplier = _clamp_float(d.lr_multiplier, 0.90, 0.85, 1.0)
                _env_clip = (
                    _mem_v89_policy_os.environ.get("MEM_V89_GRADIENT_CLIP_NORM")
                    or _mem_v89_policy_os.environ.get("MEM_GRAD_CLIP")
                    or _mem_v89_policy_os.environ.get("GRAD_CLIP")
                )
                gradient_clip_norm = _clamp_float(float(_env_clip) if _env_clip else d.gradient_clip_norm, 0.5, 0.25, 1.25)
                loss_scale_initial_power = _clamp_int(d.loss_scale_initial_power, 8, 6, 10)
                numerical_recovery_budget = _clamp_int(d.numerical_recovery_budget, 10000, 100, 30000)
                checkpoint_milestones = milestones or [100_000, 1_000_000, 5_000_000, req.max_steps]
                event_triggers = [str(x)[:80] for x in d.event_triggers[:12]]
                api_runtime_changes_count = sum([
                    lr_multiplier != 1.0,
                    gradient_clip_norm != 1.0,
                    loss_scale_initial_power != 8,
                    numerical_recovery_budget != 1000,
                    bool(checkpoint_milestones),
                    bool(event_triggers),
                    bool(str(d.dataset_directive).strip()),
                ])
                executive_runtime_directives = {
                    "enabled": True,
                    "source": d.source,
                    "authority_level": "moderate_validated",
                    "action": str(d.action)[:120],
                    "lr_multiplier": lr_multiplier,
                    "gradient_clip_norm": gradient_clip_norm,
                    "loss_scale_initial_power": loss_scale_initial_power,
                    "numerical_recovery_budget": numerical_recovery_budget,
                    "checkpoint_milestones": checkpoint_milestones,
                    "event_triggers": event_triggers,
                    "dataset_directive": str(d.dataset_directive)[:120],
                    "validated_by_local_policy": True,
                    "rejected_fields": [],
                    "api_runtime_changes_count": int(api_runtime_changes_count),
                    "api_directives_generated": 1,
                    "api_directives_applied": 1,
                    "api_directives_rejected_by_local_policy": 0,
                }
                api_exec = True

        return PolicyDecision(
            True,
            lane,
            reason,
            10_000_000,
            bool(req.persistent_checkpoint),
            llm_apply_allowed=bool(req.api_executive_moderate),
            api_executive_authority="moderate_validated" if api_exec else "none",
            applied_hyperparams={"batch_size": batch, "zero_stage": zero, "precision": precision, "gradient_accumulation_steps": grad_accum},
            executive_runtime_directives=executive_runtime_directives,
            real_limited_apply_performed=real_limited_apply,
            api_executive_directive_performed=api_exec,
            policy_rejections=rejections,
        )
