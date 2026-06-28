from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict

from core.policy_engine import LocalPolicyEngine
from core.safety_guard import EnvironmentDoctor
from core.state_manager import StateManager
from domain.models import CONFIRMATION_TOKEN, VERSION, ExecutiveDirective, RunResult, RuntimeRequest
from infrastructure.llm_client import LLMPlanner
from runtime.checkpoint_manager import CheckpointManager
from runtime.deepspeed_runner import DeepSpeedRunner


class OrchestratorContext:
    def __init__(self, base_dir: str = ".") -> None:
        self.state = StateManager(base_dir)
        self.doctor = EnvironmentDoctor()
        self.policy = LocalPolicyEngine()
        self.llm = LLMPlanner()
        self.checkpoints = CheckpointManager(self.state.checkpoints_root)
        self.runner = DeepSpeedRunner(self.checkpoints)


class MemOrchestrator:
    def __init__(self, context: OrchestratorContext | None = None) -> None:
        self.context = context or OrchestratorContext()

    def run_deepspeed(self, req: RuntimeRequest) -> Dict[str, Any]:
        req = req.normalized()
        evidence_dir = self.context.state.new_evidence_dir("v89")
        env = self.context.doctor.inspect()
        plan = self.context.llm.plan(req)
        directive = ExecutiveDirective.disabled("api_executive_disabled")
        if req.api_executive_moderate:
            directive = self.context.llm.executive_directive(req, plan, env.to_dict())
        decision = self.context.policy.evaluate(req, plan, env, directive)
        api_telemetry = self.context.llm.telemetry_summary()
        api_telemetry["api_directives_generated"] = int(decision.executive_runtime_directives.get("api_directives_generated", 0))
        api_telemetry["api_directives_applied"] = int(decision.executive_runtime_directives.get("api_directives_applied", 0))
        api_telemetry["api_directives_rejected_by_local_policy"] = int(decision.executive_runtime_directives.get("api_directives_rejected_by_local_policy", 0))
        api_telemetry["api_runtime_changes_count"] = int(decision.executive_runtime_directives.get("api_runtime_changes_count", 0))
        api_telemetry["local_policy_validated"] = bool(decision.executive_runtime_directives.get("validated_by_local_policy", False))

        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.context.state.reports_dir.mkdir(exist_ok=True)
        (evidence_dir / "api_telemetry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in api_telemetry.get("api_events", [])), encoding="utf-8")
        (evidence_dir / "api_usage_summary.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
        (self.context.state.reports_dir / "api_usage_summary_latest.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")

        runtime: Dict[str, Any] = {}
        checkpoint: Dict[str, Any] = {"checkpoint_written": False, "checkpoint_mode": "not_attempted"}
        error = None
        performed = False
        status = "NO_GO"
        effective_hyperparams = dict(decision.applied_hyperparams or {})
        executive_directives = dict(decision.executive_runtime_directives or {})

        dataset_settings = {
            "real_dataset": req.real_dataset,
            "dataset_name": req.dataset_name,
            "dataset_config": req.dataset_config,
            "dataset_split": req.dataset_split,
            "dataset_streaming": req.dataset_streaming,
            "dataset_fallback_name": req.dataset_fallback_name,
            "tokenizer_name": req.tokenizer_name,
            "sequence_length": req.sequence_length,
            "model_preset": req.model_preset,
            "benchmark_mode": req.benchmark_mode,
            "chaos_profile": req.chaos_profile,
            "dataset_mix": req.dataset_mix,
            "guardrail_mode": req.guardrail_mode,
            "guardrail_sample_interval": req.guardrail_sample_interval,
            "gradient_audit": req.gradient_audit,
            "adaptive_memory_apply_suggestions": req.adaptive_memory_apply_suggestions,
            "evidence_dir": str(evidence_dir),
            "checkpoint_label": os.environ.get("MEM_CHECKPOINT_LABEL", "v89"),
        }

        if decision.allowed:
            try:
                runtime, checkpoint = self.context.runner.run(
                    steps=req.max_steps,
                    batch_size=req.batch_size,
                    zero_stage=req.zero_stage,
                    precision=req.precision,
                    persistent_checkpoint=req.persistent_checkpoint,
                    load_checkpoint=req.load_checkpoint,
                    gradient_accumulation_steps=req.gradient_accumulation_steps,
                    applied_hyperparams=effective_hyperparams,
                    executive_directives=executive_directives,
                    dataset_settings=dataset_settings,
                )
                performed = bool(runtime.get("execution_performed"))
                checkpoint_ok = (not req.persistent_checkpoint) or bool(checkpoint.get("checkpoint_written"))
                teardown_ok = bool(runtime.get("safe_teardown_completed")) and bool(runtime.get("process_group_destroyed_or_not_initialized"))
                api_exec_ok = (not req.api_executive_moderate) or bool(decision.api_executive_directive_performed)
                status = "PASS" if performed and checkpoint_ok and teardown_ok and api_exec_ok else "REVIEW"
            except Exception as exc:
                status = "REVIEW"
                error = {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-3000:]}

        result = RunResult(
            version=VERSION,
            status=status,
            mode="v89_sustained_real_deepspeed_control",
            execution_requested=req.real_micro_train,
            execution_performed=performed,
            candidate_plan=plan.to_dict(),
            api_executive_directive=directive.to_dict(),
            aggressive_bounded_policy=decision.to_dict(),
            environment=env.to_dict(),
            requested={
                "steps": req.max_steps,
                "requested_batch_size": req.batch_size,
                "requested_zero_stage": req.zero_stage,
                "requested_precision": req.precision,
                "requested_gradient_accumulation_steps": req.gradient_accumulation_steps,
                "effective_batch_size": effective_hyperparams.get("batch_size", req.batch_size),
                "effective_zero_stage": effective_hyperparams.get("zero_stage", req.zero_stage),
                "effective_precision": effective_hyperparams.get("precision", req.precision),
                "effective_gradient_accumulation_steps": effective_hyperparams.get("gradient_accumulation_steps", req.gradient_accumulation_steps),
                "applied_hyperparams_source": "LocalPolicy.applied_hyperparams" if decision.allowed else "not_applied",
                "api_executive_moderate_requested": req.api_executive_moderate,
                "api_executive_directive_source": executive_directives.get("source", "none"),
                "api_executive_authority": decision.api_executive_authority,
                "api_executive_directive_performed": decision.api_executive_directive_performed,
                "api_calls_attempted": api_telemetry.get("api_calls_attempted", 0),
                "api_calls_succeeded": api_telemetry.get("api_calls_succeeded", 0),
                "api_total_tokens": api_telemetry.get("api_total_tokens", 0),
                "api_runtime_changes_count": api_telemetry.get("api_runtime_changes_count", 0),
                "real_limited_apply_requested": req.real_limited_apply,
                "real_dataset": req.real_dataset,
                "dataset_name": req.dataset_name,
                "dataset_config": req.dataset_config,
                "dataset_split": req.dataset_split,
                "dataset_streaming": req.dataset_streaming,
                "dataset_fallback_name": req.dataset_fallback_name,
                "tokenizer_name": req.tokenizer_name,
                "sequence_length": req.sequence_length,
                "model_preset": req.model_preset,
                "benchmark_mode": req.benchmark_mode,
                "chaos_profile": req.chaos_profile,
                "dataset_mix": req.dataset_mix,
                "guardrail_mode": req.guardrail_mode,
                "guardrail_sample_interval": req.guardrail_sample_interval,
                "gradient_audit": req.gradient_audit,
                "adaptive_memory_apply_suggestions": req.adaptive_memory_apply_suggestions,
                "real_limited_apply_performed": decision.real_limited_apply_performed,
                "persistent_checkpoint": req.persistent_checkpoint,
                "load_checkpoint": req.load_checkpoint,
                "checkpoint_label": os.environ.get("MEM_CHECKPOINT_LABEL", "v89"),
                "confirmation_expected": CONFIRMATION_TOKEN,
                "confirmation_received": bool(req.confirmation),
            },
            runtime=runtime,
            checkpoint=checkpoint,
            api_telemetry=api_telemetry,
            evidence_dir=str(evidence_dir),
            error=error,
            llm_apply_allowed=decision.llm_apply_allowed,
            api_executive_authority=decision.api_executive_authority,
            guardrail_envelope_passed=bool(performed and decision.allowed),
            production_mode=False,
        )
        payload = result.to_dict()
        api_telemetry["runtime_api_executive_enabled"] = bool(runtime.get("api_executive_enabled"))
        api_telemetry["runtime_api_runtime_changes_count"] = int(runtime.get("api_runtime_changes_count", 0) or 0)
        api_telemetry["runtime_api_directives_applied"] = int(runtime.get("api_directives_applied", 0) or 0)
        api_telemetry["runtime_tokens_processed"] = int(runtime.get("tokens_processed", 0) or 0)
        payload["api_telemetry"] = api_telemetry
        (evidence_dir / "api_usage_summary.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
        (self.context.state.reports_dir / "api_usage_summary_latest.json").write_text(json.dumps(api_telemetry, indent=2), encoding="utf-8")
        (evidence_dir / "v89_adaptive_benchmark_session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.context.state.reports_dir.mkdir(exist_ok=True)
        (self.context.state.reports_dir / "v89_adaptive_benchmark_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
