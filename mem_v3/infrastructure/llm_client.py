from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from domain.models import CandidatePlan, ExecutiveDirective, RuntimeRequest


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty LLM response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _usage_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


class LLMPlanner:
    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._events: List[Dict[str, Any]] = []

    def _client(self):
        if not os.environ.get(self.api_key_env):
            return None
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=os.environ.get(self.api_key_env), timeout=float(os.environ.get("MEM_V89_API_TIMEOUT_SECONDS", "8")))

    def _record(self, *, call_type: str, attempted: bool, succeeded: bool, usage: Dict[str, int] | None = None, error: str = "", result_source: str = "") -> None:
        self._events.append({
            "ts": round(time.time(), 3),
            "model": self.model,
            "call_type": call_type,
            "attempted": bool(attempted),
            "succeeded": bool(succeeded),
            "prompt_tokens": int((usage or {}).get("prompt_tokens", 0)),
            "completion_tokens": int((usage or {}).get("completion_tokens", 0)),
            "total_tokens": int((usage or {}).get("total_tokens", 0)),
            "error": str(error)[:500],
            "result_source": str(result_source)[:120],
        })

    def telemetry_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def telemetry_summary(self) -> Dict[str, Any]:
        attempted = [e for e in self._events if e.get("attempted")]
        succeeded = [e for e in attempted if e.get("succeeded")]
        failed = [e for e in attempted if not e.get("succeeded")]
        return {
            "api_telemetry_enabled": True,
            "api_calls_attempted": len(attempted),
            "api_calls_succeeded": len(succeeded),
            "api_calls_failed": len(failed),
            "api_prompt_tokens_total": sum(int(e.get("prompt_tokens", 0)) for e in attempted),
            "api_completion_tokens_total": sum(int(e.get("completion_tokens", 0)) for e in attempted),
            "api_total_tokens": sum(int(e.get("total_tokens", 0)) for e in attempted),
            "api_candidate_plan_calls": sum(1 for e in succeeded if e.get("call_type") == "candidate_plan"),
            "api_executive_directive_calls": sum(1 for e in succeeded if e.get("call_type") == "executive_directive"),
            "api_last_error": failed[-1].get("error", "") if failed else "",
            "api_events": self.telemetry_events(),
        }

    def plan(self, req: RuntimeRequest) -> CandidatePlan:
        if not req.llm_enabled or not os.environ.get(self.api_key_env):
            self._record(call_type="candidate_plan", attempted=False, succeeded=False, result_source="local_fallback_no_api")
            return CandidatePlan.fallback(req, "local_fallback_no_api")
        try:
            client = self._client()
            prompt = {
                "task": "Produce a bounded real-dataset Causal LM execution candidate plan. Return JSON only.",
                "operator_request": {
                    "max_steps": req.max_steps,
                    "batch_size": req.batch_size,
                    "zero_stage": req.zero_stage,
                    "precision": req.precision,
                    "persistent_checkpoint": req.persistent_checkpoint,
                    "real_dataset": req.real_dataset,
                    "dataset_name": req.dataset_name,
                    "dataset_config": req.dataset_config,
                    "fallback_dataset": req.dataset_fallback_name,
                    "tokenizer_name": req.tokenizer_name,
                    "sequence_length": req.sequence_length,
                    "task": "causal_language_modeling",
                },
                "rules": [
                    "Preserve the 10M target when requested.",
                    "Prefer FineWeb-Edu streaming; TinyStories can be used as fallback only if primary loading fails.",
                    "The API may plan and moderate, but LocalPolicy validates before runtime.",
                    "Do not disable checkpoint, teardown, evidence or guardrails.",
                    "Return decisions that can be measured through api telemetry, runtime metrics and policy acceptance.",
                ],
                "schema": {
                    "should_run_micro_train": "boolean",
                    "max_steps": "integer <= 10000000",
                    "batch_size": "integer 1..16",
                    "zero_stage": "0|1",
                    "precision": "fp32|fp16|bf16",
                    "gradient_accumulation_steps": "integer >= 1",
                    "dataset_name": "string",
                    "tokenizer_name": "string",
                    "sequence_length": "integer",
                    "expected_risk": "float 0..1",
                    "rationale": "string",
                    "safety_notes": "array of strings",
                },
            }
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                timeout=float(os.environ.get("MEM_V89_API_TIMEOUT_SECONDS", "8")),
                messages=[
                    {"role": "system", "content": "You are a bounded runtime planning agent for real-dataset Causal LM. Return strict JSON only."},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
            )
            usage = _usage_dict(response)
            data = _extract_json(response.choices[0].message.content or "")
            plan = CandidatePlan(
                source="llm_candidate_plan",
                should_run_micro_train=bool(data.get("should_run_micro_train", req.real_micro_train)),
                max_steps=min(int(data.get("max_steps", req.max_steps)), req.max_steps, 10_000_000),
                batch_size=max(1, min(int(data.get("batch_size", req.batch_size)), 16)),
                zero_stage=int(data.get("zero_stage", req.zero_stage)) if int(data.get("zero_stage", req.zero_stage)) in {0, 1} else req.zero_stage,
                precision=str(data.get("precision", req.precision)).lower() if str(data.get("precision", req.precision)).lower() in {"fp32", "fp16", "bf16"} else req.precision,
                gradient_accumulation_steps=max(1, min(int(data.get("gradient_accumulation_steps", req.gradient_accumulation_steps)), 16)),
                dataset_name=str(data.get("dataset_name", req.dataset_name)),
                tokenizer_name=str(data.get("tokenizer_name", req.tokenizer_name)),
                sequence_length=max(32, min(int(data.get("sequence_length", req.sequence_length)), 512)),
                expected_risk=float(data.get("expected_risk", 0.08)),
                rationale=str(data.get("rationale", "")),
                safety_notes=[str(x) for x in _as_list(data.get("safety_notes"))],
                raw_text_redacted="present_redacted",
            )
            self._record(call_type="candidate_plan", attempted=True, succeeded=True, usage=usage, result_source=plan.source)
            return plan
        except Exception as exc:
            plan = CandidatePlan.fallback(req, "local_fallback_llm_error")
            plan.schema_valid = False
            plan.schema_errors = [f"{type(exc).__name__}: {exc}"]
            self._record(call_type="candidate_plan", attempted=True, succeeded=False, error=f"{type(exc).__name__}: {exc}", result_source=plan.source)
            return plan

    def executive_directive(self, req: RuntimeRequest, plan: CandidatePlan, env: Dict[str, Any]) -> ExecutiveDirective:
        if not req.llm_enabled or not req.api_executive_moderate or not os.environ.get(self.api_key_env):
            self._record(call_type="executive_directive", attempted=False, succeeded=False, result_source="api_executive_not_enabled_or_no_key")
            return ExecutiveDirective.disabled("api_executive_not_enabled_or_no_key")
        try:
            client = self._client()
            prompt = {
                "task": "Return a moderate executive runtime directive for a 10M real-dataset Causal LM fire test. JSON only.",
                "operator_request": {
                    "max_steps": req.max_steps,
                    "dataset_name": req.dataset_name,
                    "dataset_config": req.dataset_config,
                    "fallback_dataset": req.dataset_fallback_name,
                    "tokenizer_name": req.tokenizer_name,
                    "sequence_length": req.sequence_length,
                    "precision": req.precision,
                    "gradient_accumulation_steps": req.gradient_accumulation_steps,
                },
                "candidate_plan": plan.to_dict(),
                "environment_summary": {"torch": env.get("torch", {}), "deepspeed": env.get("deepspeed", {}), "compatibility": env.get("compatibility", {}), "doctor_status": env.get("doctor_status")},
                "allowed_authority": [
                    "lr_multiplier between 0.20 and 1.0",
                    "gradient_clip_norm between 0.25 and 1.0",
                    "loss_scale_initial_power between 5 and 8",
                    "numerical_recovery_budget between 100 and 30000",
                    "checkpoint_milestones within max_steps",
                    "event_triggers for dataset/runtime evidence",
                    "recommend fallback handling, but LocalPolicy/runtime decides actual fallback if primary dataset load fails",
                ],
                "forbidden_authority": ["disable checkpoint", "disable teardown", "bypass confirmation", "exceed 10M steps", "mark failed run as pass", "grant production mode"],
                "output_contract": ["Every directive must be measurable by LocalPolicy and runtime telemetry.", "Prefer explicit checkpoint milestones and event triggers."],
            }
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                timeout=float(os.environ.get("MEM_V89_API_TIMEOUT_SECONDS", "8")),
                messages=[
                    {"role": "system", "content": "You are a moderate executive runtime advisor for real-dataset Causal LM. Return strict JSON only."},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
            )
            usage = _usage_dict(response)
            data = _extract_json(response.choices[0].message.content or "")
            directive = ExecutiveDirective(
                source="llm_moderate_executive_directive",
                enabled=bool(data.get("enabled", True)),
                authority_level=str(data.get("authority_level", "moderate")),
                action=str(data.get("action", "stabilize_real_dataset_10m_causal_lm_lane")),
                lr_multiplier=float(data.get("lr_multiplier", 0.65)),
                gradient_clip_norm=float(data.get("gradient_clip_norm", 0.75)),
                loss_scale_initial_power=int(data.get("loss_scale_initial_power", 7)),
                numerical_recovery_budget=int(data.get("numerical_recovery_budget", 10000)),
                checkpoint_milestones=[int(x) for x in _as_list(data.get("checkpoint_milestones")) if str(x).lstrip("-").isdigit()],
                event_triggers=[str(x) for x in _as_list(data.get("event_triggers"))],
                dataset_directive=str(data.get("dataset_directive", "real_dataset_causal_lm_observe_and_stabilize")),
                rationale=str(data.get("rationale", "")),
                safety_notes=[str(x) for x in _as_list(data.get("safety_notes"))],
                raw_text_redacted="present_redacted",
            )
            self._record(call_type="executive_directive", attempted=True, succeeded=True, usage=usage, result_source=directive.source)
            return directive
        except Exception as exc:
            directive = ExecutiveDirective.fallback("local_moderate_fallback_llm_error")
            directive.schema_valid = False
            directive.schema_errors = [f"{type(exc).__name__}: {exc}"]
            self._record(call_type="executive_directive", attempted=True, succeeded=False, error=f"{type(exc).__name__}: {exc}", result_source=directive.source)
            return directive
