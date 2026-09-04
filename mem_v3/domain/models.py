from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VERSION = "v89.0.0"
CONFIRMATION_TOKEN = "I_UNDERSTAND_V89_RECOVERY_CONTROL"
MAX_STEPS_HARD_CAP = 10_000_000


@dataclass
class RuntimeRequest:
    max_steps: int
    batch_size: int
    zero_stage: int
    precision: str
    persistent_checkpoint: bool
    load_checkpoint: str = ""
    confirmation: str = ""
    llm_enabled: bool = False
    api_executive_moderate: bool = False
    operator: bool = False
    real_micro_train: bool = False
    profile: str = "bounded"
    real_limited_apply: bool = False
    gradient_accumulation_steps: int = 1
    real_dataset: bool = False
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    dataset_streaming: bool = True
    dataset_fallback_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    sequence_length: int = 128
    model_preset: str = "tiny_decoder"
    benchmark_mode: str = "mem_real_chaos"
    chaos_profile: str = "real_streaming_mix"
    dataset_mix: str = "HuggingFaceFW/fineweb-edu:sample-10BT,roneneldan/TinyStories:"
    guardrail_mode: str = "sampled"
    guardrail_sample_interval: int = 8
    gradient_audit: bool = True
    adaptive_memory_apply_suggestions: bool = False

    def normalized(self) -> "RuntimeRequest":
        self.max_steps = max(1, min(int(self.max_steps), MAX_STEPS_HARD_CAP))
        self.batch_size = max(1, min(int(self.batch_size), 16))
        self.zero_stage = int(self.zero_stage)
        if self.zero_stage not in {0, 1}:
            self.zero_stage = 0
        self.precision = str(self.precision).lower()
        if self.precision not in {"fp32", "fp16", "bf16"}:
            self.precision = "fp32"
        self.gradient_accumulation_steps = max(1, min(int(self.gradient_accumulation_steps), 16))
        self.sequence_length = max(32, min(int(self.sequence_length), 512))
        self.dataset_name = str(self.dataset_name or "HuggingFaceFW/fineweb-edu")
        self.dataset_config = str(self.dataset_config or "")
        self.dataset_split = str(self.dataset_split or "train")
        self.dataset_fallback_name = str(self.dataset_fallback_name or "roneneldan/TinyStories")
        self.tokenizer_name = str(self.tokenizer_name or "gpt2")
        self.model_preset = str(self.model_preset or "tiny_decoder")
        self.benchmark_mode = str(self.benchmark_mode or "mem_real_chaos")
        if self.benchmark_mode not in {"mem_real_chaos", "mem_native_pytorch", "torch_native"}:
            self.benchmark_mode = "mem_real_chaos"
        self.chaos_profile = str(self.chaos_profile or "real_streaming_mix")
        if self.chaos_profile not in {"clean", "real_streaming_mix", "real_multilingual_noise", "real_checkpoint_pressure", "real_desktop_contention"}:
            self.chaos_profile = "real_streaming_mix"
        self.dataset_mix = str(self.dataset_mix or "HuggingFaceFW/fineweb-edu:sample-10BT,roneneldan/TinyStories:")
        self.guardrail_mode = str(self.guardrail_mode or "sampled").lower()
        if self.guardrail_mode not in {"full", "sampled", "minimal"}:
            self.guardrail_mode = "sampled"
        self.guardrail_sample_interval = max(1, min(int(self.guardrail_sample_interval), 1000))
        self.gradient_audit = bool(self.gradient_audit)
        self.adaptive_memory_apply_suggestions = bool(self.adaptive_memory_apply_suggestions)
        return self


@dataclass
class CandidatePlan:
    source: str
    should_run_micro_train: bool
    max_steps: int
    batch_size: int
    zero_stage: int
    precision: str
    gradient_accumulation_steps: int = 1
    expected_risk: float = 0.05
    dataset_name: str = ""
    tokenizer_name: str = ""
    sequence_length: int = 128
    task: str = "causal_language_modeling"
    rationale: str = ""
    safety_notes: List[str] = field(default_factory=list)
    schema_valid: bool = True
    schema_errors: List[str] = field(default_factory=list)
    raw_text_redacted: str = ""

    @classmethod
    def fallback(cls, req: RuntimeRequest, reason: str = "local_fallback") -> "CandidatePlan":
        return cls(
            source=reason,
            should_run_micro_train=req.real_micro_train,
            max_steps=req.max_steps,
            batch_size=req.batch_size,
            zero_stage=req.zero_stage,
            precision=req.precision,
            gradient_accumulation_steps=req.gradient_accumulation_steps,
            dataset_name=req.dataset_name,
            tokenizer_name=req.tokenizer_name,
            sequence_length=req.sequence_length,
            rationale="Local bounded fallback plan; LocalPolicy remains authoritative.",
            safety_notes=["Real dataset fire lane keeps 10M hard cap.", "Checkpoint and teardown remain mandatory."],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutiveDirective:
    source: str = "disabled"
    enabled: bool = False
    authority_level: str = "none"
    action: str = "observe_only"
    lr_multiplier: float = 1.0
    gradient_clip_norm: float = 1.0
    loss_scale_initial_power: int = 8
    numerical_recovery_budget: int = 1000
    checkpoint_milestones: List[int] = field(default_factory=list)
    event_triggers: List[str] = field(default_factory=list)
    dataset_directive: str = "real_dataset_causal_lm_observe_and_stabilize"
    rationale: str = ""
    safety_notes: List[str] = field(default_factory=list)
    schema_valid: bool = True
    schema_errors: List[str] = field(default_factory=list)
    raw_text_redacted: str = ""

    @classmethod
    def disabled(cls, reason: str = "api_executive_disabled") -> "ExecutiveDirective":
        return cls(source=reason, enabled=False, authority_level="none", action="observe_only")

    @classmethod
    def fallback(cls, reason: str = "local_moderate_fallback") -> "ExecutiveDirective":
        return cls(
            source=reason,
            enabled=True,
            authority_level="moderate_local_fallback",
            action="stabilize_real_dataset_10m_causal_lm_lane",
            lr_multiplier=0.90,
            gradient_clip_norm=1.0,
            loss_scale_initial_power=8,
            numerical_recovery_budget=10000,
            checkpoint_milestones=[100_000, 1_000_000, 5_000_000, 10_000_000],
            event_triggers=["nonfinite_gradient", "nonfinite_loss", "dataset_fallback", "checkpoint_failure", "throughput_drop"],
            rationale="Fallback moderate directive: quality-preserving real dataset lane; avoid systematic LR/clip suppression unless a critical event is observed.",
            safety_notes=["Directive is bounded by LocalPolicy.", "API cannot disable checkpoint, teardown or hard cap."],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentReport:
    python_version: str
    platform: Dict[str, Any]
    torch: Dict[str, Any]
    deepspeed: Dict[str, Any]
    cuda: Dict[str, Any]
    mpi: Dict[str, Any]
    compatibility: Dict[str, Any]
    doctor_status: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyDecision:
    allowed: bool
    lane: str
    reason: str
    max_steps_allowed: int
    persistent_checkpoint_allowed: bool
    rollback_required: bool = True
    local_policy_authoritative: bool = True
    llm_apply_allowed: bool = False
    api_executive_authority: str = "none"
    operator_note: str = "API has moderate executive authority only after LocalPolicy validates the bounded directive."
    applied_hyperparams: Dict[str, Any] = field(default_factory=dict)
    executive_runtime_directives: Dict[str, Any] = field(default_factory=dict)
    real_limited_apply_performed: bool = False
    api_executive_directive_performed: bool = False
    policy_rejections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    version: str
    status: str
    mode: str
    execution_requested: bool
    execution_performed: bool
    candidate_plan: Dict[str, Any]
    api_executive_directive: Dict[str, Any]
    aggressive_bounded_policy: Dict[str, Any]
    environment: Dict[str, Any]
    requested: Dict[str, Any]
    runtime: Dict[str, Any]
    checkpoint: Dict[str, Any]
    api_telemetry: Dict[str, Any]
    evidence_dir: str
    error: Optional[Dict[str, str]] = None
    local_policy_authoritative: bool = True
    llm_apply_allowed: bool = False
    api_executive_authority: str = "none"
    autonomous_execution_ready: bool = False
    rollback_required: bool = True
    guardrails_enabled: bool = True
    guardrail_envelope_passed: bool = False
    production_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
