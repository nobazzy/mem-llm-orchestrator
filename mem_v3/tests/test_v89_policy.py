from domain.models import RuntimeRequest, CandidatePlan, EnvironmentReport, CONFIRMATION_TOKEN, ExecutiveDirective
from core.policy_engine import LocalPolicyEngine


def _env(cuda=True):
    return EnvironmentReport("3.13", {}, {"cuda_available": cuda}, {"import_ok": True}, {}, {}, {}, "PASS" if cuda else "NO_GO", [])


def test_v89_real_dataset_request_keeps_10m_and_dataset_fields():
    req = RuntimeRequest(
        10_000_000, 8, 1, "fp16", True,
        confirmation=CONFIRMATION_TOKEN, operator=True, real_micro_train=True,
        real_limited_apply=True, api_executive_moderate=True, real_dataset=True,
        dataset_name="HuggingFaceFW/fineweb-edu", dataset_config="sample-10BT",
        tokenizer_name="gpt2", sequence_length=128,
    ).normalized()
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env(), ExecutiveDirective.fallback())
    assert decision.allowed
    assert decision.lane == "v89_real_chaos_mem_lane"
    assert decision.max_steps_allowed == 10_000_000
    assert decision.applied_hyperparams["batch_size"] <= 8
    assert decision.executive_runtime_directives["validated_by_local_policy"] is True


def test_v89_confirmation_token_rejects():
    req = RuntimeRequest(1000, 4, 1, "fp16", True, confirmation="wrong", operator=True, real_micro_train=True).normalized()
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env())
    assert not decision.allowed
    assert decision.reason == "invalid_confirmation_token"


def test_v89_no_cuda_blocks():
    req = RuntimeRequest(1000, 4, 1, "fp16", True, confirmation=CONFIRMATION_TOKEN, operator=True, real_micro_train=True).normalized()
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env(cuda=False))
    assert not decision.allowed
    assert decision.reason == "cuda_unavailable"


def test_v89_moderate_api_directive_is_clamped():
    req = RuntimeRequest(10_000_000, 8, 1, "fp16", True, confirmation=CONFIRMATION_TOKEN, operator=True, real_micro_train=True, real_limited_apply=True, api_executive_moderate=True, real_dataset=True).normalized()
    directive = ExecutiveDirective(enabled=True, authority_level="moderate", action="stabilize", lr_multiplier=0.01, gradient_clip_norm=99, loss_scale_initial_power=99, numerical_recovery_budget=999999, checkpoint_milestones=[1, 99_000_000])
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env(), directive)
    assert decision.allowed
    assert decision.executive_runtime_directives["lr_multiplier"] == 0.85
    assert decision.executive_runtime_directives["gradient_clip_norm"] == 1.25
    assert decision.executive_runtime_directives["loss_scale_initial_power"] == 10
    assert decision.executive_runtime_directives["numerical_recovery_budget"] == 30000
    assert 10_000_000 in decision.executive_runtime_directives["checkpoint_milestones"]


def test_v89_real_limited_apply_does_not_inflate_small_batch():
    # If user requests batch 1 or 2 with real_limited_apply=True, do NOT inflate to 4!
    req = RuntimeRequest(
        1000, 2, 0, "fp32", True,
        confirmation=CONFIRMATION_TOKEN, operator=True, real_micro_train=True,
        real_limited_apply=True, real_dataset=True,
    ).normalized()
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env())
    assert decision.allowed
    assert decision.applied_hyperparams["batch_size"] == 2
    assert decision.applied_hyperparams["precision"] == "fp16"


def test_v89_invalid_env_clip_norm_fallback(monkeypatch):
    monkeypatch.setenv("MEM_V89_GRADIENT_CLIP_NORM", "not-a-number-string")
    req = RuntimeRequest(
        1000, 4, 0, "fp16", True,
        confirmation=CONFIRMATION_TOKEN, operator=True, real_micro_train=True,
        api_executive_moderate=True, real_dataset=True,
    ).normalized()
    directive = ExecutiveDirective(enabled=True, authority_level="moderate", action="observe", gradient_clip_norm=0.75)
    decision = LocalPolicyEngine().evaluate(req, CandidatePlan.fallback(req), _env(), directive)
    assert decision.allowed
    # Falls back safely without throwing ValueError
    assert decision.executive_runtime_directives["gradient_clip_norm"] == 0.75


def test_orchestrator_selects_native_runner_and_runs(tmp_path):
    import pytest
    pytest.importorskip("torch")
    from core.orchestrator import MemOrchestrator, OrchestratorContext

    ctx = OrchestratorContext(base_dir=str(tmp_path))
    orchestrator = MemOrchestrator(context=ctx)

    req = RuntimeRequest(
        max_steps=5,
        batch_size=2,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        confirmation=CONFIRMATION_TOKEN,
        operator=True,
        real_micro_train=True,
        real_dataset=False,
        benchmark_mode="mem_native_pytorch",
    )
    # Validate runner selection directly on context
    selected = ctx.select_runner(req.benchmark_mode)
    assert selected is ctx.native_runner

    # Execute synthetic run through native runner to ensure complete integration
    metrics, ckpt = selected.run(
        steps=req.max_steps,
        batch_size=req.batch_size,
        zero_stage=0,
        precision="fp32",
        persistent_checkpoint=True,
        applied_hyperparams={"batch_size": 2, "precision": "fp32", "gradient_accumulation_steps": 1},
        dataset_settings={"real_dataset": False, "evidence_dir": str(tmp_path / "evidence")},
    )
    assert metrics["execution_performed"] is True
    assert ckpt["checkpoint_written"] is True
