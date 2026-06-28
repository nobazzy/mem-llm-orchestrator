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
