from __future__ import annotations

import pytest

from core.policy_engine import LocalPolicyEngine
from domain.models import (
    CONFIRMATION_TOKEN,
    CandidatePlan,
    EnvironmentReport,
    ExecutiveDirective,
    RuntimeRequest,
)


def _sample_env(cuda=True, deepspeed=True):
    return EnvironmentReport(
        python_version="3.12.0",
        platform={"system": "Linux", "release": "6.6.0"},
        torch={"cuda_available": cuda, "import_ok": True},
        deepspeed={"import_ok": deepspeed, "version": "0.19.1"},
        cuda={"nvcc_available": True},
        mpi={"import_ok": True},
        compatibility={},
        doctor_status="PASS" if cuda and deepspeed else "NO_GO",
        recommendations=[],
    )


def _base_valid_request():
    return RuntimeRequest(
        max_steps=1000,
        batch_size=2,
        zero_stage=0,
        precision="fp16",
        persistent_checkpoint=True,
        confirmation=CONFIRMATION_TOKEN,
        operator=True,
        real_micro_train=True,
        real_dataset=True,
    ).normalized()


def test_policy_allows_valid_request():
    req = _base_valid_request()
    plan = CandidatePlan.fallback(req)
    env = _sample_env()
    decision = LocalPolicyEngine().evaluate(req, plan, env)

    assert decision.allowed is True
    assert decision.lane == "v89_real_chaos_mem_lane"
    assert decision.max_steps_allowed == 10_000_000
    assert decision.applied_hyperparams["batch_size"] == 2
    assert decision.applied_hyperparams["precision"] == "fp16"


@pytest.mark.parametrize(
    "override_kwargs,expected_reason",
    [
        ({"confirmation": "BAD_TOKEN"}, "invalid_confirmation_token"),
        ({"operator": False}, "operator_mode_required"),
        ({"real_micro_train": False}, "real_micro_train_not_requested"),
        ({"real_dataset": False}, "real_dataset_required"),
        ({"max_steps": 20_000_000}, "requested_steps_exceed_hard_cap"),
    ],
)
def test_policy_rejection_reasons(override_kwargs, expected_reason):
    req_data = {
        "max_steps": 1000,
        "batch_size": 2,
        "zero_stage": 0,
        "precision": "fp16",
        "persistent_checkpoint": True,
        "confirmation": CONFIRMATION_TOKEN,
        "operator": True,
        "real_micro_train": True,
        "real_dataset": True,
    }
    # Do not normalize if testing raw boundary
    req = RuntimeRequest(**req_data)
    for k, v in override_kwargs.items():
        setattr(req, k, v)

    plan = CandidatePlan.fallback(req)
    env = _sample_env()
    decision = LocalPolicyEngine().evaluate(req, plan, env)

    assert decision.allowed is False
    assert decision.reason == expected_reason


def test_policy_blocks_when_cuda_or_deepspeed_missing():
    req = _base_valid_request()
    plan = CandidatePlan.fallback(req)

    # Missing CUDA
    decision_no_cuda = LocalPolicyEngine().evaluate(req, plan, _sample_env(cuda=False))
    assert decision_no_cuda.allowed is False
    assert decision_no_cuda.reason == "cuda_unavailable"

    # Missing DeepSpeed
    decision_no_ds = LocalPolicyEngine().evaluate(req, plan, _sample_env(deepspeed=False))
    assert decision_no_ds.allowed is False
    assert decision_no_ds.reason == "deepspeed_not_importable"


def test_policy_clamps_extreme_directive_values():
    req = _base_valid_request()
    req.api_executive_moderate = True
    plan = CandidatePlan.fallback(req)
    env = _sample_env()

    # Extreme low/high directive values
    directive = ExecutiveDirective(
        enabled=True,
        authority_level="moderate",
        action="extreme_action",
        lr_multiplier=0.0001,             # Must clamp to >= 0.85
        gradient_clip_norm=100.0,         # Must clamp to <= 1.25
        loss_scale_initial_power=99,      # Must clamp to <= 10
        numerical_recovery_budget=999999, # Must clamp to <= 30000
        checkpoint_milestones=[500, 2000, 50000],
        event_triggers=["spike"],
        dataset_directive="stabilize",
    )

    decision = LocalPolicyEngine().evaluate(req, plan, env, directive)
    assert decision.allowed is True
    assert decision.api_executive_authority == "moderate_validated"

    exec_dirs = decision.executive_runtime_directives
    assert exec_dirs["validated_by_local_policy"] is True
    assert exec_dirs["lr_multiplier"] == 0.85
    assert exec_dirs["gradient_clip_norm"] == 1.25
    assert exec_dirs["loss_scale_initial_power"] == 10
    assert exec_dirs["numerical_recovery_budget"] == 30000
    assert 500 in exec_dirs["checkpoint_milestones"]
    assert 1000 in exec_dirs["checkpoint_milestones"]  # Clamped to req.max_steps
