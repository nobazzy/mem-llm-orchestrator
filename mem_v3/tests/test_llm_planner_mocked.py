from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from domain.models import RuntimeRequest
from infrastructure.llm_client import LLMPlanner, _extract_json, _usage_dict


def test_extract_json_plain():
    raw = '{"should_run_micro_train": true, "max_steps": 100}'
    result = _extract_json(raw)
    assert result["should_run_micro_train"] is True
    assert result["max_steps"] == 100


def test_extract_json_markdown_fenced():
    raw = 'Here is the plan:\n```json\n{"should_run_micro_train": true, "batch_size": 4}\n```\nDone.'
    result = _extract_json(raw)
    assert result["should_run_micro_train"] is True
    assert result["batch_size"] == 4


def test_extract_json_embedded():
    raw = 'Prefix text {"expected_risk": 0.05, "action": "test"} suffix text'
    result = _extract_json(raw)
    assert result["expected_risk"] == 0.05
    assert result["action"] == "test"


def test_extract_json_empty_raises():
    with pytest.raises(ValueError):
        _extract_json("")


def test_usage_dict_parsing():
    mock_resp = MagicMock()
    mock_resp.usage.prompt_tokens = 120
    mock_resp.usage.completion_tokens = 80
    mock_resp.usage.total_tokens = 200
    usage = _usage_dict(mock_resp)
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 80
    assert usage["total_tokens"] == 200


def test_llm_planner_plan_mocked_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    planner = LLMPlanner(model="gpt-4o")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "should_run_micro_train": True,
        "max_steps": 500,
        "batch_size": 2,
        "zero_stage": 0,
        "precision": "fp16",
        "gradient_accumulation_steps": 2,
        "expected_risk": 0.02,
        "rationale": "Mocked safe execution plan",
        "safety_notes": ["Safety check passed"],
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50
    mock_resp.usage.total_tokens = 150
    mock_client.chat.completions.create.return_value = mock_resp

    monkeypatch.setattr(planner, "_client", lambda: mock_client)

    req = RuntimeRequest(1000, 1, 0, "fp32", False, llm_enabled=True).normalized()
    plan = planner.plan(req)

    assert plan.source == "llm_candidate_plan"
    assert plan.should_run_micro_train is True
    assert plan.max_steps == 500
    assert plan.batch_size == 2
    assert plan.precision == "fp16"
    assert plan.expected_risk == 0.02
    assert plan.schema_valid is True

    summary = planner.telemetry_summary()
    assert summary["api_calls_attempted"] == 1
    assert summary["api_calls_succeeded"] == 1
    assert summary["api_total_tokens"] == 150


def test_llm_planner_plan_handles_api_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    planner = LLMPlanner(model="gpt-4o")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI connection timeout")
    monkeypatch.setattr(planner, "_client", lambda: mock_client)

    req = RuntimeRequest(1000, 1, 0, "fp32", False, llm_enabled=True).normalized()
    plan = planner.plan(req)

    assert plan.source == "local_fallback_llm_error"
    assert plan.schema_valid is False
    assert len(plan.schema_errors) == 1
    assert "OpenAI connection timeout" in plan.schema_errors[0]

    summary = planner.telemetry_summary()
    assert summary["api_calls_attempted"] == 1
    assert summary["api_calls_failed"] == 1
    assert "OpenAI connection timeout" in summary["api_last_error"]


def test_llm_planner_executive_directive_mocked_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    planner = LLMPlanner(model="gpt-4o")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "enabled": True,
        "authority_level": "moderate",
        "action": "stabilize_lane",
        "lr_multiplier": 0.88,
        "gradient_clip_norm": 0.80,
        "loss_scale_initial_power": 8,
        "numerical_recovery_budget": 5000,
        "checkpoint_milestones": [1000, 5000],
        "event_triggers": ["loss_spike"],
        "dataset_directive": "stream_fineweb",
        "rationale": "Moderate stabilizing adjustments",
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 200
    mock_resp.usage.completion_tokens = 100
    mock_resp.usage.total_tokens = 300
    mock_client.chat.completions.create.return_value = mock_resp

    monkeypatch.setattr(planner, "_client", lambda: mock_client)

    req = RuntimeRequest(10000, 1, 0, "fp32", False, llm_enabled=True, api_executive_moderate=True).normalized()
    plan = planner.plan(req)
    directive = planner.executive_directive(req, plan, {})

    assert directive.source == "llm_moderate_executive_directive"
    assert directive.enabled is True
    assert directive.authority_level == "moderate"
    assert directive.lr_multiplier == 0.88
    assert directive.gradient_clip_norm == 0.80
    assert directive.checkpoint_milestones == [1000, 5000]
