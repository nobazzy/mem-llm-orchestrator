from domain.models import RuntimeRequest, MAX_STEPS_HARD_CAP, CONFIRMATION_TOKEN


def test_v89_request_normalization_bounds_dataset_and_steps():
    req = RuntimeRequest(99_000_000, 99, 9, "bad", True, confirmation=CONFIRMATION_TOKEN, real_dataset=True, sequence_length=9999).normalized()
    assert req.max_steps == MAX_STEPS_HARD_CAP
    assert req.batch_size == 16
    assert req.zero_stage == 0
    assert req.precision == "fp32"
    assert req.sequence_length == 512

from infrastructure.llm_client import LLMPlanner


def test_v89_api_telemetry_records_disabled_calls_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    planner = LLMPlanner()
    req = RuntimeRequest(1000, 4, 1, "fp16", True, llm_enabled=True, api_executive_moderate=True).normalized()
    planner.plan(req)
    planner.executive_directive(req, planner.plan(req), {})
    summary = planner.telemetry_summary()
    assert "api_calls_attempted" in summary
    assert "api_total_tokens" in summary
    assert summary["api_calls_attempted"] == 0
    assert summary["api_telemetry_enabled"] is True


def test_v89_runner_writes_runtime_progress_contract():
    text = __import__('pathlib').Path('runtime/deepspeed_runner.py').read_text()
    assert 'runtime_progress_latest.json' in text
    assert 'runtime_milestones.jsonl' in text
    assert 'api_runtime_changes_count' in text
    assert 'tokens_processed' in text


def test_v89_orchestrator_persists_api_usage_contract():
    text = __import__('pathlib').Path('core/orchestrator.py').read_text()
    assert 'api_telemetry.jsonl' in text
    assert 'api_usage_summary.json' in text
    assert 'api_total_tokens' in text
    assert 'evidence_dir' in text
