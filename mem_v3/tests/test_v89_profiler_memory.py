from pathlib import Path
from runtime.profiler import RuntimeProfiler
from runtime.adaptive_memory import AdaptiveRuntimeMemory


def test_v89_profiler_classifies_data_bottleneck(tmp_path):
    prof = RuntimeProfiler(tmp_path)
    for _ in range(3):
        prof.add(data_fetch=0.7, forward_loss=0.1, backward=0.1, optimizer=0.05, guardrail=0.01, total_step=1.0)
    summary = prof.summary()
    assert summary["profiler_enabled"] is True
    assert summary["bottleneck_classification"] == "data_or_tokenizer_bound"
    prof.write()
    assert (tmp_path / "profiler_report.json").exists()


def test_v89_adaptive_memory_records_effects(tmp_path):
    mem = AdaptiveRuntimeMemory(tmp_path)
    mem.record(step=1, directive="stabilize", lr=1e-4, gradient_clip_norm=0.75, loss=10.0, tokens_processed=1024, steps_per_second=1.0, tokens_per_second=1024.0)
    mem.record(step=10, directive="stabilize", lr=1e-4, gradient_clip_norm=0.75, loss=9.0, tokens_processed=10240, steps_per_second=2.0, tokens_per_second=2048.0)
    summary = mem.summary()
    assert summary["adaptive_memory_enabled"] is True
    assert summary["decision_effects_recorded"] == 2
    assert (tmp_path / "adaptive_memory.jsonl").exists()
    assert (tmp_path / "adaptive_memory_summary.json").exists()


def test_v89_runner_contains_benchmark_profiler_contract():
    text = Path('runtime/deepspeed_runner.py').read_text()
    assert 'profiler_report.json' in Path('runtime/profiler.py').read_text()
    assert 'adaptive_memory.jsonl' in Path('runtime/adaptive_memory.py').read_text()
    assert 'benchmark_mode' in text
    assert 'bottleneck_classification' in text
