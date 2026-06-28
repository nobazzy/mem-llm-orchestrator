# Architecture — MEM Orchestrator v89

MEM Orchestrator v89 is organized as an orchestration layer above a real DeepSpeed/PyTorch training runtime.

## Main flow

```txt
run script
  -> sustained controller
      -> runtime health monitoring
      -> policy/guardrail checks
      -> API Light decision path when enabled
      -> lane refresh or lane switch decision
      -> DeepSpeed runtime execution
      -> evidence/log/status output
```

## Main components

```txt
scripts/run_v89_sustained_control.sh   official launcher
scripts/v89_sustained_controller.py    long-run controller
scripts/monitor_v89_human.sh           live human monitor
runtime/deepspeed_runner.py            training runtime
runtime/real_dataset.py                real dataset/cache path
runtime/lm_model.py                    tiny causal language model runtime
core/orchestrator.py                   orchestration entrypoint
core/policy_engine.py                  local policy guard
```

## Runtime boundary

MEM does not replace PyTorch, DeepSpeed or Hugging Face tooling. It coordinates runtime behavior around them, with a focus on safety, observability and long-run completion.
