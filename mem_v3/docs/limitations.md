# Limitations — MEM v3

MEM v3 is a configurable workload release built on top of the validated mem_v2 / v89.0.0 core.

## Validation boundary

- The sustained 300k result was validated on the mem_v2 base package.
- The internal core remains `v89.0.0`.
- mem_v3 adds configuration support for custom datasets, but each new dataset/model/tokenizer combination should be validated separately.

## Hardware boundary

- The strongest local validation is WSL2 + RTX 5060 Ti 8GB.
- Other GPUs may work, but throughput and stability are not guaranteed.
- RTX 50-series / Blackwell requires attention to the PyTorch CUDA build.
- The validated install path uses Python 3.12.13 and torch nightly cu128.
- PyTorch nightly cu128 packages may change over time.

## Dataset boundary

Supported in mem_v3:

```txt
Hugging Face datasets
local .txt datasets
local .jsonl datasets
```

Limitations:

- JSONL files must contain a valid configured text field.
- Local file paths must be visible from inside WSL.
- Dataset quality, size and formatting affect training behavior.
- Fallback behavior should be checked in evidence.

## Runtime naming boundary

Some scripts still use `v89` in their names because they belong to the validated internal core.

Do not rename runtime scripts casually unless the full validation is repeated.

## Known cosmetic state note

A secondary `global_progress_latest` file can show a stale intermediate state such as `RECOVERING` even when the authoritative controller status reports:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
```

The final controller status is authoritative.

## Release hygiene

Public ZIPs should not include:

```txt
.venv/
.venv312/
__pycache__/
.pytest_cache/
dataset_cache/
checkpoints/
*.pt
*.pth
*.bin
*.safetensors
API keys
local secrets
```
