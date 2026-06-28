# MEM v3 Release Notes

**Public package:** `mem_v3`
**Internal validated core:** `v89.0.0`
**Base package:** `mem_v2` 300k validated release
**Current release line:** `mem_v3.3`
**Release type:** configurable workload layer, checkpoint continuity, recovery-lane hardening and practical 8GB presets

---

## Executive summary

MEM v3 keeps the validated MEM v89 runtime core from `mem_v2` and extends it into a configurable long-run training package.

The main goal of MEM v3 is to let users run sustained language-model training with their own workload configuration — Hugging Face datasets, local `.txt` files or local `.jsonl` files — without editing the internal runtime, controller or DeepSpeed integration code.

This release line adds four important capabilities on top of the validated v89 core:

```txt
1. Configurable dataset/workload layer
2. Checkpoint resume continuity between lane sessions
3. Live rotating checkpoints during long runs
4. Recovery-only safe lane behavior with forced escape from throughput-dead states
```

The practical target of this release is not short-lived benchmark spikes. MEM focuses on long-run survival, observability, checkpoint continuity and completion on constrained hardware.

A sustained 1M-step local run has now completed successfully:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
```

This does not mean every dataset, GPU or model size is universally validated. It means the current MEM v3/v89 line has completed a real extended endurance run under the documented local configuration.

---

## Release positioning

The correct positioning for this release is:

```txt
mem_v3 = mem_v2 validated v89 core
       + configurable workload layer
       + checkpoint continuity fixes
       + live rotating checkpoints
       + safe-lane recovery hardening
       + practical 50M preset for constrained GPUs
```

MEM v3 does not replace PyTorch, DeepSpeed or Hugging Face. It acts as an orchestration layer above them, focused on making long training runs safer, more observable and more recoverable on limited hardware.

---

## Version lineage

```txt
mem_v2
  validated 300k sustained run
  internal runtime core: v89.0.0

mem_v3.0
  configurable workload layer
  Hugging Face / local TXT / local JSONL support

mem_v3.2
  absolute safe-lane escape
  targeted hardening for safe_seq256 throughput-prison behavior

P0 hotfix
  checkpoint resume continuity between lane sessions

P0.1 hotfix
  live rotating checkpoints before lane switch

mem_v3.3
  safe_seq256 treated as recovery-only lane
  practical 50M SlimPajama preset for 8GB GPUs
  clearer 100M stress-test positioning
```

The internal controller and runtime scripts may still use `v89` names. This is intentional: the validated runtime core remains `v89.0.0`.

---

## What changed from mem_v2

MEM v3 adds a configurable workload layer.

Added configuration files include:

```txt
configs/mem_v3_default.yaml
configs/examples/huggingface_dataset.yaml
configs/examples/local_txt_dataset.yaml
configs/examples/local_jsonl_dataset.yaml
configs/validation/local_jsonl_100.yaml
configs/validation/local_txt_100.yaml
configs/validation/huggingface_1000.yaml
configs/long/huggingface_fineweb_edu_300k.yaml
configs/long/huggingface_fineweb_edu_1m.yaml
```

Added scripts include:

```txt
scripts/run_mem_v3.sh
scripts/translate_mem_v3_config.py
scripts/setup_wsl_conda_mem_v3.sh
scripts/doctor_mem_v3.sh
scripts/validate_mem_v3_configs.sh
scripts/run_v89_50m_slimpajama_1m.sh
scripts/run_v89_100m_slimpajama_1m.sh
scripts/monitor_v89_human.sh
scripts/monitor_v89_human_runtime.sh
scripts/verify_checkpoint_resume_v89.sh
```

Added documentation and samples include:

```txt
docs/custom_dataset.md
PATCH_NOTES_V3_ABSOLUTE_SAFE_ESCAPE.md
PATCH_NOTES_V3_3_SAFE_RECOVERY_50M.md
data/sample.txt
data/sample.jsonl
```

Runtime compatibility changes:

```txt
runtime/real_dataset.py can load local text/json files through Hugging Face datasets data_files
runtime/real_dataset.py supports configurable text_field through MEM_DATASET_TEXT_FIELD
runtime/real_dataset.py supports resilient dataset/cache behavior for repeated long-run reads
scripts/v89_sustained_controller.py can receive dataset/tokenizer/model settings through environment variables
scripts/run_mem_v3.sh can translate YAML workload configs into MEM runtime environment variables
```

---

## What did not change

The following were intentionally preserved:

```txt
internal validated core: v89.0.0
DeepSpeed runtime design
API Light controller path
monitoring style
baseline sustained-control architecture
dataset/tokenizer/model execution path
runtime script naming using v89 names
```

The following changed only in targeted ways:

```txt
controller policy
lane switching behavior for degraded safe_seq256
checkpoint continuity across lane sessions
live checkpoint cadence during long sessions
practical preset guidance for constrained GPUs
```

MEM v3 is not a full controller rewrite. It is a hardening and usability extension of the validated v89 runtime core.

---

## Dependency policy

MEM v3 is a language-model training workload.

It does not require vision or audio packages.

Install:

```txt
torch
deepspeed
datasets
transformers
tokenizers
accelerate
safetensors
openai
pyyaml
psutil
numpy
tqdm
pytest
```

Do not install:

```txt
torchvision
torchaudio
torchtext
```

Reason:

```txt
MEM v3 trains language models.
It does not use image operators.
It does not use audio operators.
It does not need torchvision datasets/models/transforms.
It does not need torchaudio backends.
Keeping the environment smaller reduces binary mismatch risk.
```

Recommended PyTorch installation pattern for CUDA 12.8-compatible environments:

```bash
python -m pip uninstall -y torch torchvision torchaudio torchtext

python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Avoid the generic PyTorch install command that includes `torchvision` and `torchaudio`, because those packages are not part of the MEM workload.

---

## Validated base evidence

MEM v3 is based on the `mem_v2` package that completed a sustained 300k-step run:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
internal_core: v89.0.0
```

Observed mem_v2 base summary:

```txt
Target completed: 300000 / 300000 global steps
Approximate duration: ~6h47min
Observed mean throughput: ~33000 tokens/s
Observed median throughput: ~32000 tokens/s
Observed peak throughput: ~45600 tokens/s
Fatal OOM blocking target: not observed
Fatal traceback blocking target: not observed
```

Current MEM v3 300k validation evidence:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
restart_index: 14
absolute safe-lane escape: observed and validated
fatal OOM blocking target: not observed
fatal traceback blocking target: not observed
```

Current MEM v3 1M endurance evidence:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
controller core: v89.0.0
package line: mem_v3.3 safe-recovery-50m-preset
```

The 1M evidence confirms extended local endurance under the tested configuration. It is not a universal performance guarantee for every model size, GPU, dataset or CUDA/PyTorch combination.

---

## P0 hotfix: checkpoint resume + SlimPajama source correction

This package includes a P0 continuity fix for lane-session checkpoint/resume.

Before the fix, the orchestrator could pass `load_checkpoint` between sessions, but the runtime path did not fully accept and record that resume instruction.

The fix updates the runtime so that:

```txt
runtime/deepspeed_runner.py accepts load_checkpoint
checkpoint-resume metadata is recorded
lane sessions can resume from the intended checkpoint path
global-step continuity is preserved more reliably
```

The 100M SlimPajama launcher now defaults to:

```txt
DKYoon/SlimPajama-6B
```

This is a sampled SlimPajama dataset. The previous `cerebras/SlimPajama-627B` target failed access in local testing and fell back to TinyStories. The fallback remains enabled as a safety fallback, not as the intended benchmark source.

Evidence and log directories are intentionally kept empty in release ZIPs. Runtime evidence should be generated fresh per execution.

---

## P0.1 hotfix: live rotating checkpoints before lane switch

This package adds live rotating checkpoints during long lane sessions.

Before this fix, a lane switch could happen before the child process reached its normal end-of-run checkpoint path. That could preserve the global-step counter while failing to preserve the actual model and optimizer state at the desired point.

P0.1 writes live checkpoints during training:

```txt
checkpoints/<label>_live/mem_model_optimizer.pt
```

It also updates the latest pointer:

```txt
<label>_latest.txt
```

This allows the next lane session to resume model and optimizer state instead of only preserving the global-step counter.

Default live checkpoint cadence for the 100M SlimPajama launcher:

```txt
MEM_LIVE_CHECKPOINT_EVERY_STEPS=1000
```

Checkpoint note:

```txt
Final logs bundles may include checkpoint metadata.
They may not include the heavy tensor file mem_model_optimizer.pt.
Resume requires the real checkpoint file to exist locally under checkpoints/.
```

---

## mem_v3.2: absolute safe-lane escape

`mem_v3.2-absolute-safe-escape` fixes a long-run control issue where the local controller could correctly detect degradation but the API path could still keep the current safe lane active.

Observed failing pattern:

```txt
Lane: safe_seq256
Tokens/s: ~14000-16000
Optimizer ratio: > 0.50
GPU utilization: low
Bad windows: high
Local recommendation: lane_switch_or_restart
Local target: fast_seq256_zero0_gacc4
API response: keep_current_lane or switch_lane with lane=None
Result: safe_seq256 remained active
```

New rule:

```txt
safe_seq256 + tokens/s < 25000 + 1 bad window
= forced escape to fast_seq256_zero0_gacc4
```

Controller hardening behavior:

```txt
target lane is forced to fast_seq256_zero0_gacc4
API keep_current_lane cannot veto this critical escape
API switch_lane with lane=None cannot block this critical escape
cooldown/recovery hold should not block this critical escape
```

Main events added or used by this policy:

```txt
safe_seq256_absolute_escape_armed
safe_seq256_hard_escape_override
safe_seq256_absolute_25k_floor_escape
safe_seq256_hot_optimizer_25k_attention_escape
safe_seq256_low_steps_absolute_escape
```

Rationale:

```txt
safe does not always mean healthy
stable does not always mean productive
a lane can avoid OOM while still destroying throughput
critical local degradation evidence must override API conservatism
```

This is a targeted correction, not a redesign of the full controller.

---

## mem_v3.3: safe recovery-only lane + 50M practical preset

`mem_v3.3` refines the lane policy after stress-testing larger presets on constrained 8GB hardware.

Added:

```txt
medium_50m / decoder_50m / 50m_decoder model preset in runtime/lm_model.py
scripts/run_v89_50m_slimpajama_1m.sh
PATCH_NOTES_V3_3_SAFE_RECOVERY_50M.md
```

Changed:

```txt
safe_seq256 is explicitly treated as a recovery-only lane
low safe-lane throughput after a short proof window arms a local degraded-throughput escape
generic bad_windows no longer needs to accumulate naturally before local escape can be armed
default safe-lane exit target becomes aggressive_seq256_zero0_gacc4 when checkpoint/loss remain coherent
```

Why:

A 100M SlimPajama stress run on an 8GB GPU showed that `safe_seq256` could remain operationally alive but throughput-dead:

```txt
high VRAM pressure
low tokens/s
low steps/s
low GPU work efficiency
no immediate fatal OOM
no immediate fatal traceback
no fast enough bad-window escalation
```

This patch separates operational survival from throughput health.

Evaluation interpretation:

```txt
100M preset:
  stress test for constrained 8GB environments
  useful for robustness evidence
  expected higher VRAM pressure and controlled churn

50M preset:
  practical 8GB preset
  better throughput/stability balance
  recommended comparison run after 100M stress testing
```

These conclusions are hardware-dependent. On GPUs with more VRAM, the 100M preset may behave differently.

---

## Quality Curriculum preset

The current working package includes a Quality Curriculum launcher:

```txt
scripts/run_v89_50m_quality_curriculum_1m.sh
```

This preset is intended to evaluate a mixed data curriculum using the practical 50M model size.

Expected dataset mix:

```txt
DKYoon/SlimPajama-6B: 40%
roneneldan/TinyStories: 60%
```

Intended use:

```txt
50M practical model size
1M-step endurance target
mixed broad-corpus + simpler narrative data
quality-oriented comparison against plain SlimPajama 50M
```

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_50m_quality_curriculum_1m.sh
```

This preset should be treated as an evaluation path unless a completed final evidence bundle is included for the release.

---

## Recommended validation sequence

Initial environment validation:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

python --version
which python
nvidia-smi
```

Validate PyTorch/CUDA:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("cuda matmul: OK", tuple(y.shape))
else:
    raise SystemExit("ERROR: CUDA is not available to PyTorch")
PY
```

Validate required imports:

```bash
python - <<'PY'
mods = [
    "torch",
    "deepspeed",
    "datasets",
    "transformers",
    "tokenizers",
    "openai",
    "yaml",
    "psutil",
    "numpy",
    "tqdm",
]

for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(f"{name}: FAIL -> {type(exc).__name__}: {exc}")
PY
```

Validate MEM scripts:

```bash
chmod +x scripts/*.sh scripts/*.py

python scripts/v89_static_validation.py

bash scripts/validate_mem_v3_configs.sh
```

If available:

```bash
bash scripts/doctor_v89.sh
```

Fallback doctor:

```bash
bash scripts/doctor_mem_v3.sh
```

Recommended run order:

```txt
1. local JSONL smoke test
2. Hugging Face 1000-step test
3. 300k long validation
4. 50M practical 1M-step run
5. Quality Curriculum 1M-step run
6. 100M stress test, if the target GPU has enough tolerance for pressure/churn
```

---

## Run commands

Local JSONL smoke test:

```bash
bash scripts/run_mem_v3.sh --config configs/validation/local_jsonl_100.yaml
```

Hugging Face 1000-step validation:

```bash
bash scripts/run_mem_v3.sh --config configs/validation/huggingface_1000.yaml
```

300k long validation:

```bash
bash scripts/run_mem_v3.sh --config configs/long/huggingface_fineweb_edu_300k.yaml
```

1M long validation:

```bash
bash scripts/run_mem_v3.sh --config configs/long/huggingface_fineweb_edu_1m.yaml
```

Practical 50M SlimPajama 1M-step run:

```bash
bash scripts/run_v89_50m_slimpajama_1m.sh
```

Quality Curriculum 50M 1M-step run:

```bash
bash scripts/run_v89_50m_quality_curriculum_1m.sh
```

100M SlimPajama stress run:

```bash
bash scripts/run_v89_100m_slimpajama_1m.sh
```

---

## Monitoring

Basic human monitor:

```bash
MONITOR_INTERVAL=10 bash scripts/monitor_v89_human.sh
```

Runtime/checkpoint monitor:

```bash
watch -n 10 'bash scripts/monitor_v89_human_runtime.sh'
```

Use the runtime monitor when you want to see:

```txt
current controller status
global progress
active lane
GPU/RAM pressure
latest real checkpoint
checkpoint size
checkpoint age
latest pointer
recent checkpoint files
```

Checkpoint-only monitor:

```bash
watch -n 10 'find checkpoints -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" 2>/dev/null | sort | tail -20'
```

Absolute safe-lane escape monitor:

```bash
watch -n 5 'grep -R -iE "safe_seq256_absolute_escape_armed|safe_seq256_hard_escape_override|safe_seq256_absolute_25k_floor_escape|safe_seq256_hot_optimizer_25k_attention_escape|safe_seq256_low_steps_absolute_escape|lane_switch_applied|noncritical_switch_blocked_by_api|keep_current_lane|fast_seq256_zero0_gacc4|safe_seq256|DONE|target_global_steps_reached" logs evidence 2>/dev/null | tail -150'
```

Expected safe-lane degradation behavior:

```txt
safe_seq256_absolute_escape_armed
safe_seq256_hard_escape_override
lane_switch_applied -> fast_seq256_zero0_gacc4
```

---

## Stop and resume

Stop a running MEM/DeepSpeed job:

```bash
pkill -TERM -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

sleep 8

pkill -KILL -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

ps aux | grep -E "v89_sustained_controller|main.py.*--deepspeed|deepspeed|torchrun" | grep -v grep || echo "OK: no MEM/DeepSpeed process is running"

nvidia-smi
```

Validate latest checkpoint pointer:

```bash
cat checkpoints/v89_latest.txt 2>/dev/null || true

ls -lh "$(cat checkpoints/v89_latest.txt)" 2>/dev/null || true
```

Validate controller state:

```bash
cat checkpoints/v89_controller_state.json 2>/dev/null || true
```

Run resume validator:

```bash
bash scripts/verify_checkpoint_resume_v89.sh
```

Important:

```txt
A logs/evidence ZIP is not necessarily a recovery archive.
A recovery archive must include the real checkpoint tensor file.
The heavy checkpoint file is usually named mem_model_optimizer.pt.
```

---

## Known limitations

```txt
mem_v3 inherits the validated mem_v2 runtime core, but each new dataset/config should still be validated
throughput numbers are local observations, not universal guarantees
different datasets can change data wait, memory pressure, loss behavior and lane stability
Hugging Face datasets may take time to download/cache before training starts
external issues such as WSL, CUDA, driver, API key, internet or dataset access can still affect long runs
RTX 50-series / Blackwell requires compatible PyTorch CUDA builds
100M on 8GB GPUs should be interpreted as a stress test, not the default practical configuration
```

---

## Release boundary

This release should include only the current MEM product line and its evidence.

Do not include:

```txt
historical experiments
comparison packages
separate baselines
temporary benchmark artifacts
old release branches
virtual environments
local caches
large checkpoints
machine-specific absolute paths
API keys
local secrets
dataset caches
```

Before publishing, clean:

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
```

For private recovery archives, checkpoints may be included intentionally.

For public release ZIPs, checkpoints and local caches should be excluded.

---

## Release status

```txt
Product: MEM
Public line: mem_v3
Current release line: mem_v3.3
Internal validated core: v89.0.0
Base release: mem_v2 300k validated package
Release type: configurable workload layer + checkpoint continuity + recovery-lane hardening
Status: 300k validated path available; 1M sustained local run completed
Latest completed endurance evidence: 1000000 / 1000000 global steps, state DONE
Recommended constrained-GPU preset: 50M SlimPajama / 1M steps
Stress-test preset: 100M SlimPajama / 1M steps
Quality evaluation preset: 50M Quality Curriculum / 1M steps
```

---

## Final release claim

The strongest accurate claim for this release is:

```txt
MEM v3 extends the validated v89 sustained-training runtime with configurable workloads,
checkpoint continuity, live rotating checkpoints and recovery-lane hardening.

On the documented local environment, the current line has completed a sustained 1M-step run.
```

Avoid claiming:

```txt
MEM v3 is universally validated for every dataset, GPU, CUDA build and model size.
```

Use the narrower and more accurate positioning:

```txt
MEM v3 is a long-run orchestration layer for language-model training on constrained hardware,
validated through real sustained local runs and designed around survival, observability,
checkpoint continuity and controlled recovery.
```
