## Quickstart — WSL2, Conda, CUDA and MEM runtime

This is the recommended local workflow for the current MEM v3 package on WSL2 with an NVIDIA GPU.

The validated local workflow uses:

```txt
OS: WSL2 Ubuntu
Environment manager: Miniforge / Conda
Conda env: mem312
Python: 3.12.13
GPU: NVIDIA RTX 50-series / Blackwell class
CUDA runtime target: cu128-compatible PyTorch build
DeepSpeed: enabled
Workload type: language-model training
```

MEM v3 is a **language-model training workload**. It does not require `torchvision`, `torchaudio`, image datasets, audio datasets or vision/audio operators.

Do not install:

```txt
torchvision
torchaudio
torchtext
```

Install only `torch` plus the packages required for language-model training, controller execution, datasets, tokenization, telemetry and DeepSpeed.

---

## 1. Extract the package

Extract the release ZIP and enter the project directory:

```bash
unzip mem_v3.zip

cd mem_v3
```

If the package is already extracted:

```bash
cd mem_v3
```

---

## 2. Create or activate the Conda environment

Use the validated environment name:

```bash
source ~/miniforge3/etc/profile.d/conda.sh

conda create -n mem312 python=3.12.13 -y

conda activate mem312

python --version
which python
```

Expected:

```txt
Python 3.12.13
```

If the environment already exists:

```bash
source ~/miniforge3/etc/profile.d/conda.sh

conda activate mem312

python --version
which python
```

---

## 3. Install base Python dependencies

Upgrade the Python packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install the general MEM runtime dependencies:

```bash
python -m pip install --upgrade \
  numpy \
  pyyaml \
  psutil \
  tqdm \
  requests \
  openai \
  datasets \
  transformers \
  tokenizers \
  accelerate \
  safetensors \
  pytest
```

---

## 4. Install PyTorch with CUDA

For RTX 50-series / Blackwell GPUs, use a CUDA 12.8-compatible PyTorch build.

Install **torch only**:

```bash
python -m pip uninstall -y torch torchvision torchaudio torchtext

python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Do not use the generic PyTorch command that installs `torchvision` and `torchaudio`, because MEM v3 does not need vision or audio packages.

Correct for MEM:

```bash
python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Not needed for MEM:

```bash
python -m pip install torch torchvision torchaudio
```

Validate PyTorch and CUDA:

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

Expected high-level result:

```txt
cuda available: True
cuda matmul: OK
```

---

## 5. Install DeepSpeed

Install DeepSpeed after PyTorch is working:

```bash
DS_BUILD_OPS=0 python -m pip install --upgrade deepspeed
```

Validate import:

```bash
python - <<'PY'
import deepspeed
print("deepspeed:", deepspeed.__version__)
PY
```

MEM v3 does not require DeepSpeed custom CUDA ops to be compiled for the validated local workflow.

---

## 6. Validate required imports

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

---

## 7. Validate the MEM package

From the project root:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

chmod +x scripts/*.sh scripts/*.py

python scripts/v89_static_validation.py

bash scripts/doctor_v89.sh

bash scripts/validate_mem_v3_configs.sh
```

Expected high-level result:

```txt
V89_STATIC_VALIDATION: PASS
torch: OK
cuda available: True
deepspeed: OK
```

If `doctor_v89.sh` is not present in a specific release package, use the available doctor script:

```bash
bash scripts/doctor_mem_v3.sh
```

---

## 8. Configure the OpenAI API key

MEM v3 uses the controller-assisted API path.

Required variable:

```txt
OPENAI_API_KEY
```

Set the API key in the same terminal that will run MEM:

```bash
unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY loaded"
```

Do not print the full key.

Safe validation:

```bash
python - <<'PY'
import os

key = os.getenv("OPENAI_API_KEY", "")

print("OPENAI_API_KEY loaded:", bool(key))
print("prefix:", key[:7] + "..." if key else "EMPTY")
PY
```

Never commit API keys, tokens or local secrets to the repository.

---

## 9. Run a local JSONL smoke test

This is the recommended first validation for a new environment or new package extraction.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

bash scripts/run_mem_v3.sh --config configs/validation/local_jsonl_100.yaml
```

This validates:

```txt
config translation
local JSONL dataset path
launcher compatibility
controller startup
runtime wiring
```

---

## 10. Run a Hugging Face 1000-step validation

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

bash scripts/run_mem_v3.sh --config configs/validation/huggingface_1000.yaml
```

The first Hugging Face run may spend time downloading, caching and preparing the dataset before training steps appear.

---

## 11. Run the practical 50M SlimPajama 1M-step preset

For constrained 8 GB GPUs, this is the practical long-run preset.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_50m_slimpajama_1m.sh
```

---

## 12. Run the Quality Curriculum preset

Quality Curriculum uses a mixed Hugging Face stream intended to combine broader corpus exposure with easier narrative text.

Expected dataset mix:

```txt
DKYoon/SlimPajama-6B: 40%
roneneldan/TinyStories: 60%
```

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY loaded"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_50m_quality_curriculum_1m.sh
```

---

## 13. Run the 100M SlimPajama stress preset

The 100M preset is a stress test for constrained 8 GB GPUs.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_100m_slimpajama_1m.sh
```

Recommended interpretation:

```txt
50M preset:
  practical 8 GB run
  better stability/throughput balance

100M preset:
  stress test
  useful for robustness evidence
  expected to show higher VRAM pressure
```

---

## 14. Monitor runtime and checkpoints

Basic human monitor:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

MONITOR_INTERVAL=10 bash scripts/monitor_v89_human.sh
```

Runtime/checkpoint monitor:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

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

Checkpoint-only watch:

```bash
watch -n 10 'find checkpoints -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" 2>/dev/null | sort | tail -20'
```

---

## 15. Stop a running MEM/DeepSpeed job

Use this when you want to stop the current run and resume later from checkpoint.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

echo "Stopping controller/training with TERM..."
pkill -TERM -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

sleep 8

echo "Killing remaining MEM/DeepSpeed processes..."
pkill -KILL -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

echo
echo "Remaining processes:"
ps aux | grep -E "v89_sustained_controller|main.py.*--deepspeed|deepspeed|torchrun" | grep -v grep || echo "OK: no MEM/DeepSpeed process is running"

echo
echo "GPU:"
nvidia-smi
```

---

## 16. Validate checkpoint resume

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

echo "Latest checkpoint pointer:"
cat checkpoints/v89_latest.txt 2>/dev/null || true

echo
echo "Latest checkpoint file:"
ls -lh "$(cat checkpoints/v89_latest.txt)" 2>/dev/null || true

echo
echo "Controller state:"
cat checkpoints/v89_controller_state.json 2>/dev/null || true

echo
echo "Resume validator:"
bash scripts/verify_checkpoint_resume_v89.sh
```

The final logs bundle may contain checkpoint metadata, but not necessarily the heavy checkpoint tensor file:

```txt
mem_model_optimizer.pt
```

To resume training, the real checkpoint file must exist locally under:

```txt
checkpoints/
```

---

## 17. Save a snapshot ZIP

Snapshots should be saved from the parent project directory, not inside `mem_v3`.

```bash
cd ..

TS="$(date +%Y%m%d_%H%M%S)"

zip -r "mem_v3_snapshot_${TS}.zip" \
  mem_v3/evidence \
  mem_v3/evidence_packets \
  mem_v3/reports \
  mem_v3/logs \
  mem_v3/checkpoints \
  mem_v3/README.md \
  mem_v3/EVALUATION.md \
  mem_v3/VERSION_CURRENT.txt \
  mem_v3/PATCH_APPLIED_NOTES.md \
  mem_v3/RECOVERY_NOTES_20260624.md

echo "Snapshot saved:"
ls -lh "mem_v3_snapshot_${TS}.zip"
```

For public release packages, do not include:

```txt
API keys
local secrets
dataset caches
virtual environments
large checkpoint tensor files
machine-specific temporary files
```

For private recovery archives, checkpoints may be included intentionally.

---

## 18. Current validated evidence: 1M sustained run

A completed v89 sustained training run reached the full target:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
controller core: v89.0.0
package line: mem_v3.3 safe-recovery-50m-preset
```

Runtime profile:

```txt
environment: WSL2 Ubuntu
python: 3.12.13
torch cuda target: cu128-compatible build
gpu: NVIDIA RTX 50-series / 8 GB class
precision: fp16
DeepSpeed ZeRO stage: 0
sequence length: 256
tokenizer: gpt2
model preset: medium_50m
gradient accumulation: 4
```

Checkpoint metadata:

```txt
latest pointer: checkpoints/v89_live_02/mem_model_optimizer.pt
metadata present: yes
heavy checkpoint file included in final logs ZIP: no
```

The final logs bundle records controller state, reports, evidence and checkpoint metadata. It does not replace the real local checkpoint file.

---

## 19. Troubleshooting

### CUDA is not available

Check WSL GPU visibility:

```bash
nvidia-smi
```

Check PyTorch CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

If CUDA is false, reinstall only `torch` from a CUDA-compatible index. Do not add `torchvision` or `torchaudio`.

### Wrong Conda environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

which python
python --version
```

### Missing monitor script

List available monitors:

```bash
find scripts -maxdepth 1 -type f -iname "*monitor*" -print | sort
```

### Missing checkpoint

```bash
find checkpoints -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" 2>/dev/null | sort | tail -20

cat checkpoints/v89_latest.txt 2>/dev/null || true
```

### Kill stuck run

```bash
pkill -TERM -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true
sleep 8
pkill -KILL -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true
nvidia-smi
```

---

## 20. Dependency policy

For this MEM v3 package:

```txt
Install:
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

Do not install:
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

---

## Release status

```txt
Product: MEM
Public line: mem_v3
Internal validated core: v89.0.0
Base release: mem_v2 300k validated package
Version: mem_v3.3 safe-recovery-50m-preset
New capabilities: configurable workloads, custom datasets, live rotating checkpoints, checkpoint resume continuity and recovery-lane hardening
Status: 300k validated path available; 1M sustained local run completed
Latest completed endurance evidence: 1000000 / 1000000 global steps, state DONE
```
