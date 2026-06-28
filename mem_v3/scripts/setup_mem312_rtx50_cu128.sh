#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Local conda env for MEM v89/v3 on RTX 50 / Blackwell (sm_120).
# Uses PyTorch CUDA 12.8 nightly/pre-release when stable wheels do not support sm_120 yet.
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERRO: conda/miniforge não encontrado." >&2
    exit 2
  fi
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda env list | awk '{print $1}' | grep -qx "mem312"; then
  conda create -y -n mem312 python=3.12.13 || conda create -y -n mem312 python=3.12
fi
conda activate mem312
hash -r

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip uninstall -y torch torchvision torchaudio triton deepspeed || true

# RTX 50/sm_120 needs cu128+ builds. Nightly is intentional here.
# MEM is a language-model workload, so torchvision/torchaudio are not installed.
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Project deps. Torch is installed above, so requirements intentionally avoids pinning torch.
python -m pip install -r requirements.txt

# DeepSpeed without JIT/fused CUDA ops; MEM uses external torch AdamW path.
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
export DS_SKIP_CUDA_CHECK=1
python -m pip install "deepspeed==0.19.1" || python -m pip install deepspeed

python - <<'PY'
import sys
import torch
import deepspeed
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERRO: CUDA indisponível")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.ones((512, 512), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("cuda matmul ok:", float(y.mean().item()))
print("deepspeed:", deepspeed.__version__)
PY
