#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pick_python() {
  if [ -x .venv312/bin/python ]; then
    echo "$ROOT/.venv312/bin/python"
  elif [ -x .venv/bin/python ]; then
    echo "$ROOT/.venv/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    echo "$CONDA_PREFIX/bin/python"
  else
    command -v python || true
  fi
}

PY="$(pick_python)"

run_py() {
  if [ -n "$PY" ]; then
    "$PY" "$@"
  else
    echo "Python não encontrado" >&2
    return 127
  fi
}

echo "STATUS"
echo "  Projeto: $ROOT"
echo "  Data: $(date)"
echo

echo "AMBIENTE"
echo "  CONDA_PREFIX: ${CONDA_PREFIX:-EMPTY}"
echo "  VIRTUAL_ENV: ${VIRTUAL_ENV:-EMPTY}"
echo "  Python escolhido: ${PY:-EMPTY}"
if [ -n "$PY" ]; then "$PY" --version 2>/dev/null || true; fi
echo

echo "VENVS"
[ -e .venv ] && ls -lah .venv || echo "  .venv: ausente"
[ -e .venv312 ] && ls -lah .venv312 || echo "  .venv312: ausente"
[ -x .venv312/bin/python ] && echo "  .venv312/bin/python: OK" || echo "  .venv312/bin/python: ausente"
[ -f .venv312/bin/activate ] && echo "  .venv312/bin/activate: OK" || echo "  .venv312/bin/activate: ausente"
echo

echo "PYTORCH/CUDA"
run_py - <<'PY' || true
try:
    import torch
    print("  torch:", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  cuda:", torch.version.cuda)
        print("  gpu:", torch.cuda.get_device_name(0))
        print("  capability:", torch.cuda.get_device_capability(0))
        x = torch.randn(512, 512, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("  cuda matmul: OK", tuple(y.shape))
except Exception as e:
    print("  torch FAIL:", e)
PY
echo

echo "IMPORTS"
run_py - <<'PY' || true
mods = ["torch", "deepspeed", "transformers", "datasets", "accelerate", "huggingface_hub", "openai", "numpy", "psutil", "yaml", "pytest"]
for m in mods:
    try:
        __import__(m)
        print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: FAIL -> {e}")
PY
echo

echo "API"
run_py - <<'PY' || true
import os
key = os.getenv("OPENAI_API_KEY", "")
print("  OPENAI_API_KEY loaded:", bool(key))
print("  prefix:", key[:7] + "..." if key else "EMPTY")
PY
echo

echo "SCRIPTS"
for f in \
  scripts/run_v89_sustained_control.sh \
  scripts/monitor_v89_human.sh \
  scripts/v89_static_validation.py \
  scripts/setup_wsl_conda_mem_v2.sh \
  scripts/setup_wsl_conda_v89.sh; do
  [ -f "$f" ] && echo "  $f: OK" || echo "  $f: FALTANDO"
done

echo

echo "VALIDACAO ESTATICA"
if [ -n "$PY" ] && [ -f scripts/v89_static_validation.py ]; then
  run_py scripts/v89_static_validation.py || true
else
  echo "  não executada"
fi
