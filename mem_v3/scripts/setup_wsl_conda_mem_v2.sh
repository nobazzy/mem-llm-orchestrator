#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT_NAME="mem_v2"
ENV_NAME="${MEM_V2_CONDA_ENV:-mem_v2_py312}"
PYTHON_VERSION="${MEM_V2_PYTHON_VERSION:-3.12.13}"
TORCH_INDEX="${MEM_V2_TORCH_INDEX:-https://download.pytorch.org/whl/nightly/cu128}"

load_conda() {
  if [ -n "${CONDA_EXE:-}" ]; then
    local conda_root
    conda_root="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    if [ -f "$conda_root/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "$conda_root/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  for f in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$f" ]; then
      # shellcheck disable=SC1090
      source "$f"
      return 0
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    local hook
    hook="$(conda shell.bash hook 2>/dev/null || true)"
    if [ -n "$hook" ]; then
      eval "$hook"
      return 0
    fi
  fi

  echo "ERRO: Conda/Miniforge não encontrado ou não inicializável." >&2
  echo "Instale Miniforge ou confirme se ~/miniforge3 existe." >&2
  exit 2
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

create_or_reuse_env() {
  if conda_env_exists; then
    echo "Ambiente conda já existe: $ENV_NAME"
    conda activate "$ENV_NAME"
    local pyver
    pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
    if [[ "$pyver" != 3.12.* ]]; then
      echo "Ambiente $ENV_NAME existe, mas está com Python $pyver. Recriando com $PYTHON_VERSION."
      conda deactivate || true
      conda remove -y -n "$ENV_NAME" --all
      conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
      conda activate "$ENV_NAME"
    fi
  else
    echo "Criando ambiente conda: $ENV_NAME / Python $PYTHON_VERSION"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    conda activate "$ENV_NAME"
  fi
}

make_venv_shim() {
  echo "Criando compatibilidade .venv312/.venv para o launcher"
  rm -rf .venv .venv312
  mkdir -p .venv312/bin
  ln -s "$CONDA_PREFIX/bin/python" .venv312/bin/python
  ln -s "$CONDA_PREFIX/bin/pip" .venv312/bin/pip
  cat > .venv312/bin/activate <<ACTIVATE_EOF
export CONDA_PREFIX="$CONDA_PREFIX"
export VIRTUAL_ENV="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:\$PATH"
ACTIVATE_EOF
  chmod +x .venv312/bin/activate
  ln -s .venv312 .venv
}

echo "=== MEM v2 - WSL/Conda setup ==="
echo "Projeto: $ROOT"

load_conda
create_or_reuse_env

echo "Python ativo: $(python --version)"
python -m pip install --upgrade pip setuptools wheel

echo "Instalando PyTorch: somente torch nightly cu128"
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip cache purge || true
python -m pip install --pre torch --index-url "$TORCH_INDEX"

echo "Instalando dependências da MEM v2"
python -m pip install \
  transformers \
  datasets \
  accelerate \
  huggingface_hub \
  openai \
  numpy \
  psutil \
  pydantic \
  tqdm \
  requests \
  pyyaml \
  deepspeed \
  pytest

make_venv_shim

echo "Validando CUDA/PyTorch"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda matmul: OK", tuple(y.shape))
else:
    raise SystemExit("ERRO: CUDA não está disponível para o PyTorch.")
PY

echo "Validando release"
python scripts/v89_static_validation.py
python -m compileall -q .
pytest -q

echo "=== SETUP CONCLUÍDO ==="
echo "Para rodar:"
echo "  cd $ROOT"
echo "  source .venv312/bin/activate"
echo "  export OPENAI_API_KEY='SUA_CHAVE_NOVA_AQUI'"
echo "  export API_KEY=\"\$OPENAI_API_KEY\""
echo "  bash scripts/run_v89_sustained_control.sh"
