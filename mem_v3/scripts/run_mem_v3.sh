#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/mem_v3_default.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?missing config path}"
      shift 2
      ;;
    --target-steps)
      EXTRA_ARGS+=("--target-steps" "${2:?missing target steps}")
      shift 2
      ;;
    --start-lane)
      EXTRA_ARGS+=("--start-lane" "${2:?missing start lane}")
      shift 2
      ;;
    *)
      echo "ERRO: argumento desconhecido: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "ERRO: config não encontrado: $CONFIG" >&2
  exit 1
fi

if [[ -f .venv312/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv312/bin/activate
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  export VIRTUAL_ENV="$CONDA_PREFIX"
  export PATH="$CONDA_PREFIX/bin:$PATH"
else
  echo "ERRO: ambiente Python não encontrado. Rode: bash scripts/setup_wsl_conda_mem_v2.sh" >&2
  exit 1
fi

MAP_FILE="$(mktemp)"
python scripts/translate_mem_v3_config.py "$CONFIG" > "$MAP_FILE"
# shellcheck disable=SC1090
source "$MAP_FILE"
rm -f "$MAP_FILE"

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  EXTRA_ARGS=("--target-steps" "$MEM_TARGET_GLOBAL_STEPS" "--start-lane" "$MEM_START_LANE")
fi

echo "=== MEM v3 configurable run ==="
echo "Config: $CONFIG"
echo "Run name: ${MEM_RUN_NAME:-unknown}"
echo "Dataset: ${MEM_DATASET_NAME:-unknown}"
echo "Data files: ${MEM_DATASET_DATA_FILES:-none}"
echo "Target steps: ${EXTRA_ARGS[*]}"

python scripts/v89_sustained_controller.py "${EXTRA_ARGS[@]}"
