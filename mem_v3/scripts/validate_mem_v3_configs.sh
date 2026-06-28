#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIGS=(
  "configs/examples/local_jsonl_dataset.yaml"
  "configs/examples/local_txt_dataset.yaml"
  "configs/examples/huggingface_dataset.yaml"
  "configs/validation/local_jsonl_100.yaml"
  "configs/validation/local_txt_100.yaml"
  "configs/validation/huggingface_1000.yaml"
  "configs/examples/huggingface_long_300k.yaml"
  "configs/long/local_jsonl_300k.yaml"
  "configs/long/local_txt_300k.yaml"
  "configs/long/huggingface_fineweb_edu_300k.yaml"
  "configs/mem_v3_default.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "CONFIG: $cfg"
  python scripts/translate_mem_v3_config.py "$cfg" | grep -E 'MEM_RUN_NAME|MEM_TARGET_GLOBAL_STEPS|MEM_DATASET_TYPE|MEM_DATASET_NAME|MEM_DATASET_DATA_FILES|MEM_DATASET_MIX|MEM_DATASET_PREFETCH_BATCHES|MEM_DATASET_PREWARM_BATCHES'
  echo "OK"
done

echo "============================================================"
echo "ALL MEM v3 CONFIGS TRANSLATED OK"
