#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pkill -f "scripts/v89_sustained_controller.py" 2>/dev/null || true
pkill -f "v89_sustained_controller.py" 2>/dev/null || true
pkill -f "run_v89_75m_slimpajama_1m.sh" 2>/dev/null || true
pkill -f "main.py" 2>/dev/null || true
pkill -f "deepspeed" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
pkill -f "runtime/deepspeed_runner.py" 2>/dev/null || true
pkill -f "torch/_inductor/compile_worker" 2>/dev/null || true
sleep 3

rm -rf checkpoints evidence evidence_packets logs reports chaos_tmp dataset_cache
mkdir -p checkpoints evidence evidence_packets logs reports chaos_tmp dataset_cache

echo "OK: MEM v89 limpo. Checkpoints/evidence/logs/reports/chaos_tmp/dataset_cache recriados vazios."
