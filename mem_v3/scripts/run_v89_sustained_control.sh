#!/usr/bin/env bash
set -euo pipefail
export MEM_V89_FORCE_GRAD_ACCUM=${MEM_V89_FORCE_GRAD_ACCUM:-4}
export MEM_V89_GRADIENT_ACCUMULATION_STEPS=${MEM_V89_GRADIENT_ACCUMULATION_STEPS:-4}
export MEM_V89_LR_SCHEDULE=${MEM_V89_LR_SCHEDULE:-1}
export MEM_V89_BASE_LR_REAL=${MEM_V89_BASE_LR_REAL:-5.0e-5}
export MEM_V89_LR_PEAK_CAP=${MEM_V89_LR_PEAK_CAP:-5.0e-5}
export MEM_V89_LR_WARMUP_STEPS=${MEM_V89_LR_WARMUP_STEPS:-5000}
export MEM_V89_LR_MIN_MULT=${MEM_V89_LR_MIN_MULT:-0.05}
export MEM_V89_LR_DECAY_STEPS=${MEM_V89_LR_DECAY_STEPS:-1000000}
export MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH=${MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH:-50000}
cd "$(dirname "$0")/.."
if [ -f .venv312/bin/activate ]; then
  source .venv312/bin/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  export VIRTUAL_ENV="${CONDA_PREFIX}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
  echo "ERRO: ambiente Python não encontrado (.venv312, .venv ou CONDA_PREFIX ativo)." >&2
  echo "Dica WSL/Conda: rode bash scripts/setup_wsl_conda_v89.sh" >&2
  exit 2
fi
mkdir -p logs evidence chaos_tmp evidence_packets reports dataset_cache checkpoints
export MEM_RUN_ID=${MEM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export MEM_CHECKPOINT_LABEL=${MEM_CHECKPOINT_LABEL:-v89_${MEM_RUN_ID}}
export MEM_AUTO_RESUME_CHECKPOINT=${MEM_AUTO_RESUME_CHECKPOINT:-1}
source scripts/_deepspeed_env.sh
export MASTER_PORT=${MASTER_PORT:-30079}
export MEM_V89_ENABLE_API=${MEM_V89_ENABLE_API:-1}
# Default OFF for RTX 50 / sm_120 clean starts. torch.compile can spend
# minutes in inductor workers before the first heartbeat and can look like
# a dead runtime. Enable manually only after the baseline is stable.
if [ "${MEM_ALLOW_TORCH_COMPILE:-0}" != "1" ]; then
  export MEM_TORCH_COMPILE_EXPERIMENTAL=0
else
  export MEM_TORCH_COMPILE_EXPERIMENTAL=${MEM_TORCH_COMPILE_EXPERIMENTAL:-1}
fi
export MEM_TORCH_COMPILE_MODE=${MEM_TORCH_COMPILE_MODE:-reduce-overhead}

# Keep first-progress latency bounded on empty dataset_cache. The batcher will
# keep streaming/cache-filling during training; prewarming 16 batches can stall
# startup on HF/Xet timeouts.
export MEM_DATASET_PREFETCH_BATCHES=${MEM_DATASET_PREFETCH_BATCHES:-2}
export MEM_DATASET_PREWARM_BATCHES=${MEM_DATASET_PREWARM_BATCHES:-1}
export MEM_DATASET_CACHE_MODE=${MEM_DATASET_CACHE_MODE:-memmap}
export MEM_DATASET_CACHE_DIR=${MEM_DATASET_CACHE_DIR:-dataset_cache}

# Fail fast before the controller starts. Without this, a missing/incompatible
# dependency can cause repeated lane restarts while the monitor only shows
# "AGUARDANDO RUNTIME".
python - <<'PY'
import sys
missing = []
for mod in ("torch", "deepspeed", "datasets", "transformers"):
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    print("ERRO: ambiente MEM v89 incompleto:", file=sys.stderr)
    for item in missing:
        print(" - " + item, file=sys.stderr)
    print("Rode: bash scripts/setup_mem312_rtx50_cu128.sh", file=sys.stderr)
    raise SystemExit(2)
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    cuda = getattr(torch.version, "cuda", None) or "unknown"
    print(f"MEM preflight: torch={torch.__version__} cuda={cuda} gpu={torch.cuda.get_device_name(0)} cap={cap}")
    if cap >= (12, 0) and not (str(cuda).startswith("12.8") or str(cuda).startswith("12.9") or str(cuda).startswith("13")):
        print("ERRO: GPU sm_120/RTX 50 detectada, mas este PyTorch não é cu128+.", file=sys.stderr)
        print("Rode: bash scripts/setup_mem312_rtx50_cu128.sh", file=sys.stderr)
        raise SystemExit(3)
else:
    print("ERRO: torch.cuda.is_available() = False", file=sys.stderr)
    raise SystemExit(4)
PY

CONTROLLER_ARGS=(
  --start-lane ${MEM_START_LANE:-aggressive_seq256_zero0_gacc4}
  --target-steps ${MEM_TARGET_STEPS:-300000}
  --sample-seconds 20
  --min-degrade-step 3000
  --min-lane-steps-before-switch ${MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH:-50000}
  --min-recovery-hold-steps 12000
  --drop-ratio 0.68
  --bad-windows 4
  --midband-protect-min-tokens 18000
  --midband-protect-max-tokens 23000
  --midband-required-bad-windows 4
  --midband-required-optimizer-ratio 0.45
  --health-good-tokens 19000
  --health-acceptable-tokens 17500
  --health-attention-tokens 16000
  --health-attention-optimizer-ratio 0.45
  --health-required-bad-windows 4
  --same-lane-refresh-max-attempts 2
  --same-lane-refresh-eval-steps 5000
  --stall-seconds 120
  --hard-stall-seconds 240
  --fresh-progress-grace-seconds 240
  --first-progress-hard-seconds 600
  --first-progress-min-step 1
  --min-gpu-util 35
  --min-steps-per-second 4.0
  --max-swap-mib 256
  --high-vram-mib 7600
  --max-optimizer-ratio 0.42
  --force-zero0-optimizer-ratio 0.48
  --data-wait-ratio-high 0.22
  --data-wait-ratio-severe 0.40
  --proactive-min-tokens 17000
  --slow-degradation-ratio 0.84
  --slow-degradation-optimizer-ratio 0.38
  --safe-promotion-min-lane-step 2000
  --safe-promotion-min-tokens 17000
  --safe-promotion-min-steps-per-second 6.0
  --safe-promotion-max-optimizer-ratio 0.42
  --safe-promotion-max-data-wait-ratio 0.30
  --safe-promotion-max-gpu-util 70
  --hard-override-min-step 3500
  --hard-override-tokens 13000
  --hard-override-optimizer-ratio 0.52
  --hard-override-score -0.75
  --hard-override-steps-per-second 5.0
  --hard-override-gpu-util 25
  --primary-hard-override-tokens 15000
  --primary-hard-override-optimizer-ratio 0.50
  --no-chaos
)

append_supervisor_event() {
  local event="$1"; shift || true
  python - "$event" "$@" <<'PY' || true
import json, sys, time, pathlib, os
out=pathlib.Path('evidence_packets/v89_sustained_control_events.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
payload={"event": sys.argv[1], "ts": round(time.time(),3), "version":"v89.0.0", "pid": os.getpid()}
for i,arg in enumerate(sys.argv[2:], start=1):
    payload[f"arg{i}"]=arg
with out.open('a', encoding='utf-8') as f:
    f.write(json.dumps(payload, ensure_ascii=False)+"\n")
PY
}

status_state() {
  python - <<'PY' 2>/dev/null || true
import json, pathlib
p=pathlib.Path('evidence/v89_controller_status_latest.json')
if p.exists():
    print((json.loads(p.read_text()).get('state') or '').upper())
PY
}

live_main_pids() {
  pgrep -f "main.py.*--deepspeed" 2>/dev/null || true
}

kill_live_lanes() {
  pkill -TERM -f "main.py.*--deepspeed" 2>/dev/null || true
  pkill -TERM -f "deepspeed" 2>/dev/null || true
  pkill -TERM -f "torchrun" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "main.py.*--deepspeed" 2>/dev/null || true
  pkill -KILL -f "deepspeed" 2>/dev/null || true
  pkill -KILL -f "torchrun" 2>/dev/null || true
}

supervisor_shutdown() {
  rc=${1:-130}
  append_supervisor_event "controller_supervisor_interrupted" "$rc"
  kill_live_lanes
  exit "$rc"
}

trap 'supervisor_shutdown 130' INT
trap 'supervisor_shutdown 143' TERM

# P0.2 handoff guard: if the Python controller exits while a training lane is
# still alive, do not leave an un-orchestrated orphan run. Kill the orphan lane
# and immediately restart the controller, which will auto-resume from the live
# rotating checkpoint for MEM_CHECKPOINT_LABEL.
max_supervisor_restarts=${MEM_CONTROLLER_SUPERVISOR_MAX_RESTARTS:-12}
supervisor_restart=0
while true; do
  append_supervisor_event "controller_supervisor_start" "$supervisor_restart"
  set +e
  python scripts/v89_sustained_controller.py "${CONTROLLER_ARGS[@]}"
  rc=$?
  set -e

  state="$(status_state)"
  live="$(live_main_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"

  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ] || [ "$state" = "CONTROLLER_INTERRUPTED" ]; then
    append_supervisor_event "controller_supervisor_operator_stop" "$rc" "$live" "$state"
    if [ -n "$live" ]; then
      kill_live_lanes
    fi
    exit "$rc"
  fi

  if [ "$state" = "DONE" ]; then
    append_supervisor_event "controller_supervisor_done" "$rc"
    exit "$rc"
  fi

  if [ -n "$live" ]; then
    append_supervisor_event "controller_exited_with_live_lane" "$rc" "$live" "$supervisor_restart"
    kill_live_lanes
    supervisor_restart=$((supervisor_restart + 1))
    if [ "$supervisor_restart" -gt "$max_supervisor_restarts" ]; then
      append_supervisor_event "controller_supervisor_restart_limit_reached" "$rc" "$live"
      exit 99
    fi
    sleep 3
    continue
  fi

  # Non-zero controller exit without a live lane can still be recovered if a
  # live checkpoint exists. Limit retries to avoid silent loops.
  if [ "$rc" -ne 0 ] && ls checkpoints/${MEM_CHECKPOINT_LABEL}_*/mem_model_optimizer.pt >/dev/null 2>&1; then
    append_supervisor_event "controller_exited_without_live_lane_retrying_from_checkpoint" "$rc" "$supervisor_restart"
    supervisor_restart=$((supervisor_restart + 1))
    if [ "$supervisor_restart" -gt "$max_supervisor_restarts" ]; then
      append_supervisor_event "controller_supervisor_restart_limit_reached" "$rc" "no_live_lane"
      exit 99
    fi
    sleep 3
    continue
  fi

  append_supervisor_event "controller_supervisor_exit" "$rc" "$state"
  exit "$rc"
done
