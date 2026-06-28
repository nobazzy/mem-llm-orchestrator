#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

hr(){ echo "============================================================"; }

latest_run_dir() {
  find evidence -maxdepth 1 -type d -name 'v89_wsl_deepspeed_*' 2>/dev/null | sort | tail -1
}

latest_adaptive_line() {
  local run="$1"
  local f="$run/adaptive_memory.jsonl"
  [ -f "$f" ] || return 1
  tail -1 "$f"
}

print_main() {
  local run="$1"
  local line="${2:-}"

  echo "RUN"
  echo "  Ativa:         ${run:-NENHUMA}"
  echo

  python - "$line" <<'PY'
import sys, json, time, subprocess, os, re
from pathlib import Path

raw = sys.argv[1]
latest = {}
if raw.strip():
    try:
        latest = json.loads(raw)
    except Exception:
        latest = {}

fmt = lambda x: f"{int(x):,}".replace(",", ".")

# =========================
# GLOBAL CONTROLLER SOURCE
# =========================
ctrl_step = 0
ctrl_target = 1_000_000
ctrl_restart = "?"
ctrl_lane = "?"
ctrl_reason = "?"
ctrl_state = "?"
ctrl_pid = "?"
ctrl_age_progress = "?"
ctrl_age_status = "?"

status_path = Path("evidence/v89_controller_status_latest.json")
global_path = Path("evidence/v89_controller_global_progress_latest.json")

for path in [global_path, status_path]:
    try:
        if path.exists():
            d = json.load(open(path, "r", encoding="utf-8"))
            ctrl_step = max(ctrl_step, int(d.get("global_step") or 0))
            ctrl_target = int(d.get("target_global_steps") or ctrl_target)
            ctrl_restart = d.get("restart_index", ctrl_restart)
            ctrl_lane = d.get("lane", ctrl_lane)
            ctrl_reason = d.get("reason", ctrl_reason)
            if path == status_path:
                ctrl_state = d.get("state", ctrl_state)
                ctrl_pid = d.get("lane_pid", ctrl_pid)
                ctrl_age_progress = d.get("progress_age_seconds", ctrl_age_progress)
                ts = d.get("ts")
                if ts:
                    ctrl_age_status = f"{time.time() - float(ts):.1f}s"
    except Exception:
        pass

# =========================
# EVIDENCE SUM SOURCE
# =========================
evidence_sum = 0
evidence_runs = []

for f in sorted(Path("evidence").glob("v89_wsl_deepspeed_*/adaptive_memory.jsonl")):
    max_step = 0
    lines = 0
    last_ts = 0.0
    try:
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            try:
                x = json.loads(ln)
            except Exception:
                continue
            lines += 1
            st = int(x.get("step") or 0)
            max_step = max(max_step, st)
            last_ts = max(last_ts, float(x.get("ts") or 0))
        if max_step > 0:
            evidence_sum += max_step
            evidence_runs.append((f.parent.name, max_step, lines, last_ts))
    except Exception:
        pass

# =========================
# CHECKPOINT SOURCE
# =========================
ckpt_step = 0
for mf in Path("checkpoints").glob("v89_live_*/metadata.json"):
    try:
        md = json.load(open(mf, "r", encoding="utf-8"))
        ckpt_step = max(
            ckpt_step,
            int(md.get("global_step") or 0),
            int(md.get("micro_train_steps_completed") or 0),
            int((md.get("adaptive_memory") or {}).get("last_effect", {}).get("step") or 0),
        )
    except Exception:
        pass

lane_step = int(latest.get("step") or 0)
tokens_proc = int(latest.get("tokens_processed") or 0)

# ESTE É O GLOBAL CORRETO:
# usa o maior entre controller global, soma real dos evidence runs e checkpoint.
effective_step = max(ctrl_step, evidence_sum, ckpt_step)
if effective_step <= 0:
    effective_step = lane_step

remaining = max(0, ctrl_target - effective_step)
pct = (effective_step / ctrl_target * 100.0) if ctrl_target else 0.0

loss = latest.get("loss")
tok = latest.get("tokens_per_second")
sps = latest.get("steps_per_second")
lr = latest.get("lr")
clip = latest.get("gradient_clip_norm")
effect = latest.get("effectiveness", "?")
ts = float(latest.get("ts") or 0)
age = time.time() - ts if ts else -1

if not latest:
    status = "AGUARDANDO RUNTIME"
elif 0 <= age < 90:
    status = "RODANDO"
elif 90 <= age < 300:
    status = "SEM ATUALIZAÇÃO RECENTE"
else:
    status = "PROGRESSO ANTIGO"

print("STATUS")
print(f"  Estado:        {status}")
print("  Fonte:         adaptive_memory.jsonl + evidence acumulado + controller global")
if age >= 0:
    print(f"  Última escrita:{age:.1f}s atrás")

print()
print("PROGRESSO")
print(f"  Step efetivo:  {fmt(effective_step)} / {fmt(ctrl_target)} ({pct:.3f}%)")
print(f"  Restantes:     {fmt(remaining)} steps")
print(f"  Step lane:     {fmt(lane_step)}")
print(f"  Step evidence: {fmt(evidence_sum)}")
print(f"  Step ckpt:     {fmt(ckpt_step)}")
print(f"  Step ctrl.:    {fmt(ctrl_step)}")
print(f"  Runs evidence: {len(evidence_runs)}")
print(f"  Tokens proc.:  {fmt(tokens_proc)}")

print()
print("TREINO")
print(f"  Loss:          {loss:.4f}" if isinstance(loss, (int, float)) else f"  Loss:          {loss}")
print(f"  Tokens/s:      {tok:.3f}" if isinstance(tok, (int, float)) else f"  Tokens/s:      {tok}")
print(f"  Steps/s:       {sps:.3f}" if isinstance(sps, (int, float)) else f"  Steps/s:       {sps}")
print(f"  LR:            {lr}")
print(f"  Grad clip:     {clip}")
print(f"  Tendência:     {effect}")

print()
print("GPU")
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).strip().splitlines()[0]
    a = [x.strip() for x in out.split(",")]
    print(f"  Utilização:    {a[0]}%")
    print(f"  VRAM:          {a[1]} MiB / {a[2]} MiB")
    print(f"  Temp:          {a[3]}°C")
    print(f"  Power:         {a[4]} W")
except Exception as e:
    print(f"  Status:        erro lendo GPU: {type(e).__name__}: {e}")

print()
print("CONTROLE")
print(f"  Estado ctrl.:  {ctrl_state}")
print(f"  Lane:          {ctrl_lane}")
print(f"  Motivo:        {ctrl_reason}")
print(f"  Restart:       {ctrl_restart}")
print(f"  Lane PID:      {ctrl_pid}")
print(f"  Global step:   {ctrl_step}")
print(f"  Target:        {ctrl_target}")
print(f"  Age progress:  {ctrl_age_progress}s")
print(f"  Age status:    {ctrl_age_status}")

print()
print("LANES / RESTARTS")
files = sorted(Path("logs").glob("v89_sustained_*_r*.log")) if Path("logs").exists() else []
items = []
for f in files:
    m = re.search(r"v89_sustained_(.+)_r([0-9]+)\.log$", f.name)
    if not m:
        continue
    lane = m.group(1)
    r = int(m.group(2))
    st = f.stat()
    items.append((r, lane, time.time() - st.st_mtime, st.st_size))

if not items:
    print("  Logs lane:     nenhum log de lane encontrado")
else:
    print(f"  Logs lane:     {len(items)} arquivo(s)")
    for r, lane, a, size in sorted(items)[-14:]:
        print(f"  r{r:<3} {lane:<30} {a:.0f}s atrás   {size} bytes")

try:
    gd = json.load(open(global_path, "r", encoding="utf-8")) if global_path.exists() else {}
    print(f"  Global lane:   {gd.get('lane', '?')}")
    print(f"  Global step:   {gd.get('global_step', '?')}")
    print(f"  Global reason: {gd.get('reason', '?')}")
    print(f"  Global restart:{gd.get('restart_index', '?')}")
    print(f"  Stale invalid.:{gd.get('stale_progress_invalidated', '?')}")
except Exception:
    pass

print("  Evidence runs:")
for name, st, lines, last_ts in evidence_runs[-14:]:
    a = time.time() - last_ts if last_ts else -1
    age_s = f"{a:.0f}s atrás" if a >= 0 else "?"
    print(f"    {name:<32} max_step={st:<8} linhas={lines:<5} {age_s}")
PY
}

print_config() {
  echo
  echo "CONFIG"

  local pid=""
  pid="$(pgrep -f "main.py --deepspeed" | head -1 || true)"
  if [ -z "${pid:-}" ]; then
    echo "  Status:        processo main.py --deepspeed não encontrado"
    return 0
  fi

  python - "$pid" <<'PY'
import sys, os, json
from pathlib import Path

pid = sys.argv[1]
parts = []

try:
    raw = open(f"/proc/{pid}/cmdline", "rb").read()
    arr = raw.split(bytes([0]))
    if arr and arr[-1] == b"":
        arr = arr[:-1]
    parts = [x.decode("utf-8", "replace") for x in arr]
except Exception:
    pass

def get(flag, default="?"):
    for i, x in enumerate(parts):
        if x == flag:
            if i + 1 < len(parts):
                v = parts[i + 1]
                return v if v != "" else "(vazio)"
            return "(sem valor)"
    return default

def has(flag):
    return "sim" if flag in parts else "não"

def env_value(key):
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
        prefix = (key + "=").encode()
        for item in raw.split(bytes([0])):
            if item.startswith(prefix):
                return item[len(prefix):].decode("utf-8", "replace")
    except Exception:
        pass
    return "?"

target_global = "?"
try:
    gf = Path("evidence/v89_controller_global_progress_latest.json")
    if gf.exists():
        gd = json.load(open(gf, "r", encoding="utf-8"))
        target_global = gd.get("target_global_steps", "?")
except Exception:
    pass

print(f"  PID:           {pid}")
print(f"  Python:        {os.path.realpath(f'/proc/{pid}/exe') if os.path.exists(f'/proc/{pid}/exe') else '?'}")
print(f"  Env:           {env_value('CONDA_DEFAULT_ENV')}")
print(f"  Dataset:       {get('--dataset-name')}")
print(f"  Config:        {get('--dataset-config')}")
print(f"  Split:         {get('--dataset-split')}")
print(f"  Fallback:      {get('--dataset-fallback-name')}")
print(f"  Mix:           {get('--dataset-mix')}")
print(f"  Tokenizer:     {get('--tokenizer-name')}")
print(f"  Modelo:        {get('--model-preset')}")
print(f"  Seq len:       {get('--sequence-length')}")
print(f"  Target global: {target_global}")
print(f"  Max steps lane:{get('--deepspeed-max-steps')}")
print(f"  Zero stage:    {get('--deepspeed-zero-stage')}")
print(f"  Batch:         {get('--deepspeed-batch-size')}")
print(f"  Grad accum:    {get('--deepspeed-gradient-accumulation-steps')}")
print(f"  Precision:     {get('--deepspeed-precision')}")
print(f"  Load ckpt:     {get('--deepspeed-load-checkpoint', '(não)')}")
print(f"  API LLM:       {has('--llm')}")
print(f"  API executive: {has('--api-executive-mode')}")
PY
}

print_checkpoint() {
  echo
  echo "CHECKPOINT"

  python - <<'PY'
from pathlib import Path
import time, json

now = time.time()
files = []

if Path("checkpoints").exists():
    for f in Path("checkpoints").rglob("*"):
        if f.is_file():
            try:
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f))
            except Exception:
                pass

if not files:
    print("  Status:        nenhum checkpoint encontrado ainda")
else:
    files.sort()
    heavy = [x for x in files if x[1] > 1024 * 1024]
    mtime, size, f = heavy[-1] if heavy else files[-1]
    print(f"  Último real:   {f}")
    print(f"  Tamanho:       {size / 1024 / 1024:.2f} MiB")
    print(f"  Modificado:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    print(f"  Idade:         {now - mtime:.1f}s atrás")

    latest = Path("checkpoints/v89_latest.txt")
    if latest.exists():
        val = latest.read_text(encoding="utf-8", errors="replace").strip()
        print(f"  Latest file:   {latest}")
        print(f"  Latest aponta: {val if val else '(vazio)'}")
        print(f"  Latest idade:  {now - latest.stat().st_mtime:.1f}s atrás")

    max_meta = 0
    for mf in sorted(Path("checkpoints").glob("v89_live_*/metadata.json")):
        try:
            md = json.load(open(mf, "r", encoding="utf-8"))
            stp = max(
                int(md.get("global_step") or 0),
                int(md.get("micro_train_steps_completed") or 0),
                int((md.get("adaptive_memory") or {}).get("last_effect", {}).get("step") or 0),
            )
            max_meta = max(max_meta, stp)
            print(f"  Meta {mf.parent.name}: step={stp} batch={md.get('batch_size')} tok/s={md.get('tokens_per_second')}")
        except Exception:
            pass
    print(f"  Step max meta: {max_meta}")

    print("  Recentes:")
    for mtime, size, f in files[-8:]:
        print(f"    {time.strftime('%H:%M:%S', time.localtime(mtime))}  {size/1024/1024:8.2f} MiB  {now-mtime:7.1f}s  {f}")
PY
}

print_ram() {
  echo
  echo "RAM"
  free -h | awk '
    /^Mem:/ {
      print "  Usada:         "$3" / "$2
      print "  Livre:         "$4
      print "  Disponível:    "$7
    }
    /^Swap:/ {
      print "  Swap:          "$3" / "$2
    }
  '
}

print_processes() {
  echo
  echo "PROCESSOS"

  ps -eo pid,pcpu,pmem,cmd --sort=-pcpu \
    | grep -E "v89|deepspeed|torchrun|main.py|deepspeed_runner|run_v89" \
    | grep -v grep \
    | head -10 \
    | awk '{
      pid=$1; cpu=$2; mem=$3;
      $1=$2=$3="";
      sub(/^ +/,"");
      printf "  PID %-7s CPU %-6s MEM %-6s %s\n", pid, cpu"%", mem"%", substr($0,1,180)
    }'
}

print_recent_events() {
  echo
  echo "EVENTOS RECENTES"

  if [ -f evidence_packets/v89_sustained_control_events.jsonl ]; then
    echo "  Arquivo: evidence_packets/v89_sustained_control_events.jsonl"
    tail -8 evidence_packets/v89_sustained_control_events.jsonl 2>/dev/null | sed 's/^/    /'
  else
    echo "  Sem eventos recentes em evidence_packets/"
  fi
}

clear
hr
echo "                    V89 LIVE MONITOR"
hr
date
echo

RUN="$(latest_run_dir || true)"
LINE=""
if [ -n "${RUN:-}" ]; then
  LINE="$(latest_adaptive_line "$RUN" || true)"
fi

print_main "${RUN:-}" "${LINE:-}"
print_config
print_checkpoint
print_ram
print_processes
print_recent_events

echo
hr
echo "Atualiza pelo watch. Ctrl+C para sair."
hr
