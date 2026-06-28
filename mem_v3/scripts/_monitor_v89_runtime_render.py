#!/usr/bin/env python3
import json
import pathlib
import time
import glob
import sys

root = pathlib.Path(".")
latest_dirs = sorted(root.glob("evidence/v89_wsl_deepspeed_*"), key=lambda p: p.stat().st_mtime, reverse=True)

if not latest_dirs:
    print("STATUS")
    print("  Estado:        AGUARDANDO RUNTIME")
    print("  Run:           nenhum")
    sys.exit(0)

run_dir = latest_dirs[0]
runtime_file = run_dir / "runtime_progress_latest.json"
status_file = root / "evidence/v89_controller_status_latest.json"
events_file = root / "evidence_packets/v89_sustained_control_events.jsonl"

if not runtime_file.exists():
    print("STATUS")
    print("  Estado:        AGUARDANDO RUNTIME")
    print(f"  Run:           {run_dir}")
    sys.exit(0)

try:
    d = json.loads(runtime_file.read_text(errors="ignore"))
except Exception as e:
    print("STATUS")
    print("  Estado:        ERRO LENDO RUNTIME")
    print(f"  Erro:          {e}")
    sys.exit(0)

status = {}
if status_file.exists():
    try:
        status = json.loads(status_file.read_text(errors="ignore"))
    except Exception:
        status = {}

sc = d.get("sustained_control", {}) or {}
prof = d.get("profiler", {}) or {}
ds = d.get("dataset", {}) or {}
chaos = d.get("chaos_environment", {}) or {}

step = int(d.get("step") or 0)
target = int(d.get("target_steps") or 0)
pct = (step / target * 100.0) if target else 0.0
remaining = max(target - step, 0) if target else 0
bar_len = 40
filled = int((pct / 100.0) * bar_len) if target else 0
filled = max(0, min(bar_len, filled))
bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
age = time.time() - runtime_file.stat().st_mtime

def resolve_checkpoint_path():
    # Preferir latest.txt publicado pelo CheckpointManager novo.
    latest_files = sorted(
        pathlib.Path("checkpoints").glob("*_latest.txt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for latest in latest_files:
        try:
            raw = latest.read_text(errors="ignore").strip()
            if not raw:
                continue
            candidate = pathlib.Path(raw)
            if not candidate.is_absolute():
                candidate = pathlib.Path(".") / candidate
            if candidate.exists():
                return candidate
        except Exception:
            pass

    # Fallback: padrão antigo + padrão novo com rotação live_00/live_01/live_02.
    ckpt_files = []
    ckpt_files += glob.glob("checkpoints/*_live/mem_model_optimizer.pt")
    ckpt_files += glob.glob("checkpoints/*_live_*/mem_model_optimizer.pt")
    ckpt_paths = [pathlib.Path(x) for x in ckpt_files if pathlib.Path(x).exists()]
    if ckpt_paths:
        return max(ckpt_paths, key=lambda x: x.stat().st_mtime)
    return None

ckpt_path = resolve_checkpoint_path()
ckpt_status = "não"
ckpt_size = "-"
ckpt_modify = "-"
if ckpt_path and ckpt_path.exists():
    st = ckpt_path.stat()
    ckpt_status = "sim"
    ckpt_size = f"{st.st_size / (1024**3):.2f} GB"
    ckpt_modify = time.strftime("%H:%M:%S", time.localtime(st.st_mtime))

written = sc.get("checkpoint_written")
if written is True:
    written = "sim"
elif written is False:
    written = "não"
elif written is None:
    written = ckpt_status

print("STATUS")
print("  Estado:        RUNNING")
print(f"  Controller:    {status.get('state', '-')}")
print(f"  Lane:          {status.get('lane', 'runtime-real')}")
print(f"  Razão:         {status.get('reason', '-')}")
print(f"  Restart:       {status.get('restart_index', '-')}")
print()

print("PROGRESSO")
print(f"  Step global:   {step} / {target} ({pct:.2f}%)")
print(f"  Step da lane:  {step} / {target}")
print(f"  Faltam:        {remaining} steps")
print(f"  {bar}")
print(f"  Idade:         {age:.1f}s")
print()

print("PERFORMANCE")
print(f"  Tokens/s:      {d.get('tokens_per_second')}")
print(f"  Steps/s:       {d.get('steps_per_second')}")
print(f"  Loss:          {d.get('loss')}")
print()

print("CONTROLE")
print(f"  Bottleneck:    {d.get('bottleneck')}")
print(f"  Bad windows:   {sc.get('bad_windows')}")
print(f"  Degraded:      {sc.get('degraded')}")
print(f"  Recomenda:     {sc.get('recommended_action')}")
print(f"  Optimizer:     {prof.get('optimizer_ratio')}")
print(f"  Data wait:     {prof.get('data_wait_ratio')}")
print(f"  Chaos score:   {d.get('real_chaos_score')}")
print(f"  API ativa:     {d.get('api_executive_enabled')}")
print()

print("CHECKPOINT")
print(f"  Live:          {ckpt_status}")
print(f"  Path:          {ckpt_path if ckpt_path else '-'}")
print(f"  Tamanho:       {ckpt_size}")
print(f"  Modify:        {ckpt_modify}")
print(f"  Written:       {written}")
print(f"  Mode:          {sc.get('checkpoint_mode', '-')}")
print()

print("DATASET")
print(f"  Dataset:       {ds.get('requested_dataset')}")
print(f"  Active:        {ds.get('active_dataset')}")
print(f"  Fallback:      {ds.get('fallback_used')}")
print()

print("GPU")
print(f"  Uso:           {chaos.get('gpu_util_percent')}%")
print(f"  VRAM:          {chaos.get('gpu_memory_used_mb')} / {chaos.get('gpu_memory_total_mb')} MiB")
print(f"  Temp:          {chaos.get('gpu_temperature_c')}°C")
print(f"  Power:         {chaos.get('gpu_power_w')} W")
print()

print("EVENTOS RECENTES")
print(
    "  runtime: step={} tokens/s={} loss={} bad={} action={}".format(
        d.get("step"),
        d.get("tokens_per_second"),
        d.get("loss"),
        sc.get("bad_windows"),
        sc.get("recommended_action"),
    )
)

rows = []
last_event_ts = None
if events_file.exists():
    try:
        lines = events_file.read_text(errors="ignore").splitlines()[-1200:]
    except Exception:
        lines = []
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        ts = e.get("ts")
        if ts:
            last_event_ts = max(last_event_ts or ts, ts)
        if ts and time.time() - ts > 600:
            continue
        ev = e.get("event", "")
        if ev == "controller_heartbeat":
            rows.append(
                "  controller: step={} tokens/s={} loss={} restart={}".format(
                    e.get("global_step"), e.get("tokens_per_second"), e.get("loss"), e.get("restart_index")
                )
            )
        elif ev == "sample":
            sample = e.get("decision_state", {}) or {}
            rows.append(
                "  sample: step={} tokens/s={} bad={} action={}".format(
                    sample.get("step"), sample.get("tokens"), sample.get("bad_count"), sample.get("recommended_fallback")
                )
            )
        elif ev == "checkpoint_resume_not_found_starting_from_scratch":
            rows.append("  resume: NOT_FOUND label={}".format(e.get("checkpoint_label")))
        elif ev in ("checkpoint_resume_selected", "checkpoint_resume_success", "checkpoint_resume_failed"):
            rows.append(
                "  resume: {} path={}".format(
                    ev.replace("checkpoint_resume_", ""), e.get("load_checkpoint") or e.get("checkpoint_path") or "-"
                )
            )
        elif ev == "lane_start":
            lane = e.get("lane")
            if isinstance(lane, dict):
                lane = lane.get("name")
            rows.append("  lane_start: lane={} restart={}".format(lane, e.get("restart_index")))
        elif ev == "lane_switch_decision":
            decision = e.get("decision", {}) or {}
            rows.append("  lane_switch: {} -> {}".format(e.get("current_lane"), decision.get("lane")))
        elif ev == "unexpected_exit":
            rows.append("  exit: lane={} code={} reason={}".format(e.get("lane"), e.get("returncode"), e.get("reason")))
        elif ev == "api_light_error":
            rows.append("  api_error: type={} lane={}".format(e.get("error_type"), e.get("lane")))

if rows:
    for row in rows[-7:]:
        print(row)
else:
    if last_event_ts:
        age_ev = time.time() - last_event_ts
        print("  controller: sem evento novo recente; último há {:.0f}s".format(age_ev))
    else:
        print("  controller: nenhum evento relevante recente")
