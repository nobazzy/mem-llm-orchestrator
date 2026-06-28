#!/usr/bin/env bash
INTERVAL="${MONITOR_INTERVAL:-3}"
cd "$(dirname "$0")/.."
if [ -x .venv312/bin/python ]; then
  PYTHON_BIN=".venv312/bin/python"
elif [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python3"
fi
while true; do
clear

echo "============================================================"
echo "                    V89 LIVE MONITOR"
echo "============================================================"
date
echo

"$PYTHON_BIN" - <<'PY'
import json, time, glob, os

def load_json(path):
    try:
        if not path or not os.path.exists(path): return {}
        txt=open(path,'r',encoding='utf-8').read().strip()
        return json.loads(txt) if txt else {}
    except Exception:
        return {}

def latest(pattern):
    files=glob.glob(pattern, recursive=True)
    if not files: return ''
    return max(files, key=lambda p: os.path.getmtime(p))

s=load_json('evidence/v89_controller_status_latest.json')
p=load_json(latest('evidence/**/runtime_progress_latest.json'))
lane_step=int(float(p.get('step',0) or 0))
global_step=int(float(s.get('global_step', lane_step) or 0))
target=int(float(s.get('target_global_steps', p.get('target_steps',300000)) or 300000))
remaining_target=int(float(p.get('target_steps', max(target-global_step,0)) or max(target-global_step,0)))
# When status is between writes, preserve the larger visible value.
step=max(global_step, lane_step) if global_step == 0 else global_step
pct=(step/target*100) if target else 0
faltam=max(target-step,0)
bar_len=40
bar='#'*int(pct/100*bar_len)+'-'*(bar_len-int(pct/100*bar_len))
prof=p.get('profiler',{})
sc=p.get('sustained_control',{})
ts=p.get('ts') or s.get('ts') or 0
try: age=time.time()-float(ts)
except Exception: age=-1

def fmt(x, dec=2):
    try: return f"{float(x):.{dec}f}"
    except Exception: return str(x)

print('STATUS')
print(f"  Estado:        {s.get('state','?')}")
print(f"  Lane:          {s.get('lane','?')}")
print(f"  Razão:         {s.get('reason','?')}")
print(f"  Restart:       {s.get('restart_index','?')}")
print()
print('PROGRESSO')
print(f"  Step global:   {step} / {target} ({pct:.2f}%)")
print(f"  Step da lane:  {lane_step} / {remaining_target}")
print(f"  Faltam:        {faltam} steps")
print(f"  [{bar}]")
print(f"  Idade:         {fmt(age,1)}s")
print()
print('PERFORMANCE')
print(f"  Tokens/s:      {fmt(p.get('tokens_per_second', s.get('tokens_per_second','?')),0)}")
print(f"  Steps/s:       {fmt(p.get('steps_per_second', s.get('steps_per_second','?')),2)}")
print(f"  Loss:          {fmt(p.get('loss','?'),4)}")
print()
print('CONTROLE')
print(f"  Bottleneck:    {p.get('bottleneck', p.get('bottleneck_classification','?'))}")
print(f"  Bad windows:   {sc.get('bad_windows','?')}")
print(f"  Degraded:      {sc.get('degraded','?')}")
print(f"  Recomenda:     {sc.get('recommended_action','?')}")
print(f"  Optimizer:     {fmt(prof.get('optimizer_ratio', s.get('optimizer_ratio','?')),3)}")
print(f"  Data wait:     {fmt(prof.get('data_wait_ratio', s.get('data_wait_ratio','?')),3)}")
print(f"  Chaos score:   {fmt(p.get('real_chaos_score', s.get('real_chaos_score','?')),2)}")
print(f"  API ativa:     {p.get('api_executive_enabled','?')}")
PY

echo
echo "API CHAMADAS"
python3 - <<'PY'
import json, os, time
path='evidence_packets/v89_sustained_control_events.jsonl'
if not os.path.exists(path):
    print('  nenhuma chamada API registrada')
else:
    lines=[x.strip() for x in open(path,'r',encoding='utf-8') if x.strip()]
    out=[]
    for line in lines[-200:]:
        try: d=json.loads(line)
        except Exception: continue
        ev=d.get('event')
        if ev=='api_light_called':
            out.append(f"  chamada: lane_change_approval | lane={d.get('lane','-')} | motivo={d.get('local_reason','-')}")
        elif ev=='api_light_response':
            r=d.get('response',{})
            out.append(f"  resposta: action={r.get('action','-')} | lane={r.get('lane','-')} | motivo={r.get('reason','-')}")
        elif ev=='v89_api_keep_overridden_by_floor_escape':
            out.append(f"  V89 override: API keep_current_lane -> switch_lane | alvo={d.get('target_lane','-')} | tokens={d.get('tokens','-')} | motivo=floor_escape")
        elif ev=='noncritical_switch_blocked_by_api':
            dec=d.get('decision',{})
            out.append(f"  bloqueada: action={dec.get('action','-')} | lane={d.get('lane','-')}")
    if not out:
        print('  nenhuma chamada API registrada')
    else:
        for x in out[-8:]: print(x)
PY

echo
echo "GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits | awk -F", " '{printf "  Uso:           %s%%\n  VRAM:          %s / %s MiB\n  Temp:          %s°C\n  Power:         %s W\n", $1,$2,$3,$4,$5}'

echo
echo "RAM"
free -h | awk 'NR==2{print "  Usada:         " $3 " / " $2 "\n  Livre:         " $4 "\n  Disponível:    " $7} NR==3{print "  Swap:          " $3 " / " $2}'

echo
echo "PROCESSOS"
ps aux | grep -E "v89_sustained_controller|main.py --deepspeed|torch/_inductor" | grep -v grep | awk '{printf "  PID %-8s CPU %-7s MEM %-7s %s\n", $2, $3"%", $4"%", $11}' || echo "  nenhum processo encontrado"

echo
echo "EVENTOS RECENTES"
python3 - <<'PY'
import json, os
path='evidence_packets/v89_sustained_control_events.jsonl'
if not os.path.exists(path):
    print('  nenhum evento ainda')
else:
    lines=[x.strip() for x in open(path,'r',encoding='utf-8') if x.strip()]
    evs=[]
    for line in lines[-120:]:
        try: d=json.loads(line)
        except Exception: continue
        ev=d.get('event')
        if ev=='sample':
            ds=d.get('decision_state',{})
            evs.append(f"  sample: step={ds.get('step')} tokens/s={ds.get('tokens')} bad={ds.get('bad_count')} action={ds.get('recommended_fallback')}")
        elif ev in {'recovery_escape_triggered','current_lane_temporarily_removed_from_allowed','v89_api_keep_overridden_by_floor_escape','productive_lane_switch_blocked_v89_midband','v89_absolute_health_switch_blocked','v89_same_lane_refresh_decision','v89_same_lane_refresh_exhausted','lane_switch_decision','noncritical_switch_blocked_by_api','unexpected_lane_exit_before_target'}:
            evs.append(f"  {ev}: lane={d.get('lane','-')} alvo={d.get('target_lane', d.get('decision',{}).get('lane','-'))} motivo={','.join(d.get('reasons',[])[:3])}")
    if not evs: print('  sem evento relevante recente')
    else:
        for x in evs[-12:]: print(x)
PY

echo
echo "============================================================"
echo "Atualiza em ${INTERVAL}s. Ctrl+C para sair."
echo "============================================================"
sleep "$INTERVAL"
done
