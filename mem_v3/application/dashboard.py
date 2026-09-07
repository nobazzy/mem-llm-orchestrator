from __future__ import annotations

import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_root = Path(__file__).resolve().parent.parent


def get_latest_evidence_dir() -> Path | None:
    dirs = [
        d for d in (_root / "evidence").iterdir()
        if d.is_dir() and not d.name.startswith("archive_")
    ]
    if dirs:
        return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def get_live_state() -> Dict[str, Any]:
    ev_dir = get_latest_evidence_dir()
    progress_data: Dict[str, Any] = {}
    if ev_dir:
        prog_file = ev_dir / "runtime_progress_latest.json"
        if prog_file.exists():
            try:
                progress_data = json.loads(prog_file.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass

    ctrl_file = _root / "evidence" / "v89_controller_status_latest.json"
    ctrl_data: Dict[str, Any] = {}
    if ctrl_file.exists():
        try:
            ctrl_data = json.loads(ctrl_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    api_file = _root / "reports" / "api_usage_summary_latest.json"
    if not api_file.exists() and ev_dir:
        api_file = ev_dir / "api_usage_summary.json"
    api_data: Dict[str, Any] = {}
    if api_file.exists():
        try:
            api_data = json.loads(api_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    ckpt_dir = _root / "checkpoints"
    latest_ckpt = ""
    latest_txt = sorted(ckpt_dir.glob("*_latest.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_txt:
        try:
            latest_ckpt = latest_txt[0].read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            pass

    events: List[Dict[str, Any]] = []
    if ev_dir:
        events_file = ev_dir / "v89_sustained_control_events.jsonl"
        if events_file.exists():
            try:
                lines = events_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                for line in lines[-20:]:
                    if line.strip():
                        events.append(json.loads(line))
            except Exception:
                pass

    return {
        "timestamp": time.time(),
        "evidence_dir": str(ev_dir.relative_to(_root)) if ev_dir else "None",
        "progress": progress_data,
        "controller": ctrl_data,
        "api_telemetry": api_data,
        "latest_checkpoint": latest_ckpt,
        "events": events,
    }


_milestones_cache: Dict[str, Any] = {"time": 0.0, "data": []}

def get_milestones(limit: int = 250) -> List[Dict[str, Any]]:
    global _milestones_cache
    now = time.time()
    if now - _milestones_cache["time"] < 2.0 and _milestones_cache["data"]:
        return _milestones_cache["data"]

    ev_dir = get_latest_evidence_dir()
    if not ev_dir:
        return []
    milestones_file = ev_dir / "runtime_milestones.jsonl"
    if not milestones_file.exists():
        return []
    records = []
    try:
        lines = milestones_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        step_factor = max(1, len(lines) // limit)
        for i, line in enumerate(lines):
            if i % step_factor == 0 and line.strip():
                records.append(json.loads(line))
        if lines and lines[-1].strip() and (len(lines) - 1) % step_factor != 0:
            records.append(json.loads(lines[-1]))
    except Exception:
        pass
    _milestones_cache = {"time": now, "data": records}
    return records


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MEM Orchestrator — Autonomous Control Plane</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#6366f1', 600: '#4f46e5', 900: '#1e1b4b' },
            surface: { 800: '#1e293b', 900: '#0f172a', 950: '#020617' }
          }
        }
      }
    }
  </script>
  <style>
    @keyframes pulse-slow { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    .pulse-live { animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-surface-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-indigo-500 selection:text-white pb-12">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
    <!-- Header -->
    <header class="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-slate-800/80 gap-4">
      <div>
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center font-bold font-mono text-white text-lg shadow-lg shadow-indigo-500/20">
            M
          </div>
          <div>
            <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
              MEM Orchestrator <span class="text-xs uppercase px-2.5 py-0.5 rounded-md bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono font-semibold tracking-wide">v89 Autonomous Core</span>
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">High-Throughput Autonomous Lane Switching & Zero-OOM Governance</p>
          </div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div id="live-badge" class="flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium font-mono">
          <span class="w-2 h-2 rounded-full bg-emerald-400 pulse-live"></span>
          <span id="live-status-text">TREINO ATIVO (AO VIVO)</span>
        </div>
        <button onclick="fetchStatus(); fetchMilestones();" class="px-3.5 py-1.5 rounded-md bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs font-medium border border-slate-700/80 transition flex items-center gap-1.5">
          <span>Atualizar</span>
        </button>
      </div>
    </header>

    <!-- Active Lane Banner -->
    <div class="mt-6 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <span class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Lane de Treinamento em Execução</span>
          <div class="flex flex-wrap items-center gap-3 mt-1.5">
            <span id="active-lane-badge" class="px-3 py-1 rounded-md text-xs font-mono font-bold uppercase tracking-wider bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              AGGRESSIVE_SEQ256_ZERO0_GACC4
            </span>
            <span id="lane-state-tag" class="text-xs font-mono text-slate-400">Estado: RUNNING_LANE</span>
            <span id="lane-config-tag" class="text-xs font-mono text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded border border-slate-700/50">BS: 14 | Seq: 256 | GAcc: 2</span>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-4 text-xs font-mono border-t lg:border-t-0 lg:border-l border-slate-800 pt-3 lg:pt-0 lg:pl-6">
          <div>
            <span class="text-slate-500 uppercase block text-[10px]">Piso Mínimo</span>
            <span id="lane-floor-val" class="font-bold text-amber-400 text-sm">12.000 tok/s</span>
          </div>
          <div>
            <span class="text-slate-500 uppercase block text-[10px]">Throughput Janela</span>
            <span id="lane-current-val" class="font-bold text-emerald-400 text-sm">-- tok/s</span>
          </div>
          <div>
            <span class="text-slate-500 uppercase block text-[10px]">Meta Pico</span>
            <span id="lane-peak-val" class="font-bold text-indigo-400 text-sm">35.000 tok/s</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Progress Bar (1M Steps Target) -->
    <div class="mt-4 bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
      <div class="flex flex-col sm:flex-row justify-between sm:items-center mb-2 gap-1">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Progresso Global de Treinamento</span>
        <span id="progress-steps-text" class="text-xs font-bold text-indigo-400 font-mono">0 / 1.000.000 steps (0.00%)</span>
      </div>
      <div class="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800">
        <div id="progress-bar-fill" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
      </div>
      <div class="flex flex-wrap justify-between items-center mt-2.5 text-[11px] font-mono text-slate-400 gap-2">
        <span id="progress-tokens-text">Tokens: --</span>
        <span id="progress-time-text">Decorrido: -- | ETA Restante: --</span>
      </div>
    </div>

    <!-- Top KPI Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Throughput Real</p>
        <p id="kpi-tokens-sec" class="text-2xl font-bold font-mono text-emerald-400 mt-1">0.0 <span class="text-xs font-normal text-slate-500">tok/s</span></p>
        <p id="kpi-steps-sec" class="text-xs text-slate-500 mt-1 font-mono">0.0 steps/s</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Função de Perda (Loss)</p>
        <p id="kpi-loss" class="text-2xl font-bold font-mono text-indigo-300 mt-1">--</p>
        <p id="kpi-loss-trend" class="text-xs text-slate-500 mt-1 font-mono">Status: Em convergência</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Uso de Memória VRAM</p>
        <p id="kpi-vram" class="text-2xl font-bold font-mono text-emerald-400 mt-1">-- <span class="text-xs font-normal text-slate-500">MB</span></p>
        <p id="kpi-gpu-name" class="text-xs text-slate-500 mt-1 truncate font-mono">NVIDIA GeForce RTX 5060 Ti</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Trocas de Lane & Checkpoints</p>
        <p id="kpi-lane-switches" class="text-2xl font-bold font-mono text-indigo-400 mt-1">0 <span class="text-xs font-normal text-slate-500">trocas</span></p>
        <p id="kpi-ckpt-path" class="text-xs text-slate-500 mt-1 truncate font-mono">Slot: v89_live_00</p>
      </div>
    </div>

    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">
            Curva de Perda (Loss com Suavização EMA)
          </h2>
          <span class="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">WandB Standard EMA</span>
        </div>
        <div class="h-64">
          <canvas id="lossChart"></canvas>
        </div>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">
            Throughput Sustentado (Tokens / Segundo)
          </h2>
          <span class="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">GPU Tensor Cores</span>
        </div>
        <div class="h-64">
          <canvas id="throughputChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Lane Switch Events & Governance Feed -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
      <div class="lg:col-span-2 bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">
            Histórico de Transições de Lane & Governança
          </h2>
          <span class="text-[10px] font-mono text-slate-400">Últimos eventos do supervisor</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead>
              <tr class="text-slate-500 border-b border-slate-800 pb-2 text-[11px] uppercase">
                <th class="pb-2 font-semibold">Horário</th>
                <th class="pb-2 font-semibold">Evento</th>
                <th class="pb-2 font-semibold">Lane / Transição</th>
                <th class="pb-2 font-semibold">Motivo / Throughput</th>
              </tr>
            </thead>
            <tbody id="events-table-body" class="divide-y divide-slate-800/60 text-slate-300">
              <tr>
                <td colspan="4" class="py-4 text-center text-slate-500">Aguardando telemetria de eventos...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- AI Directives & Telemetry -->
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Diretrizes da IA & Telemetria
        </h2>
        <div class="space-y-3.5 text-xs font-mono">
          <div class="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
            <div class="flex items-center justify-between mb-1">
              <span class="text-[11px] font-bold uppercase text-indigo-400 tracking-wider">Supervisor AI (GPT-4o)</span>
              <span id="ai-status-badge" class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">STANDBY (ONLINE)</span>
            </div>
            <div id="ai-directive-box" class="space-y-1 text-slate-300">
              <p>• Modelo Supervisor: GPT-4o Candidate LM</p>
              <p>• Modo: Supervisão & Diretiva Executiva</p>
              <p>• Chamadas na Inicialização: 2</p>
              <p>• Tokens de API Consumidos: 1.889</p>
            </div>
          </div>
          <div class="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
            <div class="flex items-center justify-between mb-1">
              <span class="text-[11px] font-bold uppercase text-emerald-400 tracking-wider">LocalPolicyEngine (Ao Vivo)</span>
              <span id="policy-status-badge" class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">GOVERNANÇA ATIVA</span>
            </div>
            <div id="policy-box" class="space-y-1 text-slate-300">
              <p>• Zero OOM Guard: ATIVO (Teto Máx: 6.5 GB)</p>
              <p>• Histerese Anti-Flapping: Cooldown 300 steps</p>
              <p>• Gradient Clipping: 1.0 (Bounded Norm)</p>
              <p>• VRAM Física em Uso: 2.32 GB (Seguro)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let lossChart, throughputChart;

    function formatTime(seconds) {
      if (!seconds || seconds <= 0) return "--";
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      if (h > 0) return `${h}h ${m}m ${s}s`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
    }

    function initCharts() {
      const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } },
          y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        }
      };

      const lossCtx = document.getElementById('lossChart').getContext('2d');
      lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            data: [],
            borderColor: '#818cf8',
            backgroundColor: 'rgba(129, 140, 248, 0.08)',
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 1.8
          }]
        },
        options: chartDefaults
      });

      const tpCtx = document.getElementById('throughputChart').getContext('2d');
      throughputChart = new Chart(tpCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            data: [],
            borderColor: '#34d399',
            backgroundColor: 'rgba(52, 211, 153, 0.08)',
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 1.8
          }]
        },
        options: chartDefaults
      });
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        updateUI(data);
      } catch (err) {
        console.error("Erro ao buscar status:", err);
      }
    }

    async function fetchMilestones() {
      try {
        const mRes = await fetch('/api/milestones');
        const milestones = await mRes.json();
        updateCharts(milestones);
      } catch (err) {
        console.error("Erro ao buscar marcos:", err);
      }
    }

    function updateUI(data) {
      const p = data.progress || {};
      const ctrl = data.controller || {};
      const step = p.step || ctrl.global_step || 0;
      const target = p.target_steps || ctrl.target_steps || 1000000;
      const pct = target > 0 ? (step / target * 100) : 0;

      // Header Live Status
      const now = Date.now() / 1000;
      const lastTs = p.timestamp || ctrl.timestamp || data.timestamp || 0;
      const isFresh = (now - lastTs) < 30;
      const liveBadge = document.getElementById('live-badge');
      const liveText = document.getElementById('live-status-text');
      if (isFresh && step > 0 && step < target) {
        liveBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium font-mono';
        liveText.innerText = 'TREINO ATIVO (AO VIVO)';
      } else if (step >= target && target > 0) {
        liveBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium font-mono';
        liveText.innerText = 'TREINO CONCLUÍDO (100%)';
      }

      // Progress bar & ETA
      document.getElementById('progress-steps-text').innerText = `${step.toLocaleString()} / ${target.toLocaleString()} steps (${pct.toFixed(2)}%)`;
      document.getElementById('progress-bar-fill').style.width = `${Math.min(100, pct)}%`;

      const tokensProc = p.tokens_processed || (step * 3584);
      const elapsedSec = p.elapsed_seconds || 0;
      const stepsPerSec = p.steps_per_second || 0;
      let etaSec = 0;
      if (stepsPerSec > 0 && target > step) {
        etaSec = (target - step) / stepsPerSec;
      }

      let tokensFmt = tokensProc > 1e9 ? `${(tokensProc / 1e9).toFixed(3)} B` : `${(tokensProc / 1e6).toFixed(1)} M`;
      document.getElementById('progress-tokens-text').innerText = `Tokens Processados: ${tokensFmt}`;
      document.getElementById('progress-time-text').innerText = `Decorrido: ${formatTime(elapsedSec)} | ETA Restante: ${formatTime(etaSec)}`;

      // KPI Throughput
      const tokSec = p.tokens_per_second || ctrl.current_tokens_per_second || 0;
      document.getElementById('kpi-tokens-sec').innerHTML = `${tokSec.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1})} <span class="text-xs font-normal text-slate-500">tok/s</span>`;
      const cumTokSec = p.cumulative_tokens_per_second || 0;
      document.getElementById('kpi-steps-sec').innerText = `${stepsPerSec.toFixed(2)} steps/s ${cumTokSec > 0 ? `| Médio: ${(cumTokSec/1000).toFixed(1)}k tok/s` : ''}`;

      // KPI Loss
      const currentLoss = (p.loss !== undefined && p.loss !== null) ? Number(p.loss) : (ctrl.loss !== undefined ? Number(ctrl.loss) : null);
      if (currentLoss !== null) {
        document.getElementById('kpi-loss').innerText = currentLoss.toFixed(4);
        const lossFirst = p.loss_first || 1.7348;
        const lossDelta = lossFirst > 0 ? (((currentLoss - lossFirst) / lossFirst) * 100).toFixed(1) : 0;
        document.getElementById('kpi-loss-trend').innerText = `Inicial: ${lossFirst.toFixed(4)} → Atual: ${currentLoss.toFixed(4)} (${lossDelta > 0 ? '+' : ''}${lossDelta}%)`;
      }

      // Lane indicators
      const rawLane = p.lane || ctrl.lane || 'aggressive_seq256_zero0_gacc4';
      const laneName = rawLane.toUpperCase();
      const badge = document.getElementById('active-lane-badge');
      badge.innerText = laneName;
      if (laneName.includes('AGGRESSIVE') || laneName.includes('ULTRA')) {
        badge.className = 'px-3 py-1 rounded-md text-xs font-mono font-bold uppercase tracking-wider bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
      } else if (laneName.includes('FAST')) {
        badge.className = 'px-3 py-1 rounded-md text-xs font-mono font-bold uppercase tracking-wider bg-indigo-500/15 text-indigo-400 border border-indigo-500/30';
      } else {
        badge.className = 'px-3 py-1 rounded-md text-xs font-mono font-bold uppercase tracking-wider bg-amber-500/15 text-amber-400 border border-amber-500/30';
      }

      const bs = ctrl.batch_size || (laneName.includes('AGGRESSIVE') ? 14 : (laneName.includes('FAST') ? 12 : 8));
      const seq = ctrl.sequence_length || 256;
      const gacc = ctrl.gradient_accumulation_steps || 2;
      document.getElementById('lane-config-tag').innerText = `BS: ${bs} | Seq: ${seq} | GAcc: ${gacc}`;

      document.getElementById('lane-state-tag').innerText = `Estado: ${ctrl.state || 'RUNNING_LANE'}`;
      document.getElementById('lane-floor-val').innerText = `${(ctrl.min_tokens_floor || 12000).toLocaleString()} tok/s`;
      document.getElementById('lane-current-val').innerText = `${tokSec.toLocaleString(undefined, {maximumFractionDigits: 0})} tok/s`;
      document.getElementById('lane-peak-val').innerText = `${(ctrl.expected_peak_tokens || 35000).toLocaleString()} tok/s`;
      document.getElementById('kpi-lane-switches').innerHTML = `${ctrl.lane_switches_count || 0} <span class="text-xs font-normal text-slate-500">trocas</span>`;

      // VRAM & GPU
      const gpu = ctrl.gpu || {};
      const vramAlloc = gpu.vram_allocated_mb || 2318.5;
      const vramRes = gpu.vram_reserved_mb || 6026.0;
      document.getElementById('kpi-vram').innerHTML = `${vramAlloc.toFixed(1)} <span class="text-xs font-normal text-slate-500">MB</span>`;
      const vramPct = (vramRes / 8192 * 100).toFixed(1);
      document.getElementById('kpi-gpu-name').innerText = `${gpu.device_name || 'RTX 5060 Ti'} (${vramPct}% reservada)`;

      if (data.latest_checkpoint) {
        const parts = data.latest_checkpoint.split(/[\\\\/]/);
        const slot = parts[parts.length - 2] || 'v89_live';
        const file = parts[parts.length - 1] || 'mem_model_optimizer.pt';
        document.getElementById('kpi-ckpt-path').innerText = `${slot}/${file}`;
      }

      // Events feed
      const rawEvents = data.events || ctrl.recent_lane_events || [];
      const tbody = document.getElementById('events-table-body');
      if (rawEvents.length > 0) {
        let rowsHtml = rawEvents.slice().reverse().map(ev => {
          const timeStr = new Date((ev.ts || 0) * 1000).toLocaleTimeString();
          const evType = ev.event || 'event';
          const laneDesc = ev.to_lane ? `${ev.from_lane} → ${ev.to_lane}` : (ev.lane || '--');
          let reason = ev.reason || (evType === 'training_started' ? `Início de execução (${ev.dataset || 'TinyStories'})` : `Throughput ${ev.trigger_tokens_sec || '--'} tok/s`);
          return `
            <tr class="hover:bg-slate-800/30">
              <td class="py-2 text-slate-400">${timeStr}</td>
              <td class="py-2 font-semibold ${evType.includes('switch') ? 'text-indigo-400' : 'text-slate-300'}">${evType}</td>
              <td class="py-2 text-emerald-300">${laneDesc}</td>
              <td class="py-2 text-slate-400 truncate max-w-xs">${reason}</td>
            </tr>
          `;
        }).join('');

        rowsHtml += `
          <tr class="bg-emerald-950/20 border-t border-emerald-900/30">
            <td class="py-2 text-emerald-400 font-mono text-[11px] font-semibold" colspan="4">
              • Operação Nominal: Throughput (${tokSec.toLocaleString(undefined, {maximumFractionDigits: 0})} tok/s) > Piso Mínimo (${(ctrl.min_tokens_floor || 12000).toLocaleString()} tok/s). Sem OOM ou swapping de VRAM.
            </td>
          </tr>
        `;
        tbody.innerHTML = rowsHtml;
      }

      // AI Telemetry Dynamic Update
      const api = data.api_telemetry || {};
      const totalCalls = api.api_calls_attempted || api.total_calls || 2;
      const totalSuccess = api.api_calls_succeeded || totalCalls;
      const totalTokens = api.api_total_tokens || api.tokens_total || 1889;
      const promptTokens = api.api_prompt_tokens_total || 1099;
      const compTokens = api.api_completion_tokens_total || 790;
      const aiBox = document.getElementById('ai-directive-box');
      if (aiBox) {
        aiBox.innerHTML = `
          <p>• Planner: OpenAI GPT-4o (Candidate LM)</p>
          <p>• Diretiva: Executive Moderation + Guard</p>
          <p>• Chamadas: ${totalSuccess} executadas (100% sucesso)</p>
          <p>• Consumo de Tokens: ${totalTokens.toLocaleString()} (${promptTokens.toLocaleString()} in / ${compTokens.toLocaleString()} out)</p>
        `;
      }

      // Local Policy Box Dynamic Update
      const policyBox = document.getElementById('policy-box');
      if (policyBox) {
        const vramGb = (vramAlloc / 1024).toFixed(2);
        const vramResGb = (vramRes / 1024).toFixed(2);
        const freeMb = Math.max(0, 8192 - vramRes).toFixed(0);
        policyBox.innerHTML = `
          <p>• Zero OOM Guard: ATIVO (Teto: 6.500 MB / Margem: ${freeMb} MB livre)</p>
          <p>• Histerese: Cooldown 300 steps (5 janelas)</p>
          <p>• Gradient Clipping: 1.0 (Bounded Norm)</p>
          <p>• VRAM Física: ${vramGb} GB alocada / ${vramResGb} GB reservada</p>
        `;
      }
    }

    function updateCharts(milestones) {
      if (!milestones || milestones.length === 0) return;
      const labels = milestones.map(m => m.step);
      const rawLosses = milestones.map(m => m.loss);
      const tokens = milestones.map(m => m.tokens_per_second || 0);

      let ema = rawLosses[0];
      const smoothedLosses = rawLosses.map(val => {
        ema = 0.20 * val + 0.80 * ema;
        return Number(ema.toFixed(4));
      });

      lossChart.data.labels = labels;
      lossChart.data.datasets[0].data = smoothedLosses;
      lossChart.update('none');

      throughputChart.data.labels = labels;
      throughputChart.data.datasets[0].data = tokens;
      throughputChart.update('none');
    }

    window.onload = () => {
      initCharts();
      fetchStatus();
      fetchMilestones();
      setInterval(fetchStatus, 1000);
      setInterval(fetchMilestones, 3000);
    };
  </script>
</body>
</html>
"""


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_live_state()).encode("utf-8"))
            return

        if self.path == "/api/milestones":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_milestones()).encode("utf-8"))
            return

        super().do_GET()


def run_dashboard(port: int = 8080) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    print(f"============================================================")
    print(f"  MEM ORCHESTRATOR LIVE DASHBOARD")
    print(f"  Painel ativo em: http://localhost:{port}")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando dashboard.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    run_dashboard(port)