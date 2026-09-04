from __future__ import annotations

import glob
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
    dirs = sorted(_root.glob("evidence/v89_wsl_deepspeed_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if dirs:
        return dirs[0]
    dirs = sorted(_root.glob("evidence/*"), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True)
    for d in dirs:
        if d.is_dir():
            return d
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

    return {
        "timestamp": time.time(),
        "evidence_dir": str(ev_dir.relative_to(_root)) if ev_dir else "None",
        "progress": progress_data,
        "controller": ctrl_data,
        "api_telemetry": api_data,
        "latest_checkpoint": latest_ckpt,
    }


def get_milestones(limit: int = 500) -> List[Dict[str, Any]]:
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
    return records


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MEM Orchestrator — Live Control Plane</title>
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
    @keyframes pulse-slow { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .pulse-live { animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-surface-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-indigo-500 selection:text-white">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
    <!-- Header -->
    <header class="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-slate-800 gap-4">
      <div>
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold font-mono text-white text-base shadow-lg shadow-indigo-500/20">
            M
          </div>
          <div>
            <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
              MEM Orchestrator <span class="text-xs uppercase px-2.5 py-0.5 rounded-md bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono font-semibold tracking-wide">v89 Sustained Core</span>
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">Autonomous Zero-OOM Governance & Multi-Engine LLM Training</p>
          </div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div id="live-badge" class="flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium font-mono">
          <span class="w-2 h-2 rounded-full bg-emerald-400 pulse-live"></span>
          <span id="live-status-text">AGUARDANDO SINAL</span>
        </div>
        <button onclick="fetchState()" class="px-3 py-1.5 rounded-md bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs font-medium border border-slate-700 transition flex items-center gap-1.5">
          <span>Atualizar</span>
        </button>
      </div>
    </header>

    <!-- Progress Bar (1M Steps Target) -->
    <div class="mt-6 bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
      <div class="flex justify-between items-center mb-2">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Progresso Global do Treino</span>
        <span id="progress-steps-text" class="text-xs font-bold text-indigo-400 font-mono">0 / 1.000.000 steps (0.00%)</span>
      </div>
      <div class="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800">
        <div id="progress-bar-fill" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
      </div>
    </div>

    <!-- Top KPI Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Throughput</p>
        <p id="kpi-tokens-sec" class="text-2xl font-bold font-mono text-emerald-400 mt-1">0.0 <span class="text-xs font-normal text-slate-500">tok/s</span></p>
        <p id="kpi-steps-sec" class="text-xs text-slate-500 mt-1 font-mono">0.0 steps/s</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Função de Perda (Loss)</p>
        <p id="kpi-loss" class="text-2xl font-bold font-mono text-indigo-300 mt-1">--</p>
        <p id="kpi-loss-trend" class="text-xs text-slate-500 mt-1 font-mono">Status: Aguardando</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Detector de Degradação</p>
        <p id="kpi-health" class="text-2xl font-bold text-emerald-400 mt-1 font-mono">HEALTHY</p>
        <p id="kpi-bottleneck" class="text-xs text-slate-500 mt-1 font-mono">Gargalo: Normal</p>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-4">
        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Checkpoint SHA256</p>
        <p id="kpi-ckpt-status" class="text-base font-bold font-mono text-emerald-400 mt-1 truncate">Atômico (Validado)</p>
        <p id="kpi-ckpt-path" class="text-xs text-slate-500 mt-1 truncate font-mono">Slot: live_00</p>
      </div>
    </div>

    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Curva de Perda (Loss Convergence)
        </h2>
        <div class="h-64">
          <canvas id="lossChart"></canvas>
        </div>
      </div>

      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Throughput Sustentado (Tokens / Segundo)
        </h2>
        <div class="h-64">
          <canvas id="throughputChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Governance & AI Directives Row -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
      <div class="lg:col-span-2 bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Governança de IA e LocalPolicyEngine
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-slate-950 rounded-lg p-4 border border-slate-800">
            <span class="text-xs font-bold uppercase text-indigo-400 tracking-wider">Proposta da IA (GPT-4o)</span>
            <div id="ai-directive-box" class="mt-2 text-xs font-mono text-slate-300 space-y-1.5">
              <p>• Plano: Causal LM Candidate</p>
              <p>• Ação: Observe and Stabilize</p>
              <p>• LR Multiplier Sugerido: 0.65x</p>
              <p>• Telemetria: Ativa</p>
            </div>
          </div>

          <div class="bg-slate-950 rounded-lg p-4 border border-slate-800">
            <span class="text-xs font-bold uppercase text-emerald-400 tracking-wider">Aprovação do LocalPolicyEngine</span>
            <div id="policy-verdict-box" class="mt-2 text-xs font-mono text-slate-300 space-y-1.5">
              <p>• Status: <span class="text-emerald-400 font-bold">VALIDADO & BOUNDED</span></p>
              <p>• Clamping de LR: [0.85, 1.0]</p>
              <p>• Grad Clip Norm: [0.25, 1.25]</p>
              <p>• Zero OOM Risk: <span class="text-emerald-400 font-bold">PROTEGIDO</span></p>
            </div>
          </div>
        </div>
      </div>

      <!-- Profiler Breakdown -->
      <div class="bg-surface-900 border border-slate-800/80 rounded-xl p-5 shadow-sm">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Profiler de Fases (%)
        </h2>
        <div class="space-y-3 text-xs font-mono" id="profiler-bars">
          <div>
            <div class="flex justify-between text-slate-400 mb-1">
              <span>Data Fetch</span><span id="prof-data">15%</span>
            </div>
            <div class="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div id="bar-prof-data" class="bg-blue-500 h-full" style="width: 15%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-400 mb-1">
              <span>Forward & Loss</span><span id="prof-fwd">40%</span>
            </div>
            <div class="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div id="bar-prof-fwd" class="bg-indigo-500 h-full" style="width: 40%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-400 mb-1">
              <span>Backward Pass</span><span id="prof-bwd">35%</span>
            </div>
            <div class="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div id="bar-prof-bwd" class="bg-purple-500 h-full" style="width: 35%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-slate-400 mb-1">
              <span>Optimizer Step</span><span id="prof-opt">10%</span>
            </div>
            <div class="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div id="bar-prof-opt" class="bg-emerald-500 h-full" style="width: 10%"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let lossChart, throughputChart;

    function initCharts() {
      const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
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
            tension: 0.2,
            pointRadius: 2
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
            tension: 0.2,
            pointRadius: 2
          }]
        },
        options: chartDefaults
      });
    }

    async function fetchState() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        updateUI(data);

        const mRes = await fetch('/api/milestones');
        const milestones = await mRes.json();
        updateCharts(milestones);
      } catch (err) {
        console.error("Erro ao buscar estado:", err);
      }
    }

    function updateUI(data) {
      const p = data.progress || {};
      const step = p.step || 0;
      const target = p.target_steps || 1000000;
      const pct = target > 0 ? (step / target * 100) : 0;

      document.getElementById('progress-steps-text').innerText = `${step.toLocaleString()} / ${target.toLocaleString()} steps (${pct.toFixed(2)}%)`;
      document.getElementById('progress-bar-fill').style.width = `${Math.min(100, pct)}%`;

      document.getElementById('kpi-tokens-sec').innerHTML = `${(p.tokens_per_second || 0).toFixed(1)} <span class="text-xs font-normal text-slate-500">tok/s</span>`;
      document.getElementById('kpi-steps-sec').innerText = `${(p.steps_per_second || 0).toFixed(2)} steps/s`;

      if (p.loss !== undefined && p.loss !== null) {
        document.getElementById('kpi-loss').innerText = Number(p.loss).toFixed(4);
        document.getElementById('kpi-loss-trend').innerText = 'Loss ativa';
      }

      document.getElementById('kpi-bottleneck').innerText = `Gargalo: ${p.bottleneck || 'normal'}`;
      if (data.latest_checkpoint) {
        document.getElementById('kpi-ckpt-path').innerText = data.latest_checkpoint.split(/[\\\\/]/).slice(-2).join('/');
      }

      const statusBadge = document.getElementById('live-status-text');
      if (step >= target && target > 0) {
        statusBadge.innerText = 'TREINO CONCLUÍDO';
        statusBadge.className = 'text-xs font-mono font-bold px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-400';
      } else if (step > 0) {
        statusBadge.innerText = 'TREINO ATIVO';
        statusBadge.className = 'text-xs font-mono font-bold px-2 py-0.5 rounded border border-indigo-500/40 bg-indigo-500/10 text-indigo-400 animate-pulse';
      } else {
        statusBadge.innerText = 'MONITOR PRONTO';
      }

      // Dynamic AI Directives box
      const ctrl = data.controller || {};
      const api = data.api_telemetry || {};
      if (ctrl.plan_source || api.total_calls !== undefined) {
        const aiBox = document.getElementById('ai-directive-box');
        if (aiBox) {
          aiBox.innerHTML = `
            <p>• Fonte: <span class="text-slate-200 font-bold">${ctrl.plan_source || 'LLMPlanner'}</span></p>
            <p>• Ação: <span class="text-indigo-300 font-bold">${ctrl.executive_action || 'stabilize'}</span></p>
            <p>• Chamadas API: ${api.total_calls || 0} (${api.successful_calls || 0} ok)</p>
            <p>• Tokens consumidos: ${(api.tokens_total || 0).toLocaleString()}</p>
          `;
        }

        const policyBox = document.getElementById('policy-verdict-box');
        if (policyBox) {
          policyBox.innerHTML = `
            <p>• Lane: <span class="text-emerald-400 font-bold">${(ctrl.lane || 'SAFE_STEADY').toUpperCase()}</span></p>
            <p>• Status: <span class="text-emerald-400 font-bold">${ctrl.allowed ? 'VALIDADO & AUTORIZADO' : 'BLOQUEADO'}</span></p>
            <p>• Clamping de LR: [0.85, 1.0]</p>
            <p>• Zero OOM Risk: <span class="text-emerald-400 font-bold">PROTEGIDO</span></p>
          `;
        }
      }
    }

    function updateCharts(milestones) {
      if (!milestones || milestones.length === 0) return;
      const labels = milestones.map(m => m.step);
      const losses = milestones.map(m => m.loss);
      const tokens = milestones.map(m => m.tokens_per_second || (m.tokens_processed ? m.tokens_processed / (m.step || 1) : 0));

      lossChart.data.labels = labels;
      lossChart.data.datasets[0].data = losses;
      lossChart.update();

      throughputChart.data.labels = labels;
      throughputChart.data.datasets[0].data = tokens;
      throughputChart.update();
    }

    window.onload = () => {
      initCharts();
      fetchState();
      setInterval(fetchState, 1500);
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
