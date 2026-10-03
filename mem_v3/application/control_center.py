"""
MEM ORCHESTRATOR - Interactive Control Center & Desktop Application Server.
Provides a modern visual GUI for clients to configure datasets, steps, presets,
monitor real-time training telemetry, Zero-OOM defense, and execute live inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Configure environment
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

# Global state manager
class OrchestratorSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.should_stop = False
        self.process: Optional[subprocess.Popen] = None
        self.current_step = 0
        self.total_steps = 100
        self.loss = 0.0
        self.speed_tokens_sec = 0.0
        self.vram_used_mb = 0.0
        self.vram_total_mb = 8151.0
        self.active_lane = "IDLE"
        self.guard_status = "READY"
        self.shocks_absorbed = 0
        self.history_loss: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.last_result: Dict[str, Any] = {}
        self.active_config: Dict[str, Any] = {}

    def log(self, message: str):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            entry = f"[{ts}] {message}"
            self.logs.append(entry)
            if len(self.logs) > 500:
                self.logs.pop(0)

    def reset_for_run(self, config: Dict[str, Any]):
        with self.lock:
            self.is_running = True
            self.should_stop = False
            self.current_step = 0
            self.total_steps = config.get("steps", 100)
            self.loss = 0.0
            self.speed_tokens_sec = 0.0
            self.active_lane = "INITIALIZING"
            self.guard_status = "CALIBRATING"
            self.shocks_absorbed = 0
            self.history_loss.clear()
            self.logs.clear()
            self.last_result.clear()
            self.active_config = config
        self.log(f"Iniciando orquestração: Modelo={config.get('model_preset')} | Passos={config.get('steps')} | Dataset={config.get('dataset')}")

    def update_step(self, step: int, loss: float, speed: float, vram_mb: float, vram_tot: float, lane: str, guard: str):
        with self.lock:
            self.current_step = step
            self.loss = loss
            self.speed_tokens_sec = speed
            self.vram_used_mb = vram_mb
            self.vram_total_mb = vram_tot
            self.active_lane = lane
            self.guard_status = guard
            self.history_loss.append({"step": step, "loss": round(loss, 4), "speed": round(speed, 1)})
            if len(self.history_loss) > 300:
                self.history_loss.pop(0)

    def finish_run(self, success: bool, summary: Dict[str, Any]):
        with self.lock:
            self.is_running = False
            self.guard_status = "COMPLETED" if success else "STOPPED"
            self.last_result = summary
        self.log("Orquestração concluída com sucesso." if success else "Orquestração interrompida.")

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            vram_pct = (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb > 0 else 0.0
            return {
                "is_running": self.is_running,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress_pct": round((self.current_step / max(1, self.total_steps)) * 100, 1),
                "loss": round(self.loss, 4),
                "speed_tokens_sec": round(self.speed_tokens_sec, 0),
                "vram_used_mb": round(self.vram_used_mb, 1),
                "vram_total_mb": round(self.vram_total_mb, 1),
                "vram_pct": round(vram_pct, 1),
                "active_lane": self.active_lane,
                "guard_status": self.guard_status,
                "shocks_absorbed": self.shocks_absorbed,
                "history": list(self.history_loss[-60:]),
                "recent_logs": list(self.logs[-25:]),
                "last_result": self.last_result,
                "config": self.active_config,
            }


session = OrchestratorSession()


def get_hardware_info() -> Dict[str, Any]:
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "Processador CPU (Modo Padrão)"
    vram_mb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)) if has_cuda else 0.0
    vram_used = (torch.cuda.memory_allocated(0) / (1024 ** 2)) if has_cuda else 0.0

    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024 ** 3)
    ram_free_gb = ram.available / (1024 ** 3)

    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "vram_total_mb": round(vram_mb, 1),
        "vram_used_mb": round(vram_used, 1),
        "vram_total_gb": round(vram_mb / 1024, 2),
        "cpu_count": cpu_count,
        "ram_total_gb": round(ram_total_gb, 1),
        "ram_free_gb": round(ram_free_gb, 1),
    }


def _training_worker(config: Dict[str, Any]):
    steps = int(config.get("steps", 100))
    preset = config.get("model_preset", "medium_75m")
    dataset = config.get("dataset", "roneneldan/TinyStories")
    enable_shocks = bool(config.get("enable_shocks", True))
    shock_size = int(config.get("shock_size_mb", 10000))
    shock_interval = int(config.get("shock_interval", 40))

    session.reset_for_run(config)

    # Resolve python interpreter
    venv_py = _root / ".venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    script_path = _root / "scripts" / "demo_video_chaos_defense.py"

    cmd = [
        py_exec, "-u", str(script_path),
        "--steps", str(steps),
        "--model-preset", preset,
        "--dataset", dataset,
        "--shock-interval", str(shock_interval),
        "--shock-duration", "20",
        "--shock-size-mb", str(shock_size),
    ]
    if not enable_shocks:
        cmd.append("--no-shock")

    session.log(f"Executando processo: {' '.join(cmd[1:])}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        session.process = proc

        import re
        step_pattern = re.compile(
            r"Step\s+(\d+)/(\d+).*?Speed:\s+([\d\.]+)\s+tok/s.*?VRAM:\s+([\d\.]+)MB\s+\(([\d\.]+)%\).*?Lane:\s+(.*?)\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+(.*)"
        )

        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            session.log(line_str)

            # Check if line contains step telemetry
            match = step_pattern.search(line_str)
            if match:
                s_curr = int(match.group(1))
                s_tot = int(match.group(2))
                speed = float(match.group(3))
                vram_m = float(match.group(4))
                # Strip ANSI codes from lane
                clean_lane = re.sub(r"\x1b\[[0-9;]*m", "", match.group(6)).strip("[] ")
                loss = float(match.group(7))
                clean_guard = re.sub(r"\x1b\[[0-9;]*m", "", match.group(8)).strip("[] ")

                hw = get_hardware_info()
                tot_v = hw["vram_total_mb"] or 8151.0

                session.update_step(s_curr, loss, speed, vram_m, tot_v, clean_lane, clean_guard)

            if "SHOCK INJECTED" in line_str:
                with session.lock:
                    session.shocks_absorbed += 1

            if session.should_stop:
                proc.terminate()
                session.log("Sinal de interrupção enviado ao orquestrador.")
                break

        proc.stdout.close()
        return_code = proc.wait()
        success = (return_code == 0) and not session.should_stop

        summary = {
            "exit_code": return_code,
            "steps_completed": session.current_step,
            "final_loss": session.loss,
            "shocks_absorbed": session.shocks_absorbed,
        }
        session.finish_run(success, summary)

    except Exception as e:
        session.log(f"Erro durante execução: {str(e)}")
        session.finish_run(False, {"error": str(e)})


def _run_inference_worker(prompt: str, temperature: float, max_tokens: int) -> str:
    venv_py = _root / ".venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable
    script_path = _root / "scripts" / "run_inference.py"

    cmd = [
        py_exec, "-u", str(script_path),
        "--prompt", prompt,
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
    ]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(_root),
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        output = res.stdout
        # Extract generated portion if present
        if "=== GENERATED TEXT ===" in output:
            output = output.split("=== GENERATED TEXT ===")[-1].strip()
        elif "Prompt:" in output:
            output = output.split("Prompt:")[-1].strip()
        return output.strip() or res.stdout.strip()
    except Exception as e:
        return f"Erro ao gerar inferência: {e}"


# =============================================================================
#  HTML CONTROL CENTER FRONTEND (Modern Dark-Mode UI)
# =============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MEM LLM Orchestrator — Desktop Control Center</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111726;
      --card-border: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --accent-hover: #059669;
      --cyan: #06b6d4;
      --yellow: #f59e0b;
      --red: #ef4444;
      --blue: #3b82f6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }
    header { background-color: rgba(17, 23, 38, 0.85); backdrop-filter: blur(12px); border-bottom: 1px solid var(--card-border); padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 50; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-logo { width: 34px; height: 34px; background: linear-gradient(135deg, #10b981, #06b6d4); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #fff; font-size: 16px; letter-spacing: -1px; }
    .brand-title { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
    .brand-subtitle { font-size: 11px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
    .hw-badge { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 6px 14px; font-size: 12px; display: flex; align-items: center; gap: 8px; font-weight: 600; color: #34d399; }
    .hw-dot { width: 8px; height: 8px; border-radius: 50%; background-color: #10b981; box-shadow: 0 0 10px #10b981; }

    main { flex: 1; padding: 24px 28px; display: grid; grid-template-columns: 420px 1fr; gap: 24px; max-width: 1680px; width: 100%; margin: 0 auto; }
    @media (max-width: 1024px) { main { grid-template-columns: 1fr; } }

    .card { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .card-title { font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--card-border); padding-bottom: 12px; }

    .form-group { display: flex; flex-direction: column; gap: 8px; }
    .form-label { font-size: 13px; font-weight: 600; color: var(--text); display: flex; justify-content: space-between; }
    .form-hint { font-size: 11px; color: var(--text-muted); font-weight: 400; }
    select, input[type="text"], input[type="number"] { background-color: #0c1220; border: 1px solid var(--card-border); border-radius: 8px; color: var(--text); padding: 10px 14px; font-size: 13px; outline: none; transition: border-color 0.2s; }
    select:focus, input:focus { border-color: var(--cyan); }

    .range-wrap { display: flex; align-items: center; gap: 12px; }
    input[type="range"] { flex: 1; accent-color: var(--cyan); }
    .range-val { min-width: 60px; text-align: right; font-weight: 700; color: var(--cyan); font-size: 14px; }

    .toggle-group { display: flex; align-items: center; justify-content: space-between; background: #0c1220; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--card-border); }
    .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #334155; transition: .3s; border-radius: 24px; }
    .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(20px); }

    .btn { cursor: pointer; border: none; border-radius: 8px; font-weight: 700; font-size: 14px; padding: 12px 20px; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn-primary { background: linear-gradient(135deg, #10b981, #059669); color: #fff; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35); }
    .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35); }
    .btn-secondary { background: #1e293b; color: var(--text); border: 1px solid #334155; }
    .btn-secondary:hover { background: #334155; }

    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    @media (max-width: 768px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
    .stat-box { background: #0c1220; border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 4px; }
    .stat-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }
    .stat-value { font-size: 20px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    .stat-sub { font-size: 11px; color: var(--text-muted); }

    .lane-badge { padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; display: inline-block; }
    .lane-peak { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .lane-balanced { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
    .lane-fallback { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .progress-bar-bg { width: 100%; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #06b6d4, #10b981); width: 0%; transition: width 0.3s; }

    .vram-bar-bg { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 6px; }
    .vram-bar-fill { height: 100%; background: #06b6d4; width: 0%; transition: width 0.3s; }

    .chart-container { position: relative; height: 220px; width: 100%; }

    .console-box { background-color: #040810; border: 1px solid #1a2233; border-radius: 8px; padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; color: #cbd5e1; height: 150px; overflow-y: auto; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }

    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--card-border); padding-bottom: 8px; }
    .tab { cursor: pointer; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; color: var(--text-muted); transition: 0.2s; }
    .tab.active { background: #1e293b; color: #fff; }

    .inference-box { display: flex; flex-direction: column; gap: 12px; }
    textarea { width: 100%; height: 80px; background-color: #0c1220; border: 1px solid var(--card-border); border-radius: 8px; color: var(--text); padding: 10px; font-size: 13px; resize: none; outline: none; }
    textarea:focus { border-color: var(--cyan); }
    .output-text { background: #040810; border: 1px solid var(--card-border); border-radius: 8px; padding: 12px; min-height: 80px; font-size: 13px; color: #38bdf8; line-height: 1.6; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">M</div>
      <div>
        <div class="brand-title">MEM ORCHESTRATOR</div>
        <div class="brand-subtitle">Autonomous Adaptive GPU Orchestration & Zero-OOM Defense</div>
      </div>
    </div>
    <div id="hw-badge" class="hw-badge">
      <div class="hw-dot"></div>
      <span id="hw-text">Detectando Hardware...</span>
    </div>
  </header>

  <main>
    <!-- LEFT: CONFIGURATION PANEL -->
    <div class="card">
      <div class="card-title">
        <span>Configuração de Execução</span>
        <span style="font-size: 11px; color: var(--accent); font-weight: 600;">Modo Local</span>
      </div>

      <div class="form-group">
        <label class="form-label">
          <span>Preset da Arquitetura</span>
          <span class="form-hint" id="preset-hint">75M Parâmetros</span>
        </label>
        <select id="model-preset" onchange="onPresetChange()">
          <option value="medium_75m" selected>medium_75m — Rápido & Leve (Recomendado)</option>
          <option value="large_130m">large_130m — Equilibrado (Maior coerência)</option>
          <option value="xlarge_250m">xlarge_250m — Alta Capacidade (Deep Reasoning)</option>
        </select>
      </div>

      <div class="form-group">
        <label class="form-label">
          <span>Dataset de Treinamento</span>
        </label>
        <select id="dataset-select" onchange="onDatasetChange()">
          <option value="roneneldan/TinyStories" selected>roneneldan/TinyStories (Histórias curtas)</option>
          <option value="HuggingFaceFW/fineweb-edu">HuggingFaceFW/fineweb-edu (Educacional web)</option>
          <option value="custom">Arquivo local customizado (.txt / .jsonl)...</option>
        </select>
        <input type="text" id="custom-dataset-input" placeholder="Ex: C:\\caminho\\meu_dataset.txt" style="display: none; margin-top: 6px;">
      </div>

      <div class="form-group">
        <label class="form-label">
          <span>Passos de Execução (Steps)</span>
        </label>
        <div class="range-wrap">
          <input type="range" id="steps-range" min="50" max="2000" step="50" value="150" oninput="document.getElementById('steps-val').textContent = this.value">
          <span class="range-val" id="steps-val">150</span>
        </div>
      </div>

      <div class="toggle-group">
        <div>
          <div style="font-size: 13px; font-weight: 600;">Zero-OOM Chaos Defense</div>
          <div style="font-size: 11px; color: var(--text-muted);">Simula choques repentinos de VRAM para validar o governador</div>
        </div>
        <label class="switch">
          <input type="checkbox" id="shock-toggle" checked onchange="toggleShockGroup(this.checked)">
          <span class="slider"></span>
        </label>
      </div>

      <div class="form-group" id="shock-size-group">
        <label class="form-label">
          <span>Tamanho do Choque de VRAM</span>
          <span class="form-hint" id="shock-val-hint">+10,000 MB</span>
        </label>
        <div class="range-wrap">
          <input type="range" id="shock-range" min="1000" max="15000" step="500" value="10000" oninput="document.getElementById('shock-val-hint').textContent = '+' + Number(this.value).toLocaleString() + ' MB'">
        </div>
      </div>

      <button id="btn-action" class="btn btn-primary" onclick="toggleExecution()">
        <span id="btn-action-text">INICIAR ORQUESTRAÇÃO</span>
      </button>

      <!-- INFERENCE TESTING SUB-CARD -->
      <div style="margin-top: 8px; border-top: 1px solid var(--card-border); padding-top: 14px;">
        <div class="form-label" style="margin-bottom: 8px;">
          <span>Testar Inferência com Modelo</span>
        </div>
        <div class="inference-box">
          <textarea id="prompt-input" placeholder="Digite uma frase inicial... Ex: Era uma vez em uma cidade futurista"></textarea>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-secondary" style="flex: 1;" onclick="runInference()" id="btn-infer">
              Gerar Texto
            </button>
          </div>
          <div class="output-text" id="infer-output">O texto gerado pelo modelo aparecerá aqui...</div>
        </div>
      </div>
    </div>

    <!-- RIGHT: LIVE MONITORING DASHBOARD -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      <!-- METRICS GRID -->
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Progresso do Treino</div>
          <div class="stat-value" id="val-progress">0%</div>
          <div class="stat-sub" id="val-steps">Passo 0 / 0</div>
          <div class="progress-bar-bg" style="margin-top: 8px;">
            <div id="bar-progress" class="progress-bar-fill"></div>
          </div>
        </div>

        <div class="stat-box">
          <div class="stat-label">Loss Atual</div>
          <div class="stat-value" id="val-loss">--</div>
          <div class="stat-sub">Perda de convergência</div>
        </div>

        <div class="stat-box">
          <div class="stat-label">Throughput Real</div>
          <div class="stat-value" id="val-speed">0 <span style="font-size: 12px; font-weight: 500;">tok/s</span></div>
          <div class="stat-sub">Processamento adaptativo</div>
        </div>

        <div class="stat-box">
          <div class="stat-label">Lane Ativa (Governador)</div>
          <div style="margin-top: 4px;">
            <span id="badge-lane" class="lane-badge lane-balanced">IDLE</span>
          </div>
          <div class="stat-sub" id="val-guard" style="margin-top: 4px;">Sistema Pronto</div>
        </div>
      </div>

      <!-- VRAM GAUGE CARD -->
      <div class="card" style="padding: 16px 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 13px; font-weight: 600;">Alocação de Memória VRAM (Zero-OOM Shield)</span>
          <span id="val-vram-text" style="font-size: 13px; font-weight: 700; color: var(--cyan);">0 MB / 0 MB (0%)</span>
        </div>
        <div class="vram-bar-bg">
          <div id="bar-vram" class="vram-bar-fill"></div>
        </div>
      </div>

      <!-- CHARTS -->
      <div class="card">
        <div class="card-title">
          <span>Curva de Convergência & Throughput em Tempo Real</span>
          <span style="font-size: 11px; color: var(--text-muted);" id="val-shocks">0 Choques Absorvidos</span>
        </div>
        <div class="chart-container">
          <canvas id="liveChart"></canvas>
        </div>
      </div>

      <!-- LIVE LOGS CONSOLE -->
      <div class="card" style="padding: 16px;">
        <div class="card-title" style="margin-bottom: 8px; padding-bottom: 6px;">
          <span>Terminal de Telemetria ao Vivo</span>
          <span style="font-size: 11px; cursor: pointer; color: var(--cyan);" onclick="clearLogs()">Limpar</span>
        </div>
        <div id="console-logs" class="console-box">Aguardando início da orquestração...</div>
      </div>
    </div>
  </main>

  <script>
    let isRunning = false;
    let chartInstance = null;

    function initChart() {
      const ctx = document.getElementById('liveChart').getContext('2d');
      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Loss',
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              data: [],
              yAxisID: 'yLoss',
              tension: 0.25,
              borderWidth: 2,
              pointRadius: 1,
            },
            {
              label: 'Tokens/s',
              borderColor: '#06b6d4',
              backgroundColor: 'transparent',
              data: [],
              yAxisID: 'ySpeed',
              tension: 0.25,
              borderWidth: 2,
              borderDash: [4, 4],
              pointRadius: 0,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
            yLoss: { position: 'left', grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#10b981' } },
            ySpeed: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#06b6d4' } }
          },
          plugins: {
            legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
          }
        }
      });
    }

    async function loadHardware() {
      try {
        const res = await fetch('/api/hardware');
        const hw = await res.json();
        const hwText = document.getElementById('hw-text');
        if (hw.has_cuda) {
          hwText.textContent = `${hw.device_name} — ${hw.vram_total_gb} GB VRAM GDDR6`;
        } else {
          hwText.textContent = `${hw.device_name} (${hw.cpu_count} Núcleos | ${hw.ram_free_gb} GB RAM)`;
          document.querySelector('.hw-dot').style.backgroundColor = '#f59e0b';
        }
      } catch (e) {
        console.error("Hardware fetch error", e);
      }
    }

    function onPresetChange() {
      const val = document.getElementById('model-preset').value;
      const hint = document.getElementById('preset-hint');
      if (val === 'medium_75m') hint.textContent = '75M Parâmetros (~1.8 GB VRAM)';
      if (val === 'large_130m') hint.textContent = '130M Parâmetros (~2.6 GB VRAM)';
      if (val === 'xlarge_250m') hint.textContent = '250M Parâmetros (~4.2 GB VRAM)';
    }

    function onDatasetChange() {
      const val = document.getElementById('dataset-select').value;
      const customInp = document.getElementById('custom-dataset-input');
      customInp.style.display = (val === 'custom') ? 'block' : 'none';
    }

    function toggleShockGroup(checked) {
      document.getElementById('shock-size-group').style.opacity = checked ? '1' : '0.4';
      document.getElementById('shock-size-group').style.pointerEvents = checked ? 'auto' : 'none';
    }

    async function toggleExecution() {
      if (!isRunning) {
        // Start
        const preset = document.getElementById('model-preset').value;
        const dsSelect = document.getElementById('dataset-select').value;
        const dataset = (dsSelect === 'custom') ? (document.getElementById('custom-dataset-input').value || 'roneneldan/TinyStories') : dsSelect;
        const steps = parseInt(document.getElementById('steps-range').value);
        const enableShocks = document.getElementById('shock-toggle').checked;
        const shockSize = parseInt(document.getElementById('shock-range').value);

        const payload = {
          model_preset: preset,
          dataset: dataset,
          steps: steps,
          enable_shocks: enableShocks,
          shock_size_mb: shockSize,
        };

        const res = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'started') {
          setRunningUI(true);
        }
      } else {
        // Stop
        await fetch('/api/stop', { method: 'POST' });
        setRunningUI(false);
      }
    }

    function setRunningUI(running) {
      isRunning = running;
      const btn = document.getElementById('btn-action');
      const btnText = document.getElementById('btn-action-text');
      if (running) {
        btn.className = 'btn btn-danger';
        btnText.textContent = 'INTERROMPER ORQUESTRAÇÃO';
      } else {
        btn.className = 'btn btn-primary';
        btnText.textContent = 'INICIAR ORQUESTRAÇÃO';
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/state');
        const st = await res.json();

        setRunningUI(st.is_running);

        document.getElementById('val-progress').textContent = `${st.progress_pct}%`;
        document.getElementById('val-steps').textContent = `Passo ${st.current_step} / ${st.total_steps}`;
        document.getElementById('bar-progress').style.width = `${st.progress_pct}%`;

        document.getElementById('val-loss').textContent = st.loss > 0 ? st.loss.toFixed(4) : '--';
        document.getElementById('val-speed').innerHTML = `${Number(st.speed_tokens_sec).toLocaleString()} <span style="font-size: 12px; font-weight: 500;">tok/s</span>`;

        const badge = document.getElementById('badge-lane');
        badge.textContent = `[${st.active_lane}]`;
        if (st.active_lane.includes('PEAK')) badge.className = 'lane-badge lane-peak';
        else if (st.active_lane.includes('FALLBACK') || st.active_lane.includes('SAFE')) badge.className = 'lane-badge lane-fallback';
        else badge.className = 'lane-badge lane-balanced';

        document.getElementById('val-guard').textContent = st.guard_status;

        // VRAM
        document.getElementById('val-vram-text').textContent = `${st.vram_used_mb} MB / ${st.vram_total_mb} MB (${st.vram_pct}%)`;
        document.getElementById('bar-vram').style.width = `${Math.min(100, st.vram_pct)}%`;
        if (st.vram_pct > 85) document.getElementById('bar-vram').style.backgroundColor = '#ef4444';
        else if (st.vram_pct > 65) document.getElementById('bar-vram').style.backgroundColor = '#f59e0b';
        else document.getElementById('bar-vram').style.backgroundColor = '#06b6d4';

        document.getElementById('val-shocks').textContent = `${st.shocks_absorbed} Choques Absorvidos`;

        // Update charts
        if (st.history && st.history.length > 0) {
          chartInstance.data.labels = st.history.map(h => h.step);
          chartInstance.data.datasets[0].data = st.history.map(h => h.loss);
          chartInstance.data.datasets[1].data = st.history.map(h => h.speed);
          chartInstance.update();
        }

        // Update logs
        if (st.recent_logs && st.recent_logs.length > 0) {
          const consoleEl = document.getElementById('console-logs');
          consoleEl.textContent = st.recent_logs.join('\\n');
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }
      } catch (e) {
        console.error("Telemetry update error", e);
      }
    }

    async function runInference() {
      const prompt = document.getElementById('prompt-input').value.trim();
      if (!prompt) return;
      const btn = document.getElementById('btn-infer');
      const out = document.getElementById('infer-output');
      btn.disabled = true;
      btn.textContent = 'Gerando...';
      out.textContent = 'Aguarde, processando com os pesos treinados...';

      try {
        const res = await fetch('/api/inference', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt, temperature: 0.7, max_tokens: 80 })
        });
        const data = await res.json();
        out.textContent = data.generated_text || data.error || 'Nenhum texto gerado.';
      } catch (e) {
        out.textContent = `Erro na geração: ${e}`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Gerar Texto';
      }
    }

    function clearLogs() {
      document.getElementById('console-logs').textContent = '';
    }

    window.onload = () => {
      initChart();
      loadHardware();
      onPresetChange();
      setInterval(updateTelemetry, 1000);
    };
  </script>
</body>
</html>
"""


class ControlCenterHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return  # Silence access logs

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if parsed.path == "/api/hardware":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_hardware_info()).encode("utf-8"))
            return

        if parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(session.get_state()).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed.path == "/api/start":
            if session.is_running:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Já existe um processo em execução"}).encode("utf-8"))
                return

            t = threading.Thread(target=_training_worker, args=(payload,), daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "config": payload}).encode("utf-8"))
            return

        if parsed.path == "/api/stop":
            session.should_stop = True
            if session.process:
                try:
                    session.process.terminate()
                except Exception:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))
            return

        if parsed.path == "/api/inference":
            prompt = payload.get("prompt", "")
            temp = float(payload.get("temperature", 0.7))
            tokens = int(payload.get("max_tokens", 80))

            output = _run_inference_worker(prompt, temp, tokens)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"generated_text": output}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_control_center(port: int = 8089) -> int:
    for p in [port, 8089, 8090, 8080, 8888]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), ControlCenterHandler)
            print(f"[*] MEM Control Center ativo em: http://127.0.0.1:{p}")
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return p
        except Exception:
            continue
    raise RuntimeError("Não foi possível alocar porta para o servidor.")


if __name__ == "__main__":
    p = run_control_center(8089)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Encerrando.")
