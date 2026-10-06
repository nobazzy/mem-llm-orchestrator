"""
MEM ORCHESTRATOR - Production LLM Training Studio & Control Center.
Provides a modern visual GUI for production model pre-training configuration,
hyperparameter tuning, dataset streaming, real-time telemetry, and live weight inference.
100% in English for professional deployment.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Configure environment & paths
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

# Global production session state
class ProductionTrainingSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.should_stop = False
        self.process: Optional[subprocess.Popen] = None
        self.current_step = 0
        self.total_steps = 50000
        self.loss = 0.0
        self.avg_loss = 0.0
        self.val_loss: Optional[float] = None
        self.speed_tokens_sec = 0.0
        self.step_rate = 0.0
        self.vram_used_mb = 0.0
        self.vram_total_mb = 8123.0
        self.active_lane = "IDLE"
        self.lane_switches = 0
        self.lane_events: List[Dict[str, Any]] = []
        self.eta_str = "--:--:--"
        self.history_loss: List[Dict[str, Any]] = []
        self.history_val: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.last_result: Dict[str, Any] = {}
        self.active_config: Dict[str, Any] = {}
        self.last_activity_time = time.time()
        self.ever_connected = False

    def record_activity(self):
        with self.lock:
            self.last_activity_time = time.time()
            self.ever_connected = True

    def get_last_activity(self) -> float:
        with self.lock:
            return self.last_activity_time

    def has_ever_connected(self) -> bool:
        with self.lock:
            return self.ever_connected

    def log(self, message: str):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            entry = f"[{ts}] {message}"
            self.logs.append(entry)
            if len(self.logs) > 600:
                self.logs.pop(0)

    def record_lane_switch(self, event: Dict[str, Any]):
        with self.lock:
            self.lane_switches += 1
            event["switch_id"] = self.lane_switches
            event["time"] = time.strftime("%H:%M:%S")
            self.lane_events.append(event)
            if len(self.lane_events) > 50:
                self.lane_events.pop(0)
        self.log(f"Lane Transition #{self.lane_switches}: {event.get('from_lane')} -> {event.get('to_lane')} | {event.get('reason')}")

    def reset_for_run(self, config: Dict[str, Any]):
        with self.lock:
            self.is_running = True
            self.should_stop = False
            self.current_step = 0
            self.total_steps = int(config.get("steps", 50000))
            self.loss = 0.0
            self.avg_loss = 0.0
            self.val_loss = None
            self.speed_tokens_sec = 0.0
            self.step_rate = 0.0
            self.active_lane = "INITIALIZING"
            self.lane_switches = 0
            self.lane_events.clear()
            self.eta_str = "Calculating..."
            self.history_loss.clear()
            self.history_val.clear()
            self.logs.clear()
            self.last_result.clear()
            self.active_config = config
        self.log(f"Starting Production Training: Model={config.get('model_preset')} | Steps={config.get('steps')} | Dataset={config.get('dataset')}")

    def update_telemetry(
        self,
        step: int,
        loss: float,
        avg_loss: float,
        val_loss: Optional[float],
        speed: float,
        step_rate: float,
        vram_mb: float,
        vram_tot: float,
        lane: str,
        eta: str,
        switches: Optional[int] = None,
    ):
        with self.lock:
            self.current_step = step
            self.loss = loss
            self.avg_loss = avg_loss
            if val_loss is not None:
                self.val_loss = val_loss
                self.history_val.append({"step": step, "val_loss": round(val_loss, 4)})
                if len(self.history_val) > 150:
                    self.history_val.pop(0)

            self.speed_tokens_sec = speed
            self.step_rate = step_rate
            self.vram_used_mb = vram_mb
            self.vram_total_mb = vram_tot
            self.active_lane = lane
            self.eta_str = eta
            if switches is not None and switches > self.lane_switches:
                self.lane_switches = switches

            self.history_loss.append({
                "step": step,
                "loss": round(loss, 4),
                "avg_loss": round(avg_loss, 4),
                "speed": round(speed, 1)
            })
            if len(self.history_loss) > 300:
                self.history_loss.pop(0)

    def finish_run(self, success: bool, summary: Dict[str, Any]):
        with self.lock:
            self.is_running = False
            self.active_lane = "COMPLETED" if success else "STOPPED"
            self.last_result = summary
        self.log("Training run finished successfully." if success else "Training run halted.")

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            vram_pct = (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb > 0 else 0.0
            progress_pct = round((self.current_step / max(1, self.total_steps)) * 100, 2)
            return {
                "is_running": self.is_running,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress_pct": progress_pct,
                "loss": round(self.loss, 4),
                "avg_loss": round(self.avg_loss, 4),
                "val_loss": round(self.val_loss, 4) if self.val_loss is not None else None,
                "speed_tokens_sec": round(self.speed_tokens_sec, 0),
                "step_rate": round(self.step_rate, 2),
                "vram_used_mb": round(self.vram_used_mb, 1),
                "vram_total_mb": round(self.vram_total_mb, 1),
                "vram_pct": round(vram_pct, 1),
                "active_lane": self.active_lane,
                "lane_switches": self.lane_switches,
                "lane_events": list(self.lane_events[-20:]),
                "eta_str": self.eta_str,
                "history_loss": list(self.history_loss[-80:]),
                "history_val": list(self.history_val[-40:]),
                "recent_logs": list(self.logs[-30:]),
                "last_result": self.last_result,
                "config": self.active_config,
            }


session = ProductionTrainingSession()


def get_hardware_info() -> Dict[str, Any]:
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Standard Processor)"
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
    steps = int(config.get("steps", 50000))
    preset = config.get("model_preset", "large_130m")
    dataset = config.get("dataset", "HuggingFaceFW/fineweb-edu")
    dataset_config = config.get("dataset_config", "sample-10BT")
    fallback_dataset = config.get("fallback_dataset", "roneneldan/TinyStories")
    start_lane = config.get("start_lane", "aggressive_seq256_zero0_gacc4")
    ckpt_interval = int(config.get("checkpoint_interval", 1000))
    val_interval = int(config.get("val_interval", 500))
    val_steps = int(config.get("val_steps", 10))
    cache_mode = config.get("cache_mode", "off")
    resume_latest = bool(config.get("resume_latest", True))
    clean = bool(config.get("clean", False))

    session.reset_for_run(config)

    # Resolve python interpreter
    venv_py = _root / ".venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    script_path = _root / "scripts" / "train_production_100k.py"

    cmd = [
        py_exec, "-u", str(script_path),
        "--steps", str(steps),
        "--model-preset", preset,
        "--dataset", dataset,
        "--fallback-dataset", fallback_dataset,
        "--start-lane", start_lane,
        "--checkpoint-interval", str(ckpt_interval),
        "--val-interval", str(val_interval),
        "--val-steps", str(val_steps),
        "--cache-mode", cache_mode,
        "--eval-window", "10",
    ]

    if dataset_config:
        cmd.extend(["--dataset-config", dataset_config])

    if resume_latest and not clean:
        cmd.append("--resume-latest")
    elif clean:
        cmd.append("--clean")

    session.log(f"Launching production worker: {' '.join(cmd[1:])}")

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

        ansi_cleaner = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        telemetry_pattern = re.compile(
            r"Step\s+(\d+)/(\d+)\s+\(\s*([\d\.]+)%\)\s+\|\s+Loss:\s+([\d\.]+)(?:\s+\(avg:\s+([\d\.]+)\))?(?:\s+\|\s+Val:\s+([\d\.]+))?\s+\|\s+Speed:\s+([\d\.]+)\s+tok/s\s+\(\s*([\d\.]+)\s+st/s\)\s+\|\s+VRAM:\s+([\d\.]+)MB\s+\(\s*([\d\.]+)%\)\s+\|\s+Lane:\s+(.*?)(?:\s+\(Switches:\s*(\d+)\))?\s+\|\s+ETA:\s+(.*)"
        )

        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            clean_line = ansi_cleaner.sub("", line_str).strip()
            session.log(clean_line)

            # Check if line records a lane governor switch
            if ">>> [LANE GOVERNOR]" in clean_line:
                gov_m = re.search(r"Step\s+([\d,]+):\s+(\S+)\s+->\s+(\S+)\s+\(Batch Size:\s*(\d+)\)(?:\s+\|\s+Switch\s+#(\d+))?\s+\|\s+(.*)", clean_line)
                if gov_m:
                    session.record_lane_switch({
                        "step": int(gov_m.group(1).replace(",", "")),
                        "from_lane": gov_m.group(2),
                        "to_lane": gov_m.group(3),
                        "batch_size": int(gov_m.group(4)),
                        "switch_num": int(gov_m.group(5)) if gov_m.group(5) else (session.lane_switches + 1),
                        "reason": gov_m.group(6).strip(),
                    })

            # Check if line contains telemetry
            match = telemetry_pattern.search(clean_line)
            if match:
                s_curr = int(match.group(1))
                s_tot = int(match.group(2))
                loss = float(match.group(4))
                avg_l = float(match.group(5)) if match.group(5) else loss
                val_l = float(match.group(6)) if match.group(6) else None
                tok_s = float(match.group(7))
                st_s = float(match.group(8))
                vram_m = float(match.group(9))
                clean_lane = match.group(11).strip("[] ")
                switches_val = int(match.group(12)) if match.group(12) else None
                clean_eta = match.group(13).strip()

                hw = get_hardware_info()
                tot_v = hw["vram_total_mb"] or 8123.0

                session.update_telemetry(
                    step=s_curr,
                    loss=loss,
                    avg_loss=avg_l,
                    val_loss=val_l,
                    speed=tok_s,
                    step_rate=st_s,
                    vram_mb=vram_m,
                    vram_tot=tot_v,
                    lane=clean_lane,
                    eta=clean_eta,
                    switches=switches_val,
                )
            elif "Step " in clean_line and "Loss:" in clean_line:
                # Robust keyword-based fallback extractor
                step_m = re.search(r"Step\s+(\d+)/(\d+)", clean_line)
                loss_m = re.search(r"Loss:\s+([\d\.]+)", clean_line)
                speed_m = re.search(r"Speed:\s+([\d\.]+)\s+tok/s", clean_line)
                vram_m = re.search(r"VRAM:\s+([\d\.]+)MB", clean_line)
                lane_m = re.search(r"Lane:\s+(\[.*?\]|\S+)", clean_line)
                switches_m = re.search(r"Switches:\s*(\d+)", clean_line)
                val_m = re.search(r"Val:\s+([\d\.]+)", clean_line)
                eta_m = re.search(r"ETA:\s+([^\s\|]+(?:\s+[^\s\|]+)*)", clean_line)
                if step_m and loss_m:
                    s_curr = int(step_m.group(1))
                    s_tot = int(step_m.group(2))
                    loss = float(loss_m.group(1))
                    tok_s = float(speed_m.group(1)) if speed_m else 0.0
                    vram_val = float(vram_m.group(1)) if vram_m else 0.0
                    lane_val = lane_m.group(1).strip("[] ") if lane_m else "AGGRESSIVE"
                    switches_val = int(switches_m.group(1)) if switches_m else None
                    val_val = float(val_m.group(1)) if val_m else None
                    eta_val = eta_m.group(1).strip() if eta_m else "--:--:--"
                    session.update_telemetry(
                        step=s_curr,
                        loss=loss,
                        avg_loss=loss,
                        val_loss=val_val,
                        speed=tok_s,
                        step_rate=0.0,
                        vram_mb=vram_val,
                        vram_tot=8123.0,
                        lane=lane_val,
                        eta=eta_val,
                        switches=switches_val,
                    )

            if session.should_stop:
                proc.terminate()
                session.log("Termination signal dispatched to training worker.")
                break

        proc.stdout.close()
        return_code = proc.wait()
        success = (return_code == 0) and not session.should_stop

        summary = {
            "exit_code": return_code,
            "steps_completed": session.current_step,
            "final_loss": session.loss,
            "val_loss": session.val_loss,
            "lane_switches": session.lane_switches,
        }
        session.finish_run(success, summary)

    except Exception as e:
        session.log(f"Worker runtime error: {str(e)}")
        session.finish_run(False, {"error": str(e)})


def _run_inference_worker(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 100,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    device: str = "auto",
) -> Dict[str, Any]:
    venv_py = _root / ".venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable
    script_path = _root / "scripts" / "run_inference.py"

    if not script_path.exists():
        return {"status": "error", "error": f"Inference script '{script_path.name}' not found."}

    # Intelligent device routing:
    # If training is actively running, force CPU to prevent VRAM contention.
    # If training is idle, use CUDA if available for maximum throughput.
    if device == "auto":
        target_device = "cpu" if session.is_running else ("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda" and session.is_running:
        target_device = "cuda" if (torch.cuda.is_available() and session.vram_used_mb < 6000) else "cpu"
    else:
        target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    cmd = [
        py_exec, "-u", str(script_path),
        "--prompt", prompt,
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
        "--top-k", str(top_k),
        "--repetition-penalty", str(repetition_penalty),
        "--device", target_device,
    ]

    timeout_sec = max(60, int(max_tokens * 0.75))
    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            cwd=str(_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - t0
        output = res.stdout.strip()
        if not output and res.stderr:
            output = res.stderr.strip()

        clean_text = ""
        meta_info = {}
        if "<<<GENERATION_OUTPUT>>>" in output and "<<<END_GENERATION_OUTPUT>>>" in output:
            parts = output.split("<<<GENERATION_OUTPUT>>>")
            gen_part = parts[1].split("<<<END_GENERATION_OUTPUT>>>")[0]
            clean_text = gen_part.strip()
        else:
            d_parts = output.split("-" * 70)
            if len(d_parts) >= 3:
                clean_text = d_parts[1].strip()
            else:
                clean_text = output

        if "<<<GENERATION_META>>>:" in output:
            meta_line = [l for l in output.splitlines() if "<<<GENERATION_META>>>:" in l]
            if meta_line:
                raw_meta = meta_line[0].split("<<<GENERATION_META>>>:")[1].strip()
                for item in raw_meta.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        meta_info[k.strip()] = v.strip()

        gen_tokens = int(meta_info.get("tokens", max_tokens))
        speed_str = meta_info.get("speed", f"{gen_tokens / max(0.01, elapsed):.1f} tok/s")
        dev_desc = "GPU (CUDA)" if target_device == "cuda" else "CPU (Zero-VRAM Contention)"

        return {
            "status": "success",
            "generated_text": clean_text if clean_text else "No output generated.",
            "tokens": gen_tokens,
            "elapsed_sec": round(elapsed, 2),
            "speed": speed_str,
            "device": dev_desc,
            "raw_output": output,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Inference execution timed out ({timeout_sec}s). Try reducing max tokens or switching to GPU."}
    except Exception as e:
        return {"status": "error", "error": f"Inference execution failed: {str(e)}"}


def _export_model_worker(export_format: str = "safetensors") -> Dict[str, Any]:
    """Exports latest trained checkpoint into standard deployment formats (SafeTensors + PyTorch)."""
    try:
        from runtime.checkpoint_manager import CheckpointManager
        ckpt_mgr = CheckpointManager(root=_root / "checkpoints")
        latest_ckpt = ckpt_mgr.latest_checkpoint_path()
        if not latest_ckpt or not Path(latest_ckpt).exists():
            return {"status": "error", "error": "No trained checkpoints found to export. Train the model for at least a few steps first."}

        payload = ckpt_mgr.load_torch_checkpoint(latest_ckpt, map_location="cpu")
        model_state = payload.get("model_state_dict", {})
        meta = payload.get("metadata", {})
        step = meta.get("step", 0)

        # Detect architecture preset
        d_model = 768
        num_layers = 12
        num_heads = 12
        detected_preset = "large_130m"
        if "token_embedding.weight" in model_state:
            d_model = model_state["token_embedding.weight"].shape[1]
            num_layers = len([k for k in model_state.keys() if k.endswith(".ln1.weight")])
            if d_model == 512 and num_layers == 6:
                detected_preset, num_heads = "medium_50m", 8
            elif d_model == 640 and num_layers == 8:
                detected_preset, num_heads = "medium_75m", 10
            elif d_model == 768 and num_layers == 8:
                detected_preset, num_heads = "medium_100m", 12
            elif d_model == 768 and num_layers == 12:
                detected_preset, num_heads = "large_130m", 12
            elif d_model == 1024 and num_layers == 16:
                detected_preset, num_heads = "xlarge_250m", 16
            elif d_model == 1280 and num_layers == 18:
                detected_preset, num_heads = "xxlarge_400m", 20
            elif d_model == 1280 and num_layers == 24:
                detected_preset, num_heads = "ultra_500m", 20

        export_dir = _root / "exports" / f"model_step_{step}_{detected_preset}"
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save PyTorch state dict
        pt_path = export_dir / "pytorch_model.bin"
        torch.save(model_state, pt_path)

        # 2. Save Safetensors if available
        safetensors_saved = False
        try:
            from safetensors.torch import save_file
            st_path = export_dir / "model.safetensors"
            st_state = {k: v.contiguous().clone().cpu() for k, v in model_state.items()}
            save_file(st_state, str(st_path))
            safetensors_saved = True
        except Exception:
            pass

        # 3. Save config.json
        config_data = {
            "architectures": ["MEMCausalLM"],
            "model_type": "mem-causal-lm",
            "preset": detected_preset,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "vocab_size": 50257,
            "max_position_embeddings": 256,
            "step": step,
            "loss": meta.get("loss", None),
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (export_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        # 4. Save Quickstart README in export folder
        readme_content = f"""# Exported MEM Model — Step {step}
Preset: {detected_preset} ({d_model} hidden size, {num_layers} layers)

## How to Load in Python:
```python
import torch
from mem_v3.runtime.lm_model import build_tiny_causal_lm

# Initialize model architecture
model = build_tiny_causal_lm(vocab_size=50257, seq_len=256, preset="{detected_preset}")

# Load weights
weights = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(weights)
model.eval()
print("Model loaded successfully!")
```
"""
        (export_dir / "README.md").write_text(readme_content, encoding="utf-8")

        file_size_mb = pt_path.stat().st_size / (1024 ** 2)
        session.log(f"Model exported successfully to: {export_dir.name} ({file_size_mb:.1f} MB)")
        return {
            "status": "success",
            "export_dir": str(export_dir),
            "folder_name": export_dir.name,
            "step": step,
            "preset": detected_preset,
            "size_mb": round(file_size_mb, 1),
            "safetensors": safetensors_saved,
        }
    except Exception as e:
        session.log(f"Export error: {str(e)}")
        return {"status": "error", "error": f"Model export failed: {str(e)}"}


# Production HTML5/CSS3 Dashboard - 100% in English
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MEM ORCHESTRATOR — Production LLM Training Studio</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --card-hover: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.2);
      --cyan: #06b6d4;
      --yellow: #f59e0b;
      --red: #ef4444;
      --purple: #8b5cf6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }
    
    header { background-color: rgba(17, 24, 39, 0.9); backdrop-filter: blur(12px); border-bottom: 1px solid var(--card-border); padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 50; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-logo { width: 34px; height: 34px; background: linear-gradient(135deg, #10b981, #06b6d4); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #fff; font-size: 16px; letter-spacing: -1px; }
    .brand-title { font-size: 17px; font-weight: 700; letter-spacing: -0.5px; }
    .brand-subtitle { font-size: 11px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }
    
    .hw-badge { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 6px 14px; font-size: 12px; display: flex; align-items: center; gap: 8px; font-weight: 600; color: #34d399; }
    .hw-dot { width: 8px; height: 8px; border-radius: 50%; background-color: #10b981; box-shadow: 0 0 10px #10b981; }

    main { flex: 1; padding: 22px 28px; display: grid; grid-template-columns: 430px 1fr; gap: 24px; max-width: 1720px; width: 100%; margin: 0 auto; }
    @media (max-width: 1080px) { main { grid-template-columns: 1fr; } }

    .card { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .card-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--card-border); padding-bottom: 12px; }

    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-label { font-size: 13px; font-weight: 600; color: var(--text); display: flex; justify-content: space-between; }
    .form-hint { font-size: 11px; color: var(--text-muted); font-weight: 400; }
    select, input[type="text"], input[type="number"] { background-color: #0c1220; border: 1px solid var(--card-border); border-radius: 8px; color: var(--text); padding: 10px 14px; font-size: 13px; outline: none; transition: border-color 0.2s; }
    select:focus, input:focus { border-color: var(--cyan); }

    .checkbox-group { display: flex; align-items: center; gap: 10px; background: #0c1220; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--card-border); cursor: pointer; }
    .checkbox-group input { width: 16px; height: 16px; accent-color: var(--accent); }
    .checkbox-label { font-size: 13px; font-weight: 600; cursor: pointer; }
    .checkbox-sub { font-size: 11px; color: var(--text-muted); display: block; margin-top: 2px; }

    .btn { cursor: pointer; border: none; border-radius: 8px; font-weight: 700; font-size: 14px; padding: 12px 20px; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn-primary { background: linear-gradient(135deg, #10b981, #059669); color: #fff; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35); }
    .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35); }
    .btn-danger:hover { opacity: 0.95; }
    .btn-secondary { background: #1e293b; color: var(--text); border: 1px solid #334155; }
    .btn-secondary:hover { background: #334155; }

    .stats-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; }
    @media (max-width: 1400px) { .stats-grid { grid-template-columns: repeat(4, 1fr); } }
    @media (max-width: 900px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
    
    .stat-box { background: #0c1220; border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 4px; }
    .stat-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }
    .stat-value { font-size: 19px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    .stat-sub { font-size: 11px; color: var(--text-muted); }

    .lane-badge { padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; display: inline-block; }
    .lane-aggressive { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .lane-fast { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
    .lane-safe { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .progress-bar-bg { width: 100%; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 4px; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #06b6d4, #10b981); width: 0%; transition: width 0.3s; }

    .vram-bar-bg { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 6px; }
    .vram-bar-fill { height: 100%; background-color: #06b6d4; width: 0%; transition: width 0.3s, background-color 0.3s; }

    .chart-container { position: relative; height: 260px; width: 100%; }
    .console-box { background-color: #040810; border: 1px solid #1a2233; border-radius: 8px; padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; color: #cbd5e1; height: 150px; overflow-y: auto; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }

    .pill-btn { background: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #94a3b8; padding: 3px 10px; font-size: 11px; cursor: pointer; transition: all 0.2s; font-weight: 500; }
    .pill-btn:hover { background: #334155; color: #f1f5f9; border-color: #06b6d4; }

    .inference-box { display: flex; flex-direction: column; gap: 14px; }
    textarea { width: 100%; height: 80px; background-color: #0c1220; border: 1px solid var(--card-border); border-radius: 8px; color: var(--text); padding: 12px; font-size: 13px; resize: none; outline: none; line-height: 1.5; }
    textarea:focus { border-color: var(--cyan); }
    .output-text { background: #040810; border: 1px solid var(--card-border); border-radius: 8px; padding: 14px; min-height: 90px; font-size: 13px; color: #38bdf8; line-height: 1.6; white-space: pre-wrap; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">M</div>
      <div>
        <div class="brand-title">MEM ORCHESTRATOR</div>
        <div class="brand-subtitle">Production LLM Training Studio &amp; Zero-OOM Engine</div>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
      <div id="hw-badge" class="hw-badge">
        <div class="hw-dot"></div>
        <span id="hw-text">Detecting Hardware...</span>
      </div>
      <button class="btn btn-secondary" onclick="shutdownApp()" style="padding: 6px 14px; font-size: 12px; border-color: rgba(239, 68, 68, 0.4); color: #f87171;">
        Shutdown Engine
      </button>
    </div>
  </header>

  <main>
    <!-- LEFT: PRODUCTION TRAINING CONFIGURATION -->
    <div class="card">
      <div class="card-title">
        <span>Training Configuration</span>
        <span style="font-size: 11px; color: var(--accent); font-weight: 600;">Production Mode</span>
      </div>

      <!-- 1. MODEL ARCHITECTURE -->
      <div class="form-group">
        <label class="form-label">
          <span>Model Architecture</span>
          <span class="form-hint" id="preset-hint">130M Parameters</span>
        </label>
        <select id="model-preset" onchange="onPresetChange()">
          <option value="large_130m" selected>large_130m — 130M Params (Recommended for 8GB VRAM)</option>
          <option value="medium_75m">medium_75m — 75M Params (High Throughput)</option>
          <option value="xlarge_250m">xlarge_250m — 250M Params (High Capacity)</option>
          <option value="xxlarge_400m">xxlarge_400m — 420M Params (High VRAM Stress Test)</option>
          <option value="ultra_500m">ultra_500m — 540M Params (8GB VRAM Limit & Zero-OOM Defense)</option>
          <option value="medium_50m">medium_50m — 50M Params (Lightweight)</option>
        </select>
      </div>

      <!-- 2. DATASET SOURCE -->
      <div class="form-group">
        <label class="form-label">
          <span>Dataset Source</span>
          <span class="form-hint">Hugging Face Hub or Local</span>
        </label>
        <select id="dataset-select" onchange="onDatasetChange()">
          <option value="HuggingFaceFW/fineweb-edu" selected>HuggingFaceFW/fineweb-edu (High-Quality Educational Web)</option>
          <option value="DKYoon/SlimPajama-6B">DKYoon/SlimPajama-6B (Curated Multi-Domain Corpus)</option>
          <option value="roneneldan/TinyStories">roneneldan/TinyStories (Synthetic Reasoning Corpus)</option>
          <option value="custom">Custom Local File (.txt, .jsonl, .csv)...</option>
        </select>
        <input type="text" id="custom-dataset-input" placeholder="e.g. C:\\Datasets\\my_corpus.jsonl" style="display: none; margin-top: 6px;">
      </div>

      <!-- 3. TARGET STEPS -->
      <div class="form-group">
        <label class="form-label">
          <span>Target Global Steps</span>
          <span class="form-hint">1k to 500k steps</span>
        </label>
        <input type="number" id="steps-input" min="100" max="1000000" step="500" value="50000">
      </div>

      <!-- 4. OPERATING LANE -->
      <div class="form-group">
        <label class="form-label">
          <span>Initial Operating Lane</span>
          <span class="form-hint">Autonomous Lane Control</span>
        </label>
        <select id="start-lane-select">
          <option value="aggressive_seq256_zero0_gacc4" selected>Aggressive Throughput (Seq 256, GradAcc 4 — Peak Speed)</option>
          <option value="fast_seq256_zero0_gacc4">Fast Performance (Seq 256, Balanced Headroom)</option>
          <option value="safe_seq256">Safe Mode (Seq 256, Minimal VRAM footprint)</option>
        </select>
      </div>

      <!-- 5. CHECKPOINT & VALIDATION CADENCE -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div class="form-group">
          <label class="form-label">
            <span>Checkpoint Cadence</span>
          </label>
          <select id="checkpoint-interval-select">
            <option value="500">Every 500 steps</option>
            <option value="1000" selected>Every 1,000 steps</option>
            <option value="2500">Every 2,500 steps</option>
            <option value="5000">Every 5,000 steps</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">
            <span>Validation Cadence</span>
          </label>
          <select id="val-interval-select">
            <option value="250">Every 250 steps</option>
            <option value="500" selected>Every 500 steps</option>
            <option value="1000">Every 1,000 steps</option>
          </select>
        </div>
      </div>

      <!-- 6. RESUME & CLEAN TOGGLES -->
      <label class="checkbox-group">
        <input type="checkbox" id="resume-checkbox" checked onchange="onResumeToggle()">
        <div>
          <span class="checkbox-label">Resume from Latest Checkpoint</span>
          <span class="checkbox-sub">Automatically restores model weights, optimizer, and step index</span>
        </div>
      </label>

      <label class="checkbox-group">
        <input type="checkbox" id="clean-checkbox" onchange="onCleanToggle()">
        <div>
          <span class="checkbox-label">Clean Start (Archive previous checkpoints)</span>
          <span class="checkbox-sub">Safely moves older checkpoints to archive folder</span>
        </div>
      </label>

      <!-- 7. ACTION CONTROLS -->
      <div style="display: flex; gap: 12px; margin-top: 4px;">
        <button class="btn btn-primary" id="btn-start" onclick="startTraining()" style="flex: 2;">
          Start Production Training
        </button>
        <button class="btn btn-danger" id="btn-stop" onclick="stopTraining()" style="flex: 1;" disabled>
          Pause &amp; Save
        </button>
      </div>

      <!-- 8. MODEL EXPORT & PACKAGING -->
      <div style="border-top: 1px solid var(--card-border); padding-top: 14px; margin-top: 8px; display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px;">Model Export &amp; Packaging</span>
          <span style="font-size: 10px; color: var(--cyan); font-weight: 600;">SafeTensors + PyTorch</span>
        </div>
        <button class="btn btn-secondary" id="btn-export" onclick="exportModel()" style="padding: 10px; font-size: 13px; border-color: rgba(6, 182, 212, 0.4);">
          Export Trained Checkpoint
        </button>
        <div id="export-status" style="font-size: 11px; display: none; padding: 10px; border-radius: 6px; background: #040810; border: 1px solid var(--card-border); line-height: 1.5;"></div>
      </div>
    </div>

    <!-- RIGHT: LIVE PRODUCTION TELEMETRY & ANALYTICS -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      
      <!-- KPI STATS OVERVIEW: 7 COMPREHENSIVE CARDS -->
      <div class="stats-grid">
        <div class="stat-box">
          <span class="stat-label">Global Step</span>
          <span class="stat-value" id="val-step">0 / 50,000</span>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="bar-progress"></div>
          </div>
          <span class="stat-sub" id="val-progress-text">0.0% Complete</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">Training Loss</span>
          <span class="stat-value" id="val-loss">--</span>
          <span class="stat-sub" id="val-avg-loss">Avg: --</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">Validation Loss</span>
          <span class="stat-value" id="val-val-loss" style="color: #38bdf8;">--</span>
          <span class="stat-sub">Generalization Metric</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">Throughput</span>
          <span class="stat-value" id="val-speed">0 tok/s</span>
          <span class="stat-sub" id="val-step-rate">0.0 steps/sec</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">VRAM Allocation</span>
          <span class="stat-value" id="val-vram">0 / 8,123 MB</span>
          <div class="vram-bar-bg">
            <div class="vram-bar-fill" id="bar-vram"></div>
          </div>
          <span class="stat-sub" id="val-vram-pct">0.0% Headroom Safe</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">Active Lane / ETA</span>
          <div style="margin-top: 2px;">
            <span class="lane-badge lane-aggressive" id="badge-lane">[IDLE]</span>
          </div>
          <span class="stat-sub" id="val-eta" style="margin-top: 4px;">ETA: --:--:--</span>
        </div>

        <div class="stat-box">
          <span class="stat-label">Lane Transitions</span>
          <span class="stat-value" id="val-switches" style="color: #38bdf8;">0 switches</span>
          <span class="stat-sub" id="val-switches-sub">Zero-OOM Guard Active</span>
        </div>
      </div>

      <!-- DUAL CHARTS CARD -->
      <div class="card">
        <div class="card-title">
          <span>Real-Time Production Analytics</span>
          <span style="font-size: 11px; color: var(--text-muted);">Loss &amp; Throughput Curves</span>
        </div>
        <div class="chart-container">
          <canvas id="trainingChart"></canvas>
        </div>
      </div>

      <!-- AUTONOMOUS LANE ADAPTATION LOG -->
      <div class="card">
        <div class="card-title">
          <span>Autonomous Lane Adaptation Log</span>
          <span id="lane-switches-badge" style="font-size: 11px; padding: 3px 8px; border-radius: 4px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-weight: 600;">0 Switches</span>
        </div>
        <div id="lane-history-list" style="max-height: 120px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; font-size: 12px; padding-right: 4px;">
          <div style="color: var(--text-muted); font-size: 12px; font-style: italic;">No lane switches yet. The orchestrator is running on the primary calibrated lane.</div>
        </div>
      </div>

      <!-- LIVE TERMINAL CONSOLE -->
      <div class="card">
        <div class="card-title">
          <span>Engine Execution Stream</span>
          <button class="btn btn-secondary" onclick="clearLogs()" style="padding: 4px 10px; font-size: 11px;">Clear</button>
        </div>
        <div class="console-box" id="console-logs">Engine ready. Configure hyperparameters on the left and click 'Start Production Training'.</div>
      </div>

      <!-- INTERACTIVE INFERENCE PLAYGROUND -->
      <div class="card">
        <div class="card-title">
          <span>Inference Playground (Checkpoint Verification)</span>
          <span style="font-size: 11px; color: var(--accent); font-weight: 600;">Autonomous Weight Generation</span>
        </div>
        <div class="inference-box">
          <div class="form-group">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
              <label class="form-label" style="margin: 0;">Input Prompt</label>
              <div style="display: flex; gap: 6px;">
                <button type="button" class="pill-btn" onclick="setPrompt('Once upon a time in a forgotten forest,')">Story</button>
                <button type="button" class="pill-btn" onclick="setPrompt('The artificial intelligence system discovered that')">AI</button>
                <button type="button" class="pill-btn" onclick="setPrompt('The fundamental laws of astrophysics state that')">Physics</button>
              </div>
            </div>
            <textarea id="prompt-input" placeholder="Enter prompt to test model text generation (e.g. 'Once upon a time in a distant realm...')..."></textarea>
          </div>

          <!-- Playground Controls -->
          <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; background: #0c1220; padding: 14px; border-radius: 8px; border: 1px solid var(--card-border);">
            <!-- Max Tokens Slider & Input -->
            <div class="form-group">
              <div class="form-label">
                <span>Max Tokens</span>
                <span id="tokens-display" style="color: var(--cyan); font-weight: 700;">100</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <input type="range" id="tokens-range" min="10" max="1024" step="10" value="100" oninput="syncTokens(this.value)" style="flex: 1;">
                <input type="number" id="tokens-input" min="10" max="2048" step="10" value="100" oninput="syncTokens(this.value)" style="width: 70px; text-align: center; padding: 6px;">
              </div>
              <span class="form-hint">Tokens to generate (10 - 2,048)</span>
            </div>

            <!-- Temperature Slider & Input -->
            <div class="form-group">
              <div class="form-label">
                <span>Temperature</span>
                <span id="temp-display" style="color: var(--yellow); font-weight: 700;">0.70</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <input type="range" id="temp-range" min="0.05" max="2.00" step="0.05" value="0.70" oninput="syncTemp(this.value)" style="flex: 1;">
                <input type="number" id="temp-input" min="0.05" max="2.00" step="0.05" value="0.70" oninput="syncTemp(this.value)" style="width: 70px; text-align: center; padding: 6px;">
              </div>
              <span class="form-hint" id="temp-tag">Balanced (0.70)</span>
            </div>

            <!-- Device Selector -->
            <div class="form-group">
              <div class="form-label">
                <span>Inference Device</span>
              </div>
              <select id="infer-device" style="padding: 7px 10px;">
                <option value="auto" selected>Auto (CPU if training, GPU if idle)</option>
                <option value="cuda">GPU CUDA (Ultra Fast)</option>
                <option value="cpu">CPU (Zero-VRAM Contention)</option>
              </select>
              <span class="form-hint">Prevents memory clash during training</span>
            </div>
          </div>

          <div style="display: flex; gap: 10px; align-items: center;">
            <button class="btn btn-primary" id="btn-infer" onclick="runInference()" style="padding: 9px 22px; font-size: 13px;">
              Generate Text
            </button>
            <button class="btn btn-secondary" onclick="clearInference()" style="padding: 9px 14px; font-size: 12px;">
              Clear
            </button>
            <button class="btn btn-secondary" id="btn-copy" onclick="copyInference()" style="padding: 9px 14px; font-size: 12px; margin-left: auto;">
              Copy Text
            </button>
          </div>

          <div>
            <div class="output-text" id="infer-output">Generated text output will appear here. Configure prompt, tokens, and temperature above, then click 'Generate Text'.</div>
            <div id="infer-meta" style="margin-top: 6px; font-size: 11px; color: var(--text-muted); display: none;"></div>
          </div>
        </div>
      </div>

    </div>
  </main>

  <script>
    let chartInstance = null;

    function initChart() {
      const ctx = document.getElementById('trainingChart').getContext('2d');
      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Train Loss',
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.08)',
              data: [],
              yAxisID: 'yLoss',
              tension: 0.2,
              borderWidth: 2,
              pointRadius: 0,
            },
            {
              label: 'Val Loss',
              borderColor: '#38bdf8',
              backgroundColor: 'transparent',
              data: [],
              yAxisID: 'yLoss',
              tension: 0.2,
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 3,
            },
            {
              label: 'Tokens/s',
              borderColor: '#f59e0b',
              backgroundColor: 'transparent',
              data: [],
              yAxisID: 'ySpeed',
              tension: 0.2,
              borderWidth: 1.5,
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
            ySpeed: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#f59e0b' } }
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
          hwText.textContent = `${hw.device_name} — ${hw.vram_total_gb} GB VRAM (CUDA Active)`;
        } else {
          hwText.textContent = `${hw.device_name} (${hw.cpu_count} Cores | ${hw.ram_free_gb} GB Free RAM)`;
          document.querySelector('.hw-dot').style.backgroundColor = '#f59e0b';
        }
      } catch (e) {
        console.error("Hardware fetch failed", e);
      }
    }

    function onPresetChange() {
      const val = document.getElementById('model-preset').value;
      const hint = document.getElementById('preset-hint');
      if (val === 'large_130m') hint.textContent = '130M Params (12 layers, 12 heads, 768 dim)';
      if (val === 'medium_75m') hint.textContent = '75M Params (8 layers, 10 heads, 640 dim)';
      if (val === 'xlarge_250m') hint.textContent = '250M Params (16 layers, 16 heads, 1024 dim)';
      if (val === 'xxlarge_400m') hint.textContent = '420M Params (18 layers, 20 heads, 1280 dim — ~5.4GB VRAM)';
      if (val === 'ultra_500m') hint.textContent = '540M Params (24 layers, 20 heads, 1280 dim — 8GB Ceiling)';
      if (val === 'medium_50m') hint.textContent = '50M Params (6 layers, 8 heads, 512 dim)';
    }

    function onDatasetChange() {
      const val = document.getElementById('dataset-select').value;
      const customInp = document.getElementById('custom-dataset-input');
      customInp.style.display = (val === 'custom') ? 'block' : 'none';
    }

    function onResumeToggle() {
      if (document.getElementById('resume-checkbox').checked) {
        document.getElementById('clean-checkbox').checked = false;
      }
    }

    function onCleanToggle() {
      if (document.getElementById('clean-checkbox').checked) {
        document.getElementById('resume-checkbox').checked = false;
      }
    }

    function syncTokens(val) {
      val = Math.max(10, Math.min(2048, parseInt(val) || 100));
      document.getElementById('tokens-range').value = Math.min(1024, val);
      document.getElementById('tokens-input').value = val;
      document.getElementById('tokens-display').textContent = val;
    }

    function syncTemp(val) {
      val = Math.max(0.05, Math.min(2.0, parseFloat(val) || 0.7));
      const strVal = val.toFixed(2);
      document.getElementById('temp-range').value = strVal;
      document.getElementById('temp-input').value = strVal;
      document.getElementById('temp-display').textContent = strVal;

      const tagEl = document.getElementById('temp-tag');
      if (val < 0.35) {
        tagEl.textContent = `Focused / Deterministic (${strVal})`;
        tagEl.style.color = '#38bdf8';
      } else if (val <= 0.90) {
        tagEl.textContent = `Balanced Natural (${strVal})`;
        tagEl.style.color = '#10b981';
      } else {
        tagEl.textContent = `Creative / Stochastic (${strVal})`;
        tagEl.style.color = '#f59e0b';
      }
    }

    function setPrompt(text) {
      document.getElementById('prompt-input').value = text;
      document.getElementById('prompt-input').focus();
    }

    function clearInference() {
      document.getElementById('infer-output').textContent = 'Output cleared.';
      document.getElementById('infer-meta').style.display = 'none';
    }

    function copyInference() {
      const text = document.getElementById('infer-output').textContent;
      if (!text || text.startsWith('Generated text output') || text === 'Output cleared.') return;
      navigator.clipboard.writeText(text);
      const btn = document.getElementById('btn-copy');
      const prev = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = prev; }, 1500);
    }

    async function startTraining() {
      const preset = document.getElementById('model-preset').value;
      let dataset = document.getElementById('dataset-select').value;
      if (dataset === 'custom') {
        dataset = document.getElementById('custom-dataset-input').value.trim();
        if (!dataset) {
          alert('Please enter a valid local dataset path.');
          return;
        }
      }

      const steps = parseInt(document.getElementById('steps-input').value) || 50000;
      const startLane = document.getElementById('start-lane-select').value;
      const ckptInterval = parseInt(document.getElementById('checkpoint-interval-select').value) || 1000;
      const valInterval = parseInt(document.getElementById('val-interval-select').value) || 500;
      const resumeLatest = document.getElementById('resume-checkbox').checked;
      const clean = document.getElementById('clean-checkbox').checked;

      const payload = {
        model_preset: preset,
        dataset: dataset,
        dataset_config: (dataset === 'HuggingFaceFW/fineweb-edu') ? 'sample-10BT' : '',
        fallback_dataset: 'roneneldan/TinyStories',
        steps: steps,
        start_lane: startLane,
        checkpoint_interval: ckptInterval,
        val_interval: valInterval,
        val_steps: 10,
        cache_mode: 'off',
        resume_latest: resumeLatest,
        clean: clean,
      };

      try {
        const res = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'started') {
          document.getElementById('btn-start').disabled = true;
          document.getElementById('btn-stop').disabled = false;
        } else {
          alert(data.error || 'Failed to start training run.');
        }
      } catch (e) {
        alert('Server connection error: ' + e);
      }
    }

    async function stopTraining() {
      if (confirm('Pause training and commit atomic checkpoint to disk?')) {
        try {
          await fetch('/api/stop', { method: 'POST' });
          document.getElementById('btn-stop').disabled = true;
        } catch (e) {
          alert('Error dispatching stop signal: ' + e);
        }
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/state');
        const st = await res.json();

        // Control buttons
        document.getElementById('btn-start').disabled = st.is_running;
        document.getElementById('btn-stop').disabled = !st.is_running;

        // Steps & Progress
        document.getElementById('val-step').textContent = `${st.current_step.toLocaleString()} / ${st.total_steps.toLocaleString()}`;
        document.getElementById('bar-progress').style.width = `${st.progress_pct}%`;
        document.getElementById('val-progress-text').textContent = `${st.progress_pct}% Complete`;

        // Loss
        document.getElementById('val-loss').textContent = (st.loss > 0) ? st.loss.toFixed(4) : '--';
        document.getElementById('val-avg-loss').textContent = (st.avg_loss > 0) ? `Avg: ${st.avg_loss.toFixed(4)}` : 'Avg: --';
        document.getElementById('val-val-loss').textContent = (st.val_loss !== null) ? st.val_loss.toFixed(4) : '--';

        // Speed
        document.getElementById('val-speed').textContent = `${Math.round(st.speed_tokens_sec).toLocaleString()} tok/s`;
        document.getElementById('val-step-rate').textContent = `${st.step_rate.toFixed(1)} steps/sec`;

        // VRAM
        document.getElementById('val-vram').textContent = `${Math.round(st.vram_used_mb)} / ${Math.round(st.vram_total_mb)} MB`;
        document.getElementById('bar-vram').style.width = `${Math.min(100, st.vram_pct)}%`;
        if (st.vram_pct > 88) document.getElementById('bar-vram').style.backgroundColor = '#ef4444';
        else if (st.vram_pct > 72) document.getElementById('bar-vram').style.backgroundColor = '#f59e0b';
        else document.getElementById('bar-vram').style.backgroundColor = '#06b6d4';
        document.getElementById('val-vram-pct').textContent = `${st.vram_pct}% Headroom Safe`;

        // Lane & ETA
        const badge = document.getElementById('badge-lane');
        badge.textContent = `[${st.active_lane}]`;
        if (st.active_lane.includes('AGGRESSIVE') || st.active_lane.includes('PEAK')) badge.className = 'lane-badge lane-aggressive';
        else if (st.active_lane.includes('SAFE')) badge.className = 'lane-badge lane-safe';
        else badge.className = 'lane-badge lane-fast';

        document.getElementById('val-eta').textContent = `ETA: ${st.eta_str}`;

        // Lane switches KPI
        const switchCount = st.lane_switches || 0;
        document.getElementById('val-switches').textContent = `${switchCount} switch${switchCount === 1 ? '' : 'es'}`;
        document.getElementById('val-switches-sub').textContent = switchCount > 0 ? `Dynamic Adaptation Active` : `Zero-OOM Guard Active`;
        document.getElementById('lane-switches-badge').textContent = `${switchCount} Switch${switchCount === 1 ? '' : 'es'}`;

        // Lane Adaptation Log Table
        if (st.lane_events && st.lane_events.length > 0) {
          const listEl = document.getElementById('lane-history-list');
          listEl.innerHTML = st.lane_events.map(ev => `
            <div style="background: #0c1220; border: 1px solid var(--card-border); border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <span style="font-weight: 700; color: #38bdf8;">Step ${ev.step.toLocaleString()}</span>: 
                <span style="color: #94a3b8;">${ev.from_lane}</span> &rarr; <span style="color: #10b981; font-weight: 600;">${ev.to_lane}</span>
                <span style="color: var(--text-muted); font-size: 11px; margin-left: 6px;">(${ev.reason})</span>
              </div>
              <span style="font-size: 11px; color: var(--text-muted);">${ev.time}</span>
            </div>
          `).reverse().join('');
        }

        // Update Charts
        if (st.history_loss && st.history_loss.length > 0) {
          chartInstance.data.labels = st.history_loss.map(h => h.step);
          chartInstance.data.datasets[0].data = st.history_loss.map(h => h.loss);
          chartInstance.data.datasets[2].data = st.history_loss.map(h => h.speed);

          if (st.history_val && st.history_val.length > 0) {
            const valMap = {};
            st.history_val.forEach(v => { valMap[v.step] = v.val_loss; });
            chartInstance.data.datasets[1].data = st.history_loss.map(h => valMap[h.step] !== undefined ? valMap[h.step] : null);
          }
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
      if (!prompt) {
        alert('Please enter a prompt to generate text.');
        return;
      }
      const tokens = parseInt(document.getElementById('tokens-input').value) || 100;
      const temp = parseFloat(document.getElementById('temp-input').value) || 0.70;
      const device = document.getElementById('infer-device').value;

      const btn = document.getElementById('btn-infer');
      const out = document.getElementById('infer-output');
      const metaEl = document.getElementById('infer-meta');

      btn.disabled = true;
      btn.textContent = 'Generating...';
      out.textContent = `Executing autoregressive inference (${tokens} tokens, Temp ${temp.toFixed(2)}) using latest checkpoint...`;
      metaEl.style.display = 'none';

      try {
        const res = await fetch('/api/inference', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            temperature: temp,
            max_tokens: tokens,
            device: device,
          })
        });
        const data = await res.json();
        if (data.status === 'success' || data.generated_text) {
          out.textContent = data.generated_text || 'No text returned.';
          metaEl.textContent = `Generated ${data.tokens || tokens} tokens in ${data.elapsed_sec || '?'}s (${data.speed || '? tok/s'}) on ${data.device || 'CPU'}`;
          metaEl.style.display = 'block';
        } else {
          out.textContent = data.error || 'Inference execution failed.';
        }
      } catch (e) {
        out.textContent = `Inference failed: ${e}`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Text';
      }
    }

    function clearLogs() {
      document.getElementById('console-logs').textContent = '';
    }

    async function shutdownApp() {
      if (confirm('Shutdown MEM Orchestrator engine and release GPU memory?')) {
        try {
          await fetch('/api/shutdown', { method: 'POST' });
        } catch (e) {}
        document.body.innerHTML = `
          <div style="display: flex; height: 100vh; align-items: center; justify-content: center; flex-direction: column; gap: 16px; background: #0b0f19; color: #94a3b8; font-family: sans-serif; text-align: center;">
            <h2 style="color: #f1f5f9; font-size: 24px;">Engine Shutdown Complete</h2>
            <p>Background processes terminated and GPU VRAM safely freed.</p>
            <p style="font-size: 13px; color: #64748b;">You can safely close this window now.</p>
          </div>
        `;
        setTimeout(() => { window.close(); }, 1500);
      }
    async function exportModel() {
      const btn = document.getElementById('btn-export');
      const status = document.getElementById('export-status');
      btn.disabled = true;
      btn.textContent = 'Packaging Model...';
      status.style.display = 'block';
      status.style.color = '#38bdf8';
      status.textContent = 'Extracting checkpoint weights and building deployment package...';

      try {
        const res = await fetch('/api/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format: 'safetensors' })
        });
        const data = await res.json();
        if (data.status === 'success') {
          status.style.color = '#10b981';
          status.innerHTML = `<strong>Export Complete!</strong><br>` +
            `<span style="color: #94a3b8;">Preset: <b>${data.preset}</b> | Step: <b>${data.step}</b> | Size: <b>${data.size_mb} MB</b></span><br>` +
            `<span style="color: #cbd5e1; word-break: break-all; font-family: monospace; font-size: 10px;">Path: ${data.export_dir}</span>`;
        } else {
          status.style.color = '#ef4444';
          status.textContent = data.error || 'Export failed.';
        }
      } catch (e) {
        status.style.color = '#ef4444';
        status.textContent = 'Connection error during export: ' + e;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Export Trained Checkpoint';
      }
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
        session.record_activity()
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

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        session.record_activity()
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed.path == "/api/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "shutting_down"}).encode("utf-8"))
            session.log("Shutdown signal received from GUI.")
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
            return

        if parsed.path == "/api/start":
            if session.is_running:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "A training process is already running."}).encode("utf-8"))
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
            tokens = int(payload.get("max_tokens", 100))
            top_k = int(payload.get("top_k", 40))
            rep_pen = float(payload.get("repetition_penalty", 1.15))
            dev = payload.get("device", "auto")

            result = _run_inference_worker(
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_k=top_k,
                repetition_penalty=rep_pen,
                device=dev,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed.path == "/api/export":
            export_format = payload.get("format", "safetensors")
            export_result = _export_model_worker(export_format)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(export_result).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_control_center(port: int = 8089) -> int:
    ThreadingHTTPServer.allow_reuse_address = True
    for p in [port, 8089, 8090, 8080, 8888]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), ControlCenterHandler)
            print(f"[*] MEM Production Control Center listening at: http://127.0.0.1:{p}")
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return p
        except Exception:
            continue
    raise RuntimeError("Could not allocate a port for the server.")


if __name__ == "__main__":
    p = run_control_center(8089)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Halted.")
