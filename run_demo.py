"""
MEM ORCHESTRATOR - 1-Click Interactive Demo & Zero-OOM Defense Launcher.
Automatically discovers Python environment, initializes hardware calibration,
launches the Adaptive Lane Runner, and opens the live Telemetry Dashboard in your browser.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Resolve workspace root
_script_dir = Path(__file__).resolve().parent
_mem_v3_dir = _script_dir / "mem_v3" if (_script_dir / "mem_v3").exists() else _script_dir

if str(_mem_v3_dir) not in sys.path:
    sys.path.insert(0, str(_mem_v3_dir))

# Configure Windows native SSL certificates & sanitize cert environment
for _ca_env in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
    _val = os.environ.get(_ca_env)
    if _val and not os.path.exists(_val):
        os.environ.pop(_ca_env, None)

try:
    import truststore
    truststore.inject_into_ssl()
    import urllib3.util.ssl_
    urllib3.util.ssl_.create_urllib3_context = truststore.SSLContext
except Exception:
    pass

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="MEM Orchestrator 1-Click Interactive Demo (Adaptive Lane Runner & Zero-OOM Guard)"
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of demo training steps (default: 200)")
    parser.add_argument("--model-preset", default="medium_75m", choices=["medium_75m", "large_130m", "xlarge_250m"], help="Model preset (default: medium_75m)")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="Dataset name")
    parser.add_argument("--dataset-config", default="sample-10BT", help="Dataset configuration")
    parser.add_argument("--shock-interval", type=int, default=60, help="Steps between chaos memory shocks (default: 60)")
    parser.add_argument("--shock-duration", type=int, default=30, help="Duration in steps of each shock (default: 30)")
    parser.add_argument("--shock-size-mb", type=int, default=1200, help="VRAM shock size in MB (default: 1200)")
    parser.add_argument("--no-shock", action="store_true", help="Disable synthetic chaos shocks for pure training")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser dashboard")
    parser.add_argument("--port", type=int, default=8089, help="Port for the telemetry dashboard (default: 8089)")
    args = parser.parse_args()

    # Hardware detection
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Zero-GPU Fallback Mode)"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if has_cuda else 0.0

    print("\n" + "=" * 80)
    print(f"{BOLD}{CYAN}  🚀 MEM LLM ORCHESTRATOR — 1-CLICK INTERACTIVE DEMO{RESET}")
    print(f"  {BOLD}Autonomous GPU Memory Modulation, Dynamic Lane Switching & Zero-OOM Defense{RESET}")
    print("=" * 80)
    print(f"  Hardware:      {BOLD}{device_name}{RESET} ({vram_gb:.1f} GB VRAM)" if has_cuda else f"  Hardware:      {BOLD}CPU{RESET}")
    print(f"  Model Preset:  {args.model_preset} (Causal LM)")
    print(f"  Target Steps:  {args.steps:,} steps")
    print(f"  Dashboard:     http://localhost:{args.port}")
    if not args.no_shock:
        print(f"  Chaos Defense: Active (+{args.shock_size_mb} MB VRAM shock every {args.shock_interval} steps)")
    else:
        print(f"  Chaos Defense: Disabled (Standard Adaptive Training)")
    print("=" * 80 + "\n")

    # Start live telemetry dashboard in background
    try:
        from application.dashboard import run_dashboard_server
        dash_thread = threading.Thread(
            target=run_dashboard_server,
            kwargs={"port": args.port},
            daemon=True,
        )
        dash_thread.start()
        print(f"{GREEN}>>> Telemetry Dashboard active at: http://localhost:{args.port}{RESET}")

        if not args.no_browser:
            def _open():
                time.sleep(1.5)
                try:
                    webbrowser.open(f"http://localhost:{args.port}")
                except Exception:
                    pass
            threading.Thread(target=_open, daemon=True).start()
    except Exception as e:
        print(f"{YELLOW}[Dashboard Warning] Could not start embedded dashboard: {e}{RESET}")

    # Launch demo video chaos defense script
    from scripts.demo_video_chaos_defense import main as run_chaos_demo
    sys.argv = [
        sys.argv[0],
        "--steps", str(args.steps),
        "--model-preset", args.model_preset,
        "--dataset", args.dataset,
        "--dataset-config", args.dataset_config,
        "--shock-interval", str(args.shock_interval),
        "--shock-duration", str(args.shock_duration),
        "--shock-size-mb", str(args.shock_size_mb),
        "--dashboard-port", str(args.port),
    ]
    if args.no_shock:
        sys.argv.append("--no-shock")

    run_chaos_demo()


if __name__ == "__main__":
    main()
