"""
MEM ORCHESTRATOR - Desktop Application Launcher.
Starts the Control Center server and launches the UI in native desktop window mode.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import subprocess
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PORT = 8089


def wait_for_server(port: int, timeout: float = 15.0) -> bool:
    start = time.time()
    url = f"http://127.0.0.1:{port}/api/state"
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MEM-Launcher"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    print("=" * 60)
    print("  MEM LLM ORCHESTRATOR - CONTROL CENTER DESKTOP")
    print("=" * 60)

    from mem_v3.application.control_center import run_control_center

    actual_port = run_control_center(PORT)
    app_url = f"http://127.0.0.1:{actual_port}"
    print(f"[*] Inicializando servidor backend em {app_url}...")

    # Wait for the HTTP server to answer before opening the browser window
    if not wait_for_server(actual_port, timeout=15.0):
        print(f"[ERRO] O servidor backend nao respondeu na porta {actual_port}.")
        sys.exit(1)

    print("[*] Servidor backend pronto e respondendo.")

    browser_exe = find_browser()
    if browser_exe:
        print(f"[*] Abrindo janela de aplicativo via {os.path.basename(browser_exe)}...")
        # Use an isolated user-data-dir so Edge/Chrome runs as a standalone desktop process
        profile_dir = Path(tempfile.gettempdir()) / "mem_orchestrator_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            browser_exe,
            f"--app={app_url}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1366,860",
            "--disable-extensions",
            "--disable-plugins",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
            print("[*] Janela da aplicacao encerrada pelo usuario.")
        except KeyboardInterrupt:
            proc.terminate()
    else:
        print("[*] Abrindo navegador padrao...")
        webbrowser.open(app_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
