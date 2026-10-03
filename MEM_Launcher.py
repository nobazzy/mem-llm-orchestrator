"""
MEM ORCHESTRATOR - Desktop Application Launcher.
Starts the Control Center server and launches the UI in native desktop window mode.
"""

from __future__ import annotations

import os
import sys
import time
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

    from mem_v3.application.control_center import run_control_center, session

    actual_port = run_control_center(PORT)
    app_url = f"http://127.0.0.1:{actual_port}"
    print(f"[*] Inicializando servidor backend em {app_url}...")

    # Wait for the HTTP server to answer before opening the browser window
    if not wait_for_server(actual_port, timeout=15.0):
        print(f"[ERRO] O servidor backend nao respondeu na porta {actual_port}.")
        sys.exit(1)

    print("[*] Servidor backend pronto e respondendo.")

    # Open the UI in desktop app mode or standard browser
    browser_exe = find_browser()
    if browser_exe:
        print(f"[*] Abrindo janela dedicada via: {os.path.basename(browser_exe)}")
        cmd = [
            browser_exe,
            f"--app={app_url}",
            "--window-size=1366,860",
        ]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            print(f"[AVISO] Falha ao abrir via {browser_exe}: {e}. Abrindo padrao...")
            webbrowser.open(app_url)
    else:
        print("[*] Abrindo navegador padrao...")
        webbrowser.open(app_url)

    print(f"[*] Painel de controle em execucao: {app_url}")
    print("[*] Pressione Ctrl+C para encerrar ou use o botao 'Fechar Servidor' na interface.")

    # Main watchdog loop: keeps server alive and detects when user closes window
    try:
        while True:
            time.sleep(2)
            # If client opened the dashboard and then closed all windows
            if session.has_ever_connected():
                idle_seconds = time.time() - session.get_last_activity()
                if idle_seconds > 25.0:
                    print("[*] Inatividade detectada (janela fechada). Encerrando servidor...")
                    break
    except KeyboardInterrupt:
        print("[*] Encerrado pelo usuario.")

    # Clean termination
    os._exit(0)


if __name__ == "__main__":
    main()
