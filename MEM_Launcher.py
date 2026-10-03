"""
MEM ORCHESTRATOR - Desktop Application Launcher.
Starts the Control Center server and launches the UI in native desktop window mode.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PORT = 8089


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


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
    print("  MEM LLM ORCHESTRATOR - DESKTOP CONTROL CENTER")
    print("=" * 60)

    # Start Control Center server
    from mem_v3.application.control_center import run_control_center

    actual_port = run_control_center(PORT)
    app_url = f"http://127.0.0.1:{actual_port}"
    print(f"[*] Servidor backend ativo em: {app_url}")

    # Launch in App Mode (borderless standalone window)
    browser_exe = find_browser()
    if browser_exe:
        print(f"[*] Abrindo janela dedicada via: {os.path.basename(browser_exe)}")
        cmd = [
            browser_exe,
            f"--app={app_url}",
            "--window-size=1366,860",
            "--disable-extensions",
            "--disable-plugins",
        ]
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
            print("[*] Janela da aplicacao encerrada pelo usuario.")
        except KeyboardInterrupt:
            pass
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
