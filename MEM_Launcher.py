"""
MEM ORCHESTRATOR - Production Desktop Application Launcher.
Initializes the Production Control Center and launches the UI in native desktop window mode.
100% in English.
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


def kill_stale_processes():
    try:
        import psutil
        curr_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] != curr_pid and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'control_center.py' in cmdline or 'train_production_100k.py' in cmdline:
                        proc.kill()
            except Exception:
                pass
    except Exception:
        pass


def main():
    print("=" * 66)
    print("  MEM ORCHESTRATOR - PRODUCTION LLM TRAINING STUDIO")
    print("=" * 66)

    # Clean any stale orphan processes to guarantee clean port and updated code
    kill_stale_processes()

    from mem_v3.application.control_center import run_control_center, session

    actual_port = run_control_center(PORT)
    app_url = f"http://127.0.0.1:{actual_port}"
    print(f"[*] Initializing backend engine at {app_url}...")

    # Wait for the HTTP server to respond before opening the browser window
    if not wait_for_server(actual_port, timeout=15.0):
        print(f"[ERROR] Backend engine failed to respond on port {actual_port}.")
        sys.exit(1)

    print("[*] Backend engine online and accepting requests.")

    # Open the UI in desktop app mode or standard browser
    browser_exe = find_browser()
    if browser_exe:
        print(f"[*] Opening dedicated desktop window via: {os.path.basename(browser_exe)}")
        cmd = [
            browser_exe,
            f"--app={app_url}",
            "--window-size=1400,900",
        ]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            print(f"[WARNING] Failed to open via {browser_exe}: {e}. Opening default browser...")
            webbrowser.open(app_url)
    else:
        print("[*] Opening default web browser...")
        webbrowser.open(app_url)

    print(f"[*] Control Center running at: {app_url}")
    print("[*] Press Ctrl+C in terminal or click 'Shutdown Engine' in the GUI to terminate.")

    # Main watchdog loop: keeps server alive and detects when user closes window
    try:
        while True:
            time.sleep(2)
            # If client opened the dashboard and then closed all windows
            if session.has_ever_connected():
                idle_seconds = time.time() - session.get_last_activity()
                if idle_seconds > 300.0:
                    print("[*] Inactivity detected (window closed for >5 minutes). Terminating engine...")
                    break
    except KeyboardInterrupt:
        print("[*] Halted by user.")

    # Clean termination
    os._exit(0)


if __name__ == "__main__":
    main()
