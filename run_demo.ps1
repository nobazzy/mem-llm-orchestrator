param(
    [int]$steps = 300,
    [switch]$dashboard
)
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path "$ScriptDir\mem_v3\.venv\Scripts\python.exe") {
    $WorkDir = "$ScriptDir\mem_v3"
} elseif (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $WorkDir = "$ScriptDir"
} else {
    $WorkDir = "C:\Users\vasco\.geminintigravity\scratch\mem-llm-orchestrator\mem_v3"
}

Set-Location $WorkDir
$pythonPath = Join-Path $WorkDir ".venv\Scripts\python.exe"
$demoScript = Join-Path $WorkDir "scripts\demo_video_chaos_defense.py"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Iniciando MEM Orchestrator no PowerShell" -ForegroundColor Cyan
Write-Host "  Diretório: $WorkDir" -ForegroundColor Gray
Write-Host "  Python:    $pythonPath" -ForegroundColor Gray
Write-Host "==========================================================`n" -ForegroundColor Cyan

$extraArgs = @("--steps", $steps, "--shock-interval", 100, "--shock-duration", 50)
if ($dashboard) {
    $extraArgs += "--dashboard"
}

& $pythonPath $demoScript @extraArgs
