param(
    [int]$steps = 100000,
    [string]$model = 'large_130m',
    [int]$eval_window = 25,
    [switch]$resume,
    [switch]$dashboard
)
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path "$ScriptDir\mem_v3\.venv\Scripts\python.exe") {
    $WorkDir = "$ScriptDir\mem_v3"
} elseif (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $WorkDir = "$ScriptDir"
} else {
    $WorkDir = "C:\Users\vasco\.gemini\antigravity\scratch\mem-llm-orchestrator\mem_v3"
}

Set-Location $WorkDir
$pythonPath = Join-Path $WorkDir ".venv\Scripts\python.exe"
$trainScript = Join-Path $WorkDir "scripts\train_production_100k.py"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Iniciando Treino Real de Producao (100.000 Steps) no PowerShell" -ForegroundColor Cyan
Write-Host "  Diretorio: $WorkDir" -ForegroundColor Gray
Write-Host "  Python:    $pythonPath" -ForegroundColor Gray
Write-Host "  Modelo:    $model" -ForegroundColor Gray
Write-Host "========================================================================`n" -ForegroundColor Cyan

$extraArgs = @("--steps", $steps, "--model-preset", $model, "--eval-window", $eval_window, "--checkpoint-interval", 1000)
if ($resume) {
    $extraArgs += "--resume-latest"
}
if ($dashboard) {
    $extraArgs += "--dashboard"
}

& $pythonPath $trainScript @extraArgs
