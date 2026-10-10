# ==============================================================================
# MEM Orchestrator v3 — Preset 250M Parâmetros | 10.000.000 Steps
# Dataset Ruidoso: allenai/c4 (en) [Web Corpus Desestruturado com Ruído Real]
# Otimizado para NVIDIA RTX 5060 Ti 8GB (SDPA Flash Attention + FP16 AMP)
# ==============================================================================

$ErrorActionPreference = "Stop"

# Sanitizar variáveis de CA no ambiente local do PowerShell
if ($env:CURL_CA_BUNDLE -and !(Test-Path $env:CURL_CA_BUNDLE)) {
    $env:CURL_CA_BUNDLE = ""
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
Set-Location $BaseDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  INICIANDO TREINO DE RESISTÊNCIA DE LONGA DURAÇÃO (10M STEPS)" -ForegroundColor Yellow
Write-Host "  Modelo: xlarge_250m (~255M Parâmetros, 16 camadas, d=1024, 16 heads)" -ForegroundColor Green
Write-Host "  Dataset: allenai/c4 (config: en, Streaming contínuo com ruído web)" -ForegroundColor Green
Write-Host "  Target: 10,000,000 steps" -ForegroundColor Green
Write-Host "  Sequência: 256 tokens | Precisão: FP16 | Start Lane: safe_seq256" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

python main.py `
  --steps 10000000 `
  --model-preset xlarge_250m `
  --dataset-name allenai/c4 `
  --dataset-config en `
  --sequence-length 256 `
  --start-lane safe_seq256 `
  --precision fp16 `
  --resume-latest
