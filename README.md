# ⚡ MEM v3 — Model Execution Manager & LLM Training Orchestrator

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%2B%20CPU)-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Zero--OOM-00599C)](https://www.deepspeed.ai/)
[![Tests](https://img.shields.io/badge/Tests-33%2F33%20Passing-brightgreen)](tests/)
[![Validation](https://img.shields.io/badge/Validation-1M%20Steps%20Zero%20OOM-brightgreen)](mem_v3/EVALUATION.md)
[![Status](https://img.shields.io/badge/Status-Functional%20%2F%20Active-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](mem_v3/LICENSE)

> **Autonomous, zero-OOM orchestration engine for sustained Large Language Model pre-training and fine-tuning with deterministic policy enforcement.**

---

### 🟢 Status do Projeto
* **Ambiente de Alta Performance:** Linux / WSL2 (Ubuntu 22.04+) com GPU NVIDIA (RTX 30xx/40xx/50xx / Blackwell).
* **Compatibilidade Portável:** Windows, Linux e macOS (para testes, governança, desenvolvimento e treino via PyTorch nativo).
* **Filosofia:** *AI proposes. Local policy decides. Runtime executes. Feedback improves.*

---

## 📌 Visão Geral

Treinamentos de modelos de linguagem em larga escala sofrem frequentemente com paradas catastróficas por **Out-of-Memory (OOM)**, explosão de gradientes (NaN/Inf), degradação silenciosa de hardware e corrupção de checkpoints.

O **MEM v3** atua como uma camada intermediária de governança autônoma: ele utiliza modelos de linguagem (ex: OpenAI GPT-4o) como planejadores de hiperparâmetros, mas **nunca permite que a IA execute comandos diretamente no hardware**. 

Todas as diretrizes passam pelo **LocalPolicyEngine**, que valida e aplica *clamping* determinístico antes de alcançar o runtime de execução (DeepSpeed ZeRO ou PyTorch Nativo).

```
                 LLM Planner / API Directives (GPT-4o)
                               │
                               ▼
                   ┌───────────────────────┐
                   │   LocalPolicyEngine   │ ◄── EnvironmentDoctor
                   └───────────┬───────────┘     (GPU VRAM, Temp, CUDA State)
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
            [SAFE DIRECTIVE]           [UNSAFE / OOM RISK]
                 │                           │
                 │                     Clamp to Policy Bounds
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
             ┌───────────────────────────────────┐
             │         Execution Runner          │
             │  ┌─────────────────────────────┐  │
             │  │ DeepSpeed ZeRO (Linux/GPU)  │  │
             │  ├─────────────────────────────┤  │
             │  │ PyTorch Native (Cross-plat) │  │
             │  └─────────────────────────────┘  │
             └─────────────────┬─────────────────┘
                               ▼
                 Atomic Rotating Checkpoints &
                  Comprehensive JSON Telemetry
```

---

## 📊 Evidências de Endurance & Resultados

Validado em hardware NVIDIA RTX (WSL2 / PyTorch `cu128`):

| Métrica | Resultado Validado |
| :--- | :--- |
| **Passos Globais Contínuos** | **1.000.000 / 1.000.000 steps** |
| **Throughput Médio** | **~33.000 tokens/segundo** |
| **Throughput de Pico** | **~45.600 tokens/segundo** |
| **OOM Fatais Registrados** | **0 (Zero)** |
| **Recuperação de Checkpoint** | **100% Contínua com integridade SHA256** |

---

## 🧩 Principais Componentes

1. **`LocalPolicyEngine` (Governança Determinística):** Intercepta e limita multiplicadores de taxa de aprendizado, normas de corte de gradiente e batch sizes dentro de envelopes matematicamente seguros.
2. **`CheckpointManager` (Atômico & Durável):** Gravação em diretórios temporários, validação em memória do tensor antes de publicação, verificação de integridade via SHA256 e rotação de slots (`live_00`, `live_01`, `live_02`).
3. **`Runtime Controller` Modular (`runtime/controller/`):**
   * `LaneManager`: Gerencia perfis de treino (`fast_seq256_zero0_gacc4`, `aggressive_seq256...`, `safe_seq256`) e regras de promoção.
   * `DegradationDetector`: Monitora métricas de saúde, pressão do otimizador e aciona troca de pista em tempo real.
   * `Supervisor`: Gerencia ciclo de vida dos processos e auto-recuperação pós-falha.
4. **Multi-Engine Execution:**
   * `DeepSpeedRunner`: Otimização ZeRO-0/ZeRO-1 com recuperação cross-ZeRO para servidores e clusters com GPU.
   * `PyTorchNativeRunner`: Execução pura em PyTorch para desenvolvimento, testes e máquinas sem DeepSpeed.

---

## 🛠️ Guia Rápido de Execução

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/nobazzy/mem-llm-orchestrator.git
cd mem-llm-orchestrator/mem_v3

# Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .\.venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
pip install -e .
```

### 2. Executar a Suíte de Testes Automatizados

```bash
# Executa a suíte de 33 testes funcionais e de integração
pytest -v

# Validação estática de contratos de arquitetura
python scripts/v89_static_validation.py
```

### 3. Diagnóstico do Hardware e Ambiente

```bash
# Inspeciona SO, Python, PyTorch, CUDA, GPUs e DeepSpeed
python main.py --environment-doctor
```

### 4. Executando o Orquestrador com Governança de IA

```bash
# Execução com integração OpenAI GPT-4o + Políticas Locais
export OPENAI_API_KEY="sua-chave-aqui"  # No Windows: $env:OPENAI_API_KEY="sua-chave"

python main.py --deepspeed-wsl-accelerated \
               --llm \
               --api-executive-moderate \
               --operator \
               --deepspeed-real-micro-train \
               --real-dataset \
               --confirm-deepspeed-accelerated I_UNDERSTAND_V89_RECOVERY_CONTROL \
               --deepspeed-max-steps 100
```

---

## 📂 Documentação Detalhada

* 📄 [`mem_v3/README.md`](mem_v3/README.md) — Technical architecture and module specification (English).
* 📄 [`mem_v3/README_PT.md`](mem_v3/README_PT.md) — Documentação técnica completa e guia de configuração em português.
* 📄 [`mem_v3/EVALUATION.md`](mem_v3/EVALUATION.md) — Relatório detalhado dos benchmarks e testes de endurance de 1M de steps.
* 📄 [`mem_v3/RELEASE_NOTES.md`](mem_v3/RELEASE_NOTES.md) — Histórico de versões e melhorias do Policy Engine.

---

## 📜 Licença
Distribuído sob a licença MIT. Veja [`LICENSE`](mem_v3/LICENSE) para mais detalhes.
