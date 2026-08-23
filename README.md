# ⚡ MEM v3 — Model Execution Manager & LLM Training Orchestrator

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20cu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Zero--OOM-00599C)](https://www.deepspeed.ai/)
[![Validation](https://img.shields.io/badge/Validation-1M%20Steps%20Zero%20OOM-brightgreen)](mem_v3/EVALUATION.md)
[![Status](https://img.shields.io/badge/Status-Functional%20%2F%20Active-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](mem_v3/LICENSE)

> **Autonomous, zero-OOM orchestration engine for sustained Large Language Model pre-training and fine-tuning with deterministic policy enforcement.**

---

### 🟢 Status do Projeto
* **Ambiente Alvo:** Linux / WSL2 (Ubuntu)
* **Estado:** Funcional e em desenvolvimento ativo.
* **Filosofia:** *AI proposes. Local policy decides. Runtime executes. Feedback improves.*

---

## 📌 Visão Geral

Treinamentos de modelos de linguagem em larga escala sofrem frequentemente com paradas catastróficas por **Out-of-Memory (OOM)**, oscilações de gradiente e falhas de hardware.

O **MEM v3** atua como uma camada intermediária de governança autônoma: ele utiliza um modelo de linguagem como planejador de hiperparâmetros, mas **nunca permite que a IA execute comandos diretamente no hardware**. Todas as diretrizes passam pelo **LocalPolicyEngine**, que valida e aplica *clamping* determinístico antes de chegar ao runtime PyTorch/DeepSpeed.

```
                 LLM Planner / API Directives
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
                 DeepSpeed / PyTorch Runtime
                             │
                             ▼
                 Rotating Checkpoint & Telemetry
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
| **Recuperação de Checkpoint** | **100% Contínua sem perda de progresso** |

---

## 🛠️ Guia de Execução

### 1. Pré-requisitos
* Linux ou Windows com **WSL2 (Ubuntu 22.04+)**
* Python 3.10, 3.11 ou 3.12
* GPU NVIDIA com drivers CUDA atualizados

### 2. Instalação
```bash
# Clone o repositório
git clone https://github.com/nobazzy/mem-llm-orchestrator.git
cd mem-llm-orchestrator/mem_v3

# Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instale o pacote e dependências
pip install -r requirements.txt
pip install -e .
```

### 3. Execução dos Testes e Validação
```bash
# Rodar suíte de testes unitários e de integração
pytest

# Testar módulos de resiliência e injeção de caos
python3 -m pytest tests/test_chaos.py
```

### 4. Executando o Orquestrador
```bash
# Execução padrão via CLI
python3 main.py

# Para visualizar as opções de configuração e telemetria:
python3 application/cli.py --help
```

---

## 📂 Documentação Detalhada Existente

Para aprofundamento técnico e logs de validação, consulte os documentos nos subdiretórios:
* 📄 [`mem_v3/README_PT.md`](mem_v3/README_PT.md) — Documentação técnica completa em português.
* 📄 [`mem_v3/EVALUATION.md`](mem_v3/EVALUATION.md) — Relatório detalhado dos benchmarks e testes de 1M de steps.
* 📄 [`mem_v3/RELEASE_NOTES.md`](mem_v3/RELEASE_NOTES.md) — Histórico de versões e melhorias do Policy Engine.

---

## 💼 Inquiries & Technical Contact

**MEM v3** was architected for production-grade AI infrastructure.
For consulting, custom integrations, or architecture audits:
* 📩 Reach out via GitHub ([@nobazzy](https://github.com/nobazzy)) or LinkedIn.

---

## 📜 Licença
Distribuído sob a licença MIT. Veja [`LICENSE`](mem_v3/LICENSE) para mais detalhes.
