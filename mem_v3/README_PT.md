# ⚡ MEM v3 — Guia Técnico e Manual Operacional

> **Guia completo de arquitetura, configuração de ambiente (WSL2/Linux e Multiplataforma), execução do orquestrador e políticas de segurança.**

---

## 📌 Sumário
1. [Visão Geral e Filosofia](#1-visão-geral-e-filosofia)
2. [Arquitetura e Módulos](#2-arquitetura-e-módulos)
3. [Guia de Instalação e Ambientes](#3-guia-de-instalação-e-ambientes)
4. [Governança da IA e LocalPolicyEngine](#4-governança-da-ia-e-localpolicyengine)
5. [Gerenciamento Atômico de Checkpoints](#5-gerenciamento-atômico-de-checkpoints)
6. [Suíte de Testes e Validação](#6-suíte-de-testes-e-validação)
7. [Execução do Orquestrador e CLI](#7-execução-do-orquestrador-e-cli)

---

## 1. Visão Geral e Filosofia

O **MEM v3 (Model Execution Manager)** é uma infraestrutura de governança autônoma desenvolvida para sustentação contínua de treinamentos de Modelos de Linguagem (LLMs/SLMs).

### 🎯 O Problema que Resolve
Treinos longos de IA frequentemente sofrem paradas silenciosas e catastróficas:
* **Out-of-Memory (OOM):** Fragmentação de VRAM ou aumento súbito de tamanho de contexto.
* **Instabilidade Numérica:** Explosões de gradiente gerando perdas `NaN` ou `Inf`.
* **Corrupção de Checkpoint:** Processos interrompidos no exato momento da escrita em disco.

### 🛡️ A Filosofia
> **"AI proposes. Local policy decides. Runtime executes. Feedback improves."**
* A IA (OpenAI GPT-4o) atua como consultora e planejadora de hiperparâmetros.
* O `LocalPolicyEngine` age como firewall matemático determinístico.
* O hardware só executa configurações validadas e protegidas por tetos rígidos.

---

## 2. Arquitetura e Módulos

O projeto segue os princípios de **Clean Architecture**:

* **Camada de Aplicação (`application/`):** Ponto de entrada CLI (`cli.py`), parsing de argumentos e injeção dinâmica de caminhos.
* **Camada Core (`core/`):**
  * `MemOrchestrator`: Coordena o ciclo de vida da execução.
  * `LocalPolicyEngine`: Avalia e aplica *clamping* determinístico em parâmetros da IA.
  * `EnvironmentDoctor`: Diagnostica GPU, CUDA, drivers, Python e dependências.
* **Camada de Domínio (`domain/`):** Modelos e contratos tipados (`RuntimeRequest`, `CandidatePlan`, `ExecutiveDirective`, `PolicyDecision`).
* **Camada de Infraestrutura (`infrastructure/`):** Cliente de integração OpenAI com schemas JSON estritos e telemetria de tokens.
* **Camada de Runtime (`runtime/`):**
  * `DeepSpeedRunner`: Execução otimizada com DeepSpeed ZeRO-0/ZeRO-1 para GPUs em Linux/WSL2.
  * `PyTorchNativeRunner`: Execução pura com PyTorch para desenvolvimento, testes e ambientes sem DeepSpeed.
  * `CheckpointManager`: Protocolo de publicação atômica em dois passos com verificação SHA256.
  * `Controller` (`runtime/controller/`): Módulos desacoplados de gestão de lanes (`LaneManager`), detecção de degradação (`DegradationDetector`) e supervisão de processos (`Supervisor`).

---

## 3. Guia de Instalação e Ambientes

### Ambiente A: Linux / WSL2 (Recomendado para Alta Performance com DeepSpeed)
* **SO:** Ubuntu 22.04+ (WSL2 ou Linux nativo)
* **Python:** 3.10, 3.11 ou 3.12
* **GPU:** NVIDIA com suporte a CUDA 12.8+

```bash
# Ativação do ambiente e instalação
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
DS_BUILD_OPS=0 pip install deepspeed
pip install -e .
```

### Ambiente B: Multiplataforma (Windows / macOS / Linux - Desenvolvimento e Testes)
* **Suporte Nativo:** Todos os testes unitários, diagnósticos de ambiente, chamadas de IA e o runner nativo PyTorch rodam diretamente no Windows sem necessidade de WSL.

```powershell
# No Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 4. Governança da IA e LocalPolicyEngine

O orquestrador nunca entrega o controle direto do hardware para modelos de linguagem. O `LocalPolicyEngine` impõe invariantes rígidos:

| Parâmetro | Limite Determinístico | Motivo de Segurança |
| :--- | :--- | :--- |
| **Multiplicador de LR** | Clamped entre `0.85` e `1.0` | Previne quedas bruscas ou taxas excessivas de aprendizado. |
| **Gradient Clip Norm** | Clamped entre `0.25` e `1.25` | Evita instabilidade numérica e gradientes explosivos. |
| **Loss Scale Power** | Clamped entre `6` e `10` | Protege a escala dinâmica de precisão mista FP16. |
| **Teto de Passos** | Máximo absoluto de `10.000.000` | Impede execuções infinitas descontroladas. |
| **Token de Confirmação** | `I_UNDERSTAND_V89_RECOVERY_CONTROL` | Garante validação explícita do operador. |

---

## 5. Gerenciamento Atômico de Checkpoints

O `CheckpointManager` utiliza um protocolo à prova de falhas:
1. **Gravação em Staging:** Salva o checkpoint em pasta temporária única (`.tmp.<pid>.<uuid>`).
2. **Auto-Validação:** Lê o tensor de volta para a memória RAM garantindo integridade física.
3. **Cálculo de Hash:** Gera o hash criptográfico SHA256 (`.sha256`).
4. **Promoção Atômica:** Move o diretório para o slot `live_xx` via `os.replace`. Em caso de erro, o slot anterior é preservado intacto.

---

## 6. Suíte de Testes e Validação

Execute a suíte de testes com 33 validações automatizadas:

```bash
# Rodar todos os testes unitários e funcionais
pytest -v

# Rodar validação estática de contratos e integridade de lanes
python scripts/v89_static_validation.py
```

---

## 7. Execução do Orquestrador e CLI

### Diagnóstico de Ambiente
```bash
python main.py --environment-doctor
```

### Treinamento com Governança de IA (OpenAI GPT-4o)
```bash
export OPENAI_API_KEY="sua-chave-aqui"

python main.py --deepspeed-wsl-accelerated \
               --llm \
               --api-executive-moderate \
               --operator \
               --deepspeed-real-micro-train \
               --real-dataset \
               --confirm-deepspeed-accelerated I_UNDERSTAND_V89_RECOVERY_CONTROL \
               --deepspeed-max-steps 1000
```
