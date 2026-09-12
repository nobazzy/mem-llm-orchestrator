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

```txt
Destaques da Validação Empírica (RTX 5060 Ti, 8GB GDDR6):
- Confiabilidade de Longo Prazo: 1.000.000 de passos contínuos no modelo 130M (3.49B tokens, 98% de GPU sustentada)
- Resiliência a Caos Ativo: 100.000 passos no modelo 255M sob 71 choques dinâmicos com 100% de recuperação
- Convergência sob Choque Real (FineWeb-Edu sample-10BT): 50.000 passos com 150 choques físicos ao vivo de +1.2GB de VRAM (loss 11.0 -> 0.004)
- Falhas de Processo ou OOM: 0
- Recuperação Atômica de Checkpoint: 100% Contínua com verificação SHA256
- Overhead do Control Plane: <0.5% do tempo por step (medido em janelas de avaliação de 5 passos)
```

### 📊 Evidência Empírica: Absorção de Choques de VRAM em Hardware de 8GB

![MEM Orchestrator VRAM Benchmark](assets/mem_orchestrator_vram_benchmark.png)

---

## ⚡ Início Rápido: Rodando em 60 Segundos

Você pode iniciar uma sessão de treino adaptativo imediatamente sem chaves externas de API:

```bash
# 1. Clone o repositório
git clone https://github.com/nobazzy/mem-llm-orchestrator.git
cd mem-llm-orchestrator/mem_v3

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Inicie o treino adaptativo ao vivo (100% local com PyTorch)
python scripts/run_live_training.py --steps 1000 --batch-size 6 --dataset tinystories
```

### 🔌 Uso Programático no seu Loop de Treino

```python
from runtime.controller.lane_manager import LaneManager
from runtime.controller.degradation_detector import DegradationDetector

# Inicializa o governador de runtime
lane_mgr = LaneManager(target_vram_gb=7.5)
detector = DegradationDetector(patience=3)

# Dentro do seu loop de treino PyTorch:
for step, batch in enumerate(dataloader):
    # Avalia a margem de VRAM e reduz o micro-batch dinamicamente sob pressão
    current_lane = lane_mgr.evaluate_headroom(step=step)
    batch_size = current_lane.batch_size
    
    loss = model(batch[:batch_size])
    loss.backward()
    optimizer.step()
```

---

### 🛡️ A Filosofia
> **"AI proposes. Local policy decides. Runtime executes. Feedback improves."**
* A IA (OpenAI GPT-4o) atua opcionalmente como consultora de hiperparâmetros.
* O `LocalPolicyEngine` age como firewall matemático determinístico.
* O hardware só executa configurações validadas e protegidas por tetos rígidos.

---

## 🌿 Estrutura de Branches do Repositório

O projeto possui duas branches com objetivos e ambientes específicos:

| Branch | Ambiente Alvo | Tecnologia Base | Aplicação Ideal |
| :--- | :--- | :--- | :--- |
| **`main`** | **Linux / WSL2 Produção** | DeepSpeed ZeRO-0/1/2/3 + PyTorch | Servidores, clusters distribuídos e treinamento multi-GPU de alta performance em Linux. |
| **`refactor/architecture-and-portability`** | **Multiplataforma / Hardware de Consumo** | PyTorch Nativo + `LocalPolicyEngine` | Workstations e GPUs de consumo (Windows / WSL / Linux). Desacopla dependências de compilação C++ do DeepSpeed mantendo o motor autônomo de troca de faixas. |

---

## ⚙️ Configuração de Ambiente (.env) e Modo 100% Local

**O MEM v3 opera 100% local e offline por padrão.** Nenhuma chave de API externa é necessária para treinar, gerenciar VRAM ou alternar faixas.

Caso deseje ativar o consultor executivo externo de IA (GPT-4o) para sugestões de taxa de aprendizado e clipping:

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Configure sua chave dentro do .env (nunca é enviado ao GitHub)
OPENAI_API_KEY=sua_chave_aqui
```

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

Execute a suíte de testes com 38 validações automatizadas:

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
