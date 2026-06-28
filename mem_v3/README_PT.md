## Início rápido — WSL2, Conda, CUDA e runtime MEM

Este é o fluxo local recomendado para o pacote atual do MEM v3 em WSL2 com GPU NVIDIA.

O ambiente local validado usa:

```txt
Sistema: WSL2 Ubuntu
Gerenciador de ambiente: Miniforge / Conda
Ambiente Conda: mem312
Python: 3.12.13
GPU: NVIDIA RTX série 50 / classe Blackwell
CUDA alvo: build PyTorch compatível com cu128
DeepSpeed: habilitado
Tipo de workload: treinamento de modelo de linguagem
```

O MEM v3 é um workload de **treinamento de modelo de linguagem**. Ele não precisa de `torchvision`, `torchaudio`, datasets de imagem, datasets de áudio ou operadores de visão/áudio.

Não instale:

```txt
torchvision
torchaudio
torchtext
```

Instale apenas `torch` e as dependências necessárias para treinamento de linguagem, execução do controller, datasets, tokenização, telemetria e DeepSpeed.

---

## 1. Extrair o pacote

Extraia o ZIP de release e entre na pasta do projeto:

```bash
unzip mem_v3.zip

cd mem_v3
```

Se o pacote já estiver extraído:

```bash
cd mem_v3
```

---

## 2. Criar ou ativar o ambiente Conda

Use o nome de ambiente validado:

```bash
source ~/miniforge3/etc/profile.d/conda.sh

conda create -n mem312 python=3.12.13 -y

conda activate mem312

python --version
which python
```

Esperado:

```txt
Python 3.12.13
```

Se o ambiente já existir:

```bash
source ~/miniforge3/etc/profile.d/conda.sh

conda activate mem312

python --version
which python
```

---

## 3. Instalar dependências Python base

Atualize primeiro as ferramentas de instalação:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Instale as dependências gerais do runtime MEM:

```bash
python -m pip install --upgrade \
  numpy \
  pyyaml \
  psutil \
  tqdm \
  requests \
  openai \
  datasets \
  transformers \
  tokenizers \
  accelerate \
  safetensors \
  pytest
```

---

## 4. Instalar PyTorch com CUDA

Para GPUs RTX série 50 / Blackwell, use uma build do PyTorch compatível com CUDA 12.8.

Instale **somente o torch**:

```bash
python -m pip uninstall -y torch torchvision torchaudio torchtext

python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Não use o comando genérico do PyTorch que instala `torchvision` e `torchaudio`, porque o MEM v3 não precisa de pacotes de visão ou áudio.

Correto para o MEM:

```bash
python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Não necessário para o MEM:

```bash
python -m pip install torch torchvision torchaudio
```

Valide PyTorch e CUDA:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda disponível:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("cuda matmul: OK", tuple(y.shape))
else:
    raise SystemExit("ERRO: CUDA não está disponível para o PyTorch")
PY
```

Resultado esperado em alto nível:

```txt
cuda disponível: True
cuda matmul: OK
```

---

## 5. Instalar DeepSpeed

Instale o DeepSpeed depois que o PyTorch estiver funcionando:

```bash
DS_BUILD_OPS=0 python -m pip install --upgrade deepspeed
```

Valide o import:

```bash
python - <<'PY'
import deepspeed
print("deepspeed:", deepspeed.__version__)
PY
```

O MEM v3 não exige a compilação de operadores CUDA customizados do DeepSpeed no fluxo local validado.

---

## 6. Validar imports obrigatórios

```bash
python - <<'PY'
mods = [
    "torch",
    "deepspeed",
    "datasets",
    "transformers",
    "tokenizers",
    "openai",
    "yaml",
    "psutil",
    "numpy",
    "tqdm",
]

for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(f"{name}: FALHOU -> {type(exc).__name__}: {exc}")
PY
```

---

## 7. Validar o pacote MEM

A partir da raiz do projeto:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

chmod +x scripts/*.sh scripts/*.py

python scripts/v89_static_validation.py

bash scripts/doctor_v89.sh

bash scripts/validate_mem_v3_configs.sh
```

Resultado esperado em alto nível:

```txt
V89_STATIC_VALIDATION: PASS
torch: OK
cuda available: True
deepspeed: OK
```

Se `doctor_v89.sh` não existir em algum pacote específico, use o doctor disponível:

```bash
bash scripts/doctor_mem_v3.sh
```

---

## 8. Configurar a chave da OpenAI

O MEM v3 usa o caminho assistido por API no controller.

Variável obrigatória:

```txt
OPENAI_API_KEY
```

Configure a chave no mesmo terminal que vai executar o MEM:

```bash
unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY carregada"
```

Não imprima a chave completa.

Validação segura:

```bash
python - <<'PY'
import os

key = os.getenv("OPENAI_API_KEY", "")

print("OPENAI_API_KEY carregada:", bool(key))
print("prefixo:", key[:7] + "..." if key else "EMPTY")
PY
```

Nunca envie API keys, tokens ou segredos locais para o repositório.

---

## 9. Rodar um smoke test local JSONL

Este é o primeiro teste recomendado para um ambiente novo ou uma nova extração do pacote.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

bash scripts/run_mem_v3.sh --config configs/validation/local_jsonl_100.yaml
```

Esse teste valida:

```txt
tradução da config
caminho do dataset JSONL local
compatibilidade do launcher
inicialização do controller
ligação com o runtime
```

---

## 10. Rodar validação Hugging Face de 1000 steps

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

bash scripts/run_mem_v3.sh --config configs/validation/huggingface_1000.yaml
```

A primeira execução com Hugging Face pode demorar para baixar, preparar e cachear o dataset antes dos steps de treino aparecerem.

---

## 11. Rodar o preset prático 50M SlimPajama 1M steps

Para GPUs restritas de 8 GB, este é o preset prático de run longa.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_50m_slimpajama_1m.sh
```

---

## 12. Rodar o preset Quality Curriculum

O Quality Curriculum usa um stream misto do Hugging Face para combinar corpus mais amplo com texto narrativo mais fácil.

Mix esperado:

```txt
DKYoon/SlimPajama-6B: 40%
roneneldan/TinyStories: 60%
```

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY carregada"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_50m_quality_curriculum_1m.sh
```

---

## 13. Rodar o preset 100M SlimPajama stress test

O preset 100M é um teste de estresse para GPUs restritas de 8 GB.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

unset OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export MEM_ALLOW_TORCH_COMPILE=0
export MEM_AUTO_RESUME_CHECKPOINT=1

bash scripts/run_v89_100m_slimpajama_1m.sh
```

Interpretação recomendada:

```txt
Preset 50M:
  run prática para 8 GB
  melhor equilíbrio entre estabilidade e throughput

Preset 100M:
  teste de estresse
  útil como evidência de robustez
  esperado maior consumo de VRAM
```

---

## 14. Monitorar runtime e checkpoints

Monitor humano básico:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

MONITOR_INTERVAL=10 bash scripts/monitor_v89_human.sh
```

Monitor de runtime/checkpoint:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

watch -n 10 'bash scripts/monitor_v89_human_runtime.sh'
```

Use o monitor de runtime quando quiser ver:

```txt
status atual do controller
progresso global
lane ativa
pressão de GPU/RAM
último checkpoint real
tamanho do checkpoint
idade do checkpoint
ponteiro latest
arquivos recentes de checkpoint
```

Monitor apenas de checkpoint:

```bash
watch -n 10 'find checkpoints -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" 2>/dev/null | sort | tail -20'
```

---

## 15. Parar uma run MEM/DeepSpeed

Use isto quando quiser parar a run atual e retomar depois pelo checkpoint.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

echo "Parando controller/treino com TERM..."
pkill -TERM -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

sleep 8

echo "Matando processos MEM/DeepSpeed restantes..."
pkill -KILL -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true

echo
echo "Processos restantes:"
ps aux | grep -E "v89_sustained_controller|main.py.*--deepspeed|deepspeed|torchrun" | grep -v grep || echo "OK: nenhum processo MEM/DeepSpeed rodando"

echo
echo "GPU:"
nvidia-smi
```

---

## 16. Validar retomada por checkpoint

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

echo "Ponteiro do último checkpoint:"
cat checkpoints/v89_latest.txt 2>/dev/null || true

echo
echo "Arquivo do último checkpoint:"
ls -lh "$(cat checkpoints/v89_latest.txt)" 2>/dev/null || true

echo
echo "Estado do controller:"
cat checkpoints/v89_controller_state.json 2>/dev/null || true

echo
echo "Validador de resume:"
bash scripts/verify_checkpoint_resume_v89.sh
```

O pacote final de logs pode conter metadados de checkpoint, mas não necessariamente o arquivo pesado de tensores:

```txt
mem_model_optimizer.pt
```

Para retomar o treinamento, o checkpoint real precisa existir localmente dentro de:

```txt
checkpoints/
```

---

## 17. Salvar snapshot ZIP

Snapshots devem ser salvos a partir da pasta pai do projeto, não dentro de `mem_v3`.

```bash
cd ..

TS="$(date +%Y%m%d_%H%M%S)"

zip -r "mem_v3_snapshot_${TS}.zip" \
  mem_v3/evidence \
  mem_v3/evidence_packets \
  mem_v3/reports \
  mem_v3/logs \
  mem_v3/checkpoints \
  mem_v3/README.md \
  mem_v3/EVALUATION.md \
  mem_v3/VERSION_CURRENT.txt \
  mem_v3/PATCH_APPLIED_NOTES.md \
  mem_v3/RECOVERY_NOTES_20260624.md

echo "Snapshot salvo:"
ls -lh "mem_v3_snapshot_${TS}.zip"
```

Para pacotes públicos de release, não inclua:

```txt
API keys
segredos locais
caches de dataset
ambientes virtuais
arquivos grandes de checkpoint
arquivos temporários específicos da máquina
```

Para arquivos privados de recuperação, checkpoints podem ser incluídos intencionalmente.

---

## 18. Evidência validada atual: run sustentada de 1M steps

Uma run sustentada v89 completou o alvo total:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
controller core: v89.0.0
package line: mem_v3.3 safe-recovery-50m-preset
```

Perfil da execução:

```txt
ambiente: WSL2 Ubuntu
python: 3.12.13
torch cuda target: build compatível com cu128
gpu: NVIDIA RTX série 50 / classe 8 GB
precisão: fp16
DeepSpeed ZeRO stage: 0
sequence length: 256
tokenizer: gpt2
model preset: medium_50m
gradient accumulation: 4
```

Metadados de checkpoint:

```txt
latest pointer: checkpoints/v89_live_02/mem_model_optimizer.pt
metadata present: yes
heavy checkpoint file included in final logs ZIP: no
```

O pacote final de logs registra estado do controller, reports, evidências e metadados de checkpoint. Ele não substitui o arquivo local real de checkpoint.

---

## 19. Solução de problemas

### CUDA não está disponível

Confira se a GPU aparece no WSL:

```bash
nvidia-smi
```

Confira CUDA no PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Se CUDA retornar `False`, reinstale apenas o `torch` por um índice compatível com CUDA. Não adicione `torchvision` ou `torchaudio`.

### Ambiente Conda errado

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mem312

which python
python --version
```

### Script de monitor ausente

Liste os monitores disponíveis:

```bash
find scripts -maxdepth 1 -type f -iname "*monitor*" -print | sort
```

### Checkpoint ausente

```bash
find checkpoints -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" 2>/dev/null | sort | tail -20

cat checkpoints/v89_latest.txt 2>/dev/null || true
```

### Matar run travada

```bash
pkill -TERM -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true
sleep 8
pkill -KILL -f "v89_sustained_controller.py|main.py.*--deepspeed|deepspeed|torchrun" 2>/dev/null || true
nvidia-smi
```

---

## 20. Política de dependências

Para este pacote MEM v3:

```txt
Instalar:
  torch
  deepspeed
  datasets
  transformers
  tokenizers
  accelerate
  safetensors
  openai
  pyyaml
  psutil
  numpy
  tqdm
  pytest

Não instalar:
  torchvision
  torchaudio
  torchtext
```

Motivo:

```txt
O MEM v3 treina modelos de linguagem.
Ele não usa operadores de imagem.
Ele não usa operadores de áudio.
Ele não precisa de datasets/modelos/transforms do torchvision.
Ele não precisa de backends do torchaudio.
Manter o ambiente menor reduz risco de incompatibilidade binária.
```

---

## Status de release

```txt
Produto: MEM
Linha pública: mem_v3
Core interno validado: v89.0.0
Release base: pacote mem_v2 validado em 300k
Versão: mem_v3.3 safe-recovery-50m-preset
Novas capacidades: workloads configuráveis, datasets customizados, checkpoints vivos rotativos, continuidade de resume e endurecimento da recovery-lane
Status: caminho 300k validado; run local sustentada de 1M concluída
Evidência de endurance mais recente: 1000000 / 1000000 global steps, state DONE
```
