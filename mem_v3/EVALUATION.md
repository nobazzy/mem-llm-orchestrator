# Evaluation — MEM v3

**Public package:** `mem_v3`  
**Internal validated core:** `v89.0.0`  
**Base package:** `mem_v2` 300k validated release  
**Current release line:** `mem_v3.3 safe-recovery-50m-preset`  
**Evaluation status:** 300k validated path available; local 1M sustained run completed  

---

## 1. Evaluation scope

MEM v3 should be evaluated in five layers:

```txt
1. Base sustained runtime behavior inherited from mem_v2 / v89.0.0
2. Configurable workload behavior added in mem_v3
3. Checkpoint resume continuity between lane sessions
4. Live rotating checkpoints during long runs
5. Recovery-lane hardening added in the mem_v3.3 release line
```

The core evaluation question is:

```txt
Can the system keep a real long-running language-model workload productive, safe,
observable and recoverable until the target is completed, while allowing datasets
and workloads to be configured without editing runtime/controller code?
```

For the current release line, there is also a controller-specific question:

```txt
Can the controller prevent safe_seq256 from becoming a throughput prison and treat
it as a recovery-only lane instead of a long-term productive lane?
```

---

## 2. Validation boundary

MEM v3 inherits the validated runtime core from mem_v2 and adds new configuration,
checkpoint and controller-policy layers.

This means:

```txt
Validated core evidence: inherited from mem_v2 / v89.0.0
Configurable workload layer: validate per dataset/configuration
Checkpoint resume path: validate when using lane switching or interrupted runs
Live checkpoint path: validate before relying on recovery archives
1M endurance evidence: completed locally under the documented target environment
```

Accurate claim:

```txt
MEM v3 has completed a sustained local 1M-step run on the documented target environment.
```

Do not claim:

```txt
MEM v3 is universally validated for every dataset, GPU, CUDA build and model size.
```

---

## 3. Base validated result

MEM v3 is based on mem_v2, which completed a real sustained run:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
internal_core: v89.0.0
```

Observed mem_v2 base run summary:

```txt
Target completed: 300000 / 300000 global steps
Approximate duration: ~6h47min
Observed mean throughput: ~33000 tokens/s
Observed median throughput: ~32000 tokens/s
Observed peak throughput: ~45600 tokens/s
Fatal OOM blocking target: not observed
Fatal traceback blocking target: not observed
```

Current MEM v3 300k validation evidence:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
restart_index: 14
absolute safe-lane escape: observed and validated
fatal OOM blocking target: not observed
fatal traceback blocking target: not observed
```

Current MEM v3 1M endurance evidence:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
internal_core: v89.0.0
release_line: mem_v3.3
```

These numbers describe local validation runs. They are not universal performance guarantees.

---

## 4. What should be evaluated in mem_v3

The configurable workload layer should prove that users can change dataset/workload
behavior through YAML configuration instead of editing runtime code.

Evaluation targets:

```txt
Hugging Face dataset config works
local TXT dataset config works
local JSONL dataset config works
custom text_field is respected
short smoke target can complete
long target can complete
controller still generates evidence
monitor still works
checkpoint pointer is updated
resume validation can read latest checkpoint metadata
API Light path works when enabled
no internal runtime file needs manual editing for normal workloads
```

Supported dataset types:

```txt
huggingface
local_txt
local_jsonl
```

---

## 5. Checkpoint and resume evaluation

The checkpoint path should be evaluated separately from final run status.

Minimum checkpoint checks:

```txt
checkpoints/v89_latest.txt exists when live checkpoints are enabled
latest pointer resolves to an existing checkpoint file
checkpoint metadata is present
controller state records completed/global steps
resume validator passes
lane switch can receive load_checkpoint without runtime rejection
```

Useful command:

```bash
bash scripts/verify_checkpoint_resume_v89.sh
```

Important boundary:

```txt
A final logs/evidence ZIP may contain checkpoint metadata.
It may not contain the heavy tensor file mem_model_optimizer.pt.
A recovery archive must include the real checkpoint tensor file.
```

---

## 6. Recovery-lane evaluation

The current release line treats `safe_seq256` as a recovery-only lane.

The expected behavior is:

```txt
safe_seq256 can be used to preserve execution under pressure
safe_seq256 should not remain active when throughput is clearly degraded
throughput-dead safe lane should arm a local escape
escape should target a productive lane when loss/checkpoint remain coherent
```

Known degraded pattern:

```txt
Lane: safe_seq256
Tokens/s: low relative to productive lanes
Steps/s: low
Optimizer ratio: high
GPU work efficiency: poor
Fatal OOM: not observed
Fatal traceback: not observed
Result without hardening: safe lane can remain alive but unproductive
```

Expected controller behavior:

```txt
safe_seq256_absolute_escape_armed
safe_seq256_hard_escape_override
safe_seq256_absolute_25k_floor_escape
lane_switch_applied -> fast_seq256_zero0_gacc4 or aggressive_seq256_zero0_gacc4
```

The exact exit target may differ by release/policy state, but the evaluation requirement is the same:

```txt
safe_seq256 must not become a permanent throughput prison.
```

---

## 7. Recommended validation sequence

Use short validations first. Only run long validations after short validations pass.

```txt
1. Local JSONL smoke test
2. Local TXT smoke test
3. Hugging Face 1000-step validation
4. Custom dataset short run
5. Long Hugging Face 300k run
6. Practical 50M 1M-step run
7. Quality Curriculum 1M-step run
8. 100M stress test, if the target GPU can tolerate pressure/churn
```

Recommended interpretation:

```txt
50M preset:
  practical constrained-GPU preset
  recommended for 8GB-class evaluation

100M preset:
  stress test on 8GB-class GPUs
  useful for robustness evidence
  expected higher VRAM pressure and controlled churn

Quality Curriculum preset:
  quality-oriented 50M comparison path
  mixed broad-corpus + simpler narrative data
```

---

## 8. Environment validation

The environment should validate:

```txt
Python version
Conda environment
PyTorch import
CUDA availability
GPU name/capability
DeepSpeed import
required Python imports
MEM static validation
config validation
```

Expected high-level output:

```txt
Python: 3.12.13
torch: OK
cuda available: True
deepSpeed/deepspeed: OK
V89_STATIC_VALIDATION: PASS
```

MEM v3 is a language-model workload. It should not require:

```txt
torchvision
torchaudio
torchtext
```

---

## 9. Evidence checklist

A complete evaluation bundle should include:

```txt
controller status
controller global progress
controller state
latest checkpoint pointer
checkpoint metadata
runtime logs
API usage summary, if API Light is enabled
adaptive benchmark summary, if available
monitor output or final summary
```

For public releases, do not include:

```txt
API keys
local secrets
virtual environments
dataset caches
large checkpoint tensor files
machine-specific absolute paths
```

For private recovery archives, checkpoints may be included intentionally.

---

## 10. Limitations

```txt
Throughput numbers are local observations, not universal guarantees.
Different datasets can change data wait, memory pressure, loss behavior and lane stability.
Different GPUs can change the practical model-size boundary.
RTX 50-series / Blackwell requires compatible PyTorch CUDA builds.
Hugging Face datasets may take time to download/cache before the first training step appears.
External issues such as WSL, CUDA drivers, API keys, internet access or dataset access can still affect long runs.
100M on 8GB GPUs should be interpreted as a stress test, not the default practical configuration.
```

---

# Avaliação — MEM v3

**Pacote público:** `mem_v3`  
**Core interno validado:** `v89.0.0`  
**Pacote base:** release `mem_v2` validado em 300k  
**Linha atual:** `mem_v3.3 safe-recovery-50m-preset`  
**Status de avaliação:** caminho 300k validado; run local sustentada de 1M concluída  

---

## 1. Escopo da avaliação

O MEM v3 deve ser avaliado em cinco camadas:

```txt
1. Comportamento base de runtime herdado do mem_v2 / v89.0.0
2. Camada de workload configurável adicionada no mem_v3
3. Continuidade de resume por checkpoint entre sessões de lane
4. Checkpoints vivos rotativos durante runs longas
5. Endurecimento da recovery-lane na linha mem_v3.3
```

A pergunta central é:

```txt
O sistema consegue manter um workload real de linguagem produtivo, seguro,
observável e recuperável até completar o alvo, permitindo configurar dataset e
workload sem editar código interno de runtime/controller?
```

Para a linha atual, também existe uma pergunta específica do controller:

```txt
O controller consegue impedir que a safe_seq256 vire uma prisão de throughput e
tratá-la como lane de recuperação, não como lane produtiva permanente?
```

---

## 2. Limite de validação

O MEM v3 herda o core validado do mem_v2 e adiciona novas camadas de configuração,
checkpoint e política de controller.

Isso significa:

```txt
Evidência de core validado: herdada do mem_v2 / v89.0.0
Camada de workload configurável: validar por dataset/configuração
Resume por checkpoint: validar em troca de lanes ou runs interrompidas
Checkpoint vivo: validar antes de depender de arquivos de recuperação
Evidência 1M: concluída localmente no ambiente alvo documentado
```

Afirmação correta:

```txt
O MEM v3 completou uma run local sustentada de 1M steps no ambiente alvo documentado.
```

Não afirmar:

```txt
O MEM v3 está universalmente validado para qualquer dataset, GPU, build CUDA e tamanho de modelo.
```

---

## 3. Evidência validada

O MEM v3 é baseado no mem_v2, que completou uma run real sustentada:

```txt
state: DONE
global_step: 300000 / 300000
reason: target_global_steps_reached
internal_core: v89.0.0
```

Evidência atual de endurance 1M no MEM v3:

```txt
state: DONE
reason: target_global_steps_reached
global_step: 1000000
target_global_steps: 1000000
internal_core: v89.0.0
release_line: mem_v3.3
```

Esses números descrevem runs locais validadas. Não são garantias universais de performance.

---

## 4. O que validar no mem_v3

A camada de workload configurável deve provar que o usuário consegue alterar dataset
ou workload por YAML, sem editar o runtime.

Alvos de avaliação:

```txt
config de dataset Hugging Face funciona
config de TXT local funciona
config de JSONL local funciona
text_field customizado é respeitado
smoke test curto completa
alvo longo completa
controller gera evidências
monitor funciona
ponteiro de checkpoint é atualizado
validador de resume lê metadata do último checkpoint
API Light funciona quando habilitada
nenhum arquivo interno de runtime precisa ser editado para workloads normais
```

---

## 5. Avaliação de checkpoint e resume

O caminho de checkpoint deve ser avaliado separadamente do status final da run.

Checagens mínimas:

```txt
checkpoints/v89_latest.txt existe quando checkpoints vivos estão habilitados
ponteiro latest resolve para um arquivo de checkpoint existente
metadata de checkpoint existe
estado do controller registra steps completos/globais
validador de resume passa
lane switch aceita load_checkpoint sem rejeição do runtime
```

Comando útil:

```bash
bash scripts/verify_checkpoint_resume_v89.sh
```

Limite importante:

```txt
Um ZIP final de logs/evidências pode conter metadata de checkpoint.
Ele pode não conter o arquivo pesado mem_model_optimizer.pt.
Um arquivo de recuperação precisa incluir o tensor real de checkpoint.
```

---

## 6. Avaliação da recovery-lane

A linha atual trata `safe_seq256` como lane de recuperação.

Comportamento esperado:

```txt
safe_seq256 pode preservar execução sob pressão
safe_seq256 não deve permanecer ativa quando o throughput estiver claramente degradado
safe lane improdutiva deve armar escape local
escape deve voltar para lane produtiva quando loss/checkpoint estiverem coerentes
```

Requisito principal:

```txt
safe_seq256 não pode virar uma prisão permanente de throughput.
```

---

## 7. Sequência recomendada

```txt
1. Smoke test JSONL local
2. Smoke test TXT local
3. Validação Hugging Face de 1000 steps
4. Run curta com dataset customizado
5. Run longa Hugging Face 300k
6. Run prática 50M de 1M steps
7. Run Quality Curriculum de 1M steps
8. Stress test 100M, se a GPU alvo tolerar pressão/churn
```

---

## 8. Limitações

```txt
Números de throughput são observações locais, não garantias universais.
Datasets diferentes podem alterar data wait, pressão de memória, loss e estabilidade de lane.
GPUs diferentes podem alterar o limite prático de tamanho de modelo.
RTX série 50 / Blackwell exige builds PyTorch CUDA compatíveis.
Datasets Hugging Face podem demorar para baixar/cachear antes dos steps aparecerem.
Questões externas como WSL, driver CUDA, API key, internet ou acesso a dataset ainda podem afetar runs longas.
100M em GPUs de 8GB deve ser interpretado como stress test, não como configuração prática padrão.
```
