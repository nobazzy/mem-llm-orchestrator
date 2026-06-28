# Custom datasets — MEM v3

MEM v3 adds a configuration layer so users can run the validated MEM controller with their own datasets without editing runtime code.

The supported dataset paths are:

```txt
huggingface
local_text
local_jsonl
```

---

## 1. Hugging Face dataset

Example config:

```yaml
run:
  name: hf_dataset_example
  target_global_steps: 1000
  start_lane: fast_seq256_zero0_gacc4

workload:
  dataset:
    type: huggingface
    name: HuggingFaceFW/fineweb-edu
    config: sample-10BT
    split: train
    text_field: text
    fallback_name: roneneldan/TinyStories
    mix: ""
  tokenizer:
    name: gpt2
  model:
    preset: tiny_decoder
  runtime:
    chaos_profile: clean
```

Run:

```bash
bash scripts/run_mem_v3.sh --config configs/examples/huggingface_dataset.yaml
```

---

## 2. Local TXT dataset

A local text dataset can be a plain `.txt` file.

Example file:

```txt
data/train.txt
```

Example config:

```yaml
run:
  name: local_txt_run
  target_global_steps: 1000

workload:
  dataset:
    type: local_text
    path: data/train.txt
    split: train
    text_field: text
  tokenizer:
    name: gpt2
  model:
    preset: tiny_decoder
```

Run:

```bash
bash scripts/run_mem_v3.sh --config configs/examples/local_txt_dataset.yaml
```

---

## 3. Local JSONL dataset

Recommended format for custom datasets:

```json
{"text": "First training sample."}
{"text": "Second training sample."}
{"text": "Third training sample."}
```

Example config:

```yaml
run:
  name: local_jsonl_run
  target_global_steps: 1000

workload:
  dataset:
    type: local_jsonl
    path: data/train.jsonl
    split: train
    text_field: text
  tokenizer:
    name: gpt2
  model:
    preset: tiny_decoder
```

Run:

```bash
bash scripts/run_mem_v3.sh --config configs/examples/local_jsonl_dataset.yaml
```

---

## Target steps

For quick testing, use a small target:

```yaml
run:
  target_global_steps: 100
```

For longer validation:

```yaml
run:
  target_global_steps: 10000
```

For a full long run:

```yaml
run:
  target_global_steps: 300000
```

---

## What should not be edited

To preserve the validated core, users should not edit these files for normal dataset changes:

```txt
core/
runtime/deepspeed_runner.py
scripts/v89_sustained_controller.py
scripts/run_v89_sustained_control.sh
```

Use YAML configs instead.

---

## Important limitations

- The validated lane policy currently uses seq_len 256 lanes.
- Keep `tokenizer.name: gpt2` unless you intentionally validate another tokenizer.
- Keep `model.preset: tiny_decoder` unless you intentionally validate another model preset.
- Local dataset files must be accessible from WSL paths.
- For JSONL, each line must be valid JSON and contain the configured `text_field`.
