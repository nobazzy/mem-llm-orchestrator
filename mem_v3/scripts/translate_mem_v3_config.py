#!/usr/bin/env python3
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml


def q(value: object) -> str:
    return shlex.quote(str(value))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: translate_mem_v3_config.py <config.yaml>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1]).resolve()
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    run = cfg.get("run") or {}
    workload = cfg.get("workload") or {}
    dataset = workload.get("dataset") or {}
    tokenizer = workload.get("tokenizer") or {}
    model = workload.get("model") or {}
    runtime = workload.get("runtime") or {}
    training = cfg.get("training") or workload.get("training") or {}

    ds_type = str(dataset.get("type", "huggingface")).strip().lower()
    data_files = ""
    dataset_name = str(dataset.get("name", "HuggingFaceFW/fineweb-edu"))
    dataset_config = str(dataset.get("config", "sample-10BT"))

    if ds_type == "local_jsonl":
        dataset_name = "json"
        dataset_config = ""
        data_files = str(dataset.get("path", "data/train.jsonl"))
    elif ds_type in {"local_txt", "local_text", "text"}:
        dataset_name = "text"
        dataset_config = ""
        data_files = str(dataset.get("path", "data/train.txt"))
    elif ds_type == "huggingface":
        pass
    else:
        raise SystemExit(f"Unsupported dataset.type: {ds_type}. Supported: huggingface, local_jsonl, local_txt")

    if data_files:
        data_path = Path(data_files)
        if not data_path.is_absolute():
            # Resolve relative dataset paths from project root whenever the
            # config lives under configs/, configs/examples/, configs/long/ or
            # configs/validation/. This keeps user-facing YAML stable.
            project_root = cfg_path.parent
            for parent in [cfg_path.parent, *cfg_path.parents]:
                if (parent / "configs").exists() and (parent / "scripts").exists():
                    project_root = parent
                    break
            data_path = (project_root / data_path).resolve()
        if not data_path.exists():
            raise SystemExit(f"Dataset file not found: {data_path}")
        data_files = str(data_path)

    values = {
        "MEM_V3_CONFIG": str(cfg_path),
        "MEM_RUN_NAME": run.get("name", "mem_v3_run"),
        "MEM_TARGET_GLOBAL_STEPS": int(run.get("target_global_steps", 300000)),
        "MEM_START_LANE": run.get("start_lane", "fast_seq256_zero0_gacc4"),
        "MEM_DATASET_TYPE": ds_type,
        "MEM_DATASET_NAME": dataset_name,
        "MEM_DATASET_CONFIG": dataset_config,
        "MEM_DATASET_SPLIT": dataset.get("split", "train"),
        "MEM_DATASET_TEXT_FIELD": dataset.get("text_field", "text"),
        "MEM_DATASET_FALLBACK_NAME": dataset.get("fallback_name", "roneneldan/TinyStories"),
        "MEM_DATASET_MIX": dataset.get("mix", ""),
        "MEM_DATASET_DATA_FILES": data_files,
        "MEM_TOKENIZER_NAME": tokenizer.get("name", "gpt2"),
        "MEM_MODEL_PRESET": model.get("preset", "tiny_decoder"),
        "MEM_CHAOS_PROFILE": runtime.get("chaos_profile", "real_desktop_contention"),
        "MEM_DATASET_PREFETCH_BATCHES": runtime.get("prefetch_batches", 12),
        "MEM_DATASET_PREWARM_BATCHES": runtime.get("prewarm_batches", runtime.get("prefetch_batches", 12)),
        "MEM_DATASET_CACHE_MODE": runtime.get("cache_mode", "memmap"),
        "MEM_DATASET_CACHE_DIR": runtime.get("cache_dir", "dataset_cache"),
    }
    for k, v in values.items():
        print(f"export {k}={q(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
