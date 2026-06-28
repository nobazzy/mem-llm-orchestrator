#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# MEM v89 quality/loss run: 50M + curriculum-biased SlimPajama/TinyStories mix.
# Fresh 1M run intended for comparing loss behavior after the completed endurance run.
# API key is supplied by the user before running; do not hardcode it here.

export MEM_MODEL_PRESET="${MEM_MODEL_PRESET:-medium_50m}"
export MEM_DATASET_NAME="${MEM_DATASET_NAME:-DKYoon/SlimPajama-6B}"
export MEM_DATASET_CONFIG="${MEM_DATASET_CONFIG:-}"
export MEM_DATASET_SPLIT="${MEM_DATASET_SPLIT:-train}"
export MEM_DATASET_FALLBACK_NAME="${MEM_DATASET_FALLBACK_NAME:-roneneldan/TinyStories}"

# Curriculum intent: easier mix to test whether loss stabilizes better.
# Parser-compatible format used by the existing v89 launchers.
export MEM_DATASET_MIX="${MEM_DATASET_MIX:-DKYoon/SlimPajama-6B:0.40,roneneldan/TinyStories:0.60}"

export MEM_TOKENIZER_NAME="${MEM_TOKENIZER_NAME:-gpt2}"
export MEM_SEQUENCE_LENGTH="${MEM_SEQUENCE_LENGTH:-256}"
export MEM_TARGET_STEPS="${MEM_TARGET_STEPS:-1000000}"
export MEM_START_LANE="${MEM_START_LANE:-aggressive_seq256_zero0_gacc4}"
export MEM_CHAOS_PROFILE="${MEM_CHAOS_PROFILE:-real_desktop_contention}"

# Quality/loss training defaults.
export MEM_V89_FORCE_GRAD_ACCUM="${MEM_V89_FORCE_GRAD_ACCUM:-4}"
export MEM_V89_GRADIENT_ACCUMULATION_STEPS="${MEM_V89_GRADIENT_ACCUMULATION_STEPS:-4}"

# Force quality grad clipping.
export MEM_V89_GRADIENT_CLIP_NORM="${MEM_V89_GRADIENT_CLIP_NORM:-0.5}"
export MEM_GRAD_CLIP="${MEM_GRAD_CLIP:-0.5}"
export GRAD_CLIP="${GRAD_CLIP:-0.5}"

export MEM_V89_LR_SCHEDULE="${MEM_V89_LR_SCHEDULE:-1}"
export MEM_V89_BASE_LR_REAL="${MEM_V89_BASE_LR_REAL:-1.0e-5}"
export MEM_V89_LR_PEAK_CAP="${MEM_V89_LR_PEAK_CAP:-1.0e-5}"
export MEM_V89_LR_WARMUP_STEPS="${MEM_V89_LR_WARMUP_STEPS:-8000}"
export MEM_V89_LR_DECAY_STEPS="${MEM_V89_LR_DECAY_STEPS:-1000000}"
export MEM_V89_LR_MIN_MULT="${MEM_V89_LR_MIN_MULT:-0.10}"
export MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH="${MEM_V89_MIN_LANE_STEPS_BEFORE_SWITCH:-50000}"

# Durable live checkpoint/resume. Controller searches v89_live_* and validates model_preset.
export MEM_RUN_ID="${MEM_RUN_ID:-quality_50m_$(date +%Y%m%d_%H%M%S)}"
export MEM_CHECKPOINT_LABEL="${MEM_CHECKPOINT_LABEL:-v89_quality}"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-v89_quality}"
export MEM_AUTO_RESUME_CHECKPOINT="${MEM_AUTO_RESUME_CHECKPOINT:-1}"
export MEM_LIVE_CHECKPOINT_EVERY_STEPS="${MEM_LIVE_CHECKPOINT_EVERY_STEPS:-1000}"

bash scripts/run_v89_sustained_control.sh
