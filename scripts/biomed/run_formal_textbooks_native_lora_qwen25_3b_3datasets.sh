#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CORPUS_NAME="${CORPUS_NAME:-textbooks}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-medqa_usmle headqa pubmedqa_labeled}"
export OUT_ROOT="${OUT_ROOT:-$VERL_ROOT/outputs/formal_textbooks_native_lora_qwen25_3b_500step}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-501}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export SAVE_FREQ="${SAVE_FREQ:-500}"
export TEST_FREQ="${TEST_FREQ:-500}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-1024}"

export TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"
export PROJECT_NAME="${PROJECT_NAME:-verl_biomed_search_lora_formal}"
export CLEAN_GENERATION_DIRS="${CLEAN_GENERATION_DIRS:-true}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
export BIOMED_FORMAT_REWARD="${BIOMED_FORMAT_REWARD:-0.1}"
export LAYERED_SUMMON="${LAYERED_SUMMON:-False}"

exec "$VERL_ROOT/scripts/biomed/run_textbooks_lora_flash.sh"
