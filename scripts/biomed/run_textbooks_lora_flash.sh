#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
CUDA128_HOME="${CUDA128_HOME:-$VERL_ROOT/cuda128}"

export CUDA_HOME="$CUDA128_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CORPUS_NAME="${CORPUS_NAME:-textbooks}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-501}"
export SAVE_FREQ="${SAVE_FREQ:-500}"
export TEST_FREQ="${TEST_FREQ:-500}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"

exec "$VERL_ROOT/scripts/biomed/run_qwen25_3b_lora_pubmed_3datasets.sh"
