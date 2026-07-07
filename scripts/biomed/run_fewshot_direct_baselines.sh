#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-$VERL_ROOT/outputs/baselines_fewshot_direct}"
LOG_ROOT="${LOG_ROOT:-$VERL_ROOT/outputs/baselines_fewshot_direct_logs}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

MODELS=(
  "qwen25_3b|Qwen/Qwen2.5-3B"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct"
  "deepseek_r1_qwen_15b|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
DATASETS=(medqa_usmle headqa pubmedqa_labeled)
BASELINES=(${BASELINES:-direct_1shot direct_3shot})
GPUS=(0 1 2 3)

TASK_FILE="$LOG_ROOT/tasks.tsv"
: > "$TASK_FILE"
for baseline in "${BASELINES[@]}"; do
  for model in "${MODELS[@]}"; do
    model_tag="${model%%|*}"
    model_path="${model#*|}"
    for dataset in "${DATASETS[@]}"; do
      printf '%s\t%s\t%s\t%s\n' "$baseline" "$model_tag" "$model_path" "$dataset" >> "$TASK_FILE"
    done
  done
done

run_task() {
  local gpu="$1"
  local baseline="$2"
  local model_tag="$3"
  local model_path="$4"
  local dataset="$5"
  local log="$LOG_ROOT/${baseline}__${model_tag}__${dataset}.log"

  echo "===== START gpu=$gpu baseline=$baseline model=$model_tag dataset=$dataset ====="
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VERL_ROOT/biomed/baseline_runner.py" \
    --baseline "$baseline" \
    --model-tag "$model_tag" \
    --model-path "$model_path" \
    --dataset "$dataset" \
    --output-root "$OUT_ROOT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --direct-max-tokens "${DIRECT_MAX_TOKENS:-64}" \
    > "$log" 2>&1; then
    echo "===== DONE gpu=$gpu baseline=$baseline model=$model_tag dataset=$dataset ====="
  else
    echo "===== FAILED gpu=$gpu baseline=$baseline model=$model_tag dataset=$dataset ====="
    return 1
  fi
}

worker() {
  local worker_id="$1"
  local gpu="${GPUS[$worker_id]}"
  local line_no=0

  while IFS=$'\t' read -r baseline model_tag model_path dataset; do
    if (( line_no % ${#GPUS[@]} == worker_id )); then
      run_task "$gpu" "$baseline" "$model_tag" "$model_path" "$dataset" || return 1
    fi
    line_no=$((line_no + 1))
  done < "$TASK_FILE"
}

pids=()
for worker_id in "${!GPUS[@]}"; do
  worker "$worker_id" > "$LOG_ROOT/worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/baselines_fewshot_direct")
rows = []
for path in sorted(root.glob("*/*/*/summary.json")):
    rows.append(json.loads(path.read_text()))

print("baseline\tmodel\tdataset\tN\tacc\tcorrect\tscore\tformat")
for row in rows:
    print(
        f"{row['baseline']}\t{row['model_tag']}\t{row['dataset']}\t"
        f"{row['n']}\t{row['accuracy']:.4f}\t{row['correct']}\t"
        f"{row['score']:.4f}\t{row['format']:.4f}"
    )
PY

exit "$status"
