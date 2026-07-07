#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-$VERL_ROOT/outputs/baselines_prompt_only_3methods}"
LOG_ROOT="${LOG_ROOT:-$VERL_ROOT/outputs/baselines_suspicious_rerun_logs}"

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

mkdir -p "$LOG_ROOT"
cd "$ROOT"

MODEL_TAG="deepseek_r1_qwen_15b"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASETS=(medqa_usmle headqa pubmedqa_labeled)
GPUS=(0 1 2)

run_one() {
  local gpu="$1"
  local dataset="$2"
  local log="$LOG_ROOT/direct__${MODEL_TAG}__${dataset}.log"
  echo "===== START gpu=$gpu direct $MODEL_TAG $dataset ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VERL_ROOT/biomed/baseline_runner.py" \
    --baseline direct \
    --model-tag "$MODEL_TAG" \
    --model-path "$MODEL_PATH" \
    --dataset "$dataset" \
    --output-root "$OUT_ROOT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --direct-max-tokens "${DIRECT_MAX_TOKENS:-1024}" \
    > "$log" 2>&1
  echo "===== DONE gpu=$gpu direct $MODEL_TAG $dataset ====="
}

pids=()
for i in "${!DATASETS[@]}"; do
  run_one "${GPUS[$i]}" "${DATASETS[$i]}" > "$LOG_ROOT/worker_${i}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

"$PYTHON" "$VERL_ROOT/scripts/biomed/rescore_baseline_outputs.py" \
  > "$LOG_ROOT/rescore_after_rerun.log"

exit "$status"
