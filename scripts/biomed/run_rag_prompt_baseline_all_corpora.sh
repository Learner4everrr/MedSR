#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-$VERL_ROOT/outputs/baselines_rag_prompt_all_corpora}"
LOG_ROOT="${LOG_ROOT:-$VERL_ROOT/outputs/baselines_rag_prompt_all_corpora_logs}"
RETRIEVER_LOG_ROOT="${RETRIEVER_LOG_ROOT:-$LOG_ROOT/retrievers}"

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

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$RETRIEVER_LOG_ROOT"
cd "$ROOT"

MODELS=(
  "qwen25_3b|Qwen/Qwen2.5-3B"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct"
  "deepseek_r1_qwen_15b|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
DATASETS=(medqa_usmle headqa pubmedqa_labeled)
GPUS=(0 1 2 3)
CORPORA=(${CORPORA:-textbooks pubmed statpearls wikipedia})

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
TOPK="${TOPK:-3}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-900}"
RETRIEVE_BATCH_SIZE="${RETRIEVE_BATCH_SIZE:-32}"
RETRIEVER_GPUS="${RETRIEVER_GPUS:-0,1,2,3}"
RETRIEVER_WAIT_SECONDS="${RETRIEVER_WAIT_SECONDS:-900}"

retriever_pid=""

cleanup_retriever() {
  if [[ -n "${retriever_pid:-}" ]] && kill -0 "$retriever_pid" 2>/dev/null; then
    kill "$retriever_pid" 2>/dev/null || true
    wait "$retriever_pid" 2>/dev/null || true
  fi
  retriever_pid=""
}

trap cleanup_retriever EXIT

stop_existing_retrievers() {
  if [[ "${STOP_EXISTING_RETRIEVER:-1}" == "1" ]]; then
    pkill -f "search_r1/search/retrieval_server.py" 2>/dev/null || true
  fi
}

wait_for_retriever() {
  local corpus="$1"
  local waited=0
  while (( waited < RETRIEVER_WAIT_SECONDS )); do
    if URL="$RETRIEVER_URL" "$PYTHON" -c 'import os, requests, sys
try:
    r = requests.post(os.environ["URL"], json={"queries": ["aspirin"], "topk": 1, "return_scores": False}, timeout=10)
    r.raise_for_status()
    data = r.json()
    sys.exit(0 if "result" in data else 1)
except Exception:
    sys.exit(1)
'; then
      echo "Retriever ready for corpus=$corpus after ${waited}s"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "Retriever failed to become ready for corpus=$corpus within ${RETRIEVER_WAIT_SECONDS}s" >&2
  return 1
}

start_retriever() {
  local corpus="$1"
  cleanup_retriever
  stop_existing_retrievers
  local log="$RETRIEVER_LOG_ROOT/${corpus}.log"
  echo "===== START retriever corpus=$corpus gpus=$RETRIEVER_GPUS ====="
  (
    cd "$VERL_ROOT"
    CUDA_VISIBLE_DEVICES="$RETRIEVER_GPUS" \
    CORPUS_NAME="$corpus" \
    TOPK="$TOPK" \
    RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-python}" \
    bash scripts/biomed/launch_pubmed_retriever.sh
  ) > "$log" 2>&1 &
  retriever_pid="$!"
  if ! wait_for_retriever "$corpus"; then
    echo "Retriever readiness was not confirmed for corpus=$corpus; continuing and letting retrieval tasks fail if the server is unavailable." >&2
  fi
}

run_task() {
  local corpus="$1"
  local gpu="$2"
  local model_tag="$3"
  local model_path="$4"
  local dataset="$5"
  local log="$LOG_ROOT/rag_prompt__${corpus}__${model_tag}__${dataset}.log"

  echo "===== START gpu=$gpu corpus=$corpus model=$model_tag dataset=$dataset ====="
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VERL_ROOT/biomed/baseline_runner.py" \
    --baseline rag_prompt \
    --corpus "$corpus" \
    --model-tag "$model_tag" \
    --model-path "$model_path" \
    --dataset "$dataset" \
    --output-root "$OUT_ROOT" \
    --retriever-url "$RETRIEVER_URL" \
    --topk "$TOPK" \
    --max-doc-chars "$MAX_DOC_CHARS" \
    --retrieve-batch-size "$RETRIEVE_BATCH_SIZE" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    > "$log" 2>&1; then
    echo "===== DONE gpu=$gpu corpus=$corpus model=$model_tag dataset=$dataset ====="
  else
    echo "===== FAILED gpu=$gpu corpus=$corpus model=$model_tag dataset=$dataset ====="
    return 1
  fi
}

run_corpus() {
  local corpus="$1"
  local task_file="$LOG_ROOT/tasks_${corpus}.tsv"
  : > "$task_file"
  for model in "${MODELS[@]}"; do
    local model_tag="${model%%|*}"
    local model_path="${model#*|}"
    for dataset in "${DATASETS[@]}"; do
      printf '%s\t%s\t%s\n' "$model_tag" "$model_path" "$dataset" >> "$task_file"
    done
  done

  start_retriever "$corpus"

  worker() {
    local worker_id="$1"
    local gpu="${GPUS[$worker_id]}"
    local line_no=0
    while IFS=$'\t' read -r model_tag model_path dataset; do
      if (( line_no % ${#GPUS[@]} == worker_id )); then
        run_task "$corpus" "$gpu" "$model_tag" "$model_path" "$dataset" || return 1
      fi
      line_no=$((line_no + 1))
    done < "$task_file"
  }

  local pids=()
  for worker_id in "${!GPUS[@]}"; do
    worker "$worker_id" > "$LOG_ROOT/worker_${corpus}_${worker_id}.log" 2>&1 &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  cleanup_retriever
  return "$status"
}

status=0
for corpus in "${CORPORA[@]}"; do
  if ! run_corpus "$corpus"; then
    status=1
    echo "Corpus failed: $corpus" >&2
  fi
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/baselines_rag_prompt_all_corpora")
rows = []
for path in sorted(root.glob("rag_prompt/*/*/*/summary.json")):
    rows.append(json.loads(path.read_text()))

print("baseline\tcorpus\tmodel\tdataset\tN\tacc\tcorrect\tscore\tformat")
for row in rows:
    print(
        f"{row['baseline']}\t{row.get('corpus')}\t{row['model_tag']}\t{row['dataset']}\t"
        f"{row['n']}\t{row['accuracy']:.4f}\t{row['correct']}\t"
        f"{row['score']:.4f}\t{row['format']:.4f}"
    )
PY

exit "$status"
