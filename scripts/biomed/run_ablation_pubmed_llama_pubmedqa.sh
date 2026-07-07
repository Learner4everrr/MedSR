#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-python}"
OUT_BASE="${OUT_BASE:-$VERL_ROOT/outputs/ablations_pubmed_llama_pubmedqa}"
LOG_ROOT="${LOG_ROOT:-$OUT_BASE/logs}"
TOOL_CONFIG_ROOT="${TOOL_CONFIG_ROOT:-$OUT_BASE/tool_configs}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL_TAG="${MODEL_TAG:-llama32_3b_instruct}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
DATASET="${DATASET:-pubmedqa_labeled}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
RETRIEVER_GPUS="${RETRIEVER_GPUS:-0,1,2,3}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_WAIT_SECONDS="${RETRIEVER_WAIT_SECONDS:-900}"

mkdir -p "$OUT_BASE" "$LOG_ROOT" "$TOOL_CONFIG_ROOT"
cd "$ROOT"

retriever_pid=""
current_retriever_key=""

cleanup_retriever() {
  if [[ -n "${retriever_pid:-}" ]] && kill -0 "$retriever_pid" 2>/dev/null; then
    kill "$retriever_pid" 2>/dev/null || true
    wait "$retriever_pid" 2>/dev/null || true
  fi
  retriever_pid=""
  current_retriever_key=""
}

trap cleanup_retriever EXIT

stop_existing_retrievers() {
  if [[ "${STOP_EXISTING_RETRIEVER:-1}" == "1" ]]; then
    pkill -f "search_r1/search/retrieval_server.py" 2>/dev/null || true
  fi
}

wait_for_retriever() {
  local key="$1"
  local waited=0
  while (( waited < RETRIEVER_WAIT_SECONDS )); do
    if URL="$RETRIEVER_URL" "$PYTHON" -c 'import os, requests, sys
try:
    r = requests.post(os.environ["URL"], json={"queries": ["aspirin treatment"], "topk": 1, "return_scores": False}, timeout=10)
    r.raise_for_status()
    sys.exit(0 if "result" in r.json() else 1)
except Exception:
    sys.exit(1)
'; then
      echo "Retriever ready: $key after ${waited}s"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "Retriever failed readiness check: $key" >&2
  return 1
}

model_cached() {
  local model_path="$1"
  "$PYTHON" - "$model_path" <<'PY'
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
}

start_retriever() {
  local corpus="$1"
  local retriever_name="$2"
  local retriever_model="$3"
  local retriever_pooling="$4"
  local key="${corpus}__${retriever_name}"

  if [[ "$current_retriever_key" == "$key" ]] && [[ -n "${retriever_pid:-}" ]] && kill -0 "$retriever_pid" 2>/dev/null; then
    return 0
  fi

  cleanup_retriever
  stop_existing_retrievers

  if [[ "$retriever_name" != "bm25" ]] && ! model_cached "$retriever_model"; then
    echo "SKIP retriever=$retriever_name model=$retriever_model because it is not in local HF cache." >&2
    return 2
  fi

  local log="$LOG_ROOT/retriever__${key}.log"
  echo "===== START retriever corpus=$corpus retriever=$retriever_name model=$retriever_model pooling=$retriever_pooling ====="
  (
    cd "$VERL_ROOT"
    CUDA_VISIBLE_DEVICES="$RETRIEVER_GPUS" \
    CORPUS_NAME="$corpus" \
    RETRIEVER_NAME="$retriever_name" \
    RETRIEVER_MODEL="$retriever_model" \
    RETRIEVER_POOLING="$retriever_pooling" \
    TOPK="${RETRIEVER_SERVER_TOPK:-10}" \
    RETRIEVER_PYTHON="$RETRIEVER_PYTHON" \
    bash scripts/biomed/launch_pubmed_retriever.sh
  ) > "$log" 2>&1 &
  retriever_pid="$!"
  current_retriever_key="$key"
  wait_for_retriever "$key"
}

write_tool_config() {
  local path="$1"
  local topk="$2"
  local max_doc_chars="$3"
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<YAML
tools:
  - class_name: biomed.search_tool.BiomedSearchTool
    config:
      type: native
      url: $RETRIEVER_URL
      topk: $topk
      timeout: 30
      max_doc_chars: $max_doc_chars
YAML
}

run_job() {
  local group="$1"
  local job="$2"
  local corpus="${3:-pubmed}"
  local retriever_name="${4:-e5}"
  local retriever_model="${5:-intfloat/e5-base-v2}"
  local retriever_pooling="${6:-mean}"
  local topk="${7:-2}"
  local max_doc_chars="${8:-900}"
  local steps="${9:-501}"
  local save_freq="${10:-500}"
  local test_freq="${11:-500}"
  local reward_mode="${12:-answer_plus_format}"
  local lora_rank="${13:-64}"
  local val_before_train="${14:-False}"
  local out_root="$OUT_BASE/$group/$job"
  local done_file="$out_root/$DATASET/validation_generations/501.jsonl"

  if [[ "${SKIP_FINISHED:-1}" == "1" ]] && [[ -s "$done_file" ]]; then
    echo "===== SKIP group=$group job=$job because $done_file already exists ====="
    return 0
  fi

  if (( topk > 0 )); then
    if ! start_retriever "$corpus" "$retriever_name" "$retriever_model" "$retriever_pooling"; then
      echo "===== SKIP group=$group job=$job because retriever could not start ====="
      return 0
    fi
  fi

  local tool_config="$TOOL_CONFIG_ROOT/${group}__${job}.yaml"
  local log="$LOG_ROOT/train__${group}__${job}.log"
  write_tool_config "$tool_config" "$topk" "$max_doc_chars"

  echo "===== START group=$group job=$job corpus=$corpus retriever=$retriever_name topk=$topk steps=$steps reward=$reward_mode lora=$lora_rank ====="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  CORPUS_NAME="$corpus" \
  DATASETS_OVERRIDE="$DATASET" \
  MODEL_TAG="$MODEL_TAG" \
  MODEL_PATH="$MODEL_PATH" \
  OUT_ROOT="$out_root" \
  SEARCH_TOOL_CONFIG="$tool_config" \
  TOTAL_TRAINING_STEPS="$steps" \
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}" \
  SAVE_FREQ="$save_freq" \
  TEST_FREQ="$test_freq" \
  VAL_BEFORE_TRAIN="$val_before_train" \
  CLEAN_GENERATION_DIRS=true \
  BIOMED_REWARD_MODE="$reward_mode" \
  LORA_RANK="$lora_rank" \
  LORA_ALPHA="${LORA_ALPHA:-32}" \
  TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}" \
  PROJECT_NAME="${PROJECT_NAME:-verl_biomed_ablation_pubmed_llama_pubmedqa}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}" \
  ROLLOUT_N="${ROLLOUT_N:-4}" \
  ROLLOUT_TP="${ROLLOUT_TP:-4}" \
  ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}" \
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}" \
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}" \
  MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-1024}" \
  bash "$VERL_ROOT/scripts/biomed/run_textbooks_lora_flash.sh" > "$log" 2>&1
  echo "===== DONE group=$group job=$job ====="
}

run_selected_groups() {
  local groups="${ABLATION_GROUPS:-topk steps reward retriever lora}"

  if [[ " $groups " == *" topk "* ]]; then
    run_job topk k0_no_retrieval pubmed e5 intfloat/e5-base-v2 mean 0 900 501 500 500 answer_plus_format 64 False
    run_job topk k1 pubmed e5 intfloat/e5-base-v2 mean 1 900 501 500 500 answer_plus_format 64 False
    run_job topk k2 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job topk k3 pubmed e5 intfloat/e5-base-v2 mean 3 900 501 500 500 answer_plus_format 64 False
    run_job topk k5 pubmed e5 intfloat/e5-base-v2 mean 5 900 501 500 500 answer_plus_format 64 False
  fi

  if [[ " $groups " == *" steps "* ]]; then
    run_job steps train_2000_eval250 pubmed e5 intfloat/e5-base-v2 mean 2 900 2001 250 250 answer_plus_format 64 True
  fi

  if [[ " $groups " == *" reward "* ]]; then
    run_job reward answer_plus_format pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job reward accuracy_only pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 accuracy_only 64 False
    run_job reward format_heavy pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format_heavy 64 False
    run_job reward search_light pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_search_light 64 False
    run_job reward length_penalty pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_length_penalty 64 False
  fi

  if [[ " $groups " == *" lora "* ]]; then
    run_job lora rank4 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 4 False
    run_job lora rank8 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 8 False
    run_job lora rank16 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 16 False
    run_job lora rank32 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 32 False
    run_job lora rank64 pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
  fi

  if [[ " $groups " == *" retriever "* ]]; then
    run_job retriever e5_pubmed pubmed e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job retriever e5_textbooks textbooks e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job retriever e5_statpearls statpearls e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job retriever e5_wikipedia wikipedia e5 intfloat/e5-base-v2 mean 2 900 501 500 500 answer_plus_format 64 False
    run_job retriever contriever_pubmed pubmed contriever facebook/contriever-msmarco mean 2 900 501 500 500 answer_plus_format 64 False
    run_job retriever bge_pubmed pubmed bge BAAI/bge-base-en-v1.5 cls 2 900 501 500 500 answer_plus_format 64 False
    if "$PYTHON" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pyserini") else 1)' >/dev/null 2>&1; then
      run_job retriever bm25_pubmed pubmed bm25 none mean 2 900 501 500 500 answer_plus_format 64 False
    else
      echo "SKIP retriever=bm25 because pyserini is not installed."
    fi
  fi
}

run_selected_groups
"$PYTHON" "$VERL_ROOT/scripts/biomed/summarize_ablation_pubmed_llama_pubmedqa.py" --root "$OUT_BASE" --out "$OUT_BASE/summary.tsv"
