#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-python}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_POOLING="${RETRIEVER_POOLING:-mean}"
TOPK="${TOPK:-3}"
CORPUS_NAME="${CORPUS_NAME:-pubmed}"

CORPUS_PATH="${CORPUS_PATH:-$ROOT/data/biomed_corpora/medrag_${CORPUS_NAME}/corpus.jsonl}"
INDEX_DIR="${INDEX_DIR:-$ROOT/data/biomed_indexes/medrag_${CORPUS_NAME}}"
if [[ "$RETRIEVER_NAME" == "bm25" ]]; then
  INDEX_PATH="${INDEX_PATH:-$INDEX_DIR/bm25}"
else
  INDEX_PATH="${INDEX_PATH:-$INDEX_DIR/${RETRIEVER_NAME}_Flat.index}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONUNBUFFERED=1

cd "$VERL_ROOT"

if [[ ! -e "$INDEX_PATH" ]]; then
  if [[ "$RETRIEVER_NAME" == "bm25" ]]; then
    "$RETRIEVER_PYTHON" search_r1/search/index_builder.py \
      --retrieval_method "$RETRIEVER_NAME" \
      --corpus_path "$CORPUS_PATH" \
      --save_dir "$INDEX_DIR"
  else
    "$RETRIEVER_PYTHON" search_r1/search/index_builder.py \
      --retrieval_method "$RETRIEVER_NAME" \
      --model_path "$RETRIEVER_MODEL" \
      --corpus_path "$CORPUS_PATH" \
      --save_dir "$INDEX_DIR" \
      --max_length "${INDEX_MAX_LENGTH:-180}" \
      --batch_size "${INDEX_BATCH_SIZE:-256}" \
      --use_fp16 \
      --pooling_method "$RETRIEVER_POOLING" \
      --faiss_type Flat \
      --faiss_gpu
  fi
fi

SERVER_ARGS=(
  --index_path "$INDEX_PATH"
  --corpus_path "$CORPUS_PATH"
  --topk "$TOPK"
  --retriever_name "$RETRIEVER_NAME"
  --retriever_model "$RETRIEVER_MODEL"
  --pooling_method "$RETRIEVER_POOLING"
)
if [[ "$RETRIEVER_NAME" != "bm25" ]]; then
  SERVER_ARGS+=(--faiss_gpu)
fi

exec "$RETRIEVER_PYTHON" search_r1/search/retrieval_server.py "${SERVER_ARGS[@]}"
