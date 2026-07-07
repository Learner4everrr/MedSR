#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$VERL_ROOT/.venv}"

cd "$VERL_ROOT"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$VERL_ROOT/.pip-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --no-compile --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --no-compile -e .
"$VENV_DIR/bin/python" -m pip install --no-compile "vllm==0.11.2" requests

cat <<EOF
Training environment ready:
  $VENV_DIR/bin/python

Use this for data/training:
  PYTHON=$VENV_DIR/bin/python scripts/biomed/prepare_data.sh
  PYTHON=$VENV_DIR/bin/python scripts/biomed/run_qwen25_3b_lora_pubmed_3datasets.sh

Retriever still uses the existing search env by default:
  RETRIEVER_PYTHON=$ROOT/search_env/bin/python scripts/biomed/launch_pubmed_retriever.sh
EOF
