#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

"$PYTHON" "$VERL_ROOT/biomed/prepare_biomed_multiturn.py" \
  --src-root "$ROOT/data/biomed" \
  --dst-root "$VERL_ROOT/data/biomed"
