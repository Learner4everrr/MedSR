#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
WAIT_SESSION="${WAIT_SESSION:-verl_textbooks_llama_deepseek_6runs}"
LOG_FILE="${LOG_FILE:-$VERL_ROOT/outputs/formal_textbooks_deepseek_qwen7b_waiter.log}"
RUN_LOG="${RUN_LOG:-$VERL_ROOT/outputs/formal_textbooks_deepseek_qwen7b_tmux.log}"

mkdir -p "$VERL_ROOT/outputs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for tmux session: $WAIT_SESSION" | tee -a "$LOG_FILE"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] $WAIT_SESSION ended; starting DeepSeek Qwen 7B run" | tee -a "$LOG_FILE"
cd "$ROOT"
exec bash "$VERL_ROOT/scripts/biomed/run_formal_textbooks_native_lora_deepseek_qwen7b_3datasets.sh" 2>&1 | tee "$RUN_LOG"
