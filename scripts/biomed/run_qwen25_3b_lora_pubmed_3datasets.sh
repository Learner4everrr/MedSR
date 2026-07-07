#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$ROOT}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$VERL_ROOT/data/biomed}"
CORPUS_NAME="${CORPUS_NAME:-pubmed}"
OUT_ROOT="${OUT_ROOT:-$VERL_ROOT/outputs/${CORPUS_NAME}_lora_qwen25_3b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B}"
MODEL_TAG="${MODEL_TAG:-qwen25_3b}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$OUT_ROOT/wandb}"
export PYTHONUNBUFFERED=1
export SEARCH_TOOL_CONFIG="${SEARCH_TOOL_CONFIG:-$VERL_ROOT/config/biomed/search_tool.yaml}"

mkdir -p "$OUT_ROOT"
cd "$VERL_ROOT"

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
else
  DATASETS=(medqa_usmle headqa pubmedqa_labeled)
fi

for DATASET in "${DATASETS[@]}"; do
  EXP="${MODEL_TAG}_lora_${CORPUS_NAME}_${DATASET}"
  CKPT_DIR="$OUT_ROOT/$DATASET/checkpoints"
  ROLLOUT_DIR="$OUT_ROOT/$DATASET/rollouts"
  VAL_GEN_DIR="$OUT_ROOT/$DATASET/validation_generations"
  LOG_FILE="$OUT_ROOT/$DATASET/train.log"
  if [[ "${CLEAN_GENERATION_DIRS:-true}" == "true" ]]; then
    rm -rf "$ROLLOUT_DIR" "$VAL_GEN_DIR"
  fi
  mkdir -p "$OUT_ROOT/$DATASET" "$ROLLOUT_DIR" "$VAL_GEN_DIR"

  "$PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_ROOT/$DATASET/train.parquet" \
    data.val_files="$DATA_ROOT/$DATASET/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-4}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-64}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-3e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.55}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon="${LAYERED_SUMMON:-False}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$SEARCH_TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS:-2}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-3}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-1024}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    reward.custom_reward_function.path="$VERL_ROOT/biomed/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger="${TRAINER_LOGGER:-[console,wandb]}" \
    trainer.project_name="${PROJECT_NAME:-verl_biomed_search_lora}" \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-4}" \
    trainer.nnodes=1 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
    trainer.save_freq="${SAVE_FREQ:-500}" \
    trainer.test_freq="${TEST_FREQ:-500}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-501}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.validation_data_dir="$VAL_GEN_DIR" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS:-0}" \
    trainer.default_local_dir="$CKPT_DIR" \
    2>&1 | tee "$LOG_FILE"
done
