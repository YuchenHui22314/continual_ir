#!/bin/bash
# InfoNCE 6e-6 constant-LR rerun of the 13 turn-length buckets, aligned to the
# baseline/curriculum recipe (instead of the original in-batch-CE fp32-master).
#
# Per-bucket causal-forgetting study: each bucket trained STEP-DRIVEN (470 steps,
# ckpt every 47 = 10 ckpts) with --record_grad_stats for the Fisher / EWC /
# sign-coherence analysis. fp32-master numerics via --bf16_fp32_master (precondition;
# pure-bf16 rounding floor would void the gradient analysis), weight_decay=0,
# curriculum none, NO in-training eval (offline eval is the single source of truth).
#
# Recipe = the 6e-6 const curriculum runs MINUS curriculum, PLUS step-driven + grad_stats:
#   InfoNCE (cross-GPU negs, tau=0.01, fake-neg mask 0.1, same-gold dedup), adam_beta2=0.95,
#   lr 6e-6 constant (--no_lr_schedule), bf16_fp32_master + FA2, v3, 4x120 = global 480.
#
# Naming bucket_infonce6e6_<bucket> — does NOT touch the old bucket_qwen32_* dirs.
# num_train_epochs is a high cap (200); the run actually stops at total_train_steps=470
# (3,054 pairs / 480 batch = ~6.4 steps/epoch -> 470 steps ~= 73 epochs of cycling).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/qwen_turn_buckets_infonce6e6_master_${TS}.log

PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
BUCKET_DIR=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/turn_buckets
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir
WANDB_PROJECT=topiocqa-qwen-turn-buckets-infonce6e6
CONV_INSTR="Given a conversation between a user and an AI assistant, retrieve passages that answer the user's last question."

BUCKETS=(
  turn_1 turn_2 turn_3 turn_4 turn_5
  turn_6 turn_7 turn_8 turn_9 turn_10
  turn_11_12 turn_13_14 turn_15plus
)

# Allow a single bucket as $1 (for smoke tests); default = all 13.
if [ "${1:-}" != "" ]; then BUCKETS=("$1"); fi

echo "===== InfoNCE-6e6 turn-bucket runs (${#BUCKETS[@]}) kickoff at $(date) =====" | tee -a "$MASTER_LOG"
echo "InfoNCE 4x120 (480 cross-GPU negs) tau=0.01 + mask_fake_neg(0.1) + dedup | CONSTANT LR 6e-6 | adam_beta2=0.95 | wd=0 | bf16_fp32_master+FA2 | v3 | step-driven 470/47 (10 ckpt) | record_grad_stats(cpu) | NO in-training eval" | tee -a "$MASTER_LOG"

for BUCKET in "${BUCKETS[@]}"; do
  RUN_NAME="bucket_infonce6e6_${BUCKET}"
  TRAIN_FILE="${BUCKET_DIR}/topiocqa_${BUCKET}.jsonl"
  RUN_LOG=${LOG_DIR}/run_${RUN_NAME}_${TS}.log
  echo "===== [$(date)] START $RUN_NAME (file=$TRAIN_FILE) =====" | tee -a "$MASTER_LOG"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TORCHRUN" --nproc_per_node=4 "$TRAIN_SCRIPT" \
    --pretrained_encoder_path "$PRETRAIN" \
    --training_data_file "$TRAIN_FILE" \
    --pos_neg_embedding_file "$POS_NEG" \
    --model_output_path "${OUT_BASE}/${RUN_NAME}" \
    --conv_instruction "$CONV_INSTR" \
    --template_version v3 \
    --max_query_length 32768 \
    --max_response_length 32768 \
    --max_concat_length 32768 \
    --use_data_percent 1.0 \
    --total_train_steps 470 \
    --save_every_steps 47 \
    --record_grad_stats --grad_stats_device cpu \
    --num_train_epochs 200 \
    --per_gpu_train_batch_size 120 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-6 \
    --no_lr_schedule \
    --adam_beta2 0.95 \
    --loss_type ranking \
    --negative_type none \
    --cross_device_negatives \
    --infonce_temperature 0.01 \
    --mask_fake_negative \
    --fake_neg_margin 0.1 \
    --dedup_same_gold \
    --curriculum_type none \
    --pacing_function root_2 \
    --curriculum_c0 0.2 \
    --curriculum_end_epoch 16 \
    --embed_dim 1024 \
    --encoder_type qwen3 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table \
    --dataloader_num_workers 4 --dataloader_prefetch_factor 4 \
    --n_gpu 4 \
    --save_to_wandb \
    --wandb_name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$RUN_LOG"

  RC=${PIPESTATUS[0]}
  echo "===== [$(date)] END   $RUN_NAME (exit=$RC) =====" | tee -a "$MASTER_LOG"
  sleep 5
done

echo "===== ALL ${#BUCKETS[@]} InfoNCE-6e6 TURN-BUCKET RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
