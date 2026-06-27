#!/bin/bash
# Constant-LR ablation of the canonical InfoNCE 4x120 (B) run.
# Base = run_qwen_infonce_4x120.sh (= example_paper.tex tab:hparams LAST column
# "Qwen3 (InfoNCE 4x120, B)", fig:infonce PURPLE). This run is BYTE-IDENTICAL to it
# EXCEPT the LR schedule:
#     cosine + warmup 0.1   ->   constant (--no_lr_schedule)
# --no_lr_schedule builds LambdaLR(lr_lambda=lambda step: 1.0): no warmup, no decay,
# constant lr the whole run; warmup_ratio becomes a no-op (so we drop it + the cosine flag).
# Everything else unchanged: InfoNCE tau=0.01, 480 cross-GPU negs, mask_fake_negative(0.1)
# + dedup_same_gold, adam_beta2=0.95, wd=0, bf16_fp32_master + FlashAttention-2, 4x120 global
# 480, lr 1e-5, v3, 20 epochs, --record_grad_stats --grad_stats_device cpu.
# Purpose: isolate the cosine-schedule effect (the only changed factor vs the B column).
# Usage:  ./run_qwen_infonce_4x120_constlr.sh [learning_rate]   (default 1e-5)
set -u
LR="${1:-1e-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
RUN_NAME="instruct3fp32infonce_qwen_constlr"
RUN_LOG=${LOG_DIR}/run_${RUN_NAME}_${TS}.log
PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TRAIN_FILE=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir
WANDB_PROJECT=topiocqa-qwen-infonce-fp32
CONV_INSTR="Given a conversation between a user and an AI assistant, retrieve passages that answer the user's last question."

echo "===== $RUN_NAME (InfoNCE, B=bf16_fp32_master+FA2, 4x120, CONSTANT LR) kickoff at $(date) =====" | tee -a "$RUN_LOG"
echo "4 GPU (CVD=0,1,2,3) x 120 x accum 1 = global 480 | 480 cross-device negs | InfoNCE tau=0.01 | mask_fake_negative(0.1)+dedup_same_gold | CONSTANT LR (--no_lr_schedule, no warmup, no decay) | adam_beta2=0.95 | wd=0 | lr=$LR | v3 | 20 epochs | bf16_fp32_master+FA2 | gpu_resident+workers4 | record_grad_stats(cpu)" | tee -a "$RUN_LOG"

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
  --num_train_epochs 20 \
  --per_gpu_train_batch_size 120 \
  --gradient_accumulation_steps 1 \
  --learning_rate "$LR" \
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
  --record_grad_stats --grad_stats_device cpu \
  --n_gpu 4 \
  --save_to_wandb \
  --wandb_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  2>&1 | tee -a "$RUN_LOG"

echo "===== $RUN_NAME END (exit=${PIPESTATUS[0]}) at $(date) =====" | tee -a "$RUN_LOG"
