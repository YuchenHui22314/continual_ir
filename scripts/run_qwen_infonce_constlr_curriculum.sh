#!/bin/bash
# The 7 curriculum / anti-curriculum variants under the InfoNCE 4x120 + CONSTANT-LR
# recipe. The random-order baseline for this recipe is already trained:
#   instruct3fp32infonce_qwen_constlr  (run_qwen_infonce_4x120_constlr.sh).
# These 7 + that baseline = a full 8-row curriculum table at the official-aligned
# InfoNCE recipe, replacing the old bf16/in-batch-CE "main" setting of the paper's
# curriculum table.
#
# Why constant LR (NOT cosine): a curriculum ablation must keep the LR constant so
# the PACING FUNCTION is the only thing varying along the training-step axis. A
# cosine/warmup schedule also varies with step and would confound the curriculum's
# difficulty progression with LR decay (the step-pacing phase transitions at T/3,
# 2T/3 would be indistinguishable from schedule effects). Constant LR isolates the
# ordering. (Lit review 2026-06-27: older IR-CL rerank papers all use constant LR;
# DCL/COTED use linear-decay-warmup0 and never control for this — we are stricter.)
#
# Recipe = run_qwen_infonce_4x120_constlr.sh EXACTLY, plus --curriculum_type/--pacing_function.
# Differences from the old run_qwen_8_instruct3.sh: InfoNCE (cross-GPU negs, tau=0.01,
# fake-neg mask, dedup) + bf16_fp32_master + gpu_resident_doc_table; NO in-training eval
# (invalid <|im_end|> wrapper numbers; offline eval after); NO --record_grad_stats.
# Sequential — each run uses all 4 GPUs. ~4h/run x 7 ~= 28h.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/constlr_curriculum_master_${TS}.log

PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TRAIN_FILE=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir
WANDB_PROJECT=topiocqa-qwen-infonce-fp32
CONV_INSTR="Given a conversation between a user and an AI assistant, retrieve passages that answer the user's last question."

# 7 variants (name|curriculum_type|pacing_function). Random-order baseline already done.
RUNS=(
  "cl_step|easy2hard|step"
  "cl_step_excl|easy2hard|step_exclusive"
  "cl_step_excl_2_full|easy2hard|step_exclusive_2_full"
  "cl_root2|easy2hard|root_2"
  "acl_root2|hard2easy|root_2"
  "acl_step|hard2easy|step"
  "acl_step_excl|hard2easy|step_exclusive"
)

echo "===== 7 InfoNCE-constlr curriculum runs kickoff $(date) =====" | tee -a "$MASTER_LOG"
echo "recipe: InfoNCE 4x120 (480 cross-GPU negs) tau=0.01 + mask_fake_neg(0.1) + dedup | CONSTANT LR 1e-5 (--no_lr_schedule) | adam_beta2=0.95 | bf16_fp32_master+FA2 | v3 | 20ep | NO in-training eval | NO grad_stats" | tee -a "$MASTER_LOG"

for entry in "${RUNS[@]}"; do
  IFS='|' read -r NAME CTYPE PACE <<< "$entry"
  RUN_NAME="instruct3fp32infonce_constlr_${NAME}"
  RUN_LOG=${LOG_DIR}/run_${RUN_NAME}_${TS}.log
  echo "===== [$(date)] START $RUN_NAME (curriculum=$CTYPE, pacing=$PACE) =====" | tee -a "$MASTER_LOG"

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
    --learning_rate 1e-5 \
    --no_lr_schedule \
    --adam_beta2 0.95 \
    --loss_type ranking \
    --negative_type none \
    --cross_device_negatives \
    --infonce_temperature 0.01 \
    --mask_fake_negative \
    --fake_neg_margin 0.1 \
    --dedup_same_gold \
    --curriculum_type "$CTYPE" \
    --pacing_function "$PACE" \
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
  sleep 10
done

echo "===== ALL 7 CONSTLR CURRICULUM RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
