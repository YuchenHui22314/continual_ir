#!/bin/bash
# ANCE repeat of the turn-bucket forgetting experiment (the Qwen3 version is
# scripts/run_qwen_turn_buckets.sh; findings in figures/bucket_runs_eval.json).
# Train ANCE (RoBERTa-base, 768d) on each of the 13 single-length TopiOCQA
# buckets to test whether the short-conversation-toxicity / asymmetric-transfer
# story holds on a masked-LM encoder.
#
# Protocol mirrors the April ANCE production batch (lr 1e-5, wd 0, warmup 0.06
# + no_lr_schedule, ranking loss, negative none, caps 64/64/512, effective
# global batch 480 — production used 30/GPU x 4 GPU x accum 4 on octal31's
# 24 GB cards; on octal40's 46 GB L40S we use 120/GPU x 4 x accum 1, same
# effective batch) EXCEPT the three bucket-experiment deltas:
#   - step-driven: --total_train_steps 470 --save_every_steps 47
#   - curriculum none
#   - NO in-training eval (offline eval via eval_bucket_runs_per_ckpt.py-style
#     ANCE variant afterwards)
# plus --record_grad_stats (R1/R2 parameter-update recording, ~0.5 GB/buffer).
#
# Naming:
#   - wandb project:   topiocqa-ance-turn-buckets
#   - checkpoint dirs: OUT_BASE/bucket_ance_<bucket>
#   - run names:       bucket_ance_<bucket>

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_continually_ddp_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/ance_turn_buckets_master_${TS}.log

PRETRAIN=/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234
BUCKET_DIR=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/turn_buckets
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.pt
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir

WANDB_PROJECT=topiocqa-ance-turn-buckets

BUCKETS=(
  turn_1 turn_2 turn_3 turn_4 turn_5
  turn_6 turn_7 turn_8 turn_9 turn_10
  turn_11_12 turn_13_14 turn_15plus
)

echo "===== 13 ANCE turn-bucket runs kickoff at $(date) =====" | tee -a "$MASTER_LOG"
echo "wandb project: $WANDB_PROJECT | caps 64/64/512 | 470 steps, ckpt every 47" | tee -a "$MASTER_LOG"
echo "NO in-training eval (offline eval later); --record_grad_stats ON" | tee -a "$MASTER_LOG"

for BUCKET in "${BUCKETS[@]}"; do
  RUN_NAME="bucket_ance_${BUCKET}"
  TRAIN_FILE="${BUCKET_DIR}/topiocqa_${BUCKET}.jsonl"
  RUN_LOG=${LOG_DIR}/run_${RUN_NAME}_${TS}.log
  echo "===== [$(date)] START $RUN_NAME (file=$TRAIN_FILE) =====" | tee -a "$MASTER_LOG"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$TORCHRUN" --nproc_per_node=4 "$TRAIN_SCRIPT" \
    --pretrained_encoder_path "$PRETRAIN" \
    --training_data_file "$TRAIN_FILE" \
    --pos_neg_embedding_file "$POS_NEG" \
    --model_output_path "${OUT_BASE}/${RUN_NAME}" \
    --max_query_length 64 \
    --max_response_length 64 \
    --max_concat_length 512 \
    --use_data_percent 1.0 \
    --total_train_steps 470 \
    --save_every_steps 47 \
    --record_grad_stats \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 120 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.00 \
    --warmup_ratio 0.06 \
    --no_lr_schedule \
    --loss_type ranking \
    --negative_type none \
    --curriculum_type none \
    --pacing_function root_2 \
    --curriculum_c0 0.2 \
    --curriculum_end_epoch 16 \
    --n_gpu 4 \
    --save_to_wandb \
    --wandb_name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$RUN_LOG"

  RC=${PIPESTATUS[0]}
  echo "===== [$(date)] END   $RUN_NAME (exit=$RC) =====" | tee -a "$MASTER_LOG"
  sleep 5
done

echo "===== ALL 13 ANCE TURN-BUCKET RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
