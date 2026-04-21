#!/bin/bash
# Launch 4 Qwen3-Embedding-0.6B fine-tuning runs on TopiOCQA at lr=1e-5.
# Serial execution on 4 GPUs. Each run = 20 epochs, global batch 480.
# Remaining 4 (cl_root2, acl_root2, acl_step, acl_step_excl) run on another machine.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/qwen_8runs_master_${TS}.log

# Shared args
PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TRAIN_FILE=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
TOPIOCQA_EMB=/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged
TOPIOCQA_VALID=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl
TOPIOCQA_QREL=/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir

# Subset: baseline + 3 easy2hard curriculum (excluding root_2, which runs on another machine)
# Remaining 4 (cl_root2, acl_root2, acl_step, acl_step_excl) run elsewhere.
RUNS=(
  "qwen_nosched|none|root_2"
  "qwen_cl_step|easy2hard|step"
  "qwen_cl_step_excl|easy2hard|step_exclusive"
  "qwen_cl_step_excl_2_full|easy2hard|step_exclusive_2_full"
)

echo "===== 4 Qwen runs kickoff at $(date) =====" | tee -a "$MASTER_LOG"

for entry in "${RUNS[@]}"; do
  IFS='|' read -r NAME CTYPE PACE <<< "$entry"
  RUN_LOG=${LOG_DIR}/run_${NAME}_${TS}.log
  echo "===== [$(date)] START $NAME (curriculum=$CTYPE, pacing=$PACE) =====" | tee -a "$MASTER_LOG"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=4 "$TRAIN_SCRIPT" \
    --pretrained_encoder_path "$PRETRAIN" \
    --training_data_file "$TRAIN_FILE" \
    --pos_neg_embedding_file "$POS_NEG" \
    --model_output_path "${OUT_BASE}/${NAME}" \
    --topiocqa_embedding_dir "$TOPIOCQA_EMB" \
    --topiocqa_valid_file "$TOPIOCQA_VALID" \
    --topiocqa_qrel_file "$TOPIOCQA_QREL" \
    --use_data_percent 1.0 \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 120 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.06 \
    --no_lr_schedule \
    --loss_type ranking \
    --negative_type none \
    --curriculum_type "$CTYPE" \
    --pacing_function "$PACE" \
    --curriculum_c0 0.2 \
    --curriculum_end_epoch 16 \
    --embed_dim 1024 \
    --encoder_type qwen3 \
    --use_flash_attention --use_bf16 --gradient_checkpointing \
    --activate_eval_topiocqa_while_training \
    --activate_eval_while_training \
    --beir_datasets climate-fever msmarco \
    --eval_batch_size 64 \
    --use_gpu_faiss \
    --n_gpu 4 \
    --save_to_wandb \
    --wandb_name "$NAME" \
    --wandb_project topiocqa-qwen \
    2>&1 | tee -a "$RUN_LOG"

  RC=${PIPESTATUS[0]}
  echo "===== [$(date)] END   $NAME (exit=$RC) =====" | tee -a "$MASTER_LOG"
  sleep 10
done

echo "===== ALL 4 QWEN RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
