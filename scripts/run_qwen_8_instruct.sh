#!/bin/bash
# Re-run ALL 8 Qwen3-Embedding-0.6B TopiOCQA fine-tunes WITH the official
# Qwen3 instruction-text query path (single trailing <|endoftext|>, last-token
# pooled; train==eval byte-identical). Everything else is identical to the
# original 8-run config (20 epochs, global batch 480, lr 1e-5, c0 0.2,
# curriculum_end_epoch 16, ...).
#
# Kept separate from the old runs so nothing gets mixed up:
#   - NEW wandb project:    topiocqa-qwen-instruct   (old: topiocqa-qwen)
#   - NEW checkpoint dirs:  OUT_BASE/instruct_<name>  (old: OUT_BASE/<name>)
#   - NEW wandb run names:  instruct_<name>
# QReCC eval is intentionally OFF during training (flag not passed).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
# Use the project conda env's torchrun (bare `torchrun` resolves to a python
# without faiss/torch -> ModuleNotFoundError: No module named 'faiss').
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/qwen_8runs_instruct_master_${TS}.log

# Shared args
PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TRAIN_FILE=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
TOPIOCQA_EMB=/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged
TOPIOCQA_VALID=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl
TOPIOCQA_QREL=/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir

WANDB_PROJECT=topiocqa-qwen-instruct-v2
CONV_INSTR="Given a conversation, retrieve relevant passages that help answer the user's latest question"

# All 8 runs: baseline + 4 easy2hard CL + 3 hard2easy ACL
RUNS=(
  "qwen_nosched|none|root_2"
  "qwen_cl_step|easy2hard|step"
  "qwen_cl_step_excl|easy2hard|step_exclusive"
  "qwen_cl_step_excl_2_full|easy2hard|step_exclusive_2_full"
  "qwen_cl_root2|easy2hard|root_2"
  "qwen_acl_root2|hard2easy|root_2"
  "qwen_acl_step|hard2easy|step"
  "qwen_acl_step_excl|hard2easy|step_exclusive"
)

echo "===== 8 Qwen INSTRUCT runs kickoff at $(date) =====" | tee -a "$MASTER_LOG"
echo "wandb project: $WANDB_PROJECT | conv_instruction: $CONV_INSTR" | tee -a "$MASTER_LOG"

for entry in "${RUNS[@]}"; do
  IFS='|' read -r NAME CTYPE PACE <<< "$entry"
  RUN_NAME="instruct2_${NAME}"
  RUN_LOG=${LOG_DIR}/run_${RUN_NAME}_${TS}.log
  echo "===== [$(date)] START $RUN_NAME (curriculum=$CTYPE, pacing=$PACE) =====" | tee -a "$MASTER_LOG"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TORCHRUN" --nproc_per_node=4 "$TRAIN_SCRIPT" \
    --pretrained_encoder_path "$PRETRAIN" \
    --training_data_file "$TRAIN_FILE" \
    --pos_neg_embedding_file "$POS_NEG" \
    --model_output_path "${OUT_BASE}/${RUN_NAME}" \
    --topiocqa_embedding_dir "$TOPIOCQA_EMB" \
    --topiocqa_valid_file "$TOPIOCQA_VALID" \
    --topiocqa_qrel_file "$TOPIOCQA_QREL" \
    --conv_instruction "$CONV_INSTR" \
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
    --wandb_name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$RUN_LOG"

  RC=${PIPESTATUS[0]}
  echo "===== [$(date)] END   $RUN_NAME (exit=$RC) =====" | tee -a "$MASTER_LOG"
  sleep 10
done

echo "===== ALL 8 QWEN INSTRUCT RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
