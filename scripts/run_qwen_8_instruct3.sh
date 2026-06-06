#!/bin/bash
# Re-train ALL 8 Qwen3-Embedding-0.6B TopiOCQA fine-tunes under the v3
# conversational instruct template (User:/System: per-turn role markers +
# trailing "User's last question:" anchor). Mirrors the 2026-05-19
# instruct2 batch verbatim except for:
#   1. --template_version v3 + the v3 instruction text
#      (was v1, the legacy template).
#   2. max_query / max_response / max_concat lengths bumped to 32768
#      (Qwen3-Embedding native context window) so no TopiOCQA conversation
#      is truncated. TopiOCQA training conversations are well under 1k
#      tokens, so this is effectively "no truncation".
#   3. --beir_datasets msmarco (climate-fever dropped from in-training eval).
#   4. (REMOVED 2026-06-06 after epoch-2 OOM on first attempt:
#      --keep_faiss_on_gpu sounded like a free win but the cached
#      ~13 GB/GPU TopiOCQA index collides with FA2+grad-ckpt activation
#      memory when caps > 512. Per-epoch re-transfer costs ~30 s and is
#      preferable to a crashed run. See SKILL.md gotchas.)
#
# Naming kept separate from instruct2 so nothing collides:
#   - wandb project:   topiocqa-qwen-instruct-v3
#   - checkpoint dirs: OUT_BASE/instruct3_<name>
#   - wandb run names: instruct3_<name>

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/qwen_8runs_instruct3_master_${TS}.log

PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TRAIN_FILE=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
TOPIOCQA_EMB=/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged
TOPIOCQA_VALID=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl
TOPIOCQA_QREL=/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir

WANDB_PROJECT=topiocqa-qwen-instruct-v3
# v3 instruction text — matches utils.py:CONV_INSTRUCTION_V3 (which aliases V2).
CONV_INSTR="Given a conversation between a user and an AI assistant, retrieve passages that answer the user's last question."

# All 8 runs: baseline + 4 easy2hard CL + 3 hard2easy ACL (same as instruct2).
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

echo "===== 8 Qwen INSTRUCT3 (v3 template) runs kickoff at $(date) =====" | tee -a "$MASTER_LOG"
echo "wandb project: $WANDB_PROJECT" | tee -a "$MASTER_LOG"
echo "conv_instruction: $CONV_INSTR" | tee -a "$MASTER_LOG"
echo "template_version: v3" | tee -a "$MASTER_LOG"
echo "in-training BEIR datasets: msmarco (climate-fever dropped)" | tee -a "$MASTER_LOG"
echo "FAISS index NOT cached on GPU (per-epoch re-transfer; collides with training activations under caps>512)" | tee -a "$MASTER_LOG"
echo "truncation caps: 32768 / 32768 / 32768 (Qwen3-Embedding native ctx; effectively no truncation for TopiOCQA)" | tee -a "$MASTER_LOG"

for entry in "${RUNS[@]}"; do
  IFS='|' read -r NAME CTYPE PACE <<< "$entry"
  RUN_NAME="instruct3_${NAME}"
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
    --template_version v3 \
    --max_query_length 32768 \
    --max_response_length 32768 \
    --max_concat_length 32768 \
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
    --beir_datasets msmarco \
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

echo "===== ALL 8 QWEN INSTRUCT3 RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
