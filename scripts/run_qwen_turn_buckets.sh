#!/bin/bash
# Train 13 Qwen3-Embedding-0.6B models, one per TopiOCQA turn-length bucket
# (turn_1 .. turn_10, turn_11_12, turn_13_14, turn_15plus; each 3,054 samples,
# built by preprocess/data/build_topiocqa_turn_buckets.py), to isolate the
# effect of conversation LENGTH on catastrophic forgetting.
#
# Protocol mirrors the instruct3 batch (v3 template, 32768 caps, lr 1e-5,
# global batch 480, no LR schedule, curriculum none) EXCEPT:
#   - step-driven: --total_train_steps 470 --save_every_steps 47
#     (470 steps = the update budget of 5 main-run epochs; ckpt every 47 steps
#      -> 10 save points per run, model-only except the final one)
#   - NO in-training eval (no --activate_eval_* flags): octal40 is due back,
#     so training is corpus-free and all evals run offline later
#     (preprocess/eval/eval_bucket_runs_per_ckpt.py) on another machine.
#   - --record_grad_stats: per-step per-tensor grad norms + whole-run
#     per-scalar sum_g / sum_g2 accumulators (parameter-update analysis,
#     cf. arXiv 2505.11711).
#
# Naming:
#   - wandb project:   topiocqa-qwen-turn-buckets
#   - checkpoint dirs: OUT_BASE/bucket_qwen_<bucket>
#   - run names:       bucket_qwen_<bucket>

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../src/train_qwen_cl.py"
TORCHRUN=/data/rech/huiyuche/envs/trec_ikat/bin/torchrun

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
MASTER_LOG=${LOG_DIR}/qwen_turn_buckets_master_${TS}.log

PRETRAIN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
BUCKET_DIR=/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/turn_buckets
POS_NEG=/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
OUT_BASE=/data/rech/huiyuche/huggingface/continual_ir

WANDB_PROJECT=topiocqa-qwen-turn-buckets
# v3 instruction text — matches utils.py:CONV_INSTRUCTION_V3.
CONV_INSTR="Given a conversation between a user and an AI assistant, retrieve passages that answer the user's last question."

BUCKETS=(
  turn_1 turn_2 turn_3 turn_4 turn_5
  turn_6 turn_7 turn_8 turn_9 turn_10
  turn_11_12 turn_13_14 turn_15plus
)

echo "===== 13 turn-bucket runs kickoff at $(date) =====" | tee -a "$MASTER_LOG"
echo "wandb project: $WANDB_PROJECT | template v3 | caps 32768 | 470 steps, ckpt every 47" | tee -a "$MASTER_LOG"
echo "NO in-training eval (offline eval later); --record_grad_stats ON" | tee -a "$MASTER_LOG"

for BUCKET in "${BUCKETS[@]}"; do
  RUN_NAME="bucket_qwen_${BUCKET}"
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
    --record_grad_stats \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 120 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.06 \
    --no_lr_schedule \
    --loss_type ranking \
    --negative_type none \
    --curriculum_type none \
    --pacing_function root_2 \
    --curriculum_c0 0.2 \
    --curriculum_end_epoch 16 \
    --embed_dim 1024 \
    --encoder_type qwen3 \
    --use_flash_attention --use_bf16 --gradient_checkpointing \
    --n_gpu 4 \
    --save_to_wandb \
    --wandb_name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$RUN_LOG"

  RC=${PIPESTATUS[0]}
  echo "===== [$(date)] END   $RUN_NAME (exit=$RC) =====" | tee -a "$MASTER_LOG"
  sleep 5
done

echo "===== ALL 13 TURN-BUCKET RUNS DONE at $(date) =====" | tee -a "$MASTER_LOG"
