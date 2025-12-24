#!/usr/bin/env bash
set -e # exit on error

############################
# Conda environment
############################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/rech/huiyuche/envs/trec_ikat

############################
# Project root
############################
cd /data/rech/huiyuche/continual_ir/src

############################
# Environment
############################
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

############################
# DDP config
############################
NUM_GPUS=4
MASTER_PORT=29501   # you can change this port if needed.

############################
# Paths
############################
ENCODER_PATH="/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234"
TRAIN_FILE="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl"
OUTPUT_DIR="/data/rech/huiyuche/huggingface/continual_ir/topiocqa"
POS_NEG_EMB="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.pt"

LOG_PATH="/data/rech/huiyuche/TREC_iKAT_2024/logs/train_log_topiocqa.txt"

############################
# Launch
############################
torchrun \
  --nproc_per_node ${NUM_GPUS} \
  --master_port ${MASTER_PORT} \
  train_topiocqa_ddp.py \
  --n_gpu ${NUM_GPUS} \
  --pretrained_encoder_path "${ENCODER_PATH}" \
  --training_data_file "${TRAIN_FILE}" \
  --model_output_path "${OUTPUT_DIR}" \
  --pos_neg_embedding_file "${POS_NEG_EMB}" \
  --loss_type ranking \
  --num_train_epochs 20 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 1e-5 \
  --weight_decay 0.00 \
  --warmup_ratio 0.06 \
  --max_query_length 64 \
  --max_doc_length 512 \
  --max_concat_length 512 \
  --log_print_ratio 0.1 \
  --use_data_percent 0.01 &>> $LOG_PATH


# --resume_from_checkpoint