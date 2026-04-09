#!/bin/bash
# train_continually_ddp_cl.sh
#
# Launch curriculum learning training for TopiOCQA ANCE fine-tuning.
# Aligned to baseline hyperparams (wandb run 1ppvc6dl):
#   4 GPU × per_gpu=120 (total batch=480), 20 epochs, lr=1e-5, warmup=0.06
#   loss=ranking, negative=none, Flash Attention + BF16 + TF32
#
# CURRICULUM_TYPE controls the experiment:
#   none      → standard random sampling (ablation baseline)
#   easy2hard → curriculum  (sort easy→hard, pacing with root_2)
#   hard2easy → anti-curriculum (sort hard→easy)
#
# Usage:
#   bash train_continually_ddp_cl.sh            # runs with defaults below
#   CURRICULUM_TYPE=none bash train_continually_ddp_cl.sh   # override inline
#
# Logs: /data/rech/huiyuche/TREC_iKAT_2024/logs/
# GPU check: run `nvidia-smi` before launching.

set -e

# NCCL: disable P2P (required on PCIe-only nodes without NVLink, e.g. A5000).
# Without this, DDP init hangs indefinitely on the first dist.barrier().
export NCCL_P2P_DISABLE=1

# ---- Experiment config ---- (change these between runs)
CURRICULUM_TYPE=${CURRICULUM_TYPE:-easy2hard}   # none | easy2hard | hard2easy
CURRICULUM_C0=${CURRICULUM_C0:-0.2}             # initial fraction; update after analyze_topiocqa_turns.py
CURRICULUM_END_EPOCH=${CURRICULUM_END_EPOCH:-16} # epoch at which full dataset is exposed
PACING_FUNCTION=${PACING_FUNCTION:-root_2}       # root_2 | root_5 | linear | step | standard
N_GPU=${N_GPU:-4}                                # default: 4 A5000s; override to 2 for 2-GPU nodes
PER_GPU_BATCH=${PER_GPU_BATCH:-120}              # baseline: 120 (total 480 on 4 GPU)
NUM_EPOCHS=${NUM_EPOCHS:-20}
USE_DATA_PERCENT=${USE_DATA_PERCENT:-1.0}        # set 0.05 for smoke test

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/data/rech/huiyuche/TREC_iKAT_2024/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/cl_${CURRICULUM_TYPE}_c0${CURRICULUM_C0}_${TIMESTAMP}.log"
WANDB_NAME="topiocqa_cl_${CURRICULUM_TYPE}_c0_${CURRICULUM_C0}_$(date +%m%d)"
MODEL_OUTPUT="/data/rech/huiyuche/huggingface/continual_ir/topiocqa_cl_${CURRICULUM_TYPE}"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Curriculum Learning Training"
echo "  CURRICULUM_TYPE      = ${CURRICULUM_TYPE}"
echo "  CURRICULUM_C0        = ${CURRICULUM_C0}"
echo "  CURRICULUM_END_EPOCH = ${CURRICULUM_END_EPOCH}"
echo "  PACING_FUNCTION      = ${PACING_FUNCTION}"
echo "  N_GPU                = ${N_GPU}"
echo "  PER_GPU_BATCH        = ${PER_GPU_BATCH}"
echo "  NUM_EPOCHS           = ${NUM_EPOCHS}"
echo "  USE_DATA_PERCENT     = ${USE_DATA_PERCENT}"
echo "  MODEL_OUTPUT         = ${MODEL_OUTPUT}"
echo "  LOG_FILE             = ${LOG_FILE}"
echo "============================================================"

# Check GPUs before launching
nvidia-smi

torchrun \
    --nproc_per_node ${N_GPU} \
    "${SCRIPT_DIR}/train_continually_ddp_cl.py" \
    --n_gpu ${N_GPU} \
    --per_gpu_train_batch_size ${PER_GPU_BATCH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate 1e-5 \
    --weight_decay 0.00 \
    --warmup_ratio 0.06 \
    --loss_type ranking \
    --negative_type none \
    --use_data_percent ${USE_DATA_PERCENT} \
    \
    --curriculum_type ${CURRICULUM_TYPE} \
    --scoring_function turn_length \
    --pacing_function ${PACING_FUNCTION} \
    --curriculum_c0 ${CURRICULUM_C0} \
    --curriculum_end_epoch ${CURRICULUM_END_EPOCH} \
    \
    --use_flash_attention \
    --use_bf16 \
    --use_tf32 \
    \
    --activate_eval_while_training \
    --beir_datasets climate-fever msmarco \
    --eval_batch_size 64 \
    --use_gpu_faiss \
    --keep_faiss_on_gpu \
    \
    --activate_eval_topiocqa_while_training \
    \
    --training_data_file "/part/01/Tmp/yuchenhui/continual_ir_data/topiocqa_train_oracle.jsonl" \
    --pos_neg_embedding_file "/part/01/Tmp/yuchenhui/continual_ir_data/embeddings.pt" \
    --topiocqa_valid_file "/part/01/Tmp/yuchenhui/continual_ir_data/topiocqa_valid.jsonl" \
    --topiocqa_qrel_file "/part/01/Tmp/yuchenhui/continual_ir_data/topiocqa_qrel.trec" \
    \
    --model_output_path "${MODEL_OUTPUT}" \
    --save_to_wandb \
    --wandb_project "topiocqa-ance" \
    --wandb_name "${WANDB_NAME}" \
    \
    2>&1 | tee -a "${LOG_FILE}"
