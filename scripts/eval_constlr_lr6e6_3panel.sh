#!/bin/bash
# Eval the 6e-6 constant-LR InfoNCE 4x120 run (instruct3fp32infonce_qwen_constlr_lr6e6)
# for the fig:infonce three-panel: TopiOCQA + MS MARCO + QReCC per-epoch (20 ckpts each).
# Sequential — all three share the 4-GPU faiss index. Resume-safe per (run, step).
set -u
cd /data/rech/huiyuche/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TS=$(date +%Y%m%d_%H%M%S)

echo "===== constlr_lr6e6 eval 3-panel kickoff $(date) ====="

echo "--- [1/3] TopiOCQA per-epoch ---"
$PY preprocess/eval/eval_constlr_lr6e6_qwen_topiocqa_per_epoch.py \
  2>&1 | tee "$LOGD/eval_constlr_lr6e6_topiocqa_${TS}.log"

echo "--- [2/3] MS MARCO per-epoch ---"
$PY preprocess/eval/eval_constlr_lr6e6_qwen_msmarco_per_epoch.py \
  2>&1 | tee "$LOGD/eval_constlr_lr6e6_msmarco_${TS}.log"

echo "--- [3/3] QReCC per-epoch ---"
$PY preprocess/eval/eval_qrecc_per_epoch.py --setting qwen_instr --template_version v3 \
  --runs instruct3fp32infonce_qwen_constlr_lr6e6 \
  --results_out figures/qrecc_per_epoch_infonce_v3.json \
  2>&1 | tee "$LOGD/eval_constlr_lr6e6_qrecc_${TS}.log"

echo "===== constlr_lr6e6 eval 3-panel ALL DONE $(date) ====="
