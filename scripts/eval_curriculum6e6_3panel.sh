#!/bin/bash
# Eval the 7 InfoNCE-constlr-6e6 curriculum/ACL variants for the fig:infonce-style
# three-panel: TopiOCQA + MS MARCO + QReCC per-epoch (7 runs x 20 ckpts each).
# Sequential — all three share the 4-GPU faiss index. Resume-safe per (run, step).
# The 6e-6 random-order baseline (instruct3fp32infonce_qwen_constlr_lr6e6) is already
# evaluated; these 7 + that baseline = the full 8-row InfoNCE-recipe curriculum table.
set -u
cd /data/rech/huiyuche/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TS=$(date +%Y%m%d_%H%M%S)

RUNS_QRECC="instruct3fp32infonce_constlr_lr6e6_cl_step \
instruct3fp32infonce_constlr_lr6e6_cl_step_excl \
instruct3fp32infonce_constlr_lr6e6_cl_step_excl_2_full \
instruct3fp32infonce_constlr_lr6e6_cl_root2 \
instruct3fp32infonce_constlr_lr6e6_acl_root2 \
instruct3fp32infonce_constlr_lr6e6_acl_step \
instruct3fp32infonce_constlr_lr6e6_acl_step_excl"

echo "===== curriculum6e6 eval 3-panel kickoff $(date) ====="

echo "--- [1/3] TopiOCQA (7 runs x 20 ckpts) ---"
$PY preprocess/eval/eval_curriculum6e6_qwen_topiocqa_per_epoch.py \
  2>&1 | tee "$LOGD/eval_curriculum6e6_topiocqa_${TS}.log"

echo "--- [2/3] MS MARCO (7 runs x 20 ckpts) ---"
$PY preprocess/eval/eval_curriculum6e6_qwen_msmarco_per_epoch.py \
  2>&1 | tee "$LOGD/eval_curriculum6e6_msmarco_${TS}.log"

echo "--- [3/3] QReCC (7 runs x 20 ckpts) ---"
$PY preprocess/eval/eval_qrecc_per_epoch.py --setting qwen_instr --template_version v3 \
  --runs $RUNS_QRECC \
  --results_out figures/qrecc_per_epoch_curriculum6e6.json \
  2>&1 | tee "$LOGD/eval_curriculum6e6_qrecc_${TS}.log"

echo "===== curriculum6e6 eval 3-panel ALL DONE $(date) ====="
