#!/bin/bash
# 1+3+4 对照网格:anchor 正则 × 全语料负例 × lr。conv-qwen3 init + rel_ptkb + LoRA r16 + cap all.
# 存所有 epoch(无 save_best_only、无 in-training eval)→ 训完一趟全库 eval 选 (config,epoch)。
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
REL=$D/ikat_graded_rel_ptkb.jsonl
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs

run(){  # gpu name anchor_w global_k lr
  local gpu=$1 name=$2 aw=$3 gk=$4 lr=$5
  local af=""; [ "$aw" != "0" ] && af="--ikat_anchor_emb_file $D/anchor_query_emb_conv_rel.pt"
  local gf=""; [ "$gk" != "0" ] && gf="--ikat_global_neg_file $D/global_neg_pool_50k.pt"
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $REL --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_anchor_weight $aw $af --ikat_global_neg_k $gk $gf \
    --model_output_path "$OUT/$name" --num_train_epochs 8 --per_gpu_train_batch_size 16 \
    --learning_rate $lr --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name (gpu$gpu aw=$aw gk=$gk lr=$lr)"
}

echo "=== round 1 ==="
run 0 m134_a0_g0_lr1e5    0   0    1e-5 &
run 1 m134_a0_g512_lr1e5  0   512  1e-5 &
run 2 m134_a1_g0_lr1e5    1   0    1e-5 &
run 3 m134_a1_g512_lr1e5  1   512  1e-5 &
wait
echo "=== round 2 ==="
run 0 m134_a2_g512_lr1e5  2   512  1e-5 &
run 1 m134_a1_g512_lr5e6  1   512  5e-6 &
run 2 m134_a1_g2048_lr1e5 1   2048 1e-5 &
run 3 m134_a05_g512_lr1e5 0.5 512  1e-5 &
wait
echo "ALL DONE"
