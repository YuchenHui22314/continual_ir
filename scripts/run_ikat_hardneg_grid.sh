#!/bin/bash
# round5 hard-neg: 用 oracle 检索的 hard-neg pool 替代随机 gneg(--ikat_global_neg_file 指向它, 零代码改动).
# 对症 round4 诊断: student 训在随机 easy negs -> 学不会压 false positives. 最优配方 bx1 kd2 a1 r16 lr5e-6.
# 扫 hard-neg k / epoch; 带随机 gneg 对照(应 ≈ 32.4, 确认是 hard-neg 本身带来的增益).
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
KD=$D/ikat_oracle_teacher_emb.pt
HN=$D/ikat_oracle_hardneg_pool.pt
GNEG=$D/global_neg_pool_50k.pt
ANCHOR=$D/anchor_query_emb_conv_ptkb.pt
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
run(){  # gpu name negfile k epoch
  local gpu=$1 name=$2 nf=$3 k=$4 ep=$5
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $D/ikat_graded_ptkb.jsonl --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_bixse_weight 1.0 --ikat_kd_weight 2.0 --ikat_kd_emb_file $KD \
    --ikat_anchor_weight 1.0 --ikat_anchor_emb_file $ANCHOR \
    --ikat_global_neg_file $nf --ikat_global_neg_k $k \
    --model_output_path "$OUT/$name" --num_train_epochs $ep --per_gpu_train_batch_size 16 \
    --learning_rate 5e-6 --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1: hard-neg pool, 扫 k ==="
run 0 hn_k512_20ep  $HN 512  20 &
run 1 hn_k1024_20ep $HN 1024 20 &
run 2 hn_k2048_20ep $HN 2048 20 &
run 3 hn_k2048_30ep $HN 2048 30 &
wait
echo "=== round 2: 更多 epoch / 随机对照 ==="
run 0 hn_k1024_30ep $HN 1024 30 &
run 1 hn_k512_30ep  $HN 512  30 &
run 2 rand_k2048_20ep $GNEG 2048 20 &
run 3 hn_k4096_20ep $HN 4096 20 &
wait
echo "ALL DONE"
