#!/bin/bash
# round2 KD: oracle 上界=36.3 >> round1 KD best=27.1 → student 远没学到 teacher, KD 还有 9 个点空间.
# 大力加强: 纯 KD(bixse=0, ConvDR 原版 KD 主导) / 强 KD(2~3) / 多 epoch(30, KD 慢慢把 conversation 拉到 oracle 表示).
# KD 拉向 base-qwen3 流形的 oracle teacher, 本身防崩, 纯 KD 可不要 anchor/gneg.
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
KD=$D/ikat_oracle_teacher_emb.pt
GNEG=$D/global_neg_pool_50k.pt
ANCHOR=$D/anchor_query_emb_conv_ptkb.pt
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
run(){  # gpu name bixse kd anchor gk lr ep
  local gpu=$1 name=$2 bx=$3 kd=$4 aw=$5 gk=$6 lr=$7 ep=$8
  local af=""; [ "$aw" != "0" ] && af="--ikat_anchor_emb_file $ANCHOR"
  local gf=""; [ "$gk" != "0" ] && gf="--ikat_global_neg_file $GNEG"
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $D/ikat_graded_ptkb.jsonl --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_bixse_weight $bx --ikat_kd_weight $kd --ikat_kd_emb_file $KD \
    --ikat_anchor_weight $aw $af --ikat_global_neg_k $gk $gf \
    --model_output_path "$OUT/$name" --num_train_epochs $ep --per_gpu_train_batch_size 16 \
    --learning_rate $lr --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1: 纯 KD / 弱 BiXSE + 强 KD ==="
run 0 kd2_pure_kd1_e30       0   1.0 0 0    1e-5 30 &   # 纯 KD 30ep (能学 teacher 到多高?)
run 1 kd2_pure_kd1_lr5e6_e30 0   1.0 0 0    5e-6 30 &   # 纯 KD 慢学
run 2 kd2_bx01_kd2_e30       0.1 2.0 0 512  1e-5 30 &   # 弱 BiXSE + 强 KD + gneg
run 3 kd2_bx03_kd1_e30       0.3 1.0 0 512  1e-5 30 &   # 中 BiXSE + KD + gneg
wait
echo "=== round 2: 纯强 KD / 强 KD + 强防崩 ==="
run 0 kd2_pure_kd2_e30       0   2.0 0 0    1e-5 30 &   # 纯强 KD
run 1 kd2_bx1_kd3_e20        1.0 3.0 0 512  1e-5 20 &   # 正常 BiXSE + 极强 KD
run 2 kd2_bx1_kd2_a1_g2048_e20 1.0 2.0 1 2048 5e-6 20 & # 强 KD + 强防崩(anchor1+gneg2048+慢)
run 3 kd2_pure_kd1_e15       0   1.0 0 0    1e-5 15 &   # 纯 KD 短(对照)
wait
echo "ALL DONE"
