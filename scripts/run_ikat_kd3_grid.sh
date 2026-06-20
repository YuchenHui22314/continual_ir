#!/bin/bash
# round3: round2 最优 32.3 (BiXSE1+KD2+anchor1+gneg2048+lr5e-6 @ep13) → 逼近 oracle 上界 36.3.
# 组合学习有效(BiXSE 精排 + KD 方向 + 防崩). 加容量(LoRA r32/64) + 更多 epoch(30) + 微调 KD/anchor.
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
run(){  # gpu name lora_r bixse kd anchor gk lr ep
  local gpu=$1 name=$2 r=$3 bx=$4 kd=$5 aw=$6 gk=$7 lr=$8 ep=$9
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
    --learning_rate $lr --no_lr_schedule --use_lora --lora_r $r \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1: 加容量 + 更多 epoch ==="
run 0 kd3_r32_kd2_a1_e30  32 1.0 2.0 1   2048 5e-6 30 &   # 容量 r32
run 1 kd3_r64_kd2_a1_e30  64 1.0 2.0 1   2048 5e-6 30 &   # 容量 r64
run 2 kd3_r16_kd2_a1_e30  16 1.0 2.0 1   2048 5e-6 30 &   # r16 基线更多 epoch
run 3 kd3_r32_kd3_a1_e30  32 1.0 3.0 1   2048 5e-6 30 &   # KD↑3 + r32
wait
echo "=== round 2: 微调 weight ==="
run 0 kd3_r32_kd2_a05_e30 32 1.0 2.0 0.5 2048 5e-6 30 &   # anchor↓0.5
run 1 kd3_r32_kd2_a2_e30  32 1.0 2.0 2   2048 5e-6 30 &   # anchor↑2
run 2 kd3_r32_kd15_a1_e30 32 1.0 1.5 1   2048 5e-6 30 &   # KD↓1.5
run 3 kd3_r64_kd2_a1_g4096_e30 64 1.0 2.0 1 4096 5e-6 30 & # r64 + gneg↑4096
wait
echo "ALL DONE"
