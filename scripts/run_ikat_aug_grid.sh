#!/bin/bash
# round4 数据增强 grid: aug manifest(1120 train = 280 x 3 噪声 PTKB 子集视图). 数据瓶颈的根本解.
# listwise KD 已验证没帮 -> 去掉. 纯最优配方 bx1 kd2 a1 g2048 lr5e-6; 扫 lora_r × epoch.
# 注意 aug 1120 turn -> 1 epoch ≈ 280-turn 的 4 epoch, 所以 epoch 数压低(8-20).
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
AUG_MAN=$D/ikat_graded_ptkb_aug.jsonl
AUG_KD=$D/ikat_oracle_teacher_emb_aug.pt
AUG_ANCHOR=$D/anchor_query_emb_conv_ptkb_aug.pt
GNEG=$D/global_neg_pool_50k.pt
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
run(){  # gpu name lora_r epoch
  local gpu=$1 name=$2 r=$3 ep=$4
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $AUG_MAN --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_bixse_weight 1.0 --ikat_kd_weight 2.0 --ikat_kd_emb_file $AUG_KD \
    --ikat_anchor_weight 1.0 --ikat_anchor_emb_file $AUG_ANCHOR \
    --ikat_global_neg_file $GNEG --ikat_global_neg_k 2048 \
    --model_output_path "$OUT/$name" --num_train_epochs $ep --per_gpu_train_batch_size 16 \
    --learning_rate 5e-6 --no_lr_schedule --use_lora --lora_r $r \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1: lora_r × epoch ==="
run 0 aug_r16_10ep 16 10 &
run 1 aug_r16_15ep 16 15 &
run 2 aug_r32_10ep 32 10 &
run 3 aug_r32_15ep 32 15 &
wait
echo "=== round 2: 更大容量 / 更多 epoch ==="
run 0 aug_r64_15ep 64 15 &
run 1 aug_r16_20ep 16 20 &
run 2 aug_r32_20ep 32 20 &
run 3 aug_r64_10ep 64 10 &
wait
echo "ALL DONE"
