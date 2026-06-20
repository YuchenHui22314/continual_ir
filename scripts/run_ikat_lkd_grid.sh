#!/bin/bash
# round4 listwise KD: round3 诊断=数据瓶颈(r16=32.4 > r32 > r64). 在 r16 最优配方上加 listwise KD
# (KL: student 对候选池打分分布 -> teacher 打分分布, 蒸 teacher 的相对排序, 直接优化 ranking 指标).
# 固定 bixse1 + kd(point) + anchor1 + gneg2048 + lr5e-6 + r16 + 20ep; 扫 list_weight/temp/point-kd.
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
run(){  # gpu name list_w list_t point_kd anchor
  local gpu=$1 name=$2 lw=$3 lt=$4 kd=$5 aw=$6
  local af=""; [ "$aw" != "0" ] && af="--ikat_anchor_emb_file $ANCHOR"
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $D/ikat_graded_ptkb.jsonl --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_bixse_weight 1.0 --ikat_kd_weight $kd --ikat_kd_emb_file $KD \
    --ikat_kd_list_weight $lw --ikat_kd_list_temp $lt \
    --ikat_anchor_weight $aw $af --ikat_global_neg_file $GNEG --ikat_global_neg_k 2048 \
    --model_output_path "$OUT/$name" --num_train_epochs 20 --per_gpu_train_batch_size 16 \
    --learning_rate 5e-6 --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1: list_weight × temp ==="
run 0 lkd_l05_t05 0.5 0.05 2.0 1 &
run 1 lkd_l1_t05  1.0 0.05 2.0 1 &
run 2 lkd_l2_t05  2.0 0.05 2.0 1 &
run 3 lkd_l1_t02  1.0 0.02 2.0 1 &
wait
echo "=== round 2: temp / point-kd / anchor ==="
run 0 lkd_l1_t1   1.0 0.10 2.0 1 &
run 1 lkd_l2_kd1  2.0 0.05 1.0 1 &
run 2 lkd_l3_kd0  3.0 0.05 0.0 1 &
run 3 lkd_l1_a05  1.0 0.05 2.0 0.5 &
wait
echo "ALL DONE"
