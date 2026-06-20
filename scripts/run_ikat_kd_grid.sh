#!/bin/bash
# ConvDR-KD 实验: KD(student 学 oracle-rewrite teacher emb) + graded BiXSE + 可选 anchor + 全语料负例.
# 目标: 涨过 conv ZS (full-ptkb 25.3 / rel-oracle 26.0). KD teacher = base-qwen3 encode oracle (固定).
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
BASE=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
KD=$D/ikat_oracle_teacher_emb.pt
GNEG=$D/global_neg_pool_50k.pt
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
run(){  # gpu name init form kd anchor gk lr ep
  local gpu=$1 name=$2 init=$3 form=$4 kd=$5 aw=$6 gk=$7 lr=$8 ep=$9
  local enc=$CONV; [ "$init" = base ] && enc=$BASE
  local man=$D/ikat_graded_ptkb.jsonl; [ "$form" = rel ] && man=$D/ikat_graded_rel_ptkb.jsonl
  local af=""
  if [ "$aw" != "0" ]; then
    local aff=$D/anchor_query_emb_conv_ptkb.pt; [ "$form" = rel ] && aff=$D/anchor_query_emb_conv_rel.pt
    af="--ikat_anchor_emb_file $aff"
  fi
  local gf=""; [ "$gk" != "0" ] && gf="--ikat_global_neg_file $GNEG"
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$enc" \
    --ikat_manifest_file $man --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_kd_weight $kd --ikat_kd_emb_file $KD \
    --ikat_anchor_weight $aw $af --ikat_global_neg_k $gk $gf \
    --model_output_path "$OUT/$name" --num_train_epochs $ep --per_gpu_train_batch_size 16 \
    --learning_rate $lr --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1 (conv init, full ptkb) ==="
run 0 kd_fpk_kd05_a0    conv full 0.5 0   512 1e-5 15 &
run 1 kd_fpk_kd05_a01   conv full 0.5 0.1 512 1e-5 15 &
run 2 kd_fpk_kd02_a0    conv full 0.2 0   512 1e-5 15 &
run 3 kd_fpk_kd05_lr5e6 conv full 0.5 0   512 5e-6 15 &
wait
echo "=== round 2 (variations) ==="
run 0 kd_fpk_kd1_a0     conv full 1.0 0   512 1e-5 15 &
run 1 kd_rpk_kd05_a0    conv rel  0.5 0   512 1e-5 15 &
run 2 kd_fpk_kd05_g0    conv full 0.5 0   0   1e-5 15 &
run 3 kd_bpk_kd05_a0    base full 0.5 0   512 1e-5 15 &
wait
echo "ALL DONE"
