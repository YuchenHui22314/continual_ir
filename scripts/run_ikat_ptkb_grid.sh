#!/bin/bash
# full-ptkb 下一版:在 FULL PTKB(非 oracle rel)上训 graded 微调,看能否学会 personalization
# (从一堆 profile 里挑相关) 涨过 conv ZS 的 full-ptkb 25.3。conv init + 软 anchor + 全语料负例 + 低 lr。
set -u
D=/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded
CONV=/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_qwen_nosched/checkpoint-step-94
PTKB=$D/ikat_graded_ptkb.jsonl
ANCHOR=$D/anchor_query_emb_conv_ptkb.pt
GNEG=$D/global_neg_pool_50k.pt
OUT=/part/01/Tmp/yuchen/continual_ir
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
TRAIN=/data/rech/huiyuche/continual_ir/src/train_qwen_cl.py
LOGD=/data/rech/huiyuche/TREC_iKAT_2024/logs
run(){  # gpu name aw gk lr epochs
  local gpu=$1 name=$2 aw=$3 gk=$4 lr=$5 ep=$6
  local af=""; [ "$aw" != "0" ] && af="--ikat_anchor_emb_file $ANCHOR"
  local gf=""; [ "$gk" != "0" ] && gf="--ikat_global_neg_file $GNEG"
  rm -rf "$OUT/$name" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY $TRAIN --dataset_type ikat_graded --loss_type bce --pretrained_encoder_path "$CONV" \
    --ikat_manifest_file $PTKB --ikat_doc_embedding_file $D/ikat_graded_doc_embeddings.pt \
    --ikat_split train --ikat_grade0_cap all \
    --ikat_anchor_weight $aw $af --ikat_global_neg_k $gk $gf \
    --model_output_path "$OUT/$name" --num_train_epochs $ep --per_gpu_train_batch_size 16 \
    --learning_rate $lr --no_lr_schedule --use_lora --lora_r 16 \
    --use_bf16 --use_flash_attention --bf16_fp32_master --gradient_checkpointing \
    --gpu_resident_doc_table --n_gpu 1 > "$LOGD/run_$name.log" 2>&1
  echo "done $name"
}
echo "=== round 1 ==="
run 0 pk_a0_g512_lr5e6     0   512  5e-6 15 &
run 1 pk_a05_g512_lr5e6    0.5 512  5e-6 15 &
run 2 pk_a1_g512_lr5e6     1   512  5e-6 15 &
run 3 pk_a1_g512_lr1e5     1   512  1e-5 15 &
wait
echo "=== round 2 ==="
run 0 pk_a2_g512_lr5e6     2   512  5e-6 15 &
run 1 pk_a1_g2048_lr5e6    1   2048 5e-6 15 &
run 2 pk_a05_g2048_lr1e5   0.5 2048 1e-5 15 &
run 3 pk_a1_g512_lr1e5_e20 1   512  1e-5 20 &
wait
echo "ALL DONE"
