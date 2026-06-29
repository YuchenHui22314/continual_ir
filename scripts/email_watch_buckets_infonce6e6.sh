#!/bin/bash
# Session-independent email watcher for the 13 InfoNCE-6e6 turn-bucket runs.
# Waits for the master log's ALL-DONE marker, then emails per-bucket END status +
# ckpt / grad-dump counts. Run detached: nohup bash <this> &
ML=$(ls -t /data/rech/huiyuche/TREC_iKAT_2024/logs/qwen_turn_buckets_infonce6e6_master_*.log | head -1)
ALERT_LOG=/data/rech/huiyuche/TREC_iKAT_2024/logs/buckets_infonce6e6_alert.log

until grep -q "InfoNCE-6e6 TURN-BUCKET RUNS DONE" "$ML" 2>/dev/null; do sleep 120; done

{
  echo "13 InfoNCE-6e6 turn-bucket runs finished at $(date)"
  echo "host: $(hostname)"
  echo
  echo "Per-bucket END status (from master log $ML):"
  grep -E "END .*exit=" "$ML"
  echo
  echo "ckpt + grad-dump per bucket:"
  for b in turn_1 turn_2 turn_3 turn_4 turn_5 turn_6 turn_7 turn_8 turn_9 turn_10 turn_11_12 turn_13_14 turn_15plus; do
    D=/data/rech/huiyuche/huggingface/continual_ir/bucket_infonce6e6_$b
    echo "  $b: $(ls -d $D/checkpoint-step-* 2>/dev/null | wc -l)/10 ckpt, $(ls -d $D/grad_stats-step-* 2>/dev/null | wc -l)/3 grad-dump"
  done
  echo
  echo "Next: Claude runs eval_bucket_runs_per_ckpt.py (13x13 train-bucket x eval-turn transfer matrix +"
  echo "MSMARCO forgetting) and the Fisher/EWC/sign-coherence analysis, then redraws the bucket figures."
} | mail -s "[continual_ir] 13 turn-bucket (InfoNCE 6e-6) DONE" huiyuche@iro.umontreal.ca

echo "bucket alert email sent at $(date)" >> "$ALERT_LOG"
