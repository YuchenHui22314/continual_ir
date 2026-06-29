#!/bin/bash
# Session-independent email watcher for the 7-curriculum (6e-6 const) batch.
# Waits for the master log's ALL-DONE marker, then emails per-run END status +
# final train Loss + ckpt counts. Run detached: nohup bash <this> &
ML=$(ls -t /data/rech/huiyuche/TREC_iKAT_2024/logs/constlr_curriculum_master_*.log | head -1)
ALERT_LOG=/data/rech/huiyuche/TREC_iKAT_2024/logs/curriculum6e6_alert.log

until grep -q "ALL 7 CONSTLR CURRICULUM RUNS DONE" "$ML" 2>/dev/null; do sleep 120; done

{
  echo "7 InfoNCE-constlr 6e-6 curriculum runs finished at $(date)"
  echo "host: $(hostname)"
  echo
  echo "Per-run END status (from master log $ML):"
  grep -E "END .*exit=" "$ML"
  echo
  echo "Final train Loss + ckpt count per run:"
  for n in cl_step cl_step_excl cl_step_excl_2_full cl_root2 acl_root2 acl_step acl_step_excl; do
    rl=$(ls -t /data/rech/huiyuche/TREC_iKAT_2024/logs/run_instruct3fp32infonce_constlr_lr6e6_${n}_*.log 2>/dev/null | head -1)
    loss=$(tail -c 4000 "$rl" 2>/dev/null | tr '\r' '\n' | grep -oE 'Loss=[0-9.]+' | tail -1)
    ck=$(ls -d /data/rech/huiyuche/huggingface/continual_ir/instruct3fp32infonce_constlr_lr6e6_${n}/checkpoint-step-* 2>/dev/null | wc -l)
    echo "  ${n}: ${loss:-NA}  (${ck}/20 ckpts)"
  done
  echo
  echo "ckpts under: /data/rech/huiyuche/huggingface/continual_ir/instruct3fp32infonce_constlr_lr6e6_*"
  echo "Next: Claude offline-evals the 3-panel (TopiOCQA / MS MARCO / QReCC) and assembles"
  echo "the 8-row InfoNCE-recipe curriculum table (6e-6 random baseline + these 7 variants)."
} | mail -s "[continual_ir] 7 curriculum (6e-6 const) DONE" huiyuche@iro.umontreal.ca

echo "alert email sent at $(date)" >> "$ALERT_LOG"
