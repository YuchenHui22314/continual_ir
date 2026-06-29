#!/bin/bash
# Session-independent email watcher for the 7-curriculum (6e-6 const) EVAL.
# Waits until all 3 per-epoch JSONs carry step-1880 for all 7 runs, then emails the
# 8-row final NDCG@10 table (6e-6 random baseline + 7 curriculum variants).
F=/data/rech/huiyuche/continual_ir/figures
PY=/data/rech/huiyuche/envs/trec_ikat/bin/python
ALERT_LOG=/data/rech/huiyuche/TREC_iKAT_2024/logs/curriculum6e6_eval_alert.log
RUNS="cl_step cl_step_excl cl_step_excl_2_full cl_root2 acl_root2 acl_step acl_step_excl"

chk() { $PY -c "import json,sys; d=json.load(open('$1')); p='instruct3fp32infonce_constlr_lr6e6_'; sys.exit(0 if all('1880' in d.get(p+r,{}) for r in '$RUNS'.split()) else 1)" 2>/dev/null; }
until chk $F/curriculum6e6_topiocqa_per_epoch.json && chk $F/curriculum6e6_msmarco_per_epoch.json && chk $F/qrecc_per_epoch_curriculum6e6.json; do sleep 120; done

TABLE=$($PY <<'PYEOF'
import json
F="/data/rech/huiyuche/continual_ir/figures"
runs=['cl_step','cl_step_excl','cl_step_excl_2_full','cl_root2','acl_root2','acl_step','acl_step_excl']
p='instruct3fp32infonce_constlr_lr6e6_'
files={'t':f'{F}/curriculum6e6_topiocqa_per_epoch.json','m':f'{F}/curriculum6e6_msmarco_per_epoch.json','q':f'{F}/qrecc_per_epoch_curriculum6e6.json'}
base={'t':f'{F}/constlr_lr6e6_topiocqa_per_epoch.json','m':f'{F}/constlr_lr6e6_msmarco_per_epoch.json','q':f'{F}/qrecc_per_epoch_infonce_v3.json'}
def fin(f,r):
    try: return json.load(open(f)).get(r,{}).get('1880')
    except: return None
def row(name,ft,fm,fq,r):
    g=lambda v:(v or 0)*100
    return f"{name:<26}{g(fin(ft,r)):>9.1f}{g(fin(fm,r)):>9.1f}{g(fin(fq,r)):>9.1f}"
print(f"{'row':<26}{'TopiOCQA':>9}{'MSMARCO':>9}{'QReCC':>9}")
print(row('random (baseline)',base['t'],base['m'],base['q'],'instruct3fp32infonce_qwen_constlr_lr6e6'))
for r in runs: print(row(r,files['t'],files['m'],files['q'],p+r))
PYEOF
)

{
  echo "7 curriculum (6e-6 const) eval DONE at $(date)"
  echo "host: $(hostname)"
  echo
  echo "Final NDCG@10 (x100) -- InfoNCE 6e-6 constant, 8 rows:"
  echo "$TABLE"
  echo
  echo "(per-epoch JSONs: continual_ir/figures/curriculum6e6_* ; Claude assembles the paper table + curves.)"
} | mail -s "[continual_ir] 7 curriculum eval DONE -- 8-row table" huiyuche@iro.umontreal.ca
echo "eval alert email sent at $(date)" >> "$ALERT_LOG"
