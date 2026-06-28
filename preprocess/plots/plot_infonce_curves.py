"""
plot_infonce_curves.py
======================
Figures for the official-aligned InfoNCE fp32-master runs
(instruct3fp32_infonce_lr1e5 / lr6e6). Produces:

  1. infonce_curves_3panel.png  — per-epoch NDCG@10 on TopiOCQA / MSMARCO / QReCC
     (x=epoch). Both LRs as solid lines; the OLD bf16 instruct3_nosched run
     overlaid dashed-grey for the recipe comparison; QReCC zero-shot as a
     horizontal reference.
  2. infonce_beir13_final.png   — final-checkpoint BEIR-13 NDCG@10, grouped bars
     (lr1e5 vs lr6e6) per dataset + the 13-dataset average.

All inputs are the offline per-epoch / final JSONs written by the
eval_infonce_qwen_* scripts (paper's instruct3_* JSONs untouched).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIG = "/data/rech/huiyuche/continual_ir/figures"
NEW = {
    "topiocqa": f"{FIG}/instruct3fp32_infonce_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/instruct3fp32_infonce_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_infonce_v3.json",
    "beir":     f"{FIG}/instruct3fp32_infonce_eval_results.json",
}
OLD = {  # old bf16 instruct3 (old in-batch-CE recipe) for overlay
    "topiocqa": f"{FIG}/instruct3_qwen_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/instruct3_qwen_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_qwen_instr_v3.json",
}
OLD_REF = "instruct3_qwen_nosched"          # the comparable old run (bf16, in-batch CE, 4x120)
RUN_1E5, RUN_6E6 = "instruct3fp32_infonce_lr1e5", "instruct3fp32_infonce_lr6e6"
# Aligned isolation run: in-batch CE + fp32-master at the EXACT 4x120 main setting
# (only precision differs from OLD_REF) — the clean one-factor precision comparison.
NOSCHED = {
    "topiocqa": f"{FIG}/instruct3fp32_nosched_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/instruct3fp32_nosched_msmarco_per_epoch.json",
}
NOSCHED_REF = "instruct3fp32_qwen_nosched"
# 4x120 (B): the canonical bf16_fp32_master InfoNCE (480 cross-GPU negs, FlashAttention-2) —
# the fastest + most faithful-to-official config (Table 1 last column "Qwen3 (InfoNCE 4x120, B)").
B4 = {
    "topiocqa": f"{FIG}/infonce4x120_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/infonce4x120_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_infonce_v3.json",   # 4x120 QReCC per-epoch not yet evaluated -> panel skips it
    "beir":     f"{FIG}/infonce4x120_eval_results.json",
}
RUN_4X120 = "instruct3fp32infonce_qwen_nosched"
# Constant-LR ablation of B: same InfoNCE 4x120 bf16_fp32_master + 480 cross-GPU negs,
# only the schedule differs (cosine+warmup0.1 -> constant, --no_lr_schedule).
CONSTLR = {
    "topiocqa": f"{FIG}/constlr_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/constlr_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_infonce_v3.json",
}
RUN_CONSTLR = "instruct3fp32infonce_qwen_constlr"
# 6e-6 constant-LR variant (smaller LR -> tamer late-epoch oscillation than the 1e-5 const run).
CONSTLR6 = {
    "topiocqa": f"{FIG}/constlr_lr6e6_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/constlr_lr6e6_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_infonce_v3.json",
}
RUN_CONSTLR6 = "instruct3fp32infonce_qwen_constlr_lr6e6"

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception as e:
        print(f"  [warn] {p}: {e}"); return {}

def series(d, run):
    sub = d.get(run, {})
    if not sub: return [], []
    steps = sorted(int(k) for k in sub)
    return [s // 94 for s in steps], [sub[str(s)] for s in steps]   # epoch = step/94

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C_1E5, C_6E6, C_OLD = "#1f77b4", "#e8306a", "#9e9e9e"
C_NS = "#2ca02c"   # in-batch CE + fp32-master, 4x120 (aligned isolation run)
C_B4 = "#9467bd"   # InfoNCE bf16_fp32_master, 4x120 (B) — the canonical official-aligned run
C_CONST = "#ff7f0e"  # InfoNCE 4x120, CONSTANT LR 1e-5 (schedule ablation of B)
C_CONST6 = "#8c564b"  # InfoNCE 4x120, CONSTANT LR 6e-6 (smaller LR, tamer oscillation)

# ---------------------------------------------------------------- 3-panel curves
data = {k: load(v) for k, v in NEW.items()}
old  = {k: load(v) for k, v in OLD.items()}
ns   = {k: load(v) for k, v in NOSCHED.items()}
b4   = {k: load(v) for k, v in B4.items()}
cst  = {k: load(v) for k, v in CONSTLR.items()}
cst6 = {k: load(v) for k, v in CONSTLR6.items()}
panels = [("topiocqa", "TopiOCQA (new task)"),
          ("msmarco",  "MS MARCO (previous task)"),
          ("qrecc",    "QReCC (held-out conv.)")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, (key, title) in zip(axes, panels):
    for run, c, lab in [(RUN_1E5, C_1E5, "InfoNCE (3x80) lr=1e-5"),
                        (RUN_6E6, C_6E6, "InfoNCE (3x80) lr=6e-6")]:
        xs, ys = series(data[key], run)
        if ys: ax.plot(xs, ys, color=c, lw=2.2, marker="o", ms=3, label=lab, zorder=4)
    # 4x120 (B): the canonical bf16_fp32_master InfoNCE (480 cross-GPU negs) — Table 1's last
    # column ("Qwen3 (InfoNCE 4x120, B)"). QReCC per-epoch for this run is not yet evaluated,
    # so series() returns empty there and the QReCC panel skips this curve.
    xb, yb = series(b4.get(key, {}), RUN_4X120)
    if yb: ax.plot(xb, yb, color=C_B4, lw=2.4, marker="^", ms=4,
                   label="InfoNCE (4x120) lr=1e-5 cosine", zorder=5)
    # constant-LR ablation of B: same InfoNCE 4x120 bf16_fp32_master, cosine -> constant LR.
    xc, yc = series(cst.get(key, {}), RUN_CONSTLR)
    if yc: ax.plot(xc, yc, color=C_CONST, lw=2.4, marker="D", ms=3.5,
                   label="InfoNCE (4x120) lr=1e-5 const", zorder=6)
    # 6e-6 constant-LR variant: smaller LR, expected tamer late-epoch oscillation than 1e-5 const.
    xc6, yc6 = series(cst6.get(key, {}), RUN_CONSTLR6)
    if yc6: ax.plot(xc6, yc6, color=C_CONST6, lw=2.4, marker="v", ms=3.5,
                    label="InfoNCE (4x120) lr=6e-6 const", zorder=7)
    # aligned isolation run: in-batch CE + fp32-master at the EXACT 4x120 main setting
    # (only precision differs from the bf16 main below) — the clean precision comparison.
    xn, yn = series(ns.get(key, {}), NOSCHED_REF)
    if yn: ax.plot(xn, yn, color=C_NS, lw=2.4, marker="s", ms=3,
                   label="in-batch CE (fp32, 4x120, aligned)", zorder=3)
    # old bf16 main overlay (in-batch CE, 4x120) — the precision baseline
    xo, yo = series(old.get(key, {}), OLD_REF)
    if yo: ax.plot(xo, yo, color=C_OLD, lw=1.8, ls="--", label="in-batch CE (bf16, 4x120, main)", zorder=2)
    # zero-shot reference (only QReCC json carries it)
    zs = data[key].get("zero_shot", {}).get("0")
    if zs is not None:
        ax.axhline(zs, color="k", lw=1.0, ls=":", zorder=1)
        ax.text(0.5, zs, f"zero-shot {zs:.3f}", fontsize=10, va="bottom", ha="left")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.set_xlim(0.5, 20.5)
    ax.grid(alpha=0.25, lw=0.6)
axes[0].set_ylabel("NDCG@10", fontsize=15)
axes[0].legend(fontsize=10.5, loc="lower right", framealpha=0.9)
fig.suptitle("Per-epoch NDCG@10 — precision isolation (in-batch CE: bf16 vs fp32, 4x120) + InfoNCE (3x80; 4x120 B: cosine vs constant schedule)",
             fontsize=14, y=1.02)
fig.tight_layout()
out1 = f"{FIG}/infonce_curves_3panel.png"
fig.savefig(out1, bbox_inches="tight", dpi=150); plt.close(fig)
print("Saved:", out1)

# ---------------------------------------------------------------- BEIR-13 bars
beir = data["beir"]
EXCL = {"msmarco"}
order = ["fever","climate-fever","hotpotqa","dbpedia-entity","nq","arguana",
         "webis-touche2020","fiqa","scidocs","scifact","trec-covid","nfcorpus","quora"]
def beir_vals(run):
    b = beir.get(run, {}).get("beir", {})
    vals = [b.get(d, {}).get("NDCG@10", float("nan")) for d in order]
    avg = np.nanmean([v for d, v in zip(order, vals) if d not in EXCL])
    return vals + [avg]
labels = [d.replace("webis-touche2020", "touche").replace("dbpedia-entity", "dbpedia")
          .replace("climate-fever", "climate-f") for d in order] + ["AVG-13"]
v1, v6 = beir_vals(RUN_1E5), beir_vals(RUN_6E6)
x = np.arange(len(labels)); w = 0.4
fig, ax = plt.subplots(figsize=(15, 5.2))
ax.bar(x - w/2, v1, w, color=C_1E5, label=f"lr=1e-5 (avg {v1[-1]:.3f})")
ax.bar(x + w/2, v6, w, color=C_6E6, label=f"lr=6e-6 (avg {v6[-1]:.3f})")
ax.axvline(len(order) - 0.5, color="k", lw=0.8, ls=":")  # separate AVG
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=11)
ax.set_ylabel("NDCG@10", fontsize=15)
ax.set_title("BEIR-13 (final checkpoint) — InfoNCE fp32-master", fontsize=15)
ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.25, lw=0.6)
fig.tight_layout()
out2 = f"{FIG}/infonce_beir13_final.png"
fig.savefig(out2, bbox_inches="tight", dpi=150); plt.close(fig)
print("Saved:", out2)
print("\nFinal NDCG@10 (step-1880):")
for run in (RUN_1E5, RUN_6E6):
    t = series(data['topiocqa'], run)[1]; m = series(data['msmarco'], run)[1]; q = series(data['qrecc'], run)[1]
    print(f"  {run}: TopiOCQA {t[-1]:.4f}  MSMARCO {m[-1]:.4f}  QReCC {q[-1]:.4f}  BEIR13 {beir_vals(run)[-1]:.4f}")
# schedule ablation — B (cosine) vs const, 3-panel only (no BEIR for these two)
print("\nSchedule ablation (4x120 InfoNCE, step-1880):")
_last = lambda v: v[-1] if v else float('nan')
for run, src in [(RUN_4X120, b4), (RUN_CONSTLR, cst), (RUN_CONSTLR6, cst6)]:
    t = series(src['topiocqa'], run)[1]; m = series(src['msmarco'], run)[1]; q = series(src.get('qrecc', {}), run)[1]
    print(f"  {run}: TopiOCQA {_last(t):.4f}  MSMARCO {_last(m):.4f}  QReCC {_last(q):.4f}")
