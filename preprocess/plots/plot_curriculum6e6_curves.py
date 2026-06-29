"""
plot_curriculum6e6_curves.py
============================
Per-epoch NDCG@10 for the 7 curriculum / anti-curriculum variants + the random-order
baseline, all under the InfoNCE 4x120 CONSTANT-LR 6e-6 recipe. 3 panels:
TopiOCQA (new task) / MS MARCO (previous task) / QReCC (held-out conv).
Solid = CL (easy2hard), dashed = ACL (hard2easy), black = random-order baseline.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIG = "/data/rech/huiyuche/continual_ir/figures"
P = "instruct3fp32infonce_constlr_lr6e6_"
BASE_RUN = "instruct3fp32infonce_qwen_constlr_lr6e6"
CURR = {
    "topiocqa": f"{FIG}/curriculum6e6_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/curriculum6e6_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_curriculum6e6.json",
}
BASE = {
    "topiocqa": f"{FIG}/constlr_lr6e6_topiocqa_per_epoch.json",
    "msmarco":  f"{FIG}/constlr_lr6e6_msmarco_per_epoch.json",
    "qrecc":    f"{FIG}/qrecc_per_epoch_infonce_v3.json",
}
CL = [
    ("cl_step",             "CL-step",           "#1f77b4", "-"),
    ("cl_step_excl",        "CL-step-excl",      "#2ca02c", "-"),
    ("cl_step_excl_2_full", "CL-step-excl-full", "#17becf", "-"),
    ("cl_root2",            "CL-root2",          "#9467bd", "-"),
]
ACL = [
    ("acl_root2",           "ACL-root2",         "#e8306a", "--"),
    ("acl_step",            "ACL-step",          "#ff7f0e", "--"),
    ("acl_step_excl",       "ACL-step-excl",     "#8c564b", "--"),
]

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception as e:
        print(f"[warn] {p}: {e}"); return {}

def series(d, run):
    sub = d.get(run, {})
    if not sub: return [], []
    steps = sorted(int(k) for k in sub)
    return [s // 94 for s in steps], [sub[str(s)] for s in steps]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
panels = [("topiocqa", "TopiOCQA (new task)"), ("msmarco", "MS MARCO (previous task)"),
          ("qrecc", "QReCC (held-out conv.)")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, (key, title) in zip(axes, panels):
    cd = load(CURR[key]); bd = load(BASE[key])
    xb, yb = series(bd, BASE_RUN)
    if yb: ax.plot(xb, yb, color="k", lw=2.6, marker="o", ms=3, label="w/o curriculum", zorder=5)
    for suf, lab, c, ls in CL + ACL:
        xs, ys = series(cd, P + suf)
        if ys: ax.plot(xs, ys, color=c, lw=1.8, ls=ls, marker=("o" if ls == "-" else "^"),
                       ms=2.5, label=lab, zorder=3)
    ax.set_title(title); ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4)); ax.set_xlim(0.5, 20.5)
    ax.grid(alpha=0.25, lw=0.6)
axes[0].set_ylabel("NDCG@10")
axes[0].legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.9)
fig.suptitle("Curriculum / anti-curriculum per-epoch NDCG@10 — InfoNCE 4x120 constant-LR 6e-6 "
             "(solid=CL easy2hard, dashed=ACL hard2easy, black=random)", fontsize=12.5, y=1.02)
fig.tight_layout()
out = f"{FIG}/curriculum6e6_curves_3panel.png"
fig.savefig(out, bbox_inches="tight", dpi=150); plt.close(fig)
print("Saved:", out)

print("\nFinal NDCG@10 (step-1880):")
rows = [(BASE, BASE_RUN, "w/o curriculum")] + [(CURR, P + s, l) for s, l, _, _ in CL + ACL]
for src, run, lab in rows:
    t = series(load(src['topiocqa']), run)[1]; m = series(load(src['msmarco']), run)[1]; q = series(load(src['qrecc']), run)[1]
    ff = lambda v: v[-1] * 100 if v else float('nan')
    print(f"  {lab:<20} TopiOCQA {ff(t):5.1f}  MSMARCO {ff(m):5.1f}  QReCC {ff(q):5.1f}")
