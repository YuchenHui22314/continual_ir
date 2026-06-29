"""
plot_curriculum6e6_single.py
============================
Two single-panel curves (TopiOCQA + MS MARCO) for the InfoNCE 4x120 constant-LR 6e-6
curriculum runs, matching the style of fig:curves (a)-(h) in the paper: 6 lines
(Baseline + 3 CL solid + 2 ACL dashed; the *_root2 variants are omitted because they
track the baseline and clutter the panel, exactly as in plot_qwen_instruct_curves.py),
end-of-line labels (no legend), figsize (9, 5.5).

Outputs:
  figures/training_curves_qwen_infonce6e6.png          (TopiOCQA, fig:curves panel i)
  figures/training_curves_msmarco_qwen_infonce6e6.png  (MS MARCO, fig:curves panel j)
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIG = "/data/rech/huiyuche/continual_ir/figures"
P = "instruct3fp32infonce_constlr_lr6e6_"
BASE_RUN = "instruct3fp32infonce_qwen_constlr_lr6e6"
T_CURR = f"{FIG}/curriculum6e6_topiocqa_per_epoch.json"
M_CURR = f"{FIG}/curriculum6e6_msmarco_per_epoch.json"
T_BASE = f"{FIG}/constlr_lr6e6_topiocqa_per_epoch.json"
M_BASE = f"{FIG}/constlr_lr6e6_msmarco_per_epoch.json"

# (json_key, label, color, lw, zorder, linestyle) — same palette as the other panels.
STYLE = [
    (BASE_RUN,                "Baseline",          "#4e7b8a", 2.2, 4, "-"),
    (P + "cl_step",           "CL-step",           "#4caf73", 2.0, 3, "-"),
    (P + "cl_step_excl",      "CL-step-excl",      "#f0a500", 2.0, 3, "-"),
    (P + "cl_step_excl_2_full", "CL-step-excl-full", "#e8306a", 2.0, 3, "-"),
    (P + "acl_step",          "ACL-step",          "#0097a7", 2.0, 3, "--"),
    (P + "acl_step_excl",     "ACL-step-excl",     "#5d4037", 2.0, 3, "--"),
]

def load(p):
    with open(p) as f: return json.load(f)

def series(d, run):
    sub = d.get(run, {})
    steps = sorted(int(k) for k in sub)
    return [s // 94 for s in steps], [sub[str(s)] for s in steps]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.linewidth": 1.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 1.2, "ytick.major.width": 1.2, "figure.dpi": 150})

def annotate_end(ax, xs, ys, label, color, dy=0.0):
    if not xs: return
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 0.4, ys[-1] + dy),
                color=color, fontsize=13, fontweight="bold", va="center", ha="left")

def make(curr_path, base_path, ylabel, out, yoff=None):
    cd = load(curr_path); bd = load(base_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for run, lab, c, lw, zo, ls in STYLE:
        d = bd if run == BASE_RUN else cd
        xs, ys = series(d, run)
        if not ys:
            print(f"  [SKIP] {run} (no data)"); continue
        ax.plot(xs, ys, color=c, linewidth=lw, linestyle=ls, zorder=zo)
        annotate_end(ax, xs, ys, lab, c, (yoff or {}).get(run, 0.0))
    ax.set_xlabel("Epoch", fontsize=17); ax.set_ylabel(ylabel, fontsize=17)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2)); ax.tick_params(labelsize=14)
    ax.set_xlim(0, 20 + 5.5)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", dpi=150); plt.close(fig)
    print("Saved:", out)

# TopiOCQA — end values cluster ~0.42-0.475; fan labels apart to avoid collisions.
make(T_CURR, T_BASE, "NDCG@10", f"{FIG}/training_curves_qwen_infonce6e6.png", yoff={
    BASE_RUN: 0.010, P+"acl_step": 0.004, P+"cl_step_excl": -0.002,
    P+"cl_step_excl_2_full": -0.008, P+"cl_step": -0.014, P+"acl_step_excl": -0.020,
})
# MS MARCO — end values cluster ~0.345-0.355.
make(M_CURR, M_BASE, "NDCG@10", f"{FIG}/training_curves_msmarco_qwen_infonce6e6.png", yoff={
    BASE_RUN: 0.004, P+"acl_step": 0.001, P+"cl_step_excl": -0.001,
    P+"cl_step_excl_2_full": -0.003, P+"cl_step": -0.005, P+"acl_step_excl": 0.0025,
})
print("\nDone.")
