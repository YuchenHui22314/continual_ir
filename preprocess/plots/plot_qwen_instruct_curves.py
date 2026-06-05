"""
plot_qwen_instruct_curves.py
=============================
Mirror of plot_qwen_curves.py, but for the *instruct2* Qwen3-Embedding-0.6B
runs (training and eval both wrap conversational queries with
`Instruct: Given a conversation, retrieve relevant passages that help answer
the user's latest question\nConversation:{ctx}`).

Produces two single-panel curves matching the style of Figure 4 (a) and (b):
  - training_curves_qwen_instruct.png         (TopiOCQA NDCG@10 per epoch)
  - training_curves_msmarco_qwen_instruct.png (MSMARCO NDCG@10 per epoch)

Data sources:
  - TopiOCQA per-epoch NDCG@10: parsed from each run's training log
    (the in-training eval used the correct conversational instruction,
    so the log values are trustworthy).
  - MSMARCO per-epoch NDCG@10: read from
    `figures/instruct2_qwen_msmarco_per_epoch.json`. The in-training MSMARCO
    eval missed the per-task instruction map and collapsed to ~0.13;
    the per-epoch JSON re-evaluates every checkpoint with the corrected
    instruction-aware eval (commit ca2b960 in src/utils.py).
"""

import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# 1.  Map run names -> training log files
# ---------------------------------------------------------------------------
LOG_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/logs"
# All 8 instruct2 training logs share the same suffix.
TS = "20260519_125316"

RUN_LOGS = {
    "instruct2_qwen_nosched":             f"{LOG_DIR}/run_instruct2_qwen_nosched_{TS}.log",
    "instruct2_qwen_cl_step":             f"{LOG_DIR}/run_instruct2_qwen_cl_step_{TS}.log",
    "instruct2_qwen_cl_step_excl":        f"{LOG_DIR}/run_instruct2_qwen_cl_step_excl_{TS}.log",
    "instruct2_qwen_cl_step_excl_2_full": f"{LOG_DIR}/run_instruct2_qwen_cl_step_excl_2_full_{TS}.log",
    # Anti-curriculum runs — dashed. We omit *_root2 on both CL and ACL sides
    # to mirror plot_qwen_curves.py (root2 closely tracks the baseline and
    # clutters the panel without adding signal).
    "instruct2_qwen_acl_step":            f"{LOG_DIR}/run_instruct2_qwen_acl_step_{TS}.log",
    "instruct2_qwen_acl_step_excl":       f"{LOG_DIR}/run_instruct2_qwen_acl_step_excl_{TS}.log",
}

# ---------------------------------------------------------------------------
# 2.  Parsers
# ---------------------------------------------------------------------------
# In-training eval line format (note: lowercase "topiocqa" in instruct2 logs):
#   2026-05-19 13:08:29,393 - utils - INFO - topiocqa eval: NDCG@10=0.4304 ...
TOPIOCQA_PAT = re.compile(
    r"topiocqa eval: NDCG@10=([0-9.]+)\s+Recall@100=([0-9.]+)\s+MRR@10=([0-9.]+)"
)


def parse_topiocqa_log(path):
    """Return list of per-epoch NDCG@10 floats (in training order)."""
    ndcg = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = TOPIOCQA_PAT.search(line)
                if m:
                    ndcg.append(float(m.group(1)))
    except FileNotFoundError:
        print(f"  [WARNING] File not found: {path}")
    return ndcg


# Pre-loaded MSMARCO JSON: {run_name: {step_str: ndcg}}
MSMARCO_JSON_PATH = "/data/rech/huiyuche/continual_ir/figures/instruct2_qwen_msmarco_per_epoch.json"
with open(MSMARCO_JSON_PATH) as f:
    MSMARCO_JSON = json.load(f)


def get_msmarco(run_name):
    """Return list of per-epoch MSMARCO NDCG@10 floats (sorted by step)."""
    d = MSMARCO_JSON.get(run_name, {})
    if not d:
        print(f"  [WARNING] no MSMARCO entries for {run_name}")
        return []
    steps = sorted(int(k) for k in d.keys())
    return [d[str(s)] for s in steps]


print("Parsing logs / loading JSON …")
all_data = {}
for run_name, log_path in RUN_LOGS.items():
    topiocqa = parse_topiocqa_log(log_path)
    msmarco  = get_msmarco(run_name)
    all_data[run_name] = {"topiocqa_ndcg10": topiocqa, "msmarco_ndcg10": msmarco}
    print(f"  {run_name}: {len(topiocqa)} TopiOCQA epochs, {len(msmarco)} MSMARCO epochs")

# ---------------------------------------------------------------------------
# 3.  Style — identical palette/linestyles to plot_qwen_curves.py so the
#     instruct panels match Figure 4 (a)(b) visually.
# ---------------------------------------------------------------------------
STYLE = {
    # run_name:  (label, color, linewidth, zorder, linestyle)
    "instruct2_qwen_nosched":             ("Baseline",           "#4e7b8a", 2.2, 4, "-"),
    "instruct2_qwen_cl_step":             ("CL-step",            "#4caf73", 2.0, 3, "-"),
    "instruct2_qwen_cl_step_excl":        ("CL-step-excl",       "#f0a500", 2.0, 3, "-"),
    "instruct2_qwen_cl_step_excl_2_full": ("CL-step-excl-full",  "#e8306a", 2.0, 3, "-"),
    "instruct2_qwen_acl_step":            ("ACL-step",           "#0097a7", 2.0, 3, "--"),
    "instruct2_qwen_acl_step_excl":       ("ACL-step-excl",      "#5d4037", 2.0, 3, "--"),
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         16,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# 4.  End-of-line label helper (same as plot_qwen_curves.py)
# ---------------------------------------------------------------------------
def annotate_end(ax, xs, ys, label, color, offset=(0.4, 0)):
    if not xs or not ys:
        return
    x_end, y_end = xs[-1], ys[-1]
    ax.annotate(
        label,
        xy=(x_end, y_end),
        xytext=(x_end + offset[0], y_end + offset[1]),
        color=color,
        fontsize=13,
        fontweight="bold",
        va="center",
        ha="left",
    )


# ---------------------------------------------------------------------------
# 5.  Single-panel curve plotter (identical to plot_qwen_curves.py)
# ---------------------------------------------------------------------------
def make_figure(metric_key, ylabel, out_path, y_offsets=None,
                label_x_pad=0.4):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for run_name, (label, color, lw, zo, ls) in STYLE.items():
        ys = all_data[run_name][metric_key]
        if not ys:
            print(f"  [SKIP] {run_name} has no data for {metric_key}")
            continue
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=zo)

        dy = 0.0
        if y_offsets and (run_name, metric_key) in y_offsets:
            dy = y_offsets[(run_name, metric_key)][1]
        annotate_end(ax, xs, ys, label, color, offset=(label_x_pad, dy))

    ax.set_xlabel("Epoch", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(axis="both", labelsize=14)

    all_xs = []
    for run_name in STYLE:
        ys = all_data[run_name][metric_key]
        if ys:
            all_xs.append(len(ys))
    x_max = max(all_xs) if all_xs else 20
    ax.set_xlim(left=0, right=x_max + 5.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6.  Generate the two panels (TopiOCQA + MSMARCO)
# ---------------------------------------------------------------------------
print("\nGenerating training_curves_qwen_instruct.png …")
# Label y-offsets are best tuned once we see end-of-training spreads; default
# offsets here are conservative (small, like plot_qwen_curves.py). If labels
# overlap visibly in the final PNG, hand-tune these per (run, metric).
make_figure(
    metric_key="topiocqa_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_qwen_instruct.png",
    # End-of-training values cluster in [0.452, 0.467] — labels collide unless
    # we deliberately fan them out by ~0.008 between adjacent ranks.
    #   CL-step-excl-full  0.4668  -> nudge UP   to ~0.480
    #   Baseline           0.4648  -> nudge UP   to ~0.472
    #   ACL-step           0.4590  -> nudge UP   to ~0.464
    #   CL-step-excl       0.4585  -> nudge DOWN to ~0.456
    #   CL-step            0.4571  -> nudge DOWN to ~0.450
    #   ACL-step-excl      0.4525  -> nudge DOWN to ~0.443
    y_offsets={
        ("instruct2_qwen_cl_step_excl_2_full", "topiocqa_ndcg10"): (0.4,  0.013),
        ("instruct2_qwen_nosched",             "topiocqa_ndcg10"): (0.4,  0.007),
        ("instruct2_qwen_acl_step",            "topiocqa_ndcg10"): (0.4,  0.005),
        ("instruct2_qwen_cl_step_excl",        "topiocqa_ndcg10"): (0.4, -0.003),
        ("instruct2_qwen_cl_step",             "topiocqa_ndcg10"): (0.4, -0.007),
        ("instruct2_qwen_acl_step_excl",       "topiocqa_ndcg10"): (0.4, -0.010),
    },
    label_x_pad=0.4,
)

print("Generating training_curves_msmarco_qwen_instruct.png …")
make_figure(
    metric_key="msmarco_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_msmarco_qwen_instruct.png",
    # Final MSMARCO NDCG@10 (sorted high→low; only shown 6 runs listed):
    #   ACL-step           0.3383  -> nudge UP   to ~0.341
    #   ACL-step-excl      0.3352  -> -
    #   Baseline           0.3344  -> -
    #   CL-step-excl       0.3309  -> -
    #   CL-step-excl-full  0.3286  -> -
    #   CL-step            0.3275  -> nudge DOWN to ~0.324
    y_offsets={
        ("instruct2_qwen_acl_step",            "msmarco_ndcg10"): (0.4,  0.0025),
        ("instruct2_qwen_acl_step_excl",       "msmarco_ndcg10"): (0.4,  0.0000),
        ("instruct2_qwen_nosched",             "msmarco_ndcg10"): (0.4, -0.0008),
        ("instruct2_qwen_cl_step_excl",        "msmarco_ndcg10"): (0.4,  0.0000),
        ("instruct2_qwen_cl_step_excl_2_full", "msmarco_ndcg10"): (0.4, -0.0005),
        ("instruct2_qwen_cl_step",             "msmarco_ndcg10"): (0.4, -0.0025),
    },
    label_x_pad=0.4,
)

print("\nDone.")
