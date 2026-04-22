"""
plot_qwen_curves.py
====================
Extract per-epoch eval metrics from Qwen3-Embedding-0.6B training logs and
produce two training-curve figures that match the style of the existing ANCE
figures:
  - training_curves_qwen.png       (TopiOCQA metrics)
  - training_curves_msmarco_qwen.png  (MSMARCO NDCG@10)

Log files are in: /data/rech/huiyuche/TREC_iKAT_2024/logs/run_qwen_*.log
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# 1.  Map run names -> log files (use the longest / most-complete version)
# ---------------------------------------------------------------------------
LOG_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/logs"

RUN_LOGS = {
    "qwen_nosched":              f"{LOG_DIR}/run_qwen_nosched_20260419_180606.log",
    "qwen_cl_step":              f"{LOG_DIR}/run_qwen_cl_step_20260419_180606.log",
    "qwen_cl_step_excl":         f"{LOG_DIR}/run_qwen_cl_step_excl_20260419_180606.log",
    "qwen_cl_step_excl_2_full":  f"{LOG_DIR}/run_qwen_cl_step_excl_2_full_20260419_180606.log",
    # Anti-curriculum runs (hard2easy): dashed. root2 intentionally omitted
    # (curve essentially tracks baseline — not informative to show).
    "qwen_acl_step":             f"{LOG_DIR}/run_qwen_acl_step_20260419_184548.log",
    "qwen_acl_step_excl":        f"{LOG_DIR}/run_qwen_acl_step_excl_20260419_184548.log",
}

# ---------------------------------------------------------------------------
# 2.  Parse metrics from each log
# ---------------------------------------------------------------------------
# Patterns to match the two kinds of eval lines that appear once per epoch
# (emitted by the rank-0 process only).
TOPIOCQA_PAT = re.compile(
    r"TopiOCQA eval: NDCG@10=([0-9.]+)\s+Recall@100=([0-9.]+)\s+MRR@10=([0-9.]+)"
)
CLIMATE_PAT = re.compile(
    r"eval_beir_from_cache climate-fever: NDCG@10 = ([0-9.]+)"
)
MSMARCO_PAT = re.compile(
    r"eval_beir_from_cache msmarco: NDCG@10 = ([0-9.]+)"
)


def parse_log(path):
    """Return dict with lists: ndcg, recall100, mrr, climate_ndcg, msmarco_ndcg."""
    data = {
        "topiocqa_ndcg10":   [],
        "topiocqa_recall100": [],
        "topiocqa_mrr10":    [],
        "climate_ndcg10":    [],
        "msmarco_ndcg10":    [],
    }
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  [WARNING] File not found: {path}")
        return data

    for line in lines:
        m = TOPIOCQA_PAT.search(line)
        if m:
            data["topiocqa_ndcg10"].append(float(m.group(1)))
            data["topiocqa_recall100"].append(float(m.group(2)))
            data["topiocqa_mrr10"].append(float(m.group(3)))
            continue
        m = CLIMATE_PAT.search(line)
        if m and "full eval" not in line:    # skip the final "full eval" summary line
            data["climate_ndcg10"].append(float(m.group(1)))
            continue
        m = MSMARCO_PAT.search(line)
        if m and "full eval" not in line:
            data["msmarco_ndcg10"].append(float(m.group(1)))

    return data


print("Parsing log files …")
all_data = {}
for run_name, log_path in RUN_LOGS.items():
    d = parse_log(log_path)
    all_data[run_name] = d
    n = len(d["topiocqa_ndcg10"])
    print(f"  {run_name}: {n} epochs of TopiOCQA data, "
          f"{len(d['climate_ndcg10'])} climate-fever, "
          f"{len(d['msmarco_ndcg10'])} msmarco")

# ---------------------------------------------------------------------------
# 3.  Style — match the existing ANCE figures exactly
# ---------------------------------------------------------------------------
# Colors extracted visually from the ANCE figures:
#   Baseline       -> slate/dark-teal  (#4e7b89 / "steelblue"-ish)
#   CL-step        -> green            (#4caf73)
#   CL-step-excl   -> orange           (#f0a500)
#   CL-step-excl-full -> hot-pink/crimson (#e8306a)
# For cl_root2 we add a 5th distinctive color: purple

STYLE = {
    # run_name:  (label, color, linewidth, zorder, linestyle)
    "qwen_nosched":             ("Baseline",           "#4e7b8a", 2.2, 4, "-"),
    # CL family (easy2hard) — solid
    "qwen_cl_step":             ("CL-step",            "#4caf73", 2.0, 3, "-"),
    "qwen_cl_step_excl":        ("CL-step-excl",       "#f0a500", 2.0, 3, "-"),
    "qwen_cl_step_excl_2_full": ("CL-step-excl-full",  "#e8306a", 2.0, 3, "-"),
    # ACL family (hard2easy) — dashed, with its own distinct colors
    "qwen_acl_step":            ("ACL-step",           "#0097a7", 2.0, 3, "--"),
    "qwen_acl_step_excl":       ("ACL-step-excl",      "#5d4037", 2.0, 3, "--"),
}

# Matplotlib global style tweaks (match the look: large font, clean axes)
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        16,
    "axes.linewidth":   1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi":       150,
})

# ---------------------------------------------------------------------------
# 4.  Helper — annotate last point with run label (same as ANCE figures)
# ---------------------------------------------------------------------------
def annotate_end(ax, xs, ys, label, color, offset=(0.4, 0)):
    """Place a text label at the right end of the curve."""
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

# Helper to nudge label positions so they don't overlap
LABEL_OFFSETS_TOPO = {
    # (run_name, metric_key) -> (dx, dy)  in data units
    ("qwen_nosched",             "topiocqa_ndcg10"):   (0.3,  0.004),
    ("qwen_cl_step",             "topiocqa_ndcg10"):   (0.3, -0.007),
    ("qwen_cl_step_excl",        "topiocqa_ndcg10"):   (0.3,  0.002),
    ("qwen_cl_step_excl_2_full", "topiocqa_ndcg10"):   (0.3,  0.000),
    ("qwen_cl_root2",            "topiocqa_ndcg10"):   (0.3, -0.003),
}

# ---------------------------------------------------------------------------
# 5.  Figure 1 — TopiOCQA metrics  (3 sub-plots: NDCG@10, MRR@10, Recall@100)
# ---------------------------------------------------------------------------
# We'll draw a 1-row × 3-col layout (one panel per metric) — BUT the existing
# ANCE figure is a single-panel NDCG@10 curve.  Let's match that: one panel,
# one metric (NDCG@10 on TopiOCQA), same as the existing figure.
# ----- Actually looking at the reference image: it is a SINGLE subplot -----

def make_figure(metric_key, ylabel, out_path, y_offsets=None,
                label_x_pad=0.3):
    """Draw a single-panel training curve and save to out_path."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for run_name, (label, color, lw, zo, ls) in STYLE.items():
        ys = all_data[run_name][metric_key]
        if not ys:
            print(f"  [SKIP] {run_name} has no data for {metric_key}")
            continue
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=zo)

        # label at end of line
        dy = 0.0
        if y_offsets and (run_name, metric_key) in y_offsets:
            dy = y_offsets[(run_name, metric_key)][1]
        annotate_end(ax, xs, ys, label, color, offset=(label_x_pad, dy))

    ax.set_xlabel("Epoch", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(axis="both", labelsize=14)

    # Determine actual x-range from data, then add right padding for labels
    all_xs = []
    for run_name in STYLE:
        ys = all_data[run_name][metric_key]
        if ys:
            all_xs.append(len(ys))
    x_max = max(all_xs) if all_xs else 20
    ax.set_xlim(left=0, right=x_max + 5.5)   # extra room on right for labels

    # x-axis starts at 0, same as ANCE figures
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -- Figure 1: TopiOCQA NDCG@10 (matches training_curves.png style) ---------
# Label y-offsets manually tuned so labels don't overlap at epoch 19.
# Values are in NDCG@10 data units (approx range 0.36–0.49).
# At epoch 19, approximate values:
#   nosched        ~ 0.477  -> label above
#   cl_step        ~ 0.475  -> label below
#   cl_step_excl   ~ 0.484  -> label top
#   cl_step_excl_2_full ~ 0.480  -> label middle
#   cl_root2       ~ 0.469  -> label bottom
print("\nGenerating training_curves_qwen.png …")
# End-of-training values (epoch 19, sorted high→low):
#   cl_step_excl   0.4845   -> top
#   cl_full        0.4789
#   nosched        0.4771
#   cl_step        0.4717
#   acl_root2      0.4672
#   acl_step       0.4637
#   acl_excl       0.4607   -> bottom
# Spread labels so 7 rows are legible.
make_figure(
    metric_key="topiocqa_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_qwen.png",
    y_offsets={
        ("qwen_cl_step_excl",        "topiocqa_ndcg10"): (0.4,  0.003),
        ("qwen_cl_step_excl_2_full", "topiocqa_ndcg10"): (0.4,  0.003),
        ("qwen_nosched",             "topiocqa_ndcg10"): (0.4, -0.004),
        ("qwen_cl_step",             "topiocqa_ndcg10"): (0.4, -0.004),
        ("qwen_acl_step",            "topiocqa_ndcg10"): (0.4, -0.002),
        ("qwen_acl_step_excl",       "topiocqa_ndcg10"): (0.4, -0.005),
    },
    label_x_pad=0.4,
)

# -- Figure 2: MSMARCO NDCG@10 (matches training_curves_msmarco.png style) --
# At epoch 19, approximate values:
#   nosched        ~ 0.296  -> top
#   cl_step_excl   ~ 0.296  -> just below nosched
#   cl_step_excl_2_full ~ 0.285  -> middle
#   cl_root2       ~ 0.282  -> lower
#   cl_step        ~ 0.272  -> bottom
print("Generating training_curves_msmarco_qwen.png …")
# End-of-training msmarco NDCG@10 (sorted high→low):
#   acl_step       0.3043
#   nosched        0.2960
#   cl_step_excl   0.2956   -> overlaps nosched, nudge down
#   acl_excl       0.2908
#   cl_full        0.2840
#   cl_step        0.2725   -> bottom
make_figure(
    metric_key="msmarco_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_msmarco_qwen.png",
    y_offsets={
        ("qwen_nosched",             "msmarco_ndcg10"): (0.4,  0.002),
        ("qwen_cl_step_excl",        "msmarco_ndcg10"): (0.4, -0.002),
        ("qwen_cl_step_excl_2_full", "msmarco_ndcg10"): (0.4,  0.000),
        ("qwen_cl_step",             "msmarco_ndcg10"): (0.4,  0.000),
        ("qwen_acl_step",            "msmarco_ndcg10"): (0.4,  0.000),
        ("qwen_acl_step_excl",       "msmarco_ndcg10"): (0.4,  0.000),
    },
    label_x_pad=0.4,
)

print("\nDone.")
