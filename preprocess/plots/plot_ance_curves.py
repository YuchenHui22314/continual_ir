"""
plot_ance_curves.py
====================
Extract per-epoch eval metrics from ANCE training logs (baseline + CL + ACL)
and produce two training-curve figures matching the style of the Qwen ones:
  - training_curves_ance.png          (TopiOCQA NDCG@10)
  - training_curves_msmarco_ance.png  (MSMARCO NDCG@10)

root_2 pacing intentionally excluded (curves track baseline closely — not
informative to show). cl_root2 also left out for symmetry with the Qwen figure.
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# 1.  Log files
# ---------------------------------------------------------------------------
LOG_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/logs"

RUN_LOGS = {
    "ance_nosched":          f"{LOG_DIR}/ance_topiocqa_nosched_20260409_172816.log",
    # CL family (easy2hard) — solid. These are the April-10 reruns that match
    # the wandb CSV used in training_curves.png (earlier 20260409 curriculum
    # logs under-trained; values diverge by ~0.01 NDCG@10).
    "ance_cl_step":          f"{LOG_DIR}/cl_step_20260410_114544.log",
    "ance_cl_step_excl":     f"{LOG_DIR}/cl_step_exclusive_20260410_114544.log",
    "ance_cl_step_excl_2_full": f"{LOG_DIR}/cl_step_exclusive_2_full_20260410_114544.log",
    # ACL family (hard2easy) — dashed. root_2 omitted on purpose.
    "ance_acl_step":         f"{LOG_DIR}/anticl_step_20260410_123533.log",
    "ance_acl_step_excl":    f"{LOG_DIR}/anticl_step_exclusive_20260410_123533.log",
}

# ---------------------------------------------------------------------------
# 2.  Parsing
# ---------------------------------------------------------------------------
TOPIOCQA_PAT = re.compile(
    r"TopiOCQA eval: NDCG@10=([0-9.]+)\s+Recall@100=([0-9.]+)\s+MRR@10=([0-9.]+)"
)
MSMARCO_PAT = re.compile(
    r"eval_beir_from_cache msmarco: NDCG@10 = ([0-9.]+)"
)


def parse_log(path):
    data = {"topiocqa_ndcg10": [], "msmarco_ndcg10": []}
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
            continue
        m = MSMARCO_PAT.search(line)
        if m and "full eval" not in line:
            data["msmarco_ndcg10"].append(float(m.group(1)))
    return data


print("Parsing ANCE log files …")
all_data = {}
for run_name, log_path in RUN_LOGS.items():
    d = parse_log(log_path)
    all_data[run_name] = d
    print(f"  {run_name}: topiocqa={len(d['topiocqa_ndcg10'])}  msmarco={len(d['msmarco_ndcg10'])}")

# ---------------------------------------------------------------------------
# 3.  Style — mirror plot_qwen_curves.py so the ANCE / Qwen figures look alike
# ---------------------------------------------------------------------------
STYLE = {
    # run_name:  (label, color, linewidth, zorder, linestyle)
    "ance_nosched":              ("Baseline",          "#4e7b8a", 2.2, 4, "-"),
    "ance_cl_step":              ("CL-step",           "#4caf73", 2.0, 3, "-"),
    "ance_cl_step_excl":         ("CL-step-excl",      "#f0a500", 2.0, 3, "-"),
    "ance_cl_step_excl_2_full":  ("CL-step-excl-full", "#e8306a", 2.0, 3, "-"),
    "ance_acl_step":             ("ACL-step",          "#0097a7", 2.0, 3, "--"),
    "ance_acl_step_excl":        ("ACL-step-excl",     "#5d4037", 2.0, 3, "--"),
}

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


def make_figure(metric_key, ylabel, out_path, y_offsets=None, label_x_pad=0.4, ylim=None):
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

    all_xs = [len(all_data[r][metric_key]) for r in STYLE if all_data[r][metric_key]]
    x_max = max(all_xs) if all_xs else 20
    ax.set_xlim(left=0, right=x_max + 5.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4.  TopiOCQA NDCG@10
# ---------------------------------------------------------------------------
# End-of-training TopiOCQA NDCG@10 (April-10 reruns, matches wandb CSV):
#   cl_step_excl_2_full  0.2027  -> top
#   cl_step_excl         0.1990
#   nosched              0.1974
#   cl_step              0.1965
#   acl_step             0.1956
#   acl_step_excl        0.1939  -> bottom
print("\nGenerating training_curves_ance.png …")
make_figure(
    metric_key="topiocqa_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_ance.png",
    y_offsets={
        ("ance_cl_step_excl_2_full", "topiocqa_ndcg10"): (0.4,  0.004),
        ("ance_cl_step_excl",        "topiocqa_ndcg10"): (0.4,  0.000),
        ("ance_nosched",             "topiocqa_ndcg10"): (0.4, -0.004),
        ("ance_cl_step",             "topiocqa_ndcg10"): (0.4, -0.008),
        ("ance_acl_step",            "topiocqa_ndcg10"): (0.4, -0.012),
        ("ance_acl_step_excl",       "topiocqa_ndcg10"): (0.4, -0.016),
    },
)

# ---------------------------------------------------------------------------
# 5.  MSMARCO NDCG@10
# ---------------------------------------------------------------------------
# End-of-training MSMARCO NDCG@10 (April-10 reruns):
#   acl_step_excl        0.3054  -> top
#   nosched              0.3025
#   acl_step             0.3021
#   cl_step              0.2987
#   cl_step_excl         0.2968
#   cl_step_excl_2_full  0.2965  -> bottom
print("Generating training_curves_msmarco_ance.png …")
make_figure(
    metric_key="msmarco_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_msmarco_ance.png",
    y_offsets={
        ("ance_acl_step_excl",       "msmarco_ndcg10"): (0.4,  0.004),
        ("ance_nosched",             "msmarco_ndcg10"): (0.4,  0.001),
        ("ance_acl_step",            "msmarco_ndcg10"): (0.4, -0.003),
        ("ance_cl_step",             "msmarco_ndcg10"): (0.4, -0.008),
        ("ance_cl_step_excl",        "msmarco_ndcg10"): (0.4, -0.012),
        ("ance_cl_step_excl_2_full", "msmarco_ndcg10"): (0.4, -0.016),
    },
    ylim=(0.278, 0.344),
)

print("\nDone.")
