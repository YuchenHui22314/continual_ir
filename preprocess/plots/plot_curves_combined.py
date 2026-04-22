"""
plot_curves_combined.py
========================
Combined 2x2 grid of training curves:
    (top row)    Qwen  TopiOCQA NDCG@10   |   Qwen  MSMARCO NDCG@10
    (bottom row) ANCE  TopiOCQA NDCG@10   |   ANCE  MSMARCO NDCG@10

Reuses the same log files and styling as plot_qwen_curves.py and
plot_ance_curves.py. Output: figures/training_curves_combined.png.
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/logs"

QWEN_LOGS = {
    "nosched":             f"{LOG_DIR}/run_qwen_nosched_20260419_180606.log",
    "cl_step":             f"{LOG_DIR}/run_qwen_cl_step_20260419_180606.log",
    "cl_step_excl":        f"{LOG_DIR}/run_qwen_cl_step_excl_20260419_180606.log",
    "cl_step_excl_2_full": f"{LOG_DIR}/run_qwen_cl_step_excl_2_full_20260419_180606.log",
    "acl_step":            f"{LOG_DIR}/run_qwen_acl_step_20260419_184548.log",
    "acl_step_excl":       f"{LOG_DIR}/run_qwen_acl_step_excl_20260419_184548.log",
}

ANCE_LOGS = {
    "nosched":             f"{LOG_DIR}/ance_topiocqa_nosched_20260409_172816.log",
    "cl_step":             f"{LOG_DIR}/cl_step_20260410_114544.log",
    "cl_step_excl":        f"{LOG_DIR}/cl_step_exclusive_20260410_114544.log",
    "cl_step_excl_2_full": f"{LOG_DIR}/cl_step_exclusive_2_full_20260410_114544.log",
    "acl_step":            f"{LOG_DIR}/anticl_step_20260410_123533.log",
    "acl_step_excl":       f"{LOG_DIR}/anticl_step_exclusive_20260410_123533.log",
}

TOPIOCQA_PAT = re.compile(
    r"TopiOCQA eval: NDCG@10=([0-9.]+)\s+Recall@100=([0-9.]+)\s+MRR@10=([0-9.]+)"
)
MSMARCO_PAT = re.compile(r"eval_beir_from_cache msmarco: NDCG@10 = ([0-9.]+)")


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


print("Parsing logs …")
qwen_data = {k: parse_log(v) for k, v in QWEN_LOGS.items()}
ance_data = {k: parse_log(v) for k, v in ANCE_LOGS.items()}

# Run label + color shared across both encoders so the legend reads the same.
STYLE = {
    # short_key:  (label, color, linewidth, linestyle)
    "nosched":             ("Baseline",          "#4e7b8a", 2.2, "-"),
    "cl_step":             ("CL-step",           "#4caf73", 2.0, "-"),
    "cl_step_excl":        ("CL-step-excl",      "#f0a500", 2.0, "-"),
    "cl_step_excl_2_full": ("CL-step-excl-full", "#e8306a", 2.0, "-"),
    "acl_step":            ("ACL-step",          "#0097a7", 2.0, "--"),
    "acl_step_excl":       ("ACL-step-excl",     "#5d4037", 2.0, "--"),
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        14,
    "axes.linewidth":   1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi":       150,
})


def plot_panel(ax, data_dict, metric_key, title, ylim=None):
    for key, (_, color, lw, ls) in STYLE.items():
        ys = data_dict[key][metric_key]
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("NDCG@10", fontsize=14)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.tick_params(axis="both", labelsize=12)
    all_xs = [len(data_dict[k][metric_key]) for k in STYLE if data_dict[k][metric_key]]
    x_max = max(all_xs) if all_xs else 20
    ax.set_xlim(left=0, right=x_max + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

plot_panel(axes[0, 0], qwen_data, "topiocqa_ndcg10", "Qwen3-0.6B — TopiOCQA")
plot_panel(axes[0, 1], qwen_data, "msmarco_ndcg10",  "Qwen3-0.6B — MSMARCO")
plot_panel(axes[1, 0], ance_data, "topiocqa_ndcg10", "ANCE — TopiOCQA")
plot_panel(axes[1, 1], ance_data, "msmarco_ndcg10",  "ANCE — MSMARCO", ylim=(0.278, 0.344))

# Single shared legend at the bottom
handles = []
for key, (label, color, lw, ls) in STYLE.items():
    handles.append(plt.Line2D([0], [0], color=color, linewidth=lw, linestyle=ls, label=label))
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=6,
    frameon=False,
    fontsize=13,
    bbox_to_anchor=(0.5, -0.02),
)

fig.tight_layout(rect=(0, 0.04, 1, 1))
out_path = "/data/rech/huiyuche/continual_ir/figures/training_curves_combined.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")
