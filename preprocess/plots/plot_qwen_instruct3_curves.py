"""
plot_qwen_instruct3_curves.py
==============================
Mirror of plot_qwen_instruct_curves.py, but for the *instruct3* (v3-template)
Qwen3-Embedding-0.6B runs trained 2026-06-06 / 06-07. Produces the two
single-panel curves matching Figure 4 (a)(b)/(c)(d) style:
  - training_curves_qwen_instruct3.png         (TopiOCQA NDCG@10 per epoch)
  - training_curves_msmarco_qwen_instruct3.png (MSMARCO NDCG@10 per epoch)

Data sources — BOTH metrics come from offline per-epoch JSONs; NEITHER
in-training log series is usable for the instruct3 batch:
  - TopiOCQA per-epoch NDCG@10: read from
    `figures/instruct3_qwen_topiocqa_per_epoch.json`, produced offline by
    eval_instruct3_qwen_topiocqa_per_epoch.py under the proper v3 template.
    The in-training TopiOCQA eval calls in train_qwen_cl.py (as of the
    instruct3 batch) did not pass template_version, so eval_conv_search fell
    back to "v1": v3-trained models were evaluated on v1-format queries all
    through training, depressing every reading by ~7 NDCG@10 points
    (in-training final 0.3940 vs offline-v3 0.4669 on the same
    instruct3_nosched ckpt, 2026-06-09).
  - MSMARCO per-epoch NDCG@10: read from
    `figures/instruct3_qwen_msmarco_per_epoch.json`, produced offline by
    eval_instruct3_qwen_msmarco_per_epoch.py (raw tokenizer + per-task
    instruction map). The in-training MSMARCO log values are NOT usable:
    train_qwen_cl.py passed the Qwen3TokenizerWrapper to eval_beir_from_cache,
    which brackets each BEIR query with <|im_end|> instead of the official
    trailing <|endoftext|>, shifting the last-token pool onto a position the
    instruct-template models never train. Smoke 2026-06-09 (same ckpt, same
    instruction map): wrapper=0.1586 == training log, raw=0.3363 == offline.
    (Earlier revisions of this script parsed the training logs and produced
    bogus trajectories for both panels.)
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
# All 8 instruct3 training logs share the same suffix (the relaunched batch).
TS = "20260606_005231"

RUN_LOGS = {
    "instruct3_qwen_nosched":             f"{LOG_DIR}/run_instruct3_qwen_nosched_{TS}.log",
    "instruct3_qwen_cl_step":             f"{LOG_DIR}/run_instruct3_qwen_cl_step_{TS}.log",
    "instruct3_qwen_cl_step_excl":        f"{LOG_DIR}/run_instruct3_qwen_cl_step_excl_{TS}.log",
    "instruct3_qwen_cl_step_excl_2_full": f"{LOG_DIR}/run_instruct3_qwen_cl_step_excl_2_full_{TS}.log",
    # Anti-curriculum runs — dashed. Mirroring plot_qwen_instruct_curves.py
    # we omit *_root2 on both CL and ACL sides (closely tracks the baseline
    # and clutters the panel without adding signal).
    "instruct3_qwen_acl_step":            f"{LOG_DIR}/run_instruct3_qwen_acl_step_{TS}.log",
    "instruct3_qwen_acl_step_excl":       f"{LOG_DIR}/run_instruct3_qwen_acl_step_excl_{TS}.log",
}

# ---------------------------------------------------------------------------
# 2.  Loaders — BOTH metrics come from the offline per-epoch JSONs (see module
#     docstring for why neither in-training log series is usable).
# ---------------------------------------------------------------------------
FIG_DIR = "/data/rech/huiyuche/continual_ir/figures"
TOPIOCQA_JSON_PATH = f"{FIG_DIR}/instruct3_qwen_topiocqa_per_epoch.json"
MSMARCO_JSON_PATH  = f"{FIG_DIR}/instruct3_qwen_msmarco_per_epoch.json"

with open(TOPIOCQA_JSON_PATH) as f:
    TOPIOCQA_JSON = json.load(f)
with open(MSMARCO_JSON_PATH) as f:
    MSMARCO_JSON = json.load(f)


def get_series(json_dict, run_name, metric_label):
    """Return list of per-epoch NDCG@10 floats (sorted by ckpt step)."""
    d = json_dict.get(run_name, {})
    if not d:
        print(f"  [WARNING] no {metric_label} entries for {run_name}")
        return []
    steps = sorted(int(k) for k in d.keys())
    return [d[str(s)] for s in steps]


print("Loading offline per-epoch JSONs …")
all_data = {}
for run_name in RUN_LOGS:
    d = {
        "topiocqa_ndcg10": get_series(TOPIOCQA_JSON, run_name, "TopiOCQA"),
        "msmarco_ndcg10":  get_series(MSMARCO_JSON,  run_name, "MSMARCO"),
    }
    all_data[run_name] = d
    print(f"  {run_name}: {len(d['topiocqa_ndcg10'])} TopiOCQA epochs, "
          f"{len(d['msmarco_ndcg10'])} MSMARCO epochs")

# ---------------------------------------------------------------------------
# 3.  Style — identical palette/linestyles to plot_qwen_instruct_curves.py
#     so the instruct3 panels match Figure 4 (c)(d) visually.
# ---------------------------------------------------------------------------
STYLE = {
    # run_name:  (label, color, linewidth, zorder, linestyle)
    "instruct3_qwen_nosched":             ("Baseline",           "#4e7b8a", 2.2, 4, "-"),
    "instruct3_qwen_cl_step":             ("CL-step",            "#4caf73", 2.0, 3, "-"),
    "instruct3_qwen_cl_step_excl":        ("CL-step-excl",       "#f0a500", 2.0, 3, "-"),
    "instruct3_qwen_cl_step_excl_2_full": ("CL-step-excl-full",  "#e8306a", 2.0, 3, "-"),
    "instruct3_qwen_acl_step":            ("ACL-step",           "#0097a7", 2.0, 3, "--"),
    "instruct3_qwen_acl_step_excl":       ("ACL-step-excl",      "#5d4037", 2.0, 3, "--"),
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
# 4.  End-of-line label helper
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
# 5.  Single-panel curve plotter
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
print("\nGenerating training_curves_qwen_instruct3.png …")
make_figure(
    metric_key="topiocqa_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_qwen_instruct3.png",
    label_x_pad=0.4,
)

print("Generating training_curves_msmarco_qwen_instruct3.png …")
make_figure(
    metric_key="msmarco_ndcg10",
    ylabel="NDCG@10",
    out_path="/data/rech/huiyuche/continual_ir/figures/training_curves_msmarco_qwen_instruct3.png",
    label_x_pad=0.4,
)

print("\nDone.")
