"""
plot_curves_static_styles.py
=============================
Render the same 2x2 training-curve grid in three different static styles:

  - figures/training_curves_seaborn.png    (seaborn whitegrid)
  - figures/training_curves_nature.png     (Nature/IEEE paper style)
  - figures/training_curves_ggplot.png     (plotnine / ggplot2-style)
"""

import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

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

TOPIOCQA_PAT = re.compile(r"TopiOCQA eval: NDCG@10=([0-9.]+)")
MSMARCO_PAT  = re.compile(r"eval_beir_from_cache msmarco: NDCG@10 = ([0-9.]+)")


def parse_log(path):
    data = {"topiocqa_ndcg10": [], "msmarco_ndcg10": []}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = TOPIOCQA_PAT.search(line)
            if m:
                data["topiocqa_ndcg10"].append(float(m.group(1)))
                continue
            m = MSMARCO_PAT.search(line)
            if m and "full eval" not in line:
                data["msmarco_ndcg10"].append(float(m.group(1)))
    return data


def build_long_df():
    """Long-form DataFrame: encoder, dataset, run, epoch, ndcg."""
    label_map = {
        "nosched":             "Baseline",
        "cl_step":             "CL-step",
        "cl_step_excl":        "CL-step-excl",
        "cl_step_excl_2_full": "CL-step-excl-full",
        "acl_step":            "ACL-step",
        "acl_step_excl":       "ACL-step-excl",
    }
    rows = []
    for encoder, logs in [("Qwen3-0.6B", QWEN_LOGS), ("ANCE", ANCE_LOGS)]:
        for key, path in logs.items():
            d = parse_log(path)
            for dataset, metric in [("TopiOCQA", "topiocqa_ndcg10"),
                                    ("MS MARCO", "msmarco_ndcg10")]:
                for i, v in enumerate(d[metric], start=1):
                    rows.append({
                        "encoder": encoder,
                        "dataset": dataset,
                        "run":     label_map[key],
                        "run_key": key,
                        "epoch":   i,
                        "ndcg":    v,
                    })
    return pd.DataFrame(rows)


df = build_long_df()
print(f"Parsed {len(df)} rows.")

# Consistent palette + linestyle mapping used by all three styles
RUN_ORDER = ["Baseline", "CL-step", "CL-step-excl", "CL-step-excl-full",
             "ACL-step", "ACL-step-excl"]
PALETTE = {
    "Baseline":          "#2F4858",
    "CL-step":           "#2E8B57",
    "CL-step-excl":      "#D99000",
    "CL-step-excl-full": "#C9186D",
    "ACL-step":          "#1F9FB8",
    "ACL-step-excl":     "#8B5A3C",
}
DASHES = {
    "Baseline":          (1, 0),
    "CL-step":           (1, 0),
    "CL-step-excl":      (1, 0),
    "CL-step-excl-full": (1, 0),
    "ACL-step":          (5, 2),
    "ACL-step-excl":     (5, 2),
}

# y-ranges for the two MS MARCO panels (narrow so small deltas show)
YLIM = {
    ("Qwen3-0.6B", "TopiOCQA"): (0.34, 0.50),
    ("Qwen3-0.6B", "MS MARCO"): (0.268, 0.335),
    ("ANCE",       "TopiOCQA"): (0.09, 0.22),
    ("ANCE",       "MS MARCO"): (0.284, 0.330),
}


# ---------------------------------------------------------------------------
#  Style 1: seaborn whitegrid
# ---------------------------------------------------------------------------
def plot_seaborn():
    sns.set_theme(style="whitegrid", context="paper",
                  font="DejaVu Sans", font_scale=1.25)

    g = sns.FacetGrid(
        df, row="encoder", col="dataset",
        row_order=["Qwen3-0.6B", "ANCE"],
        col_order=["TopiOCQA", "MS MARCO"],
        height=3.4, aspect=1.55,
        sharex=False, sharey=False,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.lineplot,
        x="epoch", y="ndcg", hue="run", style="run",
        hue_order=RUN_ORDER, style_order=RUN_ORDER,
        palette=PALETTE, dashes=DASHES,
        linewidth=2.2, marker="o", markersize=4.5, markeredgewidth=0,
    )
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NDCG@10")
        ax.set_ylim(*YLIM[(row_val, col_val)])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(True, alpha=0.35, linewidth=0.7)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.add_legend(title=None, loc="lower center",
                 bbox_to_anchor=(0.5, -0.04), ncol=6, frameon=False)
    g.figure.subplots_adjust(bottom=0.14, wspace=0.22, hspace=0.28)
    out = "/data/rech/huiyuche/continual_ir/figures/training_curves_seaborn.png"
    g.figure.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
#  Style 2: Nature / IEEE paper style (serif, minimal, ~3.5" column width)
# ---------------------------------------------------------------------------
def plot_nature():
    # Reset rcParams to a clean state, then apply paper style
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "axes.titleweight":  "bold",
        "axes.labelweight":  "regular",
        "axes.linewidth":    0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.labelsize":   7.5,
        "ytick.labelsize":   7.5,
        "legend.fontsize":   7.5,
        "legend.frameon":    False,
        "lines.linewidth":   1.3,
        "lines.markersize":  3,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

    # IEEE/Nature double-column width ~7.2 inches, keep aspect compact
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4),
                             constrained_layout=True)
    panels = [
        ("Qwen3-0.6B", "TopiOCQA", axes[0, 0], "(a) Qwen3-0.6B \u00b7 TopiOCQA"),
        ("Qwen3-0.6B", "MS MARCO", axes[0, 1], "(b) Qwen3-0.6B \u00b7 MS MARCO"),
        ("ANCE",       "TopiOCQA", axes[1, 0], "(c) ANCE \u00b7 TopiOCQA"),
        ("ANCE",       "MS MARCO", axes[1, 1], "(d) ANCE \u00b7 MS MARCO"),
    ]
    for encoder, dataset, ax, title in panels:
        sub = df[(df.encoder == encoder) & (df.dataset == dataset)]
        for run in RUN_ORDER:
            s = sub[sub.run == run].sort_values("epoch")
            if s.empty:
                continue
            ls = "-" if run.startswith(("Baseline", "CL-")) else (0, (4, 1.5))
            ax.plot(s["epoch"], s["ndcg"],
                    color=PALETTE[run], linestyle=ls,
                    marker="o", markersize=2.5, markeredgewidth=0,
                    label=run)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NDCG@10")
        ax.set_ylim(*YLIM[(encoder, dataset)])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.4)

    # One shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), handlelength=2.4,
               columnspacing=1.2)
    out = "/data/rech/huiyuche/continual_ir/figures/training_curves_nature.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    # PDF too (journals want vector)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out.replace('.png', '.pdf')}")


# ---------------------------------------------------------------------------
#  Style 3: plotnine (ggplot2 grammar-of-graphics)
# ---------------------------------------------------------------------------
def plot_ggplot():
    from plotnine import (
        ggplot, aes, geom_line, geom_point, facet_grid, scale_color_manual,
        scale_linetype_manual, scale_x_continuous, labs, theme_bw, theme,
        element_text, element_rect, element_line, element_blank, guides,
        guide_legend,
    )

    df2 = df.copy()
    df2["run"] = pd.Categorical(df2["run"], categories=RUN_ORDER, ordered=True)
    df2["encoder"] = pd.Categorical(df2["encoder"],
                                    categories=["Qwen3-0.6B", "ANCE"], ordered=True)
    df2["dataset"] = pd.Categorical(df2["dataset"],
                                    categories=["TopiOCQA", "MS MARCO"], ordered=True)
    linetype_map = {r: ("solid" if not r.startswith("ACL") else "dashed")
                    for r in RUN_ORDER}

    p = (
        ggplot(df2, aes("epoch", "ndcg", color="run", linetype="run"))
        + geom_line(size=0.9)
        + geom_point(size=1.2, stroke=0)
        + facet_grid("encoder ~ dataset", scales="free")
        + scale_color_manual(values=PALETTE)
        + scale_linetype_manual(values=linetype_map)
        + scale_x_continuous(breaks=[0, 4, 8, 12, 16, 20])
        + labs(x="Epoch", y="NDCG@10", color=None, linetype=None)
        + theme_bw(base_size=11)
        + theme(
            figure_size=(11, 7.2),
            panel_grid_major=element_line(color="#e3e3e3", size=0.4),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(color="#bbbbbb", size=0.5),
            strip_background=element_rect(fill="#f3f4f6", color="#bbbbbb"),
            strip_text=element_text(size=11, weight="bold"),
            axis_title=element_text(size=11),
            axis_text=element_text(size=9, color="#333"),
            legend_position="bottom",
            legend_title=element_blank(),
            legend_key=element_rect(fill="white", color="white"),
            legend_background=element_rect(fill="white", color="white"),
            legend_box_spacing=0.25,
        )
        + guides(color=guide_legend(nrow=1), linetype=guide_legend(nrow=1))
    )
    out = "/data/rech/huiyuche/continual_ir/figures/training_curves_ggplot.png"
    p.save(filename=out, dpi=220, width=11, height=7.2, units="in", verbose=False)
    print(f"Saved: {out}")


plot_seaborn()
plot_nature()
plot_ggplot()
print("\nDone.")
