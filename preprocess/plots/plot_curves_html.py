"""
plot_curves_html.py
====================
Interactive Plotly 2x2 training-curve dashboard for the paper:
    (top row)    Qwen  TopiOCQA   |   Qwen  MSMARCO
    (bottom row) ANCE  TopiOCQA   |   ANCE  MSMARCO

Reads the same log files as plot_curves_combined.py. Emits a single
self-contained HTML: figures/training_curves_dashboard.html.
"""

import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


qwen_data = {k: parse_log(v) for k, v in QWEN_LOGS.items()}
ance_data = {k: parse_log(v) for k, v in ANCE_LOGS.items()}

# Palette: colorblind-safe, high-contrast between solid (CL) and dashed (ACL).
# Baseline = neutral slate so it doesn't dominate.
STYLE = {
    # key:                   (label,               color,     dash)
    "nosched":             ("Baseline",           "#2F4858", "solid"),
    "cl_step":             ("CL-step",            "#2E8B57", "solid"),
    "cl_step_excl":        ("CL-step-excl",       "#D99000", "solid"),
    "cl_step_excl_2_full": ("CL-step-excl-full",  "#C9186D", "solid"),
    "acl_step":            ("ACL-step",           "#1F9FB8", "dash"),
    "acl_step_excl":       ("ACL-step-excl",      "#8B5A3C", "dash"),
}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Qwen3-Embedding-0.6B</b> &mdash; TopiOCQA (new task)",
        "<b>Qwen3-Embedding-0.6B</b> &mdash; MS MARCO (previous task)",
        "<b>ANCE</b> &mdash; TopiOCQA (new task)",
        "<b>ANCE</b> &mdash; MS MARCO (previous task)",
    ),
    horizontal_spacing=0.09,
    vertical_spacing=0.14,
    shared_xaxes=False,
)

PANELS = [
    (qwen_data, "topiocqa_ndcg10", 1, 1),
    (qwen_data, "msmarco_ndcg10",  1, 2),
    (ance_data, "topiocqa_ndcg10", 2, 1),
    (ance_data, "msmarco_ndcg10",  2, 2),
]

for pi, (data, metric, r, c) in enumerate(PANELS):
    first_panel = (pi == 0)
    for key, (label, color, dash) in STYLE.items():
        ys = data[key][metric]
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                name=label,
                legendgroup=key,
                showlegend=first_panel,
                mode="lines+markers",
                line=dict(color=color, width=2.6, dash=dash, shape="spline", smoothing=0.3),
                marker=dict(size=5, color=color, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Epoch %{x}<br>"
                    "NDCG@10 = %{y:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=r, col=c,
        )

# Axis styling: subtle grid, no top/right spine, unified font
for r in (1, 2):
    for c in (1, 2):
        fig.update_xaxes(
            title_text="Epoch",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            showline=True, linewidth=1, linecolor="rgba(0,0,0,0.4)",
            mirror=False,
            dtick=4,
            row=r, col=c,
        )
        fig.update_yaxes(
            title_text="NDCG@10",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            showline=True, linewidth=1, linecolor="rgba(0,0,0,0.4)",
            mirror=False,
            tickformat=".3f",
            row=r, col=c,
        )

# Tighten the MS MARCO y-range on both rows so the small deltas are visible
fig.update_yaxes(range=[0.268, 0.335], row=1, col=2)
fig.update_yaxes(range=[0.284, 0.330], row=2, col=2)
# TopiOCQA: starts near 0.10-0.12, end ~0.20 (ANCE) / ~0.49 (Qwen)
fig.update_yaxes(range=[0.34, 0.50], row=1, col=1)
fig.update_yaxes(range=[0.09, 0.22], row=2, col=1)

fig.update_layout(
    template="plotly_white",
    height=760,
    width=1180,
    font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=13, color="#222"),
    title=dict(
        text="<b>Curriculum &amp; Anti-Curriculum Training Curves</b>"
             "<br><span style='font-size:13px;color:#666'>"
             "Per-epoch eval on TopiOCQA (new task) and MS MARCO (previous task). "
             "Solid = easy&rarr;hard (CL), dashed = hard&rarr;easy (ACL)."
             "</span>",
        x=0.5, xanchor="center",
        font=dict(size=19),
        y=0.97,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.10,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0)",
        font=dict(size=13),
        itemwidth=80,
    ),
    margin=dict(l=70, r=30, t=110, b=80),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font_size=12, font_family="Inter"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Subplot titles are an annotations list — bump them up slightly
for ann in fig["layout"]["annotations"]:
    ann["font"] = dict(size=14, color="#333")

out = "/data/rech/huiyuche/continual_ir/figures/training_curves_dashboard.html"
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print(f"Saved: {out}")
