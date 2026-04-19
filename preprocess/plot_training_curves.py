"""
Plot TopiOCQA NDCG@10 training curves from wandb CSV export.
No title. Inline labels. Colors match pacing visualization.
Output: ../figures/training_curves.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- data ----------
CSV = os.path.join(os.path.dirname(__file__), '..', 'figures',
                   'wandb_export_2026-04-13T23_10_21.038-04_00.csv')
df = pd.read_csv(CSV)
steps = df['Step'].astype(float)
STEPS_PER_EPOCH = 94.0
epochs = steps / STEPS_PER_EPOCH

# ---------- visual settings ----------
# colors aligned with plot_pacing_all4.py
COLORS = {
    'nosched':               '#546E7A',   # blue-gray baseline
    'step':                  '#4CAF50',   # green
    'step_exclusive':        '#FF9800',   # orange
    'step_exclusive_2_full': '#E91E63',   # pink
}
LABELS = {
    'nosched':               'Baseline',
    'step':                  'CL-step',
    'step_exclusive':        'CL-step-excl',
    'step_exclusive_2_full': 'CL-step-excl-full',
}
# (key → column prefix)
COL = {
    'step_exclusive_2_full': 'ance_curriculum_step_exclusive_2_full - eval/topiocqa_NDCG@10',
    'step_exclusive':        'ance_curriculum_step_exclusive - eval/topiocqa_NDCG@10',
    'step':                  'ance_curriculum_step - eval/topiocqa_NDCG@10',
    'nosched':               'ance_topiocqa_nosched - eval/topiocqa_NDCG@10',
}

# draw order: nosched first so curriculum runs sit on top
DRAW_ORDER = ['nosched', 'step', 'step_exclusive', 'step_exclusive_2_full']

# ---------- label positions (epoch, ndcg offset) ----------
# place labels at a point where lines are well separated
LABEL_AT_EPOCH = {
    'nosched':               4.0,     # early, clearly above curriculum
    'step':                  11.0,
    'step_exclusive':        13.5,
    'step_exclusive_2_full': 2.0,    # lower-left flat region
}
LABEL_YOFFSET = {          # fine-tune vertical shift (in NDCG units)
    'nosched':               0.004,
    'step':                 -0.006,
    'step_exclusive':       -0.006,
    'step_exclusive_2_full':-0.007,  # sit just below the flat cluster
}

# ---------- plot ----------
sns.set_theme(style='ticks', font_scale=1.35)
fig, ax = plt.subplots(figsize=(9, 5))

series_cache = {}
for key in DRAW_ORDER:
    raw = df[COL[key]]
    mask = raw.notna() & (raw.astype(str) != '')
    x = epochs[mask].values
    y = raw[mask].astype(float).values
    series_cache[key] = (x, y)

# truncate all series to the length of the shortest one
min_len = min(len(v[0]) for v in series_cache.values())
series_cache = {k: (x[:min_len], y[:min_len]) for k, (x, y) in series_cache.items()}

for key in DRAW_ORDER:
    x, y = series_cache[key]
    ax.plot(x, y,
            color=COLORS[key],
            linewidth=2.4,
            alpha=0.92,
            zorder=3 if key != 'nosched' else 2)

# inline labels
for key in DRAW_ORDER:
    x, y = series_cache[key]
    target_epoch = LABEL_AT_EPOCH[key]
    # find closest index
    idx = int(np.argmin(np.abs(x - target_epoch)))
    lx, ly = x[idx], y[idx] + LABEL_YOFFSET[key]
    ax.text(lx, ly, LABELS[key],
            color=COLORS[key],
            fontsize=12.5,
            fontweight='bold',
            va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

# axis labels
ax.set_xlabel('Epoch', fontsize=15, labelpad=6)
ax.set_ylabel('NDCG@10', fontsize=15, labelpad=6)
ax.set_xlim(0, 20.5)
ax.set_ylim(0.08, 0.215)
ax.set_xticks(range(0, 21, 2))
ax.tick_params(labelsize=12)

sns.despine(offset=6)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'figures', 'training_curves.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
