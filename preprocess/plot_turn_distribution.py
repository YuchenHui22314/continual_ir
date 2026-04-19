"""
Plot TopiOCQA turn-length distribution with step pacing cut annotations.
Secondary axis: step pacing schedule (3-stage step curve). Cut lines at c0=20% and mid=60%.
Output: ../figures/turn_distribution.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------- data (from analyze_topiocqa_turns.py output) ----------
turns  = list(range(1, 26))
counts = [3509,3509,3509,3509,3509,3462,3421,3374,3333,3285,
          2472,2025,1650,1404,1191, 880, 678, 545, 109,  56,
            11,   5,   2,   1,   1]
total  = sum(counts)

cumul  = np.cumsum(counts) / total * 100   # cumulative % (used only to find cut positions)

# ---------- step pacing thresholds ----------
c0  = 0.20   # stage 1 → 2 cut
mid = 0.60   # stage 2 → 3 cut

# Find fractional x positions of the cuts in bar-chart coords
# Bar for turn k is centered at k, spanning [k-0.5, k+0.5]
def cut_xpos(threshold_pct):
    """Return the x position (in turn units) where cumulative% crosses threshold."""
    pct = threshold_pct * 100
    for i, (c, cnt) in enumerate(zip(cumul, counts)):
        if c >= pct:
            prev = cumul[i-1] if i > 0 else 0.0
            frac = (pct - prev) / (c - prev)   # fraction into this bar
            return turns[i] - 0.5 + frac       # left edge + fraction
    return turns[-1] + 0.5

x_cut1 = cut_xpos(c0)   # ~3.1
x_cut2 = cut_xpos(mid)  # ~8.3

# epoch labels for each stage  (end_epoch=16, t*0.33=5.3, t*0.66=10.6)
stage_labels = [
    (x_cut1 / 2,               'Stage 1\n(epoch 0–5)',   '#4CAF50', 0.97),  # green
    ((x_cut1 + x_cut2) / 2,    'Stage 2\n(epoch 5–11)',  '#FF9800', 0.97),  # orange
    ((x_cut2 + turns[-1]) / 2, 'Stage 3\n(epoch 11–16)', '#9E9E9E', 0.55),  # gray — moved down
]

# ---------- plot ----------
sns.set_theme(style='ticks', font_scale=1.3)
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

# background shading
ax1.axvspan(0.5,      x_cut1,         color='#4CAF50', alpha=0.10, zorder=0)
ax1.axvspan(x_cut1,   x_cut2,         color='#FF9800', alpha=0.10, zorder=0)
ax1.axvspan(x_cut2,   turns[-1]+0.5,  color='#9E9E9E', alpha=0.08, zorder=0)

# bars
ax1.bar(turns, counts, color='#546E7A', alpha=0.82, width=0.7, zorder=2)

# step pacing curve on secondary axis
# stage 1: 20%  (turns 0.5 → x_cut1)
# stage 2: 60%  (x_cut1 → x_cut2)
# stage 3: 100% (x_cut2 → turns[-1]+0.5)
step_x = [0.5,    x_cut1, x_cut1, x_cut2, x_cut2, turns[-1]+0.5]
step_y = [c0*100, c0*100, mid*100, mid*100, 100.0,  100.0]
step_clr = '#37474F'
ax2.plot(step_x, step_y, color=step_clr, linewidth=2.2, zorder=4,
         solid_capstyle='butt', solid_joinstyle='miter')

# cut lines
for xc, label_pct, clr in [(x_cut1, c0*100, '#4CAF50'), (x_cut2, mid*100, '#FF9800')]:
    ax1.axvline(xc, color=clr, linestyle='--', linewidth=1.6, zorder=3)
    ax2.axhline(label_pct, color=clr, linestyle=':', linewidth=1.2, alpha=0.7, zorder=3)

# stage labels at top of plot
ymax_counts = max(counts) * 1.12
for xc, label, clr, yfrac in stage_labels:
    ax1.text(xc, ymax_counts * yfrac, label,
             color=clr, fontsize=12, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

# axes formatting
ax1.set_xlabel('Conversation turn', fontsize=14, labelpad=6)
ax1.set_ylabel('# examples', fontsize=14, labelpad=6)
ax2.set_ylabel('Active data (step pacing)', fontsize=14, labelpad=6, color=step_clr)
ax2.tick_params(axis='y', labelcolor=step_clr)
ax2.set_ylim(0, 108)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
ax1.set_xlim(0.5, turns[-1] + 0.5)
ax1.set_xticks(turns)
ax1.set_ylim(0, ymax_counts)
ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)

# color the 20% and 60% tick labels on the secondary axis
tick_colors = {20: '#4CAF50', 60: '#FF9800', 100: step_clr}
fig.canvas.draw()
for label in ax2.get_yticklabels():
    try:
        val = int(label.get_text().replace('%', ''))
        if val in tick_colors:
            label.set_color(tick_colors[val])
            label.set_fontweight('bold')
    except ValueError:
        pass

sns.despine(ax=ax1, right=False)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'figures', 'turn_distribution.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
