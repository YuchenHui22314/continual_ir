"""
Plot all 4 curriculum pacing schedules side by side:
  root_2, step, step_exclusive, step_exclusive_2_full
Output: ../figures/curriculum_pacing_all4.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Settings
num_epochs = 20
end_epoch  = 16
c0         = 0.2
mid        = c0 + (1.0 - c0) / 2.0  # 0.6

epochs = np.arange(num_epochs)
curriculum_steps = end_epoch


def pacing_root2(x, t, c0):
    x = min(x, t)
    return ((x * (1 - c0**2) / t) + c0**2) ** 0.5


def pacing_step(x, t, c0):
    mid = c0 + (1.0 - c0) / 2.0
    x = min(x, t)
    if x <= t * 0.33:   return c0
    elif x <= t * 0.66: return mid
    else:               return 1.0


def pacing_step_exclusive(x, t, c0):
    mid = c0 + (1.0 - c0) / 2.0
    if x <= t * 0.33:   return (0.0, c0)
    elif x <= t * 0.66: return (c0, mid)
    else:               return (mid, 1.0)


def pacing_step_exclusive_2_full(x, t, c0):
    mid = c0 + (1.0 - c0) / 2.0
    if x >= t:           return (0.0, 1.0)
    elif x <= t * 0.33:  return (0.0, c0)
    elif x <= t * 0.66:  return (c0, mid)
    else:                return (mid, 1.0)


root2_vals = [pacing_root2(e, curriculum_steps, c0) for e in epochs]
step_vals  = [pacing_step(e, curriculum_steps, c0)  for e in epochs]
excl_vals  = [pacing_step_exclusive(e, curriculum_steps, c0) for e in epochs]
excl2_vals = [pacing_step_exclusive_2_full(e, curriculum_steps, c0) for e in epochs]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

colors = {
    'root_2':           '#2196F3',
    'step':             '#4CAF50',
    'step_exclusive':   '#FF9800',
    'step_excl_2_full': '#E91E63',
}

titles = ['root_2\n(cumulative, smooth)', 'step\n(cumulative, 3-stage)',
          'step_exclusive\n(exclusive slice)', 'step_exclusive_2_full\n(exclusive + full)']
fns    = [root2_vals, step_vals, excl_vals, excl2_vals]
clrs   = [colors['root_2'], colors['step'], colors['step_exclusive'], colors['step_excl_2_full']]

for ax, title, fn_vals, clr in zip(axes, titles, fns, clrs):
    ax.set_xlim(-0.5, num_epochs - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.axvline(x=end_epoch - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label=f'end_epoch={end_epoch}')
    ax.set_xticks([0, 4, 8, 12, 16, 19])
    ax.yaxis.set_tick_params(labelsize=9)
    ax.xaxis.set_tick_params(labelsize=9)

    for e, v in enumerate(fn_vals):
        if isinstance(v, tuple):
            lo, hi = v
        else:
            lo, hi = 0.0, v
        ax.barh(y=(lo + hi) / 2, width=0.8, height=(hi - lo),
                left=e - 0.4, color=clr, alpha=0.75, align='center')
        if lo > 0:
            ax.barh(y=lo / 2, width=0.8, height=lo,
                    left=e - 0.4, color='#EEEEEE', alpha=0.9, align='center', zorder=2)
        if hi < 1.0:
            ax.barh(y=(hi + 1.0) / 2, width=0.8, height=(1.0 - hi),
                    left=e - 0.4, color='#EEEEEE', alpha=0.9, align='center', zorder=2)

    for yval in [c0, mid, 1.0]:
        ax.axhline(y=yval, color='black', linestyle=':', alpha=0.3, linewidth=0.8)

    if not isinstance(fn_vals[0], tuple):
        ax.step(np.arange(num_epochs), fn_vals, where='post',
                color=clr, linewidth=2.0, zorder=5)

axes[0].set_ylabel('Dataset fraction', fontsize=11)
for ax in axes:
    ax.set_yticks([0, c0, mid, 1.0])
    ax.set_yticklabels(['0%', f'{c0*100:.0f}%', f'{mid*100:.0f}%', '100%'])

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'curriculum_pacing_all4.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
