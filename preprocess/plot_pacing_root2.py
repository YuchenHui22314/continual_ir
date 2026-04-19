"""
Plot root_2 pacing function curve.
Output: ../figures/root2_pacing.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

c0 = 0.2
end_epoch = 16
total_epochs = 20
epochs = np.arange(0, total_epochs + 1)


def pacing_root2(x, t, c0):
    x_clamped = np.minimum(x, t)
    return np.minimum(1.0, np.sqrt(x_clamped * (1 - c0**2) / t + c0**2))


pacing = pacing_root2(epochs, end_epoch, c0)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(epochs, pacing * 100, color='#2196F3', linewidth=2.5, marker='o', markersize=5, zorder=3)
ax.axvline(end_epoch, color='gray', linestyle='--', linewidth=1.2, label=f'curriculum end (epoch {end_epoch})')
ax.axhline(100, color='#ccc', linestyle=':', linewidth=1)

ax.annotate(f'start: {pacing[0]*100:.0f}%', xy=(0, pacing[0]*100),
            xytext=(0.5, pacing[0]*100 + 4), fontsize=9, color='#2196F3')
ax.annotate(f'epoch 8: {pacing[8]*100:.1f}%', xy=(8, pacing[8]*100),
            xytext=(8.3, pacing[8]*100 - 6), fontsize=9, color='#2196F3')
ax.annotate(f'epoch 16: {pacing[16]*100:.0f}%', xy=(16, pacing[16]*100),
            xytext=(16.1, pacing[16]*100 - 6), fontsize=9, color='gray')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Active training data (%)', fontsize=12)
ax.set_title('root_2 pacing function  (c₀=0.2, end_epoch=16, k=2)', fontsize=13)
ax.set_xlim(-0.5, 20.5)
ax.set_ylim(0, 110)
ax.set_xticks(range(0, 21, 2))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

import os
out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'root2_pacing.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

for e in [0, 4, 8, 12, 16, 17, 18, 19, 20]:
    print(f"  epoch {e:2d}: {pacing[e]*100:.1f}%")
