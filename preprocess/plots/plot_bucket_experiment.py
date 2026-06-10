"""
plot_bucket_experiment.py
=========================
Figures for the per-turn-length forgetting experiment (Part D):

  bucket_transfer_matrix.png   13 train-bucket x 13 eval-turn NDCG@10 heatmap
                               at step 470, with the zero-shot row appended —
                               cell (i,j) answers "training only on length i,
                               how do length-j queries fare?"
  bucket_forgetting_curves.png MSMARCO NDCG@10 vs step, one line per bucket
                               (the causal version of Figure 4 panel h).
  bucket_inlength_gain.png     in-length gain (diag - zero-shot diag) and
                               MSMARCO drop vs bucket length, twin axes.
  bucket_fisher_overlap.png    top-p% Fisher-mask overlap F_bucket vs F_old
                               as a function of bucket length.

Inputs: figures/bucket_runs_eval.json (B5 offline eval),
        figures/analysis/ewc_forgetting_predictor.json (overlaps; optional).
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "/data/rech/huiyuche/continual_ir/figures"
EVAL_JSON = f"{FIG_DIR}/bucket_runs_eval.json"
EWC_JSON  = f"{FIG_DIR}/analysis/ewc_forgetting_predictor.json"

BUCKETS = [*(f"turn_{t}" for t in range(1, 11)), "turn_11_12", "turn_13_14", "turn_15plus"]
LABELS  = [*(str(t) for t in range(1, 11)), "11-12", "13-14", "15+"]
STEPS   = [47 * i for i in range(1, 11)]

with open(EVAL_JSON) as f:
    D = json.load(f)
ZS = D["zero_shot"]["0"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

# ── 1. transfer matrix ───────────────────────────────────────────────────────
M = np.zeros((len(BUCKETS) + 1, len(BUCKETS)))
for j, eb in enumerate(BUCKETS):
    M[0, j] = ZS["topiocqa"][eb]["NDCG@10"]
for i, tb in enumerate(BUCKETS):
    e = D[f"bucket_qwen_{tb}"]["470"]["topiocqa"]
    for j, eb in enumerate(BUCKETS):
        M[i + 1, j] = e[eb]["NDCG@10"]

fig, ax = plt.subplots(figsize=(10.5, 9))
im = ax.imshow(M, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(LABELS)), LABELS)
ax.set_yticks(range(len(LABELS) + 1), ["zero-shot"] + LABELS)
ax.set_xlabel("eval turn length")
ax.set_ylabel("training bucket (conversation length)")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center",
                color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=9)
# mark the in-length diagonal
for k in range(len(BUCKETS)):
    ax.add_patch(plt.Rectangle((k - 0.5, k + 0.5), 1, 1, fill=False,
                               edgecolor="red", lw=1.6))
ax.set_title("TopiOCQA NDCG@10 (x100) after 470 steps on a single length bucket")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_transfer_matrix.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_transfer_matrix.png")

# ── 2. forgetting curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.5))
cmap = plt.get_cmap("plasma")
for i, b in enumerate(BUCKETS):
    ys = [D[f"bucket_qwen_{b}"][str(s)]["msmarco"] for s in STEPS]
    ax.plot([0] + STEPS, [ZS["msmarco"]] + ys,
            color=cmap(i / (len(BUCKETS) - 1)), lw=1.8,
            label=LABELS[i])
ax.axhline(ZS["msmarco"], color="gray", ls=":", lw=1)
ax.text(2, ZS["msmarco"] + 0.0006, "zero-shot", color="gray", fontsize=10)
ax.set_xlabel("optimizer step")
ax.set_ylabel("MSMARCO NDCG@10")
ax.legend(title="train bucket", ncol=4, fontsize=9, title_fontsize=10,
          loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_forgetting_curves.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_forgetting_curves.png")

# ── 3. in-length gain + MSMARCO drop vs length ───────────────────────────────
gain = [D[f"bucket_qwen_{b}"]["470"]["topiocqa"][b]["NDCG@10"]
        - ZS["topiocqa"][b]["NDCG@10"] for b in BUCKETS]
drop = [ZS["msmarco"] - D[f"bucket_qwen_{b}"]["470"]["msmarco"] for b in BUCKETS]
x = np.arange(len(BUCKETS))
fig, ax1 = plt.subplots(figsize=(9.5, 5))
ax1.bar(x - 0.18, np.array(gain) * 100, width=0.36, color="#4caf73",
        label="in-length TopiOCQA gain")
ax1.set_ylabel("in-length NDCG@10 gain (x100)", color="#2e7d4f")
ax1.set_xticks(x, LABELS)
ax1.set_xlabel("training bucket (conversation length)")
ax2 = ax1.twinx()
ax2.bar(x + 0.18, np.array(drop) * 100, width=0.36, color="#e8306a",
        label="MSMARCO drop")
ax2.set_ylabel("MSMARCO NDCG@10 drop (x100)", color="#b02050")
ax2.spines["right"].set_visible(True)
fig.legend(loc="upper center", ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_inlength_gain.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_inlength_gain.png")

# ── 4. Fisher overlap vs length ──────────────────────────────────────────────
try:
    with open(EWC_JSON) as f:
        ewc = json.load(f)
    ov = ewc["fisher_overlap_vs_bucket"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for p, marker in zip(["0.01", "0.05", "0.2"], ["o", "s", "^"]):
        ys = [ov[b][p] for b in BUCKETS if b in ov]
        xs = [LABELS[i] for i, b in enumerate(BUCKETS) if b in ov]
        ax.plot(xs, ys, marker=marker, lw=1.8, label=f"top-{float(p)*100:.0f}%")
    ax.set_xlabel("bucket (conversation length)")
    ax.set_ylabel("Fisher-mask overlap with MSMARCO (F_old)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bucket_fisher_overlap.png", bbox_inches="tight")
    plt.close(fig)
    print("saved bucket_fisher_overlap.png")
except FileNotFoundError:
    print("EWC json not found yet — skipping fisher overlap plot")

print("Done.")
