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

# ═════════════════════════════════════════════════════════════════════════════
# Second-generation figures (one-message-per-figure versions; the originals
# above are kept for the appendix).
# ═════════════════════════════════════════════════════════════════════════════

# ── 5. dose-response: MSMARCO drop vs training-conversation length ───────────
x = np.arange(len(BUCKETS))
drop470 = [(ZS["msmarco"] - D[f"bucket_qwen_{b}"]["470"]["msmarco"]) * 100 for b in BUCKETS]
drop47  = [(ZS["msmarco"] - D[f"bucket_qwen_{b}"]["47"]["msmarco"]) * 100 for b in BUCKETS]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(x, drop470, "o-",  color="#b02050", lw=2.2, ms=7, label="after 470 steps")
ax.plot(x, drop47,  "s--", color="#e58aa8", lw=1.8, ms=6, label="after 47 steps")
ax.set_xticks(x, LABELS)
ax.set_xlabel("training-conversation length (bucket)")
ax.set_ylabel("MS MARCO NDCG@10 drop (×100)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_dose_response.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_dose_response.png")

# ── 6. mechanism scatter: Fisher overlap vs measured forgetting ──────────────
try:
    with open(EWC_JSON) as f:
        ewc = json.load(f)
    ov = [ewc["fisher_overlap_vs_bucket"][b]["0.01"] for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(ov, drop470, c=np.arange(len(BUCKETS)), cmap="plasma",
                    s=80, zorder=3)
    for i, b in enumerate(LABELS):
        ax.annotate(b, (ov[i], drop470[i]), xytext=(6, 5),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("Fisher-mask overlap with MS MARCO (top-1%, base encoder)")
    ax.set_ylabel("MS MARCO NDCG@10 drop after 470 steps (×100)")
    cb = fig.colorbar(sc, ax=ax, ticks=[0, len(BUCKETS) - 1])
    cb.ax.set_yticklabels(["short", "long"])
    cb.set_label("conversation length")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bucket_overlap_vs_drop.png", bbox_inches="tight")
    plt.close(fig)
    print("saved bucket_overlap_vs_drop.png")
except FileNotFoundError:
    print("EWC json missing — skipped overlap-vs-drop scatter")

# ── 7. Δ-heatmap: transfer matrix RELATIVE TO ZERO-SHOT ──────────────────────
# Cell (i, j) = NDCG(train bucket i, eval turn j) − zero-shot(eval turn j):
# blue = trained model HELPS that eval length, red = HURTS it. Removes the
# per-column baseline difficulty that dominates the absolute heatmap.
Md = (M[1:, :] - M[0:1, :]) * 100
vmax = np.abs(Md).max()
fig, ax = plt.subplots(figsize=(10.5, 8.5))
im = ax.imshow(Md, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(LABELS)), LABELS)
ax.set_yticks(range(len(LABELS)), LABELS)
ax.set_xlabel("eval turn length")
ax.set_ylabel("training bucket (conversation length)")
for i in range(Md.shape[0]):
    for j in range(Md.shape[1]):
        ax.text(j, i, f"{Md[i, j]:+.0f}", ha="center", va="center",
                color="black", fontsize=9)
for k in range(len(BUCKETS)):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.6))
ax.set_title("Δ TopiOCQA NDCG@10 vs zero-shot (×100) after 470 single-bucket steps")
fig.colorbar(im, ax=ax, shrink=0.8, label="Δ NDCG@10 (×100)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_transfer_delta.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_transfer_delta.png")

# ── 8. generalization profiles: Δ vs eval length, representative buckets ─────
REPS = ["turn_1", "turn_3", "turn_5", "turn_10", "turn_15plus"]
REP_COLORS = {"turn_1": "#b02050", "turn_3": "#e8a13a", "turn_5": "#4caf73",
              "turn_10": "#0097a7", "turn_15plus": "#5e35b1"}
fig, ax = plt.subplots(figsize=(9, 5.2))
for b in REPS:
    i = BUCKETS.index(b)
    ax.plot(range(len(LABELS)), Md[i, :], "o-", lw=2,
            color=REP_COLORS[b], label=f"train {LABELS[i]}")
ax.axhline(0, color="gray", lw=1, ls=":")
ax.set_xticks(range(len(LABELS)), LABELS)
ax.set_xlabel("eval turn length")
ax.set_ylabel("Δ NDCG@10 vs zero-shot (×100)")
ax.legend(frameon=False, title="training bucket", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/bucket_generalization_profiles.png", bbox_inches="tight")
plt.close(fig)
print("saved bucket_generalization_profiles.png")

print("Done.")
