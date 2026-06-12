"""
plot_bucket_ance_beir.py
========================
ANCE turn-bucket MSMARCO + BEIR figures (counterparts of the Qwen
bucket_forgetting_curves / bucket_transfer_delta), from
figures/bucket_ance_beir_eval.json (produced by
eval_bucket_ance_beir_per_ckpt.py):

  bucket_ance_msmarco_forgetting.png  MSMARCO NDCG@10 vs step, 13 train buckets
  bucket_ance_beir_forgetting.png     BEIR-13 avg NDCG@10 vs step, 13 buckets
  bucket_ance_beir_transfer_delta.png 13 train-bucket x 13 BEIR-dataset
                                      Delta-vs-zero-shot heatmap (final ckpt)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "/data/rech/huiyuche/continual_ir/figures"
EVAL_JSON = f"{FIG_DIR}/bucket_ance_beir_eval.json"

BUCKETS = [f"bucket_ance_turn_{k}" for k in
           [*(str(t) for t in range(1, 11)), "11_12", "13_14", "15plus"]]
LABELS  = [*(str(t) for t in range(1, 11)), "11-12", "13-14", "15+"]
STEPS   = [47 * i for i in range(1, 11)]
BEIR13  = ["scifact", "trec-covid", "nfcorpus", "fiqa", "arguana",
           "webis-touche2020", "quora", "scidocs", "nq", "hotpotqa",
           "dbpedia-entity", "fever", "climate-fever"]

with open(EVAL_JSON) as f:
    D = json.load(f)
ZS = D["zero_shot"]["0"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def beir_avg(metric_dict):
    vals = [metric_dict[d] for d in BEIR13 if d in metric_dict]
    return sum(vals) / max(1, len(vals))


def forgetting_curve(metric_fn, ylabel, out, zs_val):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    cmap = plt.get_cmap("plasma")
    for i, b in enumerate(BUCKETS):
        ys = [metric_fn(D[b][str(s)]) for s in STEPS if str(s) in D.get(b, {})]
        xs = [s for s in STEPS if str(s) in D.get(b, {})]
        ax.plot([0] + xs, [zs_val] + ys, color=cmap(i / (len(BUCKETS) - 1)),
                lw=1.8, label=LABELS[i])
    ax.axhline(zs_val, color="gray", ls=":", lw=1)
    ax.text(2, zs_val, " zero-shot", color="gray", fontsize=10, va="bottom")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.legend(title="train bucket", ncol=4, fontsize=9, title_fontsize=10,
              loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


# ── 1 + 2: forgetting curves ─────────────────────────────────────────────────
forgetting_curve(lambda m: m["msmarco"], "MSMARCO NDCG@10",
                 f"{FIG_DIR}/bucket_ance_msmarco_forgetting.png", ZS["msmarco"])
forgetting_curve(beir_avg, "BEIR-13 avg NDCG@10",
                 f"{FIG_DIR}/bucket_ance_beir_forgetting.png", beir_avg(ZS))

# ── 3: BEIR transfer-delta heatmap (final ckpt, Delta vs zero-shot) ──────────
# Skip buckets whose final checkpoint isn't evaluated yet (partial-preview safe).
done_buckets = [(i, b) for i, b in enumerate(BUCKETS)
                if str(STEPS[-1]) in D.get(b, {})]
M = np.zeros((len(done_buckets), len(BEIR13)))
row_labels = []
for r, (i, b) in enumerate(done_buckets):
    fin = D[b][str(STEPS[-1])]
    row_labels.append(LABELS[i])
    for j, ds in enumerate(BEIR13):
        M[r, j] = (fin[ds] - ZS[ds]) * 100
if len(done_buckets) < len(BUCKETS):
    print(f"[partial] transfer_delta: {len(done_buckets)}/{len(BUCKETS)} buckets have a final ckpt")
vmax = np.abs(M).max()
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(BEIR13)), BEIR13, rotation=45, ha="right")
ax.set_yticks(range(len(row_labels)), row_labels)
ax.set_xlabel("BEIR dataset")
ax.set_ylabel("training bucket (conversation length)")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7,
                color="black")
ax.set_title(r"$\Delta$ BEIR NDCG@10 vs zero-shot ($\times$100) after 470 single-bucket steps")
fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\Delta$ NDCG@10 ($\times$100)")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/bucket_ance_beir_transfer_delta.png",
                                bbox_inches="tight"); plt.close(fig)
print(f"saved {FIG_DIR}/bucket_ance_beir_transfer_delta.png")
print("Done.")
