"""
gradient_direction_analysis.py
==============================
Mechanistic "why is turn_1 toxic" analysis for the turn-bucket experiment.
All inputs are on disk (final checkpoints, recorded sum_g/sum_g2, MSMARCO
Fisher); this is a CPU-only post-hoc analysis and does not touch the GPUs.

For each bucket it answers two questions about the realised update
Delta-theta = theta_final - theta_base and the recorded training gradients:

  (A) DIRECTION — do different conversation lengths push the model in
      different directions? -> 13x13 cosine-similarity matrix of the flattened
      Delta-theta vectors (restricted to the shared updated subnetwork is not
      needed; cosine over the full vector already captures it).

  (B) MSMARCO-SUBSPACE OVERLAP of the actual update — what fraction of each
      bucket's update ENERGY ||Delta-theta||^2 lands on MSMARCO's most-important
      parameters (top-p% of F_old)? The format-overlap hypothesis predicts this
      fraction is highest for short conversations. This is the realised-update
      analogue of the base-model Fisher-mask overlap, and a candidate
      forgetting predictor to compare against the (weak, rho=0.38) EWC
      quadratic.

  (C) SIGN COHERENCE per scalar = |sum_g| / sqrt(T * sum_g2) in [0,1]: 1 = the
      parameter was pushed consistently one way every step, ~0 = oscillating /
      cancelling updates. Reported as a mean overall and a mean restricted to
      the top-1% MSMARCO-Fisher parameters -> does turn_1 drive the MSMARCO
      parameters with unusually coherent (one-directional) gradients?

Outputs figures/analysis/gradient_direction_analysis.json and three PNGs.

Usage:
    python preprocess/analysis/gradient_direction_analysis.py
"""

import os, json, glob
import numpy as np
import torch
from safetensors import safe_open

CKPT_BASE  = "/data/rech/huiyuche/huggingface/continual_ir"
BASE_ST    = ("/data/rech/huiyuche/huggingface/"
              "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
              "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418/model.safetensors")
ANALYSIS   = "/data/rech/huiyuche/continual_ir/figures/analysis"
FIG_DIR    = "/data/rech/huiyuche/continual_ir/figures"
EVAL_JSON  = f"{FIG_DIR}/bucket_runs_eval.json"
FISHER_OLD = f"{ANALYSIS}/fisher_msmarco.pt"

BUCKETS = [*(f"turn_{t}" for t in range(1, 11)), "turn_11_12", "turn_13_14", "turn_15plus"]
LABELS  = [*(str(t) for t in range(1, 11)), "11-12", "13-14", "15+"]
FINAL_STEP = 470
TOP_PS = [0.01, 0.05, 0.20]

DEV = "cpu"


# ── helpers ──────────────────────────────────────────────────────────────────
def ckpt_st(bucket):
    return os.path.join(CKPT_BASE, f"bucket_qwen_{bucket}",
                        f"checkpoint-step-{FINAL_STEP}", "model.safetensors")


def load_delta_flat(bucket, name_order):
    """Flatten theta_final - theta_base into one vector (float32), in name_order."""
    parts = []
    with safe_open(BASE_ST, framework="pt", device="cpu") as fb, \
         safe_open(ckpt_st(bucket), framework="pt", device="cpu") as fc:
        for name in name_order:
            a = fb.get_tensor(name).float()
            b = fc.get_tensor(name).float()
            parts.append((b - a).flatten())
    return torch.cat(parts)


def load_fold_flat(name_order):
    blob = torch.load(FISHER_OLD, map_location="cpu", weights_only=False)
    f = blob["fisher"]
    # F_old keys may be prefixed 'model.' relative to the safetensors names.
    def lookup(n):
        if n in f: return f[n]
        alt = n[len("model."):] if n.startswith("model.") else f"model.{n}"
        return f.get(alt)
    parts = []
    for name in name_order:
        t = lookup(name)
        parts.append((t.float().flatten() if t is not None
                      else torch.zeros(_numel[name])))
    return torch.cat(parts)


def _resolve(d, name):
    """The grad-stat dicts are keyed by the QwenQueryEncoder.named_parameters()
    names (prefixed 'model.'), the safetensors NAME_ORDER is not. Try both."""
    if name in d: return d[name]
    for alt in (f"model.{name}", name[len("model."):] if name.startswith("model.") else None):
        if alt and alt in d: return d[alt]
    raise KeyError(name)


def load_sumg_flat(bucket, name_order):
    gs = os.path.join(CKPT_BASE, f"bucket_qwen_{bucket}", f"grad_stats-step-{FINAL_STEP}")
    sg = torch.load(f"{gs}/sum_g.pt", map_location="cpu", weights_only=False)
    sg2 = torch.load(f"{gs}/sum_g2.pt", map_location="cpu", weights_only=False)
    T = torch.load(f"{gs}/meta.pt", map_location="cpu", weights_only=False)["n_steps"]
    g = torch.cat([_resolve(sg, n).float().flatten() for n in name_order])
    g2 = torch.cat([_resolve(sg2, n).float().flatten() for n in name_order])
    return g, g2, T


# ── parameter name order + sizes (from the base model) ───────────────────────
with safe_open(BASE_ST, framework="pt", device="cpu") as fb:
    NAME_ORDER = list(fb.keys())
    _numel = {n: fb.get_tensor(n).numel() for n in NAME_ORDER}
N = sum(_numel.values())
print(f"{len(NAME_ORDER)} tensors, {N/1e6:.1f}M scalars")

# F_old top-p% masks (boolean over the flat vector)
fold = load_fold_flat(NAME_ORDER)
fold_top = {}
for p in TOP_PS:
    k = max(1, int(p * fold.numel()))
    idx = torch.topk(fold, k).indices
    m = torch.zeros(fold.numel(), dtype=torch.bool)
    m[idx] = True
    fold_top[p] = m
    print(f"  F_old top-{p*100:.0f}% mask: {int(m.sum())} params")

# ── per-bucket metrics ───────────────────────────────────────────────────────
deltas = {}          # bucket -> normalised Delta-theta (for cosine matrix)
records = {}
for b in BUCKETS:
    d = load_delta_flat(b, NAME_ORDER)
    energy = float((d * d).sum())
    rec = {"delta_norm": energy ** 0.5}
    # (B) fraction of update energy in F_old top-p%
    for p in TOP_PS:
        rec[f"msmarco_energy_frac_top{int(p*100)}"] = float((d[fold_top[p]] ** 2).sum() / energy)
    # (C) sign coherence overall + on MSMARCO top-1%
    g, g2, T = load_sumg_flat(b, NAME_ORDER)
    coh = g.abs() / (T * g2).clamp_min(1e-20).sqrt()   # in [0,1] per scalar
    moved = g2 > 0                                       # params that ever moved
    rec["coherence_mean"] = float(coh[moved].mean())
    rec["coherence_msmarco_top1"] = float(coh[fold_top[0.01] & moved].mean())
    records[b] = rec
    deltas[b] = (d / (energy ** 0.5)).numpy()           # unit vector
    print(f"  {b:<12} |Δθ|={rec['delta_norm']:.3f}  "
          f"E@MSMARCO(top1%)={rec['msmarco_energy_frac_top1']*100:.2f}%  "
          f"coh_all={rec['coherence_mean']:.3f}  coh_MS={rec['coherence_msmarco_top1']:.3f}")

# 13x13 cosine similarity of unit Delta-theta vectors
COS = np.zeros((len(BUCKETS), len(BUCKETS)))
for i, bi in enumerate(BUCKETS):
    for j, bj in enumerate(BUCKETS):
        COS[i, j] = float(np.dot(deltas[bi], deltas[bj]))

# ── correlate MSMARCO-energy-fraction with measured forgetting ───────────────
ev = json.load(open(EVAL_JSON))
zs_ms = ev["zero_shot"]["0"]["msmarco"]
drop = [zs_ms - ev[f"bucket_qwen_{b}"][str(FINAL_STEP)]["msmarco"] for b in BUCKETS]
frac = [records[b]["msmarco_energy_frac_top1"] for b in BUCKETS]
def spear(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx*ry).sum()/np.sqrt((rx**2).sum()*(ry**2).sum()))
rho_frac = spear(frac, drop)
print(f"\nSpearman(MSMARCO top-1% update-energy fraction, measured drop) = {rho_frac:.3f}")

out = {"buckets": records, "cosine_matrix": COS.tolist(),
       "spearman_energyfrac_drop": rho_frac,
       "msmarco_drop": dict(zip(BUCKETS, drop))}
json.dump(out, open(f"{ANALYSIS}/gradient_direction_analysis.json", "w"), indent=2)
print(f"saved {ANALYSIS}/gradient_direction_analysis.json")

# ── figures ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

# fig 1: update-energy-fraction in MSMARCO params vs length (+ coherence)
x = np.arange(len(BUCKETS))
fig, ax = plt.subplots(figsize=(9, 5))
for p, c in zip(TOP_PS, ["#b02050", "#e8801a", "#4caf73"]):
    ax.plot(x, [records[b][f"msmarco_energy_frac_top{int(p*100)}"]*100 for b in BUCKETS],
            "-o", color=c, label=f"top-{int(p*100)}% F_old")
ax.set_xticks(x, LABELS); ax.set_xlabel("training bucket (conversation length)")
ax.set_ylabel(r"% of $\|\Delta\theta\|^2$ on MSMARCO-important params")
ax.legend(frameon=False, title="MSMARCO importance set")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/bucket_update_energy_msmarco.png", bbox_inches="tight")
plt.close(fig); print("saved bucket_update_energy_msmarco.png")

# fig 2: 13x13 cosine similarity of update directions
fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(COS, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(x, LABELS); ax.set_yticks(x, LABELS)
ax.set_xlabel("training bucket"); ax.set_ylabel("training bucket")
for i in range(len(BUCKETS)):
    for j in range(len(BUCKETS)):
        ax.text(j, i, f"{COS[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(COS[i,j]) > 0.6 else "black", fontsize=7)
ax.set_title(r"cosine similarity of update directions $\Delta\theta$")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/bucket_update_cosine.png", bbox_inches="tight")
plt.close(fig); print("saved bucket_update_cosine.png")

# fig 3: forgetting vs MSMARCO-update-energy-fraction scatter
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sc = ax.scatter([f*100 for f in frac], [dd*100 for dd in drop],
                c=x, cmap="plasma", s=80, zorder=3)
for i, b in enumerate(LABELS):
    ax.annotate(b, (frac[i]*100, drop[i]*100), xytext=(5, 4),
                textcoords="offset points", fontsize=9)
ax.set_xlabel(r"% of $\|\Delta\theta\|^2$ on MSMARCO top-1% params")
ax.set_ylabel("MSMARCO NDCG@10 drop (×100)")
ax.set_title(f"Spearman = {rho_frac:.2f}")
cb = fig.colorbar(sc, ax=ax, ticks=[0, len(BUCKETS)-1]); cb.ax.set_yticklabels(["short","long"])
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/bucket_energyfrac_vs_drop.png", bbox_inches="tight")
plt.close(fig); print("saved bucket_energyfrac_vs_drop.png")

print("Done.")
