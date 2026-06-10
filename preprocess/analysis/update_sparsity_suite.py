"""
update_sparsity_suite.py
========================
Post-hoc parameter-update analysis over saved checkpoints, following the
measurement methodology of "Reinforcement Learning Finetunes Small Subnetworks
in Large Language Models" (arXiv 2505.11711):

  update sparsity(theta0, theta1) := 1 - ||theta1 - theta0||_0 / n,
  counted per SCALAR parameter in bf16, treating two values as equal when
  |a - b| <= --tolerance (default 1e-5, their default).

For each requested run it produces:
  - sparsity trajectory: overall sparsity of every checkpoint vs theta_base
    (their Fig. 5 analogue);
  - layerwise x matrix-type breakdown (q/k/v/o_proj, gate/up/down_proj,
    embed_tokens, *norm*) for the FINAL checkpoint (their Fig. 3 analogue);
  - (optional, --rank) effective rank of the per-matrix update DeltaW at the
    final checkpoint (their Tab. 2 analogue);
  - a bit-packed per-scalar update MASK of the final checkpoint, saved to
    disk, from which pairwise subnetwork overlaps o1/o2 (their Sec. 5):
        o1 = |I1 ∩ I2| / |I1|,   o2 = |I1 ∩ I2| / |I2|
    are computed across all requested runs.

Everything is streamed tensor-by-tensor via safetensors lazy loading — at no
point are two full models resident simultaneously.

Usage examples:
    # instruct3 family, all 20 epoch ckpts each:
    python preprocess/analysis/update_sparsity_suite.py \
        --runs instruct3_qwen_nosched instruct3_qwen_acl_step_excl \
        --steps 94:1880:94
    # turn-bucket runs (470-step batch):
    python preprocess/analysis/update_sparsity_suite.py \
        --runs bucket_qwen_turn_1 bucket_qwen_turn_5 bucket_qwen_turn_15plus \
        --steps 47:470:47 --rank
"""

import sys, os, json, argparse, logging
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CKPT_BASE  = "/data/rech/huiyuche/huggingface/continual_ir"
BASE_MODEL = ("/data/rech/huiyuche/huggingface/"
              "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
              "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
OUT_DIR    = "/data/rech/huiyuche/continual_ir/figures/analysis"

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True,
                help="run dir names under huggingface/continual_ir")
ap.add_argument("--steps", type=str, default="94:1880:94",
                help="start:stop:step for checkpoint-step-N dirs")
ap.add_argument("--tolerance", type=float, default=1e-5)
ap.add_argument("--rank", action="store_true",
                help="also compute effective rank of DeltaW per 2D matrix "
                     "(final ckpt; SVD-based, slow on CPU)")
ap.add_argument("--device", type=str,
                default="cuda" if torch.cuda.is_available() else "cpu",
                help="device for the elementwise comparisons / SVD")
args = ap.parse_args()

# "start:stop:step" with INCLUSIVE stop (94:1880:94 -> 94, 188, ..., 1880).
_start, _stop, _step = (int(x) for x in args.steps.split(":"))
STEPS = list(range(_start, _stop + 1, _step))


def matrix_type(name):
    """Map a parameter name to its coarse matrix type (their Fig. 3 legend)."""
    for key in ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"):
        if key in name:
            return key
    if "embed_tokens" in name:
        return "embed"
    if "norm" in name.lower():
        return "norm"
    return "other"


def layer_index(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1   # embeddings / final norm


def st_path(model_dir):
    return os.path.join(model_dir, "model.safetensors")


def iter_tensors(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            yield name, f.get_tensor(name)


def compare_ckpt(base_path, ckpt_path, device, tol, want_mask=False,
                 want_rank=False):
    """Stream-compare two safetensors files. Returns per-tensor stats and
    (optionally) bit-packed update masks / DeltaW ranks."""
    stats = {}
    masks = {} if want_mask else None
    ranks = {} if want_rank else None
    with safe_open(base_path, framework="pt", device="cpu") as fb, \
         safe_open(ckpt_path, framework="pt", device="cpu") as fc:
        names = list(fb.keys())
        for name in names:
            a = fb.get_tensor(name).to(device)
            b = fc.get_tensor(name).to(device)
            diff = (a.float() - b.float()).abs()
            updated = diff > tol
            n_upd = int(updated.sum().item())
            stats[name] = {"n": a.numel(), "updated": n_upd}
            if want_mask:
                masks[name] = np.packbits(
                    updated.flatten().cpu().numpy().astype(np.uint8))
            if want_rank and a.dim() == 2 and min(a.shape) > 1:
                delta = (b.float() - a.float())
                # effective rank = #singular values > rcond * smax (torch default tol)
                r = int(torch.linalg.matrix_rank(delta).item())
                ranks[name] = {"rank": r, "max_rank": min(a.shape)}
            del a, b, diff, updated
    return stats, masks, ranks


def aggregate(stats):
    total_n = sum(s["n"] for s in stats.values())
    total_u = sum(s["updated"] for s in stats.values())
    overall_sparsity = 1.0 - total_u / total_n
    by_type  = defaultdict(lambda: [0, 0])
    by_layer = defaultdict(lambda: [0, 0])
    for name, s in stats.items():
        t = matrix_type(name)
        by_type[t][0]  += s["updated"]; by_type[t][1]  += s["n"]
        l = layer_index(name)
        by_layer[l][0] += s["updated"]; by_layer[l][1] += s["n"]
    return {
        "overall_sparsity": overall_sparsity,
        "by_type":  {t: 1.0 - u / n for t, (u, n) in sorted(by_type.items())},
        "by_layer": {str(l): 1.0 - u / n for l, (u, n) in sorted(by_layer.items())},
    }


def overlap(mask_a, mask_b):
    """o1, o2 between two {name: packedbits} masks (their Sec. 5)."""
    i1 = i2 = inter = 0
    for name in mask_a:
        a = np.unpackbits(mask_a[name]).astype(bool)
        b = np.unpackbits(mask_b[name]).astype(bool)
        i1 += int(a.sum()); i2 += int(b.sum())
        inter += int((a & b).sum())
    return (inter / i1 if i1 else float("nan"),
            inter / i2 if i2 else float("nan"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base_st = st_path(BASE_MODEL)
    results = {}
    final_masks = {}

    for run in args.runs:
        run_dir = os.path.join(CKPT_BASE, run)
        traj = {}
        final_step = None
        for step in STEPS:
            ckpt = os.path.join(run_dir, f"checkpoint-step-{step}")
            if not os.path.isdir(ckpt):
                continue
            final_step = step
        if final_step is None:
            logger.warning(f"{run}: no checkpoints found for steps {STEPS} — skip")
            continue

        for step in STEPS:
            ckpt = os.path.join(run_dir, f"checkpoint-step-{step}")
            if not os.path.isdir(ckpt):
                continue
            is_final = (step == final_step)
            stats, masks, ranks = compare_ckpt(
                base_st, st_path(ckpt), args.device, args.tolerance,
                want_mask=is_final, want_rank=(is_final and args.rank))
            agg = aggregate(stats)
            traj[str(step)] = agg["overall_sparsity"]
            logger.info(f"{run} step-{step}: sparsity={agg['overall_sparsity']:.4f}")
            if is_final:
                results[run] = {
                    "trajectory": None,        # filled below
                    "final_step": step,
                    "final": agg,
                }
                final_masks[run] = masks
                if ranks:
                    rank_pct = [100.0 * v["rank"] / v["max_rank"]
                                for v in ranks.values()]
                    results[run]["final"]["mean_update_rank_pct"] = \
                        float(np.mean(rank_pct))
                    results[run]["ranks"] = {k: v for k, v in ranks.items()}
        results[run]["trajectory"] = traj

        # persist the final-ckpt mask for later cross-experiment overlaps
        mask_path = os.path.join(OUT_DIR, f"update_mask_{run}.npz")
        np.savez_compressed(mask_path, **final_masks[run])
        logger.info(f"{run}: final-ckpt update mask -> {mask_path}")

    # pairwise overlaps among the runs of THIS invocation
    runs_with_masks = list(final_masks.keys())
    overlaps = {}
    for i, ra in enumerate(runs_with_masks):
        for rb in runs_with_masks[i + 1:]:
            o1, o2 = overlap(final_masks[ra], final_masks[rb])
            overlaps[f"{ra}__VS__{rb}"] = {"o1": o1, "o2": o2}
            logger.info(f"overlap {ra} vs {rb}: o1={o1:.3f} o2={o2:.3f}")

    out = os.path.join(OUT_DIR, "update_sparsity_"
                       + "_".join(args.runs[:2])
                       + (f"_plus{len(args.runs)-2}" if len(args.runs) > 2 else "")
                       + ".json")
    with open(out, "w") as f:
        json.dump({"tolerance": args.tolerance, "runs": results,
                   "overlaps": overlaps}, f, indent=2)
    logger.info(f"results -> {out}")


if __name__ == "__main__":
    main()
