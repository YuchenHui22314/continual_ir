"""
ewc_forgetting_predictor.py
===========================
Two analyses linking parameter updates to IR catastrophic forgetting:

1. EWC-style forgetting prediction (Kirkpatrick et al., PNAS 2017):
       predicted_forgetting(run, step) = sum_i F_old_i * (theta_step - theta_0)_i^2
   computed per (run, checkpoint) by streaming Delta-theta against the saved
   F_old = fisher_msmarco.pt (from fisher_importance.py --data msmarco), then
   correlated (Spearman) with the MEASURED MSMARCO NDCG@10 drop of the same
   checkpoint from the offline eval JSON. If the diagonal-Fisher quadratic
   form explains the measured forgetting across runs x checkpoints, the
   parameter-level account of forgetting holds.

2. Fisher-mask overlap vs conversation length:
       overlap(p, k) = |top-p%(F_bucket_k)  ∩  top-p%(F_old)| / |top-p%|
   for each turn-bucket k. The format-overlap hypothesis predicts overlap
   decreases with conversation length (short conversations recruit the
   MSMARCO-aligned parameters).

Usage (after fisher_importance.py runs and the offline bucket eval exists):
    python preprocess/analysis/ewc_forgetting_predictor.py \
        --runs bucket_qwen_turn_1 ... --steps 47:470:47 \
        --eval_json figures/bucket_runs_eval.json
"""

import sys, os, json, argparse, logging

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
ANALYSIS_DIR = "/data/rech/huiyuche/continual_ir/figures/analysis"

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--steps", type=str, default="47:470:47")
ap.add_argument("--fisher_old", type=str,
                default=os.path.join(ANALYSIS_DIR, "fisher_msmarco.pt"))
ap.add_argument("--eval_json", type=str,
                default="/data/rech/huiyuche/continual_ir/figures/bucket_runs_eval.json",
                help="offline eval results with per-(run,step) msmarco NDCG@10; "
                     "zero_shot/0 entry supplies the no-forgetting reference")
ap.add_argument("--bucket_fishers", nargs="*", default=None,
                help="fisher_<bucket>.pt paths for analysis 2; default: every "
                     "fisher_turn_*.pt found in the analysis dir")
ap.add_argument("--top_p", type=float, nargs="+", default=[0.01, 0.05, 0.20])
ap.add_argument("--device", type=str,
                default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

_start, _stop, _step = (int(x) for x in args.steps.split(":"))
STEPS = list(range(_start, _stop + 1, _step))


def load_fisher(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {n: t.to(device=device, dtype=torch.float32)
            for n, t in blob["fisher"].items()}


def ewc_quadratic(base_st, ckpt_st, fisher, device):
    """sum_i F_i * Delta_i^2, streamed per tensor."""
    total = 0.0
    with safe_open(base_st, framework="pt", device="cpu") as fb, \
         safe_open(ckpt_st, framework="pt", device="cpu") as fc:
        for name in fb.keys():
            f_key = name if name in fisher else None
            if f_key is None:
                # ckpt names may carry/lack a "model." prefix vs the live module
                alt = name[len("model."):] if name.startswith("model.") else f"model.{name}"
                f_key = alt if alt in fisher else None
            if f_key is None:
                continue
            a = fb.get_tensor(name).to(device).float()
            b = fc.get_tensor(name).to(device).float()
            d = b - a
            total += float((fisher[f_key] * d * d).sum().item())
            del a, b, d
    return total


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def topk_mask_overlap(f_a, f_b, p):
    """|top-p%(a) ∩ top-p%(b)| / k, computed over the concatenated flat vector."""
    va = torch.cat([t.flatten() for _, t in sorted(f_a.items())])
    vb = torch.cat([t.flatten() for _, t in sorted(f_b.items())])
    k = max(1, int(p * va.numel()))
    ia = set(torch.topk(va, k).indices.cpu().numpy().tolist())
    ib = set(torch.topk(vb, k).indices.cpu().numpy().tolist())
    return len(ia & ib) / k


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    device = args.device
    base_st = os.path.join(BASE_MODEL, "model.safetensors")
    fisher_old = load_fisher(args.fisher_old, device)
    logger.info(f"F_old loaded ({len(fisher_old)} tensors) from {args.fisher_old}")

    # ── analysis 1: EWC predictor vs measured forgetting ─────────────────────
    with open(args.eval_json) as f:
        eval_results = json.load(f)
    ms_zero = eval_results["zero_shot"]["0"]["msmarco"]

    records = []
    for run in args.runs:
        for step in STEPS:
            ckpt = os.path.join(CKPT_BASE, run, f"checkpoint-step-{step}")
            key = str(step)
            if not os.path.isdir(ckpt) or key not in eval_results.get(run, {}):
                continue
            pred = ewc_quadratic(base_st, os.path.join(ckpt, "model.safetensors"),
                                 fisher_old, device)
            measured_drop = ms_zero - eval_results[run][key]["msmarco"]
            records.append({"run": run, "step": step,
                            "ewc_pred": pred, "msmarco_drop": measured_drop})
            logger.info(f"{run} step-{step}: EWC={pred:.4e}  "
                        f"measured drop={measured_drop:.4f}")

    rho = spearman([r["ewc_pred"] for r in records],
                   [r["msmarco_drop"] for r in records]) if len(records) > 2 else float("nan")
    logger.info(f"Spearman(EWC prediction, measured MSMARCO drop) over "
                f"{len(records)} points: rho={rho:.3f}")

    # ── analysis 2: Fisher-mask overlap vs bucket length ─────────────────────
    if args.bucket_fishers is None:
        import glob
        bucket_paths = sorted(glob.glob(os.path.join(ANALYSIS_DIR, "fisher_turn_*.pt")))
    else:
        bucket_paths = args.bucket_fishers
    overlaps = {}
    for path in bucket_paths:
        name = os.path.basename(path)[len("fisher_"):-len(".pt")]
        f_b = load_fisher(path, device)
        overlaps[name] = {str(p): topk_mask_overlap(f_b, fisher_old, p)
                          for p in args.top_p}
        logger.info(f"F_{name} vs F_old top-p overlap: {overlaps[name]}")
        del f_b
        torch.cuda.empty_cache()

    out = os.path.join(ANALYSIS_DIR, "ewc_forgetting_predictor.json")
    with open(out, "w") as f:
        json.dump({"records": records, "spearman_rho": rho,
                   "msmarco_zero_shot": ms_zero,
                   "fisher_overlap_vs_bucket": overlaps}, f, indent=2)
    logger.info(f"results -> {out}")


if __name__ == "__main__":
    main()
