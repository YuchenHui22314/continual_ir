"""
eval_ance_beir_full.py
======================
Evaluate fine-tuned ANCE checkpoints (anti-curriculum runs) on the full BEIR
benchmark (14 datasets, cqadupstack excluded), then print a LaTeX results table.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0 python preprocess/eval_ance_beir_full.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_ance_beir.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from transformers import RobertaTokenizer

from models import ANCE
from utils import build_beir_eval_cache, eval_beir_from_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/ance"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/ance_anticl_eval_results.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
    # cqadupstack excluded: subdirectory structure not supported
]
BEIR_AVG_EXCLUDE = {"msmarco"}

RUNS = {
    "topiocqa_anticl_root_2":       "ACL-root$_2$",
    "topiocqa_anticl_step":         "ACL-step",
    "topiocqa_anticl_step_exclusive": "ACL-step-excl",
}

# TopiOCQA NDCG@10 from final wandb metrics logged at end of training
TOPO_NDCG10 = {
    "topiocqa_anticl_root_2":         0.19940,
    "topiocqa_anticl_step":           0.19738,
    "topiocqa_anticl_step_exclusive": 0.19328,
}

EMBED_DIM    = 768
EVAL_BS      = 64
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_encoder(ckpt_path):
    logger.info(f"Loading ANCE encoder from {ckpt_path}")
    tokenizer = RobertaTokenizer.from_pretrained(ckpt_path)
    encoder = ANCE.from_pretrained(ckpt_path).to(DEVICE).eval()
    encoder.config.use_cache = False
    return tokenizer, encoder


# ── build BEIR cache once ──────────────────────────────────────────────────────
logger.info("Building BEIR eval cache (loading all corpus indices) ...")
beir_cache = build_beir_eval_cache(
    dataset_list        = BEIR_DATASETS,
    embedding_base_path = BEIR_EMB,
    beir_data_path      = BEIR_DATA,
    embed_dim           = EMBED_DIM,
    use_gpu             = False,
)

all_results = {}

for run_name, label in RUNS.items():
    ckpt_path = os.path.join(CKPT_BASE, run_name, "checkpoint-step-1880")
    if not os.path.isdir(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path} — skipping")
        continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {run_name} ...")
    tokenizer, encoder = load_encoder(ckpt_path)

    with torch.no_grad():
        beir_metrics = eval_beir_from_cache(
            beir_cache       = beir_cache,
            query_encoder    = encoder,
            tokenizer        = tokenizer,
            device           = DEVICE,
            eval_batch_size  = EVAL_BS,
            use_gpu_faiss    = True,
            keep_faiss_on_gpu= False,
            full_eval        = True,
        )

    all_results[run_name] = {
        "label": label,
        "beir":  beir_metrics,
    }

    beir_avg = sum(
        v["NDCG@10"] for k, v in beir_metrics.items() if k not in BEIR_AVG_EXCLUDE
    ) / max(1, sum(1 for k in beir_metrics if k not in BEIR_AVG_EXCLUDE))
    ms_ndcg = beir_metrics.get("msmarco", {}).get("NDCG@10", float("nan"))
    logger.info(
        f"  {run_name}: MSMARCO NDCG@10={ms_ndcg:.4f}  Avg BEIR* NDCG@10={beir_avg:.4f}"
    )

    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ── save JSON ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump(all_results, f, indent=2)
logger.info(f"\nResults saved to {RESULTS_OUT}")

# ── print LaTeX snippet (rows to paste into the existing table) ────────────────
def fmt(val, bold=False, underline=False):
    s = f"{val*100:.2f}"
    if bold:      s = r"\bf " + s
    if underline: s = r"\underline{" + s + "}"
    return s

print("\n\n% ── ANCE Anti-Curriculum rows (paste into existing table) ──────────")

rows = {}
for run_name, data in all_results.items():
    beir = data["beir"]
    beir_avg = sum(
        v["NDCG@10"] for k, v in beir.items() if k not in BEIR_AVG_EXCLUDE
    ) / max(1, sum(1 for k in beir if k not in BEIR_AVG_EXCLUDE))
    rows[run_name] = {
        "label":     data["label"],
        "topo_ndcg": TOPO_NDCG10.get(run_name, 0.0),
        "ms_ndcg":   beir.get("msmarco", {}).get("NDCG@10", 0.0),
        "beir_avg":  beir_avg,
    }

for run_name, r in rows.items():
    topo = fmt(r["topo_ndcg"])
    ms   = fmt(r["ms_ndcg"])
    beir = fmt(r["beir_avg"])
    print(f"        {r['label']:<40} & {topo:<10} & {ms:<10} & {beir} \\\\")

# also save rows to JSON for downstream use
rows_out = "/data/rech/huiyuche/continual_ir/figures/ance_anticl_rows.json"
with open(rows_out, "w") as f:
    json.dump(rows, f, indent=2)
logger.info(f"Row data saved to {rows_out}")
