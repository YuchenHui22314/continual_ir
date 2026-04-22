"""
eval_qwen_beir_full_root2.py
============================
Single-run variant of eval_qwen_beir_full.py — evaluates ONLY the
`qwen_cl_root2` final checkpoint on the full BEIR benchmark and writes
to a dedicated JSON so the aggregate file with the other 7 runs is not
clobbered. The caller merges the two JSONs afterwards.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval_qwen_beir_full_root2.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_qwen_cl_root2_<ts>.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import build_beir_eval_cache, eval_beir_from_cache  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/qwen_eval_results_root2.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
]
BEIR_AVG_EXCLUDE = {"msmarco"}

RUNS = {
    "qwen_cl_root2": "CL-root2",
}

EMBED_DIM    = 1024
EVAL_BS      = 64
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        seq_len = attention_mask.sum(dim=1) - 1
        hidden  = out.last_hidden_state
        emb     = hidden[torch.arange(hidden.size(0)), seq_len]
        return F.normalize(emb, p=2, dim=-1)


def load_encoder(ckpt_path):
    logger.info(f"Loading encoder from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    base = AutoModel.from_pretrained(
        ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    base.config.use_cache = False
    encoder = QwenQueryEncoder(base)
    return tokenizer, encoder


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
    msmarco_ndcg = beir_metrics.get("msmarco", {}).get("NDCG@10", float("nan"))
    logger.info(
        f"  {run_name}: MSMARCO NDCG@10={msmarco_ndcg:.4f}  Avg BEIR* NDCG@10={beir_avg:.4f}"
    )

    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump(all_results, f, indent=2)
logger.info(f"\nResults saved to {RESULTS_OUT}")
