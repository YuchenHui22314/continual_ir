"""
eval_bucket_ance_beir_per_ckpt.py
=================================
Offline BEIR-14 (MSMARCO + 13 BEIR) evaluation of every ANCE turn-bucket
checkpoint, for the ANCE MSMARCO/BEIR forgetting curves and the
train-bucket x BEIR-dataset transfer-delta heatmap.

Mirrors the Qwen bucket eval but for ANCE: RobertaTokenizer + the ANCE model
(models.ANCE, CLS-pooled query_emb, 768d), no instruction prefix (ANCE is not
instruction-aware). Doc corpora are the base-ANCE-encoded BEIR embeddings
(query tower is the only thing fine-tuned; doc tower frozen — same protocol as
eval_ance_beir_full.py and as training, which uses pre-encoded pos/neg docs).

All 14 corpora are loaded once and kept on the GPUs as fp16 sharded indexes
across the ~131 checkpoint evals (BEIR-14 fp16 ~52 GB across 4 GPUs ~ 13 GB/
card). Resume-safe incremental JSON.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval/eval_bucket_ance_beir_per_ckpt.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_bucket_ance_beir.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from transformers import RobertaTokenizer

from models import ANCE
from utils import build_beir_eval_cache, eval_beir_from_cache

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BASE_ANCE   = ("/data/rech/huiyuche/huggingface/"
               "models--castorini--ance-msmarco-passage/snapshots/"
               "6d7e7d6b6c59dd691671f280bc74edb4297f8234")
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/ance"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/bucket_ance_beir_eval.json"

# msmarco + the 13 BEIR datasets used elsewhere in the paper (BEIR avg excludes
# msmarco, matching the main-results convention).
BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
]

BUCKETS = [f"bucket_ance_turn_{k}" for k in
           [*(str(t) for t in range(1, 11)), "11_12", "13_14", "15plus"]]
CKPT_STEPS = [47 * i for i in range(1, 11)]   # 47 .. 470

EMBED_DIM = 768
EVAL_BS   = 128
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_encoder(path):
    logger.info(f"Loading ANCE encoder from {path}")
    tok = RobertaTokenizer.from_pretrained(path, do_lower_case=True)
    enc = ANCE.from_pretrained(path).to(DEVICE).eval()
    enc.config.use_cache = False
    return tok, enc


def _save(state):
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    tmp = RESULTS_OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RESULTS_OUT)


def _load_existing():
    if os.path.exists(RESULTS_OUT):
        try:
            with open(RESULTS_OUT) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {RESULTS_OUT}: {e} — starting fresh")
    return {}


# ── one-time setup: load all 14 corpora to CPU RAM ────────────────────────────
logger.info("Building BEIR-14 eval cache (ANCE 768d) — loading all corpora ...")
beir_cache = build_beir_eval_cache(
    dataset_list        = BEIR_DATASETS,
    embedding_base_path = BEIR_EMB,
    beir_data_path      = BEIR_DATA,
    embed_dim           = EMBED_DIM,
    use_gpu             = False,
)
logger.info("BEIR cache ready.")

# Shared GPU index cache — every corpus is transferred to the GPUs (fp16,
# sharded) on the first checkpoint and reused for all later checkpoints.
_gpu_cache = {}


def eval_one(tok, enc):
    with torch.no_grad():
        m = eval_beir_from_cache(
            beir_cache        = beir_cache,
            query_encoder     = enc,
            tokenizer         = tok,
            device            = DEVICE,
            eval_batch_size   = EVAL_BS,
            use_gpu_faiss     = True,
            keep_faiss_on_gpu = True,
            gpu_index_cache   = _gpu_cache,
            full_eval         = False,        # NDCG@10 per dataset
            query_instruction_map = None,     # ANCE: no instruction prefix
            use_gpu_fp16      = True,
        )
    # m is {dataset: ndcg10}
    return {k: float(v) for k, v in m.items()}


state = _load_existing()

# zero-shot base ANCE first.
if "zero_shot" not in state:
    tok, enc = load_encoder(BASE_ANCE)
    state["zero_shot"] = {"0": eval_one(tok, enc)}
    _save(state)
    del tok, enc; gc.collect(); torch.cuda.empty_cache()
    logger.info(f"[ok ] zero-shot: msmarco={state['zero_shot']['0'].get('msmarco', float('nan')):.4f}")

for run in BUCKETS:
    state.setdefault(run, {})
    for step in CKPT_STEPS:
        key = str(step)
        if key in state[run]:
            logger.info(f"[skip] {run} step-{step}")
            continue
        ckpt = os.path.join(CKPT_BASE, run, f"checkpoint-step-{step}")
        if not os.path.isdir(ckpt):
            logger.warning(f"missing: {ckpt} — skipping")
            continue
        logger.info(f"\n--- {run} step-{step} ---")
        tok, enc = load_encoder(ckpt)
        state[run][key] = eval_one(tok, enc)
        _save(state)
        ms = state[run][key].get("msmarco", float("nan"))
        beir_avg = sum(v for k, v in state[run][key].items() if k != "msmarco") / \
                   max(1, len([k for k in state[run][key] if k != "msmarco"]))
        logger.info(f"[ok ] {run} step-{step}: msmarco={ms:.4f}  beir13_avg={beir_avg:.4f}")
        del tok, enc; gc.collect(); torch.cuda.empty_cache()

logger.info(f"\nAll done. Results: {RESULTS_OUT}")
