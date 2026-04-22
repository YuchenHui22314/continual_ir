"""
eval_qwen_base_full.py
======================
Evaluate the base (zero-shot) Qwen3-Embedding-0.6B model on:
- TopiOCQA (NDCG@10)
- Full BEIR benchmark (14 datasets, cqadupstack excluded)

Saves results to figures/qwen_base_eval_results.json.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0 python -u preprocess/eval_qwen_base_full.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_qwen_base.log
"""

import sys, os, json, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import (
    build_beir_eval_cache,
    eval_beir_from_cache,
    load_corpus_into_faiss,
    _compute_full_metrics,
)
import numpy as np
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
BEIR_EMB   = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA  = "/data/rech/huiyuche/beir"
TOPIOCQA_EMB_DIR = "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged"
TOPIOCQA_VALID   = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl"
TOPIOCQA_QREL    = "/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec"
RESULTS_OUT      = "/data/rech/huiyuche/continual_ir/figures/qwen_base_eval_results.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
]
BEIR_AVG_EXCLUDE = {"msmarco"}

EMBED_DIM = 1024
EVAL_BS   = 64
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Qwen3 query encoder wrapper: last-token pooling + L2 normalize
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


logger.info(f"Loading base Qwen3 encoder from {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Qwen3 tokenizer has no CLS/SEP tokens — fall back to BOS/EOS so the
# TopiOCQA query builder (utils._build_topiocqa_query_tokens) doesn't insert None.
if tokenizer.cls_token_id is None:
    tokenizer.cls_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
if tokenizer.sep_token_id is None:
    tokenizer.sep_token_id = tokenizer.eos_token_id

base = AutoModel.from_pretrained(
    BASE_MODEL,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval()
base.config.use_cache = False
encoder = QwenQueryEncoder(base)

# ── BEIR ──────────────────────────────────────────────────────────────────────
# Skip BEIR re-run if cached results exist (from an earlier successful BEIR pass
# that crashed later in TopiOCQA).
BEIR_CACHE_JSON = "/data/rech/huiyuche/continual_ir/figures/qwen_base_beir_cache.json"
if os.path.exists(BEIR_CACHE_JSON):
    logger.info(f"Loading cached BEIR results from {BEIR_CACHE_JSON}")
    with open(BEIR_CACHE_JSON) as f:
        beir_metrics = json.load(f)
else:
    logger.info("Building BEIR eval cache (14 datasets) ...")
    beir_cache = build_beir_eval_cache(
        dataset_list        = BEIR_DATASETS,
        embedding_base_path = BEIR_EMB,
        beir_data_path      = BEIR_DATA,
        embed_dim           = EMBED_DIM,
        use_gpu             = False,
    )

    logger.info("Running BEIR eval ...")
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

    # free BEIR cache before loading TopiOCQA
    del beir_cache
    import gc; gc.collect()
    torch.cuda.empty_cache()

# ── TopiOCQA ──────────────────────────────────────────────────────────────────
logger.info(f"Loading TopiOCQA corpus index from {TOPIOCQA_EMB_DIR} ...")
topi_index, topi_dids = load_corpus_into_faiss(
    TOPIOCQA_EMB_DIR, embed_dim=EMBED_DIM, use_gpu=False,
)

logger.info("Running TopiOCQA eval (Qwen3 plain-text path) ...")

# Zero-shot Qwen3 on TopiOCQA: build a plain-text conversational query (no CLS/SEP
# wrapping), tokenize like BEIR does, last-token pool + L2-normalize via the encoder.
# This matches how the corpus was encoded (plain text → last-token pool).
TOPI_MAX_LEN = 512
TOPI_TOPK    = 1000

# 1) qrels
topi_qrels = {}
with open(TOPIOCQA_QREL, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        qid, did, rel = parts[0], parts[2], int(parts[3])
        topi_qrels.setdefault(qid, {})[did] = rel

# 2) build plain-text queries: "ctx1\nctx2\n...\ncurrent"
topi_qids, topi_texts = [], []
with open(TOPIOCQA_VALID, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        qid = f"{rec['Conversation_no']}-{rec['Turn_no']}"
        if qid not in topi_qrels:
            continue
        parts = list(rec.get("Context", [])) + [rec["Question"]]
        topi_qids.append(qid)
        topi_texts.append("\n".join(parts))

logger.info(f"TopiOCQA: encoding {len(topi_texts)} valid queries ...")

# 3) encode queries (batched, plain-text tokenization — Qwen3 tokenizer is left-padding by default)
topi_embs_list = []
with torch.no_grad():
    for i in range(0, len(topi_texts), EVAL_BS):
        batch = topi_texts[i : i + EVAL_BS]
        enc   = tokenizer(batch, max_length=TOPI_MAX_LEN, padding=True,
                          truncation=True, return_tensors="pt")
        embs  = encoder(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE))
        topi_embs_list.append(embs.float().cpu().numpy())
topi_query_embs = np.concatenate(topi_embs_list, axis=0)

# 4) FAISS search — shard CPU index across all visible GPUs
logger.info("TopiOCQA: transferring index to GPUs (sharded) ...")
co = faiss.GpuMultipleClonerOptions()
co.shard = True
idx_gpu = faiss.index_cpu_to_all_gpus(topi_index, co=co)
scores, indices = idx_gpu.search(topi_query_embs.astype(np.float32), TOPI_TOPK)
del idx_gpu
torch.cuda.empty_cache()

# 5) build run dict and compute metrics
topi_run = {}
for q_idx, qid in enumerate(topi_qids):
    topi_run[qid] = {
        str(topi_dids[int(idx)]): float(scores[q_idx, r])
        for r, idx in enumerate(indices[q_idx])
        if 0 <= int(idx) < len(topi_dids)
    }
topi_metrics = _compute_full_metrics(topi_qrels, topi_run)
logger.info(
    f"TopiOCQA full eval: NDCG@10={topi_metrics['NDCG@10']:.4f}  "
    f"Recall@100={topi_metrics['Recall@100']:.4f}  "
    f"MRR@10={topi_metrics['MRR@10']:.4f}  "
    f"MAP@10={topi_metrics['MAP@10']:.4f}"
)

# ── save & report ─────────────────────────────────────────────────────────────
beir_avg = sum(
    v["NDCG@10"] for k, v in beir_metrics.items() if k not in BEIR_AVG_EXCLUDE
) / max(1, sum(1 for k in beir_metrics if k not in BEIR_AVG_EXCLUDE))
ms_ndcg = beir_metrics.get("msmarco", {}).get("NDCG@10", float("nan"))
topi_ndcg = topi_metrics.get("NDCG@10", float("nan"))

logger.info(f"\n{'='*60}")
logger.info(f"Qwen3 (zero-shot) results:")
logger.info(f"  TopiOCQA NDCG@10      = {topi_ndcg:.4f}")
logger.info(f"  MSMARCO NDCG@10       = {ms_ndcg:.4f}")
logger.info(f"  Avg BEIR* NDCG@10     = {beir_avg:.4f}")

results = {
    "label":    "Qwen3 (w/o Conv. data)",
    "topiocqa": topi_metrics,
    "beir":     beir_metrics,
}
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"Results saved to {RESULTS_OUT}")

print(f"\n% Zero-shot Qwen3 row:")
print(f"  Qwen3 (w/o Conv. data) & {topi_ndcg*100:.2f} & {ms_ndcg*100:.2f} & {beir_avg*100:.2f} \\\\")
