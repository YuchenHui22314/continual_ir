"""
eval_bucket_runs_per_ckpt.py
============================
Offline 3-axis evaluation of the 13 turn-bucket runs (bucket_qwen_turn_*),
every saved checkpoint (steps 47, 94, ..., 470), plus the zero-shot base
encoder:

  1. TopiOCQA valid, FULL set + 13 per-turn-length subsets (one FAISS search
     per checkpoint; subset metrics are computed from the same run dict via
     utils.eval_conv_search(report_qid_subsets=...)).
  2. MSMARCO dev (BEIR protocol, raw tokenizer + per-task instruction map).

Training (scripts/run_qwen_turn_buckets.sh) deliberately ran with NO
in-training eval — octal40 had to be returned, and the in-training eval path
produced two measurement bugs in the past (wrapper tokenizer; missing
template_version). This script is the single source of truth for the bucket
experiment's numbers.

Designed to run on octal31 (4x A5000 24 GB): both corpora are loaded once and
kept on the GPUs as fp16 sharded indexes across all ~130 checkpoint evals
(TopiOCQA fp16 ~13.2 GB/card + MSMARCO fp16 ~3.3 GB/card + model ~1.2 GB
fits in 24 GB). Resume-safe: results JSON is written incrementally and
(run, step) pairs already present are skipped.

Usage:
    cd /data/rech/huiyuche/continual_ir
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval/eval_bucket_runs_per_ckpt.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_bucket_runs_per_ckpt.log
"""

import sys, os, json, gc, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import (
    eval_conv_search,
    load_corpus_into_faiss,
    build_beir_eval_cache,
    eval_beir_from_cache,
    build_qwen_instruction_map,
    CONV_INSTRUCTION_V3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--corpus_dir", type=str, default=None,
                help="TopiOCQA Qwen corpus index dir. Default: first existing of "
                     "the octal40 /part path, the octal31 /part path, the NFS copy.")
ap.add_argument("--results_out", type=str,
                default="/data/rech/huiyuche/continual_ir/figures/bucket_runs_eval.json")
ap.add_argument("--topiocqa_bs", type=int, default=64,
                help="Query-encoding batch size for TopiOCQA (long v3 queries).")
ap.add_argument("--msmarco_bs", type=int, default=128)
ap.add_argument("--no_gpu_fp16", action="store_true",
                help="Use fp32 GPU indexes (needs >40 GB cards).")
args = ap.parse_args()

CORPUS_CANDIDATES = [
    "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged",        # octal40 local SSD
    "/part/01/Tmp/yuchenhui/indexes/topiocqa_qwen_merged",     # octal31 local SSD
    "/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_qwen_merged",  # NFS
]
CORPUS_DIR = args.corpus_dir or next(p for p in CORPUS_CANDIDATES if os.path.isdir(p))

CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BASE_MODEL  = ("/data/rech/huiyuche/huggingface/"
               "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
               "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
TOPIOCQA_VALID = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
                  "topiocqa_valid.jsonl")
TOPIOCQA_QREL  = ("/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/"
                  "topiocqa_qrel.trec")
BEIR_EMB  = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA = "/data/rech/huiyuche/beir"

RUNS = [f"bucket_qwen_turn_{k}" for k in
        [*(str(t) for t in range(1, 11)), "11_12", "13_14", "15plus"]]
CKPT_STEPS = [47 * i for i in range(1, 11)]   # 47 .. 470

CAPS      = 32768   # match the bucket training protocol exactly
EMBED_DIM = 1024
USE_FP16  = not args.no_gpu_fp16
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── eval-turn subsets (same bucketing as the training sets) ───────────────────
def build_turn_subsets(valid_file):
    """qid -> turn from the valid jsonl; returns {subset_name: set(qids)}."""
    subsets = {}

    def bucket_name(turn):
        if turn <= 10:
            return f"turn_{turn}"
        if turn <= 12:
            return "turn_11_12"
        if turn <= 14:
            return "turn_13_14"
        return "turn_15plus"

    n_mismatch = 0
    with open(valid_file) as f:
        for line in f:
            rec = json.loads(line)
            qid = f"{rec['Conversation_no']}-{rec['Turn_no']}"
            turn = 1 + len(rec.get("Context", [])) // 2
            if int(rec["Turn_no"]) != turn:
                n_mismatch += 1
            subsets.setdefault(bucket_name(turn), set()).add(qid)
    if n_mismatch:
        logger.warning(f"{n_mismatch} valid records have Turn_no != 1+len(Context)//2 "
                       f"(using the Context-derived turn).")
    total = sum(len(v) for v in subsets.values())
    logger.info("Eval-turn subsets: " +
                ", ".join(f"{k}={len(v)}" for k, v in sorted(subsets.items())) +
                f"  (total {total})")
    return subsets


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


def load_encoder(path):
    logger.info(f"Loading encoder from {path}")
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    base = AutoModel.from_pretrained(
        path, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    base.config.use_cache = False
    return tok, QwenQueryEncoder(base)


def _save(state):
    tmp = args.results_out + ".tmp"
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, args.results_out)


def _load_existing():
    if os.path.exists(args.results_out):
        try:
            with open(args.results_out) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {args.results_out}: {e} — starting fresh")
    return {}


def eval_one(tokenizer, encoder):
    """Run both axes for one loaded encoder; returns the per-ckpt result dict."""
    with torch.no_grad():
        topi = eval_conv_search(
            query_encoder       = encoder,
            tokenizer           = tokenizer,
            test_data_file      = TOPIOCQA_VALID,
            qrel_file           = TOPIOCQA_QREL,
            faiss_index         = topi_index,
            doc_ids             = topi_doc_ids,
            device              = DEVICE,
            eval_batch_size     = args.topiocqa_bs,
            max_query_length    = CAPS,
            max_response_length = CAPS,
            max_concat_length   = CAPS,
            use_gpu_faiss       = True,
            use_gpu_fp16        = USE_FP16,
            keep_faiss_on_gpu   = True,
            gpu_index_cache     = _gpu_cache,
            full_eval           = False,
            left_padding        = True,
            dataset_tag         = "topiocqa",
            conv_instruction    = CONV_INSTRUCTION_V3,
            template_version    = "v3",
            report_qid_subsets  = TURN_SUBSETS,
        )
        ms = eval_beir_from_cache(
            beir_cache            = beir_cache,
            query_encoder         = encoder,
            tokenizer             = tokenizer,
            device                = DEVICE,
            eval_batch_size       = args.msmarco_bs,
            use_gpu_faiss         = True,
            keep_faiss_on_gpu     = True,
            gpu_index_cache       = _gpu_cache,
            full_eval             = False,
            query_instruction_map = instruction_map,
            use_gpu_fp16          = USE_FP16,
        )
    return {"topiocqa": topi, "msmarco": float(ms["msmarco"])}


# ── one-time setup ────────────────────────────────────────────────────────────
TURN_SUBSETS = build_turn_subsets(TOPIOCQA_VALID)

logger.info(f"Loading TopiOCQA corpus from {CORPUS_DIR} ...")
topi_index, topi_doc_ids = load_corpus_into_faiss(CORPUS_DIR, embed_dim=EMBED_DIM,
                                                  use_gpu=False)
logger.info(f"TopiOCQA: {topi_index.ntotal} docs.")

logger.info("Loading MSMARCO BEIR cache ...")
beir_cache = build_beir_eval_cache(
    dataset_list=["msmarco"], embedding_base_path=BEIR_EMB,
    beir_data_path=BEIR_DATA, embed_dim=EMBED_DIM, use_gpu=False,
)
instruction_map = build_qwen_instruction_map()

_gpu_cache = {}   # shared: both indexes live on the GPUs across all ckpts

state = _load_existing()

# zero-shot row first (base encoder under the same protocol).
if "zero_shot" not in state:
    tok, enc = load_encoder(BASE_MODEL)
    state["zero_shot"] = {"0": eval_one(tok, enc)}
    _save(state)
    del tok, enc
    gc.collect(); torch.cuda.empty_cache()

# ── 13 runs × 10 ckpts ────────────────────────────────────────────────────────
for run_name in RUNS:
    state.setdefault(run_name, {})
    for step in CKPT_STEPS:
        key = str(step)
        if key in state[run_name]:
            logger.info(f"[skip] {run_name} step-{step} already computed.")
            continue
        ckpt = os.path.join(CKPT_BASE, run_name, f"checkpoint-step-{step}")
        if not os.path.isdir(ckpt):
            logger.warning(f"missing: {ckpt} — skipping")
            continue
        logger.info(f"\n--- {run_name} step-{step} ---")
        tok, enc = load_encoder(ckpt)
        state[run_name][key] = eval_one(tok, enc)
        _save(state)
        full = state[run_name][key]["topiocqa"]["__full__"]["NDCG@10"]
        ms   = state[run_name][key]["msmarco"]
        logger.info(f"[ok ] {run_name} step-{step}: topiocqa_full={full:.4f} "
                    f"msmarco={ms:.4f}")
        del tok, enc
        gc.collect(); torch.cuda.empty_cache()

logger.info(f"\nAll done. Results: {args.results_out}")
