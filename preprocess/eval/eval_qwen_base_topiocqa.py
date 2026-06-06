"""
eval_qwen_base_topiocqa.py
==========================
Focused zero-shot eval of Qwen3-Embedding-0.6B on TopiOCQA only, under a
chosen conversational instruct template (v1 or v2). Sidesteps the BEIR
cache build and the QReCC 160 GB corpus load that eval_qwen_base_full.py
incurs when you only need the TopiOCQA number for a quick A/B (e.g.
comparing v1 vs v2 prompt wording on the same base encoder).

All eval logic is delegated to utils.eval_conv_search, which threads
template_version through to build_qwen_instruct_query_ids; we just supply
the TopiOCQA corpus + valid + qrel.

Usage:
    cd /data/rech/huiyuche/continual_ir
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
      python -u preprocess/eval/eval_qwen_base_topiocqa.py \\
        --template_version v2 \\
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/topiocqa_zeroshot_v2.log
"""

import sys, os, json, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import (
    eval_conv_search,
    load_corpus_into_faiss,
    CONV_INSTRUCTION_V1,
    CONV_INSTRUCTION_V2,
    CONV_INSTRUCTION_V3,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--template_version", type=str, default="v1",
                choices=["v1", "v2", "v3"],
                help="Conversational instruct template (see "
                     "src/utils.py:build_qwen_instruct_query_ids).")
ap.add_argument("--results_out", type=str, default=None,
                help="Default: figures/qwen_base_topiocqa_<template_version>.json")
ap.add_argument("--eval_bs", type=int, default=64)
ap.add_argument("--corpus_dir", type=str,
                default="/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged",
                help="Prefer the local-SSD copy at /part/01/Tmp/...; "
                     "the network copy at /data/rech/...//embeddings/"
                     "topiocqa_qwen_merged is a slower fallback.")
args = ap.parse_args()
TEMPLATE_VERSION = args.template_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL = ("/data/rech/huiyuche/huggingface/"
              "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
              "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
TOPIOCQA_VALID = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
                  "topiocqa_valid.jsonl")
TOPIOCQA_QREL  = ("/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/"
                  "topiocqa_qrel.trec")
RESULTS_OUT = args.results_out or (
    "/data/rech/huiyuche/continual_ir/figures/"
    f"qwen_base_topiocqa_{TEMPLATE_VERSION}.json"
)

CONV_INSTR = {"v1": CONV_INSTRUCTION_V1,
              "v2": CONV_INSTRUCTION_V2,
              "v3": CONV_INSTRUCTION_V3}[TEMPLATE_VERSION]

EMBED_DIM = 1024
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Same QwenQueryEncoder wrapper as the rest of the codebase: left-padded
# last-token pool + fp32 normalize. Matches src/train_qwen_cl.py's encoder.
class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


logger.info(f"=== TopiOCQA zero-shot, template_version={TEMPLATE_VERSION} ===")
logger.info(f"conv_instruction: {CONV_INSTR!r}")

logger.info(f"Loading Qwen3-Embedding base from {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# left-pad: required by FlashAttention 2 AND by the left-padded last-token
# pool the encoder above performs (position -1 must be the last real token).
tokenizer.padding_side = "left"
base = AutoModel.from_pretrained(
    BASE_MODEL,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval()
base.config.use_cache = False
encoder = QwenQueryEncoder(base)

logger.info(f"Loading TopiOCQA corpus index from {args.corpus_dir} ...")
faiss_index, doc_ids = load_corpus_into_faiss(
    args.corpus_dir, embed_dim=EMBED_DIM, use_gpu=False,
)
logger.info(f"Loaded {faiss_index.ntotal} doc embeddings.")

# eval_conv_search internally:
#   - opens TOPIOCQA_VALID jsonl, picks qrel-eligible queries
#   - builds tokens via _build_topiocqa_query_tokens (which dispatches to
#     build_qwen_instruct_query_ids with the requested template_version
#     because conv_instruction is non-empty)
#   - left-pads, encodes (last-token pool + L2 normalize)
#   - searches the (sharded GPU fp16) FAISS index
#   - returns the requested metric set
logger.info("Running TopiOCQA eval via eval_conv_search ...")
with torch.no_grad():
    metrics = eval_conv_search(
        query_encoder       = encoder,
        tokenizer           = tokenizer,
        test_data_file      = TOPIOCQA_VALID,
        qrel_file           = TOPIOCQA_QREL,
        faiss_index         = faiss_index,
        doc_ids             = doc_ids,
        device              = DEVICE,
        eval_batch_size     = args.eval_bs,
        # Match training distribution exactly (same caps as the qwen_instr
        # setting in eval_qrecc_per_epoch.py / train_qwen_cl.py defaults).
        max_query_length    = 64,
        max_response_length = 64,
        max_concat_length   = 512,
        use_gpu_faiss       = True,
        use_gpu_fp16        = True,
        keep_faiss_on_gpu   = False,
        full_eval           = True,
        left_padding        = True,
        dataset_tag         = "topiocqa",
        conv_instruction    = CONV_INSTR,
        template_version    = TEMPLATE_VERSION,
    )

logger.info(
    f"TopiOCQA full eval: NDCG@10={metrics['NDCG@10']:.4f}  "
    f"Recall@100={metrics['Recall@100']:.4f}  "
    f"MRR@10={metrics['MRR@10']:.4f}  "
    f"MAP@10={metrics['MAP@10']:.4f}"
)

os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump({
        "template_version": TEMPLATE_VERSION,
        "conv_instruction": CONV_INSTR,
        "max_query_length": 64,
        "max_response_length": 64,
        "max_concat_length": 512,
        "topiocqa": metrics,
    }, f, indent=2)
logger.info(f"Saved {RESULTS_OUT}")

print(f"\n% TopiOCQA zero-shot Qwen3-base (template={TEMPLATE_VERSION}):")
print(f"  NDCG@10 = {metrics['NDCG@10']*100:.2f}")
