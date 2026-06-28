"""
eval_instruct3_qwen_topiocqa_per_epoch.py
=========================================
For each of the 8 *instruct3_* (v3-template) Qwen3-Embedding-0.6B fine-tuning
runs, evaluate EVERY saved per-epoch checkpoint on TopiOCQA valid under the
PROPER v3 template.

Purpose: rebuild the correct per-epoch TopiOCQA trajectory for Figure 4 panel
(g) and the Table 1 v3 TopiOCQA column. The in-training TopiOCQA eval calls in
train_qwen_cl.py did not pass `template_version`, so eval_conv_search fell back
to its "v1" default: the instruct3 models were TRAINED on v3-format queries
(role markers + `User's last question:` anchor) but EVALUATED on v1-format
queries (newline-joined, no role markers) throughout training. This train/eval
format mismatch depressed every in-training TopiOCQA reading by ~7 NDCG@10
points: in-training final for instruct3_qwen_nosched = 0.3940 vs 0.4669 when
the same checkpoint-step-1880 is evaluated offline under proper v3
(eval_topiocqa_caps_ablation.py, 2026-06-09; truncation caps were ruled out as
a factor in the same ablation, B@512 = 0.4666 vs C@32768 = 0.4669).
train_qwen_cl.py is fixed to thread template_version; this script repairs the
already-trained batch's trajectories.

Mirrors eval_instruct3_qwen_msmarco_per_epoch.py: one corpus load, 8 runs x 20
ckpts, resume-safe incremental JSON, FAISS index cached on GPU across ckpts.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval/eval_instruct3_qwen_topiocqa_per_epoch.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_instruct3_qwen_topiocqa_per_epoch.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import (
    eval_conv_search,
    load_corpus_into_faiss,
    CONV_INSTRUCTION_V3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
CORPUS_DIR  = "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged"
TOPIOCQA_VALID = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
                  "topiocqa_valid.jsonl")
TOPIOCQA_QREL  = ("/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/"
                  "topiocqa_qrel.trec")
RESULTS_OUT = ("/data/rech/huiyuche/continual_ir/figures/"
               "constlr_lr6e6_topiocqa_per_epoch.json")

# All 8 instruct3 fine-tuned runs.
RUNS = [
    "instruct3fp32infonce_qwen_constlr_lr6e6",
]

# 20 epoch-end ckpts per run: 94 × epoch (1..20).
CKPT_STEPS = [94 * i for i in range(1, 21)]   # [94, 188, ..., 1880]

# Match the instruct3 training protocol exactly: v3 template, 32768 caps.
# (Caps don't change the result — see eval_topiocqa_caps_ablation.py — but we
# stay byte-faithful to the training-side query construction.)
CAPS      = 32768
EMBED_DIM = 1024
EVAL_BS   = 128   # L40S 46 GB
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


def load_encoder(ckpt_path):
    logger.info(f"Loading encoder from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = "left"
    base = AutoModel.from_pretrained(
        ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    base.config.use_cache = False
    return tokenizer, QwenQueryEncoder(base)


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
            logger.warning(f"Could not load existing {RESULTS_OUT}: {e} — starting fresh")
    return {}


# ── one-time setup ────────────────────────────────────────────────────────────
logger.info(f"Loading TopiOCQA corpus index from {CORPUS_DIR} ...")
faiss_index, doc_ids = load_corpus_into_faiss(
    CORPUS_DIR, embed_dim=EMBED_DIM, use_gpu=False,
)
logger.info(f"Loaded {faiss_index.ntotal} doc embeddings.")

state = _load_existing()
logger.info("Existing results for %d runs already on disk; will skip computed (run,step).",
            len(state))

# Shared GPU index cache — the TopiOCQA index (fp16, sharded) lives on the
# 4 GPUs once and is reused by every checkpoint eval below.
_gpu_faiss_cache = {}

# ── nested loop: 8 runs × 20 ckpts ────────────────────────────────────────────
for run_name in RUNS:
    state.setdefault(run_name, {})
    for step in CKPT_STEPS:
        step_key = str(step)
        if step_key in state[run_name]:
            logger.info(f"[skip] {run_name} step-{step} already computed: "
                        f"NDCG@10={state[run_name][step_key]:.4f}")
            continue

        ckpt_path = os.path.join(CKPT_BASE, run_name, f"checkpoint-step-{step}")
        if not os.path.isdir(ckpt_path):
            logger.warning(f"Checkpoint missing: {ckpt_path} — skipping")
            continue

        logger.info(f"\n--- {run_name} step-{step} ---")
        tokenizer, encoder = load_encoder(ckpt_path)
        with torch.no_grad():
            metrics = eval_conv_search(
                query_encoder       = encoder,
                tokenizer           = tokenizer,
                test_data_file      = TOPIOCQA_VALID,
                qrel_file           = TOPIOCQA_QREL,
                faiss_index         = faiss_index,
                doc_ids             = doc_ids,
                device              = DEVICE,
                eval_batch_size     = EVAL_BS,
                max_query_length    = CAPS,
                max_response_length = CAPS,
                max_concat_length   = CAPS,
                use_gpu_faiss       = True,
                use_gpu_fp16        = True,
                keep_faiss_on_gpu   = True,
                gpu_index_cache     = _gpu_faiss_cache,
                full_eval           = False,
                left_padding        = True,
                dataset_tag         = "topiocqa",
                conv_instruction    = CONV_INSTRUCTION_V3,
                template_version    = "v3",
            )
        ndcg = float(metrics["NDCG@10"])
        state[run_name][step_key] = ndcg
        _save(state)
        logger.info(f"[ok ] {run_name} step-{step}: TopiOCQA NDCG@10={ndcg:.4f}")

        del encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

logger.info(f"\nResults saved to {RESULTS_OUT}")

# Compact summary
print("\n" + "=" * 80)
print(f"{'run':<42} {'step→ndcg@10 (per epoch)':>30}")
print("-" * 80)
for run_name in RUNS:
    if run_name not in state:
        continue
    series = " ".join(f"{state[run_name].get(str(s), float('nan')):.3f}" for s in CKPT_STEPS)
    print(f"{run_name:<42} {series}")
print("=" * 80)
