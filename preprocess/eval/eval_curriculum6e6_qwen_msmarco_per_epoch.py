"""
eval_instruct3_qwen_msmarco_per_epoch.py
========================================
For each of the 8 *instruct3_* (v3-template) Qwen3-Embedding-0.6B fine-tuning
runs, evaluate EVERY saved per-epoch checkpoint on MSMARCO dev (BEIR protocol)
with the official per-task Qwen3 instruction map and the RAW AutoTokenizer.

Purpose: rebuild the correct per-epoch MSMARCO trajectory for Figure 4 panel
(h). The in-training BEIR eval passed the Qwen3TokenizerWrapper to
eval_beir_from_cache, which appends <|im_end|> (151645) to every BEIR query
instead of the official trailing <|endoftext|> (151643) that the raw
tokenizer's post-processor adds and that the corpus embeddings were built
with. Last-token pooling at the never-trained <|im_end|> position deflates
MSMARCO NDCG@10 from ~0.34 to ~0.16 (smoke-verified 2026-06-09 on
instruct3_qwen_nosched step-1880: wrapper=0.1586 == training log, raw=0.3363
== offline eval). The same wrapper artifact explains the instruct2 in-training
~0.13 readings (there compounded by the then-missing instruction map).

Mirrors eval_instruct2_qwen_msmarco_per_epoch.py, with three deltas:
  - RUNS/RESULTS_OUT target the instruct3 family.
  - EVAL_BS = 128 (octal40, 4x L40S 46 GB — query encoding is light).
  - keep_faiss_on_gpu=True with a shared gpu_index_cache: the MSMARCO fp32
    index (~36 GB) is sharded once across the 4 GPUs (~9 GB/card) and reused
    by all 160 checkpoint evals, saving the ~12 s per-ckpt re-transfer.
    (Unlike during training there are no activation buffers competing for
    VRAM here, so caching the index is safe — cf. the instruct3 OOM postmortem
    in the qwen-instruct-training skill.)

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval/eval_instruct3_qwen_msmarco_per_epoch.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_instruct3_qwen_msmarco_per_epoch.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import (
    build_beir_eval_cache,
    eval_beir_from_cache,
    build_qwen_instruction_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/curriculum6e6_msmarco_per_epoch.json"

# Only MSMARCO; per-epoch BEIR-13 would be ~14× the runtime.
BEIR_DATASETS = ["msmarco"]

# All 8 instruct3 fine-tuned runs.
RUNS = [
    "instruct3fp32infonce_constlr_lr6e6_cl_step",
    "instruct3fp32infonce_constlr_lr6e6_cl_step_excl",
    "instruct3fp32infonce_constlr_lr6e6_cl_step_excl_2_full",
    "instruct3fp32infonce_constlr_lr6e6_cl_root2",
    "instruct3fp32infonce_constlr_lr6e6_acl_root2",
    "instruct3fp32infonce_constlr_lr6e6_acl_step",
    "instruct3fp32infonce_constlr_lr6e6_acl_step_excl",
]

# 20 epoch-end ckpts per run: 94 × epoch (1..20).
CKPT_STEPS = [94 * i for i in range(1, 21)]   # [94, 188, ..., 1880]

EMBED_DIM = 1024
EVAL_BS   = 128   # L40S 46 GB; query encoding alone, generous headroom.
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Match training: left-padded last-token pool, fp32 normalize.
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


def load_encoder(ckpt_path):
    logger.info(f"Loading encoder from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    # Match training: left-pad so QwenQueryEncoder.forward's `last_hidden_state[:, -1, :]`
    # pool always lands on the last real token, regardless of batch heterogeneity.
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
logger.info("Building BEIR cache for MSMARCO ...")
beir_cache = build_beir_eval_cache(
    dataset_list        = BEIR_DATASETS,
    embedding_base_path = BEIR_EMB,
    beir_data_path      = BEIR_DATA,
    embed_dim           = EMBED_DIM,
    use_gpu             = False,
)

instruction_map = build_qwen_instruction_map()
logger.info("Loaded per-task BEIR instructions for %d datasets.", len(instruction_map))

state = _load_existing()
logger.info("Existing results for %d runs already on disk; will skip computed (run,step).",
            len(state))

# Shared GPU index cache — the MSMARCO index is sharded to the 4 GPUs once and
# reused by every checkpoint eval below.
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
            metrics = eval_beir_from_cache(
                beir_cache            = beir_cache,
                query_encoder         = encoder,
                tokenizer             = tokenizer,
                device                = DEVICE,
                eval_batch_size       = EVAL_BS,
                use_gpu_faiss         = True,
                keep_faiss_on_gpu     = True,
                gpu_index_cache       = _gpu_faiss_cache,
                full_eval             = False,           # only NDCG@10 needed per epoch
                query_instruction_map = instruction_map,
            )
        ndcg = float(metrics.get("msmarco", float("nan")))
        state[run_name][step_key] = ndcg
        _save(state)
        logger.info(f"[ok ] {run_name} step-{step}: MSMARCO NDCG@10={ndcg:.4f}")

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
