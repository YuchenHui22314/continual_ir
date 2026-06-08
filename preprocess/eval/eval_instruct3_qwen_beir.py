"""
eval_instruct3_qwen_beir.py
===========================
Evaluate the 8 *instruct3_* fine-tuned Qwen3-Embedding-0.6B checkpoints (trained
WITH the v3 conversational instruction prefix: User:/System: role markers +
trailing `User's last question:` anchor) on the full BEIR benchmark.

Eval-side uses the per-task BEIR instruction prefix (same recipe as the
zero-shot baseline in eval_qwen_base_full.py and the instruct2 family in
eval_instruct2_qwen_beir.py) for an apples-to-apples comparison against the
official Qwen3-Embedding-0.6B MTEB numbers. The v3 conversational template is
NOT used here: BEIR queries are standalone questions and the eval recipe
mirrors zero-shot exactly so that the only difference between instruct2 and
instruct3 BEIR rows in the paper table is the underlying ckpt weights.

Differences from eval_instruct2_qwen_beir.py:
  - CKPT prefix instruct3_qwen_* (was instruct2_qwen_*).
  - RESULTS_OUT lands in figures/instruct3_qwen_eval_results.json.
  - EVAL_BS = 32 (was 64), since this batch will run on the octal31 4xA5000-24G
    box; halving the batch leaves enough VRAM for the larger BEIR corpora
    (msmarco fp16 ~13 GB sharded across the four GPUs gives ~3.3 GB/card, plus
    1.2 GB model on cuda:0, plus query-encoding activations under bf16 +
    FlashAttention 2 + grad-checkpointing-disabled at eval).

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python preprocess/eval/eval_instruct3_qwen_beir.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_instruct3_qwen_beir.log
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
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/instruct3_qwen_eval_results.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
]
BEIR_AVG_EXCLUDE = {"msmarco"}

# All 8 instruct3 fine-tuned runs (wandb project topiocqa-qwen-instruct-v3).
RUNS = {
    "instruct3_qwen_nosched":             "Conv-Qwen3 v3 (w/o curriculum, instruct)",
    "instruct3_qwen_cl_step":              "CL-step (instruct, v3)",
    "instruct3_qwen_cl_step_excl":         "CL-step-excl (instruct, v3)",
    "instruct3_qwen_cl_step_excl_2_full":  "CL-step-excl-full (instruct, v3)",
    "instruct3_qwen_cl_root2":             "CL-root_2 (instruct, v3)",
    "instruct3_qwen_acl_root2":            "ACL-root_2 (instruct, v3)",
    "instruct3_qwen_acl_step":             "ACL-step (instruct, v3)",
    "instruct3_qwen_acl_step_excl":        "ACL-step-excl (instruct, v3)",
}

EMBED_DIM = 1024
# A5000-24G: halved from 64 (instruct2) to keep activations + msmarco-fp16
# corpus shard + model resident on cuda:0 comfortably under 24 GB.
EVAL_BS   = 32
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
    encoder = QwenQueryEncoder(base)
    return tokenizer, encoder


logger.info("Building BEIR eval cache (14 datasets) ...")
beir_cache = build_beir_eval_cache(
    dataset_list        = BEIR_DATASETS,
    embedding_base_path = BEIR_EMB,
    beir_data_path      = BEIR_DATA,
    embed_dim           = EMBED_DIM,
    use_gpu             = False,
)

instruction_map = build_qwen_instruction_map()
logger.info("Loaded per-task BEIR instructions for %d datasets.", len(instruction_map))

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
            beir_cache            = beir_cache,
            query_encoder         = encoder,
            tokenizer             = tokenizer,
            device                = DEVICE,
            eval_batch_size       = EVAL_BS,
            use_gpu_faiss         = True,
            keep_faiss_on_gpu     = False,
            full_eval             = True,
            query_instruction_map = instruction_map,
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

# Compact summary log
print("\n" + "=" * 80)
print(f"{'run':<42} {'msmarco':>10} {'beir13_avg':>12}")
print("-" * 80)
for run_name, data in all_results.items():
    b = data["beir"]
    avg = sum(v["NDCG@10"] for k, v in b.items() if k not in BEIR_AVG_EXCLUDE) / 13
    ms  = b.get("msmarco", {}).get("NDCG@10", float("nan"))
    print(f"{run_name:<42} {ms*100:>10.2f} {avg*100:>12.2f}")
print("=" * 80)
