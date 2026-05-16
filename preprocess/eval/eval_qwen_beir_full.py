"""
eval_qwen_beir_full.py
======================
Evaluate fine-tuned Qwen3-Embedding-0.6B checkpoints on the full BEIR benchmark
and TopiOCQA, then print a LaTeX results table.

Usage (from continual_ir/):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval_qwen_beir_full.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_qwen_beir.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import build_beir_eval_cache, eval_beir_from_cache, load_corpus_into_faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/qwen_eval_results.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
    # cqadupstack excluded: it has per-subdataset subdirs, not supported by load_corpus_into_faiss
]
# BEIR avg excludes msmarco (matches ANCE table)
BEIR_AVG_EXCLUDE = {"msmarco"}

RUNS = {
    "qwen_nosched":             "Conv-Qwen3 (w/o curriculum)",
    "qwen_cl_step":             "CL-step",
    "qwen_cl_step_excl":        "CL-step-excl",
    "qwen_cl_step_excl_2_full": "CL-step-excl-full",
    "qwen_acl_root2":           "ACL-root2",
    "qwen_acl_step":            "ACL-step",
    "qwen_acl_step_excl":       "ACL-step-excl",
}

EMBED_DIM    = 1024
EVAL_BS      = 64
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Qwen3 query encoder wrapper ────────────────────────────────────────────────
class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last-token pooling (left-padded → last non-pad token = position -1)
        seq_len = attention_mask.sum(dim=1) - 1          # index of last real token
        hidden  = out.last_hidden_state                  # (B, L, H)
        emb     = hidden[torch.arange(hidden.size(0)), seq_len]  # (B, H)
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


# ── build BEIR cache once ──────────────────────────────────────────────────────
logger.info("Building BEIR eval cache (loading all corpus indices) ...")
beir_cache = build_beir_eval_cache(
    dataset_list        = BEIR_DATASETS,
    embedding_base_path = BEIR_EMB,
    beir_data_path      = BEIR_DATA,
    embed_dim           = EMBED_DIM,
    use_gpu             = False,   # CPU indices; moved to GPU per-dataset during search
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

    # BEIR — one dataset at a time, free GPU memory after each.
    #
    # NOTE on instruction consistency: these are *fine-tuned* checkpoints. They
    # were trained (src/data.py) WITHOUT the Qwen3 "Instruct: ...\nQuery:..."
    # prefix, so we deliberately do NOT pass query_instruction_map here —
    # evaluating with an instruction the model never saw in Stage-2 training
    # would be a train/test mismatch. The instruction-aware path is only correct
    # for the zero-shot base model (see preprocess/eval/eval_qwen_base_full.py).
    # Fixing this properly requires retraining with the instruction prefix.
    with torch.no_grad():
        beir_metrics = eval_beir_from_cache(
            beir_cache       = beir_cache,
            query_encoder    = encoder,
            tokenizer        = tokenizer,
            device           = DEVICE,
            eval_batch_size  = EVAL_BS,
            use_gpu_faiss    = True,
            keep_faiss_on_gpu= False,   # free after each dataset
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

# ── save JSON ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump(all_results, f, indent=2)
logger.info(f"\nResults saved to {RESULTS_OUT}")

# ── print LaTeX table ──────────────────────────────────────────────────────────
def fmt(val, bold=False, underline=False):
    s = f"{val*100:.2f}"
    if bold:      s = r"\bf " + s
    if underline: s = r"\underline{" + s + "}"
    return s

print("\n\n% ── Qwen3-Embedding-0.6B Results Table ──────────────────────────────")

# TopiOCQA NDCG@10 from final wandb metrics logged at end of training
TOPO_NDCG10 = {
    "qwen_nosched":             0.47351,
    "qwen_cl_step":             0.47164,
    "qwen_cl_step_excl":        0.48248,
    "qwen_cl_step_excl_2_full": 0.47761,
    "qwen_acl_root2":           0.46720,
    "qwen_acl_step":            0.46294,
    "qwen_acl_step_excl":       0.45951,
}

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

# find best/worst per column
for col in ["topo_ndcg", "ms_ndcg", "beir_avg"]:
    vals = {k: v[col] for k, v in rows.items()}
    best = max(vals, key=vals.get)
    worst = min(vals, key=vals.get)
    for k in rows:
        rows[k][f"{col}_best"]  = (k == best)
        rows[k][f"{col}_worst"] = (k == worst)

latex = r"""\begin{table}[t]
    \centering
    \caption{Qwen3-Embedding-0.6B results. *MS MARCO is excluded from the average BEIR score. Bold = best, underlined = worst.}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{@{}lccc}
    \toprule
        \multicolumn{1}{c}{\multirow{2}[2]{*}{Retriever ($\downarrow$)}} & \multicolumn{1}{c}{New Task} & \multicolumn{1}{c}{Previous Task} & \multicolumn{1}{c}{Out-of-Domain} \\
        \cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4}
        & (TopiOCQA) & (MS MARCO) & (Avg. BEIR*)  \\
        & NDCG@10 & NDCG@10 & NDCG@10 \\
    \midrule
"""

for run_name, r in rows.items():
    topo = fmt(r["topo_ndcg"], bold=r["topo_ndcg_best"], underline=r["topo_ndcg_worst"])
    ms   = fmt(r["ms_ndcg"],   bold=r["ms_ndcg_best"],   underline=r["ms_ndcg_worst"])
    beir = fmt(r["beir_avg"],  bold=r["beir_avg_best"],   underline=r["beir_avg_worst"])
    latex += f"        {r['label']:<40} & {topo:<25} & {ms:<25} & {beir} \\\\\n"

latex += r"""    \bottomrule
     \end{tabular}}
     \label{tab:qwen_results}
\end{table}"""

print(latex)

# also save to file
tex_path = "/data/rech/huiyuche/continual_ir/figures/qwen_results_table.tex"
with open(tex_path, "w") as f:
    f.write(latex)
logger.info(f"LaTeX table saved to {tex_path}")
