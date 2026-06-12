"""
eval_bucket_qwen_beir_per_ckpt.py
=================================
Qwen3-Embedding counterpart of eval_bucket_ance_beir_per_ckpt.py: BEIR-14
(MSMARCO + 13 BEIR) per-checkpoint evaluation of the 13 Qwen turn-bucket runs,
for the Qwen MSMARCO/BEIR forgetting curves and BEIR transfer-delta (the Qwen
bucket eval only covered TopiOCQA + MSMARCO; BEIR was never run per bucket).

Qwen specifics (mirroring eval_instruct3_qwen_beir.py): raw AutoTokenizer
(left-pad, trailing <|endoftext|>), last-token-pooled QwenQueryEncoder (1024d),
and the official MTEB per-task BEIR instruction map (build_qwen_instruction_map)
— BEIR queries are standalone and use the instruction prefix, NOT the v3 conv
template the buckets trained with.

All 14 corpora kept on GPU fp16 across all ckpts. Resume-safe incremental JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/eval/eval_bucket_qwen_beir_per_ckpt.py \
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_bucket_qwen_beir.log
"""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import build_beir_eval_cache, eval_beir_from_cache, build_qwen_instruction_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CKPT_BASE   = "/data/rech/huiyuche/huggingface/continual_ir"
BASE_QWEN   = ("/data/rech/huiyuche/huggingface/"
               "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
               "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
BEIR_EMB    = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
BEIR_DATA   = "/data/rech/huiyuche/beir"
RESULTS_OUT = "/data/rech/huiyuche/continual_ir/figures/bucket_qwen_beir_eval.json"

BEIR_DATASETS = [
    "msmarco", "scifact", "trec-covid", "nfcorpus", "fiqa",
    "arguana", "webis-touche2020", "quora", "scidocs", "nq",
    "hotpotqa", "dbpedia-entity", "fever", "climate-fever",
]

BUCKETS = [f"bucket_qwen_turn_{k}" for k in
           [*(str(t) for t in range(1, 11)), "11_12", "13_14", "15plus"]]
CKPT_STEPS = [47 * i for i in range(1, 11)]   # 47 .. 470

EMBED_DIM = 1024
EVAL_BS   = 128
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


def load_encoder(path):
    logger.info(f"Loading Qwen encoder from {path}")
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    base = AutoModel.from_pretrained(path, attn_implementation="flash_attention_2",
                                     torch_dtype=torch.bfloat16).to(DEVICE).eval()
    base.config.use_cache = False
    return tok, QwenQueryEncoder(base)


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


logger.info("Building BEIR-14 eval cache (Qwen 1024d) ...")
beir_cache = build_beir_eval_cache(
    dataset_list=BEIR_DATASETS, embedding_base_path=BEIR_EMB,
    beir_data_path=BEIR_DATA, embed_dim=EMBED_DIM, use_gpu=False)
instruction_map = build_qwen_instruction_map()
logger.info("BEIR cache ready.")

_gpu_cache = {}


def eval_one(tok, enc):
    with torch.no_grad():
        m = eval_beir_from_cache(
            beir_cache=beir_cache, query_encoder=enc, tokenizer=tok, device=DEVICE,
            eval_batch_size=EVAL_BS, use_gpu_faiss=True, keep_faiss_on_gpu=True,
            gpu_index_cache=_gpu_cache, full_eval=False,
            query_instruction_map=instruction_map, use_gpu_fp16=True)
    return {k: float(v) for k, v in m.items()}


state = _load_existing()

if "zero_shot" not in state:
    tok, enc = load_encoder(BASE_QWEN)
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
