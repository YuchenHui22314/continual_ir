"""
eval_topiocqa_caps_ablation.py
==============================
Disentangle the two confounded variables behind the instruct3 in-domain gap
(TopiOCQA 39.68 under v3 vs 46.51 under v1): the *template* (v1 vs v3) and the
*truncation caps* (64/64/512 vs 32768).

Every TopiOCQA number in the current result set was measured at 64/64/512
EXCEPT the instruct3 in-training one (32768), which is also the only anomalous
one. Under smart truncation, 512-caps inputs keep the current question plus
only the most-recent turns — a free recency filter that drops earlier
(topic-switched, off-topic) history; 32768 feeds the full conversation.

Three runs on one corpus load (all v3 template, all smart truncation):
  A. zero-shot base            @ 32768   — compare to the known 24.92 @ 512.
       Drops a lot  => long inputs dilute the base encoder's last-token pool.
  B. instruct3_nosched final   @ 512     — compare to the known 39.68 @ 32768.
       Recovers to ~45+ => the in-domain gap is mostly the length regime,
       not the v3 template.
  C. instruct3_nosched final   @ 32768   — sanity: must reproduce ~39.7
       (validates this standalone harness against the in-training eval).

Usage:
    cd /data/rech/huiyuche/continual_ir
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
      python -u preprocess/eval/eval_topiocqa_caps_ablation.py \\
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/topiocqa_caps_ablation.log
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

BASE_MODEL = ("/data/rech/huiyuche/huggingface/"
              "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
              "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
INSTRUCT3_FINAL = ("/data/rech/huiyuche/huggingface/continual_ir/"
                   "instruct3_qwen_nosched/checkpoint-step-1880")
TOPIOCQA_VALID = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
                  "topiocqa_valid.jsonl")
TOPIOCQA_QREL  = ("/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/"
                  "topiocqa_qrel.trec")
CORPUS_DIR  = "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged"
RESULTS_OUT = ("/data/rech/huiyuche/continual_ir/figures/"
               "topiocqa_caps_ablation_v3.json")

EMBED_DIM = 1024
EVAL_BS   = 64
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (label, model_path, (max_query_length, max_response_length, max_concat_length))
# Reference points measured elsewhere:
#   zero-shot v3 @ (64,64,512)            = 0.2492  (prompt-ablation table)
#   instruct3_nosched final @ (32768,...) = 0.3968  (in-training eval, valid)
#   instruct2_nosched final v1 @ (64,64,512) = 0.4651 (in-training eval, valid)
CONFIGS = [
    ("A_zeroshot_v3_caps32768",         BASE_MODEL,      (32768, 32768, 32768)),
    ("B_instruct3_final_v3_caps512",    INSTRUCT3_FINAL, (64,    64,    512)),
    ("C_instruct3_final_v3_caps32768",  INSTRUCT3_FINAL, (32768, 32768, 32768)),
]


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


logger.info(f"Loading TopiOCQA corpus index from {CORPUS_DIR} ...")
faiss_index, doc_ids = load_corpus_into_faiss(
    CORPUS_DIR, embed_dim=EMBED_DIM, use_gpu=False,
)
logger.info(f"Loaded {faiss_index.ntotal} doc embeddings.")

results = {}
cur_model_path = None
tokenizer = encoder = None

for label, model_path, (mq, mr, mc) in CONFIGS:
    if model_path != cur_model_path:
        if encoder is not None:
            del encoder, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        tokenizer, encoder = load_encoder(model_path)
        cur_model_path = model_path

    logger.info(f"\n=== {label}: caps=({mq},{mr},{mc}) ===")
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
            max_query_length    = mq,
            max_response_length = mr,
            max_concat_length   = mc,
            use_gpu_faiss       = True,
            use_gpu_fp16        = True,
            keep_faiss_on_gpu   = False,
            full_eval           = True,
            left_padding        = True,
            dataset_tag         = "topiocqa",
            conv_instruction    = CONV_INSTRUCTION_V3,
            template_version    = "v3",
        )
    results[label] = {"caps": [mq, mr, mc], "metrics": metrics}
    logger.info(f"[{label}] NDCG@10={metrics['NDCG@10']:.4f}  "
                f"Recall@100={metrics['Recall@100']:.4f}  "
                f"MRR@10={metrics['MRR@10']:.4f}")
    # incremental save
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)

logger.info(f"\nSaved {RESULTS_OUT}")

print("\n" + "=" * 78)
print("REFERENCE: zeroshot v3 @512 = 0.2492 | instruct3 final @32768 = 0.3968 "
      "| instruct2 final v1 @512 = 0.4651")
for label, r in results.items():
    print(f"{label:<38} NDCG@10 = {r['metrics']['NDCG@10']:.4f}")
print("=" * 78)
