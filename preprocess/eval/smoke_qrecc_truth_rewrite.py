"""
smoke_qrecc_truth_rewrite.py
============================
Diagnostic smoke. Question:
    is the 10x gap vs ChatRetriever (NDCG@10≈0.05 vs reported ~0.5) caused by
    our raw-conversation query format vs literature's "oracle rewrite"
    (Truth_rewrite field, a human-written standalone version of the current
    question that resolves coreference)?

What we test (zero-shot Qwen3-Embedding-0.6B base, no fine-tune):
  variant A — current default: full conversation history + current question
              with the conversational instruction.
  variant B — Truth_rewrite as a standalone Query with the BEIR-style
              "Given a question, retrieve relevant passages…" instruction
              (matches how literature uses "oracle rewrite").

Both variants are scored on the SAME 8,282-query QReCC answerable subset, same
corpus (qrecc_qwen_merged 54.6M docs, fp16 sharded GPU FAISS), same eval code.

Output: prints both NDCG@10/Recall@100/MRR@10 side-by-side.
        Also saves figures/SMOKE_qrecc_truth_rewrite.json.

Run AFTER qwen_no_instr finishes (corpus + GPUs are tied up while it's running).
"""
import sys, os, json, gc, time, logging, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from beir.retrieval.evaluation import EvaluateRetrieval

from utils import load_corpus_into_faiss, CONV_INSTRUCTION_V1, CONV_INSTRUCTION_V2
import faiss

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# CLI — pick conversational template version used for variant A.
# v1 (default) is byte-identical to the original smoke that produced the
# pooling-fix verification numbers (49.36 NDCG@10 etc.); v2 swaps in the
# role-marker template introduced 2026-06-05.
_ap = argparse.ArgumentParser()
_ap.add_argument("--template_version", type=str, default="v1",
                 choices=["v1", "v2"],
                 help="Conversational template for variant A. v1=legacy "
                      "newline-joined; v2=User:/System: role markers, single "
                      "space turn separator. v1-trained checkpoints are OOD "
                      "under v2.")
_args = _ap.parse_args()
TEMPLATE_VERSION = _args.template_version

QWEN_BASE = "/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
QRECC_QWEN_EMB = "/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/qrecc_qwen_merged"
QRECC_VAL  = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/qrecc/qrecc_valid.jsonl"
QRECC_QREL = "/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/qrecc_qrel.trec"
# Avoid clobbering the v1 SMOKE JSON when running v2.
OUT_JSON   = ("/data/rech/huiyuche/continual_ir/figures/"
              "SMOKE_qrecc_truth_rewrite"
              + ("" if TEMPLATE_VERSION == "v1" else f"_{TEMPLATE_VERSION}")
              + ".json")

# Variant A uses the conversational instruction (v1 or v2). Variant B is a
# single-query format unaffected by the template change.
CONV_INSTR = CONV_INSTRUCTION_V1 if TEMPLATE_VERSION == "v1" else CONV_INSTRUCTION_V2
QUERY_INSTR = "Given a question, retrieve relevant passages that answer the question"

EMBED_DIM = 1024
EVAL_BS   = 64
TOPK      = 1000
MAX_LEN   = 512
DEVICE    = torch.device("cuda:0")


class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Match training: left-padded last-token pool, fp32 normalize.
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


# ── load model
log.info("Loading Qwen3-Embedding-0.6B base ...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE)
tokenizer.padding_side = "left"
base = AutoModel.from_pretrained(
    QWEN_BASE, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
).to(DEVICE).eval()
base.config.use_cache = False
encoder = QwenQueryEncoder(base)

# ── load QReCC corpus into FAISS GPU fp16
log.info("Loading QReCC corpus → CPU FAISS ...")
t0 = time.time()
cpu_idx, doc_ids = load_corpus_into_faiss(QRECC_QWEN_EMB, embed_dim=EMBED_DIM, use_gpu=False)
log.info("Loaded %d docs in %.1f min", cpu_idx.ntotal, (time.time() - t0) / 60)

log.info("Sharding to GPU sharded fp16 ...")
co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
gpu_idx = faiss.index_cpu_to_all_gpus(cpu_idx, co=co)
del cpu_idx; gc.collect()

# ── load qrels (answerable subset filter)
qrels = {}
with open(QRECC_QREL) as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 4:
            qid, did, rel = p[0], p[2], int(p[3])
            qrels.setdefault(qid, {})[did] = rel

# ── build TWO sets of queries on same answerable subset
log.info("Building query sets ...")
qids = []; q_text_A = []; q_text_B = []
with open(QRECC_VAL) as f:
    for line in f:
        r = json.loads(line.strip())
        qid = f"{r['Conversation_no']}-{r['Turn_no']}"
        if qid not in qrels: continue
        ctx = list(r.get("Context", []))
        cur = r["Question"]
        tr  = r.get("Truth_rewrite") or cur

        # A: conversation. Template depends on --template_version. We inline
        # the construction here (instead of calling build_qwen_instruct_query_ids)
        # to keep the v1 byte sequence identical to the original smoke; the v2
        # branch mirrors utils.build_qwen_instruct_query_ids' non-smart path.
        if TEMPLATE_VERSION == "v2":
            parts = []
            for j, utt in enumerate(ctx):
                role = "User" if (j % 2 == 0) else "System"
                parts.append(f"{role}: {utt}")
            parts.append(f"User: {cur}")
            conv = " ".join(parts)
            q_text_A.append(f"Instruct: {CONV_INSTR}\nConversation: {conv}")
        else:
            conv = "\n".join(ctx + [cur])
            q_text_A.append(f"Instruct: {CONV_INSTR}\nConversation:{conv}")
        # B: Truth_rewrite as standalone Query
        q_text_B.append(f"Instruct: {QUERY_INSTR}\nQuery:{tr}")
        qids.append(qid)
log.info("%d queries in answerable subset", len(qids))


def encode(texts):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), EVAL_BS):
            batch = texts[i:i+EVAL_BS]
            enc = tokenizer(batch, max_length=MAX_LEN, padding=True,
                            truncation=True, return_tensors="pt")
            e = encoder(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE))
            embs.append(e.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def eval_variant(label, texts):
    log.info("=== %s ===", label)
    t0 = time.time()
    q_emb = encode(texts)
    log.info("[%s] encoded %d queries in %.0fs", label, len(texts), time.time() - t0)
    t0 = time.time()
    scores, indices = gpu_idx.search(q_emb.astype(np.float32), TOPK)
    log.info("[%s] FAISS search in %.0fs", label, time.time() - t0)
    run = {}
    for qi, qid in enumerate(qids):
        run[qid] = {str(doc_ids[int(idx)]): float(scores[qi, r])
                    for r, idx in enumerate(indices[qi])
                    if 0 <= int(idx) < len(doc_ids)}
    retr = EvaluateRetrieval(None)
    ndcg, _, recall, _ = retr.evaluate(qrels, run, [10, 100, 1000])
    mrr = retr.evaluate_custom(qrels, run, [10], metric="mrr")
    out = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"],
           "Recall@1000": recall["Recall@1000"], "MRR@10": mrr["MRR@10"]}
    log.info("[%s] %s", label, out)
    return out


res = {}
res["A_conversation"]    = eval_variant("A: full conversation + CONV_INSTR", q_text_A)
res["B_truth_rewrite"]   = eval_variant("B: Truth_rewrite + QUERY_INSTR",    q_text_B)

print("\n" + "=" * 80)
print(f"{'variant':<40}{'NDCG@10':>10}{'R@100':>10}{'R@1000':>10}{'MRR@10':>10}")
print("-" * 80)
for k, v in res.items():
    print(f"{k:<40}{v['NDCG@10']*100:>10.2f}{v['Recall@100']*100:>10.2f}{v['Recall@1000']*100:>10.2f}{v['MRR@10']*100:>10.2f}")
print("=" * 80)

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(res, f, indent=2)
log.info("Saved %s", OUT_JSON)
