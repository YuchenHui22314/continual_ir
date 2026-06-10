"""
fisher_importance.py
====================
Diagonal Fisher-information proxy F = sum_t g_t^2 per scalar parameter
(the FISH-mask quantity of Sung et al. 2021), computed for the BASE
Qwen3-Embedding-0.6B encoder on a chosen data distribution:

  --data msmarco         : the OLD task. F_old underlies the EWC-style
                           forgetting predictor sum(F_old * Delta-theta^2)
                           (Kirkpatrick et al., PNAS 2017).
  --data turn_<k>        : one TopiOCQA turn-bucket (v3 conversational
                           queries). Comparing top-p% F_bucket(k) masks with
                           the top-p% F_old mask tests the format-overlap
                           hypothesis at the parameter level: short-conv
                           buckets should overlap MSMARCO-important
                           parameters the most.

The loss whose gradients are accumulated is the SAME in-batch ranking loss
used in training (cal_ranking_loss): queries and their positive passages are
both encoded by the model, so no precomputed document embeddings are needed.
Gradients flow through both towers; F is accumulated post-backward per batch
on fp32 accumulators and saved as bf16.

Usage (needs 1+ GPU):
    python preprocess/analysis/fisher_importance.py --data msmarco --n_batches 250
    python preprocess/analysis/fisher_importance.py --data turn_1   --n_batches 250
"""

import sys, os, json, argparse, logging, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import build_qwen_instruct_query_ids, CONV_INSTRUCTION_V3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = ("/data/rech/huiyuche/huggingface/"
              "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
              "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
BUCKET_DIR = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
              "turn_buckets")
MSMARCO_TRAIN = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/msmarco/"
                 "msmarco_train.jsonl")
OUT_DIR = "/data/rech/huiyuche/continual_ir/figures/analysis"

# Official MTEB instruction for MSMARCO-style web queries (matches the
# query_instruction used when the BEIR corpora were encoded).
MSMARCO_INSTRUCTION = ("Given a web search query, retrieve relevant passages "
                       "that answer the query")

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=str, required=True,
                help="'msmarco' or a bucket name like 'turn_1' / 'turn_15plus'")
ap.add_argument("--n_batches", type=int, default=250)
ap.add_argument("--batch_size", type=int, default=16,
                help="query-passage pairs per batch (both towers are encoded "
                     "with gradients, so keep modest)")
ap.add_argument("--max_len", type=int, default=2048,
                help="token cap for both towers during Fisher estimation")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

DEVICE = torch.device("cuda:0")


def load_pairs():
    """Return list of (query_ids_list, pos_doc_text). query_ids_list is a
    ready token-id list (so conv queries use the exact v3 builder)."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pairs = []
    if args.data == "msmarco":
        import json as _json
        with open(MSMARCO_TRAIN) as f:
            for line in f:
                rec = _json.loads(line)
                # schema check at first record
                if not pairs:
                    logger.info(f"msmarco_train fields: {list(rec.keys())}")
                q = rec.get("query") or rec.get("question") or rec.get("cur_utt_text")
                pos = rec.get("pos_docs") or rec.get("positive_passages") or rec.get("pos_doc")
                if isinstance(pos, list):
                    pos = pos[0]
                if isinstance(pos, dict):
                    pos = pos.get("text") or pos.get("passage")
                if not q or not pos:
                    continue
                text = f"Instruct: {MSMARCO_INSTRUCTION}\nQuery:{q}"
                ids = tok.encode(text, truncation=True, max_length=args.max_len)
                pairs.append((ids, pos))
                if len(pairs) >= args.n_batches * args.batch_size * 2:
                    break
    else:
        import json as _json
        path = os.path.join(BUCKET_DIR, f"topiocqa_{args.data}.jsonl")
        with open(path) as f:
            for line in f:
                rec = _json.loads(line)
                ids = build_qwen_instruct_query_ids(
                    tok, rec["cur_utt_text"], rec["ctx_utts_text"],
                    CONV_INSTRUCTION_V3, max_length=args.max_len,
                    max_query_length=args.max_len, max_response_length=args.max_len,
                    template_version="v3")
                pos = rec["pos_docs"][0] if isinstance(rec["pos_docs"], list) else rec["pos_docs"]
                pairs.append((ids, pos))
    random.Random(args.seed).shuffle(pairs)
    logger.info(f"{args.data}: {len(pairs)} query-passage pairs loaded.")
    return tok, pairs


def left_pad(id_lists, pad_id):
    L = max(len(x) for x in id_lists)
    input_ids = torch.tensor([[pad_id] * (L - len(x)) + x for x in id_lists])
    attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in id_lists])
    return input_ids, attn


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok, pairs = load_pairs()
    pad_id = tok.pad_token_id

    model = AutoModel.from_pretrained(
        BASE_MODEL, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16).to(DEVICE)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    fisher = {n: torch.zeros(p.shape, dtype=torch.float32, device=DEVICE)
              for n, p in model.named_parameters() if p.requires_grad}

    n_done = 0
    for b in range(args.n_batches):
        batch = pairs[b * args.batch_size:(b + 1) * args.batch_size]
        if len(batch) < 2:
            break
        q_ids, q_mask = left_pad([x[0] for x in batch], pad_id)
        d_enc = tok([x[1] for x in batch], truncation=True,
                    max_length=args.max_len, padding=True, return_tensors="pt")

        def encode(input_ids, attn):
            out = model(input_ids=input_ids.to(DEVICE),
                        attention_mask=attn.to(DEVICE))
            if bool(attn[:, -1].all()):
                embs = out.last_hidden_state[:, -1, :]
            else:  # doc tower may be right-padded by tok(); pool at last real token
                lengths = attn.sum(dim=1) - 1
                embs = out.last_hidden_state[
                    torch.arange(len(lengths)), lengths.to(DEVICE)]
            return F.normalize(embs.float(), p=2, dim=-1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = encode(q_ids, q_mask)
            d_emb = encode(d_enc["input_ids"], d_enc["attention_mask"])
        scores = q_emb @ d_emb.T
        labels = torch.arange(len(batch), device=DEVICE)
        loss = torch.nn.CrossEntropyLoss()(scores, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n].addcmul_(p.grad.float(), p.grad.float())
        n_done += 1
        if (b + 1) % 25 == 0:
            logger.info(f"batch {b+1}/{args.n_batches} loss={loss.item():.4f}")

    out_path = os.path.join(OUT_DIR, f"fisher_{args.data}.pt")
    torch.save({"fisher": {n: t.to(torch.bfloat16).cpu() for n, t in fisher.items()},
                "n_batches": n_done, "batch_size": args.batch_size,
                "data": args.data, "max_len": args.max_len},
               out_path)
    logger.info(f"Fisher ({n_done} batches) -> {out_path}")


if __name__ == "__main__":
    main()
