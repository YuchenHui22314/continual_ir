"""
Build Qwen3-Embedding-0.6B pos/neg/oracle embedding table for TopiOCQA.

- pos/neg: looked up directly from the merged Qwen3 corpus (no GPU needed)
- oracle: encoded on GPU (oracle utterances are not in the passage corpus)

Output format: {sample_id: {"pos": Tensor(1024,), "neg": Tensor(1024,), "oracle": Tensor(1024,)}}
  Same format as the ANCE embeddings.pt, just 1024-dim instead of 768.

Run single-process (no DDP needed since corpus lookup is pure CPU):
  python extract_qwen_pos_neg_from_corpus.py \
    --corpus_dir  /part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged \
    --train_file  /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl \
    --model_path  /data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 \
    --output_file /data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt
"""

import os
import gc
import sys
import json
import glob
import argparse
import pickle
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build in-memory {doc_id → (emb: np.ndarray[1024])} lookup table
# ---------------------------------------------------------------------------

def build_lookup_table(corpus_dir: str, needed_ids: set) -> dict:
    """
    Stream through all merged corpus blocks and collect only the needed embeddings.
    Returns {doc_id_str: np.ndarray shape (1024,)}.

    We avoid loading the full corpus into RAM by scanning id blocks first and
    then loading only the required row slices from the emb blocks.
    """
    id_files  = sorted(glob.glob(os.path.join(corpus_dir, "doc_embid_block.*.pb")))
    emb_files = sorted(glob.glob(os.path.join(corpus_dir, "doc_emb_block.*.pb")))
    assert len(id_files) == len(emb_files), "Mismatch between id and emb block counts"
    logger.info(f"Scanning {len(id_files)} blocks for {len(needed_ids)} doc ids …")

    lookup = {}  # doc_id_str → np.ndarray(1024,)
    remaining = set(needed_ids)

    for id_file, emb_file in tqdm(zip(id_files, emb_files), total=len(id_files),
                                   desc="Scanning blocks"):
        ids = pickle.load(open(id_file, "rb"))  # list[str]

        # Quick check: does this block contain any needed ids?
        block_set = set(ids)
        hits = remaining & block_set
        if not hits:
            continue  # skip loading the embedding block

        embs = pickle.load(open(emb_file, "rb"))  # np.ndarray (N, 1024) float32
        for row, doc_id in enumerate(ids):
            if doc_id in hits:
                lookup[doc_id] = embs[row].copy()
                remaining.discard(doc_id)
        del embs
        gc.collect()

        if not remaining:
            logger.info("All needed doc ids found. Early stop.")
            break

    if remaining:
        logger.warning(f"{len(remaining)} doc ids NOT found in corpus: {list(remaining)[:5]} …")

    logger.info(f"Lookup table built: {len(lookup)} entries")
    return lookup


# ---------------------------------------------------------------------------
# Step 2: Encode oracle utterances with Qwen3-Embedding (GPU)
# ---------------------------------------------------------------------------

def last_token_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Last-token pooling for left-padded sequences (FlashAttn requires left padding)."""
    return hidden_states[:, -1, :]


def encode_oracle_utterances(texts: list, model_path: str, batch_size: int,
                              max_length: int, device: torch.device) -> dict:
    """
    Encode oracle utterances with Qwen3-Embedding.
    Returns {index_in_texts → np.ndarray(1024,)}.
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Loading Qwen3 tokenizer + model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # FlashAttention 2 requires left-padding; last real token is always at position -1.
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    logger.info("Model loaded.")

    results = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding oracle"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        embs = last_token_pool(out.last_hidden_state)
        embs = F.normalize(embs, p=2, dim=-1).float().cpu()
        for i, emb in enumerate(embs):
            results[start + i] = emb.numpy()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ---- Load training file ----
    logger.info(f"Reading {args.train_file} …")
    records = []
    with open(args.train_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    logger.info(f"Total records: {len(records)}")

    # ---- Collect needed doc ids (pos + neg) ----
    # Each record has: pos_docs_pids: [int], bm25_hard_neg_docs_pids: [int, ...]
    # Corpus ids are strings "wiki:{pid}"
    needed_ids = set()
    for rec in records:
        needed_ids.add(f"wiki:{rec['pos_docs_pids'][0]}")
        needed_ids.add(f"wiki:{rec['bm25_hard_neg_docs_pids'][0]}")
    logger.info(f"Need {len(needed_ids)} unique doc ids from corpus")

    # ---- Build corpus lookup ----
    lookup = build_lookup_table(args.corpus_dir, needed_ids)

    # ---- Collect oracle utterances ----
    oracle_texts  = [rec["oracle_utt_text"] for rec in records]
    oracle_sample_ids = [rec["sample_id"] for rec in records]

    # ---- Encode oracle utterances on GPU ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_embs = encode_oracle_utterances(
        oracle_texts, args.model_path, args.oracle_batch_size, args.max_oracle_length, device
    )

    # ---- Assemble output dict ----
    output = {}
    missing_pos = 0
    missing_neg = 0
    for i, rec in enumerate(tqdm(records, desc="Assembling embeddings")):
        sid  = rec["sample_id"]
        p_id = f"wiki:{rec['pos_docs_pids'][0]}"
        n_id = f"wiki:{rec['bm25_hard_neg_docs_pids'][0]}"

        pos_emb = lookup.get(p_id)
        neg_emb = lookup.get(n_id)
        oracle_emb = oracle_embs[i]

        if pos_emb is None:
            missing_pos += 1
            pos_emb = np.zeros(1024, dtype=np.float32)
        if neg_emb is None:
            missing_neg += 1
            neg_emb = np.zeros(1024, dtype=np.float32)

        output[sid] = {
            "pos":    torch.from_numpy(pos_emb),
            "neg":    torch.from_numpy(neg_emb),
            "oracle": torch.from_numpy(oracle_emb),
        }

    logger.info(f"Missing pos: {missing_pos}, missing neg: {missing_neg}")
    logger.info(f"Saving {len(output)} entries to {args.output_file} …")
    torch.save(output, args.output_file)
    logger.info("Done.")

    # Sanity check
    sample_sid = next(iter(output))
    for k in ["pos", "neg", "oracle"]:
        assert output[sample_sid][k].shape == (1024,), \
            f"Unexpected shape: {output[sample_sid][k].shape}"
    logger.info(f"Sanity check passed. Sample: {sample_sid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir",       type=str, required=True,
                        help="Merged Qwen corpus dir with doc_emb_block.{B}.pb files")
    parser.add_argument("--train_file",       type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl")
    parser.add_argument("--model_path",       type=str,
                        default="/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
    parser.add_argument("--output_file",      type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt")
    parser.add_argument("--oracle_batch_size", type=int, default=256)
    parser.add_argument("--max_oracle_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
