"""
Merge Qwen3-Embedding-0.6B TopiOCQA corpus shards.

Reuses `merge_blocks_to_large_blocks` from the reference indexer at
  /data/rech/huiyuche/TREC_iKAT_2024/src/apcir/indexing/dense/distributed_dense_index.py

Input layout  (4 ranks x 26 blocks = 208 shards):
    doc_emb_block.rank_{R}.{B}.pb   shape=(~250k, 1024) float32
    doc_embid_block.rank_{R}.{B}.pb list[str] length ~250k

Output layout (26 merged blocks, ~1M docs per block to match ANCE merged layout):
    doc_emb_block.{B}.pb
    doc_embid_block.{B}.pb
"""
import os
import sys

# make `from utils import pstore, pload` work (same as distributed_dense_index.py)
APCIR_DENSE = "/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/indexing/dense"
sys.path.insert(0, APCIR_DENSE)

from distributed_dense_index import merge_blocks_to_large_blocks  # noqa: E402

INPUT_DIR  = "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_emb_0.6"         # local SSD copy of the unmerged shards
OUTPUT_DIR = "/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged"
NUM_BLOCK  = 26
NUM_RANK   = 4
EXPECTED_NUM_DOC_PER_BLOCK = 1_000_000  # matches topiocqa_ance_merged layout

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[merge] input  = {INPUT_DIR}")
    print(f"[merge] output = {OUTPUT_DIR}")
    print(f"[merge] num_block={NUM_BLOCK}  num_rank={NUM_RANK}  per_block={EXPECTED_NUM_DOC_PER_BLOCK}")
    merge_blocks_to_large_blocks(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        num_block=NUM_BLOCK,
        num_rank=NUM_RANK,
        expected_num_doc_per_block=EXPECTED_NUM_DOC_PER_BLOCK,
    )
    print("[merge] done.")
