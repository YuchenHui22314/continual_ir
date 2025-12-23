import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
from models import ANCE
import torch.distributed as dist


def batch_encode(texts, tokenizer, model, device, max_len):
    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embs = model(**inputs)

    return embs.cpu()


def main(args):
    # --------
    # Distributed setup (no DDP!)
    # --------

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = RobertaTokenizer.from_pretrained(args.encoder_path)
    model = ANCE.from_pretrained(args.encoder_path).to(device)
    model.eval()

    # --------
    # Load dataset lines
    # --------
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --------
    # Shard data by rank
    # --------
    lines = lines[local_rank::world_size]

    sample_embeddings = {}

    batch_texts = []
    batch_meta = []  # (sample_id, field)

    def flush():
        nonlocal batch_texts, batch_meta
        if not batch_texts:
            return

        embs = batch_encode(
            batch_texts,
            tokenizer,
            model,
            device,
            args.max_doc_length
        )

        for (sid, field), emb in zip(batch_meta, embs):
            if sid not in sample_embeddings:
                sample_embeddings[sid] = {
                    "pos": None,
                    "neg": None,
                    "oracle": None
                }
            sample_embeddings[sid][field] = emb

        batch_texts.clear()
        batch_meta.clear()

    for line in tqdm(lines, desc=f"Rank {local_rank} encoding", total=len(lines)):
        record = json.loads(line)
        sid = record["sample_id"]

        batch_texts.append(record["pos_docs"][0])
        batch_meta.append((sid, "pos"))

        if "bm25_hard_neg_docs" in record and record["bm25_hard_neg_docs"]:
            batch_texts.append(record["bm25_hard_neg_docs"][0])
            batch_meta.append((sid, "neg"))

        if "oracle_utt_text" in record:
            batch_texts.append(record["oracle_utt_text"])
            batch_meta.append((sid, "oracle"))

        if len(batch_texts) >= args.encode_batch_size:
            flush()

    flush()

    # --------
    # Save per-rank file
    # --------
    out_file = f"{args.output_embedding_file}.rank{local_rank}.pt"
    torch.save(sample_embeddings, out_file)

    print(f"[Rank {local_rank}] Saved {len(sample_embeddings)} samples to {out_file}")

    # --------
    # Synchronize before merge
    # --------

    if world_size > 1:
        dist.barrier()

    # --------
    # Merge on rank 0
    # --------
    if local_rank == 0 and world_size > 1:
        print("Merging embeddings from all ranks...")
        merged = {}

        for r in range(world_size):
            part = torch.load(f"{args.output_embedding_file}.rank{r}.pt")
            merged.update(part)

        torch.save(merged, args.output_embedding_file)
        print(f"Final merged embeddings saved to {args.output_embedding_file}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_embedding_file", type=str, required=True)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--encode_batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args)

'''
torchrun --nproc_per_node=4 encode_pos_neg_docs_batched.py \
  --encoder_path /data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234 \
  --dataset_file /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl \
  --output_embedding_file /data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings_multi_GPU.pt \
  --max_doc_length 512 \
  --encode_batch_size 512
'''
