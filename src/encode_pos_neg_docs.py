import json
import torch
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
from models import ANCE


def encode_text(text, tokenizer, model, device, max_len):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        emb = model(**inputs)          # [1, dim]
        emb = emb.squeeze(0).cpu()     # [dim]

    return emb


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(args.encoder_path)
    model = ANCE.from_pretrained(args.encoder_path).to(device)
    model.eval()

    # Final table: sample_id -> {pos, neg, oracle}
    sample_embeddings = {}

    with open(args.dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding pos / neg / oracle embeddings"):
            record = json.loads(line)
            sample_id = record["sample_id"]

            entry = {}

            # --------
            # Positive document (must exist)
            # --------
            pos_text = record["pos_docs"][0]
            entry["pos"] = encode_text(
                pos_text, tokenizer, model, device, args.max_doc_length
            )

            # --------
            # Negative document (optional)
            # --------
            if "bm25_hard_neg_docs" in record and len(record["bm25_hard_neg_docs"]) > 0:
                neg_text = record["bm25_hard_neg_docs"][0]
                entry["neg"] = encode_text(
                    neg_text, tokenizer, model, device, args.max_doc_length
                )
            else:
                entry["neg"] = None

            # --------
            # Oracle utterance (optional)
            # --------
            if "oracle_utt_text" in record:
                entry["oracle"] = encode_text(
                    record["oracle_utt_text"],
                    tokenizer,
                    model,
                    device,
                    args.max_doc_length
                )
            else:
                entry["oracle"] = None

            sample_embeddings[sample_id] = entry

    # Save as a single torch file (recommended)
    torch.save(sample_embeddings, args.output_embedding_file)

    print(f"Saved embeddings for {len(sample_embeddings)} samples.")
    print(f"Output file: {args.output_embedding_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_embedding_file", type=str, required=True)
    parser.add_argument("--max_doc_length", type=int, default=512)

    args = parser.parse_args()
    main(args)

### Example usage:
'''
python encode_pos_neg_docs.py \
--encoder_path /data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234 \
--dataset_file /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl \
--output_embedding_file /data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.pt \
--max_doc_length 512
'''