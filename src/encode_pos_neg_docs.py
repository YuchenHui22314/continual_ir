import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from models import ANCE


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(args.encoder_path)
    model = ANCE.from_pretrained(args.encoder_path).to(device)
    model.eval()

    doc_embeddings = {}
    doc_id_map = {}

    with open(args.dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding documents"):
            record = json.loads(line)

            # collect pos + neg docs
            docs = []
            if "pos_docs" in record:
                docs.append(record["pos_docs"][0])
            if "bm25_hard_neg_docs" in record:
                docs.extend(record["bm25_hard_neg_docs"][0])
            if "oracle_utt_text" in record:
                docs.append(record["oracle_utt_text"])
            
            sample_id = record["sample_id"]

            for doc_text in docs:
                if doc_text in doc_id_map:
                    continue

                inputs = tokenizer(
                    doc_text,
                    truncation=True,
                    padding=False,
                    max_length=args.max_doc_length,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    emb = model(**inputs)
                    emb = emb.squeeze(0).cpu().numpy()

                doc_id_map[doc_text] = sample_id
                doc_embeddings[sample_id] = emb
                print(emb.shape)
                exit()

    # save
    np.save(args.output_embedding_file, doc_embeddings)
    json.dump(doc_id_map, open(args.output_id_map_file, "w"))

    print(f"Saved {len(doc_embeddings)} document embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_embedding_file", type=str, required=True)
    parser.add_argument("--output_id_map_file", type=str, required=True)
    parser.add_argument("--max_doc_length", type=int, default=512)

    args = parser.parse_args()
    main(args)

### Example usage:
'''
 python encode_pos_neg_docs.py \
     --encoder_path /data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234" \
     --dataset_file /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl \
     --output_embedding_file /data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.pt \
     --max_doc_length 512  
'''