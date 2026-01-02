import os
import json
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_parquets(parquet_dir: str) -> pd.DataFrame:
    """
    Load all parquet files under a directory into a single DataFrame.
    """
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    assert len(parquet_files) == 7, f"Expected 7 parquet files, got {len(parquet_files)}"

    dfs = []
    for pf in parquet_files:
        print(f"Loading {pf}")
        dfs.append(pd.read_parquet(pf))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total rows: {len(df)}")
    return df


def has_any_positive_is_selected(passages: dict) -> bool:
    """
    Check whether passages['is_selected'] contains at least one non-zero element.
    """
    is_selected = passages.get("is_selected", [])
    return any(x != 0 for x in is_selected)


def build_pos_neg_docs(passages: dict):
    """
    Build positive and negative document lists based on is_selected.
    """
    is_selected = passages["is_selected"]
    texts = passages["passage_text"]

    pos_docs = []
    neg_docs = []

    for flag, text in zip(is_selected, texts):
        if flag != 0:
            pos_docs.append(text)
        else:
            neg_docs.append(text)

    return pos_docs, neg_docs


def main():
    parquet_dir = "./"
    output_jsonl = "msmarco_train.jsonl"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_all_parquets(parquet_dir)

    # ------------------------------------------------------------------
    # 2. Statistics and filtering condition check
    # ------------------------------------------------------------------
    mask_has_pos = df["passages"].apply(has_any_positive_is_selected)

    df = df[mask_has_pos].reset_index(drop=True)

    print(f"Remaining rows after filtering: {len(df)}")

    # ------------------------------------------------------------------
    # 3. Statistics on is_selected
    # ------------------------------------------------------------------
    non_zero_counts = []
    lengths = []

    for passages in df["passages"]:
        is_selected = passages["is_selected"]
        lengths.append(len(is_selected))
        non_zero_counts.append(sum(1 for x in is_selected if x != 0))

    avg_non_zero = sum(non_zero_counts) / len(non_zero_counts)
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    print("Average number of positive passages:", avg_non_zero)
    print("Average is_selected length:", avg_len)
    print("Max is_selected length:", max_len)
    print("Min is_selected length:", min_len)

    # ------------------------------------------------------------------
    # 4. Generate JSONL
    # ------------------------------------------------------------------
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            passages = row["passages"]

            pos_docs, neg_docs = build_pos_neg_docs(passages)

            #  numpy array to list
            passages_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in passages.items()
            }

            out_record = {
                "query_id": int(row["query_id"]),
                "query": row["query"],
                "pos_docs": pos_docs,
                "bing_hard_neg_docs": neg_docs,
                # keep original fields unchanged
                "passages": passages_serializable,
                "query_type": row["query_type"],
                "answers": row["answers"].tolist() if isinstance(row["answers"], np.ndarray) else row["answers"],
                "wellFormedAnswers": row["wellFormedAnswers"].tolist() if isinstance(row["wellFormedAnswers"], np.ndarray) else row["wellFormedAnswers"],
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"JSONL written to: {output_jsonl}")


if __name__ == "__main__":
    main()
