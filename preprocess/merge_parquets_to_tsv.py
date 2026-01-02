import os
import glob
import pandas as pd

# ==========================
# Configuration
# ==========================

# Directory where your parquet files are stored
PARQUET_DIR = "/data/rech/huiyuche/topiocqa_wiki_collection"

# Output TSV path
OUTPUT_TSV = "topiocqa_wiki_collection.tsv"

# ==========================
# End of configuration
# ==========================

def load_all_parquets(parquet_dir):
    """
    Load all parquet files in the given directory.

    Files are sorted to ensure consistent merging order.
    """
    # Find all files ending with .parquet
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f" Found {len(files)} parquet files in {parquet_dir}")

    # List to accumulate all dataframes
    df_list = []
    for file in files:
        print(f" Loading {os.path.basename(file)}...")
        # Read only the necessary columns to save memory
        df = pd.read_parquet(file, columns = ["id", "title", "sub_title", "contents"])
        
        df_list.append(df)

    # Concatenate all dataframes into one
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def main():
    # Load all parquet files
    df = load_all_parquets(PARQUET_DIR)

    # Drop any rows where 'id' or 'contents' is missing
    df = df.dropna(subset=["id", "contents"])

    # Combine title + sub_title + contents into one text
    print("Concatenating title, sub_title, and contents ...")
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["sub_title"].fillna("") + " " +
        df["contents"].fillna("")
    ).str.strip()

    # Count total passages
    total_passages = len(df)
    print(f"\n Total number of passages: {total_passages:,}")

    # Save to TSV file
    print(f" Writing data to {OUTPUT_TSV}...")
    df[["id", "combined_text"]].to_csv(OUTPUT_TSV, sep="\t", index=False)

    print("\n Finished successfully!")

if __name__ == "__main__":
    main()
