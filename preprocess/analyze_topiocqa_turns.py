"""
analyze_topiocqa_turns.py

Analyze the turn-length distribution of the TopiOCQA training set.
This is used to determine the initial pacing fraction (c0 / delta_p) for curriculum learning.

The "turn number" of a training example = 1 + len(ctx_utts_text) // 2
  - Turn 1: no history (ctx_utts_text is empty) → easiest
  - Turn 2: 1 prior Q+A pair → moderate
  - Turn N: N-1 prior Q+A pairs → harder

Output:
  - Per-turn-length count and percentage (printed to stdout)
  - Cumulative percentage (helps choose c0: fraction of easy data to start curriculum with)
  - Saved to data analysis log

Usage:
  python analyze_topiocqa_turns.py \
    --data_file /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl \
    --output_log /data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log
"""

import json
import argparse
import os
from collections import Counter
from datetime import datetime


def analyze_turn_distribution(data_file: str, output_log: str):
    """
    Load the JSONL file, compute turn-length distribution, and print/save stats.
    """
    print(f"Loading data from: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    turn_counts = Counter()

    for line in lines:
        record = json.loads(line.strip())
        # ctx_utts_text is a list of alternating questions and responses from prior turns.
        # [q1, r1, q2, r2, ..., q_{n-1}, r_{n-1}]
        # Turn number for current example = 1 + prior_pairs
        ctx_utts = record.get("ctx_utts_text", [])
        turn_number = 1 + len(ctx_utts) // 2
        turn_counts[turn_number] += 1

    # Build sorted output
    sorted_turns = sorted(turn_counts.keys())
    max_turn = sorted_turns[-1]

    lines_out = []
    lines_out.append(f"\n{'='*60}")
    lines_out.append(f"TopiOCQA Turn-Length Distribution")
    lines_out.append(f"Dataset: {data_file}")
    lines_out.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines_out.append(f"{'='*60}")
    lines_out.append(f"Total examples: {total}")
    lines_out.append(f"Max turn number: {max_turn}")
    lines_out.append(f"{'='*60}")
    lines_out.append(f"{'Turn':>6}  {'Count':>7}  {'Pct (%)':>8}  {'Cumul (%)':>10}")
    lines_out.append(f"{'-'*40}")

    cumulative = 0
    for turn in sorted_turns:
        count = turn_counts[turn]
        pct = 100.0 * count / total
        cumulative += pct
        lines_out.append(f"{turn:>6}  {count:>7}  {pct:>8.2f}%  {cumulative:>9.2f}%")

    lines_out.append(f"{'='*60}")
    lines_out.append(f"")
    lines_out.append(f"Curriculum Learning Hint:")
    lines_out.append(f"  c0 = fraction of 'easy' data at curriculum start.")
    lines_out.append(f"  If starting with Turn-1 only: c0 = {100.0*turn_counts[1]/total:.2f}%")
    turn1_plus2 = turn_counts.get(1, 0) + turn_counts.get(2, 0)
    lines_out.append(f"  If starting with Turn 1+2:    c0 = {100.0*turn1_plus2/total:.2f}%")
    turn1_plus3 = turn1_plus2 + turn_counts.get(3, 0)
    lines_out.append(f"  If starting with Turn 1+2+3:  c0 = {100.0*turn1_plus3/total:.2f}%")
    lines_out.append(f"{'='*60}\n")

    output_str = "\n".join(lines_out)
    print(output_str)

    # Save to data analysis log
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "a", encoding="utf-8") as f:
        f.write(output_str)
    print(f"[INFO] Results appended to: {output_log}")

    return turn_counts, total


def main():
    parser = argparse.ArgumentParser(description="Analyze TopiOCQA turn-length distribution.")
    parser.add_argument(
        "--data_file",
        type=str,
        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl",
        help="Path to TopiOCQA training JSONL file."
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default="/data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log",
        help="Path to data analysis log file (results will be appended)."
    )
    args = parser.parse_args()
    analyze_turn_distribution(args.data_file, args.output_log)


if __name__ == "__main__":
    main()
