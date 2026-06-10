"""
build_topiocqa_turn_buckets.py
==============================
Split the TopiOCQA train set into 13 equal-size single-length buckets for the
per-turn-length forgetting experiment:

    turn 1, 2, ..., 10        (each ~3.3-3.5k samples natively)
    turns 11-12 merged        (4,497 natively)
    turns 13-14 merged        (3,054 natively  <- smallest, sets N)
    turns 15+   merged        (3,479 natively)

Every bucket is downsampled (fixed seed) to N = size of the smallest bucket
(13-14 = 3,054), so all 13 training sets have IDENTICAL size. Rationale: the
bucket runs train for a fixed number of steps with a cycling iterator, so
unequal bucket sizes would mean unequal per-sample repetition rates (a
confound with conversation length). Equal size + equal steps => the only
varying factor is conversation length.

turn = 1 + len(ctx_utts_text) // 2   (same formula as data.py:286 /
curriculum.py score_by_turn_length).

Outputs:
    /data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/turn_buckets/
        topiocqa_turn_1.jsonl ... topiocqa_turn_10.jsonl
        topiocqa_turn_11_12.jsonl
        topiocqa_turn_13_14.jsonl
        topiocqa_turn_15plus.jsonl
        manifest.json   (per-bucket native/sampled counts, seed, source path)
Also appends the bucket statistics to the shared data-analysis log
(/data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log).

Usage:
    python preprocess/data/build_topiocqa_turn_buckets.py
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime

TRAIN_FILE = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
              "topiocqa_train_oracle.jsonl")
OUT_DIR    = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/"
              "turn_buckets")
ANALYSIS_LOG = "/data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log"
SEED = 42

# bucket name -> membership predicate on turn number
BUCKETS = {f"turn_{t}": (lambda t0: (lambda turn: turn == t0))(t) for t in range(1, 11)}
BUCKETS["turn_11_12"]  = lambda turn: turn in (11, 12)
BUCKETS["turn_13_14"]  = lambda turn: turn in (13, 14)
BUCKETS["turn_15plus"] = lambda turn: turn >= 15


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) read + group raw lines by bucket (keep raw lines so output records are
    #    byte-identical to the source — the Topiocqa dataset class consumes
    #    them unchanged).
    grouped = defaultdict(list)
    total = 0
    with open(TRAIN_FILE) as f:
        for line in f:
            rec = json.loads(line)
            turn = 1 + len(rec["ctx_utts_text"]) // 2
            for name, pred in BUCKETS.items():
                if pred(turn):
                    grouped[name].append(line)
                    break
            total += 1

    native_counts = {name: len(lines) for name, lines in grouped.items()}
    n_target = min(native_counts.values())
    print(f"source: {TRAIN_FILE} ({total} records)")
    print(f"equalized bucket size N = {n_target} (= smallest bucket "
          f"{min(native_counts, key=native_counts.get)})")

    # 2) downsample each bucket to n_target with a fixed seed and write.
    rng = random.Random(SEED)
    manifest = {
        "source": TRAIN_FILE,
        "seed": SEED,
        "n_per_bucket": n_target,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "buckets": {},
    }
    for name in BUCKETS:
        lines = grouped[name]
        sampled = rng.sample(lines, n_target) if len(lines) > n_target else list(lines)
        out_path = os.path.join(OUT_DIR, f"topiocqa_{name}.jsonl")
        with open(out_path, "w") as f:
            f.writelines(sampled)
        manifest["buckets"][name] = {
            "native": len(lines),
            "sampled": len(sampled),
            "file": out_path,
        }
        print(f"  {name:<14} native={len(lines):>5}  sampled={len(sampled):>5}  -> {out_path}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {os.path.join(OUT_DIR, 'manifest.json')}")

    # 3) append stats to the shared data-analysis log.
    with open(ANALYSIS_LOG, "a") as f:
        f.write(f"\n[{manifest['created']}] TopiOCQA turn-bucket split "
                f"(build_topiocqa_turn_buckets.py, seed={SEED})\n")
        f.write(f"  source: {TRAIN_FILE} ({total} records); "
                f"equalized N={n_target} per bucket\n")
        for name, info in manifest["buckets"].items():
            f.write(f"  {name:<14} native={info['native']:>5} "
                    f"sampled={info['sampled']:>5}\n")
    print(f"stats appended to {ANALYSIS_LOG}")


if __name__ == "__main__":
    main()
