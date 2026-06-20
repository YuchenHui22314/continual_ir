"""round5 hard-neg pool: 从 train 280 turn 的 oracle 全库 ranking 抽 hard negatives
(rank 20-150, 去掉 judged), 用 build_lookup_table 从 /part SSD blocks reconstruct base-qwen3 emb,
存成 (N,1024) L2-normed tensor —— 与 global_neg_pool 同格式, 训练时 --ikat_global_neg_file 指向它即可
(零训练代码改动). 对症 round4 诊断: student 训在随机 easy negs -> 学不会压 false positives;
换成 oracle 检索的 hard docs(全库真实分布的高分干扰)就直接攻击这个 gap."""
import sys, json, collections, random, torch
import torch.nn.functional as F
sys.path.insert(0, "/data/rech/huiyuche/TREC_iKAT_2024/src")
from apcir.preprocess.build_ikat_graded_training import build_lookup_table

REPO = "/data/rech/huiyuche/TREC_iKAT_2024"; D = f"{REPO}/data/ikat_graded"
CORPUS = "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_qwen_merged"   # /part SSD (fast)
RANK = (f"{REPO}/results/ClueWeb_ikat/perso_dense_train/ranking/"
        "S1[oracle_qwen_instruct]-S2[none]-g[none]-[qwen3]-[none_4_1_none]-[s2_top50].txt")
RMIN, RMAX, POOL_CAP = 20, 150, 50000
random.seed(42)

# 1. judged docid per turn (drop these — they're the turn's own labeled docs)
man = [json.loads(l) for l in open(f"{D}/ikat_graded_ptkb.jsonl")]
judged = {r["sample_id"]: set(d for d, g in r["candidates"])
          for r in man if r["split"] == "train" and "_aug" not in r["sample_id"]}

# 2. read train oracle ranking, collect hard-neg docid (rank 20-150, drop judged)
rank_by_q = collections.defaultdict(list)
for line in open(RANK):
    p = line.split()
    rank_by_q[p[0]].append((int(p[3]), p[2]))
hard = set()
for qid, rl in rank_by_q.items():
    jd = judged.get(qid, set())
    for rank, docid in rl:
        if RMIN <= rank <= RMAX and docid not in jd:
            hard.add(docid)
print(f"{len(rank_by_q)} train queries; {len(hard)} unique hard-neg docids (rank {RMIN}-{RMAX}, judged dropped)")

# 3. cap pool size
if len(hard) > POOL_CAP:
    hard = set(random.sample(sorted(hard), POOL_CAP))
    print(f"capped pool to {len(hard)}")

# 4. reconstruct base-qwen3 emb from /part SSD blocks (same source as the 46946 judged embs)
lookup = build_lookup_table(CORPUS, hard)
print(f"reconstructed {len(lookup)}/{len(hard)} emb from blocks")

# 5. stack + L2-norm + save (global_neg_pool format)
embs = torch.stack([torch.from_numpy(lookup[d]) for d in lookup]).float()
embs = F.normalize(embs, dim=-1)
out = f"{D}/ikat_oracle_hardneg_pool.pt"
torch.save(embs, out)
print(f"saved {out}: {tuple(embs.shape)}  (use as --ikat_global_neg_file)")
