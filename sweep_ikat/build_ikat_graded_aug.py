"""round4 数据增强 (PTKB-subset aug): round3 诊断=数据瓶颈(280 turn 太少, 大容量过拟合).
每个 train turn 生成 K 个噪声 PTKB 子集视图 —— 保留 gold-relevant(ptkb_provenance), 随机保留部分
噪声 persona —— 重构 query_text. 教模型"从任意噪声 profile 子集映射到同一 personalized 表示"
(同一 oracle teacher / anchor / candidates). val 不增强. by-conversation split 防泄漏(aug 复用原 conv_id)."""
import sys, json, random, torch
sys.path.insert(0, "/data/rech/huiyuche/TREC_iKAT_2024/src")
from apcir.preprocess.build_ikat_graded_training import collect_kept_turns, build_query_text

REPO = "/data/rech/huiyuche/TREC_iKAT_2024"; D = f"{REPO}/data/ikat_graded"
K = 3; random.seed(42)

man = [json.loads(l) for l in open(f"{D}/ikat_graded_ptkb.jsonl")]
train = {r["sample_id"]: r for r in man if r["split"] == "train"}
kept, _ = collect_kept_turns(["23", "24", "25"], drop_minus1=True)
turn_by_sid = {f"{k['year']}_{k['qid']}": k for k in kept}
teacher = torch.load(f"{D}/ikat_oracle_teacher_emb.pt")
anchor = torch.load(f"{D}/anchor_query_emb_conv_ptkb.pt")

aug = []; n_skip = 0
for sid, r in train.items():
    k = turn_by_sid.get(sid)
    if k is None or sid not in teacher or sid not in anchor:
        n_skip += 1; continue
    turn = k["turn"]; year = k["year"]; full = dict(turn.ptkb)
    prov = set(str(i) for i in (turn.ptkb_provenance or []))
    gold = [kk for kk in full if kk in prov]
    noise = [kk for kk in full if kk not in prov]
    for j in range(K):
        keep = random.sample(noise, random.randint(1, len(noise))) if noise else []
        sub = {kk: full[kk] for kk in full if kk in set(gold) | set(keep)}
        if not sub:
            sub = full
        turn.ptkb = sub
        _, q = build_query_text(turn, year, "ptkb")
        turn.ptkb = full
        asid = f"{sid}_aug{j}"
        aug.append(dict(sample_id=asid, qid=r["qid"], conversation_id=r["conversation_id"],
                        year=r["year"], query_text=q, candidates=r["candidates"],
                        split="train", template=r["template"]))
        teacher[asid] = teacher[sid]   # same oracle teacher (the invariance target)
        anchor[asid] = anchor[sid]     # same anchor (full-ptkb conv emb; anti-collapse only)

with open(f"{D}/ikat_graded_ptkb_aug.jsonl", "w") as f:
    for r in man: f.write(json.dumps(r) + "\n")          # original train + val (val untouched)
    for r in aug: f.write(json.dumps(r) + "\n")          # aug (train only)
torch.save(teacher, f"{D}/ikat_oracle_teacher_emb_aug.pt")
torch.save(anchor, f"{D}/anchor_query_emb_conv_ptkb_aug.pt")
print(f"train {len(train)} x{K} = {len(aug)} aug (skip {n_skip}); total train = {len(train)+len(aug)}")
print(f"teacher dict {len(teacher)}, anchor dict {len(anchor)}; manifest -> ikat_graded_ptkb_aug.jsonl")
