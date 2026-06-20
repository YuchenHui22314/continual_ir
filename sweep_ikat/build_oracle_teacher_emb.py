"""ConvDR-KD teacher: base-qwen3 + web-search instruction encode 各年 oracle_utterance。
key = sample_id = f"{year}_{turn_id}" (对齐 ikat_graded manifest)。输出 L2-normed,落 base-qwen3 doc 流形。"""
import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
BASE="/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
D="/data/rech/huiyuche/TREC_iKAT_2024/data"
INSTR="Given a web search query, retrieve relevant passages that answer the query"
tok=AutoTokenizer.from_pretrained(BASE); tok.padding_side="left"
model=AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
items=[]
for y in ['23','24','25']:
    d=json.load(open(f"{D}/topics/ikat{y}/ikat_20{y}_test.json"))
    for t in d:
        if t.get('oracle_utterance'):
            items.append((f"{y}_{t['turn_id']}", t['oracle_utterance']))
print(f"{len(items)} oracle utterances across 3 years")
emb={}
with torch.no_grad():
    for i in range(0,len(items),64):
        b=items[i:i+64]
        qs=[f"Instruct: {INSTR}\nQuery:{o}" for _,o in b]
        x=tok(qs, add_special_tokens=True, max_length=32768, padding=True, truncation=True, return_tensors="pt")
        o=model(input_ids=x["input_ids"].cuda(), attention_mask=x["attention_mask"].cuda())
        e=F.normalize(o.last_hidden_state[:,-1,:].float(),dim=-1).cpu()
        for (sid,_),ee in zip(b,e): emb[sid]=ee
out=f"{D}/ikat_graded/ikat_oracle_teacher_emb.pt"
torch.save(emb,out)
print(f"saved {out}: {len(emb)} teacher embs, dim={list(emb.values())[0].shape}")
# 对齐自检: 280 train sample_id 命中率
man=[json.loads(l) for l in open(f"{D}/ikat_graded/ikat_graded_ptkb.jsonl") if json.loads(l)['split']=='train']
hit=sum(1 for r in man if r['sample_id'] in emb)
print(f"train sample_id 命中 teacher: {hit}/{len(man)}")
