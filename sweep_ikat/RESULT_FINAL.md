# iKAT personalized conv dense retriever — 最终结论 (2026-06-20)
## 战绩: conv ZS 25.3 -> 最优 32.4 (oracle 上界 36.3 的 88%)
锚点: base-qwen3 ZS 12.9 / conv-qwen3 ZS 25.3 (ptkb) / oracle 上界 36.3 (base-qwen3 encode oracle rewrite).
最优 = 32.4: round2 kd2_bx1_kd2_a1_g2048_e20 @ep13 / round3 kd3_r16_kd2_a1_e30 @ep13.
配方: BiXSE 1.0 + KD 2.0(cos, 拉向 oracle-rewrite teacher) + anchor 1.0(防崩) + 随机 gneg 2048
      + LoRA r16 + lr 5e-6 + conv-qwen3 init + FULL ptkb form.

## 关键发现 (论文可写)
1. 必须 conv init + 防崩(anchor+gneg): 裸 graded-ft 必崩(全库 ~1-5). 
2. 必须 FULL ptkb(非 rel=oracle): rel 上 conv ZS 已 26.0 无空间; full(25.3,带噪声)有 personalization 空间.
3. KD 拉向 oracle teacher 是核心增量: 27.1->32.4 (KD2 + BiXSE 互补, 纯 KD 只 28.5).
4. oracle 上界 36.3 robust: 加 rel-PTKB(33.6)/换 instruction(35.7) 都抬不动 -> oracle rewrite 已最优.
5. 数据瓶颈: r16 > r32 > r64 (大容量过拟合 280 turn).
6. 不再涨的方法(全部验证): listwise KD 31.7, PTKB-subset aug 32.0, hard-neg 29.6(false-neg 反伤).
7. 剩余 gap 3.9 本质: student(长噪声 full-ptkb conv) vs teacher(短 clean oracle) 的 input 信息差
   + 仅 280 train turn + query-only on FROZEN base-qwen3 doc index.

## 最优 ckpt
/part/01/Tmp/yuchen/continual_ir/kd3_r16_kd2_a1_e30/  (ep13 = 32.4) — LoRA, AutoModel 可加载.
