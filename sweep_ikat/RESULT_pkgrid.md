# full-ptkb 网格 (anchor+gneg, conv init) — 突破, 2026-06-20
锚点(ptkb form): base=12.9, conv=25.3, rel-oracle=26.0.
8/8 config best 都涨过 conv ptkb 25.3! 最高 pk_a1_g2048_lr5e6 @ep4=26.8 (> rel oracle 26.0).
  pk_a1_g2048_lr5e6  @ep4 = 26.8  逐epoch后期稳~21 (anchor1+gneg2048+lr5e-6, 最优)
  pk_a05_g2048_lr1e5 @ep2 = 26.7
  pk_a0_g512_lr5e6   @ep2 = 26.6  (无anchor, 后期崩~1)
  pk_a2_g512_lr5e6   @ep3 = 26.6  (后期稳~18)
  pk_a1_g512_lr1e5_e20 @ep1=26.6 ; pk_a1_g512_lr5e6 @ep2=26.2 ; 其余 25.6~26.7
结论:
- full ptkb 微调有 personalization 空间(conv 扛全部 profile 噪声只 25.3) → 涨过 conv, 甚至超 oracle (26.8 > 26.0).
- best 在 ep4 = 训练有正增量 (rel_ptkb 上 best=ep1 啥也没学到 -- 因为 rel 是 oracle 已最优).
- 多 gneg(2048) + anchor 后期最稳(~21); 无 anchor 仍崩.
- 这还没用 KD. 下一步 KD 实验看 26.8 能否再涨.
