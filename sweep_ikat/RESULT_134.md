# 1+3+4 (anchor 正则 + 全语料负例 + 低 lr) 全库结果 — 2026-06-20
锚点: base_ZS rel=13.4, conv_ZS rel=26.0 (校验通过). 8 config × 8 epoch, conv init, rel_ptkb, LoRA r16.

逐 epoch 全库 NDCG@3 (rel ×100):
  baseline a0 g0     : 25.6 23.0 17.8  1.1  1.8  0.9  0.5  0.0   <- ep4 崩 (老问题)
  only gneg a0 g512  : 24.9 24.0 21.1  0.5  1.3  1.4  1.4  1.4
  only anchor a1 g0  : 25.1 23.3 18.3  2.9  4.2  3.4  4.2  3.8
  a0.5 g512          : 24.9 24.3 20.3  3.7  9.8  7.4  8.6  6.1
  both a1 g512       : 24.2 23.8 20.9 10.4 13.7 10.6 12.2 10.4
  strong a2 g512     : 24.6 23.0 20.5 17.9 17.5 17.1 16.3 14.9   <- 强 anchor 后期稳
  a1 g2048           : 24.5 24.9 20.5 18.1 19.0 17.9 19.0 20.1   <- 多负例后期稳
  a1 g512 lr5e-6     : 25.8 24.9 25.0 24.6 22.0 17.1 11.7 13.8   <- 低 lr 最稳, best=25.8

结论:
- anchor(λ↑)+ 全语料负例 + 低 lr 成功防住崩塌 (后期 epoch 从崩到 ~1 → 保持 15-25)。机制验证成功:崩=query 旋转离 base-doc 空间;修=anchor 拉住 init + 全语料负例校准。
- 但 best 都在 ep1 (~25 ≈ conv ZS),没有 config 超过 conv 26.0。训练防崩成功但无正增量。
- 即:从"训练必崩"→"训练不崩、持平 conv ZS";还差"涨过 conv ZS"。
下一步候选: anchor 软化(L2/只罚大偏移)、加强 PTKB personalization 信号、更多 held-out 数据(44 太少)。
artifacts: scripts/run_ikat_134_grid.sh, perso_eval_134.yaml, alias_map_134.json, tuning_134.png
