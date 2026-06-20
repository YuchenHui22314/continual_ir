# round2 KD 突破 (2026-06-20): 32.3, 逼近 oracle 上界 36.3
锚点: base 12.9 / conv 25.3 / pkgrid(anchor+gneg) 26.8 / KD-r1 27.1 / oracle 上界 36.3
最优: kd2_bx1_kd2_a1_g2048_e20 @ep13 = 32.3 (gap to oracle = 4.0)
  配方: BiXSE 1.0 + KD 2.0 + anchor 1.0 + gneg 2048 + lr 5e-6 + 20ep (best ep13, ep1=25.6 涨到 32.3)
次优: kd2_bx01_kd2_e30 @ep6 = 30.1 (弱 BiXSE 0.1 + KD2 + gneg512)
纯 KD: kd2_pure_kd{1,2} 28.3~28.5 (不如组合! 说明 BiXSE 精排 + KD 方向 互补, 缺一不可)
结论:
- 27.1 → 32.3 涨 5.2; best 在 ep13 = 真正学习, 不是 ep1 退化.
- 最优 = 组合 (BiXSE ranking + KD personalized 方向 + anchor/gneg 防崩), 纯 KD 不够.
- round3: 加容量(LoRA r32/64) + 更多 epoch(30) + 微调 KD/anchor weight, 逼近 oracle 上界 36.3.
