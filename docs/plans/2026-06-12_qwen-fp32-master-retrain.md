# Plan: 修复纯-bf16 更新地板 (fp32 master weights) → 重训 Qwen 桶实验 → 重评/重画/重分析

## Context (为什么做)

- **Qwen3-Embedding-0.6B 原生 bf16**:config `torch_dtype=bfloat16`,磁盘 310 个权重全 BF16(已核实)。无官方 fp32 版。
- **我们的桶训练是纯 bf16(有 bug)**:launcher 传 `--use_flash_attention --use_bf16` → `QwenQueryEncoder` 用 `torch_dtype=bf16` 加载 → 参数 + AdamW 一二阶矩**全 bf16**(已核实:ckpt bf16,optimizer exp_avg/exp_avg_sq bf16)。**无 fp32 master,无 GradScaler**。
- **实测地板效应**:lr=1e-5 时单步 AdamW 更新(~1e-5)小于 bf16 在 |θ|≥0.0026 处的 ULP(bf16 仅 8 位尾数)。实测 `layers.13.mlp.gate_proj`:|θ|<0.0026 的参数 98.9% 动了;|θ|≥0.0026 的只有 **4.5%** 动 → 我们测的 ~90% "稀疏度" 大部分是 **bf16 舍入假象**,污染参数分析。
- **官方实证(`QwenLM/Qwen3-Embedding/docs/training/SWIFT.md`)**:官方 FT = ms-swift `swift sft --train_type full --learning_rate 6e-6 --loss_type infonce --deepspeed zero2/3`。**DeepSpeed ZeRO + bf16 = 标准保留 fp32 master weights**。官方 LR 6e-6 < 我们 1e-5 → 6e-6 在纯 bf16 里更会被舍没,它能 work **正是因为有 fp32 master 托底**。所以我们的纯 bf16(无 master)是偏离官方的 bug。
- **哪些结果不受影响**:行为层(forgetting 曲线/transfer 矩阵/长度依赖)= 模型输出 → 有效;Fisher 重叠(在 base 模型梯度 Σg² 上算,不碰 Δθ)→ 有效。只有用 Δθ 的参数分析(稀疏度/子网络 overlap/EWC)被污染。

## Goal
给 `train_qwen_cl.py` 加 fp32 master weights(纯 PyTorch 复现官方 DeepSpeed-ZeRO 的 fp32-master 数值行为,0.6B 不需要 ZeRO 分片故不上 DeepSpeed),重训 13 个 Qwen 桶,重评 + 重画 + 重跑(此时干净的)参数分析。**保留旧 bf16 产物做 bf16-vs-fp32 对照**(本身是一个发现)。

## 决策(已与用户确认)
- 方案:**自改 fp32 master**(非 HF Trainer / 非 DeepSpeed —— 数值等价但保住全部 infra)。
- GPU:**3 张**(SLURM job 6902 分配 `gres/gpu:ls40:3`;未 pin,故用 `CUDA_VISIBLE_DEVICES=1,2,3` 留 GPU0)。RAM cap 300G(BEIR cache ~120G 在范围内)。
- 保持 **global batch 480** 与原桶/ANCE 一致。
- **超参:只改 fp32,其余一切不变**(LR 1e-5、现有 in-batch CE loss、batch 480、v3 模板、无假负例屏蔽)→ 重训后任何变化都可干净归因于"修掉 bf16 地板",bf16-vs-fp32 受控对照,且与 ANCE(1e-5) 可比。

## 官方 tips 调研结论(QwenLM/Qwen3-Embedding/docs/training/SWIFT.md + ms-swift Embedding.md;存档,本次不采用)
- **6e-6 无 rationale**:官方 embedding/reranker 都用 6e-6,文档没解释,就是保守 full-FT LR。我们 1e-5 略高、合理区间,保留以维持对照。
- **fp32 master via DeepSpeed ZeRO** = 我们要修的(本计划核心)。
- **infonce τ=0.01 + 跨 GPU in-batch 负例 + INFONCE_MASK_FAKE_NEGATIVE(相似度>正例+0.1 的负例置 -inf)**:假负例屏蔽或与 turn_1 自伤相关(未来可试),本次不动。
- **模板结尾 `<|endoftext|>` 非 `<|im_end|>`**:印证之前 wrapper bug 修复方向正确。

## 修复 (src/train_qwen_cl.py)
- 加 CLI `--fp32_master_weights`(默认 False,不破坏旧复现 + ANCE)。
- `QwenQueryEncoder.__init__`(行 66-71):当 fp32_master → **不设 torch_dtype**(fp32 加载,把发布的 bf16 权重上采样到 fp32)+ `attn_implementation="sdpa"`(FA2 需半精度权重;sdpa 在 autocast 下正常)。保留 `autocast(bf16)`(由 `--use_bf16` 控制)→ bf16 算前向/反向。同步 resume-load 路径(行 ~412-414)保持一致。
- 净效果:参数 fp32(master)+ AdamW 矩 fp32 + autocast bf16 计算 → 1e-5 更新能正确累积。行 82 的 `embs.float()` 不变。
- `get_optimizer` / DDP / 梯度记录不变(梯度记录现在读 fp32 `p.grad`,更干净)。

### 保证"官方其他逻辑一致"(硬约束)
- **精度逻辑 = 官方**:fp32 master + autocast(bf16) 矩阵运算 = DeepSpeed ZeRO bf16 的数值行为(= HF Trainer `bf16=True` 标准做法)。这就是官方精度逻辑,不是自造的。
- **唯一机械改动 = FA2→sdpa,且数学等价**:两者都算精确 `softmax(QK^T/√d)V`,非近似;换 sdpa 仅因 FA2 强制半精度权重、与 fp32 master 不兼容。
- **模型机制全对齐官方且不变**:last-token pool(`last_hidden_state[:,-1,:]`)、L2 normalize、结尾 `<|endoftext|>`(v3 builder 已加)、left-pad —— 逐项与官方 SWIFT/ms-swift 文档一致,本次都不动。
- **实验超参不动**(只改 fp32):LR 1e-5、in-batch CE loss、batch 480、v3 模板。
- **硬验证(重训前必做)**:对 base 模型用 (旧)bf16+FA2 与 (新)fp32+sdpa+autocast 各编码同一批 ~20 条 query,比对 embedding 余弦相似度,要求 **>0.999**(证明前向数值等价,换 kernel/精度路径没改变模型行为);否则停下排查。

## 重训 (新 scripts/run_qwen_turn_buckets_fp32.sh)
- 复制 `run_qwen_turn_buckets.sh`,改:`CUDA_VISIBLE_DEVICES=1,2,3` + `--n_gpu 3` + batch 凑 global 480(先试 `3×160×accum1`;OOM 退 `3×80×accum2`)+ `--use_bf16 --fp32_master_weights`(**去掉 `--use_flash_attention`** → sdpa)+ 输出 `OUT_BASE/bucket_qwen32_turn_*` + wandb `topiocqa-qwen-turn-buckets-fp32`。保留 v3 模板、caps 32768、470 步、每 47 存、`--record_grad_stats`。
- **先 smoke**(turn_15plus,47 步):验证 (a) bf16 地板测试现在通过(|θ|≥0.0026 参数会动,对比之前 4.5%),(b) 显存装得下(160 或退 80),(c) 速度。3 GPU fp32+sdpa 估 ~70-95min/桶 → 13 桶 ≈ 15-20h。
- watcher 邮件 huiyuche@iro.umontreal.ca。

## 重评 (复用现有脚本,指向 bucket_qwen32_*)
- `eval_bucket_runs_per_ckpt.py`(TopiOCQA 全集+子集 + MSMARCO):BUCKETS 前缀 → bucket_qwen32_*,输出 → `bucket_qwen32_runs_eval.json`。
- `eval_bucket_qwen_beir_per_ckpt.py`(BEIR-14):同 → `bucket_qwen32_beir_eval.json`。
- 都需 GPU,训练后跑。

## 重画 + 重分析
- `plot_bucket_experiment.py` + `plot_bucket_qwen_beir.py` 跑新 JSON(图名加 `_fp32` 后缀;保留 bf16 图做对照)。
- `gradient_direction_analysis.py` + `update_sparsity_suite.py` 跑 bucket_qwen32_* → 干净的稀疏度/子网络/coherence。报告 **bf16-vs-fp32 稀疏度差**(量化舍入假象幅度)——这本身写进 paper。

## 验证
- **(重训前)前向等价**:旧 bf16+FA2 vs 新 fp32+sdpa+autocast 编码同批 query,cosine >0.999(见"保证官方逻辑一致")。
- **Smoke 直接证明地板消失**:对 fp32 turn_15plus step-47 ckpt 重跑 |θ|-阈值 moved-fraction 测试;|θ|≥0.0026 参数的 moved% 应远超之前的 4.5%(跟随其梯度大小)。
- 显存在 3×L40S 46GB 内;loss 下降。
- Eval sanity:turn_1 等 in-length vs 旧 bf16(预期有变化——大权重被更充分训练)。
- 落地前对账一个 final-ckpt 的 offline 数字(防温度计事故)。

## 改动文件
| 文件 | 动作 |
|---|---|
| `src/train_qwen_cl.py` | 加 `--fp32_master_weights` flag + 加载分支(~10 行;默认行为不变) |
| `scripts/run_qwen_turn_buckets_fp32.sh` | 新建 |
| `preprocess/eval/eval_bucket_runs_per_ckpt.py` | BUCKETS 前缀 + 输出路径(小改或加参数) |
| `preprocess/eval/eval_bucket_qwen_beir_per_ckpt.py` | 同上 |
| `preprocess/plots/plot_bucket_experiment.py`, `plot_bucket_qwen_beir.py` | 输出 `_fp32` 后缀 |
| 分析脚本 | 不改,`--runs bucket_qwen32_*` |

## 不在范围
- 切 ms-swift / 真上 DeepSpeed(要重写数据格式 + 丢 curriculum/step-mode/梯度记录/已验证 eval;fp32-master 自改对 0.6B 数值等价于 ZeRO 的 fp32-master)。
- 重训主表 instruct2/3(行为结果有效;只有桶的参数分析需要此修复)。
- ANCE(已 fp32)。
