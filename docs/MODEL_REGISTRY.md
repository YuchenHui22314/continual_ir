# continual_ir — Model Registry (all training runs)

Every fine-tuning run we have on disk, grouped by family, with its recipe and checkpoint
location. Updated 2026-06-19.

## family `ikat_graded` (sweep, 2026-06-19) — graded personalized dense retriever
28 throwaway runs in **`/part/01/Tmp/yuchen/continual_ir/{s1_*,s1b_*,s1c_*,s2_*}/best/`** (NOT
huggingface/; `--ikat_save_best_only` keeps only the best-proxy epoch). Recipe: `--dataset_type
ikat_graded --loss_type bce` (BiXSE), query-encoder-only, frozen base-qwen3 docs, 12–20 ep, bs16,
bf16_fp32_master+FA2+grad-ckpt, single-GPU. Sweep = init{base,conv-qwen3 step-94} × form{ptkb,rel_ptkb}
× LoRA r{8,16,32,64}/full-FT × cap{16,32,64,all} × LR{6e-6,1e-5,1e-4,5e-4}. Launcher
`scripts/sweep_ikat_graded.sh`, driver `scripts/sweep_ikat_driver.py`, results `sweep_ikat/stage*_results.json`.
**RESULT: approach fails on full retrieval — every trained config collapses (full-ClueWeb NDCG@3 1–5 ≪
base 13.4); conv-qwen3 ZS is best (26.0). Proxy anti-correlated w/ full retrieval. See
`sweep_ikat/SUMMARY.md` + memory `project_ikat_graded_retriever`.** Safe to `rm -rf` /part runs (ask first). **When in doubt about a run's exact settings, the ground
truth is its training log** (the full `argparse.Namespace` is logged at startup):
`grep -m1 "Args: Namespace" /data/rech/huiyuche/TREC_iKAT_2024/logs/run_<name>_*.log`
(or the wandb config). This doc is the map; the log is the territory.

## Checkpoint location (all runs)
```
/data/rech/huiyuche/huggingface/continual_ir/<run_name>/checkpoint-step-<N>/
```
- Epoch-mode runs: 20 ckpts at step `94*epoch` (94,188,…,1880). `model.safetensors`
  (+ config + tokenizer + `trainer_state.pt`). **No `optimizer.pt`** unless `--save_optimizer`
  (we don't) → model-only, not resumable.
- Bucket runs: step-driven, 10 ckpts at step `47*k` (47…470).
- Sweeps/probes/smokes: 1–2 ckpts (deletable, see bottom).
- Trainers: Qwen → `src/train_qwen_cl.py`; ANCE → `src/train_continually_ddp_cl.py`
  (curriculum) / `train_topiocqa_ddp.py` (baseline) / `train_continually_ddp.py` (ER).
- Eval JSONs + figures live in `continual_ir/figures/` (skill `qwen-continual-eval`).

## Naming decode
`<prefix>_<encoder>_<schedule>` where schedule ∈ {`nosched` (random order baseline),
`cl_<pacing>` (curriculum easy→hard), `acl_<pacing>` (anti-curriculum hard→easy)},
pacing ∈ {`root2`, `step`, `step_excl`, `step_excl_2_full`}. An **"8-run family"** =
1 nosched + 4 CL (`cl_root2,cl_step,cl_step_excl,cl_step_excl_2_full`) + 3 ACL
(`acl_root2,acl_step,acl_step_excl`). All share the family recipe; they differ ONLY in
sample ordering.

---

## A. Qwen3-Embedding-0.6B families (1024-d, last-token pool, L2)
Shared unless noted: TopiOCQA, 20 epochs, LR 1e-5 constant (`--no_lr_schedule`, 0.06 warmup
ratio is a no-op), AdamW (0.9,0.999), wd 0, in-batch CE (`--loss_type ranking
--negative_type none`), frozen pre-encoded docs, grad-ckpt on.

| family (dirs) | template | precision / attn | GPUs / batch | negs/q | launcher | wandb | notes |
|---|---|---|---|---|---|---|---|
| `qwen_*` (8) | none (v0) | bf16 / FA2 | 4×120 = 480 | 120 | `run_qwen_8_experiments.sh` | topiocqa-qwen | original 2026-04 batch, no instruction prefix |
| `instruct_qwen_*` (8) | v1 | bf16 / FA2 | 4×120 | 120 | `run_qwen_8_instruct.sh` (early) | topiocqa-qwen-instruct | FIRST v1-instruct batch |
| `instruct2_qwen_*` (8) | v1 | bf16 / FA2 | 4×120 | 120 | `run_qwen_8_instruct.sh` | topiocqa-qwen-instruct(-v2) | 2026-05-19 v1 redo |
| `instruct3_qwen_*` (8) | **v3** | bf16 / FA2 | 4×120 | 120 | `run_qwen_8_instruct3.sh` | topiocqa-qwen-instruct-v3 | 2026-06-06. **bf16 reference for the fp32 study** |
| `instruct3fp32_qwen_nosched` | v3 | **fp32-master** / sdpa | **4×120** | **120** | `run_qwen_nosched_fp32_4x120.sh` | topiocqa-qwen-instruct-v3-fp32 | **clean fp32-isolation, exact main-batch match** (2026-06-18) |
| `instruct3fp32_qwen_nosched_3x80` | v3 | fp32-master / sdpa | 3×80×accum2 | **80** | `run_qwen_nosched_fp32.sh` | …-fp32 | earlier 3-GPU version (fewer negs); superseded by 4×120 |
| `instruct3fp32_infonce_lr1e5` / `_lr6e6` | v3 | fp32-master / sdpa | 3×80×accum2 | **240 (cross-GPU)** | `run_qwen_infonce.sh`, `run_qwen_infonce_BOTH.sh` | topiocqa-qwen-infonce-fp32-lr{1e5,6e6} | InfoNCE τ=0.01, cosine, β₂=0.95, fake-neg mask, same-gold dedup |
| `instruct3fp32infonce_qwen_nosched` | v3 | **B = bf16-master + FA2** | **4×120** | **480 (cross-GPU)** | `run_qwen_infonce_4x120.sh` | topiocqa-qwen-infonce-fp32 | LR 1e-5. Same InfoNCE recipe as the 3×80 rows but at the FULL 480-neg main batch + the canonical fast path B (official DeepSpeed-ZeRO bf16, ~30% faster). **Launched 2026-06-18**; first of the 4×120 InfoNCE 8-run batch (`instruct3fp32infonce_qwen_*`) |
| `instruct3fp32infonce_qwen_constlr` | v3 | B = bf16-master + FA2 | **4×120** | **480 (cross-GPU)** | `run_qwen_infonce_4x120_constlr.sh` | topiocqa-qwen-infonce-fp32 | LR 1e-5 **constant** (`--no_lr_schedule`). Byte-identical to `instruct3fp32infonce_qwen_nosched` (the "B" column = `tab:hparams` last col) EXCEPT cosine+warmup0.1 → constant LR; isolates the schedule effect. **2026-06-27** (exit 0, ~4h9m, 20 per-epoch ckpts). `--record_grad_stats`: kept R1 `grad_norms_per_step.npz` (2.4 MB), **deleted R2 sum_g/sum_g2 accumulators** (5.4 GB) per user. New `tab:hparams` col "Qwen3 (InfoNCE 4×120, const)" |
| `instruct3fp32infonce_qwen_constlr_lr6e6` | v3 | B = bf16-master + FA2 | **4×120** | **480 (cross-GPU)** | `run_qwen_infonce_4x120_constlr.sh 6e-6` | topiocqa-qwen-infonce-fp32 | LR **6e-6** constant. Same recipe as `instruct3fp32infonce_qwen_constlr` but lr 6e-6 (smaller → tamer late-epoch oscillation, the 1e-5 const TopiOCQA curve oscillates). **2026-06-28** (exit 0, 20 ckpts). Final NDCG@10 **TopiOCQA 0.473 / MSMARCO 0.354 / QReCC 0.426** — beats 1e-5 const on all three (0.454/0.341/0.414) and forgets MSMARCO least of the 4×120 const/cosine runs. R2 deleted/R1 kept. `tab:hparams` const col LR now {1e-5,6e-6} |
| `instruct3fp32infonce_constlr_lr6e6_{cl_step,cl_step_excl,cl_step_excl_2_full,cl_root2,acl_root2,acl_step,acl_step_excl}` (7) | v3 | B = bf16-master + FA2 | **4×120** | **480 (cross-GPU)** | `run_qwen_infonce_constlr_curriculum.sh 6e-6` | topiocqa-qwen-infonce-fp32 | The **7 curriculum/ACL variants** of the 6e-6 const baseline (= row above is the random-order baseline). easy2hard (CL) / hard2easy (ACL) × pacing {step, step\_exclusive, step\_exclusive\_2\_full, root\_2}. **2026-06-28** (~20h, all exit 0, 20 ckpts each). CL early-short-conv pacing makes CL ~2.5h/run vs ACL ~3.7h. Eval'd → `fig:curves` InfoNCE panels (i,j) + 8-row final table. NO grad_stats, NO in-training eval. |
| `bucket_infonce6e6_{turn_1..turn_10,turn_11_12,turn_13_14,turn_15plus}` (13) | v3 | B = bf16-master + FA2 | **4×120** | **480 (cross-GPU)** | `run_qwen_turn_buckets_infonce6e6.sh` | topiocqa-qwen-turn-buckets-infonce6e6 | **InfoNCE 6e-6 const rerun of the turn-bucket causal-forgetting study** (orig was in-batch-CE fp32-master `bucket_qwen32_*`, untouched). 13 equal 3,054-pair buckets, **step-driven 470/47 → 10 ckpts**, **`--record_grad_stats --grad_stats_device cpu`** (R1 per-step norms + R2 Σg/Σg² @ step 47/235/470 → Fisher/EWC/sign-coherence). wd=0, curriculum none, NO in-training eval. NB step 47 ≈ epoch 7.4 (3054/480=6.4 steps/ep), NOT epoch 1. **2026-06-29**. |
| `bucket_qwen_turn_{1..10,11_12,13_14,15plus}` (13) | v3 | fp32-master | 3-GPU, step-driven | — | `run_qwen_turn_buckets_fp32.sh` | — | per-turn-length forgetting study; `--total_train_steps 470`, save every 47 (10 ckpts), each bucket downsampled to 3,054 pairs. See skill `qwen-gradient-analysis` |

**fp32 vs bf16 finding (2026-06-18):** at the EXACT same 4×120 in-batch-CE setting, fp32-master
alone raises TopiOCQA (~0.467→0.498) but INCREASES MS MARCO forgetting (0.336→0.271) — bf16's
"stability" was under-training (the update floor froze ~90% of weights). See
`docs/qwen_vs_official_training_divergences.md`.

**Future fast path (verified, not yet used in a batch):** `--use_bf16 --use_flash_attention
--bf16_fp32_master --gradient_checkpointing --gpu_resident_doc_table --dataloader_num_workers 4`
= same fp32-master numerics, bf16 model + FA2, ~30% faster (~8 vs ~11.4 s/step), ~25 GB. grad-ckpt
must stay ON (off OOMs at 4×120). See [[project_qwen_training_speed]], skill `qwen-instruct-training`.

⚠️ **In-training BEIR/MSMARCO numbers for any instruct* run are INVALID** (Qwen3TokenizerWrapper
`<|im_end|>` bug). Use the offline evals only (skill `qwen-continual-eval`).

## B. ANCE families (RoBERTa-base, 768-d, CLS pool) — older, less actively used
Settings less current; confirm from each run's log / wandb (baseline config = wandb `1ppvc6dl`,
in CLAUDE.md). Trainer: `train_continually_ddp_cl.py` / `train_topiocqa_ddp.py`.

| family (dirs) | what it is |
|---|---|
| `ance_topiocqa_nosched`, `ance_topiocqa_lrsched` | ANCE TopiOCQA baselines (random order; with/without LR schedule) |
| `ance_curriculum_{root2,step,step_excl_2_full,step_exclusive}` | ANCE curriculum runs (the ANCE counterpart of the qwen 8-run batch) |
| `bucket_ance_turn_{1..10,11_12,13_14,15plus}` (13) | ANCE per-turn-length forgetting study (mirror of `bucket_qwen_turn_*`) |
| `continual_{100_20,110_10,90_30,80_40,70_50,60_60,50_70,40_80}_no_negative` (8) | ANCE continual / experience-replay runs; the `A_B` numbers encode a stage-1/stage-2 (or replay) split — **confirm meaning from the log before citing** |
| `topiocqa`, `topiocqa_120bs_{beir,no_negatives}`, `topiocqa_cl_*`, `topiocqa_anticl_*`, `topiocqa_cl_hard2easy_*`, `topiocqa_anticurr_*` | EARLIEST ANCE TopiOCQA experiments (pre-naming-convention); settings vary — log is authoritative |

## C. Deletable (smoke / probe / sweep — not experiments)
`SMOKE_fp32_4x120`, `qwen_smoke4`, `qwen_smoke_eval` (smokes); `qwen_probe_bs{48,96,120}`
(batch-fit probes, 1 ckpt); `qwen_lrsweep_{1e-4,1e-5,5e-6}` (LR sweep, 1 ckpt);
`topiocqa_cl_easy2hard` (2 ckpts, aborted). Safe to `rm -rf` to reclaim disk if needed
(ask first — shared NFS).
