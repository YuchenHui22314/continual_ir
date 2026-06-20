# CLAUDE.md — continual_ir Project

## Project Overview

Continual learning for Information Retrieval. Fine-tune a dense query encoder — either
ANCE (RoBERTa-base, 768d) or Qwen3-Embedding-0.6B (LLM-based, 1024d) — on conversational
search data (TopiOCQA) while retaining performance on the previous ad-hoc task (MSMARCO)
and OOD generalisation (BEIR).

Current focus: **Curriculum Learning** on Stage-2 TopiOCQA fine-tuning.
- Scoring: conversation turn length (`turn = 1 + |ctx|//2`, lower = easier).
- Pacing: `root_2`, `step`, `step_exclusive`, `step_exclusive_2_full` (see `src/curriculum.py`).
- Ordering: `easy2hard` (curriculum, CL) or `hard2easy` (anti-curriculum, ACL) or `none` (random baseline).
- Experience Replay from Stage 1 is supported on the ANCE branch (`train_continually_ddp.py`).

## Key File Paths

| File | Description |
|------|-------------|
| `src/data.py` | Dataset classes (Topiocqa, MSMARCODataset), collation logic |
| `src/models.py` | ANCE model (RobertaForSequenceClassification-based) |
| `src/train_topiocqa_ddp.py` | **Baseline** TopiOCQA-only training — DO NOT MODIFY |
| `src/train_continually_ddp.py` | Continual training w/ ER — DO NOT MODIFY |
| `src/train_continually_ddp_cl.py` | ANCE curriculum learning training script |
| `src/train_qwen_cl.py` | Qwen3-Embedding-0.6B curriculum learning training script (bf16, FlashAttention-2) |
| `src/curriculum.py` | Scoring functions + pacing functions (`root_2`, `step`, `step_exclusive`, `step_exclusive_2_full`) |
| `src/utils.py` | BEIR eval, optimizer, in-memory FAISS eval functions |
| `preprocess/data/analyze_topiocqa_turns.py` | Turn-length statistics script |
| `preprocess/data/extract_qwen_pos_neg_from_corpus.py` | Build Qwen pos/neg embedding cache directly from corpus index |
| `preprocess/plots/plot_pacing_all4.py` | Paper Fig.2 — all 4 pacing functions |
| `preprocess/plots/plot_turn_distribution.py` | Paper Fig.3 — TopiOCQA turn histogram |
| `preprocess/plots/plot_{ance,qwen}_curves.py` | Paper Fig.4 — per-encoder per-epoch NDCG@10 curves |
| `preprocess/plots/plot_curves_static_styles.py` | Combined Nature/IEEE/seaborn/ggplot styles for paper figure |
| `preprocess/eval/eval_{ance,qwen}_beir_full.py` | Offline BEIR + MSMARCO eval over saved checkpoints |
| `preprocess/eval/eval_qwen_base_full.py` | Qwen3 zero-shot (no Stage 2) BEIR + MSMARCO eval |
| `scripts/run_qwen_8_experiments.sh` | Batch launcher for the 8 Qwen3 curriculum runs |
| `figures/` | Paper-ready figures, tables, aggregated eval JSONs |

## Key Data Paths

| Data | Path |
|------|------|
| TopiOCQA train | `/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl` |
| TopiOCQA valid | `/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl` |
| TopiOCQA qrel | `/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec` |
| TopiOCQA corpus embeddings | `/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_ance_merged/` (26 blocks, .pb = pickle) |
| TopiOCQA pos/neg embeddings | `/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.pt` |
| ANCE pretrained | `/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/...` |
| BEIR corpus embeddings | `/data/rech/huiyuche/beir/embeddings/ance/{dataset_name}/` |
| BEIR dataset text | `/data/rech/huiyuche/beir/{dataset_name}/` |
| MSMARCO train | `/data/rech/huiyuche/TREC_iKAT_2024/data/topics/msmarco/msmarco_train.jsonl` |
| MSMARCO embeddings | `/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/msmarco_pos_neg_docs_ance/embeddings_multi_GPU.pt` |
| Model output dir | `/data/rech/huiyuche/huggingface/continual_ir/` |
| Logs | `/data/rech/huiyuche/TREC_iKAT_2024/logs/` |

## Baseline Configuration (wandb run 1ppvc6dl)

```
train_topiocqa_ddp.py
--n_gpu 4  --per_gpu_train_batch_size 120  (total batch = 480)
--num_train_epochs 20  --learning_rate 1e-5  --weight_decay 0.00
--warmup_ratio 0.06  --loss_type ranking  --negative_type none
--beir_datasets climate-fever msmarco
```

## Curriculum Learning Details

- **Difficulty score**: `turn_number = 1 + len(ctx_utts_text) // 2` (1 = easiest)
- **Pacing functions** (in `src/curriculum.py`):
  - `root_2(x, t, c0) = ((x*(1-c0²)/t) + c0²)^0.5` — smooth cumulative
  - `step(x, t, c0)` — 3-stage cumulative: c0 → (c0+1)/2 → 1.0 at T/3 and 2T/3
  - `step_exclusive(x, t, c0)` — 3-stage exclusive slicing: [0,c0), [c0,(c0+1)/2), [(c0+1)/2,1)
  - `step_exclusive_2_full` — like `step_exclusive` during curriculum, reverts to [0,1) after `curriculum_end_epoch`
  - `x` = epoch_start_step, `t` = curriculum_steps (epoch 16 × steps_per_epoch by default)
  - `c0` (delta_p) = initial fraction; default 0.2 (motivated by turn histogram)
- **Curriculum type**: `easy2hard` (CL), `hard2easy` (ACL), or `none` (random baseline)
- **DataLoader**: recreated each epoch with Subset of examples indexed by the pacing function

## Data Analysis Log

Analysis results (turn distribution, dataset stats, etc.) are saved to:
`/data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log`

## Environment

- Conda env: `/data/rech/huiyuche/envs/trec_ikat`
- 2× NVIDIA RTX A6000 (48GB each, compute cap 8.6) — Flash Attention 2 supported
- `/data/rech` = network disk (slow for large I/O)
- `/part/0x` = local SSD (fast — put training data here for production runs)

---

## General Instructions (apply to ALL tasks)

1. **tmux sessions**: For any long task (training, large inference, data processing):
   - Use tmux session named `yuchen`. If it exists, create a new window inside it.
   - Create a log file in `/data/rech/huiyuche/TREC_iKAT_2024/logs/` with a meaningful name.
   - Redirect all output: `2>&1 | tee -a /path/to/logfile.log`
   - Log key info (progress, metrics, errors) inside the code itself.

2. **Code style**: Match the existing code style (readability, comments, variable names).
   - Be generous with comments. No obscure idioms.
   - Never delete existing code or comments.
   - Add new features via arguments (e.g., `--use_flash_attention`) so they can be enabled/disabled.

3. **Test environment**: Always test GPU-dependent code in a real GPU environment, not sandbox.

4. **Show diffs before modifying**: For any existing file modification, show the planned diff
   and get user approval first. For file deletions, always ask before deleting.

5. **Concurrency by default**: Code should support multi-GPU (DDP/torchrun) and multi-thread.
   Default to DDP-compatible patterns.

6. **Minimal edits to existing files**: Preserve original code structure. Minimal targeted changes.

7. **Clarify when unsure**: If requirements are ambiguous and cannot be verified from context,
   ask before implementing. Do NOT make large assumptions and act.

8. **Data analysis log**: Save important dataset statistics (turn distributions, counts, ratios)
   to `/data/rech/huiyuche/TREC_iKAT_2024/logs/data_analysis.log`.
   Include timestamps and dataset name in each entry.

9. **End-to-end smoke test**: Before any full run (training, large inference, data processing),
   run a small-scale test (~5 min, small data fraction) to catch save/output bugs early.
   Typically: `--use_data_percent 0.05 --num_train_epochs 2`

10. **Disk persistence + resume**: Do not keep large inference results in RAM.
    Write intermediate results to disk periodically. Implement resume-from-checkpoint.
    If the program crashes, prior work should be recoverable.

11. **DataLoader for GPU processing**: Use `torch.utils.data.DataLoader` for large-scale
    GPU inference (batching, prefetching, pin_memory).

12. **Keep .gitignore clean**: Large files (*.pt, *.pkl, *.pb, *.npy, model checkpoints,
    logs, embeddings) should not be pushed to git.

13. **Network vs local disk**: `/data/rech` is network-mounted (slow for large I/O).
    `/part/0x` is local SSD (fast). For training runs with frequent reads, copy data to `/part`.

14. **No overhead before experiments**: Do not run `du` on large directories or recursive `ls`
    before GPU tasks. `/data/rech` network disk is slow — every unnecessary stat call adds latency.

15. **Check GPUs before running**: Before any GPU job, check `nvidia-smi` to see if GPUs are
    in use. If occupied, report the process info (pid, user, memory usage), guess what it is,
    and ask the user whether to kill it.

16. **Log EVERY experiment in the model registry** (standing rule, 2026-06-18): whenever you
    launch any training/fine-tuning run, ADD it to `docs/MODEL_REGISTRY.md` — its run name,
    family, recipe (template / precision / GPUs×batch / loss / LR / negatives), launcher,
    wandb project, and ckpt path `huggingface/continual_ir/<run>/`. Keep the registry current
    so the ~100-dir sprawl stays navigable. The ground truth for any run's exact args is its
    startup log (`grep -m1 "Args: Namespace" logs/run_<name>_*.log`). See the doc + skill
    `continual-ir-model-registry` + memory `project_continual_ir_model_registry`.

17. **IR experiments + paper writing — clean protocol** (standing rule, 2026-06-20): before
    running sweeps/ablations or writing them up, follow skill `ir-experiment-and-paper`. Core
    rules: (a) ablation = leave-one-out with all OTHER hyperparameters FIXED, kept SEPARATE from
    the cumulative build-up table — never a mixed "ablation of ours" bag; (b) model selection on
    FULL-corpus retrieval ONLY (the pool-rerank proxy is anti-correlated; never select on it);
    (c) log every run's config + full-corpus result immediately, never stitch an ablation from
    cross-round runs whose configs differ in several places; (d) trace every number to (run,
    metric, dataset, epoch) before quoting — don't fabricate a selection rationale (the
    QReCC 0.48-vs-0.546 in-batch-CE-vs-InfoNCE lesson); (e) paper structure = Dataset → Method →
    Experiments (ablation goes in Experiments, not Method), with loss formulas + citations for
    every borrowed method, unified terminology, and no internal codenames (e.g. `perso_dense_val`).
