# CLAUDE.md — continual_ir Project

## Project Overview

Continual learning for Information Retrieval. Fine-tune an ANCE (RoBERTa-base) query encoder
on conversational search datasets (TopiOCQA, MSMARCO) using Experience Replay.

Currently implementing: **Curriculum Learning** on TopiOCQA fine-tuning.
- Scoring: conversation turn length (# turns, lower = easier)
- Pacing: root_2 pacing function (from DCL 2208 / transformers_cl 1912, same formula)
- Baselines: random (original), easy2hard (curriculum), hard2easy (anti-curriculum)

## Key File Paths

| File | Description |
|------|-------------|
| `src/data.py` | Dataset classes (Topiocqa, MSMARCODataset), collation logic |
| `src/models.py` | ANCE model (RobertaForSequenceClassification-based) |
| `src/train_topiocqa_ddp.py` | **Baseline** TopiOCQA-only training — DO NOT MODIFY |
| `src/train_continually_ddp.py` | Continual training w/ ER — DO NOT MODIFY |
| `src/train_continually_ddp_cl.py` | NEW: Curriculum learning training script |
| `src/curriculum.py` | NEW: Scoring functions + pacing functions |
| `src/utils.py` | BEIR eval, optimizer, in-memory FAISS eval functions |
| `preprocess/analyze_topiocqa_turns.py` | Turn-length statistics script |

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
- **Pacing function**: `root_2(x, t, c0) = ((x*(1-c0^2)/t) + c0^2)^0.5`
  - `x` = epoch_start_step, `t` = curriculum_steps (epoch 16 × steps_per_epoch)
  - `c0` (delta_p) = initial fraction; tune based on `analyze_topiocqa_turns.py` output
- **Curriculum type**: `easy2hard` (ascending) or `hard2easy` (descending)
- **DataLoader**: recreated each epoch with Subset of first `n_active` sorted examples

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
