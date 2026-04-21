"""
Training script for Qwen3-Embedding-0.6B with curriculum learning on TopiOCQA.

This is a direct adaptation of train_continually_ddp_cl.py for Qwen3.
Key differences from the ANCE version:
  - Model: Qwen3-Embedding-0.6B (0.6B params, 1024-dim) wrapped in QwenQueryEncoder
  - Pooling: last-token + L2-normalize (vs. first-token + linear head for ANCE)
  - Tokenizer: Qwen2Tokenizer wrapped in Qwen3TokenizerWrapper (adds EOS as CLS/SEP substitute)
  - Padding: dynamic=True (batch-max pad, hard-cap 512) → ~3x memory saving vs fixed 512
  - Embedding dim: 1024 (corpus blocks and FAISS indices must match)

DO NOT MODIFY: train_continually_ddp_cl.py (ANCE experiments must stay reproducible)
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys, argparse, contextlib, time, os, gc
from os.path import join as oj

sys.path.append('..')
sys.path.append('.')

from tqdm import tqdm, trange
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import wandb

from utils import (
    eval_beir_datasets,
    build_beir_eval_cache,
    eval_beir_from_cache,
    eval_topiocqa,
    load_corpus_into_faiss,
    set_seed,
    get_optimizer,
    optimizer_to,
)
from data import Topiocqa
from curriculum import SCORING_FUNCTIONS, PACING_FUNCTIONS, get_pacing_value, log_pacing_schedule


# ---------------------------------------------------------------------------
# Qwen3-Embedding-0.6B wrapper
# ---------------------------------------------------------------------------

class QwenQueryEncoder(nn.Module):
    """
    Thin nn.Module wrapper around Qwen3-Embedding backbone.

    forward() returns L2-normalized last-token embeddings of shape (B, 1024),
    matching the interface expected by cal_ranking_loss() and the eval functions.
    """
    def __init__(self, model_path: str, use_flash_attention: bool = False,
                 use_bf16: bool = False):
        super().__init__()
        kwargs = {"trust_remote_code": True}
        if use_bf16 or use_flash_attention:
            kwargs["torch_dtype"] = torch.bfloat16
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModel.from_pretrained(model_path, **kwargs)
        # Disable KV cache — we only do forward/backward for embedding, no generation.
        # Without this, Qwen decoder still materializes past_key_values → extra activation memory.
        # Also a prerequisite for gradient_checkpointing to be effective.
        self.model.config.use_cache = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # Last-token pooling with left-padding: last position is always the last real token.
        # FlashAttention 2 requires left-padding (padding_side='left' set on tokenizer).
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)    # always float32 for loss stability

    def save_pretrained(self, path: str, **kwargs):
        """Delegate to inner model so HF checkpoint format is preserved."""
        self.model.save_pretrained(path, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)


class Qwen3TokenizerWrapper:
    """
    Wraps Qwen2Tokenizer to produce [EOS, tok1, …, tokN, EOS] from each .encode() call,
    mimicking RoBERTa's [CLS]…[SEP] format that build_conv_query_tokens() expects.

    cls_token_id = sep_token_id = eos_token_id = 151645 (<|im_end|>)
    pad_token_id = 151643 (<|endoftext|>)
    """
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.eos_token_id = tokenizer.eos_token_id    # 151645
        self.cls_token_id = tokenizer.eos_token_id
        self.sep_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id    # 151643
        self.bos_token_id = tokenizer.eos_token_id
        self.vocab_size   = tokenizer.vocab_size

    def encode(self, text, add_special_tokens=True, max_length=512, truncation=True):
        # Reserve 2 positions for the boundary EOS tokens
        budget = max(1, max_length - 2) if add_special_tokens else max_length
        ids = self._tok.encode(text, add_special_tokens=False,
                               max_length=budget, truncation=truncation)
        if add_special_tokens:
            return [self.eos_token_id] + ids + [self.eos_token_id]
        return ids

    def __call__(self, texts, max_length=512, padding=True, truncation=True,
                 return_tensors=None, **kwargs):
        """
        Batch encode for eval_beir_from_cache() which calls tokenizer(batch, ...).
        Adds EOS boundary tokens and left-pads (required by FlashAttention 2).
        """
        encoded_ids = [self.encode(t, add_special_tokens=True,
                                   max_length=max_length, truncation=truncation)
                       for t in (texts if isinstance(texts, list) else [texts])]
        if padding:
            max_len = max(len(ids) for ids in encoded_ids)
            input_ids = [
                [self.pad_token_id] * (max_len - len(ids)) + ids   # left-pad
                for ids in encoded_ids
            ]
        else:
            input_ids = encoded_ids
        attention_mask = [[0] * (len(row) - len(ids)) + [1] * len(ids)
                          for row, ids in zip(input_ids, encoded_ids)]
        if return_tensors == "pt":
            import torch
            return {
                "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path):
        self._tok.save_pretrained(path)

    def __getattr__(self, name):
        return getattr(self._tok, name)


# ---------------------------------------------------------------------------
# Loss functions (unchanged from ANCE version)
# ---------------------------------------------------------------------------

def make_cycling_iter(loader, n_steps, epoch):
    """Yield exactly n_steps batches, cycling through loader with fresh shuffles."""
    yielded = 0
    cycle   = 0
    while yielded < n_steps:
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch * 10000 + cycle)
        for batch in loader:
            if yielded >= n_steps:
                return
            yield batch
            yielded += 1
        cycle += 1
        if cycle > 1000:
            raise RuntimeError("Cycling iterator exceeded 1000 cycles — subset too small for drop_last=True.")


def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs=None):
    batch_size = query_embs.size(0)
    pos_scores = query_embs @ pos_doc_embs.T
    if neg_doc_embs is not None:
        neg_scores = torch.sum(query_embs * neg_doc_embs, dim=1, keepdim=True)
        score_mat  = torch.cat([pos_scores, neg_scores], dim=1)
    else:
        score_mat = pos_scores
    labels = torch.arange(batch_size, device=query_embs.device)
    return nn.CrossEntropyLoss()(score_mat, labels)


def cal_kd_loss(query_embs, oracle_query_embs):
    return nn.MSELoss()(query_embs, oracle_query_embs)


def save_model(args, model_output_path, model, query_tokenizer, optimizer, scheduler,
               cur_step, cur_epoch, best_loss):
    output_dir = oj(model_output_path, f"checkpoint-step-{cur_step}")
    if os.path.exists(output_dir):
        logger.info(f"Checkpoint {output_dir} already exists, skipping save.")
        return
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)           # saves inner Qwen3 AutoModel
    query_tokenizer.save_pretrained(output_dir)         # saves Qwen2Tokenizer files
    torch.save(optimizer.state_dict(), oj(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), oj(output_dir, "scheduler.pt"))
    torch.save({"step": cur_step, "epoch": cur_epoch, "best_loss": best_loss},
               oj(output_dir, "trainer_state.pt"))
    logger.info(f"Saved checkpoint at {output_dir}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    is_main_process = (args.n_gpu == 1) or (dist.get_rank() == 0)

    # 0. Pre-encoded pos/neg/oracle embeddings
    if args.pos_neg_embedding_file is not None:
        sample_emb_table = torch.load(args.pos_neg_embedding_file, map_location="cpu")

    # 1. Load Qwen3-Embedding model + tokenizer
    query_encoder = QwenQueryEncoder(
        model_path         = args.pretrained_encoder_path,
        use_flash_attention = args.use_flash_attention,
        use_bf16           = args.use_bf16,
    ).to(args.device)

    raw_tokenizer  = AutoTokenizer.from_pretrained(args.pretrained_encoder_path,
                                                   trust_remote_code=True)
    # FlashAttention 2 requires left-padding; last real token is always at position -1.
    raw_tokenizer.padding_side = "left"
    query_tokenizer = Qwen3TokenizerWrapper(raw_tokenizer)

    if args.n_gpu > 1:
        # find_unused_parameters=False: the forward path is unconditional (no LM head,
        # no branches) so every param contributes to the loss every step.
        # gradient_as_bucket_view=True: lets DDP share memory between grads and buckets
        # (saves a copy of gradient tensors in the reducer).
        query_encoder = DDP(query_encoder, device_ids=[args.local_rank],
                            output_device=args.local_rank,
                            find_unused_parameters=False,
                            gradient_as_bucket_view=True)
        dist.barrier()

    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 matmul enabled.")

    if args.gradient_checkpointing:
        model_gc = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        # use_reentrant=False: non-reentrant checkpoint. Better for DDP, torch.compile,
        # mixed-grad inputs, and dynamic control flow. PyTorch is deprecating reentrant=True.
        model_gc.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled (use_reentrant=False).")

    if args.use_compile:
        query_encoder = torch.compile(query_encoder)
        logger.info("torch.compile enabled.")

    # 2. Dataset
    train_dataset = Topiocqa(args, query_tokenizer, args.training_data_file)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.curriculum_type != "none":
        if args.scoring_function not in SCORING_FUNCTIONS:
            raise ValueError(f"Unknown scoring function '{args.scoring_function}'")
        scoring_fn = SCORING_FUNCTIONS[args.scoring_function]
        ascending  = (args.curriculum_type == "easy2hard")
        train_dataset.sort_by_difficulty(scoring_fn, ascending=ascending)
        logger.info(f"Curriculum: {args.curriculum_type}, scoring={args.scoring_function}, "
                    f"pacing={args.pacing_function}, c0={args.curriculum_c0}")

    # Step counts (same as ANCE: full-dataset basis for consistent LR schedule)
    raw_micro_steps       = len(train_dataset) // args.batch_size
    micro_steps_per_epoch = (raw_micro_steps // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
    steps_per_epoch       = micro_steps_per_epoch // args.gradient_accumulation_steps
    total_training_steps  = args.num_train_epochs * steps_per_epoch
    curriculum_steps      = args.curriculum_end_epoch * steps_per_epoch

    if steps_per_epoch == 0:
        raise ValueError(f"steps_per_epoch=0: dataset too small or gradient_accumulation_steps too large.")

    if is_main_process and args.curriculum_type != "none":
        log_pacing_schedule(args.num_train_epochs, steps_per_epoch, curriculum_steps,
                            args.curriculum_c0, args.pacing_function)

    # Collate function: dynamic (batch-max, hard-cap 512) + left-padding (required by FlashAttn2)
    collate_fn_kwargs = dict(pad_token_id=query_tokenizer.pad_token_id,
                             dynamic_padding=True, left_padding=True)

    if args.curriculum_type == "none":
        if args.n_gpu > 1:
            sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size = args.per_gpu_train_batch_size,
            sampler    = sampler,
            collate_fn = train_dataset.get_collate_fn(args, **collate_fn_kwargs),
            drop_last  = True,
            pin_memory = True,
        )

    # 3. Optimizer
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay,
                              fused=args.use_fused_optimizer)

    # 4. LR scheduler
    if args.no_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        if is_main_process:
            logger.info("LR scheduler: constant (no warmup, no decay).")
    else:
        num_warmup = int(args.warmup_ratio * total_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_training_steps)
        if is_main_process:
            logger.info(f"LR scheduler: linear warmup ({num_warmup} steps) + linear decay.")

    # 5. In-memory eval cache (corpus loaded once; per-epoch eval only re-encodes queries)
    beir_eval_cache    = None
    topiocqa_faiss_idx = None
    topiocqa_doc_ids   = None

    if is_main_process:
        if args.activate_eval_while_training and len(args.beir_datasets) > 0:
            logger.info("Building BEIR eval cache (embed_dim=1024)...")
            beir_eval_cache = build_beir_eval_cache(
                dataset_list        = args.beir_datasets,
                embedding_base_path = args.beir_embedding_dir,
                beir_data_path      = args.beir_query_corpus_path,
                use_gpu             = False,
                embed_dim           = args.embed_dim,
            )

        if args.activate_eval_topiocqa_while_training:
            logger.info(f"Loading TopiOCQA corpus into FAISS (embed_dim={args.embed_dim})...")
            topiocqa_faiss_idx, topiocqa_doc_ids = load_corpus_into_faiss(
                embedding_dir = args.topiocqa_embedding_dir,
                embed_dim     = args.embed_dim,
                use_gpu       = False,
            )
            logger.info(f"TopiOCQA FAISS ready: {topiocqa_faiss_idx.ntotal} docs.")

    _gpu_faiss_cache: dict = {}

    if args.n_gpu > 1:
        dist.barrier()

    # 6. Resume from checkpoint
    if args.resume_from_latest and args.resume_from_checkpoint is None:
        ckpt_dirs = sorted(
            [d for d in (os.listdir(args.model_output_path) if os.path.isdir(args.model_output_path) else [])
             if d.startswith("checkpoint-step-")
             and os.path.exists(oj(args.model_output_path, d, "trainer_state.pt"))],
            key=lambda d: int(d.split("checkpoint-step-")[-1])
        )
        if ckpt_dirs:
            args.resume_from_checkpoint = oj(args.model_output_path, ckpt_dirs[-1])
            logger.info(f"Resuming from latest: {args.resume_from_checkpoint}")

    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint
        # Load inner Qwen3 model via AutoModel (avoids key-prefix issues from QwenQueryEncoder wrapper)
        model_core = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        model_core.model = AutoModel.from_pretrained(ckpt, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
                                                     **( {"attn_implementation": "flash_attention_2"} if args.use_flash_attention else {}))
        model_core.model = model_core.model.to(args.device)
        model_core.model.config.use_cache = False  # consistent with initial load
        optimizer.load_state_dict(torch.load(oj(ckpt, "optimizer.pt"), map_location="cpu"))
        optimizer_to(optimizer, args.device)
        scheduler.load_state_dict(torch.load(oj(ckpt, "scheduler.pt"), map_location="cpu"))
        state = torch.load(oj(ckpt, "trainer_state.pt"), map_location="cpu")
        cur_step, best_loss, start_epoch = state["step"], state["best_loss"], state["epoch"] + 1
        logger.info(f"Resumed from {ckpt}: step={cur_step}, epoch={start_epoch}")
    else:
        cur_step, start_epoch, best_loss = 0, 0, float("inf")

    # --------------------------------------------------------------------------
    logger.info(f"Start training (Qwen3-Embedding-0.6B): {len(train_dataset)} samples, "
                f"{args.num_train_epochs} epochs, {total_training_steps} steps total.")
    args.log_print_steps = max(1, int(args.log_print_ratio * steps_per_epoch))

    epoch_iterator = trange(start_epoch, args.num_train_epochs, desc="Epoch")

    for epoch in epoch_iterator:
        query_encoder.train()

        # Per-epoch curriculum DataLoader
        if args.curriculum_type != "none":
            epoch_start_step = epoch * steps_per_epoch
            pacing_value = get_pacing_value(epoch_start_step, curriculum_steps,
                                            args.curriculum_c0, args.pacing_function)
            N = len(train_dataset)
            if isinstance(pacing_value, tuple):
                start_frac, end_frac = pacing_value
                start_idx = int(start_frac * N)
                end_idx   = max(min(int(end_frac * N), N), start_idx + args.batch_size)
                subset    = Subset(train_dataset, range(start_idx, end_idx))
                n_active  = end_idx - start_idx
                pacing_log = f"{start_frac*100:.0f}%–{end_frac*100:.0f}%"
                wandb_pacing = end_frac - start_frac
            else:
                n_active  = max(args.batch_size, min(int(pacing_value * N), N))
                subset    = Subset(train_dataset, range(n_active))
                pacing_log = f"0%–{pacing_value*100:.1f}%"
                wandb_pacing = pacing_value

            if is_main_process:
                logger.info(f"Epoch {epoch}: curriculum {pacing_log} ({n_active}/{N} examples).")
                if args.save_to_wandb:
                    wandb.log({"curriculum/pacing_value": wandb_pacing,
                               "curriculum/n_active": n_active,
                               "curriculum/data_pct": wandb_pacing * 100.0,
                               "curriculum/direction": 1 if args.curriculum_type == "easy2hard" else -1},
                              step=cur_step)

            sampler = (DistributedSampler(subset, shuffle=True, drop_last=True)
                       if args.n_gpu > 1 else RandomSampler(subset))
            current_loader = DataLoader(subset,
                                        batch_size = args.per_gpu_train_batch_size,
                                        sampler    = sampler,
                                        collate_fn = train_dataset.get_collate_fn(args, **collate_fn_kwargs),
                                        drop_last  = True,
                                        pin_memory = True)
        else:
            current_loader = train_loader

        if args.curriculum_type != "none":
            batch_iter = make_cycling_iter(current_loader, micro_steps_per_epoch, epoch)
        else:
            if args.n_gpu > 1:
                current_loader.sampler.set_epoch(epoch)
            batch_iter = iter(current_loader)

        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if args.use_bf16 else contextlib.nullcontext())

        optimizer.zero_grad(set_to_none=True)
        n_batches = micro_steps_per_epoch
        epoch_step_start = epoch * steps_per_epoch

        for micro_step, batch in enumerate(tqdm(batch_iter, total=micro_steps_per_epoch, desc="Step")):
            is_last_in_accum = (
                (micro_step + 1) % args.gradient_accumulation_steps == 0
                or (micro_step + 1) == n_batches
            )

            sample_ids         = batch["sample_ids"]
            complex_query      = batch["complex_query"].to(args.device, non_blocking=True)
            complex_query_mask = batch["complex_query_mask"].to(args.device, non_blocking=True)

            ddp_sync_ctx = (contextlib.nullcontext()
                            if args.n_gpu <= 1 or is_last_in_accum
                            else query_encoder.no_sync())

            with ddp_sync_ctx:
                with autocast_ctx:
                    complex_query_embs = query_encoder(complex_query, complex_query_mask)

                # Pre-encoded pos/neg/oracle
                pos_doc_embs = torch.stack(
                    [sample_emb_table[sid]["pos"] for sid in sample_ids], dim=0
                ).to(args.device)

                if sample_emb_table[sample_ids[0]]["neg"] is not None and args.negative_type != "none":
                    neg_doc_embs = torch.stack(
                        [sample_emb_table[sid]["neg"] for sid in sample_ids], dim=0
                    ).to(args.device)
                else:
                    neg_doc_embs = None

                kd_loss = torch.tensor(0.0)
                if "kd" in args.loss_type:
                    oracle_utt_embs = torch.stack(
                        [sample_emb_table[sid]["oracle"] for sid in sample_ids], dim=0
                    ).to(args.device)

                with autocast_ctx:
                    ranking_loss = cal_ranking_loss(complex_query_embs, pos_doc_embs, neg_doc_embs)
                    if "kd" in args.loss_type:
                        kd_loss = cal_kd_loss(complex_query_embs, oracle_utt_embs)
                    loss = ranking_loss + kd_loss

                (loss / args.gradient_accumulation_steps).backward()

            if is_last_in_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                loss_val  = loss.item()
                optimizer_step_in_epoch = (micro_step + 1) // args.gradient_accumulation_steps
                cur_step  = epoch_step_start + min(optimizer_step_in_epoch, steps_per_epoch)

                if is_main_process and cur_step % args.log_print_steps == 0:
                    logger.info(f"Epoch={epoch} Step={cur_step}/{total_training_steps} Loss={loss_val:.7f}")
                    if args.save_to_wandb:
                        wandb.log({"train/loss": loss_val, "train/ranking_loss": ranking_loss.item(),
                                   "train/lr": scheduler.get_last_lr()[0],
                                   "train/grad_norm": grad_norm.item(), "epoch": epoch},
                                  step=cur_step)

        # End of epoch
        cur_step = (epoch + 1) * steps_per_epoch
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        if is_main_process:
            is_last_epoch = (epoch == args.num_train_epochs - 1)

            if args.activate_eval_while_training and beir_eval_cache is not None:
                with torch.no_grad():
                    # BEIR: force keep_faiss_on_gpu=False so each dataset's index is
                    # freed after search. Qwen3 (1024-dim) + TopiOCQA cache + BEIR cache
                    # together overflow 46 GB; loading one at a time peaks at ~40 GB.
                    metric_numbers = eval_beir_from_cache(
                        beir_cache = beir_eval_cache, query_encoder = query_encoder,
                        tokenizer = query_tokenizer, device = args.device,
                        eval_batch_size = args.eval_batch_size,
                        use_gpu_faiss = args.use_gpu_faiss, keep_faiss_on_gpu = False,
                        gpu_index_cache = _gpu_faiss_cache, full_eval = is_last_epoch,
                    )
                if args.save_to_wandb:
                    if is_last_epoch:
                        for ds, dmetrics in metric_numbers.items():
                            for mk, mv in dmetrics.items():
                                wandb.run.summary[f"final/{ds}/{mk}"] = mv
                    else:
                        for ds, ndcg in metric_numbers.items():
                            wandb.log({f"eval/{ds}_ndcg@10": ndcg}, step=cur_step)
                gc.collect(); torch.cuda.empty_cache()

            if args.activate_eval_topiocqa_while_training and topiocqa_faiss_idx is not None:
                with torch.no_grad():
                    topiocqa_metrics = eval_topiocqa(
                        query_encoder = query_encoder, tokenizer = query_tokenizer,
                        test_data_file = args.topiocqa_valid_file, qrel_file = args.topiocqa_qrel_file,
                        faiss_index = topiocqa_faiss_idx, doc_ids = topiocqa_doc_ids,
                        device = args.device, eval_batch_size = args.eval_batch_size,
                        max_query_length = args.max_query_length,
                        max_response_length = args.max_response_length,
                        max_concat_length = args.max_concat_length,
                        use_gpu_faiss = args.use_gpu_faiss, keep_faiss_on_gpu = args.keep_faiss_on_gpu,
                        gpu_index_cache = _gpu_faiss_cache, full_eval = is_last_epoch,
                        left_padding = True,
                    )
                if args.save_to_wandb:
                    if is_last_epoch:
                        for mk, mv in topiocqa_metrics.items():
                            wandb.run.summary[f"final/topiocqa/{mk}"] = mv
                    else:
                        wandb.log({f"eval/topiocqa_{k}": v for k, v in topiocqa_metrics.items()},
                                  step=cur_step)
                gc.collect(); torch.cuda.empty_cache()

            if best_loss > loss_val:
                best_loss = loss_val
            save_model(args, args.model_output_path, query_encoder, query_tokenizer,
                       optimizer, scheduler, cur_step, epoch, best_loss)
            if args.save_to_wandb:
                wandb.run.summary["best_loss"] = best_loss

        if args.n_gpu > 1:
            dist.barrier()

    logger.info("Training finished!")
    if args.save_to_wandb:
        wandb.finish()
    if args.n_gpu > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument("--pretrained_encoder_path", type=str,
                        default="/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_latest", action="store_true")
    parser.add_argument("--training_data_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl")
    parser.add_argument("--pos_neg_embedding_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_qwen/embeddings.pt")
    parser.add_argument("--model_output_path", type=str,
                        default="/data/rech/huiyuche/huggingface/continual_ir/topiocqa_qwen_cl")
    parser.add_argument("--encoder_type", type=str, default="qwen3",
                        help="Encoder type: 'qwen3' (default) for this script.")
    parser.add_argument("--embed_dim", type=int, default=1024,
                        help="Embedding dimension (1024 for Qwen3-Embedding-0.6B).")

    # Training recipe (same defaults as ANCE runs for fair comparison)
    parser.add_argument("--loss_type", type=str, default="ranking")
    parser.add_argument("--negative_type", type=str, default="none")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--no_lr_schedule", action="store_true")

    # Curriculum learning
    parser.add_argument("--curriculum_type", type=str, default="none",
                        choices=["none", "easy2hard", "hard2easy"])
    parser.add_argument("--scoring_function", type=str, default="turn_length",
                        choices=list(SCORING_FUNCTIONS.keys()))
    parser.add_argument("--pacing_function", type=str, default="root_2",
                        choices=list(PACING_FUNCTIONS.keys()))
    parser.add_argument("--curriculum_c0", type=float, default=0.2)
    parser.add_argument("--curriculum_end_epoch", type=int, default=16)

    # Memory / speed
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_tf32", action="store_true")
    parser.add_argument("--use_fused_optimizer", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_compile", action="store_true")

    # Eval
    parser.add_argument("--activate_eval_while_training", action="store_true")
    parser.add_argument("--beir_embedding_dir", type=str,
                        default="/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--beir_datasets", type=str, nargs='+',
                        default=["climate-fever", "msmarco"])
    parser.add_argument("--beir_query_corpus_path", type=str, default="/data/rech/huiyuche/beir")
    parser.add_argument("--use_gpu_faiss", action="store_true")
    parser.add_argument("--keep_faiss_on_gpu", action="store_true")

    # TopiOCQA eval
    parser.add_argument("--activate_eval_topiocqa_while_training", action="store_true")
    parser.add_argument("--topiocqa_valid_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl")
    parser.add_argument("--topiocqa_qrel_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec")
    parser.add_argument("--topiocqa_embedding_dir", type=str,
                        default="/part/01/Tmp/yuchen/indexes/topiocqa_qwen_merged",
                        help="Dir with Qwen3 corpus blocks (doc_emb_block.*.pb) for TopiOCQA eval.")

    # Misc
    parser.add_argument("--log_print_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_data_percent", type=float, default=1.0)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)
    parser.add_argument("--save_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="topiocqa-qwen")
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()

    if args.n_gpu > 1:
        from datetime import timedelta
        dist.init_process_group(backend='nccl', init_method='env://',
                                timeout=timedelta(hours=2))
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("Args: %s", args)
    return args


if __name__ == '__main__':
    args = get_args()
    is_main_process = (args.n_gpu == 1) or (dist.get_rank() == 0)

    if is_main_process and args.save_to_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, resume="allow")
        wandb.config.update(vars(args))

    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    set_seed(args)
    train(args)
