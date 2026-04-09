import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
import argparse
import contextlib

sys.path.append('..')
sys.path.append('.')

import time
import os
from os.path import join as oj

import gc

from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer

import wandb

from models import ANCE
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


def save_model(args, model_output_path, model, query_tokenizer, optimizer, scheduler, cur_step, cur_epoch, best_loss):
    """
    Save a full training checkpoint including:
      - Model weights
      - Tokenizer
      - Optimizer state
      - Scheduler state
      - Current step and best loss (trainer state)

    This structure allows you to resume training later.
    """

    # Create a unique directory per checkpoint (e.g., by step)
    output_dir = oj(model_output_path, f"checkpoint-step-{cur_step}")

    # if exit, skip
    if os.path.exists(output_dir):
        logger.info(f"Checkpoint {output_dir} already exists, skipping save.")
        return

    else:

        os.makedirs(output_dir, exist_ok=True)

        # ====== Save model and tokenizer ======

        # If model is wrapped in DistributedDataParallel, unwrap it
        model_to_save = model.module if hasattr(model, "module") else model

        # Save the model in HuggingFace format
        model_to_save.save_pretrained(output_dir)

        # Save the tokenizer as well
        query_tokenizer.save_pretrained(output_dir)

        # ====== Save optimizer and scheduler states ======

        # Optimizer state is required to resume training with momentum/Adam states
        torch.save(optimizer.state_dict(), oj(output_dir, "optimizer.pt"))

        # Scheduler state is required to resume the learning rate schedule
        torch.save(scheduler.state_dict(), oj(output_dir, "scheduler.pt"))

        # ====== Save training state info ======

        # This file stores metadata about training progress (e.g., step, best loss)
        torch.save({
            "step": cur_step,
            "epoch": cur_epoch,
            "best_loss": best_loss,
        }, oj(output_dir, "trainer_state.pt"))

        logger.info(f"Saved checkpoint at {output_dir}")


def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs=None):
    # get batch size
    batch_size = query_embs.size(0)

    # relevance scores: B * B
    pos_scores = query_embs @ pos_doc_embs.T  # [B, B]

    if neg_doc_embs is not None:
        # B * 1 hard negative scores
        neg_scores = torch.sum(query_embs * neg_doc_embs, dim=1, keepdim=True)
        # B * (B + 1), scores = in_batch negatives + 1 BM25 hard negative
        score_mat = torch.cat([pos_scores, neg_scores], dim=1)
    else:
        score_mat = pos_scores

    # where exist the positive documents (the diagonal elements)
    labels = torch.arange(batch_size, device=query_embs.device)

    return nn.CrossEntropyLoss()(score_mat, labels)


def train(args):
    # -1
    is_main_process = (args.n_gpu == 1) or (dist.get_rank() == 0)

    # 0. load pre-encoded pos, neg, oracle embeddings if any
    if args.pos_neg_embedding_file is not None:
        # dict: sample_id -> (pos_emb, neg_emb, oracle_emb)
        sample_emb_table = torch.load(
            args.pos_neg_embedding_file,
            map_location= "cpu"
            )

    # 1. Load query encoder (with optional Flash Attention 2 + bf16)
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    query_tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)

    # Build kwargs for ANCE.from_pretrained.
    # --use_bf16 loads weights in BF16 (halves weight memory).
    # --use_flash_attention additionally requests FA2; falls back gracefully if unsupported.
    model_kwargs = {}
    if args.use_bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.use_flash_attention:
        if getattr(ANCE, "_supports_flash_attn_2", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled.")
        else:
            logger.warning("--use_flash_attention set but ANCE does not support FA2; "
                           "falling back to standard attention. BF16 weights still active if --use_bf16.")

    query_encoder = ANCE.from_pretrained(
        args.pretrained_encoder_path, config=config, **model_kwargs
    ).to(args.device)

    # if we did not encode the pos neg and oracle embeddings beforehand
    if args.pos_neg_embedding_file is None:
        passage_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    if args.n_gpu > 1:
        query_encoder = DDP(query_encoder, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        dist.barrier()

    # Optional: TF32 for faster matmul on Ampere (A6000) — negligible precision loss
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 matmul enabled.")

    # Optional: gradient checkpointing — saves ~50% activation memory at ~20% compute cost
    if args.gradient_checkpointing:
        model_to_gc = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        model_to_gc.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    # Optional: torch.compile — 20–40% forward pass speedup (PyTorch 2.0+)
    if args.use_compile:
        query_encoder = torch.compile(query_encoder)
        logger.info("torch.compile enabled.")

    # 2. Prepare training data
    train_dataset = Topiocqa(
        args,
        query_tokenizer,
        args.training_data_file
    )

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # ---- Curriculum: sort dataset once at startup ----
    if args.curriculum_type != "none":
        if args.scoring_function not in SCORING_FUNCTIONS:
            raise ValueError(f"Unknown scoring function '{args.scoring_function}'. "
                             f"Available: {list(SCORING_FUNCTIONS.keys())}")
        scoring_fn = SCORING_FUNCTIONS[args.scoring_function]
        ascending   = (args.curriculum_type == "easy2hard")  # False → hard2easy
        train_dataset.sort_by_difficulty(scoring_fn, ascending=ascending)
        logger.info(f"Curriculum type={args.curriculum_type}, "
                    f"scoring={args.scoring_function}, "
                    f"pacing={args.pacing_function}, "
                    f"c0={args.curriculum_c0}, "
                    f"end_epoch={args.curriculum_end_epoch}")

    # ---- Step counts ----
    # steps_per_epoch counts *optimizer* steps (i.e., micro-batches / gradient_accumulation_steps).
    # Using full dataset length keeps LR schedule consistent regardless of curriculum subset.
    micro_steps_per_epoch = len(train_dataset) // args.batch_size
    steps_per_epoch       = micro_steps_per_epoch // args.gradient_accumulation_steps
    total_training_steps  = args.num_train_epochs * steps_per_epoch
    curriculum_steps      = args.curriculum_end_epoch * steps_per_epoch

    if is_main_process and args.curriculum_type != "none":
        log_pacing_schedule(
            num_epochs       = args.num_train_epochs,
            steps_per_epoch  = steps_per_epoch,
            curriculum_steps = curriculum_steps,
            c0               = args.curriculum_c0,
            pacing_fn_name   = args.pacing_function,
        )

    # ---- Standard (non-curriculum) DataLoader, built once ----
    # For curriculum training, per-epoch loaders are built inside the epoch loop.
    if args.curriculum_type == "none":
        if args.n_gpu > 1:
            sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size   = args.per_gpu_train_batch_size,
            sampler      = sampler,
            collate_fn   = train_dataset.get_collate_fn(args, pad_token_id=query_tokenizer.pad_token_id),
            drop_last    = True,
            pin_memory   = True,
        )

    # 3. Optimizer (optional fused AdamW for faster optimizer step on CUDA)
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay,
                              fused=args.use_fused_optimizer)

    # 4. LR scheduler (based on full dataset steps — unaffected by curriculum subset)
    if args.no_lr_schedule:
        # Constant LR — no warmup, no decay.
        # Useful to isolate whether LR schedule contributes to any observed gain.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        if is_main_process:
            logger.info("LR scheduler: constant (no warmup, no decay).")
            if args.save_to_wandb:
                wandb.config.update({"num_warmup_steps": 0, "lr_schedule": "constant"})
    else:
        # Linear warmup + linear decay (default, matches baseline wandb run 1ppvc6dl)
        num_warmup = int(args.warmup_ratio * total_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = num_warmup,
            num_training_steps = total_training_steps,
        )
        if is_main_process:
            logger.info("LR scheduler: linear warmup ({} steps) + linear decay.".format(num_warmup))
            if args.save_to_wandb:
                wandb.config.update({"num_warmup_steps": num_warmup, "lr_schedule": "linear_warmup_decay"})

    # ---- Load corpus into memory for fast per-epoch evaluation ----
    # This is done once at startup; per-epoch eval only re-encodes queries.
    beir_eval_cache    = None
    topiocqa_faiss_idx = None
    topiocqa_doc_ids   = None

    if is_main_process:

        if args.activate_eval_while_training and len(args.beir_datasets) > 0:
            logger.info("Building in-memory BEIR eval cache (loads corpus to CPU RAM once) ...")
            beir_eval_cache = build_beir_eval_cache(
                dataset_list        = args.beir_datasets,
                embedding_base_path = args.beir_embedding_dir,
                beir_data_path      = args.beir_query_corpus_path,
                use_gpu             = False,  # always CPU; GPU transfer handled per-eval via use_gpu_faiss
            )
            logger.info("BEIR eval cache ready.")

        if args.activate_eval_topiocqa_while_training:
            logger.info(f"Loading TopiOCQA corpus into FAISS (CPU) from {args.topiocqa_embedding_dir} ...")
            topiocqa_faiss_idx, topiocqa_doc_ids = load_corpus_into_faiss(
                embedding_dir = args.topiocqa_embedding_dir,
                use_gpu       = False,  # always CPU; GPU transfer handled per-eval via use_gpu_faiss
            )
            logger.info(f"TopiOCQA FAISS index ready: {topiocqa_faiss_idx.ntotal} docs.")

    # Mutable dict for caching GPU FAISS indices across epochs (used when keep_faiss_on_gpu=True).
    # Keys: dataset_name (BEIR) or "topiocqa".  Values: GPU faiss index objects.
    _gpu_faiss_cache: dict = {}

    # Wait for rank 0 to finish loading all corpora before any rank enters training.
    # Without this barrier, non-main ranks would hit the first epoch-end dist.barrier()
    # while rank 0 is still loading embeddings → NCCL timeout.
    if args.n_gpu > 1:
        dist.barrier()

    # begin to train
    logger.info("Start training...")
    logger.info("Total training samples = {}".format(len(train_dataset)))
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    logger.info("Num steps per epoch = {}".format(steps_per_epoch))
    args.log_print_steps = max(1, int(args.log_print_ratio * steps_per_epoch))

    # if we want to resume
    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint

        # 1) Load model weights
        model_to_load = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        model_to_load.load_state_dict(
            torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
        )

        # 2) Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.pt"), map_location="cpu"))
        optimizer_to(optimizer, args.device)
        scheduler.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.pt"), map_location="cpu"))

        # 3) Load trainer state (global step, best loss)
        state = torch.load(os.path.join(ckpt, "trainer_state.pt"), map_location="cpu")
        cur_step  = state["step"]
        best_loss = state["best_loss"]
        epoch     = state["epoch"]
        start_epoch = epoch + 1  # next epoch to run

        logger.info(f"Resumed from checkpoint {ckpt}: step={cur_step}, best_loss={best_loss}")

    else:
        cur_step    = 0
        start_epoch = 0
        best_loss   = float("inf")


    ######################################
    #### Main Training Loop: For each Epoch
    ######################################
    epoch_iterator = trange(start_epoch, args.num_train_epochs, desc="Epoch")

    for epoch in epoch_iterator:

        query_encoder.train()

        # ---- Per-epoch curriculum DataLoader ----
        # For curriculum training, we rebuild the DataLoader each epoch with a
        # Subset whose size grows according to the pacing function.
        if args.curriculum_type != "none":
            epoch_start_step = epoch * steps_per_epoch
            pacing_value = get_pacing_value(
                global_step      = epoch_start_step,
                curriculum_steps = curriculum_steps,
                c0               = args.curriculum_c0,
                pacing_fn_name   = args.pacing_function,
            )
            # Clamp: at least one full batch, at most the full dataset
            n_active = max(args.batch_size, int(pacing_value * len(train_dataset)))
            n_active = min(n_active, len(train_dataset))
            subset   = Subset(train_dataset, range(n_active))

            if is_main_process:
                logger.info(f"Epoch {epoch}: curriculum pacing_value={pacing_value:.4f}, "
                            f"using {n_active}/{len(train_dataset)} examples "
                            f"({100*pacing_value:.1f}%).")
                if args.save_to_wandb:
                    wandb.log({
                        "curriculum/pacing_value": pacing_value,
                        "curriculum/n_active":     n_active,
                        "curriculum/data_pct":     pacing_value * 100.0,
                    }, step=cur_step)

            if args.n_gpu > 1:
                sampler = DistributedSampler(subset, shuffle=True, drop_last=True)
            else:
                sampler = RandomSampler(subset)

            current_loader = DataLoader(
                subset,
                batch_size = args.per_gpu_train_batch_size,
                sampler    = sampler,
                collate_fn = train_dataset.get_collate_fn(args, pad_token_id=query_tokenizer.pad_token_id),
                drop_last  = True,
                pin_memory = True,
            )
        else:
            # Standard training: use the pre-built loader
            current_loader = train_loader

        # yuchen: necessary to make the distribution of samples correct in DDP
        if args.n_gpu > 1:
            current_loader.sampler.set_epoch(epoch)

        # BF16 autocast context for memory optimization (used if --use_bf16)
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.use_bf16
            else contextlib.nullcontext()
        )

        optimizer.zero_grad()  # zero once before the loop; re-zero after each optimizer step
        n_batches = len(current_loader)

        for micro_step_in_epoch, batch in enumerate(tqdm(current_loader, desc="Step")):

            # Is this the last micro-batch in the current accumulation window?
            is_last_in_accum = (
                (micro_step_in_epoch + 1) % args.gradient_accumulation_steps == 0
                or (micro_step_in_epoch + 1) == n_batches  # last batch of epoch
            )

            #########################
            ### Take inputs
            #########################
            sample_ids = batch['sample_ids']  # B

            complex_query      = batch['complex_query'].to(args.device, non_blocking=True)
            complex_query_mask = batch['complex_query_mask'].to(args.device, non_blocking=True)

            # Suppress DDP gradient all-reduce until the last step in the accumulation window.
            # This avoids redundant cross-GPU communication on intermediate backward passes.
            ddp_sync_ctx = (
                contextlib.nullcontext()
                if args.n_gpu <= 1 or is_last_in_accum
                else query_encoder.no_sync()
            )

            # ddp_sync_ctx wraps the full forward + backward so that DDP gradient
            # all-reduce is suppressed on intermediate accumulation steps.
            with ddp_sync_ctx:
                with autocast_ctx:
                    complex_query_embs = query_encoder(complex_query, complex_query_mask)  # B * dim

                ###############################
                ### calcualte (get) embeddings.
                ###############################

                # if not pre-encoded, encode on the fly
                if args.pos_neg_embedding_file is None:

                    pos_docs      = batch['pos_docs'].to(args.device, non_blocking=True)
                    pos_docs_mask = batch['pos_docs_mask'].to(args.device, non_blocking=True)

                    if args.negative_type != "none":
                        neg_docs      = batch['neg_docs'].to(args.device, non_blocking=True)
                        neg_docs_mask = batch['neg_docs_mask'].to(args.device, non_blocking=True)

                    if "kd" in args.loss_type:
                        oracle_query      = batch['oracle_qr'].to(args.device, non_blocking=True)
                        oracle_query_mask = batch['oracle_qr_mask'].to(args.device, non_blocking=True)

                    passage_encoder.eval()
                    with torch.no_grad():
                        # doc encoder's parameters are frozen
                        pos_doc_embs = passage_encoder(pos_docs, pos_docs_mask).detach()  # B * dim

                        if args.negative_type != "none":
                            neg_doc_embs = passage_encoder(neg_docs, neg_docs_mask).detach()  # B * dim
                        else:
                            neg_doc_embs = None

                        if "kd" in args.loss_type:
                            oracle_utt_embs = passage_encoder(oracle_query, oracle_query_mask).detach()

                else:  # pre-encoded pos, neg, oracle embeddings

                    pos_doc_embs = torch.stack(
                        [sample_emb_table[sid]["pos"] for sid in sample_ids], dim=0
                    ).to(args.device)   # [B, dim]

                    if sample_emb_table[sample_ids[0]]["neg"] is not None and args.negative_type != "none":
                        neg_doc_embs = torch.stack(
                            [sample_emb_table[sid]["neg"] for sid in sample_ids], dim=0
                        ).to(args.device)
                    else:
                        neg_doc_embs = None

                    if "kd" in args.loss_type:
                        oracle_utt_embs = torch.stack(
                            [sample_emb_table[sid]["oracle"] for sid in sample_ids], dim=0
                        ).to(args.device)

                # end if: pre-encoded pos neg oracle embeddings

                if args.negative_type == "none":
                    assert neg_doc_embs is None, "neg_doc_embs should be None when negative_type is 'none'."

                with autocast_ctx:
                    ranking_loss = cal_ranking_loss(complex_query_embs, pos_doc_embs, neg_doc_embs)

                    kd_loss = torch.tensor(0.0)
                    if "kd" in args.loss_type:
                        kd_loss = cal_kd_loss(complex_query_embs, oracle_utt_embs)
                        loss = ranking_loss + kd_loss
                    else:
                        loss = ranking_loss

                # Scale by 1/accumulation_steps so averaged gradients match a single large batch.
                (loss / args.gradient_accumulation_steps).backward()

            # end ddp_sync_ctx

            # Only update weights after accumulating gradient_accumulation_steps micro-batches.
            if is_last_in_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss         = loss.item()
                ranking_loss = ranking_loss.item()
                kd_loss      = kd_loss.item()
                grad_norm    = grad_norm.item()

                # print info
                if is_main_process and cur_step % args.log_print_steps == 0:
                    logger.info("Epoch = {}, Current Step = {}, Total Step = {}, Loss = {}".format(
                        epoch, cur_step, total_training_steps, round(loss, 7))
                    )
                    if args.save_to_wandb:
                        wandb.log(
                            {
                                "train/loss":         loss,
                                "train/ranking_loss": ranking_loss,
                                "train/kd_loss":      kd_loss,
                                "train/lr":           scheduler.get_last_lr()[0],
                                "train/grad_norm":    grad_norm,
                                "epoch":              epoch,
                            },
                            step=cur_step
                        )
                cur_step += 1

        # end for: each batch

        # Per-epoch cleanup (zero_grad now happens inside loop after each optimizer step,
        # but call again here to handle any leftover gradients from incomplete accum window)
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        if is_main_process:

            # ---- BEIR eval (in-memory, fast) ----
            if args.activate_eval_while_training and beir_eval_cache is not None:
                with torch.no_grad():
                    metric_numbers = eval_beir_from_cache(
                        beir_cache         = beir_eval_cache,
                        query_encoder      = query_encoder,
                        tokenizer          = query_tokenizer,
                        device             = args.device,
                        eval_batch_size    = args.eval_batch_size,
                        use_gpu_faiss      = args.use_gpu_faiss,
                        keep_faiss_on_gpu  = args.keep_faiss_on_gpu,
                        gpu_index_cache    = _gpu_faiss_cache,
                    )

                if args.save_to_wandb:
                    for dataset_name, ndcg10 in metric_numbers.items():
                        wandb.log({f"eval/{dataset_name}_ndcg@10": ndcg10}, step=cur_step)
                    wandb.log({"eval/loss": loss}, step=cur_step)

                gc.collect()
                torch.cuda.empty_cache()

            # ---- TopiOCQA eval (in-memory, fast) ----
            if args.activate_eval_topiocqa_while_training and topiocqa_faiss_idx is not None:
                with torch.no_grad():
                    topiocqa_metrics = eval_topiocqa(
                        query_encoder       = query_encoder,
                        tokenizer           = query_tokenizer,
                        test_data_file      = args.topiocqa_valid_file,
                        qrel_file           = args.topiocqa_qrel_file,
                        faiss_index         = topiocqa_faiss_idx,
                        doc_ids             = topiocqa_doc_ids,
                        device              = args.device,
                        eval_batch_size     = args.eval_batch_size,
                        max_query_length    = args.max_query_length,
                        max_response_length = args.max_response_length,
                        max_concat_length   = args.max_concat_length,
                        use_gpu_faiss       = args.use_gpu_faiss,
                        keep_faiss_on_gpu   = args.keep_faiss_on_gpu,
                        gpu_index_cache     = _gpu_faiss_cache,
                    )

                if args.save_to_wandb:
                    wandb.log({
                        f"eval/topiocqa_{k}": v
                        for k, v in topiocqa_metrics.items()
                    }, step=cur_step)

                gc.collect()
                torch.cuda.empty_cache()

            # end if: eval while training

            # Update best_loss before saving so the checkpoint reflects the true best
            if best_loss > loss:
                best_loss = loss
            save_model(
                args,
                args.model_output_path,
                query_encoder,
                query_tokenizer,
                optimizer,
                scheduler,
                cur_step,
                epoch,
                best_loss
            )
            if args.save_to_wandb:
                wandb.run.summary["best_loss"] = best_loss

        # end if: is main process

        # wait for all processes before next epoch
        if args.n_gpu > 1:
            dist.barrier()


    logger.info("Training finish!")

    if args.save_to_wandb:
        wandb.finish()

    if args.n_gpu > 1:
        dist.barrier()
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.')

    parser.add_argument('--n_gpu', type=int, default=4, help='The number of used GPU.')
    parser.add_argument("--pretrained_encoder_path", type=str, default="/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--training_data_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl")
    parser.add_argument("--pos_neg_embedding_file", type=str, default=None)
    parser.add_argument("--model_output_path", type=str, default="/data/rech/huiyuche/huggingface/continual_ir/topiocqa_cl")

    parser.add_argument("--loss_type", type=str, default="ranking", help="Options: kd, kd+ranking, ranking")
    parser.add_argument("--negative_type", type=str, default="none", help="Options: bm25_hard, none")

    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N micro-batches before each optimizer step. "
                             "Effective batch = per_gpu_batch * n_gpu * gradient_accumulation_steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warm up ratio w.r.t total training steps.")
    parser.add_argument("--no_lr_schedule", action="store_true",
                        help="Use constant LR (no warmup, no linear decay). "
                             "Overrides --warmup_ratio. Use to isolate LR schedule contribution.")

    # ---- Curriculum learning ----
    parser.add_argument("--curriculum_type", type=str, default="none",
                        choices=["none", "easy2hard", "hard2easy"],
                        help="Curriculum type: none = standard random, easy2hard = curriculum, hard2easy = anti-curriculum.")
    parser.add_argument("--scoring_function", type=str, default="turn_length",
                        choices=list(SCORING_FUNCTIONS.keys()),
                        help="Difficulty scoring function.")
    parser.add_argument("--pacing_function", type=str, default="root_2",
                        choices=list(PACING_FUNCTIONS.keys()),
                        help="Pacing function controlling how fast the data fraction grows.")
    parser.add_argument("--curriculum_c0", type=float, default=0.2,
                        help="Initial data fraction (delta_p). Set based on analyze_topiocqa_turns.py output.")
    parser.add_argument("--curriculum_end_epoch", type=int, default=16,
                        help="Epoch at which the full dataset is exposed (pacing_value reaches 1.0).")

    # ---- Memory / speed optimizations ----
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Enable Flash Attention 2 (requires bf16; A6000 compute cap 8.6 supported).")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Enable bfloat16 autocast for forward/loss passes.")
    parser.add_argument("--use_tf32", action="store_true",
                        help="Enable TF32 matmul on Ampere GPUs (~20% speedup, negligible precision loss).")
    parser.add_argument("--use_fused_optimizer", action="store_true",
                        help="Use fused AdamW (PyTorch >= 2.0; faster optimizer step on CUDA).")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (~50% activation memory, ~20% slower).")
    parser.add_argument("--use_compile", action="store_true",
                        help="Enable torch.compile for forward pass (PyTorch 2.0+; ~20-40% speedup).")

    # ---- Eval settings ----
    parser.add_argument("--activate_eval_while_training", action="store_true",
                        help="Evaluate on BEIR datasets each epoch (corpus loaded into RAM at startup).")
    parser.add_argument("--beir_embedding_dir", type=str,
                        default="/data/rech/huiyuche/beir/embeddings/ance",
                        help="Base dir for BEIR corpus embeddings (corpus.*.pkl blocks).")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Batch size for query encoding during eval.")
    parser.add_argument("--beir_datasets", type=str, nargs='+',
                        default=["climate-fever", "msmarco"],
                        help="Which BEIR datasets to eval on.")
    parser.add_argument("--beir_query_corpus_path", type=str,
                        default="/data/rech/huiyuche/beir",
                        help="Base dir for BEIR text data (queries + qrels).")
    parser.add_argument("--use_gpu_faiss", action="store_true",
                        help="Shard FAISS indices across GPUs at eval time (requires faiss-gpu). "
                             "Corpus is always loaded into CPU RAM; GPU transfer happens per eval call.")
    parser.add_argument("--keep_faiss_on_gpu", action="store_true",
                        help="If --use_gpu_faiss, keep GPU indices cached across epochs instead of "
                             "freeing after each eval. Saves ~20-25s per epoch on high-VRAM machines.")

    # ---- TopiOCQA eval ----
    parser.add_argument("--activate_eval_topiocqa_while_training", action="store_true",
                        help="Evaluate on TopiOCQA validation set each epoch.")
    parser.add_argument("--topiocqa_valid_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_valid.jsonl")
    parser.add_argument("--topiocqa_qrel_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/topiocqa_qrel.trec")
    parser.add_argument("--topiocqa_embedding_dir", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_ance_merged",
                        help="Directory with TopiOCQA corpus embedding blocks (doc_emb_block.*.pb).")

    # ---- Misc ----
    parser.add_argument("--log_print_ratio", type=float, default=0.1,
                        help="Fraction of steps per epoch at which to print a log line.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0,
                        help="Fraction of training data to use. Set < 1.0 for smoke tests.")
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=512, help="Max doc length.")
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)

    parser.add_argument("--save_to_wandb", action="store_true", help="Log results to wandb.")
    parser.add_argument("--wandb_project", type=str, default="topiocqa-ance")
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()

    # pytorch parallel gpu
    if args.n_gpu > 1:
        # Set a long timeout so rank 0's per-epoch eval (corpus loading, FAISS search)
        # doesn't cause other ranks to hit the default 10-min NCCL watchdog timeout.
        from datetime import timedelta
        dist.init_process_group(backend='nccl', init_method='env://',
                                timeout=timedelta(hours=2))
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
        args.device = device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device

    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    return args


if __name__ == '__main__':
    args = get_args()
    is_main_process = (args.n_gpu == 1) or (dist.get_rank() == 0)

    if is_main_process and args.save_to_wandb:
        run = wandb.init(
            project = args.wandb_project,
            name    = args.wandb_name,
            resume  = "allow",
        )
        wandb.config.update(vars(args))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    args.run_id = run_id

    set_seed(args)
    train(args)
