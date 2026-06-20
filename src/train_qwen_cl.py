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
import numpy as np

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

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
)
import wandb

from utils import (
    eval_beir_datasets,
    build_beir_eval_cache,
    eval_beir_from_cache,
    build_qwen_instruction_map,
    eval_conv_search,
    load_corpus_into_faiss,
    set_seed,
    get_optimizer,
    optimizer_to,
)
from data import Topiocqa, IKATGradedDataset
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
                 use_bf16: bool = False, fp32_master_weights: bool = False,
                 bf16_fp32_master: bool = False):
        super().__init__()
        kwargs = {"trust_remote_code": True}
        if fp32_master_weights:
            # fp32 MASTER WEIGHTS: load the (released bf16) weights upcast to fp32 so
            # the optimizer keeps fp32 master params + fp32 Adam moments. With
            # autocast(bf16) around the forward (controlled by --use_bf16), matmuls
            # still run in bf16, so this reproduces the numerics of the official
            # DeepSpeed-ZeRO bf16 setup (= HF Trainer bf16=True) WITHOUT pure-bf16's
            # update-rounding floor: at lr=1e-5 a step is below bf16 ULP for
            # |theta|>=0.0026 and gets rounded away, so ~90% of weights never move.
            # FlashAttention-2 requires half-precision weights, so we fall back to
            # sdpa, which computes the SAME exact softmax(QK^T/sqrt(d))V (different
            # kernel, numerically equivalent — verified by a forward cos>0.999 check).
            kwargs["attn_implementation"] = "sdpa"
            # NB: do NOT set torch_dtype -> model loads in fp32.
        else:
            # Normal bf16 / FA2 path. The CANONICAL fp32-master mode (bf16_fp32_master)
            # also lives here: the MODEL is bf16 (so FlashAttention-2 works) and a
            # separate fp32 master copy + fp32 AdamW are held by train() — that copy,
            # not this module, is what avoids the bf16 update floor. The fp32 master
            # re-adds ~1 model-worth of fp32 state, so the VRAM win over the fp32-master
            # path is NOT in fixed state but in activations: FA2 (vs sdpa) avoids the
            # O(L^2) attention buffer and lets us turn gradient_checkpointing OFF — the
            # net is measured with --profile_timing. Update numerics stay identical.
            if use_bf16 or use_flash_attention or bf16_fp32_master:
                kwargs["torch_dtype"] = torch.bfloat16
            if use_flash_attention:
                kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModel.from_pretrained(model_path, **kwargs)
        # Disable KV cache — we only do forward/backward for embedding, no generation.
        # Without this, Qwen decoder still materializes past_key_values → extra activation memory.
        # Also a prerequisite for gradient_checkpointing to be effective.
        self.model.config.use_cache = False
        # Learnable scalar bias beta for the BiXSE (graded BCE) loss; lives on the OUTER
        # module (not self.model) so save_pretrained never serializes it. train() toggles
        # its requires_grad: True only for the graded `bce` loss, else frozen+unused (so DDP
        # find_unused_parameters=False stays valid and TopiOCQA runs are unaffected).
        self.bce_beta = nn.Parameter(torch.zeros(()))

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


def _all_gather_detached(t, world_size):
    """All-gather a DETACHED tensor across all DDP ranks.

    The doc embeddings are pre-encoded constants (loaded from sample_emb_table,
    requires_grad=False), so a non-differentiable all_gather is sufficient — no
    gradient needs to flow back through the gathered docs. Returns the
    rank-ordered concatenation [rank0 block, rank1 block, ...] so the global
    index of local row `i` on rank `r` is `r * B + i`.
    """
    if t is None or world_size <= 1:
        return t
    t = t.contiguous()
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0)


def _apply_fake_negative_mask(scores, labels, temperature, *, pos_pids, neg_pids,
                              world_size, neg_present, mask_fake_negative,
                              fake_neg_margin, dedup_same_gold):
    """Set false-negative candidate logits to -inf; the positive column is never masked.

    Two complementary signals (official Qwen3-Embedding + RocketQA-style de-dup):
    - similarity mask (INFONCE_MASK_FAKE_NEGATIVE): mask candidate j of query i when
      sim(q_i, cand_j) > sim(q_i, pos_i) + margin (margin on raw cosine). `scores` are
      already divided by temperature, so the cosine margin becomes margin/temperature.
    - same-gold de-dup: mask candidate j when its passage id == query i's gold pid. This
      catches EXACT duplicate positives the similarity mask misses (a duplicate has
      sim == pos, which never exceeds pos+margin).
    The own-positive column (labels[i]) is excluded from masking via `not_self`, so the
    cross-entropy target always keeps a finite logit.
    """
    B, K = scores.shape
    col = torch.arange(K, device=scores.device).unsqueeze(0)          # (1, K)
    not_self = col != labels.unsqueeze(1)                            # (B, K): False at own positive
    mask = torch.zeros_like(scores, dtype=torch.bool)

    if mask_fake_negative:
        own_pos = scores.gather(1, labels.unsqueeze(1))             # (B, 1) own positive logit (/temp)
        mask |= (scores > own_pos + fake_neg_margin / temperature) & not_self

    if dedup_same_gold and pos_pids is not None:
        # candidate pids aligned with the K columns: gathered pos pids (then neg pids).
        pos_pid_pool = _all_gather_detached(pos_pids, world_size)            # (W*B,)
        cand_pids = pos_pid_pool
        if neg_present and neg_pids is not None:
            neg_pid_pool = _all_gather_detached(neg_pids, world_size)        # (W*B,)
            cand_pids = torch.cat([pos_pid_pool, neg_pid_pool], dim=0)       # (K,)
        # Guard the -1 "missing pid" sentinel so it can never participate in the
        # equality (a -1==-1 match would false-mask placeholder columns). No-op on
        # the current TopiOCQA data (no -1 pids), but keeps de-dup robust to any
        # future data with empty pos/neg pids.
        valid = (pos_pids.unsqueeze(1) != -1) & (cand_pids.unsqueeze(0) != -1)
        same_gold = (cand_pids.unsqueeze(0) == pos_pids.unsqueeze(1)) & not_self & valid  # (B, K)
        mask |= same_gold

    return scores.masked_fill(mask, float("-inf"))


def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs=None,
                     temperature=1.0, cross_device=False, world_size=1, rank=0,
                     pos_pids=None, neg_pids=None,
                     mask_fake_negative=False, fake_neg_margin=0.1,
                     dedup_same_gold=False):
    """InfoNCE / in-batch contrastive loss.

    temperature: similarity logits are divided by this (InfoNCE). 1.0 = legacy
        no-op; official Qwen3-Embedding uses 0.01.
    cross_device: when True and world_size>1, all_gather the (frozen) pos/neg
        doc embeddings across ranks so each query's in-batch negatives are the
        GLOBAL batch (official INFONCE_USE_BATCH recipe). Only the local queries
        carry gradient; DDP all-reduce averages the per-rank gradients, giving
        the effect of a global B*world_size InfoNCE batch.
    """
    batch_size = query_embs.size(0)
    if cross_device and world_size > 1:
        # Gather frozen doc embeddings → global candidate pool. Cast to fp32 so
        # the gathered tensors and the (fp32-normalized) query embs share dtype
        # under autocast, and the all_gather is dtype-consistent across ranks.
        pos_pool = _all_gather_detached(pos_doc_embs.float(), world_size)        # (W*B, d)
        candidates = pos_pool
        if neg_doc_embs is not None:
            # Hard negatives become GLOBAL shared negatives (FlagEmbedding/official
            # style), unlike the legacy single per-query diagonal neg below.
            neg_pool = _all_gather_detached(neg_doc_embs.float(), world_size)    # (W*B, d)
            candidates = torch.cat([pos_pool, neg_pool], dim=0)                  # (2WB, d)
        scores = (query_embs.float() @ candidates.T) / temperature              # (B, K)
        # Local query i's positive sits at global column rank*B + i (positives
        # occupy the first W*B columns; negatives, if any, follow).
        labels = rank * batch_size + torch.arange(batch_size, device=query_embs.device)
        if mask_fake_negative or dedup_same_gold:
            scores = _apply_fake_negative_mask(
                scores, labels, temperature,
                pos_pids=pos_pids, neg_pids=neg_pids, world_size=world_size,
                neg_present=(neg_doc_embs is not None),
                mask_fake_negative=mask_fake_negative, fake_neg_margin=fake_neg_margin,
                dedup_same_gold=dedup_same_gold)
        return nn.CrossEntropyLoss()(scores, labels)

    # ---- legacy local-only path (now temperature-aware) ----
    pos_scores = (query_embs @ pos_doc_embs.T) / temperature
    if neg_doc_embs is not None:
        neg_scores = torch.sum(query_embs * neg_doc_embs, dim=1, keepdim=True) / temperature
        score_mat  = torch.cat([pos_scores, neg_scores], dim=1)
    else:
        score_mat = pos_scores
    labels = torch.arange(batch_size, device=query_embs.device)
    return nn.CrossEntropyLoss()(score_mat, labels)


def cal_kd_loss(query_embs, oracle_query_embs):
    return nn.MSELoss()(query_embs, oracle_query_embs)


def cal_graded_loss(query_embs, cand_embs, cand_grades, cand_mask, loss_type,
                    bce_logit_scale=20.0, bce_beta=None, graded_temperature=0.01,
                    extra_neg_embs=None):
    """
    Graded-relevance loss over a per-query candidate pool (iKAT TREC grades 0-4).
    Gradient flows ONLY through query_embs; cand_embs / extra_neg_embs are frozen docs.
      query_embs     (B, d)        L2-normalized, carries grad
      cand_embs      (B, Cmax, d)  frozen doc embeddings gathered for each query's pool
      cand_grades    (B, Cmax)     int64 grade 0..4, pad = -1
      cand_mask      (B, Cmax)     bool, True = real candidate
      bce_beta       scalar nn.Parameter (learnable BCE bias) or None
      extra_neg_embs (E, d)        optional in-batch / cross-device negatives (treated grade 0)
    """
    B, Cmax, d = cand_embs.shape
    scores = torch.einsum("bd,bcd->bc", query_embs.float(), cand_embs.float())   # (B, Cmax) = <q,d>

    if loss_type == "bce":                              # BiXSE (primary): pointwise graded BCE
        z      = cand_grades.clamp(min=0).float() / 4.0                          # grade/4 in [0,1]
        beta   = bce_beta if bce_beta is not None else 0.0
        logits = bce_logit_scale * scores + beta
        m      = cand_mask
        if extra_neg_embs is not None and extra_neg_embs.numel() > 0:
            ns = query_embs.float() @ extra_neg_embs.float().T                   # (B, E), grade 0
            logits = torch.cat([logits, bce_logit_scale * ns + beta], dim=1)
            z      = torch.cat([z, torch.zeros_like(ns)], dim=1)
            m      = torch.cat([m, torch.ones_like(ns, dtype=torch.bool)], dim=1)
        per_elem = F.binary_cross_entropy_with_logits(logits, z, reduction="none") * m.float()
        return per_elem.sum() / m.float().sum().clamp(min=1.0)                   # mean over valid (q,cand) pairs

    elif loss_type == "graded_infonce":                 # Soft-InfoNCE (ablation): listwise softmax CE
        grades, m = cand_grades, cand_mask
        if extra_neg_embs is not None and extra_neg_embs.numel() > 0:
            ns = query_embs.float() @ extra_neg_embs.float().T
            scores = torch.cat([scores, ns], dim=1)
            grades = torch.cat([grades, torch.zeros(B, ns.size(1), dtype=grades.dtype,
                                                    device=grades.device)], dim=1)
            m      = torch.cat([m, torch.ones_like(ns, dtype=torch.bool)], dim=1)
        gain = torch.where(grades >= 1, (2.0 ** grades.clamp(min=0).float()) - 1.0,
                           torch.zeros_like(scores)) * m.float()                 # gain = 2^g - 1 for positives
        tgt  = gain / gain.sum(dim=1, keepdim=True).clamp(min=1e-9)              # target distribution over positives
        logits = (scores / graded_temperature).masked_fill(~m, float("-inf"))
        logp   = F.log_softmax(logits, dim=1)
        has_pos = (gain.sum(dim=1) > 0).float()                                 # turns with >=1 positive
        loss_per_q = -(tgt * logp.masked_fill(~m, 0.0)).sum(dim=1)
        return (loss_per_q * has_pos).sum() / has_pos.sum().clamp(min=1.0)

    raise ValueError(f"unknown graded loss_type {loss_type!r}")


def eval_ikat_graded_ndcg(encoder, val_examples, doc_matrix, device, pad_token_id,
                          eval_autocast_ctx, batch_size=64, k=3):
    """
    Per-turn POOL-RERANK graded NDCG@k over the iKAT val split (cheap; no corpus FAISS).
    For each val turn: encode its query, score ONLY its own judged-doc pool by <q,d>, and
    compute graded NDCG@k (TREC grades 0-4 as gain) with pytrec_eval. Returns
    (mean_ndcg_over_turns_with_a_positive, n_turns_scored). Rank-0 only; sets eval()/train().
    """
    import pytrec_eval
    was_training = encoder.training
    encoder.eval()
    dm = doc_matrix.to(device)
    qrels, run = {}, {}
    with torch.no_grad(), eval_autocast_ctx:
        for i in range(0, len(val_examples), batch_size):
            chunk = val_examples[i:i + batch_size]
            seqs  = [ex["query_tokens"] for ex in chunk]
            L     = max(len(s) for s in seqs)
            ids   = torch.tensor([[pad_token_id] * (L - len(s)) + s for s in seqs], device=device)   # left-pad
            msk   = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
            q_emb = encoder(ids, msk).float()                                   # (b,1024), already L2-normalized
            for j, ex in enumerate(chunk):
                rows = torch.tensor(ex["cand_rows"], device=device)
                d    = dm[rows].float()                                          # (Cj,1024) frozen, L2-normalized
                sc   = (q_emb[j] @ d.T).cpu().tolist()
                sid  = ex["sample_id"]                                           # UNIQUE key ("{year}_{qid}")
                qrels[sid] = {f"d{t}": int(g) for t, g in enumerate(ex["cand_grades"])}
                run[sid]   = {f"d{t}": float(sc[t]) for t in range(len(sc))}
    if was_training:
        encoder.train()
    ev  = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    res = ev.evaluate(run)
    # only count turns with >=1 positive (a turn of all grade-0 has no attainable NDCG)
    vals = [m[f"ndcg_cut_{k}"] for sid, m in res.items() if any(g >= 1 for g in qrels[sid].values())]
    return (sum(vals) / max(1, len(vals)), len(vals))


def save_model(args, model_output_path, model, query_tokenizer, optimizer, scheduler,
               cur_step, cur_epoch, best_loss, save_optimizer=True, output_subdir=None):
    # save_optimizer=False is used by the step-driven turn-bucket runs for the
    # intermediate (every --save_every_steps) checkpoints: the 2.4 GB AdamW
    # state is only needed at the final checkpoint, and Delta-theta analysis
    # only needs the model weights. Default True keeps the historical behavior
    # of the epoch-mode runs.
    # output_subdir (e.g. "best"): a FIXED dir that we OVERWRITE on every call
    # (used by --ikat_save_best_only to keep only the best-NDCG@3 epoch); when
    # None, the historical per-step "checkpoint-step-N" dir (skipped if it exists).
    if output_subdir is not None:
        output_dir = oj(model_output_path, output_subdir)
        if os.path.isdir(output_dir):
            import shutil
            shutil.rmtree(output_dir)               # overwrite the previous best
    else:
        output_dir = oj(model_output_path, f"checkpoint-step-{cur_step}")
        if os.path.exists(output_dir):
            logger.info(f"Checkpoint {output_dir} already exists, skipping save.")
            return
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    if getattr(args, "use_lora", False):
        # LoRA: merge the adapter into the base so the saved dir is a STANDALONE HF model
        # loadable by AutoModel.from_pretrained (the offline fuse_then_eval / perso-dense-val
        # needs a full model, NOT a PEFT adapter). deepcopy so the live training module is
        # not mutated (merge_and_unload is in-place and strips the adapter).
        import copy
        merged = copy.deepcopy(model_to_save.model).merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        model_to_save.save_pretrained(output_dir)       # saves inner Qwen3 AutoModel (full-FT)
    query_tokenizer.save_pretrained(output_dir)         # saves Qwen2Tokenizer files
    # learnable BCE bias (graded path) is NOT in the HF model -> record it as a sidecar
    # for reproducibility (eval ranks by cosine, so it doesn't need beta).
    _beta = getattr(model_to_save, "bce_beta", None)
    if _beta is not None and getattr(args, "dataset_type", "topiocqa") == "ikat_graded":
        torch.save({"bce_beta": float(_beta.detach().cpu())}, oj(output_dir, "bce_beta.pt"))
    if save_optimizer:
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

    # Fail fast on incompatible precision flags: fp32_master_weights forces sdpa
    # (FA2 needs half-precision weights), so --use_flash_attention would be
    # silently ignored. Hard-error instead of quietly changing the attention
    # kernel (codex review #4).
    if args.fp32_master_weights and args.use_flash_attention:
        raise ValueError(
            "--fp32_master_weights is incompatible with --use_flash_attention "
            "(FlashAttention-2 requires half-precision weights; the fp32-master path "
            "uses sdpa, which is numerically equivalent). Drop --use_flash_attention.")

    # The two fp32-master modes are mutually exclusive (both keep fp32 master params
    # + fp32 Adam moments; they differ only in HOW). bf16_fp32_master is the canonical/
    # faster one (bf16 model + FA2 + a separate fp32 master in the optimizer, like
    # DeepSpeed's bf16_optimizer); fp32_master_weights (load whole model fp32 + sdpa)
    # is kept only as the A/B-verification reference.
    if args.fp32_master_weights and args.bf16_fp32_master:
        raise ValueError(
            "--fp32_master_weights and --bf16_fp32_master are mutually exclusive "
            "(both are fp32-master modes). Use --bf16_fp32_master for production "
            "(recovers FlashAttention-2 + frees VRAM); --fp32_master_weights is the "
            "reference path kept only for A/B numeric verification.")
    if args.bf16_fp32_master and not args.use_bf16:
        raise ValueError(
            "--bf16_fp32_master requires --use_bf16: the model is loaded in bf16 and the "
            "forward runs under autocast(bf16); fp32 precision lives ONLY in the master "
            "copy + Adam moments. Add --use_bf16.")

    # Graded-path (iKAT) argument validation — fail fast (codex review):
    #  - iKAT requires a graded loss; TopiOCQA must NOT use one (a trainable-but-unused
    #    bce_beta would break DDP find_unused_parameters=False).
    #  - grade0 cap 0 with no extra negatives = degenerate (positive-only targets).
    #  - resume is not supported for the graded / LoRA path (cheap runs; re-run from scratch).
    _GRADED = ("bce", "graded_infonce")
    if args.dataset_type == "ikat_graded":
        if args.loss_type not in _GRADED:
            raise ValueError(f"--dataset_type ikat_graded requires --loss_type in {_GRADED} "
                             f"(got {args.loss_type!r}).")
        if args.ikat_manifest_file is None:
            raise ValueError("--dataset_type ikat_graded requires --ikat_manifest_file.")
        if str(args.ikat_grade0_cap) == "0" and args.ikat_extra_negatives == "none":
            raise ValueError("--ikat_grade0_cap 0 with --ikat_extra_negatives none leaves no "
                             "negatives (degenerate all-positive targets). Use a cap > 0 or "
                             "add extra negatives (v2).")
        if args.resume_from_checkpoint is not None or args.resume_from_latest:
            raise NotImplementedError("Resume is not supported for the graded/iKAT path "
                                      "(best-state + bce_beta are not checkpointed; re-run from scratch).")
        if args.ikat_save_best_only and not args.activate_eval_ikat_graded_while_training:
            raise ValueError("--ikat_save_best_only needs --activate_eval_ikat_graded_while_training "
                             "(the val hook is what selects the best epoch to save).")
    elif args.loss_type in _GRADED:
        raise ValueError(f"--loss_type {args.loss_type!r} is only valid with --dataset_type ikat_graded.")
    if args.use_lora and (args.resume_from_checkpoint is not None or args.resume_from_latest):
        raise NotImplementedError("Resume is not supported with --use_lora (the saved ckpt is a "
                                  "merged full model, not an adapter; re-run from scratch).")

    # 0. Frozen doc embeddings
    sample_emb_table = None     # TopiOCQA: {sample_id: {"pos"/"neg"/"oracle": Tensor|None}}
    doc_matrix       = None     # iKAT graded: (N, 1024) frozen base-qwen3 doc-embedding matrix
    docid_to_row     = None
    ikat_anchor_emb  = None     # iKAT anti-collapse: {sample_id: init query emb} for the anchor term
    ikat_global_neg  = None     # iKAT anti-collapse: (N,1024) corpus negative pool
    ikat_kd_emb      = None     # iKAT ConvDR-KD: {sample_id: oracle-rewrite teacher emb (base-qwen3)}
    if args.dataset_type == "ikat_graded":
        # shared frozen base-qwen3 judged-doc embeddings {docid: (1024,) f32} -> a contiguous
        # (N,1024) matrix + docid->row map (built once; candidates index into it by row).
        _dd = torch.load(args.ikat_doc_embedding_file, map_location="cpu")
        _docids = list(_dd.keys())
        docid_to_row = {d: i for i, d in enumerate(_docids)}
        doc_matrix = torch.stack([_dd[d] for d in _docids], dim=0).contiguous().float()  # (N,1024)
        del _dd
        if args.gpu_resident_doc_table:
            doc_matrix = doc_matrix.to(args.device)
        if is_main_process:
            logger.info(f"iKAT graded: doc_matrix {tuple(doc_matrix.shape)} on "
                        f"{'GPU' if args.gpu_resident_doc_table else 'CPU'} ({len(docid_to_row)} docids).")
        if args.ikat_anchor_emb_file:
            ikat_anchor_emb = torch.load(args.ikat_anchor_emb_file, map_location="cpu")   # {sid: (1024,)}
            if is_main_process:
                logger.info(f"iKAT anchor: {len(ikat_anchor_emb)} init query embs, lambda={args.ikat_anchor_weight}.")
        if args.ikat_global_neg_file and args.ikat_global_neg_k > 0:
            ikat_global_neg = torch.load(args.ikat_global_neg_file, map_location="cpu").float()   # (N,1024)
            if args.gpu_resident_doc_table:
                ikat_global_neg = ikat_global_neg.to(args.device)
            if is_main_process:
                logger.info(f"iKAT global-neg pool: {tuple(ikat_global_neg.shape)}, K={args.ikat_global_neg_k}/step.")
        if args.ikat_kd_emb_file:
            ikat_kd_emb = torch.load(args.ikat_kd_emb_file, map_location="cpu")   # {sid: (1024,)} oracle teacher
            if is_main_process:
                logger.info(f"iKAT KD teacher (oracle rewrite): {len(ikat_kd_emb)} embs, lambda={args.ikat_kd_weight}.")
    elif args.pos_neg_embedding_file is not None:
        sample_emb_table = torch.load(args.pos_neg_embedding_file, map_location="cpu")
        if args.gpu_resident_doc_table:
            # Move the whole frozen table onto the GPU once, so the per-step gather
            # (torch.stack([... for sid in sample_ids]).to(device)) becomes a pure
            # on-device op instead of a CPU->GPU copy every step. ~570 MB for
            # TopiOCQA, replicated per rank. Each entry is a dict with "pos"/"neg"/
            # "oracle" tensors, any of which may be None.
            _moved = 0
            for _sid, _entry in sample_emb_table.items():
                if not isinstance(_entry, dict):
                    continue
                for _k in ("pos", "neg", "oracle"):
                    _v = _entry.get(_k)
                    if torch.is_tensor(_v):
                        _entry[_k] = _v.to(args.device, non_blocking=True)
                        _moved += 1
            if is_main_process:
                logger.info(f"gpu_resident_doc_table ON: moved {_moved} frozen "
                            f"embedding tensors onto {args.device}.")

    # 1. Load Qwen3-Embedding model + tokenizer
    query_encoder = QwenQueryEncoder(
        model_path         = args.pretrained_encoder_path,
        use_flash_attention = args.use_flash_attention,
        use_bf16           = args.use_bf16,
        fp32_master_weights = args.fp32_master_weights,
        bf16_fp32_master   = args.bf16_fp32_master,
    ).to(args.device)

    raw_tokenizer  = AutoTokenizer.from_pretrained(args.pretrained_encoder_path,
                                                   trust_remote_code=True)
    # FlashAttention 2 requires left-padding; last real token is always at position -1.
    raw_tokenizer.padding_side = "left"
    query_tokenizer = Qwen3TokenizerWrapper(raw_tokenizer)

    # Graded-BCE learnable bias: trainable ONLY for --loss_type bce. Otherwise frozen +
    # unused → DDP find_unused_parameters=False stays valid and TopiOCQA is unaffected.
    query_encoder.bce_beta.requires_grad_(args.loss_type == "bce")

    if args.use_lora:
        # Parameter-efficient FT: wrap the inner Qwen3 model so only LoRA adapters (+ bce_beta)
        # train. get_optimizer + the bf16_fp32_master master_map both filter requires_grad, so
        # they naturally restrict to the adapter params. enable_input_require_grads is added in
        # the gradient-checkpointing block below (PEFT+GC requirement).
        from peft import LoraConfig, get_peft_model
        _targets = args.lora_target_modules or \
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        query_encoder.model = get_peft_model(query_encoder.model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", target_modules=_targets))
        if is_main_process:
            logger.info("LoRA enabled (r=%d, alpha=%d, dropout=%.2f, targets=%s):",
                        args.lora_r, args.lora_alpha, args.lora_dropout, _targets)
            query_encoder.model.print_trainable_parameters()

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
        if args.use_lora:
            # PEFT + gradient checkpointing: the frozen base produces activations with
            # requires_grad=False, which breaks the checkpointed autograd graph. Force the
            # input embeddings to require grad so the graph stays connected (canonical order:
            # AFTER gradient_checkpointing_enable).
            model_gc.model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (use_reentrant=False).")

    if args.use_compile:
        query_encoder = torch.compile(query_encoder)
        logger.info("torch.compile enabled.")

    # 2. Dataset
    if args.dataset_type == "ikat_graded":
        train_dataset = IKATGradedDataset(args, query_tokenizer, args.ikat_manifest_file,
                                          docid_to_row, args.ikat_split)
    else:
        train_dataset = Topiocqa(args, query_tokenizer, args.training_data_file)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # sample_id -> gold/neg passage id maps, for same-gold false-negative de-dup in the
    # InfoNCE loss (TopiOCQA only; iKAT graded has no pos_pid). Built once; -1 when absent.
    if args.dataset_type == "topiocqa":
        sid_to_pos_pid = {ex['sample_id']: ex.get('pos_pid', -1) for ex in train_dataset.examples}
        sid_to_neg_pid = {ex['sample_id']: ex.get('neg_pid', -1) for ex in train_dataset.examples}
    else:
        sid_to_pos_pid = sid_to_neg_pid = {}

    # iKAT graded val hook: build the held-out (val-split) turns once, with the FULL judged
    # pool (no grade-0 cap), rank-0 only. Drives the per-epoch NDCG@3 selection metric.
    ikat_val_examples = None
    if (args.dataset_type == "ikat_graded" and args.activate_eval_ikat_graded_while_training
            and is_main_process):
        _vf = args.ikat_val_manifest_file or args.ikat_manifest_file
        ikat_val_examples = IKATGradedDataset(args, query_tokenizer, _vf, docid_to_row,
                                              "val", grade0_cap_override="all").examples
        logger.info(f"iKAT graded val hook: {len(ikat_val_examples)} val turns (full pools).")

    # Fake-negative masking is implemented only in the cross-device InfoNCE path;
    # fail loudly rather than silently no-op in the legacy single-GPU branch.
    if (args.mask_fake_negative or args.dedup_same_gold) and not (args.cross_device_negatives and args.n_gpu > 1):
        raise ValueError("--mask_fake_negative / --dedup_same_gold require "
                         "--cross_device_negatives with n_gpu>1 (masking lives in the "
                         "cross-device branch of cal_ranking_loss).")

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

    # ── Step-driven mode (--total_train_steps > 0) — used by the turn-bucket
    # runs, whose datasets (~3k samples ≈ 6 steps/pass at batch 480) are far
    # smaller than one conventional epoch. We repurpose the existing epoch
    # loop: one "epoch" = --save_every_steps optimizer steps drawn from a
    # cycling iterator over the full (small) dataset, so the end-of-epoch
    # save_model() call produces a checkpoint every --save_every_steps steps
    # (checkpoint-step-{47, 94, ..., 470} with the defaults).
    if args.total_train_steps > 0:
        if args.curriculum_type != "none":
            raise ValueError("--total_train_steps only supports --curriculum_type none "
                             "(the turn-bucket runs are single-difficulty by construction).")
        steps_per_epoch       = args.save_every_steps
        micro_steps_per_epoch = steps_per_epoch * args.gradient_accumulation_steps
        args.num_train_epochs = (args.total_train_steps + args.save_every_steps - 1) \
                                // args.save_every_steps
        total_training_steps  = args.num_train_epochs * steps_per_epoch
        if is_main_process:
            logger.info(f"Step-driven mode: {total_training_steps} optimizer steps total, "
                        f"checkpoint every {steps_per_epoch} steps "
                        f"({args.num_train_epochs} save points), cycling over "
                        f"{len(train_dataset)} samples.")

    if is_main_process and args.curriculum_type != "none":
        log_pacing_schedule(args.num_train_epochs, steps_per_epoch, curriculum_steps,
                            args.curriculum_c0, args.pacing_function)

    # Collate function: dynamic (batch-max, hard-cap 512) + left-padding (required by FlashAttn2)
    collate_fn_kwargs = dict(pad_token_id=query_tokenizer.pad_token_id,
                             dynamic_padding=True, left_padding=True)

    # DataLoader worker/prefetch kwargs (shared by the nosched loader below and the
    # per-epoch curriculum loader). prefetch_factor/persistent_workers are only valid
    # when num_workers > 0, so build them conditionally.
    loader_perf_kwargs = dict(num_workers=args.dataloader_num_workers)
    if args.dataloader_num_workers > 0:
        loader_perf_kwargs["prefetch_factor"]    = args.dataloader_prefetch_factor
        loader_perf_kwargs["persistent_workers"] = True
    if is_main_process:
        logger.info(f"DataLoader: num_workers={args.dataloader_num_workers}"
                    + (f", prefetch_factor={args.dataloader_prefetch_factor}, "
                       f"persistent_workers=True" if args.dataloader_num_workers > 0 else ""))

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
            **loader_perf_kwargs,
        )

    # 3. Optimizer
    # master_map: in --bf16_fp32_master mode, a list of (name, model_param_bf16,
    # master_param_fp32). The optimizer is built over the FP32 master copies; each
    # step copies the DDP-averaged bf16 grads into the master grads, clips+steps the
    # fp32 AdamW, then writes the updated fp32 master back into the bf16 model. None
    # in the legacy / fp32_master_weights paths (optimizer is over the model params).
    master_map = None
    if args.bf16_fp32_master:
        _base = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        master_map = []
        for _n, _p in _base.named_parameters():
            if not _p.requires_grad:
                continue
            _m = _p.detach().clone().float()      # fp32 master copy of this param
            _m.requires_grad_(True)
            master_map.append((_n, _p, _m))
        # Same no_decay grouping as get_optimizer(), but over the master tensors.
        _no_decay = ['bias', 'LayerNorm.weight']
        _groups = [
            {'params': [_m for _n, _, _m in master_map if not any(nd in _n for nd in _no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [_m for _n, _, _m in master_map if any(nd in _n for nd in _no_decay)],
             'weight_decay': 0.0},
        ]
        _betas = (args.adam_beta1, args.adam_beta2)
        optimizer = torch.optim.AdamW(_groups, lr=args.learning_rate, betas=_betas,
                                      eps=args.adam_epsilon,
                                      fused=bool(args.use_fused_optimizer))
        if is_main_process:
            logger.info(f"bf16_fp32_master ON: {len(master_map)} fp32 master tensors; "
                        f"AdamW(fused={bool(args.use_fused_optimizer)}, betas={_betas}) "
                        f"over master params; bf16 model + "
                        f"{'FA2' if args.use_flash_attention else 'sdpa'}.")
    else:
        optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay,
                                  fused=args.use_fused_optimizer)

    # 4. LR scheduler
    if args.no_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        if is_main_process:
            logger.info("LR scheduler: constant (no warmup, no decay).")
    else:
        num_warmup = int(args.warmup_ratio * total_training_steps)
        if args.lr_scheduler_type == "cosine":
            # Matches the official Qwen3-Embedding ms-swift default (cosine).
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, total_training_steps)
            if is_main_process:
                logger.info(f"LR scheduler: linear warmup ({num_warmup} steps) + cosine decay.")
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_training_steps)
            if is_main_process:
                logger.info(f"LR scheduler: linear warmup ({num_warmup} steps) + linear decay.")

    # 5. In-memory eval cache (corpus loaded once; per-epoch eval only re-encodes queries)
    beir_eval_cache    = None
    topiocqa_faiss_idx = None
    topiocqa_doc_ids   = None
    qrecc_faiss_idx    = None
    qrecc_doc_ids      = None
    # Per-task official Qwen3 BEIR instructions; passed to eval_beir_from_cache so
    # in-training BEIR/MSMARCO eval matches the zero-shot baseline's instruction setup.
    _beir_instruction_map = build_qwen_instruction_map()

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

        if args.activate_eval_qrecc_while_training:
            logger.info(f"Loading QReCC corpus into FAISS (embed_dim={args.embed_dim})...")
            logger.warning(
                "QReCC full flat FAISS can be very large. "
                "For Qwen3 1024-dim, 54.6M docs require about 208GiB just for vectors, "
                "before doc ids and FAISS/Python overhead."
            )
            qrecc_faiss_idx, qrecc_doc_ids = load_corpus_into_faiss(
                embedding_dir = args.qrecc_embedding_dir,
                embed_dim     = args.embed_dim,
                use_gpu       = False,
            )
            logger.info(f"QReCC FAISS ready: {qrecc_faiss_idx.ntotal} docs.")

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
        model_core = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        # Load the checkpoint weights IN-PLACE via load_state_dict rather than
        # REPLACING model_core.model with a fresh AutoModel. The optimizer param
        # groups and the DDP reducer hold references to the EXISTING parameter
        # tensors; swapping the module out would orphan them (gradients computed
        # on the new params, optimizer/DDP still bound to the old ones), silently
        # breaking resume. load_state_dict overwrites values while preserving
        # tensor identity, and casts to the existing param dtype (bf16 or fp32),
        # so the attention impl / dtype set at construction are kept untouched
        # (codex review #1).
        _src = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        model_core.model.load_state_dict(_src.state_dict())
        del _src
        model_core.model.config.use_cache = False  # consistent with initial load
        if args.bf16_fp32_master and master_map is not None:
            # The fp32 master was cloned from the PRE-resume (pretrained-init) weights;
            # re-sync it from the just-loaded checkpoint weights so the optimizer's
            # master matches the resumed model. (Rebuilt from bf16, so the sub-bf16-ULP
            # bits of the original fp32 master are lost — a negligible perturbation; the
            # Adam moments restored below from optimizer.pt are exact. Resume is best-
            # effort here since runs default to --save_optimizer OFF.)
            with torch.no_grad():
                for _n, _p, _m in master_map:
                    _m.data.copy_(_p.data.float())
            logger.info("bf16_fp32_master: re-synced fp32 master from resumed bf16 weights.")
        if not os.path.exists(oj(ckpt, "optimizer.pt")):
            raise FileNotFoundError(
                f"Resume from {ckpt} requested but optimizer.pt is missing. Optimizer/"
                f"scheduler state is only written when training ran with --save_optimizer; "
                f"a model-only checkpoint cannot resume optimizer state (codex review).")
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

    # ── Gradient-statistics recorder (--record_grad_stats) ───────────────────
    # R1: per-step per-tensor L2 grad norms (~310 floats/step, negligible).
    # R2: whole-run per-scalar signed gradient sum (sum_g) and sum of squares
    #     (sum_g2), fp32 accumulators. sum_g captures the NET update direction,
    #     sum_g2 the total gradient energy (diagonal-Fisher proxy, cf. Sung et
    #     al. 2021 FISH mask); |sum_g|/sqrt(T*sum_g2) in [0,1] measures sign
    #     coherence (1 = consistently same-direction drift, ~0 = oscillating /
    #     cancelling updates — cf. the ~8% cancelled-out parameters reported in
    #     arXiv 2505.11711 Fig. 2). Grads are recorded AFTER clip_grad_norm_,
    #     i.e. the effective signal that reaches the optimizer. Under DDP the
    #     post-all-reduce grads are identical on every rank, so rank 0 records.
    grad_stats = None
    _model_for_stats = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
    if args.record_grad_stats and is_main_process:
        _gs_device = args.device if args.grad_stats_device == "gpu" else torch.device("cpu")
        grad_stats = {
            "names":  [n for n, p in _model_for_stats.named_parameters() if p.requires_grad],
            "per_step_norms": [],
            "sum_g":  {n: torch.zeros(p.shape, dtype=torch.float32, device=_gs_device)
                       for n, p in _model_for_stats.named_parameters() if p.requires_grad},
            "sum_g2": {n: torch.zeros(p.shape, dtype=torch.float32, device=_gs_device)
                       for n, p in _model_for_stats.named_parameters() if p.requires_grad},
            "n_steps": 0,
        }
        logger.info(f"record_grad_stats ON: {len(grad_stats['names'])} parameter tensors, "
                    f"accumulators on {_gs_device}.")

    # In-training eval must run the forward under the SAME bf16 autocast as the
    # offline eval scripts (which load checkpoints in bf16). Under
    # --fp32_master_weights the model params are fp32, so without autocast the
    # in-training eval would run a full-fp32 forward and report metrics from a
    # different numeric path than deployment/offline (codex review #3). A reusable
    # autocast instance (torch.autocast objects can be re-entered) wraps all three
    # eval blocks below.
    eval_autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if (args.use_bf16 or args.fp32_master_weights)
                         else contextlib.nullcontext())

    # --- iKAT graded val hook: zero-shot NDCG@3 floor (before any training) + best tracking ---
    best_ikat_ndcg3, best_ikat_epoch = -1.0, -1
    best_saved_ndcg3 = -1.0   # best TRAINED epoch (for --ikat_save_best_only; saves even if < floor)
    if ikat_val_examples is not None:
        _enc = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        _floor, _n = eval_ikat_graded_ndcg(_enc, ikat_val_examples, doc_matrix, args.device,
                                           query_tokenizer.pad_token_id, eval_autocast_ctx,
                                           batch_size=args.eval_batch_size)
        best_ikat_ndcg3, best_ikat_epoch = _floor, -1      # zero-shot = the collapse floor
        logger.info(f"[ikat val] zero-shot (epoch -1) NDCG@3={_floor:.4f} over {_n} turns (collapse floor).")
        if args.save_to_wandb:
            wandb.log({"eval/ikat_ndcg@3": _floor}, step=cur_step)
            wandb.run.summary["floor/ikat_ndcg@3"] = _floor

    # --- Optional per-phase step profiler (opt-in via --profile_timing) ---
    # Accumulates wall-clock for the data wait, forward+backward, and optimizer
    # phases, then logs rolling averages (+ GPU util/peak-mem) every
    # --profile_every_steps optimizer steps. Used to pick the optimal
    # num_workers / gpu_resident / bf16_fp32_master config. Adds cuda.syncs, so
    # only enabled when requested. Rank-0 only.
    _prof = None
    if args.profile_timing and is_main_process:
        _prof = {"data": 0.0, "fb": 0.0, "opt": 0.0, "step": 0.0,
                 "nmicro": 0, "nopt": 0}

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
                                        pin_memory = True,
                                        **loader_perf_kwargs)
        else:
            current_loader = train_loader

        if args.curriculum_type != "none" or args.total_train_steps > 0:
            # Curriculum subsets AND step-driven (turn-bucket) runs both need
            # the cycling iterator: the active dataset is smaller than the
            # number of micro-steps the "epoch" must yield.
            batch_iter = make_cycling_iter(current_loader, micro_steps_per_epoch, epoch)
        else:
            if args.n_gpu > 1:
                current_loader.sampler.set_epoch(epoch)
            batch_iter = iter(current_loader)

        # fp32_master_weights IMPLIES bf16 autocast for the forward/loss: the whole
        # point is "fp32 master + bf16 matmuls" (the official DeepSpeed-ZeRO bf16
        # numerics). Without this, `--fp32_master_weights` alone (no `--use_bf16`)
        # would silently run a full-fp32 forward — slower, more memory, and NOT the
        # intended numeric path (codex review #2).
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if (args.use_bf16 or args.fp32_master_weights)
                        else contextlib.nullcontext())

        optimizer.zero_grad(set_to_none=True)
        n_batches = micro_steps_per_epoch
        epoch_step_start = epoch * steps_per_epoch
        if _prof is not None:
            _t_prev_end = time.perf_counter()   # marks end of previous work (for data_time)

        for micro_step, batch in enumerate(tqdm(batch_iter, total=micro_steps_per_epoch, desc="Step")):
            is_last_in_accum = (
                (micro_step + 1) % args.gradient_accumulation_steps == 0
                or (micro_step + 1) == n_batches
            )

            if _prof is not None:
                # data_time = wait for this batch (iterator yield + host-side gap).
                _t_data = time.perf_counter()
                _prof["data"] += _t_data - _t_prev_end
                if (micro_step % args.gradient_accumulation_steps) == 0:
                    _t_step_start = _t_data        # start of this optimizer-step window
                _t_fb0 = _t_data                   # forward+backward starts now

            sample_ids         = batch["sample_ids"]
            complex_query      = batch["complex_query"].to(args.device, non_blocking=True)
            complex_query_mask = batch["complex_query_mask"].to(args.device, non_blocking=True)

            ddp_sync_ctx = (contextlib.nullcontext()
                            if args.n_gpu <= 1 or is_last_in_accum
                            else query_encoder.no_sync())

            with ddp_sync_ctx:
                with autocast_ctx:
                    complex_query_embs = query_encoder(complex_query, complex_query_mask)

                if args.dataset_type == "ikat_graded":
                    # ── Graded iKAT path: rank each query's OWN judged-doc pool by <q,d> ──
                    # cand_embs gathered by integer row from the frozen doc_matrix (grad-free);
                    # gradient flows only through complex_query_embs.
                    # gather frozen candidate embeddings by row; index on the matrix's device, then
                    # (CPU-resident case) transfer only the gathered rows -- NOT the whole 192 MB
                    # matrix every step (codex review).
                    if args.gpu_resident_doc_table:
                        cand_embs = doc_matrix[batch["cand_rows"].to(args.device)]   # (B, Cmax, 1024) frozen
                    else:
                        cand_embs = doc_matrix[batch["cand_rows"]].to(args.device, non_blocking=True)
                    cand_grades = batch["cand_grades"].to(args.device)
                    cand_mask   = batch["cand_mask"].to(args.device)
                    # extra grade-0 negatives beyond each query's own judged pool:
                    #   global   = K docs sampled from the corpus pool -> FULL-corpus calibration
                    #              (the fix for "query rotates off the base-qwen3 doc manifold");
                    #   in_batch = all judged docs in the batch (cheap hard negs; may include this
                    #              query's own relevant docs -> mild false-negative risk).
                    extra_neg = None
                    if ikat_global_neg is not None:
                        gi = torch.randint(0, ikat_global_neg.shape[0], (args.ikat_global_neg_k,),
                                           device=ikat_global_neg.device)
                        extra_neg = ikat_global_neg[gi].to(args.device)                  # (K, 1024) frozen
                    elif args.ikat_extra_negatives == "in_batch":
                        extra_neg = cand_embs[cand_mask]                                 # (sum_valid, 1024)
                    elif args.ikat_extra_negatives == "cross_device":
                        raise NotImplementedError("cross_device extra negs not wired; use --ikat_global_neg_file.")
                    _base = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
                    with autocast_ctx:
                        ranking_loss = cal_graded_loss(
                            complex_query_embs, cand_embs, cand_grades, cand_mask,
                            loss_type=args.loss_type, bce_logit_scale=args.bce_logit_scale,
                            bce_beta=(_base.bce_beta if args.loss_type == "bce" else None),
                            graded_temperature=args.graded_temperature, extra_neg_embs=extra_neg)
                        loss = args.ikat_bixse_weight * ranking_loss   # bixse_weight=0 -> pure KD (ConvDR-style KD-dominant)
                        # anchor regularization: keep the trained query near init (cos-to-init),
                        # so it stays compatible with the frozen base-qwen3 doc index.
                        if args.ikat_anchor_weight > 0 and ikat_anchor_emb is not None:
                            qa = torch.stack([ikat_anchor_emb[sid] for sid in sample_ids]).to(args.device).float()
                            anchor_loss = (1.0 - (complex_query_embs.float() * qa).sum(-1)).mean()
                            loss = loss + args.ikat_anchor_weight * anchor_loss
                        # ConvDR-style KD: pull the query toward the oracle-rewrite teacher emb (base-qwen3,
                        # web-search instruction). The oracle fuses the relevant persona -> teaches
                        # personalization, AND stays on the base-qwen3 doc manifold -> a *directed* anti-collapse.
                        if args.ikat_kd_weight > 0 and ikat_kd_emb is not None:
                            qt = torch.stack([ikat_kd_emb[sid] for sid in sample_ids]).to(args.device).float()
                            kd_loss = (1.0 - (complex_query_embs.float() * qt).sum(-1)).mean()   # cos KD
                            loss = loss + args.ikat_kd_weight * kd_loss
                        # listwise KD (round4 C2): match student's score distribution over the candidate
                        # pool to the teacher's -> distill the teacher's RELATIVE ranking, not one point.
                        # Reuses the same frozen cand_embs/cand_mask as BiXSE (the eval metric is ranking).
                        if args.ikat_kd_list_weight > 0 and ikat_kd_emb is not None:
                            qtl  = torch.stack([ikat_kd_emb[sid] for sid in sample_ids]).to(args.device).float()  # (B,1024)
                            t_sc = (qtl.unsqueeze(1) * cand_embs.float()).sum(-1)                                  # (B,Cmax) teacher-pool sims
                            s_sc = (complex_query_embs.float().unsqueeze(1) * cand_embs.float()).sum(-1)           # (B,Cmax) student-pool sims
                            neg_inf = -1e4   # large-but-finite: softmax->0 at masked slots, avoids -inf in log_softmax
                            t_sc = t_sc.masked_fill(~cand_mask, neg_inf)
                            s_sc = s_sc.masked_fill(~cand_mask, neg_inf)
                            T = args.ikat_kd_list_temp
                            # reduction='none' + remask: masked slots have teacher prob ~0, whose 0*log0
                            # would be nan under batchmean -> zero them out, then batch-mean over queries.
                            kd_el = F.kl_div(F.log_softmax(s_sc / T, dim=-1),
                                             F.softmax(t_sc / T, dim=-1), reduction="none")   # (B,Cmax)
                            kd_list = kd_el.masked_fill(~cand_mask, 0.0).sum(dim=-1).mean()
                            loss = loss + args.ikat_kd_list_weight * kd_list
                else:
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
                        if args.cross_device_negatives and args.n_gpu > 1:
                            # all_gather needs every rank's micro-batch to have the
                            # SAME row count (DistributedSampler + DataLoader both use
                            # drop_last=True, so this holds); fail loudly if a future
                            # edit relaxes drop_last instead of silently NCCL-hanging.
                            assert complex_query_embs.size(0) == args.per_gpu_train_batch_size, (
                                f"cross_device_negatives requires fixed per-rank batch "
                                f"(got {complex_query_embs.size(0)} vs {args.per_gpu_train_batch_size}); "
                                f"ensure drop_last=True on sampler+loader.")
                        # passage-id tensors for same-gold de-dup masking (only built when needed)
                        if args.dedup_same_gold:
                            pos_pids = torch.tensor([sid_to_pos_pid[sid] for sid in sample_ids],
                                                    device=args.device, dtype=torch.long)
                            neg_pids = (torch.tensor([sid_to_neg_pid[sid] for sid in sample_ids],
                                                     device=args.device, dtype=torch.long)
                                        if neg_doc_embs is not None else None)
                        else:
                            pos_pids = neg_pids = None
                        ranking_loss = cal_ranking_loss(
                            complex_query_embs, pos_doc_embs, neg_doc_embs,
                            temperature=args.infonce_temperature,
                            cross_device=args.cross_device_negatives,
                            world_size=args.n_gpu,
                            rank=(dist.get_rank() if args.n_gpu > 1 else 0),
                            pos_pids=pos_pids, neg_pids=neg_pids,
                            mask_fake_negative=args.mask_fake_negative,
                            fake_neg_margin=args.fake_neg_margin,
                            dedup_same_gold=args.dedup_same_gold,
                        )
                        if "kd" in args.loss_type:
                            kd_loss = cal_kd_loss(complex_query_embs, oracle_utt_embs)
                        loss = ranking_loss + kd_loss

                (loss / args.gradient_accumulation_steps).backward()

            if _prof is not None:
                torch.cuda.synchronize(args.device)
                _prof["fb"] += time.perf_counter() - _t_fb0
                _prof["nmicro"] += 1
                _t_opt0 = time.perf_counter()        # optimizer phase starts now

            if is_last_in_accum:
                if args.bf16_fp32_master:
                    # Copy the DDP-averaged bf16 model grads into the fp32 master grads;
                    # clip + step + write-back then run entirely in fp32 (= DeepSpeed's
                    # bf16_optimizer). For grad_accum > 1 the per-micro grads were summed
                    # in bf16 (matching DeepSpeed's bf16 grad buffer); our runs use accum=1.
                    with torch.no_grad():
                        for _n, _p, _m in master_map:
                            if _p.grad is None:
                                _m.grad = None
                            elif _m.grad is None:
                                _m.grad = _p.grad.detach().float()     # fp32 <- bf16
                            else:
                                _m.grad.copy_(_p.grad.detach())
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [_m for _, _, _m in master_map], args.max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)

                if grad_stats is not None:
                    # In bf16_fp32_master mode the meaningful (post-clip, fp32) grads
                    # live on the master copies, not the bf16 model; their names match.
                    if args.bf16_fp32_master:
                        _stats_params = [(_n, _m) for _n, _, _m in master_map]
                    else:
                        _stats_params = [(n, p) for n, p in _model_for_stats.named_parameters()
                                         if p.requires_grad]
                    with torch.no_grad():
                        # R1: per-tensor grad norms (post-clip). Guard against
                        # requires_grad params that never receive a gradient
                        # (p.grad is None) so the per-step row keeps a fixed
                        # length; record 0.0 for them.
                        norms = torch.stack([
                            (p.grad.detach().norm().float().cpu() if p.grad is not None
                             else torch.zeros((), dtype=torch.float32))
                            for _, p in _stats_params
                        ])
                        grad_stats["per_step_norms"].append(norms)
                        # R2: per-scalar accumulators (skip None-grad params).
                        for n, p in _stats_params:
                            if p.grad is None:
                                continue
                            g = p.grad.detach().to(grad_stats["sum_g"][n].device,
                                                   dtype=torch.float32)
                            grad_stats["sum_g"][n].add_(g)
                            grad_stats["sum_g2"][n].addcmul_(g, g)
                        grad_stats["n_steps"] += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if args.bf16_fp32_master:
                    # Write the updated fp32 master back into the bf16 model, then clear
                    # the model's bf16 grads (the optimizer.zero_grad above only cleared
                    # the master grads — the optimizer owns the master params, not the model).
                    with torch.no_grad():
                        for _n, _p, _m in master_map:
                            _p.data.copy_(_m.data)             # bf16 <- fp32
                    (query_encoder.module if hasattr(query_encoder, "module")
                     else query_encoder).zero_grad(set_to_none=True)

                if _prof is not None:
                    torch.cuda.synchronize(args.device)
                    _now = time.perf_counter()
                    _prof["opt"]  += _now - _t_opt0
                    _prof["step"] += _now - _t_step_start
                    _prof["nopt"] += 1

                loss_val  = loss.item()
                optimizer_step_in_epoch = (micro_step + 1) // args.gradient_accumulation_steps
                cur_step  = epoch_step_start + min(optimizer_step_in_epoch, steps_per_epoch)

                if _prof is not None and _prof["nopt"] >= args.profile_every_steps:
                    _nmi = max(_prof["nmicro"], 1)
                    _nop = max(_prof["nopt"], 1)
                    _alloc = torch.cuda.max_memory_allocated(args.device) / 1e9
                    _resv  = torch.cuda.max_memory_reserved(args.device) / 1e9
                    try:
                        _util = torch.cuda.utilization(args.device)
                    except Exception:
                        _util = -1
                    logger.info(
                        "[profile] step=%d  data=%.1fms  fwd_bwd=%.1fms/micro  opt=%.1fms  "
                        "total=%.1fms/optstep | GPU util=%s%%  peak_alloc=%.1fGB  peak_resv=%.1fGB"
                        % (cur_step, 1000 * _prof["data"] / _nmi, 1000 * _prof["fb"] / _nmi,
                           1000 * _prof["opt"] / _nop, 1000 * _prof["step"] / _nop,
                           _util, _alloc, _resv))
                    torch.cuda.reset_peak_memory_stats(args.device)
                    for _kk in ("data", "fb", "opt", "step"):
                        _prof[_kk] = 0.0
                    _prof["nmicro"] = 0
                    _prof["nopt"] = 0

                if is_main_process and cur_step % args.log_print_steps == 0:
                    logger.info(f"Epoch={epoch} Step={cur_step}/{total_training_steps} Loss={loss_val:.7f}")
                    if args.save_to_wandb:
                        wandb.log({"train/loss": loss_val, "train/ranking_loss": ranking_loss.item(),
                                   "train/lr": scheduler.get_last_lr()[0],
                                   "train/grad_norm": grad_norm.item(), "epoch": epoch},
                                  step=cur_step)

            if _prof is not None:
                _t_prev_end = time.perf_counter()   # end of this micro-step -> next data_time

        # End of epoch
        cur_step = (epoch + 1) * steps_per_epoch
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        if is_main_process:
            is_last_epoch = (epoch == args.num_train_epochs - 1)

            if args.activate_eval_while_training and beir_eval_cache is not None:
                with torch.no_grad(), eval_autocast_ctx:
                    # BEIR: force keep_faiss_on_gpu=False so each dataset's index is
                    # freed after search. Qwen3 (1024-dim) + TopiOCQA cache + BEIR cache
                    # together overflow 46 GB; loading one at a time peaks at ~40 GB.
                    # Pass the per-task Qwen3 BEIR instruction map so MSMARCO and
                    # BEIR datasets are evaluated the same way as the zero-shot
                    # baseline (apples-to-apples).
                    # BEIR queries must use the RAW tokenizer, NOT the
                    # Qwen3TokenizerWrapper: the raw tokenizer's post-processor
                    # appends the official trailing <|endoftext|> (151643) that the
                    # BEIR corpus embeddings were built with, whereas the wrapper
                    # brackets the query with <|im_end|> (151645) boundary tokens and
                    # shifts the last-token pool onto a position the instruct-template
                    # models never train. Smoke 2026-06-09 (instruct3_qwen_nosched
                    # step-1880, same instruction map): wrapper=0.1586 vs raw=0.3363.
                    # This wrapper artifact (not the instruction map alone) is what
                    # made instruct2/instruct3 in-training MSMARCO read ~0.13-0.17;
                    # the map fix (ca2b960) only accounted for 0.13->0.16.
                    metric_numbers = eval_beir_from_cache(
                        beir_cache = beir_eval_cache, query_encoder = query_encoder,
                        tokenizer = raw_tokenizer, device = args.device,
                        eval_batch_size = args.eval_batch_size,
                        use_gpu_faiss = args.use_gpu_faiss, keep_faiss_on_gpu = False,
                        gpu_index_cache = _gpu_faiss_cache, full_eval = is_last_epoch,
                        query_instruction_map = _beir_instruction_map,
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
                with torch.no_grad(), eval_autocast_ctx:
                    topiocqa_metrics = eval_conv_search(
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
                        dataset_tag = "topiocqa",
                        conv_instruction = args.conv_instruction,
                        # MUST match the training-side template (data.py reads
                        # args.template_version). Omitting this left eval on the
                        # "v1" default while instruct3 trained on v3 — a train/eval
                        # format mismatch that depressed every in-training TopiOCQA
                        # reading by ~7 NDCG@10 points (in-training 0.3940 vs
                        # offline 0.4669 on the same instruct3_nosched final ckpt,
                        # 2026-06-09).
                        template_version = args.template_version,
                    )
                if args.save_to_wandb:
                    if is_last_epoch:
                        for mk, mv in topiocqa_metrics.items():
                            wandb.run.summary[f"final/topiocqa/{mk}"] = mv
                    else:
                        wandb.log({f"eval/topiocqa_{k}": v for k, v in topiocqa_metrics.items()},
                                  step=cur_step)
                gc.collect(); torch.cuda.empty_cache()

            if ikat_val_examples is not None:
                # graded NDCG@3 over the held-out val turns (per-turn pool rerank). Cheap selection
                # signal; final reporting is the offline full-ClueWeb fuse_then_eval (perso_dense_val).
                _enc = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
                _ndcg3, _n = eval_ikat_graded_ndcg(_enc, ikat_val_examples, doc_matrix, args.device,
                                                   query_tokenizer.pad_token_id, eval_autocast_ctx,
                                                   batch_size=args.eval_batch_size)
                _is_new_best = _ndcg3 > best_saved_ndcg3   # best trained epoch -> save to <out>/best
                if _is_new_best:
                    best_saved_ndcg3 = _ndcg3
                if _ndcg3 > best_ikat_ndcg3:
                    best_ikat_ndcg3, best_ikat_epoch = _ndcg3, epoch
                logger.info(f"[ikat val] epoch={epoch} NDCG@3={_ndcg3:.4f} "
                            f"(best {best_ikat_ndcg3:.4f} @ep{best_ikat_epoch})")
                if args.save_to_wandb:
                    wandb.log({"eval/ikat_ndcg@3": _ndcg3}, step=cur_step)
                    wandb.run.summary["best/ikat_ndcg@3"] = best_ikat_ndcg3
                    wandb.run.summary["best/ikat_epoch"]  = best_ikat_epoch
                gc.collect(); torch.cuda.empty_cache()

            if args.activate_eval_qrecc_while_training and qrecc_faiss_idx is not None:
                if args.use_gpu_faiss:
                    logger.warning(
                        "QReCC full flat index is large. Make sure the current GPUs have enough "
                        "memory before using --use_gpu_faiss."
                    )
                with torch.no_grad(), eval_autocast_ctx:
                    qrecc_metrics = eval_conv_search(
                        query_encoder = query_encoder, tokenizer = query_tokenizer,
                        test_data_file = args.qrecc_valid_file, qrel_file = args.qrecc_qrel_file,
                        faiss_index = qrecc_faiss_idx, doc_ids = qrecc_doc_ids,
                        device = args.device, eval_batch_size = args.eval_batch_size,
                        max_query_length = args.max_query_length,
                        max_response_length = args.max_response_length,
                        max_concat_length = args.max_concat_length,
                        use_gpu_faiss = args.use_gpu_faiss, keep_faiss_on_gpu = args.keep_faiss_on_gpu,
                        gpu_index_cache = _gpu_faiss_cache, full_eval = is_last_epoch,
                        left_padding = True,
                        dataset_tag = "qrecc",
                        conv_instruction = args.conv_instruction,
                        # Same train/eval template alignment as the TopiOCQA call
                        # above — see the comment there.
                        template_version = args.template_version,
                    )
                if args.save_to_wandb:
                    if is_last_epoch:
                        for mk, mv in qrecc_metrics.items():
                            wandb.run.summary[f"final/qrecc/{mk}"] = mv
                    else:
                        wandb.log({f"eval/qrecc_{k}": v for k, v in qrecc_metrics.items()},
                                  step=cur_step)
                gc.collect(); torch.cuda.empty_cache()

            if best_loss > loss_val:
                best_loss = loss_val
            # Save the heavy AdamW state (~4.8 GB fp32 / ~2.4 GB bf16) ONLY at the
            # FINAL checkpoint, in BOTH epoch and step-driven mode; intermediate
            # checkpoints are model-only. The Delta-theta / per-epoch-eval
            # analyses need just the model weights, and a 20-epoch fp32 run that
            # saved the optimizer every epoch would waste ~96 GB and ~40 min of
            # NFS write time per run (the old `(total_train_steps==0) or
            # is_last_epoch` made epoch mode save it every epoch). We do not
            # resume these runs (long SLURM allocation); if mid-run resume is
            # ever needed, save the optimizer on a coarse epoch cadence instead.
            # --save_optimizer (default OFF) gates this entirely: training is cheap
            # and we don't resume, so by default NO optimizer.pt is ever written.
            if getattr(args, "ikat_save_best_only", False) and args.dataset_type == "ikat_graded":
                # sweep mode: keep ONLY the best-NDCG@3 epoch in <out>/best (overwrite on
                # improvement). 27 runs x ~12 epochs x 2.4 GB would be ~800 GB; this is ~65 GB.
                if _is_new_best:
                    save_model(args, args.model_output_path, query_encoder, query_tokenizer,
                               optimizer, scheduler, cur_step, epoch, best_loss,
                               save_optimizer=False, output_subdir="best")
                    logger.info(f"[ikat val] new best NDCG@3 -> saved <out>/best (epoch {epoch})")
            else:
                _save_opt = args.save_optimizer and is_last_epoch
                save_model(args, args.model_output_path, query_encoder, query_tokenizer,
                           optimizer, scheduler, cur_step, epoch, best_loss,
                           save_optimizer=_save_opt)
            if args.save_to_wandb:
                wandb.run.summary["best_loss"] = best_loss

            # Milestone dumps of the R2 accumulators (step-driven runs): after
            # the 1st, middle and last save points (steps 47/235/470 with the
            # defaults). Stored in bf16 to halve disk (1.19 GB per buffer).
            if grad_stats is not None:
                # Dump the R2 accumulators (sum_g / sum_g2) at the first/middle/
                # last save point. This must fire in BOTH epoch mode (e.g. the
                # nosched full runs: 20 epochs -> milestones {0,9,19} = steps
                # 94/940/1880) and step-driven mode (the buckets), so the gate is
                # NOT restricted to --total_train_steps>0 anymore — otherwise the
                # whole-run gradient direction/coherence data is silently never
                # saved for epoch-mode runs.
                milestone_epochs = {0, max(0, args.num_train_epochs // 2 - 1),
                                    args.num_train_epochs - 1}
                if epoch in milestone_epochs:
                    gs_dir = oj(args.model_output_path, f"grad_stats-step-{cur_step}")
                    os.makedirs(gs_dir, exist_ok=True)
                    torch.save({n: t.to(torch.bfloat16).cpu()
                                for n, t in grad_stats["sum_g"].items()},
                               oj(gs_dir, "sum_g.pt"))
                    torch.save({n: t.to(torch.bfloat16).cpu()
                                for n, t in grad_stats["sum_g2"].items()},
                               oj(gs_dir, "sum_g2.pt"))
                    torch.save({"n_steps": grad_stats["n_steps"], "step": cur_step},
                               oj(gs_dir, "meta.pt"))
                    logger.info(f"Dumped grad-stat accumulators at {gs_dir} "
                                f"(n_steps={grad_stats['n_steps']}).")

        if args.n_gpu > 1:
            dist.barrier()

    if grad_stats is not None:
        # R1 dump: (n_steps, n_tensors) matrix of per-step per-tensor grad
        # norms + the tensor-name list.
        norms_path = oj(args.model_output_path, "grad_norms_per_step.npz")
        np.savez(norms_path,
                 norms=torch.stack(grad_stats["per_step_norms"]).numpy(),
                 names=np.array(grad_stats["names"]))
        logger.info(f"Saved per-step grad norms ({grad_stats['n_steps']} steps) "
                    f"to {norms_path}")

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
    parser.add_argument("--loss_type", type=str, default="ranking",
                        help="ranking / kd (TopiOCQA) | bce (BiXSE graded) | graded_infonce (Soft-InfoNCE) for ikat_graded.")
    parser.add_argument("--negative_type", type=str, default="none")

    # --- Graded personalized retriever on iKAT qrels (--dataset_type ikat_graded). All default to TopiOCQA. ---
    parser.add_argument("--dataset_type", type=str, default="topiocqa",
                        choices=["topiocqa", "ikat_graded"],
                        help="topiocqa (default, unchanged) | ikat_graded (per-query graded candidate pool).")
    parser.add_argument("--ikat_manifest_file", type=str, default=None,
                        help="iKAT graded train manifest JSONL ({sample_id,query_text,candidates:[[docid,grade]],split,...}).")
    parser.add_argument("--ikat_val_manifest_file", type=str, default=None,
                        help="Val manifest for the NDCG@3 hook; default None -> reuse --ikat_manifest_file (split=val).")
    parser.add_argument("--ikat_doc_embedding_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/ikat_graded/ikat_graded_doc_embeddings.pt",
                        help="Frozen base-qwen3 judged-doc embeddings {docid: Tensor(1024)}.")
    parser.add_argument("--ikat_split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--ikat_max_query_length", type=int, default=32768,
                        help="Tokenize/pad cap for the (long, full-PTKB) iKAT query; matches the offline eval.")
    parser.add_argument("--ikat_grade0_cap", type=str, default="all",
                        help="Per-query gold grade-0 cap: 'all' | int K | '0'. Sweep {0,16,32,64,all}.")
    parser.add_argument("--ikat_extra_negatives", type=str, default="none",
                        choices=["none", "in_batch", "cross_device"],
                        help="Extra negatives beyond the per-query pool. v1 supports 'none' only "
                             "(in_batch/cross_device are v2).")
    parser.add_argument("--bce_logit_scale", type=float, default=20.0,
                        help="BiXSE logit scale alpha (temperature 1/alpha); paper default ~20.")
    parser.add_argument("--graded_temperature", type=float, default=0.01,
                        help="Softmax temperature tau for graded_infonce.")
    parser.add_argument("--activate_eval_ikat_graded_while_training", action="store_true",
                        help="Per-epoch graded NDCG@3 on the val split (pool rerank) + epoch-0 floor + best tracking.")
    parser.add_argument("--ikat_save_best_only", action="store_true",
                        help="Sweep mode: save ONLY the best-NDCG@3 trained epoch to <out>/best (overwrite "
                             "on improvement), skip per-epoch ckpts. Requires --activate_eval_ikat_graded_while_training.")
    # --- anti-collapse extras (2026-06-20): keep the query aligned with the frozen base-qwen3 doc space ---
    parser.add_argument("--ikat_anchor_weight", type=float, default=0.0,
                        help="Anchor reg weight lambda: loss += lambda*mean(1-cos(q, q_init)); keeps the trained "
                             "query from rotating away from init (init stays compatible with the base-qwen3 doc index).")
    parser.add_argument("--ikat_anchor_emb_file", type=str, default=None,
                        help="Precomputed init query embeddings {sample_id: Tensor(1024)} for the anchor term.")
    parser.add_argument("--ikat_global_neg_file", type=str, default=None,
                        help="Precomputed corpus doc-emb pool Tensor(N,1024); sampled as grade-0 negatives so the "
                             "query is calibrated against the FULL corpus, not only the judged pool.")
    parser.add_argument("--ikat_global_neg_k", type=int, default=0,
                        help="Number of global corpus negatives sampled per step (0 = off).")
    # --- ConvDR-style KD: distill query toward the oracle-rewrite teacher embedding ---
    parser.add_argument("--ikat_kd_weight", type=float, default=0.0,
                        help="ConvDR-KD weight: loss += w*mean(1-cos(q, q_oracle_teacher)). Pulls the query "
                             "toward the oracle-rewrite teacher emb -> teaches personalization AND keeps it on "
                             "the base-qwen3 doc manifold (anti-collapse with a direction).")
    parser.add_argument("--ikat_kd_emb_file", type=str, default=None,
                        help="Precomputed oracle-rewrite teacher embeddings {sample_id: Tensor(1024)} "
                             "(base-qwen3, web-search instruction).")
    parser.add_argument("--ikat_bixse_weight", type=float, default=1.0,
                        help="Weight on the BiXSE/graded ranking loss. 1.0 = normal; 0.0 = pure KD "
                             "(only the oracle-rewrite distillation term drives training, ConvDR-style).")
    parser.add_argument("--ikat_kd_list_weight", type=float, default=0.0,
                        help="Listwise KD weight (round4 C2): KL of student's candidate-pool score "
                             "distribution to the teacher's. Distills the teacher's relative ranking.")
    parser.add_argument("--ikat_kd_list_temp", type=float, default=0.05,
                        help="Temperature for the listwise KD softmax over the candidate pool.")

    # --- LoRA (parameter-efficient FT) ---
    parser.add_argument("--use_lora", action="store_true",
                        help="Wrap the inner Qwen3 model with LoRA; only adapter params (+bce_beta) train.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="*", default=None,
                        help="Default = [q,k,v,o,gate,up,down]_proj.")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--no_lr_schedule", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="LR decay after warmup. linear = current/BGE default; "
                             "cosine = official Qwen3-Embedding ms-swift default. "
                             "Ignored when --no_lr_schedule is set.")

    # Official Qwen3-Embedding InfoNCE alignment (all default to legacy behavior)
    parser.add_argument("--infonce_temperature", type=float, default=1.0,
                        help="Divide similarity logits by this temperature (InfoNCE). "
                             "1.0 = legacy no-op; official Qwen3-Embedding uses 0.01.")
    parser.add_argument("--cross_device_negatives", action="store_true",
                        help="All-gather the frozen pos/neg doc embeddings across DDP "
                             "ranks so each query's in-batch negatives = the GLOBAL "
                             "batch (official INFONCE_USE_BATCH). Requires n_gpu>1 and "
                             "DistributedSampler(drop_last=True).")
    parser.add_argument("--save_optimizer", action="store_true",
                        help="Save optimizer.pt + scheduler.pt at the final epoch (for "
                             "resume). Default OFF — training is cheap and we don't "
                             "resume, so no optimizer state is written (~2.4 GB/ckpt saved).")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="AdamW beta1 (default 0.9).")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="AdamW beta2. Default 0.999; official Qwen3-Embedding uses 0.95.")
    parser.add_argument("--mask_fake_negative", action="store_true",
                        help="Official Qwen3-Embedding INFONCE_MASK_FAKE_NEGATIVE: in the "
                             "cross-device InfoNCE loss, set a candidate's logit to -inf when "
                             "its cosine similarity exceeds the positive's by --fake_neg_margin "
                             "(likely an unlabeled relevant). Requires --cross_device_negatives.")
    parser.add_argument("--fake_neg_margin", type=float, default=0.1,
                        help="Cosine margin for --mask_fake_negative (official default 0.1).")
    parser.add_argument("--dedup_same_gold", action="store_true",
                        help="Mask any candidate whose passage id == the query's gold pid "
                             "(exact duplicate positive, or a BM25 negative that is actually "
                             "the gold). Catches false negatives the similarity mask misses. "
                             "Requires --cross_device_negatives.")

    # Curriculum learning
    parser.add_argument("--curriculum_type", type=str, default="none",
                        choices=["none", "easy2hard", "hard2easy"])
    parser.add_argument("--scoring_function", type=str, default="turn_length",
                        choices=list(SCORING_FUNCTIONS.keys()))
    parser.add_argument("--pacing_function", type=str, default="root_2",
                        choices=list(PACING_FUNCTIONS.keys()))
    parser.add_argument("--curriculum_c0", type=float, default=0.2)
    parser.add_argument("--curriculum_end_epoch", type=int, default=16)
    parser.add_argument("--conv_instruction", type=str, default="",
                        help="If non-empty, build conversational queries with the "
                             "official Qwen3-Embedding instruct-text path. The "
                             "exact template depends on --template_version: "
                             "v1 (legacy) -> 'Instruct: {conv_instruction}\\n"
                             "Conversation:{q1}\\n{r1}\\n...{q_cur}<|endoftext|>'; "
                             "v2 -> 'Instruct: {conv_instruction}\\n"
                             "Conversation: User: {q1} System: {r1} ... User: {q_cur}"
                             "<|endoftext|>'. Single trailing <|endoftext|>, "
                             "last-token pooled, byte-identical for training and eval. "
                             "Empty = legacy [CLS]..[SEP] ANCE-style path.")
    parser.add_argument("--template_version", type=str, default="v1",
                        choices=["v1", "v2", "v3"],
                        help="Conversational instruct template version (see "
                             "--conv_instruction). v1 (default) is byte-identical "
                             "to the 2026-05-19 instruct2_qwen_* checkpoint family; "
                             "v2 (added 2026-06-05) inserts explicit User:/System: "
                             "role markers and uses single-space turn separators; "
                             "v3 (added 2026-06-05) is v2 plus the trailing user "
                             "utterance gets `User's last question:` instead of "
                             "`User:`. Ignored when --conv_instruction is empty.")

    # Memory / speed
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--fp32_master_weights", action="store_true",
                        help="Keep fp32 master weights + fp32 AdamW moments (load model in "
                             "fp32 + sdpa attention) while using autocast(bf16) for compute. "
                             "Reproduces the official DeepSpeed-ZeRO bf16 numerics and avoids "
                             "pure-bf16's update-rounding floor (lr=1e-5 steps fall below bf16 "
                             "ULP for |theta|>=0.0026). Incompatible with --use_flash_attention "
                             "(FA2 needs half-precision weights); sdpa is used instead.")
    parser.add_argument("--use_tf32", action="store_true")
    parser.add_argument("--use_fused_optimizer", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_compile", action="store_true")

    # --- Speed: canonical bf16 + fp32-master (the DeepSpeed-ZeRO bf16 way) ---
    parser.add_argument("--bf16_fp32_master", action="store_true",
                        help="CANONICAL fp32-master mode: load the model in bf16 (so "
                             "FlashAttention-2 works) but keep a SEPARATE fp32 master copy "
                             "of every trainable param + fp32 AdamW moments. Each optimizer "
                             "step copies the (DDP-averaged) bf16 grads into the fp32 master "
                             "grads, clips on the master, steps the fp32 AdamW, then copies "
                             "the updated fp32 master back into the bf16 model — exactly like "
                             "DeepSpeed's bf16_optimizer. Same numerics as --fp32_master_weights "
                             "(no bf16 update floor) but recovers FlashAttention-2: the VRAM win "
                             "is in activations (FA2 avoids the O(L^2) attention buffer and lets "
                             "grad-checkpointing be turned off), not fixed state. Mutually exclusive "
                             "with --fp32_master_weights (kept only for A/B verification).")

    # --- Speed: DataLoader workers / prefetch ---
    # Tokenization is precomputed at dataset-build time, so per-batch worker
    # work is only padding+stacking pre-tokenized ids (light). A few workers +
    # prefetch fully overlap that with the (long) GPU step; many workers only
    # waste RAM (each fork copies the example list) — sweep with --profile_timing.
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader worker processes (default 0 = load in the main "
                             "process, current behavior). Tokenization is precomputed so the "
                             "only per-batch CPU work is padding; 2-8 workers + prefetch are "
                             "plenty to hide it behind the GPU step. Persistent workers are "
                             "enabled automatically when >0.")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4,
                        help="Batches each DataLoader worker prefetches ahead. Only used when "
                             "--dataloader_num_workers > 0 (ignored otherwise).")

    # --- Speed: keep the frozen pos/neg/oracle embedding table on GPU ---
    parser.add_argument("--gpu_resident_doc_table", action="store_true",
                        help="Move the whole pre-encoded pos/neg/oracle embedding table "
                             "(~570 MB for TopiOCQA) onto the GPU once at startup, so the "
                             "per-step gather is a pure on-device index instead of a "
                             "CPU->GPU copy every step. Costs ~570 MB VRAM per rank; trivially "
                             "fits a 46 GB card. Default OFF (current CPU-gather behavior).")

    # --- Profiling: per-phase step timing + GPU util/mem ---
    parser.add_argument("--profile_timing", action="store_true",
                        help="Log rolling averages of data_time / forward_backward_time / "
                             "optimizer_time / step_time plus GPU util and peak memory every "
                             "--profile_every_steps optimizer steps. Adds a cuda.synchronize "
                             "per phase (small overhead) — use it to pick the optimal "
                             "num_workers / gpu_resident / bf16_fp32_master config, then turn off.")
    parser.add_argument("--profile_every_steps", type=int, default=20,
                        help="How often (in optimizer steps) to emit the --profile_timing line.")

    # Eval
    # ── Step-driven mode + gradient recording (turn-bucket experiment) ──────
    parser.add_argument("--total_train_steps", type=int, default=0,
                        help="If >0: train for exactly this many optimizer steps, cycling "
                             "over the (small) dataset, instead of epoch mode. A checkpoint "
                             "is saved every --save_every_steps steps. Requires "
                             "--curriculum_type none.")
    parser.add_argument("--save_every_steps", type=int, default=47,
                        help="Checkpoint cadence in step-driven mode (intermediate ckpts "
                             "are model-only; the final one also stores optimizer state).")
    parser.add_argument("--record_grad_stats", action="store_true",
                        help="Record per-step per-tensor grad norms (R1) and whole-run "
                             "per-scalar sum_g / sum_g2 accumulators (R2); see comments at "
                             "the recorder init.")
    parser.add_argument("--grad_stats_device", type=str, default="gpu",
                        choices=["gpu", "cpu"],
                        help="Where the two fp32 R2 accumulators (2.4 GB each) live. 'gpu' "
                             "costs ~4.8 GB VRAM on rank 0 but adds ~10 ms/step; 'cpu' is "
                             "VRAM-free but adds ~0.5 s/step for the device copy.")

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

    # QReCC eval
    parser.add_argument("--activate_eval_qrecc_while_training", action="store_true")
    parser.add_argument("--qrecc_valid_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/qrecc/qrecc_valid.jsonl")
    parser.add_argument("--qrecc_qrel_file", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/qrecc_qrel.trec")
    parser.add_argument("--qrecc_embedding_dir", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/qrecc_qwen_merged",
                        help="Dir with Qwen3 corpus blocks (doc_emb_block.*.pb) for QReCC eval.")

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
