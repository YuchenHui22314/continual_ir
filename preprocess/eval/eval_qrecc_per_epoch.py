"""
eval_qrecc_per_epoch.py
=======================
Per-epoch QReCC NDCG@10 eval for ONE of three encoder regimes (one process per
setting because each needs its own ~160 GB resident QReCC corpus):

    --setting ance            : 8 ANCE fine-tunes (ckpts under huggingface/continual_ir/)
    --setting qwen_no_instr   : 8 OLD Qwen runs trained WITHOUT conv instruction
    --setting qwen_instr      : 8 NEW instruct2 Qwen runs trained WITH conv instruction

For each setting we load the QReCC corpus once (qrecc_ance_merged for ANCE,
qrecc_qwen_merged for both Qwen settings), then iterate 8 runs × 20
checkpoint steps + (optionally) the corresponding zero-shot baseline,
encoding queries via the regime-appropriate path (legacy [CLS]..[SEP] for
ANCE & Qwen-no-instr; official Instruct: ...\\nConversation: text path for
Qwen-instr). Results are saved incrementally to a per-setting JSON.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python preprocess/eval/eval_qrecc_per_epoch.py \\
        --setting ance         --include_zero_shot \\
        2>&1 | tee /data/rech/huiyuche/TREC_iKAT_2024/logs/eval_qrecc_perepoch_ance_$(date +%Y%m%d_%H%M%S).log
"""

import sys, os, json, gc, time, argparse, logging, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer

from utils import load_corpus_into_faiss, eval_conv_search
from models import ANCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── shared paths ──────────────────────────────────────────────────────────────
CKPT_BASE      = "/data/rech/huiyuche/huggingface/continual_ir"
QRECC_DATA     = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/qrecc/qrecc_valid.jsonl"
# Derived QReCC valid file with `Context` field rebuilt as RAW conversation history
# (interleaved raw [Q_{1}, A_{1}, ..., Q_{N-1}, A_{N-1}] from earlier turns of the
# same Conversation_no). The dataset-provided Context embeds Truth_rewrite (oracle
# coref-resolved) for prior turns, which would make our eval comparable to
# "oracle-history + raw-current" baselines rather than the raw-conversation
# setting used by TopiOCQA and by literature reporting on QReCC. We reconstruct
# the raw history once and cache to this derived path.
QRECC_DATA_RAW = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/qrecc/qrecc_valid_rawconv.jsonl"
QRECC_QREL     = "/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/qrecc_qrel.trec"
QRECC_ANCE_EMB = "/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/qrecc_ance_merged"
QRECC_QWEN_EMB = "/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/qrecc_qwen_merged"
RESULTS_DIR    = "/data/rech/huiyuche/continual_ir/figures"

ANCE_BASE_CKPT = "/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234"
QWEN_BASE_CKPT = "/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

CKPT_STEPS_DEFAULT = [94 * i for i in range(1, 21)]   # [94, 188, ..., 1880]


def discover_ckpt_steps(run_name):
    """Return the sorted list of step ints found under <CKPT_BASE>/<run_name>/.
    Use this when step numbering varies per run (e.g. ANCE curriculum runs whose
    early epochs have fewer optimizer steps due to pacing-restricted dataset
    size). Each run on disk is expected to have exactly 20 checkpoint-step-N
    directories, one per training epoch."""
    pat = os.path.join(CKPT_BASE, run_name, "checkpoint-step-*")
    steps = []
    for d in glob.glob(pat):
        m = re.search(r"checkpoint-step-(\d+)$", d)
        if m and os.path.isdir(d):
            steps.append(int(m.group(1)))
    return sorted(steps)

# Conversational-instruction strings live in utils; we import both so the
# --template_version CLI flag can swap between them without duplicating text.
from utils import CONV_INSTRUCTION_V1, CONV_INSTRUCTION_V2, CONV_INSTRUCTION_V3
CONV_INSTRUCTION = CONV_INSTRUCTION_V1   # legacy alias for the doc-string above


def build_rawconv_qrecc(src: str = QRECC_DATA,
                        dst: str = QRECC_DATA_RAW,
                        force: bool = False) -> str:
    """
    Reconstruct *raw-conversation* history for the QReCC valid set.

    QReCC's `Context` field in qrecc_valid.jsonl embeds Truth_rewrite (oracle
    coref-resolved versions) of all prior turns, not the literal raw Questions
    the user typed. That makes our eval an "oracle-history + raw-current" setting
    rather than the all-raw setting used by TopiOCQA and by QReCC literature.

    This helper walks the valid set once, builds an index
    `(Conversation_no, Turn_no) -> {Question, Answer}`, then rewrites each
    record's `Context` to the raw interleaved list
    `[Q_1, A_1, Q_2, A_2, ..., Q_{N-1}, A_{N-1}]`
    using only earlier turns of the same conversation. Other fields are kept
    verbatim so the file stays a drop-in for eval_conv_search().

    Cached at `dst`; only rebuilt if missing or `force=True`.
    """
    if os.path.exists(dst) and not force:
        logger.info("Reusing cached raw-conv QReCC valid jsonl: %s", dst)
        return dst

    # 1) index every turn by (conversation, turn) so prior turns are recoverable.
    by_conv: dict = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cn, tn = r["Conversation_no"], r["Turn_no"]
            by_conv.setdefault(cn, {})[tn] = {
                "Q": r.get("Question") or "",
                "A": r.get("Answer")   or "",
            }

    # 2) emit, overriding Context with raw history reconstruction.
    n_written = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            r  = json.loads(line)
            cn, tn = r["Conversation_no"], r["Turn_no"]
            raw_ctx = []
            for prev_t in sorted(by_conv[cn].keys()):
                if prev_t >= tn:
                    break
                qa = by_conv[cn][prev_t]
                raw_ctx.append(qa["Q"])   # even index = user turn
                raw_ctx.append(qa["A"])   # odd  index = answer
            r["Context"] = raw_ctx
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += 1
    logger.info("Wrote %d raw-conv QReCC records to %s", n_written, dst)
    return dst


def _debug_print_sample_query(test_data_file: str,
                              tokenizer,
                              conv_instruction: str,
                              max_query_length: int,
                              max_response_length: int,
                              max_concat_length: int,
                              setting_label: str,
                              template_version: str = "v1",
                              sample_qid: str = None,
                              min_turn_for_sample: int = 3) -> None:
    """
    Print one fully-prepared QReCC encoder input so the human can sanity-check
    that the raw-conversation reconstruction + truncation caps produce the
    expected text and token budget BEFORE launching the long eval.

    sample_qid: 'C-N' string (e.g. '64-11') — exact qid to pull. If None,
                pick the first record with Turn_no >= min_turn_for_sample.
    Mirrors exactly what eval_conv_search() will feed the model.
    """
    from utils import _build_topiocqa_query_tokens   # local: avoids re-export

    with open(test_data_file, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    if sample_qid is not None:
        try:
            cn, tn = sample_qid.split("-", 1)
            cn, tn = int(cn), int(tn)
            sample = next(
                (r for r in records if r["Conversation_no"] == cn and r["Turn_no"] == tn),
                None,
            )
            if sample is None:
                logger.warning("sample_qid=%s not found — falling back to first record"
                               " with Turn_no >= %d", sample_qid, min_turn_for_sample)
        except (ValueError, AttributeError):
            logger.warning("sample_qid=%r unparsable — falling back to min_turn search",
                           sample_qid)
            sample = None
    else:
        sample = None
    if sample is None:
        sample = next(
            (r for r in records if r["Turn_no"] >= min_turn_for_sample
                                  and len(r.get("Context", [])) >= 2),
            records[0],
        )

    logger.info("=" * 90)
    logger.info("SAMPLE TURN — setting=%s; verify encoder input before launching full eval",
                setting_label)
    logger.info("=" * 90)
    logger.info("qid=%d-%d   Conversation_source=%s",
                sample["Conversation_no"], sample["Turn_no"],
                sample.get("Conversation_source"))
    logger.info("Question  (raw current)            : %r", sample["Question"])
    logger.info("Truth_rewrite (oracle, NOT used)   : %r", sample.get("Truth_rewrite"))
    logger.info("Raw Context (%d items, even=Q odd=A):", len(sample["Context"]))
    for i, c in enumerate(sample["Context"]):
        role  = "Q" if i % 2 == 0 else "A"
        short = (c[:160] + "…") if len(c) > 160 else c
        logger.info("  Context[%d] (%s) = %r", i, role, short)

    tokens = _build_topiocqa_query_tokens(
        tokenizer,
        cur_utt_text        = sample["Question"],
        ctx_utts_text       = sample["Context"],
        max_query_length    = max_query_length,
        max_response_length = max_response_length,
        max_concat_length   = max_concat_length,
        conv_instruction    = conv_instruction,
        template_version    = template_version,
    )

    raw_tok = getattr(tokenizer, "_tok", tokenizer)
    decoded = raw_tok.decode(tokens, skip_special_tokens=False)
    logger.info("Total token count : %d   (cap max_concat_length=%d)",
                len(tokens), max_concat_length)
    logger.info("Per-utt caps      : max_query_length=%d   max_response_length=%d   template_version=%s",
                max_query_length, max_response_length, template_version)
    logger.info("Decoded encoder input (verbatim, with special tokens):")
    logger.info("---------- BEGIN encoder input ----------\n%s", decoded)
    logger.info("---------- END   encoder input ----------")
    logger.info("=" * 90)


# Reproduces the caps used at TRAINING time for Qwen instruct (train_qwen_cl.py).
# Used by the alignment-check printer below: we tokenize ONE TopiOCQA training
# record with these caps so the reader can verify the eval-time encoder input
# follows the exact same prefix / separator / EOS pattern (length aside).
TOPIOCQA_TRAIN_FILE = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl"
TRAIN_TIME_CAPS = dict(
    max_query_length    = 64,
    max_response_length = 64,
    max_concat_length   = 512,
)


def _debug_print_topiocqa_training_sample(tokenizer,
                                          conv_instruction: str,
                                          setting_label: str,
                                          template_version: str = "v1",
                                          min_ctx: int = 4) -> None:
    """
    Print one TopiOCQA TRAINING-time encoder input under the exact training
    caps (64/64/512), so the human can confirm the QReCC eval-time input
    follows the same `Instruct: ...\\nConversation:Q\\nA\\n...\\nQ_cur<|endoftext|>`
    pattern. Length-budget is different (QReCC eval bumps to 8192) and that
    is flagged explicitly.
    """
    from utils import _build_topiocqa_query_tokens   # local import: avoid re-export

    sample = None
    with open(TOPIOCQA_TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("ctx_utts_text", [])) >= min_ctx:
                sample = r
                break
    if sample is None:
        logger.warning("No TopiOCQA training record with ctx>=%d found — skipping",
                       min_ctx)
        return

    logger.info("=" * 90)
    logger.info("TOPIOCQA TRAINING-TIME encoder input — for alignment check vs QReCC eval")
    logger.info("setting=%s   training caps: max_query=%d max_resp=%d max_concat=%d",
                setting_label,
                TRAIN_TIME_CAPS["max_query_length"],
                TRAIN_TIME_CAPS["max_response_length"],
                TRAIN_TIME_CAPS["max_concat_length"])
    logger.info("=" * 90)
    logger.info("sample_id     : %s", sample.get("sample_id"))
    logger.info("cur_utt_text  : %r", sample["cur_utt_text"])
    logger.info("ctx_utts_text (%d items, even=Q odd=A):", len(sample["ctx_utts_text"]))
    for i, c in enumerate(sample["ctx_utts_text"]):
        role  = "Q" if i % 2 == 0 else "A"
        short = (c[:160] + "…") if len(c) > 160 else c
        logger.info("  ctx_utts_text[%d] (%s) = %r", i, role, short)

    tokens = _build_topiocqa_query_tokens(
        tokenizer,
        cur_utt_text        = sample["cur_utt_text"],
        ctx_utts_text       = sample["ctx_utts_text"],
        conv_instruction    = conv_instruction,
        template_version    = template_version,
        **TRAIN_TIME_CAPS,
    )
    raw_tok = getattr(tokenizer, "_tok", tokenizer)
    decoded = raw_tok.decode(tokens, skip_special_tokens=False)
    logger.info("Total token count : %d   (cap max_concat_length=%d)",
                len(tokens), TRAIN_TIME_CAPS["max_concat_length"])
    logger.info("Decoded encoder input (verbatim, with special tokens):")
    logger.info("---------- BEGIN training encoder input ----------\n%s", decoded)
    logger.info("---------- END   training encoder input ----------")
    logger.info("NOTE: QReCC eval (Qwen settings) uses the SAME caps 64/64/512 as training")
    logger.info("      — eval is byte-identical structure AND budget to training.")
    logger.info("=" * 90)


# ── per-setting run lists (8 runs each, all paper rows) ───────────────────────
RUNS_BY_SETTING = {
    "ance": [
        "ance_topiocqa_nosched",
        "ance_curriculum_step",
        "ance_curriculum_step_exclusive",
        "ance_curriculum_step_excl_2_full",
        "ance_curriculum_root2",
        "topiocqa_anticl_root_2",
        "topiocqa_anticl_step",
        "topiocqa_anticl_step_exclusive",
    ],
    "qwen_no_instr": [
        "qwen_nosched",
        "qwen_cl_step",
        "qwen_cl_step_excl",
        "qwen_cl_step_excl_2_full",
        "qwen_cl_root2",
        "qwen_acl_root2",
        "qwen_acl_step",
        "qwen_acl_step_excl",
    ],
    "qwen_instr": [
        "instruct2_qwen_nosched",
        "instruct2_qwen_cl_step",
        "instruct2_qwen_cl_step_excl",
        "instruct2_qwen_cl_step_excl_2_full",
        "instruct2_qwen_cl_root2",
        "instruct2_qwen_acl_root2",
        "instruct2_qwen_acl_step",
        "instruct2_qwen_acl_step_excl",
    ],
}

# ── Qwen3 query encoder + tokenizer wrapper (verbatim copy of training defs) ──
class QwenQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Match training: left-padded last-token pool, fp32 normalize.
        # NOTE: prior version used `attention_mask.sum(dim=1)-1` indexing,
        # which is RIGHT-padding semantics and silently produces zero
        # vectors when tokenizer.padding_side='left' (the actual setup).
        embs = out.last_hidden_state[:, -1, :]
        return F.normalize(embs.float(), p=2, dim=-1)


class Qwen3TokenizerWrapper:
    """Verbatim copy of the wrapper used during qwen_no_instr training so the
    legacy [CLS]..[SEP] path in _build_topiocqa_query_tokens sees the same
    [EOS, tokens, EOS] sequences the model was trained on.
    cls=sep=eos=151645 (<|im_end|>), pad=151643 (<|endoftext|>).
    """
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.cls_token_id = tokenizer.eos_token_id
        self.sep_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.eos_token_id
        self.vocab_size   = tokenizer.vocab_size

    def encode(self, text, add_special_tokens=True, max_length=512, truncation=True):
        budget = max(1, max_length - 2) if add_special_tokens else max_length
        ids = self._tok.encode(text, add_special_tokens=False,
                               max_length=budget, truncation=truncation)
        if add_special_tokens:
            return [self.eos_token_id] + ids + [self.eos_token_id]
        return ids

    def __call__(self, texts, max_length=512, padding=True, truncation=True,
                 return_tensors=None, **kwargs):
        encoded_ids = [self.encode(t, add_special_tokens=True,
                                   max_length=max_length, truncation=truncation)
                       for t in (texts if isinstance(texts, list) else [texts])]
        if padding:
            max_len = max(len(ids) for ids in encoded_ids)
            input_ids = [
                [self.pad_token_id] * (max_len - len(ids)) + ids
                for ids in encoded_ids
            ]
        else:
            input_ids = encoded_ids
        attention_mask = [[0] * (len(row) - len(ids)) + [1] * len(ids)
                          for row, ids in zip(input_ids, encoded_ids)]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask)}
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ── setting → (corpus_dir, embed_dim, conv_instruction, left_padding, loader) ─
def setting_spec(setting, device, template_version="v1"):
    """
    template_version only affects the qwen_instr setting:
      v1 (default, byte-identical to the 2026-05-19 instruct2_qwen_* training):
        CONV_INSTRUCTION_V1 + newline-joined turns.
      v2 (added 2026-06-05):
        CONV_INSTRUCTION_V2 + 'User: ... System: ...' role-marker template,
        single-space turn separators. OOD for any v1-trained checkpoint;
        intended for fresh zero-shot evals or a future instruct3 retrain.
    The ance and qwen_no_instr settings carry conv_instruction="" so the
    template_version flag has no effect on them.
    """
    if setting == "ance":
        def loader(ckpt_path):
            tok = RobertaTokenizer.from_pretrained(ckpt_path)
            enc = ANCE.from_pretrained(ckpt_path).to(device).eval()
            enc.config.use_cache = False
            return tok, enc
        return dict(
            corpus_dir       = QRECC_ANCE_EMB,
            embed_dim        = 768,
            conv_instruction = "",
            left_padding     = False,
            zero_shot_ckpt   = ANCE_BASE_CKPT,
            loader           = loader,
            template_version = template_version,
            # ANCE backbone (RoBERTa-base) is hard-capped at 512 positions;
            # we keep the same per-utt budget the TopiOCQA training used.
            max_query_length    = 32,
            max_response_length = 32,
            max_concat_length   = 512,
        )
    if setting == "qwen_no_instr":
        def loader(ckpt_path):
            tok = AutoTokenizer.from_pretrained(ckpt_path)
            tok.padding_side = "left"
            wrap = Qwen3TokenizerWrapper(tok)
            base = AutoModel.from_pretrained(
                ckpt_path, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            ).to(device).eval()
            base.config.use_cache = False
            return wrap, QwenQueryEncoder(base)
        return dict(
            corpus_dir       = QRECC_QWEN_EMB,
            embed_dim        = 1024,
            conv_instruction = "",
            left_padding     = True,
            zero_shot_ckpt   = QWEN_BASE_CKPT,
            loader           = loader,
            template_version = template_version,
            # Match training distribution exactly: train_qwen_cl.py defaults
            # were max_query_length=64, max_response_length=64,
            # max_concat_length=512 (see train_qwen_cl.py:731-734). Eval at
            # these same caps keeps the model in-distribution; smart
            # truncation drops oldest history first when total > 512.
            max_query_length    = 64,
            max_response_length = 64,
            max_concat_length   = 512,
        )
    if setting == "qwen_instr":
        def loader(ckpt_path):
            tok = AutoTokenizer.from_pretrained(ckpt_path)
            tok.padding_side = "left"
            base = AutoModel.from_pretrained(
                ckpt_path, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            ).to(device).eval()
            base.config.use_cache = False
            return tok, QwenQueryEncoder(base)
        # Pick the conv-instruction string that matches the requested template.
        conv_instr = {"v1": CONV_INSTRUCTION_V1,
                      "v2": CONV_INSTRUCTION_V2,
                      "v3": CONV_INSTRUCTION_V3}[template_version]
        return dict(
            corpus_dir       = QRECC_QWEN_EMB,
            embed_dim        = 1024,
            conv_instruction = conv_instr,
            left_padding     = True,
            zero_shot_ckpt   = QWEN_BASE_CKPT,
            loader           = loader,
            template_version = template_version,
            # Match training distribution exactly (instruct2_qwen runs trained
            # with max_query_length=64, max_response_length=64,
            # max_concat_length=512). Smart truncation drops oldest history
            # first when total > 512.
            max_query_length    = 64,
            max_response_length = 64,
            max_concat_length   = 512,
        )
    raise ValueError(f"Unknown setting: {setting}")


# ── checkpointed state helpers ────────────────────────────────────────────────
def _save(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _load_existing(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing {path}: {e} — starting fresh")
    return {}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True,
                    choices=list(RUNS_BY_SETTING.keys()))
    ap.add_argument("--include_zero_shot", action="store_true", default=True)
    ap.add_argument("--no_zero_shot", dest="include_zero_shot", action="store_false")
    ap.add_argument("--ckpt_steps", type=int, nargs="+", default=None,
                    help="explicit step numbers (default: auto-discover per run from disk; "
                         "ANCE curriculum runs have variable per-epoch step counts).")
    ap.add_argument("--runs", type=str, nargs="+", default=None,
                    help="optional: limit to a subset of run names (default: all 8).")
    ap.add_argument("--results_out", type=str, default=None,
                    help="default figures/qrecc_per_epoch_<setting>.json")
    ap.add_argument("--eval_bs", type=int, default=64)
    # GPU FAISS strategy:
    #   --gpu_fp16 (DEFAULT): shard index across all visible GPUs in fp16.
    #     Qwen 1024d × 54.6M × 2B = 106 GB → ~26.5 GB/GPU (fits 46 GB L40S).
    #     ANCE 768d × 54.6M × 2B =  84 GB → ~21   GB/GPU (easy fit).
    #     Single GPU transfer + sharded fp16 stays resident for every ckpt.
    #   --gpu_fp32: same but fp32 (Qwen WON'T fit at 211 GB / 184 GB GPU total).
    #   --cpu_faiss: CPU IndexFlatIP (slow, ~10-20 min/ckpt).
    ap.add_argument("--faiss_backend", choices=["gpu_fp16", "gpu_fp32", "cpu"],
                    default="gpu_fp16")
    # Conversational instruct template version (only affects qwen_instr).
    # v1 = byte-identical to the 2026-05-19 instruct2_qwen_* training family;
    # v2 = new role-marker template (added 2026-06-05). v1-trained checkpoints
    # are OOD under v2, so v2 is most informative on zero-shot Qwen3-base.
    ap.add_argument("--template_version", type=str, default="v1",
                    choices=["v1", "v2", "v3"],
                    help="Conversational instruct template version "
                         "(see src/utils.py:build_qwen_instruct_query_ids).")
    # Dry-run: build raw-conv jsonl, print one sample encoder input, then exit
    # WITHOUT loading the 160 GB corpus or any checkpoint past the tokenizer.
    # Used for fast code-review of the query-construction pipeline.
    ap.add_argument("--dry_run", action="store_true", default=False,
                    help="print sample encoder input and exit (no corpus load, no eval).")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = setting_spec(args.setting, device, template_version=args.template_version)
    runs = args.runs or RUNS_BY_SETTING[args.setting]
    # Avoid clobbering v1 results when running a v2 eval: default output now
    # carries a `_v2` suffix unless the user passes --results_out explicitly.
    default_suffix = "" if args.template_version == "v1" else f"_{args.template_version}"
    results_out = args.results_out or os.path.join(
        RESULTS_DIR, f"qrecc_per_epoch_{args.setting}{default_suffix}.json"
    )

    logger.info("Setting=%s; output=%s", args.setting, results_out)
    logger.info("Runs: %s", runs)
    logger.info("Ckpt steps: %s", args.ckpt_steps)

    # 0) one-time: build the raw-conversation QReCC valid jsonl (overrides
    # dataset Context = Truth_rewrite-history with raw [Q_i, A_i] history)
    # so our eval matches TopiOCQA's raw-conv setting and QReCC literature.
    qrecc_test_file = build_rawconv_qrecc()

    # Sanity-print one fully-prepared encoder input under this setting's caps
    # BEFORE loading the 160 GB corpus, so a bad config aborts fast.
    # spec["loader"] also instantiates the encoder; we free it right after.
    _sample_tok, _sample_enc = spec["loader"](spec["zero_shot_ckpt"])
    # Two samples: an easy short one (early in conversation, all under per-utt
    # caps) AND a deep worst-case one (10 turns, ~700 raw tokens — guaranteed
    # to exercise per-utt 64-cap and total 512-cap drop-oldest behavior).
    for label, qid in [("LIGHT", "1-3"), ("WORST_CASE_DEEP", "64-11")]:
        logger.info(">>> printing %s sample (qid=%s) <<<", label, qid)
        _debug_print_sample_query(
            test_data_file      = qrecc_test_file,
            tokenizer           = _sample_tok,
            conv_instruction    = spec["conv_instruction"],
            max_query_length    = spec["max_query_length"],
            max_response_length = spec["max_response_length"],
            max_concat_length   = spec["max_concat_length"],
            setting_label       = args.setting,
            template_version    = spec["template_version"],
            sample_qid          = qid,
        )
    # Companion print: the SAME tokenization on a TopiOCQA TRAINING record
    # at the actual training caps (64/64/512). Lets the reader confirm the
    # QReCC eval encoder input above follows the identical prefix/separator/
    # EOS structure that the model was actually tuned on.
    _debug_print_topiocqa_training_sample(
        tokenizer        = _sample_tok,
        conv_instruction = spec["conv_instruction"],
        setting_label    = args.setting,
        template_version = spec["template_version"],
    )
    del _sample_tok, _sample_enc
    gc.collect()
    torch.cuda.empty_cache()

    if args.dry_run:
        logger.info("--dry_run: skipping corpus load and full eval. Exiting.")
        return

    # 1) one-time: QReCC corpus into CPU FAISS; eval_conv_search transfers
    # it to GPU sharded once (and keeps it via gpu_index_cache).
    t0 = time.time()
    logger.info("Loading QReCC corpus into CPU FAISS from %s (~160 GB on disk) ...",
                spec["corpus_dir"])
    faiss_index, doc_ids = load_corpus_into_faiss(
        spec["corpus_dir"], embed_dim=spec["embed_dim"], use_gpu=False,
    )
    logger.info("CPU FAISS loaded in %.1f min (%d docs).",
                (time.time() - t0) / 60, faiss_index.ntotal)

    # GPU sharded transfer happens inside eval_conv_search on the first ckpt;
    # gpu_index_cache + keep_faiss_on_gpu=True keep it resident across all ckpts.
    use_gpu_faiss_in_eval = (args.faiss_backend != "cpu")
    use_gpu_fp16          = (args.faiss_backend == "gpu_fp16")
    gpu_index_cache       = {} if use_gpu_faiss_in_eval else None
    logger.info("FAISS backend = %s (use_gpu_faiss=%s, useFloat16=%s)",
                args.faiss_backend, use_gpu_faiss_in_eval, use_gpu_fp16)

    # 3) resume previous state
    state = _load_existing(results_out)
    if "zero_shot" not in state:
        state["zero_shot"] = {}

    def _eval_one(label, ckpt_path):
        tokenizer, encoder = spec["loader"](ckpt_path)
        with torch.no_grad():
            metrics = eval_conv_search(
                query_encoder       = encoder,
                tokenizer           = tokenizer,
                # Use the raw-conversation reconstruction (Truth_rewrite history
                # stripped out, replaced with literal raw [Q_i, A_i] turns).
                test_data_file      = qrecc_test_file,
                qrel_file           = QRECC_QREL,
                faiss_index         = faiss_index,
                doc_ids             = doc_ids,
                device              = device,
                eval_batch_size     = args.eval_bs,
                # Truncation budget pulled from setting_spec so ANCE keeps its
                # 512-position cap while Qwen runs effectively un-truncated.
                max_query_length    = spec["max_query_length"],
                max_response_length = spec["max_response_length"],
                max_concat_length   = spec["max_concat_length"],
                use_gpu_faiss       = use_gpu_faiss_in_eval,
                use_gpu_fp16        = use_gpu_fp16,
                keep_faiss_on_gpu   = use_gpu_faiss_in_eval,
                gpu_index_cache     = gpu_index_cache,
                full_eval           = False,         # only NDCG@10 needed per epoch
                left_padding        = spec["left_padding"],
                dataset_tag         = "qrecc",
                conv_instruction    = spec["conv_instruction"],
                template_version    = spec["template_version"],
            )
        ndcg = float(metrics["NDCG@10"])
        del encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return ndcg

    # 4) zero-shot baseline
    if args.include_zero_shot:
        zs_key = "zero_shot"
        if str(0) in state.get(zs_key, {}):
            logger.info("[skip] zero-shot already computed: NDCG@10=%.4f",
                        state[zs_key][str(0)])
        else:
            logger.info("\n=== zero-shot %s ===", spec["zero_shot_ckpt"])
            ndcg = _eval_one("zero_shot", spec["zero_shot_ckpt"])
            state.setdefault(zs_key, {})[str(0)] = ndcg
            _save(results_out, state)
            logger.info("[ok ] zero-shot QReCC NDCG@10=%.4f", ndcg)

    # 5) main grid: 8 runs × 20 epoch-ckpts
    for run_name in runs:
        state.setdefault(run_name, {})
        ckpt_steps = args.ckpt_steps if args.ckpt_steps is not None else discover_ckpt_steps(run_name)
        if not ckpt_steps:
            logger.warning("No ckpts found for %s — skipping run", run_name)
            continue
        logger.info("[%s] will evaluate %d ckpts (steps %d..%d)",
                    run_name, len(ckpt_steps), ckpt_steps[0], ckpt_steps[-1])
        for step in ckpt_steps:
            sk = str(step)
            if sk in state[run_name]:
                logger.info("[skip] %s step-%d already=%.4f",
                            run_name, step, state[run_name][sk])
                continue
            ckpt_path = os.path.join(CKPT_BASE, run_name, f"checkpoint-step-{step}")
            if not os.path.isdir(ckpt_path):
                logger.warning("Missing: %s — skipping", ckpt_path)
                continue
            logger.info("\n--- %s step-%d ---", run_name, step)
            ndcg = _eval_one(f"{run_name}:step-{step}", ckpt_path)
            state[run_name][sk] = ndcg
            _save(results_out, state)
            logger.info("[ok ] %s step-%d: QReCC NDCG@10=%.4f", run_name, step, ndcg)

    logger.info("\nResults saved to %s", results_out)

    # compact summary
    print("\n" + "=" * 100)
    print(f"{'run':<42} {'zero-shot':>12} {'step→ndcg@10 (per epoch)':>40}")
    print("-" * 100)
    if args.include_zero_shot:
        zs = state.get("zero_shot", {}).get("0", float("nan"))
        print(f"{'zero_shot ('+args.setting+')':<42} {zs:>12.4f}")
    for run_name in runs:
        if run_name not in state:
            continue
        steps_for_run = args.ckpt_steps if args.ckpt_steps is not None \
                        else discover_ckpt_steps(run_name)
        series = " ".join(f"{state[run_name].get(str(s), float('nan')):.3f}"
                          for s in steps_for_run)
        print(f"{run_name:<42} {'':<12} {series}")
    print("=" * 100)


if __name__ == "__main__":
    main()
