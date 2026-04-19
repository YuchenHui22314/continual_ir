"""
curriculum.py

Curriculum learning utilities for continual IR training.

Provides two registries:
  - SCORING_FUNCTIONS: assign a numeric difficulty score to each training example.
  - PACING_FUNCTIONS:  given current step x, total curriculum steps t, and initial
                       fraction c0, return the fraction of the (sorted) dataset to
                       sample from at this step.

The pacing formulas follow:
  - root_2  : "positive pacing" from DCL (arXiv 2208.10226) = root_2 from
               transformers_cl (arXiv 1912.08555). Both papers show this works well.
  - root_5  : slower warm-up variant (also recommended in 1912 paper).
  - linear  : linear increase from c0 to 1.0 over curriculum_steps.
  - step    : coarse 3-stage step function (0.33 → 0.66 → 1.0).
  - standard: always return 1.0 (no curriculum, standard training). Useful for ablation.

All pacing functions satisfy:
  f(0, t, c0) = c0      (start with c0 fraction of data)
  f(t, t, c0) = 1.0     (use full dataset at end of curriculum phase)
  f(x, t, c0) clipped to [0, 1] when called via get_pacing_value()

Usage example (in training script):
    from curriculum import SCORING_FUNCTIONS, PACING_FUNCTIONS, get_pacing_value

    # Sort dataset once at startup
    train_dataset.sort_by_difficulty(SCORING_FUNCTIONS["turn_length"], ascending=True)

    # Per epoch: compute what fraction of data to expose
    pacing_value = get_pacing_value(
        global_step      = epoch * steps_per_epoch,
        curriculum_steps = curriculum_end_epoch * steps_per_epoch,
        c0               = args.curriculum_c0,
        pacing_fn_name   = args.pacing_function,
    )
    n_active = max(args.batch_size, int(pacing_value * len(train_dataset)))
    subset   = Subset(train_dataset, range(n_active))
"""

import math
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Scoring Functions
# Maps example dict → numeric difficulty score (higher = harder).
# ============================================================

def score_by_turn_length(example: dict) -> float:
    """
    Difficulty = conversation turn number.
    Turn 1 (no history, ctx_utts_text is empty) is the easiest.
    Longer conversations (more prior turns) are harder.

    turn_number is pre-computed in Topiocqa.__init__ as:
        1 + len(ctx_utts_text) // 2
    """
    return float(example["turn_number"])


SCORING_FUNCTIONS = {
    "turn_length": score_by_turn_length,
    # Add new scoring functions here for future experiments, e.g.:
    # "query_length": score_by_query_length,
    # "bm25_score":   score_by_bm25_difficulty,
}


# ============================================================
# Pacing Functions
# Signature: (x: int, t: int, c0: float) -> float
#   x  = current global step
#   t  = total curriculum steps (curriculum ends here)
#   c0 = initial data fraction (delta_p in DCL paper)
# ============================================================

def pacing_root_2(x: float, t: float, c0: float) -> float:
    """
    Root-2 (square-root) pacing.
    Identical to the "positive pacing function" in DCL (arXiv:2208.10226)
    and root_2 in transformers_cl (arXiv:1912.08555).
    Starts fast, then slows down as it approaches the full dataset.

        f(x) = ( x*(1-c0^2)/t + c0^2 )^(1/2)
    """
    return ((x * (1.0 - c0 ** 2.0) / t) + c0 ** 2.0) ** 0.5


def pacing_root_5(x: float, t: float, c0: float) -> float:
    """
    Root-5 pacing. Slower initial warm-up than root_2.
    Also recommended in transformers_cl (arXiv:1912.08555).

        f(x) = ( x*(1-c0^5)/t + c0^5 )^(1/5)
    """
    return ((x * (1.0 - c0 ** 5.0) / t) + c0 ** 5.0) ** (1.0 / 5.0)


def pacing_linear(x: float, t: float, c0: float) -> float:
    """
    Linear pacing. Increases fraction uniformly from c0 to 1.0.

        f(x) = x*(1-c0)/t + c0
    """
    return x * (1.0 - c0) / t + c0


def pacing_step(x: float, t: float, c0: float) -> float:
    """
    Coarse 3-stage step pacing: c0 → mid → 1.0 over the curriculum period.
    Breakpoints at 33% and 66% of curriculum_steps.
    """
    mid = c0 + (1.0 - c0) / 2.0
    if x <= t * 0.33:
        return c0
    elif x <= t * 0.66:
        return mid
    else:
        return 1.0


def pacing_standard(x: float, t: float, c0: float) -> float:
    """
    No curriculum (standard training). Always uses 100% of the data.
    Useful as ablation baseline: same code path, no data restriction.
    """
    return 1.0


def pacing_step_exclusive(x: float, t: float, c0: float):
    """
    Exclusive 3-stage step pacing — each stage trains on ONE slice only.

    Unlike cumulative `step` (0–c0 → 0–mid → 0–1.0), here each stage is
    an exclusive window.  Stage 3 continues until the end of training
    (no switch to full data after curriculum ends).

    Slices are index-based (relative to the sorted dataset), so the semantic
    meaning depends on sort direction:
      - easy2hard (ascending):  stage 1 = easiest,  stage 3 = hardest
      - hard2easy (descending): stage 1 = hardest,  stage 3 = easiest

        Stage 1  (x ≤ t×0.33):  [  0,  c0 )  — first c0 fraction of sorted data
        Stage 2  (x ≤ t×0.66):  [ c0, mid )  — middle slice
        Stage 3  (x >  t×0.66):  [mid, 1.0 )  — last slice (stays here)

    where  mid = c0 + (1 − c0) / 2

    With c0=0.2, end_epoch=16, total=20:
        epochs  0–5:   [0%,  20%)
        epochs  6–10:  [20%, 60%)
        epochs 11–19:  [60%, 100%)   ← includes post-curriculum epochs

    Returns:
        (start_frac, end_frac) tuple.
    """
    mid = c0 + (1.0 - c0) / 2.0
    if x <= t * 0.33:
        return (0.0, c0)           # stage 1: first slice of sorted data
    elif x <= t * 0.66:
        return (c0, mid)           # stage 2: middle slice
    else:
        return (mid, 1.0)          # stage 3: last slice (no full-data transition)


def pacing_step_exclusive_2_full(x: float, t: float, c0: float):
    """
    Exclusive 3-stage step pacing with a final full-data phase.

    Same 3 exclusive stages as step_exclusive, but after the curriculum
    ends (x ≥ t, i.e., epoch ≥ curriculum_end_epoch), switches to full
    data for consolidation.

    Slices are index-based (relative to the sorted dataset):
      - easy2hard (ascending):  stage 1 = easiest,  stage 3 = hardest
      - hard2easy (descending): stage 1 = hardest,  stage 3 = easiest

        Stage 1  (x ≤ t×0.33):  [  0,  c0 )  — first c0 fraction of sorted data
        Stage 2  (x ≤ t×0.66):  [ c0, mid )  — middle slice
        Stage 3  (x <  t     ):  [mid, 1.0 )  — last slice
        Full     (x ≥  t     ):  [  0, 1.0 )  — full data consolidation

    With c0=0.2, end_epoch=16, total=20:
        epochs  0–5:   [0%,  20%)
        epochs  6–10:  [20%, 60%)
        epochs 11–15:  [60%, 100%)
        epochs 16–19:  [0%,  100%)   ← full data consolidation

    Returns:
        (start_frac, end_frac) tuple.
    """
    mid = c0 + (1.0 - c0) / 2.0
    if x >= t:
        return (0.0, 1.0)          # curriculum over → full data
    elif x <= t * 0.33:
        return (0.0, c0)           # stage 1: easiest slice
    elif x <= t * 0.66:
        return (c0, mid)           # stage 2: middle slice
    else:
        return (mid, 1.0)          # stage 3: hardest slice


PACING_FUNCTIONS = {
    "root_2":                  pacing_root_2,                 # DCL 2208 / transformers_cl 1912
    "root_5":                  pacing_root_5,                 # slower warm-up variant
    "linear":                  pacing_linear,                 # uniform linear increase
    "step":                    pacing_step,                   # cumulative 3-stage step
    "step_exclusive":          pacing_step_exclusive,         # exclusive 3-stage (stage 3 till end)
    "step_exclusive_2_full":   pacing_step_exclusive_2_full,  # exclusive 3-stage + full-data phase
    "standard":                pacing_standard,               # no curriculum (baseline)
}


# ============================================================
# Helper: compute the current pacing value
# ============================================================

def get_pacing_value(
    global_step:      int,
    curriculum_steps: int,
    c0:               float,
    pacing_fn_name:   str = "root_2",
):
    """
    Compute which slice of the sorted training dataset to expose at the
    current training step.

    Args:
        global_step:      Current training step (e.g., epoch * steps_per_epoch).
        curriculum_steps: Step at which the curriculum ends (full data exposed).
                          Computed as: curriculum_end_epoch * steps_per_epoch.
        c0:               Initial data fraction (delta_p). At step 0, this fraction
                          of data is used. Typically set based on the fraction of
                          "easy" examples in the training set.
        pacing_fn_name:   Key in PACING_FUNCTIONS. Default: "root_2".

    Returns:
        For most pacing functions: float in [c0, 1.0] — use dataset[:end_frac].
        For step_exclusive:        (start_frac, end_frac) tuple — use dataset[start:end].
        Use is_exclusive_pacing() to distinguish, or handle both in the training loop.

    Example (root_2, c0=0.2):
        Step 0       → 0.20 (use first 20% of sorted data)
        Step t/2     → ~0.73
        Step t       → 1.00 (full dataset)

    Example (step_exclusive, c0=0.2, end_epoch=16, total=20):
        Epochs 0–5   → (0.0, 0.2)  — train only on easiest 20%
        Epochs 6–10  → (0.2, 0.6)  — train only on 20–60%
        Epochs 11+   → (0.6, 1.0)  — train only on hardest 40% (stays here, no full-data transition)

    Example (step_exclusive_2_full, c0=0.2, end_epoch=16, total=20):
        Epochs 0–5   → (0.0, 0.2)  — train only on easiest 20%
        Epochs 6–10  → (0.2, 0.6)  — train only on 20–60%
        Epochs 11–15 → (0.6, 1.0)  — train only on hardest 40%
        Epochs 16+   → (0.0, 1.0)  — full data (curriculum over)
    """
    if pacing_fn_name not in PACING_FUNCTIONS:
        raise ValueError(
            f"Unknown pacing function '{pacing_fn_name}'. "
            f"Available: {list(PACING_FUNCTIONS.keys())}"
        )
    if curriculum_steps <= 0:
        # Degenerate: no curriculum — return appropriate "full data" value.
        # Both exclusive pacing functions return tuples; all others return a scalar.
        _tuple_pacings = ("step_exclusive", "step_exclusive_2_full")
        return (0.0, 1.0) if pacing_fn_name in _tuple_pacings else 1.0

    fn = PACING_FUNCTIONS[pacing_fn_name]
    x = min(global_step, curriculum_steps)   # cap at t
    result = fn(float(x), float(curriculum_steps), float(c0))

    if isinstance(result, tuple):
        # Exclusive-range pacing: clamp both ends to [0, 1]
        start, end = result
        return (max(0.0, float(start)), min(1.0, float(end)))
    else:
        # Scalar pacing: clamp to [c0, 1.0]
        return float(min(1.0, max(c0, result)))


def log_pacing_schedule(
    num_epochs:       int,
    steps_per_epoch:  int,
    curriculum_steps: int,
    c0:               float,
    pacing_fn_name:   str = "root_2",
):
    """
    Log the pacing schedule for each epoch. Useful for debugging and transparency.
    Call this once at training startup when curriculum is enabled.
    """
    logger.info("=" * 60)
    logger.info(f"Curriculum pacing schedule ({pacing_fn_name}, c0={c0:.3f}):")
    logger.info(f"  curriculum_steps={curriculum_steps}, steps_per_epoch={steps_per_epoch}")
    logger.info(f"  {'Epoch':>6}  {'Step':>8}  {'Active data':>20}")
    for epoch in range(num_epochs):
        step = epoch * steps_per_epoch
        pv = get_pacing_value(step, curriculum_steps, c0, pacing_fn_name)
        if isinstance(pv, tuple):
            start, end = pv
            data_str = f"{start*100:.0f}%–{end*100:.0f}%  ({(end-start)*100:.0f}% of data)"
        else:
            data_str = f"0%–{pv*100:.1f}%  ({pv*100:.1f}% of data)"
        logger.info(f"  {epoch:>6}  {step:>8}  {data_str}")
    logger.info("=" * 60)
