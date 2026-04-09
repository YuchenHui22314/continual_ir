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


PACING_FUNCTIONS = {
    "root_2":    pacing_root_2,     # DCL 2208 "positive pacing" = transformers_cl root_2
    "root_5":    pacing_root_5,     # slower warm-up, also recommended in 1912
    "linear":    pacing_linear,     # uniform linear increase
    "step":      pacing_step,       # coarse 3-stage step
    "standard":  pacing_standard,   # no curriculum (baseline)
}


# ============================================================
# Helper: compute the current pacing value
# ============================================================

def get_pacing_value(
    global_step:      int,
    curriculum_steps: int,
    c0:               float,
    pacing_fn_name:   str = "root_2",
) -> float:
    """
    Compute the fraction of the sorted training dataset to expose at the
    current training step.

    Args:
        global_step:      Current training step (e.g., epoch * steps_per_epoch).
        curriculum_steps: Step at which the curriculum ends (full data exposed).
                          Computed as: curriculum_end_epoch * steps_per_epoch.
        c0:               Initial data fraction (delta_p). At step 0, this fraction
                          of data is used. Typically set based on the fraction of
                          "easy" examples in the training set (see analyze_topiocqa_turns.py).
        pacing_fn_name:   Key in PACING_FUNCTIONS. Default: "root_2".

    Returns:
        pacing_value: float in [c0, 1.0]. Fraction of (sorted) training examples to use.

    Example:
        # With c0=0.2 and root_2 pacing:
        # Step 0             → 0.20 (use 20% of data)
        # Step t/4           → ~0.53
        # Step t/2           → ~0.73
        # Step t (epoch 16)  → 1.00 (use full dataset)
        # Step > t           → 1.00 (capped)
    """
    if pacing_fn_name not in PACING_FUNCTIONS:
        raise ValueError(
            f"Unknown pacing function '{pacing_fn_name}'. "
            f"Available: {list(PACING_FUNCTIONS.keys())}"
        )
    if curriculum_steps <= 0:
        return 1.0  # degenerate case: no curriculum

    fn = PACING_FUNCTIONS[pacing_fn_name]
    # Cap x at curriculum_steps so f(x) never exceeds 1.0 by formula
    x = min(global_step, curriculum_steps)
    value = fn(float(x), float(curriculum_steps), float(c0))
    return float(min(1.0, max(c0, value)))  # clamp to [c0, 1.0]


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
    logger.info(f"  {'Epoch':>6}  {'Step':>8}  {'Pacing':>8}  {'Data %':>8}")
    for epoch in range(num_epochs):
        step = epoch * steps_per_epoch
        pv = get_pacing_value(step, curriculum_steps, c0, pacing_fn_name)
        logger.info(f"  {epoch:>6}  {step:>8}  {pv:>8.4f}  {pv*100:>7.2f}%")
    logger.info("=" * 60)
