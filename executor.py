"""
executor.py — Run a hypothesis against training pairs and score it.

Takes a Hypothesis and a list of (input, expected_output) training pairs.
Applies the hypothesis to each input, compares with the expected output,
and returns an ExecutionResult with per-pair and aggregate scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.grid import Grid
from hypotheses.base import Hypothesis


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairResult:
    pair_index: int
    exact:      bool    # predicted grid == expected grid (shape + every cell)
    score:      float   # fraction of cells correct (0.0 when shapes differ)


@dataclass(frozen=True)
class ExecutionResult:
    hypothesis:   Hypothesis
    pair_results: Tuple[PairResult, ...]

    all_pass:   bool    # every training pair is an exact match
    pass_count: int     # number of pairs that are exact matches
    avg_score:  float   # mean cell-accuracy across all pairs


# ── Core executor ─────────────────────────────────────────────────────────────

def execute(
    hypothesis: Hypothesis,
    train_pairs: List[Tuple[Grid, Grid]],
) -> ExecutionResult:
    """
    Apply *hypothesis* to every training input and compare with the expected output.

    Parameters
    ----------
    hypothesis:
        The candidate rule to test.
    train_pairs:
        List of ``(input_grid, expected_output_grid)`` training examples.

    Returns
    -------
    :class:`ExecutionResult` with per-pair and aggregate scores.
    """
    pair_results: List[PairResult] = []

    for i, (inp, expected) in enumerate(train_pairs):
        try:
            predicted = hypothesis.apply(inp)
        except Exception:
            pair_results.append(PairResult(pair_index=i, exact=False, score=0.0))
            continue

        if predicted.shape != expected.shape:
            pair_results.append(PairResult(pair_index=i, exact=False, score=0.0))
            continue

        total = expected.rows * expected.cols
        correct = int((predicted.data == expected.data).sum())
        exact = correct == total
        score = correct / total if total > 0 else 0.0

        pair_results.append(PairResult(pair_index=i, exact=exact, score=score))

    all_pass = bool(pair_results) and all(r.exact for r in pair_results)
    pass_count = sum(1 for r in pair_results if r.exact)
    avg_score = (
        sum(r.score for r in pair_results) / len(pair_results)
        if pair_results
        else 0.0
    )

    return ExecutionResult(
        hypothesis=hypothesis,
        pair_results=tuple(pair_results),
        all_pass=all_pass,
        pass_count=pass_count,
        avg_score=avg_score,
    )
