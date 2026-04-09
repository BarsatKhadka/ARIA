"""
solver.py — Milestone 3 solver: perception + differential → hypotheses → executor.

Pipeline for each task:
  1. Run perception (all 6 modules on every training pair).
  2. Run differential (cross-pair feature classification).
  3. Generate candidate hypotheses from all 5 hypothesis agents.
  4. Execute each hypothesis against all training pairs.
  5. If any hypothesis passes all training pairs → apply to test inputs
     (prefer the simplest by MDL complexity; carry the second-simplest as
     attempt_2).
  6. If none pass → return the top two by average training score as
     attempt_1 / attempt_2, giving the evaluator the best partial match.
  7. Fallback: identity (copy input) + all-zeros when everything fails.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from core.grid import Grid
from core.task import Task
from differential.feature_classifier import classify
from executor import execute, ExecutionResult
from hypotheses.base import Hypothesis
import hypotheses.global_color_map as _global_color_map
import hypotheses.mirror          as _mirror
import hypotheses.object_move     as _object_move
import hypotheses.object_recolor  as _object_recolor
import hypotheses.pattern_fill    as _pattern_fill
from perception.report import perceive


# ── Hypothesis generators (ordered: cheapest signal first) ────────────────────

_GENERATORS = [
    _global_color_map,
    _object_recolor,
    _object_move,
    _pattern_fill,
    _mirror,
]


# ── Fallback grids ─────────────────────────────────────────────────────────────

def _zeros(inp: Grid) -> Grid:
    return Grid([[0] * inp.cols for _ in range(inp.rows)])


def _identity(inp: Grid) -> Grid:
    return Grid(inp.data.tolist())


# ── Core solver ────────────────────────────────────────────────────────────────

def solve(task: Task) -> List[Union[Grid, Tuple[Grid, Grid]]]:
    """
    Solve *task* and return one prediction per test input.

    Each element may be:
      - a single ``Grid``              → used as both attempt_1 and attempt_2
      - ``Tuple[Grid, Grid]``          → attempt_1, attempt_2

    The :class:`~core.evaluator.Evaluator` normalises either form.
    """
    # ── Step 1–2: perception + differential ───────────────────────────────────
    try:
        perception   = perceive(task)
        differential = classify(task)
    except Exception:
        return [_zeros(inp) for inp in task.test_inputs]

    # ── Step 3: collect hypotheses ────────────────────────────────────────────
    all_hypotheses: List[Hypothesis] = []
    for module in _GENERATORS:
        try:
            all_hypotheses.extend(module.generate(perception, differential))
        except Exception:
            pass

    if not all_hypotheses:
        return [_zeros(inp) for inp in task.test_inputs]

    # ── Step 4: execute against training pairs ────────────────────────────────
    results: List[ExecutionResult] = []
    for h in all_hypotheses:
        try:
            results.append(execute(h, task.train_pairs))
        except Exception:
            pass

    if not results:
        return [_zeros(inp) for inp in task.test_inputs]

    # ── Step 5: use passing hypotheses (all training pairs exact match) ────────
    passing = [r for r in results if r.all_pass]
    if passing:
        # MDL: prefer simpler hypothesis; carry the second-simplest as attempt_2
        passing.sort(key=lambda r: (r.hypothesis.complexity, r.hypothesis.name))
        best   = passing[0].hypothesis
        second = passing[1].hypothesis if len(passing) > 1 else best
        return [
            (best.apply(inp), second.apply(inp))
            for inp in task.test_inputs
        ]

    # ── Step 6: best two partial-match hypotheses ──────────────────────────────
    results.sort(key=lambda r: (-r.avg_score, r.hypothesis.complexity))
    attempt_1 = results[0].hypothesis
    attempt_2 = results[1].hypothesis if len(results) > 1 else attempt_1

    return [
        (attempt_1.apply(inp), attempt_2.apply(inp))
        for inp in task.test_inputs
    ]
