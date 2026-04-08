from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

from core.grid import Grid
from core.task import Task


# ── Per-pair diff ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairDiff:
    # ── Dimensions ────────────────────────────────────────────────────────────
    input_shape:   Tuple[int, int]
    output_shape:  Tuple[int, int]
    shape_changed: bool
    rows_delta:    int               # output_rows - input_rows
    cols_delta:    int               # output_cols - input_cols
    scale_rows:    float             # output_rows / input_rows
    scale_cols:    float             # output_cols / input_cols

    # ── Colors ────────────────────────────────────────────────────────────────
    input_colors:   FrozenSet[int]   # all colors present in input
    output_colors:  FrozenSet[int]   # all colors present in output
    colors_added:   FrozenSet[int]   # in output but not input
    colors_removed: FrozenSet[int]   # in input but not output
    colors_kept:    FrozenSet[int]   # in both

    # ── Cell-level change ─────────────────────────────────────────────────────
    # None when shapes differ (can't compare cell-by-cell)
    change_fraction: Optional[float]  # fraction of cells whose value differs
    cells_changed:   Optional[int]    # raw count of changed cells
    total_cells:     int              # input rows × cols

    # ── Background ────────────────────────────────────────────────────────────
    background: int                  # most frequent color in the input


# ── Cross-pair task diff ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskDiff:
    pairs:   Tuple[PairDiff, ...]
    n_pairs: int

    # ── Dimension invariants ──────────────────────────────────────────────────
    shape_always_changes:  bool      # every pair has a shape change
    shape_never_changes:   bool      # no pair has a shape change

    rows_delta_consistent: bool      # every pair has the same rows_delta
    cols_delta_consistent: bool      # every pair has the same cols_delta
    scale_rows_consistent: bool      # every pair has the same scale_rows (±1e-6)
    scale_cols_consistent: bool      # every pair has the same scale_cols (±1e-6)

    consistent_rows_delta: Optional[int]    # the shared value, or None
    consistent_cols_delta: Optional[int]
    consistent_scale_rows: Optional[float]
    consistent_scale_cols: Optional[float]

    # ── Color invariants ──────────────────────────────────────────────────────
    colors_always_added:      FrozenSet[int]  # added in every pair
    colors_always_removed:    FrozenSet[int]  # removed in every pair
    colors_always_kept:       FrozenSet[int]  # kept in every pair
    colors_sometimes_added:   FrozenSet[int]  # added in at least one pair
    colors_sometimes_removed: FrozenSet[int]  # removed in at least one pair

    # ── Change-fraction invariants (same-shape pairs only) ────────────────────
    comparable_pairs:           int            # pairs where shapes match
    change_fraction_min:        Optional[float]
    change_fraction_max:        Optional[float]
    change_fraction_avg:        Optional[float]
    change_fraction_range:      Optional[float]  # max - min
    change_fraction_consistent: bool             # range < 0.05

    # ── Background ────────────────────────────────────────────────────────────
    background_consistent: bool
    background:            Optional[int]  # shared value if consistent, else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _most_frequent(grid: Grid) -> int:
    counts: Counter[int] = Counter()
    for r in range(grid.rows):
        for c in range(grid.cols):
            counts[grid.get(r, c)] += 1
    return counts.most_common(1)[0][0]


def _all_same(values: list, tol: float = 0.0) -> bool:
    if not values:
        return True
    if tol == 0.0:
        return all(v == values[0] for v in values)
    return max(values) - min(values) <= tol


# ── Public: single pair ───────────────────────────────────────────────────────

def diff_pair(inp: Grid, out: Grid) -> PairDiff:
    """
    Compute the difference between one input/output training pair.

    Parameters
    ----------
    inp, out:
        The input and output :class:`~core.grid.Grid` of a single training pair.

    Returns
    -------
    :class:`PairDiff`
    """
    in_r,  in_c  = inp.shape
    out_r, out_c = out.shape

    in_colors  = frozenset(int(c) for c in np.unique(inp.data))
    out_colors = frozenset(int(c) for c in np.unique(out.data))

    # Cell-level change — only possible when shapes match
    change_fraction: Optional[float] = None
    cells_changed:   Optional[int]   = None
    if inp.shape == out.shape:
        changed      = int(np.sum(inp.data != out.data))
        cells_changed   = changed
        change_fraction = changed / (in_r * in_c)

    return PairDiff(
        input_shape=inp.shape,
        output_shape=out.shape,
        shape_changed=inp.shape != out.shape,
        rows_delta=out_r - in_r,
        cols_delta=out_c - in_c,
        scale_rows=out_r / in_r,
        scale_cols=out_c / in_c,
        input_colors=in_colors,
        output_colors=out_colors,
        colors_added=out_colors - in_colors,
        colors_removed=in_colors - out_colors,
        colors_kept=in_colors & out_colors,
        change_fraction=change_fraction,
        cells_changed=cells_changed,
        total_cells=in_r * in_c,
        background=_most_frequent(inp),
    )


# ── Public: full task ─────────────────────────────────────────────────────────

def diff_task(task: Task) -> TaskDiff:
    """
    Compute cross-pair invariants for all training pairs in *task*.

    Runs :func:`diff_pair` on every training pair, then analyses what is
    consistent (invariant) versus variable (incidental) across pairs.

    Parameters
    ----------
    task:
        A :class:`~core.task.Task` with at least one training pair.

    Returns
    -------
    :class:`TaskDiff`
    """
    pairs = tuple(diff_pair(inp, out) for inp, out in task.train_pairs)
    n = len(pairs)

    # ── Dimension invariants ──────────────────────────────────────────────────
    shape_always_changes = all(p.shape_changed for p in pairs)
    shape_never_changes  = not any(p.shape_changed for p in pairs)

    rows_deltas  = [p.rows_delta  for p in pairs]
    cols_deltas  = [p.cols_delta  for p in pairs]
    scale_rows_v = [p.scale_rows  for p in pairs]
    scale_cols_v = [p.scale_cols  for p in pairs]

    rows_delta_consistent = _all_same(rows_deltas)
    cols_delta_consistent = _all_same(cols_deltas)
    scale_rows_consistent = _all_same(scale_rows_v, tol=1e-6)
    scale_cols_consistent = _all_same(scale_cols_v, tol=1e-6)

    consistent_rows_delta  = rows_deltas[0]  if rows_delta_consistent  else None
    consistent_cols_delta  = cols_deltas[0]  if cols_delta_consistent  else None
    consistent_scale_rows  = scale_rows_v[0] if scale_rows_consistent  else None
    consistent_scale_cols  = scale_cols_v[0] if scale_cols_consistent  else None

    # ── Color invariants ──────────────────────────────────────────────────────
    # Intersection = present in ALL pairs; union = present in ANY pair
    colors_always_added      = frozenset.intersection(*(p.colors_added    for p in pairs)) if n else frozenset()
    colors_always_removed    = frozenset.intersection(*(p.colors_removed  for p in pairs)) if n else frozenset()
    colors_always_kept       = frozenset.intersection(*(p.colors_kept     for p in pairs)) if n else frozenset()
    colors_sometimes_added   = frozenset.union(*(p.colors_added    for p in pairs)) if n else frozenset()
    colors_sometimes_removed = frozenset.union(*(p.colors_removed  for p in pairs)) if n else frozenset()

    # ── Change-fraction invariants ────────────────────────────────────────────
    fractions = [p.change_fraction for p in pairs if p.change_fraction is not None]
    comparable_pairs = len(fractions)

    if fractions:
        cf_min   = min(fractions)
        cf_max   = max(fractions)
        cf_avg   = sum(fractions) / len(fractions)
        cf_range = cf_max - cf_min
        cf_consistent = cf_range < 0.05
    else:
        cf_min = cf_max = cf_avg = cf_range = None
        cf_consistent = False

    # ── Background ────────────────────────────────────────────────────────────
    backgrounds = [p.background for p in pairs]
    bg_consistent = _all_same(backgrounds)
    background    = backgrounds[0] if bg_consistent else None

    return TaskDiff(
        pairs=pairs,
        n_pairs=n,
        shape_always_changes=shape_always_changes,
        shape_never_changes=shape_never_changes,
        rows_delta_consistent=rows_delta_consistent,
        cols_delta_consistent=cols_delta_consistent,
        scale_rows_consistent=scale_rows_consistent,
        scale_cols_consistent=scale_cols_consistent,
        consistent_rows_delta=consistent_rows_delta,
        consistent_cols_delta=consistent_cols_delta,
        consistent_scale_rows=consistent_scale_rows,
        consistent_scale_cols=consistent_scale_cols,
        colors_always_added=colors_always_added,
        colors_always_removed=colors_always_removed,
        colors_always_kept=colors_always_kept,
        colors_sometimes_added=colors_sometimes_added,
        colors_sometimes_removed=colors_sometimes_removed,
        comparable_pairs=comparable_pairs,
        change_fraction_min=cf_min,
        change_fraction_max=cf_max,
        change_fraction_avg=cf_avg,
        change_fraction_range=cf_range,
        change_fraction_consistent=cf_consistent,
        background_consistent=bg_consistent,
        background=background,
    )
