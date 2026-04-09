"""
mirror.py — Hypotheses: reflect or rotate the input grid.

Generates six whole-grid transformation hypotheses unconditionally:
  flip_vertical     (left ↔ right)
  flip_horizontal   (top ↔ bottom)
  rotate_180
  rotate_90_cw
  rotate_90_ccw
  transpose         (rows ↔ cols)

And two "append mirror" hypotheses that produce a wider/taller grid:
  concat_h          input | flip_vertical(input)   side-by-side
  concat_v          input / flip_horizontal(input) stacked

These are always generated because they are cheap to apply and cover a wide
class of ARC tasks (symmetry completion, rotation tasks).
"""
from __future__ import annotations

from typing import List

import numpy as np

from core.grid import Grid
from differential.feature_classifier import DifferentialReport
from hypotheses.base import Hypothesis
from perception.report import PerceptionReport


# ── Apply helpers ─────────────────────────────────────────────────────────────

def _flip_v(grid: Grid) -> Grid:
    """Flip left ↔ right (vertical axis of reflection)."""
    return Grid(np.fliplr(grid.data).tolist())


def _flip_h(grid: Grid) -> Grid:
    """Flip top ↔ bottom (horizontal axis of reflection)."""
    return Grid(np.flipud(grid.data).tolist())


def _rot180(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.data, 2).tolist())


def _rot90_cw(grid: Grid) -> Grid:
    """Rotate 90° clockwise."""
    return Grid(np.rot90(grid.data, -1).tolist())


def _rot90_ccw(grid: Grid) -> Grid:
    """Rotate 90° counter-clockwise."""
    return Grid(np.rot90(grid.data, 1).tolist())


def _transpose(grid: Grid) -> Grid:
    return Grid(grid.data.T.tolist())


def _concat_h(grid: Grid) -> Grid:
    """Concatenate input and its left-right flip side by side."""
    flipped = np.fliplr(grid.data)
    return Grid(np.concatenate([grid.data, flipped], axis=1).tolist())


def _concat_v(grid: Grid) -> Grid:
    """Stack input and its top-bottom flip vertically."""
    flipped = np.flipud(grid.data)
    return Grid(np.concatenate([grid.data, flipped], axis=0).tolist())


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    perception: PerceptionReport,
    differential: DifferentialReport,
) -> List[Hypothesis]:
    """
    Return all mirror/rotation hypotheses (always generated; executor filters).
    """
    return [
        Hypothesis(name="flip_vertical",   apply=_flip_v,     complexity=2),
        Hypothesis(name="flip_horizontal",  apply=_flip_h,     complexity=2),
        Hypothesis(name="rotate_180",       apply=_rot180,     complexity=2),
        Hypothesis(name="rotate_90_cw",     apply=_rot90_cw,   complexity=2),
        Hypothesis(name="rotate_90_ccw",    apply=_rot90_ccw,  complexity=2),
        Hypothesis(name="transpose",        apply=_transpose,  complexity=2),
        Hypothesis(name="concat_mirror_h",  apply=_concat_h,   complexity=3),
        Hypothesis(name="concat_mirror_v",  apply=_concat_v,   complexity=3),
    ]
