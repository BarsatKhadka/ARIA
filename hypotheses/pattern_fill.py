"""
pattern_fill.py — Hypotheses: fill regions with a pattern or scale the grid.

Three hypothesis families:
  1. Pixel zoom (upscale): each cell becomes an N×N block. Fires when output
     is exactly N× input in both dimensions.
  2. Tile: output is the input tiled N×M times. Fires when output dimensions
     are integer multiples of input dimensions.
  3. Flood fill: fill the largest connected background region in the input with
     each color that appears in training outputs. Fires for same-shape pairs.

Signals used:
  - PairDiff.scale_rows / scale_cols (from DifferentialReport.pair_analyses)
  - DifferentialReport.pair_analyses[*].pair_diff.colors_added
  - PerceptionReport for background detection
"""
from __future__ import annotations

from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np

from core.grid import Grid
from differential.feature_classifier import DifferentialReport, Confidence
from hypotheses.base import Hypothesis
from perception.report import PerceptionReport


# ── Apply helpers ─────────────────────────────────────────────────────────────

def _apply_pixel_zoom(grid: Grid, scale: int) -> Grid:
    """Each cell becomes a scale×scale block."""
    new_data = np.repeat(np.repeat(grid.data, scale, axis=0), scale, axis=1)
    return Grid(new_data.tolist())


def _apply_tile(grid: Grid, reps_r: int, reps_c: int) -> Grid:
    """Tile the grid reps_r × reps_c times."""
    new_data = np.tile(grid.data, (reps_r, reps_c))
    return Grid(new_data.tolist())


def _flood_fill_bg(
    grid: Grid,
    bg: int,
) -> List[Set[Tuple[int, int]]]:
    """
    Return all connected background regions (4-connectivity) as a list of cell sets,
    sorted largest first.
    """
    rows, cols = grid.rows, grid.cols
    visited: Set[Tuple[int, int]] = set()
    regions: List[Set[Tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid.data[r, c] == bg and (r, c) not in visited:
                region: Set[Tuple[int, int]] = set()
                queue: deque = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    region.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and (nr, nc) not in visited
                                and grid.data[nr, nc] == bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                regions.append(region)

    regions.sort(key=len, reverse=True)
    return regions


def _apply_flood_fill(grid: Grid, bg: int, fill_color: int) -> Grid:
    """Fill the largest background-connected region with fill_color."""
    regions = _flood_fill_bg(grid, bg)
    if not regions:
        return grid
    largest = regions[0]
    new_data = grid.data.copy()
    for r, c in largest:
        new_data[r, c] = fill_color
    return Grid(new_data.tolist())


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    perception: PerceptionReport,
    differential: DifferentialReport,
) -> List[Hypothesis]:
    """
    Generate pattern-fill hypotheses based on shape-change and color signals.
    """
    if not perception.pair_perceptions:
        return []

    bg = perception.pair_perceptions[0].input.background
    hypotheses: List[Hypothesis] = []

    # Collect per-pair scale and color info from the differential
    pair_analyses = differential.pair_analyses

    # ── Scale/tile hypotheses (shape-change pairs) ────────────────────────────
    if pair_analyses:
        scale_rows_vals = [pa.pair_diff.scale_rows for pa in pair_analyses]
        scale_cols_vals = [pa.pair_diff.scale_cols for pa in pair_analyses]

        # Check for consistent integer scale
        def _all_close(vals, target) -> bool:
            return all(abs(v - target) < 1e-6 for v in vals)

        first_sr = scale_rows_vals[0]
        first_sc = scale_cols_vals[0]
        consistent_scale = (
            _all_close(scale_rows_vals, first_sr)
            and _all_close(scale_cols_vals, first_sc)
        )

        if consistent_scale and first_sr > 1.0 and first_sc > 1.0:
            sr_int = int(round(first_sr))
            sc_int = int(round(first_sc))
            sr_exact = abs(first_sr - sr_int) < 1e-6
            sc_exact = abs(first_sc - sc_int) < 1e-6

            # ── Pixel zoom (same integer scale in both dims) ──────────────────
            if sr_exact and sc_exact and sr_int == sc_int and sr_int >= 2:
                scale = sr_int
                hypotheses.append(Hypothesis(
                    name=f"pixel_zoom(×{scale})",
                    apply=lambda g, _s=scale: _apply_pixel_zoom(g, _s),
                    complexity=2,
                ))

            # ── Tile (integer repeats in each dim) ────────────────────────────
            if sr_exact and sc_exact and sr_int >= 1 and sc_int >= 1:
                rr, rc = sr_int, sc_int
                hypotheses.append(Hypothesis(
                    name=f"tile(×{rr}r×{rc}c)",
                    apply=lambda g, _r=rr, _c=rc: _apply_tile(g, _r, _c),
                    complexity=2,
                ))

    # ── Flood-fill hypotheses (same-shape pairs) ──────────────────────────────
    # Collect all colors that were added in output but not present in input,
    # across all training pairs, as candidate fill colors.
    fill_colors: Set[int] = set()
    shape_change_pairs = 0
    for pa in pair_analyses:
        if pa.pair_diff.shape_changed:
            shape_change_pairs += 1
            continue
        for c in pa.pair_diff.colors_added:
            if c != bg:
                fill_colors.add(int(c))

    # Only generate flood-fill when most pairs have the same shape
    same_shape_pairs = len(pair_analyses) - shape_change_pairs
    if same_shape_pairs > 0 and fill_colors:
        for fill_color in sorted(fill_colors):
            hypotheses.append(Hypothesis(
                name=f"flood_fill(bg={bg} → {fill_color})",
                apply=lambda g, _bg=bg, _fc=fill_color: _apply_flood_fill(g, _bg, _fc),
                complexity=3,
            ))

    return hypotheses
