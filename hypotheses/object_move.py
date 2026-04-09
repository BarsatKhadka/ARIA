"""
object_move.py — Hypotheses: move objects by a constant offset or with gravity.

Signals used:
  - DifferentialReport.movement_vector is RELEVANT (constant offset)
  - DifferentialReport.movement_direction is RELEVANT (gravity direction)

Two hypothesis types:
  1. Constant offset: every non-background cell shifts by (dr, dc).
  2. Gravity: every non-background object slides in a direction until it hits
     the border or another object.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from core.grid import Grid
from differential.feature_classifier import DifferentialReport, Confidence
from hypotheses.base import Hypothesis
from perception.objects import extract_objects
from perception.report import PerceptionReport


# ── Apply helpers ─────────────────────────────────────────────────────────────

def _apply_offset(grid: Grid, bg: int, dr: int, dc: int) -> Grid:
    """Shift every non-background cell by (dr, dc); vacated cells become bg."""
    new_data = np.full_like(grid.data, bg)
    rows, cols = grid.rows, grid.cols
    for r in range(rows):
        for c in range(cols):
            color = int(grid.data[r, c])
            if color == bg:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_data[nr, nc] = color
    return Grid(new_data.tolist())


def _apply_gravity(grid: Grid, bg: int, dr: int, dc: int) -> Grid:
    """
    Slide every non-background object in direction (dr, dc) until blocked by
    the border or another object (already placed in the output).

    Objects are processed from the leading edge first so they don't collide
    with each other during placement.
    """
    objects = extract_objects(grid, background=bg)
    if not objects:
        return grid

    # Sort objects: process the one closest to the destination edge first
    if dr > 0:      # moving down → process bottom-most first
        objects = sorted(objects, key=lambda o: -o.centroid[0])
    elif dr < 0:    # moving up
        objects = sorted(objects, key=lambda o:  o.centroid[0])
    elif dc > 0:    # moving right
        objects = sorted(objects, key=lambda o: -o.centroid[1])
    else:           # moving left
        objects = sorted(objects, key=lambda o:  o.centroid[1])

    rows, cols = grid.rows, grid.cols
    new_data = np.full_like(grid.data, bg)

    for obj in objects:
        pixels = set(obj.pixels)
        # Slide until blocked
        while True:
            next_pixels = {(r + dr, c + dc) for r, c in pixels}
            # Out of bounds?
            if any(nr < 0 or nr >= rows or nc < 0 or nc >= cols
                   for nr, nc in next_pixels):
                break
            # Blocked by already-placed object?
            if any(new_data[nr, nc] != bg for nr, nc in next_pixels):
                break
            pixels = next_pixels
        for r, c in pixels:
            new_data[r, c] = obj.dominant_color

    return Grid(new_data.tolist())


# ── Direction helpers ──────────────────────────────────────────────────────────

_DIRECTION_VECTORS: dict = {
    "up":           (-1,  0),
    "down":         ( 1,  0),
    "left":         ( 0, -1),
    "right":        ( 0,  1),
    "diagonal":     ( 1,  1),   # generic fallback; real diagonal is per-pair
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    perception: PerceptionReport,
    differential: DifferentialReport,
) -> List[Hypothesis]:
    """
    Generate movement hypotheses.

    If movement_vector is RELEVANT → constant-offset hypothesis.
    If movement_direction is RELEVANT → gravity hypothesis.
    Both may fire simultaneously (gravity is more robust than exact offset).
    """
    bg = (
        perception.pair_perceptions[0].input.background
        if perception.pair_perceptions
        else 0
    )

    hypotheses: List[Hypothesis] = []

    # ── Constant-offset hypothesis ─────────────────────────────────────────────
    mv = differential.movement_vector
    if mv.confidence == Confidence.RELEVANT and mv.value and len(mv.value) == 1:
        dr_f, dc_f = next(iter(mv.value))
        dr, dc = int(round(dr_f)), int(round(dc_f))
        if dr != 0 or dc != 0:
            hypotheses.append(Hypothesis(
                name=f"move_offset({dr:+d},{dc:+d})",
                apply=lambda g, _bg=bg, _dr=dr, _dc=dc: _apply_offset(g, _bg, _dr, _dc),
                complexity=2,
            ))

    # ── Gravity hypothesis ─────────────────────────────────────────────────────
    md = differential.movement_direction
    if md.confidence == Confidence.RELEVANT and md.value in _DIRECTION_VECTORS:
        direction = md.value
        dr, dc = _DIRECTION_VECTORS[direction]
        hypotheses.append(Hypothesis(
            name=f"gravity_{direction}",
            apply=lambda g, _bg=bg, _dr=dr, _dc=dc: _apply_gravity(g, _bg, _dr, _dc),
            complexity=3,
        ))

    return hypotheses
