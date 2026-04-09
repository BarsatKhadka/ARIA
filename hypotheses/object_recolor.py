"""
object_recolor.py — Hypotheses: select objects by predicate and recolor them.

Signal used: DifferentialReport.color_mapping is RELEVANT + various selection
features (selection_is_largest, selected_colors, etc.).

Generates one hypothesis per selection predicate. The executor filters down
to whichever predicates actually reproduce the training outputs.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Set, Tuple

import numpy as np

from core.grid import Grid
from differential.feature_classifier import DifferentialReport, Confidence
from hypotheses.base import Hypothesis
from perception.objects import Object, extract_objects
from perception.report import PerceptionReport
from perception.topology import relate_all


# ── Internal helpers ──────────────────────────────────────────────────────────

def _touches_border(obj: Object, rows: int, cols: int) -> bool:
    return any(
        r == 0 or r == rows - 1 or c == 0 or c == cols - 1
        for r, c in obj.pixels
    )


def _enclosed_ids(objects: List[Object], grid: Grid, bg: int) -> Set[int]:
    if len(objects) < 2:
        return set()
    rels = relate_all(objects, grid, background=bg)
    return {id(r.b) for r in rels if r.a_encloses_b}


def _apply_recolor(
    grid: Grid,
    bg: int,
    selector: Callable[[List[Object], Grid, int], List[Object]],
    mapping: Dict[int, int],
) -> Grid:
    objects = extract_objects(grid, background=bg)
    if not objects:
        return grid
    selected = selector(objects, grid, bg)
    if not selected:
        return grid
    new_data = grid.data.copy()
    for obj in selected:
        to_c = mapping.get(obj.dominant_color)
        if to_c is not None:
            for r, c in obj.pixels:
                new_data[r, c] = to_c
    return Grid(new_data.tolist())


# ── Selector functions (named so Hypothesis.name is readable) ─────────────────

def _sel_all(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    return objects


def _sel_largest(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    if not objects:
        return []
    max_sz = max(o.size for o in objects)
    return [o for o in objects if o.size == max_sz]


def _sel_smallest(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    if not objects:
        return []
    min_sz = min(o.size for o in objects)
    return [o for o in objects if o.size == min_sz]


def _sel_border(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    return [o for o in objects if _touches_border(o, grid.rows, grid.cols)]


def _sel_interior(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    return [o for o in objects if not _touches_border(o, grid.rows, grid.cols)]


def _sel_enclosed(objects: List[Object], grid: Grid, bg: int) -> List[Object]:
    enc = _enclosed_ids(objects, grid, bg)
    return [o for o in objects if id(o) in enc]


def _make_color_selector(
    colors: FrozenSet[int],
) -> Callable[[List[Object], Grid, int], List[Object]]:
    def sel(objects: List[Object], grid: Grid, bg: int,
            _c: FrozenSet[int] = colors) -> List[Object]:
        return [o for o in objects if o.dominant_color in _c]
    return sel


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    perception: PerceptionReport,
    differential: DifferentialReport,
) -> List[Hypothesis]:
    """
    Generate object-recolor hypotheses for each selection predicate.

    Requires a RELEVANT color_mapping in the differential. Generates hypotheses
    for: all, largest, smallest, border-touching, interior, enclosed, and
    (if available) the color-filter predicate from selected_colors.
    """
    cm = differential.color_mapping
    if cm.confidence != Confidence.RELEVANT or not cm.value:
        return []

    mapping: Dict[int, int] = {int(f): int(t) for f, t in cm.value}
    if not mapping:
        return []

    # Determine background from perception
    bg = (
        perception.pair_perceptions[0].input.background
        if perception.pair_perceptions
        else 0
    )

    hypotheses: List[Hypothesis] = []

    # ── Predicate: recolor ALL objects ────────────────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_all({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_all, _m),
        complexity=1,
    ))

    # ── Predicate: largest ────────────────────────────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_largest({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_largest, _m),
        complexity=2,
    ))

    # ── Predicate: smallest ───────────────────────────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_smallest({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_smallest, _m),
        complexity=2,
    ))

    # ── Predicate: border-touching ────────────────────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_border_touching({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_border, _m),
        complexity=3,
    ))

    # ── Predicate: interior (not border-touching) ─────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_interior({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_interior, _m),
        complexity=3,
    ))

    # ── Predicate: enclosed ───────────────────────────────────────────────────
    hypotheses.append(Hypothesis(
        name=f"recolor_enclosed({mapping})",
        apply=lambda g, _bg=bg, _m=mapping: _apply_recolor(g, _bg, _sel_enclosed, _m),
        complexity=3,
    ))

    # ── Predicate: specific color filter (if differential knows which colors) ──
    sc = differential.selected_colors
    if sc.confidence == Confidence.RELEVANT and sc.value:
        colors: FrozenSet[int] = sc.value
        sel_fn = _make_color_selector(colors)
        hypotheses.append(Hypothesis(
            name=f"recolor_color_filter({set(colors)} → {mapping})",
            apply=lambda g, _bg=bg, _sel=sel_fn, _m=mapping: _apply_recolor(g, _bg, _sel, _m),
            complexity=2,
        ))

    return hypotheses
