from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, List, Optional, Set, Tuple

from core.grid import Grid
from core.task import Task
from perception.objects import Object, extract_objects
from perception.topology import relate_all
from differential.grid_diff import PairDiff, diff_pair
from differential.object_alignment import PairAlignment, ObjectMatch, align_pair


# ── Confidence ────────────────────────────────────────────────────────────────

class Confidence(Enum):
    RELEVANT   = "RELEVANT"    # same in every pair — likely part of the rule
    INCIDENTAL = "INCIDENTAL"  # varies across pairs — task-specific data
    AMBIGUOUS  = "AMBIGUOUS"   # only one pair, cannot confirm either way


# ── Feature ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Feature:
    name:       str
    confidence: Confidence
    value:      Any                  # shared value when RELEVANT; None otherwise
    per_pair:   Tuple[Any, ...]      # raw observation from each training pair


def _classify(name: str, observations: List[Any]) -> Feature:
    """Classify a list of per-pair observations into a Feature."""
    n = len(observations)
    if n == 0:
        return Feature(name, Confidence.AMBIGUOUS, None, ())
    if n == 1:
        return Feature(name, Confidence.AMBIGUOUS, observations[0], tuple(observations))
    all_same = all(o == observations[0] for o in observations)
    confidence = Confidence.RELEVANT if all_same else Confidence.INCIDENTAL
    value = observations[0] if all_same else None
    return Feature(name, confidence, value, tuple(observations))


# ── Per-pair analysis ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairAnalysis:
    pair_index: int
    pair_diff:  PairDiff
    alignment:  PairAlignment          # full object-level correspondence

    # Color mapping — from survived objects that changed color
    color_transitions: FrozenSet[Tuple[int, int]]   # {(from, to), ...}

    # Input colors of all "selected" objects (those that changed in any way)
    colors_with_changes: FrozenSet[int]

    # Selection properties of the selected input objects
    # None when there are no selected objects
    selected_object_colors:      Optional[FrozenSet[int]]
    selected_is_largest:         Optional[bool]
    selected_is_smallest:        Optional[bool]
    selected_is_border_touching: Optional[bool]
    selected_is_enclosed:        Optional[bool]

    # What kinds of transformations occurred
    transform_types: FrozenSet[str]    # subset of {"recolor","move","resize","delete","create"}

    # Centroid displacement of moved objects
    movement_vectors: FrozenSet[Tuple[float, float]]  # {(Δrow, Δcol), ...}

    # Object counts
    n_input_objects:  int
    n_output_objects: int
    object_count_delta: int


# ── DifferentialReport ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DifferentialReport:
    n_pairs:       int
    pair_analyses: Tuple[PairAnalysis, ...]

    # Color mapping
    color_mapping:   Feature    # invariant {(from, to)} across pairs
    selected_colors: Feature    # input colors always involved in changes

    # Selection criterion
    selection_is_largest:         Feature
    selection_is_smallest:        Feature
    selection_is_border_touching: Feature
    selection_is_enclosed:        Feature

    # Transformation type
    transform_type: Feature     # invariant frozenset of type strings

    # Spatial pattern
    movement_vector:    Feature  # invariant {(Δrow, Δcol)}
    movement_direction: Feature  # "up"/"down"/"left"/"right"/"diagonal"/None

    # Object counts
    object_count_delta: Feature  # n_output_objects − n_input_objects


# ── Direction helper ──────────────────────────────────────────────────────────

_CARDINAL_THRESHOLD = 2.0

def _vec_to_direction(dr: float, dc: float) -> Optional[str]:
    if abs(dr) < 0.5 and abs(dc) < 0.5:
        return None
    if abs(dr) >= _CARDINAL_THRESHOLD * abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) >= _CARDINAL_THRESHOLD * abs(dr):
        return "left" if dc < 0 else "right"
    return "diagonal"


# ── Border / enclosure helpers ────────────────────────────────────────────────

def _touches_border(obj: Object, rows: int, cols: int) -> bool:
    return any(
        r == 0 or r == rows - 1 or c == 0 or c == cols - 1
        for r, c in obj.pixels
    )


def _enclosed_ids(all_objs: List[Object], inp: Grid) -> Set[int]:
    """Return id() of every object that is enclosed by another object."""
    if len(all_objs) < 2:
        return set()
    rels = relate_all(all_objs, inp)
    return {id(r.b) for r in rels if r.a_encloses_b}


# ── Selected-object extraction ────────────────────────────────────────────────

def _selected_input_objects(pa: PairAlignment) -> List[Object]:
    """
    The input objects that were "selected" by the transformation.

    Selected = recolored ∪ moved ∪ resized ∪ destroyed.
    Duplicates (an object both moved and recolored) are deduplicated by id.
    """
    seen: Set[int] = set()
    selected: List[Object] = []
    candidates = (
        [m.input_obj for m in pa.recolored]
        + [m.input_obj for m in pa.moved]
        + [m.input_obj for m in pa.resized]
        + list(pa.destroyed)
    )
    for obj in candidates:
        if id(obj) not in seen:
            seen.add(id(obj))
            selected.append(obj)
    return selected


# ── Per-pair computation ──────────────────────────────────────────────────────

def _analyze_pair(
    index: int,
    inp: Grid,
    out: Grid,
    pd: PairDiff,
    pa: PairAlignment,
) -> PairAnalysis:

    # ── Color transitions from alignment ──────────────────────────────────────
    transitions: FrozenSet[Tuple[int, int]] = frozenset(
        (m.change.color_from, m.change.color_to)
        for m in pa.recolored
    )

    # ── Selected input objects ────────────────────────────────────────────────
    selected = _selected_input_objects(pa)
    all_inp_objs = extract_objects(inp)

    colors_changed = frozenset(o.dominant_color for o in selected)

    # Selection criterion properties — only if we have selected objects
    sel_colors:   Optional[FrozenSet[int]] = None
    sel_largest:  Optional[bool]           = None
    sel_smallest: Optional[bool]           = None
    sel_border:   Optional[bool]           = None
    sel_enclosed: Optional[bool]           = None

    if selected and all_inp_objs:
        sizes = [o.size for o in all_inp_objs]
        max_sz, min_sz = max(sizes), min(sizes)

        sel_colors   = frozenset(o.dominant_color for o in selected)
        sel_largest  = all(o.size == max_sz for o in selected)
        sel_smallest = all(o.size == min_sz for o in selected)
        sel_border   = all(_touches_border(o, inp.rows, inp.cols) for o in selected)

        enc_ids = _enclosed_ids(all_inp_objs, inp)
        sel_enclosed = all(id(o) in enc_ids for o in selected)

    # ── Transformation types from alignment ───────────────────────────────────
    types: Set[str] = set()
    if pa.recolored:  types.add("recolor")
    if pa.moved:      types.add("move")
    if pa.resized:    types.add("resize")
    if pa.destroyed:  types.add("delete")
    if pa.created:    types.add("create")
    if not types:
        # Shape-changing pairs: infer from diff
        if pd.shape_changed:
            if pd.scale_rows > 1.0 or pd.scale_cols > 1.0:
                types.add("expand")
            elif pd.scale_rows < 1.0 or pd.scale_cols < 1.0:
                types.add("shrink")
        if not types:
            types.add("none")

    # ── Movement vectors from alignment ───────────────────────────────────────
    movement_vecs: FrozenSet[Tuple[float, float]] = frozenset(
        m.change.position_delta for m in pa.moved
    )

    return PairAnalysis(
        pair_index=index,
        pair_diff=pd,
        alignment=pa,
        color_transitions=transitions,
        colors_with_changes=colors_changed,
        selected_object_colors=sel_colors,
        selected_is_largest=sel_largest,
        selected_is_smallest=sel_smallest,
        selected_is_border_touching=sel_border,
        selected_is_enclosed=sel_enclosed,
        transform_types=frozenset(types),
        movement_vectors=movement_vecs,
        n_input_objects=len(all_inp_objs),
        n_output_objects=pa.n_survived + pa.n_created,
        object_count_delta=pa.n_created - pa.n_destroyed,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def classify(task: Task) -> DifferentialReport:
    """
    Classify features across all training pairs of *task*.

    Runs object alignment on every pair, then for each feature collects the
    per-pair observation and determines whether it is
    :attr:`Confidence.RELEVANT` (invariant), :attr:`Confidence.INCIDENTAL`
    (varies), or :attr:`Confidence.AMBIGUOUS` (single pair).

    Parameters
    ----------
    task:
        A :class:`~core.task.Task` with at least one training pair.

    Returns
    -------
    :class:`DifferentialReport`
    """
    analyses: List[PairAnalysis] = []
    for i, (inp, out) in enumerate(task.train_pairs):
        pd = diff_pair(inp, out)
        pa = align_pair(i, inp, out)
        analyses.append(_analyze_pair(i, inp, out, pd, pa))

    # ── Color mapping ─────────────────────────────────────────────────────────
    color_mapping   = _classify("color_mapping",   [a.color_transitions   for a in analyses])
    selected_colors = _classify("selected_colors", [a.colors_with_changes for a in analyses])

    # ── Selection criterion (only pairs with a selection) ─────────────────────
    sel_pairs = [a for a in analyses if a.selected_object_colors is not None]

    selection_is_largest = _classify(
        "selection_is_largest",
        [a.selected_is_largest for a in sel_pairs],
    )
    selection_is_smallest = _classify(
        "selection_is_smallest",
        [a.selected_is_smallest for a in sel_pairs],
    )
    selection_is_border_touching = _classify(
        "selection_is_border_touching",
        [a.selected_is_border_touching for a in sel_pairs],
    )
    selection_is_enclosed = _classify(
        "selection_is_enclosed",
        [a.selected_is_enclosed for a in sel_pairs],
    )

    # ── Transformation type ───────────────────────────────────────────────────
    transform_type = _classify(
        "transform_type",
        [a.transform_types for a in analyses],
    )

    # ── Spatial pattern ───────────────────────────────────────────────────────
    movement_vector = _classify(
        "movement_vector",
        [a.movement_vectors for a in analyses],
    )

    directions: List[Optional[str]] = []
    for a in analyses:
        if len(a.movement_vectors) == 1:
            dr, dc = next(iter(a.movement_vectors))
            directions.append(_vec_to_direction(dr, dc))
        elif len(a.movement_vectors) == 0:
            directions.append(None)
        else:
            directions.append("mixed")
    movement_direction = _classify("movement_direction", directions)

    # ── Object count ──────────────────────────────────────────────────────────
    object_count_delta = _classify(
        "object_count_delta",
        [a.object_count_delta for a in analyses],
    )

    return DifferentialReport(
        n_pairs=len(analyses),
        pair_analyses=tuple(analyses),
        color_mapping=color_mapping,
        selected_colors=selected_colors,
        selection_is_largest=selection_is_largest,
        selection_is_smallest=selection_is_smallest,
        selection_is_border_touching=selection_is_border_touching,
        selection_is_enclosed=selection_is_enclosed,
        transform_type=transform_type,
        movement_vector=movement_vector,
        movement_direction=movement_direction,
        object_count_delta=object_count_delta,
    )
