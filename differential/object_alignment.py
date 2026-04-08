from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.grid import Grid
from core.task import Task
from perception.objects import Object, extract_objects


# ── Tuning constants ──────────────────────────────────────────────────────────

# Weights for the composite match score (must sum to 1.0)
_W_IOU     = 0.35   # bounding-box overlap
_W_SHAPE   = 0.40   # shape-signature Jaccard (position-invariant)
_W_COLOR   = 0.25   # dominant-color agreement

# Below this score the assignment is discarded — objects treated as unmatched
_MIN_SCORE = 0.10


# ── Change record ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObjectChange:
    """What differed between a matched input and output object."""
    color_changed:  bool
    color_from:     int
    color_to:       int
    position_delta: Tuple[float, float]   # (Δrow, Δcol) of centroids
    size_delta:     int                   # output.size − input.size
    shape_changed:  bool                  # shape_signature differs
    iou:            float                 # bounding-box IoU of the matched pair
    shape_jaccard:  float                 # shape-signature Jaccard of the matched pair


# ── Match record ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObjectMatch:
    """One matched (input → output) object pair."""
    input_obj:  Object
    output_obj: Object
    score:      float         # composite match score in [0, 1]
    change:     ObjectChange


# ── Pair-level alignment ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairAlignment:
    """
    Complete object-level correspondence for one training pair.

    Attributes
    ----------
    pair_index:
        Zero-based index of this pair within the task.
    survived:
        Input objects successfully matched to output objects.
        Each entry records what changed (or didn't) between the two.
    created:
        Output objects that could not be matched to any input object.
    destroyed:
        Input objects that could not be matched to any output object.
    """
    pair_index: int
    survived:   Tuple[ObjectMatch, ...]
    created:    Tuple[Object, ...]
    destroyed:  Tuple[Object, ...]

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_survived(self) -> int:
        return len(self.survived)

    @property
    def n_created(self) -> int:
        return len(self.created)

    @property
    def n_destroyed(self) -> int:
        return len(self.destroyed)

    @property
    def recolored(self) -> Tuple[ObjectMatch, ...]:
        """Survived objects whose color changed."""
        return tuple(m for m in self.survived if m.change.color_changed)

    @property
    def moved(self) -> Tuple[ObjectMatch, ...]:
        """Survived objects whose centroid shifted by more than 0.5 cells."""
        return tuple(
            m for m in self.survived
            if abs(m.change.position_delta[0]) > 0.5
            or abs(m.change.position_delta[1]) > 0.5
        )

    @property
    def resized(self) -> Tuple[ObjectMatch, ...]:
        """Survived objects whose pixel count changed."""
        return tuple(m for m in self.survived if m.change.size_delta != 0)

    @property
    def reshaped(self) -> Tuple[ObjectMatch, ...]:
        """Survived objects whose shape_signature changed."""
        return tuple(m for m in self.survived if m.change.shape_changed)

    @property
    def unchanged(self) -> Tuple[ObjectMatch, ...]:
        """Survived objects that are identical in color, position, and shape."""
        return tuple(
            m for m in self.survived
            if not m.change.color_changed
            and not m.change.shape_changed
            and abs(m.change.position_delta[0]) <= 0.5
            and abs(m.change.position_delta[1]) <= 0.5
        )


# ── Scoring primitives ────────────────────────────────────────────────────────

def _bbox_iou(a: Object, b: Object) -> float:
    """Intersection-over-Union of the two objects' bounding boxes."""
    min_r_a, min_c_a, max_r_a, max_c_a = a.bounding_box
    min_r_b, min_c_b, max_r_b, max_c_b = b.bounding_box

    inter_r = max(0, min(max_r_a, max_r_b) - max(min_r_a, min_r_b) + 1)
    inter_c = max(0, min(max_c_a, max_c_b) - max(min_c_a, min_c_b) + 1)
    inter   = inter_r * inter_c

    area_a = (max_r_a - min_r_a + 1) * (max_c_a - min_c_a + 1)
    area_b = (max_r_b - min_r_b + 1) * (max_c_b - min_c_b + 1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _shape_jaccard(a: Object, b: Object) -> float:
    """
    Jaccard similarity of shape signatures.

    Shape signatures are position-invariant (offsets from top-left corner),
    so this is high when the two objects have the same shape regardless of
    where they appear in the grid.
    """
    inter = len(a.shape_signature & b.shape_signature)
    union = len(a.shape_signature | b.shape_signature)
    return inter / union if union > 0 else 1.0   # both empty → identical


def _match_score(a: Object, b: Object) -> float:
    """
    Composite similarity in [0, 1].

    Weights
    -------
    _W_IOU   (0.35) × bounding-box IoU
    _W_SHAPE (0.40) × shape-signature Jaccard  ← position-invariant
    _W_COLOR (0.25) × color agreement (binary)
    """
    iou     = _bbox_iou(a, b)
    jaccard = _shape_jaccard(a, b)
    color   = 1.0 if a.dominant_color == b.dominant_color else 0.0
    return _W_IOU * iou + _W_SHAPE * jaccard + _W_COLOR * color


def _build_change(a: Object, b: Object) -> ObjectChange:
    dr = b.centroid[0] - a.centroid[0]
    dc = b.centroid[1] - a.centroid[1]
    return ObjectChange(
        color_changed=a.dominant_color != b.dominant_color,
        color_from=a.dominant_color,
        color_to=b.dominant_color,
        position_delta=(dr, dc),
        size_delta=b.size - a.size,
        shape_changed=a.shape_signature != b.shape_signature,
        iou=_bbox_iou(a, b),
        shape_jaccard=_shape_jaccard(a, b),
    )


# ── Core alignment ────────────────────────────────────────────────────────────

def align_pair(pair_index: int, inp: Grid, out: Grid) -> PairAlignment:
    """
    Align input objects to output objects for one training pair.

    Builds an ``n_inp × n_out`` score matrix, runs the Hungarian algorithm
    (minimising ``1 − score``) for optimal assignment, then discards any
    assignment whose score falls below :data:`_MIN_SCORE`.

    Parameters
    ----------
    pair_index:
        The pair's position within the task (for bookkeeping).
    inp, out:
        Input and output :class:`~core.grid.Grid`.

    Returns
    -------
    :class:`PairAlignment`
    """
    inp_objs = extract_objects(inp)
    out_objs = extract_objects(out)

    # Degenerate cases — no objects on one or both sides.
    if not inp_objs and not out_objs:
        return PairAlignment(pair_index, (), (), ())
    if not inp_objs:
        return PairAlignment(pair_index, (), tuple(out_objs), ())
    if not out_objs:
        return PairAlignment(pair_index, (), (), tuple(inp_objs))

    n_inp, n_out = len(inp_objs), len(out_objs)

    # Build score matrix.
    score_matrix = np.zeros((n_inp, n_out), dtype=np.float64)
    for i, a in enumerate(inp_objs):
        for j, b in enumerate(out_objs):
            score_matrix[i, j] = _match_score(a, b)

    # Hungarian: linear_sum_assignment minimises cost.
    row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)

    matched_inp: set[int] = set()
    matched_out: set[int] = set()
    survived: List[ObjectMatch] = []

    for i, j in zip(row_ind, col_ind):
        s = score_matrix[i, j]
        if s >= _MIN_SCORE:
            a, b = inp_objs[i], out_objs[j]
            survived.append(ObjectMatch(
                input_obj=a,
                output_obj=b,
                score=float(s),
                change=_build_change(a, b),
            ))
            matched_inp.add(i)
            matched_out.add(j)

    destroyed = tuple(inp_objs[i] for i in range(n_inp) if i not in matched_inp)
    created   = tuple(out_objs[j] for j in range(n_out) if j not in matched_out)

    return PairAlignment(pair_index, tuple(survived), created, destroyed)


# ── Task-level alignment ──────────────────────────────────────────────────────

def align_task(task: Task) -> Tuple[PairAlignment, ...]:
    """
    Align objects for every training pair in *task*.

    Parameters
    ----------
    task:
        A :class:`~core.task.Task` with at least one training pair.

    Returns
    -------
    ``Tuple[PairAlignment, ...]``, one per training pair, in order.
    """
    return tuple(
        align_pair(i, inp, out)
        for i, (inp, out) in enumerate(task.train_pairs)
    )
