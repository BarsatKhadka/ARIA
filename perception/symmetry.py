from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

from core.grid import Grid
from perception.objects import Object


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SymmetryResult:
    vertical: bool             # left-right reflection across a vertical axis
    horizontal: bool           # top-bottom reflection across a horizontal axis
    rotation_180: bool         # 180° rotational symmetry
    rotation_90: bool          # 90° rotational symmetry (implies 180°)
    vertical_score: float      # fraction of cells matching under vertical reflection
    horizontal_score: float    # fraction of cells matching under horizontal reflection
    rotation_180_score: float  # fraction of cells matching under 180° rotation
    rotation_90_score: float   # fraction of cells matching under 90° rotation
    vertical_axis: Optional[float]    # col index of reflection axis (may be .5 for even width)
    horizontal_axis: Optional[float]  # row index of reflection axis (may be .5 for even height)
    rotation_center: Optional[Tuple[float, float]]  # (row, col) center of rotation


# ── Score helpers ─────────────────────────────────────────────────────────────

def _score_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of positions where a == b."""
    return float(np.sum(a == b)) / a.size


# ── Grid-level symmetry ───────────────────────────────────────────────────────

def _grid_to_array(grid: Grid) -> np.ndarray:
    return grid.data.astype(np.int8)


def analyze_grid_symmetry(grid: Grid, threshold: float = 1.0) -> SymmetryResult:
    """
    Test all four symmetry types for a full :class:`~core.grid.Grid`.

    Parameters
    ----------
    grid:
        The source grid.
    threshold:
        Minimum symmetry score (0.0–1.0) required to declare a symmetry present.
        Default ``1.0`` means perfect symmetry only. Lower values allow
        near-symmetric grids (e.g. ``0.9`` = 90 % of cells match).

    Returns
    -------
    :class:`SymmetryResult`
    """
    arr = _grid_to_array(grid)
    return _analyze_array_symmetry(arr, threshold)


def _analyze_array_symmetry(arr: np.ndarray, threshold: float) -> SymmetryResult:
    rows, cols = arr.shape

    # ── Vertical reflection (left ↔ right) ────────────────────────────────────
    v_reflected = np.fliplr(arr)
    v_score = _score_arrays(arr, v_reflected)
    vertical = v_score >= threshold
    v_axis: Optional[float] = (cols - 1) / 2.0 if vertical else None

    # ── Horizontal reflection (top ↔ bottom) ──────────────────────────────────
    h_reflected = np.flipud(arr)
    h_score = _score_arrays(arr, h_reflected)
    horizontal = h_score >= threshold
    h_axis: Optional[float] = (rows - 1) / 2.0 if horizontal else None

    # ── 180° rotation ─────────────────────────────────────────────────────────
    rot180 = np.rot90(arr, 2)
    r180_score = _score_arrays(arr, rot180)
    has_180 = r180_score >= threshold

    # ── 90° rotation ──────────────────────────────────────────────────────────
    # 90° symmetry requires the grid to be square and match under 90° rotation.
    if rows == cols:
        rot90 = np.rot90(arr, 1)
        r90_score = _score_arrays(arr, rot90)
        has_90 = r90_score >= threshold
    else:
        r90_score = 0.0
        has_90 = False

    rot_center: Optional[Tuple[float, float]] = (
        ((rows - 1) / 2.0, (cols - 1) / 2.0)
        if (has_180 or has_90)
        else None
    )

    return SymmetryResult(
        vertical=vertical,
        horizontal=horizontal,
        rotation_180=has_180,
        rotation_90=has_90,
        vertical_score=v_score,
        horizontal_score=h_score,
        rotation_180_score=r180_score,
        rotation_90_score=r90_score,
        vertical_axis=v_axis,
        horizontal_axis=h_axis,
        rotation_center=rot_center,
    )


# ── Object-level symmetry ─────────────────────────────────────────────────────

def _object_to_array(obj: Object) -> np.ndarray:
    """Render an Object into a tight bounding-box array using its pixel set.

    Non-member cells within the bounding box are filled with -1 so they are
    never treated as matching any ARC color (0–9).
    """
    min_r, min_c, max_r, max_c = obj.bounding_box
    height = max_r - min_r + 1
    width  = max_c - min_c + 1
    arr = np.full((height, width), -1, dtype=np.int8)
    for r, c in obj.pixels:
        arr[r - min_r, c - min_c] = 1  # presence marker; color irrelevant for shape symmetry
    return arr


def analyze_object_symmetry(obj: Object, threshold: float = 1.0) -> SymmetryResult:
    """
    Test all four symmetry types for the *shape* of an :class:`~perception.objects.Object`.

    The object is rendered into its bounding box before testing. Only the
    presence or absence of pixels is considered — color is ignored — so two
    objects of different colors with the same shape report identical symmetry.

    Parameters
    ----------
    obj:
        The object to test.
    threshold:
        Minimum symmetry score to declare a symmetry present. Default ``1.0``.

    Returns
    -------
    :class:`SymmetryResult`
    """
    arr = _object_to_array(obj)
    return _analyze_array_symmetry(arr, threshold)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_symmetry(
    target: Union[Grid, Object],
    threshold: float = 1.0,
) -> SymmetryResult:
    """
    Test all four symmetry types for a :class:`~core.grid.Grid` or
    :class:`~perception.objects.Object`.

    Parameters
    ----------
    target:
        A ``Grid`` (full grid analysis) or ``Object`` (shape analysis within
        bounding box).
    threshold:
        Minimum fraction of matching cells required to declare a symmetry
        present. ``1.0`` = perfect only; ``0.9`` = near-symmetric allowed.

    Returns
    -------
    :class:`SymmetryResult`
    """
    if isinstance(target, Grid):
        return analyze_grid_symmetry(target, threshold)
    elif isinstance(target, Object):
        return analyze_object_symmetry(target, threshold)
    else:
        raise TypeError(f"target must be Grid or Object, got {type(target).__name__!r}")
