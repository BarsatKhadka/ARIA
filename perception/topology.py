from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple

from core.grid import Grid
from perception.objects import Object

# ── Types ─────────────────────────────────────────────────────────────────────

Direction = Literal["above", "below", "left", "right",
                    "above-left", "above-right", "below-left", "below-right",
                    "same"]


@dataclass(frozen=True)
class ObjectRelation:
    a: Object                    # the reference object
    b: Object                    # the comparison object
    direction: Direction         # position of b relative to a
    distance: float              # Euclidean distance between centroids
    touches: bool                # share at least one 4-adjacent cell pair across their borders
    a_encloses_b: bool           # a's bounding box contains b and a forms a closed border around b
    b_encloses_a: bool           # b's bounding box contains a and b forms a closed border around a


# ── Direction ─────────────────────────────────────────────────────────────────

def _direction(a: Object, b: Object) -> Direction:
    """Position of *b* relative to *a*, based on centroid delta."""
    dr = b.centroid[0] - a.centroid[0]   # positive → b is below a
    dc = b.centroid[1] - a.centroid[1]   # positive → b is right of a

    # Use 2:1 ratio: primary axis must be at least 2× the secondary to count
    # as a cardinal direction; otherwise it's diagonal.
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return "same"

    primarily_vertical   = abs(dr) >= 2 * abs(dc)
    primarily_horizontal = abs(dc) >= 2 * abs(dr)

    if primarily_vertical:
        return "below" if dr > 0 else "above"
    if primarily_horizontal:
        return "right" if dc > 0 else "left"
    # Diagonal
    vert = "below" if dr > 0 else "above"
    horiz = "right" if dc > 0 else "left"
    return f"{vert}-{horiz}"  # type: ignore[return-value]


# ── Distance ──────────────────────────────────────────────────────────────────

def _distance(a: Object, b: Object) -> float:
    """Euclidean distance between centroids."""
    dr = b.centroid[0] - a.centroid[0]
    dc = b.centroid[1] - a.centroid[1]
    return math.sqrt(dr * dr + dc * dc)


# ── Touching ──────────────────────────────────────────────────────────────────

_CARDINAL: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _touches(a: Object, b: Object) -> bool:
    """
    True if any pixel of *a* is 4-adjacent to any pixel of *b*.

    Uses a set-membership check: build the neighbor set of *a* once, then
    test membership of every pixel of *b*.
    """
    a_neighbors: Set[Tuple[int, int]] = set()
    for r, c in a.pixels:
        for dr, dc in _CARDINAL:
            a_neighbors.add((r + dr, c + dc))
    return bool(a_neighbors & b.pixels)


# ── Enclosure ─────────────────────────────────────────────────────────────────

def _bbox_contains(outer: Object, inner: Object) -> bool:
    """True if outer's bounding box strictly contains inner's bounding box."""
    o_minr, o_minc, o_maxr, o_maxc = outer.bounding_box
    i_minr, i_minc, i_maxr, i_maxc = inner.bounding_box
    return (o_minr < i_minr and o_minc < i_minc
            and o_maxr > i_maxr and o_maxc > i_maxc)


def _encloses(outer: Object, inner: Object, grid: Grid, background: int = 0) -> bool:
    """
    True if *outer* forms a closed background-free border around *inner*.

    Steps
    -----
    1. Check that *outer*'s bounding box strictly contains *inner*'s.
    2. Flood-fill background cells starting from any cell **outside** the
       outer bounding box.  If the flood fill cannot reach any background
       cell **inside** the inner bounding box, *outer* forms a sealed wall
       and truly encloses *inner*.
    """
    if not _bbox_contains(outer, inner):
        return False

    o_minr, o_minc, o_maxr, o_maxc = outer.bounding_box
    rows, cols = grid.rows, grid.cols

    # All non-outer-object cells that are background-colored.
    outer_pixels = outer.pixels

    def is_bg(r: int, c: int) -> bool:
        return (r, c) not in outer_pixels and grid.get(r, c) == background

    # Start the flood fill from a corner guaranteed to be outside the outer
    # bounding box.  Walk the full grid border to find a valid seed.
    seed: Optional[Tuple[int, int]] = None
    for c in range(cols):
        if is_bg(0, c):
            seed = (0, c)
            break
    if seed is None:
        for r in range(rows):
            if is_bg(r, 0):
                seed = (r, 0)
                break
    if seed is None:
        # Entire border is non-background — can't prove enclosure this way.
        return False

    # BFS from seed through background cells.
    visited: Set[Tuple[int, int]] = {seed}
    queue = [seed]
    head = 0
    while head < len(queue):
        r, c = queue[head]; head += 1
        for dr, dc in _CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and is_bg(nr, nc):
                visited.add((nr, nc))
                queue.append((nr, nc))

    # Check whether any background cell strictly inside the inner bbox was reached.
    i_minr, i_minc, i_maxr, i_maxc = inner.bounding_box
    for r in range(i_minr, i_maxr + 1):
        for c in range(i_minc, i_maxc + 1):
            if (r, c) not in inner.pixels and grid.get(r, c) == background:
                if (r, c) in visited:
                    return False  # outside connected to inside → not enclosed

    return True


# ── Public API ────────────────────────────────────────────────────────────────

def relate(a: Object, b: Object, grid: Grid, background: int = 0) -> ObjectRelation:
    """
    Compute the full spatial relationship between objects *a* and *b*.

    Parameters
    ----------
    a, b:
        The two :class:`~perception.objects.Object` instances to compare.
    grid:
        The source grid (needed for enclosure checking).
    background:
        Background color value. Default ``0``.

    Returns
    -------
    :class:`ObjectRelation`
    """
    return ObjectRelation(
        a=a,
        b=b,
        direction=_direction(a, b),
        distance=_distance(a, b),
        touches=_touches(a, b),
        a_encloses_b=_encloses(a, b, grid, background),
        b_encloses_a=_encloses(b, a, grid, background),
    )


def relate_all(
    objects: List[Object],
    grid: Grid,
    background: int = 0,
) -> List[ObjectRelation]:
    """
    Compute :class:`ObjectRelation` for every ordered pair of objects.

    Returns one ``ObjectRelation`` per ordered pair ``(a, b)`` where ``a ≠ b``,
    so ``len(objects) * (len(objects) - 1)`` results in total.

    Parameters
    ----------
    objects:
        List of objects, typically from :func:`~perception.objects.extract_objects`.
    grid:
        The source grid.
    background:
        Background color value. Default ``0``.

    Returns
    -------
    List of :class:`ObjectRelation`.
    """
    relations: List[ObjectRelation] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i != j:
                relations.append(relate(a, b, grid, background))
    return relations
