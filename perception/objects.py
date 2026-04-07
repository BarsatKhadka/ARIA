from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import FrozenSet, List, Literal, Set, Tuple

from core.grid import Grid


@dataclass(frozen=True)
class Object:
    pixels: FrozenSet[Tuple[int, int]]          # (row, col) members
    bounding_box: Tuple[int, int, int, int]      # (min_row, min_col, max_row, max_col)
    centroid: Tuple[float, float]                # (row, col) float centre-of-mass
    dominant_color: int                          # most-frequent ARC color in the object
    size: int                                    # number of pixels
    shape_signature: FrozenSet[Tuple[int, int]]  # positions relative to top-left corner


# ── Connectivity kernels ───────────────────────────────────────────────────────

_NEIGHBORS_4: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ── BFS flood-fill ────────────────────────────────────────────────────────────

def _flood_fill(
    grid: Grid,
    start: Tuple[int, int],
    visited: Set[Tuple[int, int]],
    connectivity: Literal[4, 8],
    background: int,
) -> Set[Tuple[int, int]]:
    """Return the connected component reachable from *start* (same color, non-background)."""
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    target_color = grid.get(*start)
    component: Set[Tuple[int, int]] = set()
    queue: deque[Tuple[int, int]] = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        component.add((r, c))
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid.rows
                and 0 <= nc < grid.cols
                and (nr, nc) not in visited
                and grid.get(nr, nc) == target_color
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return component


# ── Object construction helper ────────────────────────────────────────────────

def _make_object(pixels: Set[Tuple[int, int]], grid: Grid) -> Object:
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    centroid = (sum(rows) / len(rows), sum(cols) / len(cols))

    color_counts: Counter[int] = Counter(grid.get(r, c) for r, c in pixels)
    dominant_color = color_counts.most_common(1)[0][0]

    shape_signature = frozenset((r - min_r, c - min_c) for r, c in pixels)

    return Object(
        pixels=frozenset(pixels),
        bounding_box=(min_r, min_c, max_r, max_c),
        centroid=centroid,
        dominant_color=dominant_color,
        size=len(pixels),
        shape_signature=shape_signature,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_objects(
    grid: Grid,
    connectivity: Literal[4, 8] = 4,
    background: int = 0,
    include_background: bool = False,
) -> List[Object]:
    """
    Extract connected components from *grid* as a list of :class:`Object`.

    Parameters
    ----------
    grid:
        The source :class:`~core.grid.Grid`.
    connectivity:
        ``4`` for 4-connectivity (cardinal neighbours only);
        ``8`` for 8-connectivity (diagonal neighbours included).
    background:
        Color value treated as background. Cells of this color are skipped
        unless *include_background* is ``True``.
    include_background:
        When ``True``, background-colored cells are also segmented into
        connected components.

    Returns
    -------
    List of :class:`Object`, ordered top-to-bottom then left-to-right by the
    top-left corner of each bounding box.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    visited: Set[Tuple[int, int]] = set()
    objects: List[Object] = []

    for r in range(grid.rows):
        for c in range(grid.cols):
            if (r, c) in visited:
                continue
            color = grid.get(r, c)
            if color == background and not include_background:
                visited.add((r, c))
                continue
            component = _flood_fill(grid, (r, c), visited, connectivity, background)
            objects.append(_make_object(component, grid))

    return objects
