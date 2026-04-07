from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from core.grid import Grid


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColorAnalysis:
    histogram: Dict[int, int]                             # color → pixel count
    background: int                                       # detected background color
    color_regions: Dict[int, FrozenSet[Tuple[int, int]]]  # color → all (row,col) cells


# ── Background detection ──────────────────────────────────────────────────────

def detect_background_by_frequency(grid: Grid) -> int:
    """Return the most frequent color in the grid."""
    counter: Counter[int] = Counter()
    for r in range(grid.rows):
        for c in range(grid.cols):
            counter[grid.get(r, c)] += 1
    return counter.most_common(1)[0][0]


def detect_background_by_border(grid: Grid) -> int:
    """Return the most frequent color among border (edge) cells."""
    counter: Counter[int] = Counter()
    rows, cols = grid.rows, grid.cols

    for c in range(cols):          # top row
        counter[grid.get(0, c)] += 1
    for c in range(cols):          # bottom row
        counter[grid.get(rows - 1, c)] += 1
    for r in range(1, rows - 1):   # left column (skip corners already counted)
        counter[grid.get(r, 0)] += 1
    for r in range(1, rows - 1):   # right column
        counter[grid.get(r, cols - 1)] += 1

    return counter.most_common(1)[0][0]


# ── Color histogram ───────────────────────────────────────────────────────────

def color_histogram(grid: Grid) -> Dict[int, int]:
    """Return a dict mapping each present color to its pixel count."""
    counter: Counter[int] = Counter()
    for r in range(grid.rows):
        for c in range(grid.cols):
            counter[grid.get(r, c)] += 1
    return dict(counter)


# ── Color regions ─────────────────────────────────────────────────────────────

def color_regions(grid: Grid) -> Dict[int, FrozenSet[Tuple[int, int]]]:
    """
    Return a dict mapping each color to the frozenset of all (row, col) cells
    that carry that color, regardless of adjacency.
    """
    buckets: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            color = grid.get(r, c)
            if color not in buckets:
                buckets[color] = []
            buckets[color].append((r, c))
    return {color: frozenset(cells) for color, cells in buckets.items()}


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_colors(
    grid: Grid,
    background_method: str = "frequency",
) -> ColorAnalysis:
    """
    Run all color analyses on *grid* and return a :class:`ColorAnalysis`.

    Parameters
    ----------
    grid:
        The source :class:`~core.grid.Grid`.
    background_method:
        How to detect the background color.

        - ``"frequency"`` — most frequent color overall (default).
        - ``"border"``    — most frequent color among edge cells.

    Returns
    -------
    :class:`ColorAnalysis` with histogram, background color, and color regions.
    """
    if background_method == "frequency":
        bg = detect_background_by_frequency(grid)
    elif background_method == "border":
        bg = detect_background_by_border(grid)
    else:
        raise ValueError(
            f"background_method must be 'frequency' or 'border', got {background_method!r}"
        )

    return ColorAnalysis(
        histogram=color_histogram(grid),
        background=bg,
        color_regions=color_regions(grid),
    )
