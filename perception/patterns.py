from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.grid import Grid


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PeriodicityResult:
    found: bool                 # True if a repeating tile was discovered
    tile: Optional[Grid]        # the minimal repeating tile (None if not found)
    period_rows: Optional[int]  # tile height — vertical repetition period
    period_cols: Optional[int]  # tile width  — horizontal repetition period
    score: float                # fraction of cells consistent with the best tile found


# ── Core check ────────────────────────────────────────────────────────────────

def _tile_score(arr: np.ndarray, th: int, tw: int) -> float:
    """
    Fraction of cells in *arr* that equal ``arr[r % th, c % tw]``.

    Builds the infinite tiling of the top-left ``th × tw`` subtile and
    crops it back to the original shape, then counts matches.
    """
    rows, cols = arr.shape
    tile = arr[:th, :tw]
    reps_r = -(-rows // th)   # ceil division
    reps_c = -(-cols // tw)
    tiled = np.tile(tile, (reps_r, reps_c))[:rows, :cols]
    return float(np.sum(arr == tiled)) / arr.size


# ── Public API ────────────────────────────────────────────────────────────────

def detect_periodicity(
    grid: Grid,
    threshold: float = 1.0,
) -> PeriodicityResult:
    """
    Test whether *grid* tiles with a repeating pattern.

    Tries every 2D tile size ``(th, tw)`` where ``2 ≤ th ≤ rows // 2`` and
    ``2 ≤ tw ≤ cols // 2``. Returns the smallest tile (by area) whose tiling
    of the grid achieves *threshold*.

    When no tile reaches *threshold*, ``found`` is ``False`` and ``score``
    reports the highest match fraction seen across all tested sizes — useful
    for detecting near-periodic grids.

    Parameters
    ----------
    grid:
        The source :class:`~core.grid.Grid`.
    threshold:
        Minimum fraction of cells that must match the tiled pattern to declare
        periodicity found. Default ``1.0`` (perfect tiling only).

    Returns
    -------
    :class:`PeriodicityResult`
    """
    arr = grid.data
    rows, cols = arr.shape

    # Need at least one candidate tile size in each dimension.
    if rows < 4 or cols < 4:
        return PeriodicityResult(found=False, tile=None,
                                 period_rows=None, period_cols=None, score=0.0)

    best_found: Optional[tuple[int, int, float]] = None  # (th, tw, score)
    best_score = 0.0  # best score seen even if below threshold

    for th in range(2, rows // 2 + 1):
        for tw in range(2, cols // 2 + 1):
            score = _tile_score(arr, th, tw)

            if score > best_score:
                best_score = score

            if score >= threshold:
                # Prefer smaller area; break ties by smaller th then tw.
                if best_found is None or th * tw < best_found[0] * best_found[1]:
                    best_found = (th, tw, score)

    if best_found is not None:
        th, tw, score = best_found
        return PeriodicityResult(
            found=True,
            tile=Grid(arr[:th, :tw].tolist()),
            period_rows=th,
            period_cols=tw,
            score=score,
        )

    return PeriodicityResult(
        found=False,
        tile=None,
        period_rows=None,
        period_cols=None,
        score=best_score,
    )
