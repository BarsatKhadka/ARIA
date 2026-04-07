from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from core.grid import Grid

# ── Types ─────────────────────────────────────────────────────────────────────

LineDirection = Literal["horizontal", "vertical", "diagonal-down", "diagonal-up"]
#  horizontal   : same row,    cols increase      →
#  vertical     : same col,    rows increase      ↓
#  diagonal-down: rows+cols both increase         ↘
#  diagonal-up  : rows decrease, cols increase    ↗


@dataclass(frozen=True)
class Line:
    start: Tuple[int, int]      # (row, col) of the first cell
    end: Tuple[int, int]        # (row, col) of the last cell
    direction: LineDirection
    color: int                  # ARC color (0–9)
    length: int                 # number of cells
    cells: Tuple[Tuple[int, int], ...]  # ordered (row, col) sequence


# ── Run scanner ───────────────────────────────────────────────────────────────

def _scan_runs(
    grid: Grid,
    sequence: List[Tuple[int, int]],
    min_length: int,
    background: int,
    include_background: bool,
) -> List[Tuple[int, int, int]]:
    """
    Scan *sequence* of (row, col) positions and return same-color runs.

    Returns list of ``(start_idx, end_idx, color)`` for every run whose
    length is ≥ *min_length* and whose color passes the background filter.
    """
    if not sequence:
        return []

    results: List[Tuple[int, int, int]] = []
    run_start = 0
    run_color = grid.get(*sequence[0])

    for i in range(1, len(sequence)):
        color = grid.get(*sequence[i])
        if color != run_color:
            length = i - run_start
            if length >= min_length and (include_background or run_color != background):
                results.append((run_start, i - 1, run_color))
            run_start = i
            run_color = color

    # Flush the final run.
    length = len(sequence) - run_start
    if length >= min_length and (include_background or run_color != background):
        results.append((run_start, len(sequence) - 1, run_color))

    return results


def _make_line(
    sequence: List[Tuple[int, int]],
    start_idx: int,
    end_idx: int,
    color: int,
    direction: LineDirection,
) -> Line:
    cells = tuple(sequence[start_idx : end_idx + 1])
    return Line(
        start=cells[0],
        end=cells[-1],
        direction=direction,
        color=color,
        length=len(cells),
        cells=cells,
    )


# ── Direction scanners ────────────────────────────────────────────────────────

def _horizontal_sequences(grid: Grid) -> List[List[Tuple[int, int]]]:
    return [[(r, c) for c in range(grid.cols)] for r in range(grid.rows)]


def _vertical_sequences(grid: Grid) -> List[List[Tuple[int, int]]]:
    return [[(r, c) for r in range(grid.rows)] for c in range(grid.cols)]


def _diagonal_down_sequences(grid: Grid) -> List[List[Tuple[int, int]]]:
    """All ↘ diagonals (row and col both increase)."""
    rows, cols = grid.rows, grid.cols
    sequences: List[List[Tuple[int, int]]] = []
    # Diagonals starting on the top row.
    for start_c in range(cols):
        seq = []
        r, c = 0, start_c
        while r < rows and c < cols:
            seq.append((r, c))
            r += 1; c += 1
        if len(seq) >= 2:
            sequences.append(seq)
    # Diagonals starting on the left column (skip corner already added).
    for start_r in range(1, rows):
        seq = []
        r, c = start_r, 0
        while r < rows and c < cols:
            seq.append((r, c))
            r += 1; c += 1
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def _diagonal_up_sequences(grid: Grid) -> List[List[Tuple[int, int]]]:
    """All ↗ diagonals (row decreases, col increases)."""
    rows, cols = grid.rows, grid.cols
    sequences: List[List[Tuple[int, int]]] = []
    # Diagonals starting on the bottom row.
    for start_c in range(cols):
        seq = []
        r, c = rows - 1, start_c
        while r >= 0 and c < cols:
            seq.append((r, c))
            r -= 1; c += 1
        if len(seq) >= 2:
            sequences.append(seq)
    # Diagonals starting on the left column (skip corner already added).
    for start_r in range(rows - 2, -1, -1):
        seq = []
        r, c = start_r, 0
        while r >= 0 and c < cols:
            seq.append((r, c))
            r -= 1; c += 1
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


# ── Public API ────────────────────────────────────────────────────────────────

def find_lines(
    grid: Grid,
    min_length: int = 3,
    background: int = 0,
    include_background: bool = False,
) -> List[Line]:
    """
    Find all horizontal, vertical, and diagonal runs of the same color.

    A *line* is a maximal run of consecutive same-colored cells along one of
    the four directions whose length is ≥ *min_length*.

    Parameters
    ----------
    grid:
        The source :class:`~core.grid.Grid`.
    min_length:
        Minimum run length to report. Default ``3``.
    background:
        Color treated as background. Background-colored runs are excluded
        unless *include_background* is ``True``.
    include_background:
        When ``True``, background-colored runs are also returned.

    Returns
    -------
    List of :class:`Line`, ordered: horizontal first (top→bottom),
    then vertical (left→right), then diagonal-down, then diagonal-up.
    """
    if min_length < 2:
        raise ValueError(f"min_length must be ≥ 2, got {min_length}")

    lines: List[Line] = []

    scanners: List[Tuple[LineDirection, List[List[Tuple[int, int]]]]] = [
        ("horizontal",    _horizontal_sequences(grid)),
        ("vertical",      _vertical_sequences(grid)),
        ("diagonal-down", _diagonal_down_sequences(grid)),
        ("diagonal-up",   _diagonal_up_sequences(grid)),
    ]

    for direction, sequences in scanners:
        for seq in sequences:
            for start_idx, end_idx, color in _scan_runs(
                grid, seq, min_length, background, include_background
            ):
                lines.append(_make_line(seq, start_idx, end_idx, color, direction))

    return lines
