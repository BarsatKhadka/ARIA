import numpy as np
from typing import List, Tuple


class Grid:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.int8)

    # ── Dimensions ────────────────────────────────────────────────────────────

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    # ── Cell access ───────────────────────────────────────────────────────────

    def get(self, row: int, col: int) -> int:
        return int(self.data[row, col])

    def update(self, row: int, col: int, value: int) -> "Grid":
        """Return a NEW Grid with one cell changed. Original is unchanged."""
        if not (0 <= value <= 9):
            raise ValueError(f"ARC colors must be 0–9, got {value}.")
        new_data = self.data.copy()
        new_data[row, col] = value
        return Grid(new_data.tolist())

    # ── Color queries ─────────────────────────────────────────────────────────

    def cells_of_color(self, color: int) -> List[Tuple[int, int]]:
        """Return all (row, col) positions where the cell equals color."""
        rows, cols = np.where(self.data == color)
        return list(zip(rows.tolist(), cols.tolist()))

    def unique_colors(self) -> List[int]:
        return sorted(np.unique(self.data).tolist())

    # ── Comparison ────────────────────────────────────────────────────────────

    def equals(self, other: "Grid") -> bool:
        return self.shape == other.shape and np.array_equal(self.data, other.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.equals(other)

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def diff_mask(self, other: "Grid") -> np.ndarray:
        """
        Boolean mask where True means the cell differs between self and other.
        Both grids must have the same shape.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return self.data != other.data

    def diff_cells(self, other: "Grid") -> List[Tuple[int, int]]:
        """Return (row, col) positions where self and other differ."""
        mask = self.diff_mask(other)
        rows, cols = np.where(mask)
        return list(zip(rows.tolist(), cols.tolist()))

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_list(self) -> List[List[int]]:
        """Convert to nested Python list (for JSON submission)."""
        return self.data.tolist()

    @classmethod
    def from_list(cls, data: List[List[int]]) -> "Grid":
        return cls(data)

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        chars = ".123456789"
        rows = [" ".join(chars[c] for c in row) for row in self.data]
        return "\n".join(rows)
