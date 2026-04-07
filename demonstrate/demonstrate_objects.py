"""
demonstrate_objects.py — Visual walkthrough of perception/objects.py

Shows:
  1. The raw grid (colour-coded ASCII)
  2. Each extracted Object: isolated view, bounding box, centroid,
     dominant colour, size, shape signature
  3. Difference between 4-connectivity and 8-connectivity

Run from the ARIA root:
    python demonstrate/demonstrate_objects.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core.grid import Grid
from perception.objects import extract_objects, Object

# ── ARC colour names (0-9) ────────────────────────────────────────────────────
COLOR_NAME = {
    0: "black",  1: "blue",   2: "red",    3: "green",  4: "yellow",
    5: "grey",   6: "fuchsia",7: "orange", 8: "azure",  9: "maroon",
}

# ── Terminal ANSI colours (one per ARC colour index) ─────────────────────────
ANSI = {
    0: "\033[90m",   # dark grey  (background / black)
    1: "\033[94m",   # blue
    2: "\033[91m",   # red
    3: "\033[92m",   # green
    4: "\033[93m",   # yellow
    5: "\033[37m",   # light grey
    6: "\033[95m",   # magenta / fuchsia
    7: "\033[33m",   # orange-ish
    8: "\033[96m",   # cyan / azure
    9: "\033[31m",   # dark red / maroon
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


# ── Rendering helpers ─────────────────────────────────────────────────────────

CHARS = ".123456789"

def _cell(value: int, highlight: bool = False) -> str:
    ch = CHARS[value]
    colour = ANSI[value]
    if highlight:
        return f"{BOLD}{colour}{ch}{RESET}"
    return f"{DIM}{colour}{ch}{RESET}"


def render_grid(grid: Grid, highlight_pixels=None, indent: int = 4) -> None:
    """Print the grid; highlight_pixels (set of (r,c)) shown bold, rest dimmed."""
    pad = " " * indent
    hi = set(highlight_pixels) if highlight_pixels else None
    for r in range(grid.rows):
        row_str = " ".join(
            _cell(grid.get(r, c), highlight=(hi is None or (r, c) in hi))
            for c in range(grid.cols)
        )
        print(f"{pad}{row_str}")


def render_object_crop(obj: Object, grid: Grid, indent: int = 4) -> None:
    """Render just the bounding-box region; object pixels shown bold, rest dimmed."""
    min_r, min_c, max_r, max_c = obj.bounding_box
    pad = " " * indent
    for r in range(min_r, max_r + 1):
        row_str = " ".join(
            _cell(grid.get(r, c), highlight=((r, c) in obj.pixels))
            for c in range(min_c, max_c + 1)
        )
        print(f"{pad}{row_str}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}--- {title} ---{RESET}")


# ── Demo grids ────────────────────────────────────────────────────────────────

GRIDS = {
    "Simple shapes": Grid([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 2, 0, 0],
        [0, 1, 1, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
    ]),

    "Diagonal touch (4 vs 8)": Grid([
        [0, 0, 0, 0, 0],
        [0, 5, 0, 5, 0],
        [0, 0, 5, 0, 0],
        [0, 5, 0, 5, 0],
        [0, 0, 0, 0, 0],
    ]),

    "Multi-color cluster": Grid([
        [0, 1, 1, 0, 6, 6, 6],
        [0, 1, 0, 0, 6, 0, 6],
        [0, 0, 0, 0, 6, 6, 6],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 8, 0, 8],
        [0, 0, 0, 0, 0, 8, 0],
    ]),
}


# ── Main demo ─────────────────────────────────────────────────────────────────

def show_objects(grid: Grid, connectivity: int) -> None:
    objects = extract_objects(grid, connectivity=connectivity)
    print(f"\n  Connectivity={connectivity}  →  {len(objects)} object(s) found\n")

    for i, obj in enumerate(objects):
        cr, cc = obj.centroid
        name = COLOR_NAME[obj.dominant_color]
        min_r, min_c, max_r, max_c = obj.bounding_box
        bb_h = max_r - min_r + 1
        bb_w = max_c - min_c + 1

        subsection(f"Object {i}  |  color={obj.dominant_color} ({name})  size={obj.size}")
        print(f"    bounding box : rows {min_r}–{max_r}, cols {min_c}–{max_c}  "
              f"({bb_h}×{bb_w})")
        print(f"    centroid     : ({cr:.2f}, {cc:.2f})")
        print(f"    shape sig    : {sorted(obj.shape_signature)}")

        print(f"\n    Full grid  (object = bright, rest = dim):")
        render_grid(grid, highlight_pixels=obj.pixels, indent=6)

        print(f"\n    Bounding-box crop:")
        render_object_crop(obj, grid, indent=6)


for grid_name, grid in GRIDS.items():
    section(f"GRID: {grid_name}  {grid.shape}")

    print(f"\n  Full grid:")
    render_grid(grid, indent=4)

    print(f"\n  Colors present: "
          + ", ".join(f"{c} ({COLOR_NAME[c]})" for c in grid.unique_colors() if c != 0))

    show_objects(grid, connectivity=4)

    # Only show 8-connectivity if result differs
    objs4 = extract_objects(grid, connectivity=4)
    objs8 = extract_objects(grid, connectivity=8)
    if len(objs4) != len(objs8):
        section(f"GRID: {grid_name}  — 8-connectivity comparison")
        show_objects(grid, connectivity=8)
        print(f"\n  {BOLD}4-conn: {len(objs4)} objects  vs  8-conn: {len(objs8)} objects{RESET}")
    else:
        print(f"\n  (8-connectivity gives the same {len(objs8)} object(s) for this grid)")
