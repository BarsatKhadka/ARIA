"""
demonstrate_perception.py — Module-by-module visual test of all perception modules.

Tests each module against 20 hand-crafted grids chosen to exercise specific behaviours.
Run from the ARIA root:
    python demonstrate/demonstrate_perception.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core.grid import Grid
from perception.objects  import extract_objects
from perception.colors   import analyze_colors, detect_background_by_border, detect_background_by_frequency
from perception.symmetry import analyze_symmetry
from perception.patterns import detect_periodicity
from perception.topology import relate_all
from perception.lines    import find_lines

# ── ANSI helpers ──────────────────────────────────────────────────────────────

ANSI = {
    0:"\033[90m", 1:"\033[94m", 2:"\033[91m", 3:"\033[92m", 4:"\033[93m",
    5:"\033[37m", 6:"\033[95m", 7:"\033[33m", 8:"\033[96m", 9:"\033[31m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

COLOR_NAME = {
    0:"black", 1:"blue", 2:"red", 3:"green", 4:"yellow",
    5:"grey",  6:"fuchsia", 7:"orange", 8:"azure", 9:"maroon",
}
CHARS = ".123456789"

def _cell(v, hi=True):
    return f"{''+BOLD if hi else DIM}{ANSI[v]}{CHARS[v]}{RESET}"

def render(grid, highlight=None, indent=4):
    pad = " " * indent
    hi = set(highlight) if highlight is not None else None
    for r in range(grid.rows):
        print(pad + " ".join(
            _cell(grid.get(r, c), hi is None or (r, c) in hi)
            for c in range(grid.cols)
        ))

def section(title):
    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'═'*64}{RESET}")

def sub(title):
    print(f"\n  {BOLD}── {title}{RESET}")

def ok(msg):   print(f"    {BOLD}\033[92m✓{RESET} {msg}")
def info(msg): print(f"    {msg}")


# ── Test grids ────────────────────────────────────────────────────────────────
# Each entry: (name, Grid, notes)

GRIDS = [
    # ── objects ───────────────────────────────────────────────────────────────
    ("01 · two separate blobs", Grid([
        [0,0,0,0,0,0,0],
        [0,1,1,0,0,2,0],
        [0,1,1,0,0,2,0],
        [0,0,0,0,0,0,0],
    ]), "expect 2 objects: blue 2×2, red 2×1"),

    ("02 · diagonal touch (4 vs 8)", Grid([
        [0,3,0,0],
        [0,0,3,0],
        [0,0,0,3],
        [0,0,0,0],
    ]), "4-conn: 3 objects; 8-conn: 1 object"),

    ("03 · L-shape + dot", Grid([
        [0,0,0,0,0],
        [0,2,0,0,0],
        [0,2,0,0,4],
        [0,2,2,0,0],
        [0,0,0,0,0],
    ]), "expect 2 objects: red L, yellow dot"),

    ("04 · frame enclosing dot", Grid([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,3,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ]), "blue frame encloses green dot"),

    ("05 · three same-shape blobs", Grid([
        [0,1,1,0,1,1,0,1,1,0],
        [0,1,1,0,1,1,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]), "3 objects, identical shape_signature"),

    # ── colors ────────────────────────────────────────────────────────────────
    ("06 · background by frequency", Grid([
        [0,0,0,0,0],
        [0,0,3,0,0],
        [0,0,0,0,0],
    ]), "frequency → 0; border → 0"),

    ("07 · background trap (large foreground)", Grid([
        [2,2,2,2,2],
        [2,1,1,1,2],
        [2,1,1,1,2],
        [2,1,1,1,2],
        [2,2,2,2,2],
    ]), "frequency picks 1 (bigger blob); border correctly picks 2"),

    ("08 · five distinct colors", Grid([
        [1,1,0,2,2],
        [0,0,0,0,0],
        [3,0,4,0,5],
    ]), "histogram shows 5 colors + background"),

    # ── symmetry ──────────────────────────────────────────────────────────────
    ("09 · perfect vertical symmetry", Grid([
        [0,1,0,1,0],
        [0,2,0,2,0],
        [0,1,0,1,0],
    ]), "vertical=True, score=1.0"),

    ("10 · perfect horizontal symmetry", Grid([
        [1,2,3,2,1],
        [0,0,0,0,0],
        [1,2,3,2,1],
    ]), "horizontal=True, score=1.0"),

    ("11 · 180° rotational symmetry", Grid([
        [1,0,0,0,2],
        [0,0,0,0,0],
        [2,0,0,0,1],
    ]), "rotation_180=True"),

    ("12 · 90° rotational symmetry (square)", Grid([
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ]), "rotation_90=False (no symmetry), check scores"),

    ("13 · near-symmetric (one wrong cell)", Grid([
        [0,1,0,1,0],
        [0,2,0,9,0],  # 9 breaks vertical symmetry
        [0,1,0,1,0],
    ]), "vertical_score ≈ 0.87, not 1.0"),

    # ── patterns ──────────────────────────────────────────────────────────────
    ("14 · perfect 2×2 tile", Grid([
        [1,2,1,2,1,2],
        [3,4,3,4,3,4],
        [1,2,1,2,1,2],
        [3,4,3,4,3,4],
    ]), "tile=(2,2), score=1.0"),

    ("15 · 2×3 tile", Grid([
        [1,2,3,1,2,3],
        [4,5,6,4,5,6],
        [1,2,3,1,2,3],
        [4,5,6,4,5,6],
    ]), "tile=(2,3), score=1.0"),

    ("16 · near-periodic (one broken cell)", Grid([
        [1,2,1,2,1,2],
        [3,4,3,4,3,4],
        [1,2,1,2,1,2],
        [3,4,3,9,3,4],  # 9 breaks the pattern
    ]), "found=False, score≈0.96"),

    # ── lines ─────────────────────────────────────────────────────────────────
    ("17 · horizontal and vertical lines", Grid([
        [0,0,0,0,0,0],
        [3,3,3,3,3,3],
        [0,0,0,2,0,0],
        [0,0,0,2,0,0],
        [0,0,0,2,0,0],
        [0,0,0,0,0,0],
    ]), "1 horizontal green line (len 6), 1 vertical red line (len 3)"),

    ("18 · diagonal lines", Grid([
        [7,0,0,0,0],
        [0,7,0,0,0],
        [0,0,7,0,0],
        [0,0,0,7,0],
        [0,0,0,0,7],
    ]), "1 diagonal-down orange line (len 5)"),

    # ── topology ──────────────────────────────────────────────────────────────
    ("19 · above/below/left/right", Grid([
        [0,0,1,0,0],
        [0,0,0,0,0],
        [2,0,0,0,3],
        [0,0,0,0,0],
        [0,0,4,0,0],
    ]), "1=above 4, 4=below 1, 2=left of 3, 3=right of 2"),

    ("20 · touching vs non-touching", Grid([
        [0,1,1,0,0,2,2],
        [0,1,1,3,0,2,2],
        [0,0,0,3,0,0,0],
    ]), "blue touches orange (3), red does not touch blue"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Module runners
# ══════════════════════════════════════════════════════════════════════════════

def run_objects(grid, notes):
    objs4 = extract_objects(grid, connectivity=4)
    objs8 = extract_objects(grid, connectivity=8)
    ok(f"4-conn: {len(objs4)} object(s)   8-conn: {len(objs8)} object(s)")
    for i, o in enumerate(objs4):
        name = COLOR_NAME[o.dominant_color]
        info(f"  obj{i}: color={o.dominant_color}({name}) size={o.size} "
             f"bbox={o.bounding_box} centroid=({o.centroid[0]:.1f},{o.centroid[1]:.1f})")
    if len(objs4) != len(objs8):
        ok(f"  connectivity matters: 4→{len(objs4)}  8→{len(objs8)}")


def run_colors(grid, notes):
    ca_freq   = analyze_colors(grid, background_method="frequency")
    ca_border = analyze_colors(grid, background_method="border")
    ok(f"background: frequency={ca_freq.background}  border={ca_border.background}")
    hist_str = "  ".join(f"{COLOR_NAME[c]}={v}" for c, v in sorted(ca_freq.histogram.items()))
    info(f"  histogram: {hist_str}")
    for color, cells in sorted(ca_freq.color_regions.items()):
        if color != 0:
            info(f"  color {color} ({COLOR_NAME[color]}): {len(cells)} cell(s)")


def run_symmetry(grid, notes):
    sr = analyze_symmetry(grid)
    flags = []
    if sr.vertical:    flags.append(f"vertical(axis={sr.vertical_axis})")
    if sr.horizontal:  flags.append(f"horizontal(axis={sr.horizontal_axis})")
    if sr.rotation_180:flags.append(f"rot180(center={sr.rotation_center})")
    if sr.rotation_90: flags.append(f"rot90")
    ok("symmetries: " + (", ".join(flags) if flags else "none"))
    info(f"  scores — V:{sr.vertical_score:.2f}  H:{sr.horizontal_score:.2f}  "
         f"R180:{sr.rotation_180_score:.2f}  R90:{sr.rotation_90_score:.2f}")


def run_patterns(grid, notes):
    pr = detect_periodicity(grid)
    if pr.found:
        ok(f"periodic tile ({pr.period_rows}×{pr.period_cols})  score={pr.score:.3f}")
        info("  tile:")
        render(pr.tile, indent=6)
    else:
        ok(f"no perfect tiling  best_score={pr.score:.3f}")


def run_lines(grid, notes):
    lines = find_lines(grid, min_length=3)
    ok(f"{len(lines)} line(s) found")
    for ln in lines:
        info(f"  {ln.direction:15s} color={ln.color}({COLOR_NAME[ln.color]}) "
             f"len={ln.length}  {ln.start}→{ln.end}")


def run_topology(grid, notes):
    objs = extract_objects(grid)
    if len(objs) < 2:
        info("  < 2 objects, nothing to relate")
        return
    rels = relate_all(objs, grid)
    # Only print each unordered pair once (lower id first)
    seen = set()
    for r in rels:
        key = (id(r.a), id(r.b)) if id(r.a) < id(r.b) else (id(r.b), id(r.a))
        if key in seen:
            continue
        seen.add(key)
        a_name = COLOR_NAME[r.a.dominant_color]
        b_name = COLOR_NAME[r.b.dominant_color]
        enc = ""
        if r.a_encloses_b: enc = "  [A encloses B]"
        if r.b_encloses_a: enc = "  [B encloses A]"
        ok(f"{a_name} → {b_name}: dir={r.direction}  dist={r.distance:.1f}  "
           f"touches={r.touches}{enc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for name, grid, notes in GRIDS:
        section(name)
        info(f"notes: {notes}")
        info(f"grid {grid.rows}×{grid.cols}:")
        render(grid)

        # Pick the relevant modules based on grid index (embedded in name)
        idx = int(name.split("·")[0].strip())

        sub("objects")
        run_objects(grid, notes)

        if idx in (6, 7, 8):
            sub("colors")
            run_colors(grid, notes)

        if idx in (9, 10, 11, 12, 13):
            sub("symmetry")
            run_symmetry(grid, notes)

        if idx in (14, 15, 16):
            sub("patterns")
            run_patterns(grid, notes)

        if idx in (17, 18):
            sub("lines")
            run_lines(grid, notes)

        if idx in (19, 20):
            sub("topology")
            run_topology(grid, notes)

        # Always run lines + symmetry + topology as bonus where interesting
        if idx in (4,):
            sub("topology (enclosure check)")
            run_topology(grid, notes)
        if idx in (1, 2, 3, 4, 5):
            sub("lines (quick check)")
            run_lines(grid, notes)

    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  All 20 grids done.{RESET}")
    print(f"{BOLD}{'═'*64}{RESET}\n")


if __name__ == "__main__":
    main()
