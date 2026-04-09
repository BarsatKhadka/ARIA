# Perception

The `perception/` package breaks a raw `Grid` into structured observations — objects, colors, symmetries, relations — that a solver can reason about. Each file is one focused analysis step. None of them modify grids; they only read and describe.

---

## Package layout

| File | What it produces |
|---|---|
| `perception/objects.py` | Connected components → list of `Object` dataclasses |
| `perception/colors.py` | Background detection, color histogram, color regions → `ColorAnalysis` |
| `perception/symmetry.py` | Vertical, horizontal, 180°, 90° symmetry tests + scores → `SymmetryResult` |
| `perception/patterns.py` | Tile periodicity detection — smallest repeating tile and period → `PeriodicityResult` |
| `perception/topology.py` | Pairwise spatial relations — direction, distance, touching, enclosure → `ObjectRelation` |
| `perception/lines.py` | Horizontal, vertical, diagonal run detection (3+ cells, same color) → `Line` |

---
---

# objects.py

**File:** `perception/objects.py`

Extracts connected components from a `Grid` and returns them as `Object` dataclasses. An object is a maximal group of same-colored, connected cells. This is usually the first thing a solver needs to know: how many objects are there, where are they, what do they look like?

---

## Why objects.py exists

Almost every ARC task is described in terms of objects: "move the red object to the right", "copy the small shape into each quadrant", "find the two objects that match and keep only them". Before a solver can reason at that level it needs a plain list of objects — their positions, sizes, shapes — rather than a raw grid of numbers. `objects.py` produces that list in one call.

---

## The `Object` dataclass

```python
@dataclass(frozen=True)
class Object:
    pixels:          FrozenSet[Tuple[int, int]]
    bounding_box:    Tuple[int, int, int, int]
    centroid:        Tuple[float, float]
    dominant_color:  int
    size:            int
    shape_signature: FrozenSet[Tuple[int, int]]
```

`Object` is frozen (immutable) and hashable — you can put objects in sets, use them as dict keys, or compare them directly.

### `pixels`

`FrozenSet` of every `(row, col)` that belongs to this object. These are absolute grid coordinates.

```python
obj.pixels  # frozenset({(1, 2), (1, 3), (2, 2)})
```

Use this to read the color of any member cell:

```python
color_at = {grid.get(r, c) for r, c in obj.pixels}  # all colors (usually one)
```

### `bounding_box`

`(min_row, min_col, max_row, max_col)` — the tightest rectangle that contains all pixels. Inclusive on all sides.

```python
min_r, min_c, max_r, max_c = obj.bounding_box
height = max_r - min_r + 1
width  = max_c - min_c + 1
```

### `centroid`

`(row, col)` as floats — the arithmetic mean of all pixel positions. Useful for computing distances between objects or checking alignment.

```python
cr, cc = obj.centroid  # e.g. (2.5, 4.0)
```

### `dominant_color`

The most common ARC color value (0–9) across all member pixels. For single-color objects (the common case) this is just the object's color. For multi-color objects it is the plurality color.

```python
obj.dominant_color  # e.g. 3
```

### `size`

Number of pixels in the object. Equivalent to `len(obj.pixels)`.

```python
obj.size  # e.g. 6
```

### `shape_signature`

`FrozenSet` of `(row, col)` offsets relative to the object's top-left corner — i.e. relative to `(min_row, min_col)`. Two objects have the same `shape_signature` if and only if they have the same shape, regardless of where they appear in the grid or what color they are.

```python
# An L-shape at (3,5) and the same L-shape at (0,0) have identical signatures
obj_a.shape_signature == obj_b.shape_signature  # True if same shape
```

This is the right field to use for grouping objects by shape, checking for mirror images (by comparing to a reflected signature), or checking whether a training object reappears in the test input.

---

## `extract_objects`

```python
def extract_objects(
    grid: Grid,
    connectivity: Literal[4, 8] = 4,
    background: int = 0,
    include_background: bool = False,
) -> List[Object]:
```

The main entry point. Runs BFS flood-fill over every unvisited non-background cell and returns one `Object` per connected component.

### Parameters

**`grid`**

The `Grid` to segment.

**`connectivity`**

How cells connect to their neighbors.

- `4` — cardinal directions only (up, down, left, right). Two cells that only touch diagonally are *not* connected. This is the default and matches most ARC tasks.
- `8` — cardinal *and* diagonal directions. Two cells touching only at a corner are connected.

```python
# 4-connectivity: these 5 cells → 5 separate objects (no cardinal adjacency)
# . 5 . 5 .
# . . 5 . .
# . 5 . 5 .
objs4 = extract_objects(grid, connectivity=4)  # 5 objects

# 8-connectivity: all 5 cells are diagonally adjacent → 1 object
objs8 = extract_objects(grid, connectivity=8)  # 1 object
```

**`background`**

The color value treated as background. Cells of this color are skipped and never become objects. Default is `0` (black), which is the background in essentially all ARC tasks.

```python
# Treat color 5 as background instead
objects = extract_objects(grid, background=5)
```

**`include_background`**

When `True`, background-colored cells are also segmented. Useful if you need to reason about the shape or size of the black regions.

```python
all_objects = extract_objects(grid, include_background=True)
```

### Return value

`List[Object]`, ordered top-to-bottom then left-to-right by discovery order (the order BFS encounters the first cell of each component, scanning row by row).

### Raises

`ValueError` if `connectivity` is not `4` or `8`.

---

## Common patterns

### Count objects of each color

```python
from collections import Counter

objects = extract_objects(grid)
by_color = Counter(o.dominant_color for o in objects)
# Counter({1: 3, 2: 1, 4: 2})
```

### Find the largest object

```python
largest = max(objects, key=lambda o: o.size)
```

### Find objects with a given shape

```python
target_sig = some_known_object.shape_signature
matches = [o for o in objects if o.shape_signature == target_sig]
```

### Check whether two objects are the same shape

```python
o1.shape_signature == o2.shape_signature
```

### Get the bounding-box subgrid for an object

```python
min_r, min_c, max_r, max_c = obj.bounding_box
crop = Grid(grid.data[min_r:max_r+1, min_c:max_c+1].tolist())
```

### Sort objects top-to-bottom

```python
objects_sorted = sorted(objects, key=lambda o: (o.bounding_box[0], o.bounding_box[1]))
```

### Find objects that touch a specific cell

```python
target = (3, 5)
touching = [o for o in objects if target in o.pixels]
```

---

## Connectivity: when to use 4 vs 8

Use **4-connectivity** (default) unless you have a specific reason not to. The vast majority of ARC tasks use cardinal adjacency when describing connected groups.

Use **8-connectivity** when:
- The task description or training examples imply diagonal connections are meaningful.
- Objects are defined by diagonal chains.
- You are segmenting a grid that uses diagonal lines (color 5 "grey" borders, for example).

When in doubt, extract under both and compare: if `len(objs4) == len(objs8)`, the choice doesn't matter for this grid. If they differ, inspect which segmentation matches the task's intent.

---

## Full API reference

### `Object` fields

| Field | Type | Description |
|---|---|---|
| `pixels` | `FrozenSet[Tuple[int,int]]` | Absolute `(row, col)` positions of all member cells |
| `bounding_box` | `Tuple[int,int,int,int]` | `(min_row, min_col, max_row, max_col)`, inclusive |
| `centroid` | `Tuple[float,float]` | Mean `(row, col)` across all pixels |
| `dominant_color` | `int` | Most frequent ARC color (0–9) in the object |
| `size` | `int` | Number of pixels |
| `shape_signature` | `FrozenSet[Tuple[int,int]]` | Offsets from top-left corner — position-invariant shape fingerprint |

### `extract_objects` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `Grid` | required | Source grid to segment |
| `connectivity` | `4` or `8` | `4` | Neighbor kernel: cardinal only or cardinal + diagonal |
| `background` | `int` | `0` | Color to skip; cells of this value are never returned as objects |
| `include_background` | `bool` | `False` | When True, background cells are also segmented |

Returns `List[Object]`.

---

---

# colors.py

**File:** `perception/colors.py`

Analyses the color distribution of a `Grid` and returns a `ColorAnalysis` dataclass. Three concerns are covered in one pass: which color is the background, how many cells of each color exist, and where every cell of each color sits.

---

## Why colors.py exists

Object segmentation (`objects.py`) already needs a background color to know what to skip. But the background is not always obvious — a large foreground object can dominate the pixel count and fool a naive check. `colors.py` provides two detection strategies and, in the same sweep, builds the histogram and region index that many solvers need before they can even frame the problem.

---

## The `ColorAnalysis` dataclass

```python
@dataclass(frozen=True)
class ColorAnalysis:
    histogram:     Dict[int, int]
    background:    int
    color_regions: Dict[int, FrozenSet[Tuple[int, int]]]
```

`ColorAnalysis` is frozen and can be passed around freely without risk of mutation.

### `histogram`

`Dict[int, int]` — maps each color present in the grid to its pixel count.

```python
ca.histogram  # {0: 48, 3: 6, 7: 2}
```

Use this to rank colors by frequency, detect singleton colors, or check whether the grid is monochromatic.

### `background`

`int` — the color identified as the grid's background by the chosen method.

```python
ca.background  # 0
```

Pass this directly to `extract_objects(grid, background=ca.background)` so object extraction uses the right background.

### `color_regions`

`Dict[int, FrozenSet[Tuple[int, int]]]` — maps each color to the complete set of `(row, col)` cells carrying that color. Adjacency is irrelevant: every cell of that color is included regardless of whether it touches another.

```python
ca.color_regions[3]  # frozenset({(1,2), (1,3), (4,7)})
```

Use this when you need "all red cells" without caring whether they form one object or many.

---

## `analyze_colors`

```python
def analyze_colors(
    grid: Grid,
    background_method: str = "frequency",
) -> ColorAnalysis:
```

The main entry point. Computes histogram, background, and color regions in a single grid scan.

### Parameters

**`grid`**

The `Grid` to analyse.

**`background_method`**

Controls how the background color is chosen.

- `"frequency"` *(default)* — most frequent color in the entire grid. Works well when the background floods most of the grid.
- `"border"` — most frequent color among the cells on the four edges of the grid. Use this when a large foreground object dominates the pixel count and would fool the frequency check. The background almost always appears on the grid's outermost ring even when it is outnumbered overall.

```python
# Default: whole-grid frequency
ca = analyze_colors(grid)

# Border heuristic — immune to large central objects
ca = analyze_colors(grid, background_method="border")
```

### Return value

`ColorAnalysis` with all three fields populated.

### Raises

`ValueError` if `background_method` is not `"frequency"` or `"border"`.

---

## When to use each background method

**`"frequency"`** — the standard. If a 30×30 grid has 850 black cells out of 900, black is obviously the canvas. Use this by default.

**`"border"`** — the trap-killer. Imagine a puzzle where the background is red but a massive blue square covers 60 % of the grid. `"frequency"` gets tricked into calling blue the background. `"border"` scans only the outermost ring, sees it is 100 % red, and correctly identifies red as the canvas.

A safe strategy when unsure: run both and compare.

```python
ca_freq   = analyze_colors(grid, background_method="frequency")
ca_border = analyze_colors(grid, background_method="border")

if ca_freq.background != ca_border.background:
    # The two methods disagree — likely a large dominant foreground object.
    # Prefer the border result.
    ca = ca_border
else:
    ca = ca_freq
```

---

## Background detection in isolation

Both heuristics are also available as standalone functions:

```python
from perception.colors import detect_background_by_frequency, detect_background_by_border

bg = detect_background_by_frequency(grid)  # most frequent color overall
bg = detect_background_by_border(grid)     # most frequent color on the four edges
```

---

## Common patterns

### Use detected background for object extraction

```python
from perception.colors import analyze_colors
from perception.objects import extract_objects

ca = analyze_colors(grid)
objects = extract_objects(grid, background=ca.background)
```

### Find the rarest (likely signal) color

```python
ca = analyze_colors(grid)
rarest = min(ca.histogram, key=ca.histogram.get)
```

### Check how many distinct non-background colors exist

```python
ca = analyze_colors(grid)
signal_colors = [c for c in ca.histogram if c != ca.background]
len(signal_colors)  # e.g. 3
```

### Get all cells of a specific color

```python
ca = analyze_colors(grid)
red_cells = ca.color_regions.get(2, frozenset())
```

### Check whether two colors always appear together (same cells)

```python
ca = analyze_colors(grid)
ca.color_regions[2] == ca.color_regions[5]
```

---

## Full API reference

### `ColorAnalysis` fields

| Field | Type | Description |
|---|---|---|
| `histogram` | `Dict[int, int]` | Color → pixel count for every color present |
| `background` | `int` | Detected background color (0–9) |
| `color_regions` | `Dict[int, FrozenSet[Tuple[int,int]]]` | Color → all `(row, col)` cells of that color |

### `analyze_colors` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `Grid` | required | Source grid to analyse |
| `background_method` | `str` | `"frequency"` | `"frequency"` or `"border"` — how to pick the background color |

Returns `ColorAnalysis`.

### Standalone helpers

| Function | Returns | Description |
|---|---|---|
| `detect_background_by_frequency(grid)` | `int` | Most frequent color in the whole grid |
| `detect_background_by_border(grid)` | `int` | Most frequent color among border cells |
| `color_histogram(grid)` | `Dict[int, int]` | Color counts |
| `color_regions(grid)` | `Dict[int, FrozenSet[...]]` | All cells grouped by color |

---

---

# symmetry.py

**File:** `perception/symmetry.py`

Tests a `Grid` or `Object` for four symmetry types and returns a `SymmetryResult` with boolean flags, axis/center coordinates, and per-symmetry scores. A score measures what fraction of cells match their reflected or rotated counterpart, so near-symmetric grids can be detected too.

---

## Why symmetry.py exists

Many ARC tasks are built around symmetric structure: "the grid is almost vertically symmetric — find the cell that breaks it and fix it", or "copy this shape which has 180° symmetry". Before a solver can act on that structure it needs a reliable yes/no answer per symmetry type, plus enough quantitative detail (score, axis) to handle imperfect cases and to describe where the symmetry lives.

---

## The `SymmetryResult` dataclass

```python
@dataclass(frozen=True)
class SymmetryResult:
    vertical:           bool
    horizontal:         bool
    rotation_180:       bool
    rotation_90:        bool
    vertical_score:     float
    horizontal_score:   float
    rotation_180_score: float
    rotation_90_score:  float
    vertical_axis:      Optional[float]
    horizontal_axis:    Optional[float]
    rotation_center:    Optional[Tuple[float, float]]
```

`SymmetryResult` is frozen. All fields are always populated — scores are present even when the symmetry flag is `False`.

### Boolean flags

| Field | Meaning |
|---|---|
| `vertical` | True if the grid/object is symmetric across a vertical axis (left ↔ right) |
| `horizontal` | True if the grid/object is symmetric across a horizontal axis (top ↔ bottom) |
| `rotation_180` | True if the grid/object maps onto itself under 180° rotation |
| `rotation_90` | True if the grid/object maps onto itself under 90° rotation (implies 180°; only possible for square targets) |

### Score fields

Each score is a `float` in `[0.0, 1.0]` — the fraction of cells that match their reflected/rotated counterpart. A score of `1.0` is perfect symmetry; `0.5` means half the cells agree.

```python
sr.vertical_score      # 1.0  → perfect left-right symmetry
sr.rotation_180_score  # 0.87 → nearly, but not quite, 180°-symmetric
```

Scores are computed regardless of the boolean result, so you can use them as continuous signals even when `threshold=1.0`.

### Axis / center fields

| Field | Type | Meaning |
|---|---|---|
| `vertical_axis` | `Optional[float]` | Column index of the vertical reflection axis. `None` if `vertical` is `False`. May be `.5` for even-width targets (axis lies between two columns). |
| `horizontal_axis` | `Optional[float]` | Row index of the horizontal reflection axis. `None` if `horizontal` is `False`. |
| `rotation_center` | `Optional[Tuple[float, float]]` | `(row, col)` center of rotation. `None` if neither `rotation_180` nor `rotation_90` is `True`. |

---

## `analyze_symmetry`

```python
def analyze_symmetry(
    target: Union[Grid, Object],
    threshold: float = 1.0,
) -> SymmetryResult:
```

The single entry point. Pass either a `Grid` or an `Object`; the function dispatches automatically.

### Parameters

**`target`**

A `Grid` (full-grid analysis) or an `Object` (shape analysis within the object's bounding box, color-independent).

**`threshold`**

Minimum score to declare a symmetry present. Default `1.0` means perfect only.

```python
# Perfect symmetry only (default)
sr = analyze_symmetry(grid)

# Allow up to 10 % mismatch
sr = analyze_symmetry(grid, threshold=0.9)
```

### Return value

`SymmetryResult`.

### Raises

`TypeError` if `target` is not a `Grid` or `Object`.

---

## Grid vs Object analysis

**Grid** — the full `rows × cols` array is tested. Axes and centers are expressed in absolute grid coordinates.

**Object** — the object is rendered into its tight bounding box (non-member cells filled with a sentinel). Only the *shape* (pixel presence/absence) is tested; color is ignored. This means two objects of different colors with the same shape report identical symmetry. Axes and centers are expressed in bounding-box-local coordinates.

```python
# Grid symmetry
sr = analyze_symmetry(grid)

# Object shape symmetry
objects = extract_objects(grid)
sr = analyze_symmetry(objects[0])
```

---

## Rotation symmetry and non-square targets

90° rotational symmetry is only possible when the target is square (`rows == cols`). For non-square grids or objects, `rotation_90` is always `False` and `rotation_90_score` is `0.0`. 180° symmetry has no such restriction.

---

## Common patterns

### Check if a grid has any symmetry

```python
sr = analyze_symmetry(grid)
any_symmetry = sr.vertical or sr.horizontal or sr.rotation_180 or sr.rotation_90
```

### Find which objects are vertically symmetric

```python
objects = extract_objects(grid)
symmetric = [o for o in objects if analyze_symmetry(o).vertical]
```

### Rank objects by how close they are to 180° symmetric

```python
objects = extract_objects(grid)
ranked = sorted(objects, key=lambda o: analyze_symmetry(o).rotation_180_score, reverse=True)
```

### Detect near-symmetry (broken by one misplaced cell)

```python
sr = analyze_symmetry(grid, threshold=0.0)  # get raw scores
if sr.vertical_score > 0.95:
    # Almost vertically symmetric — probably one cell is wrong
    pass
```

### Get the vertical axis of a symmetric grid

```python
sr = analyze_symmetry(grid)
if sr.vertical:
    print(f"Vertical axis at column {sr.vertical_axis}")
    # Even-width 4-col grid → axis = 1.5 (between cols 1 and 2)
    # Odd-width 5-col grid  → axis = 2.0 (col 2 is the mirror column)
```

---

## Full API reference

### `SymmetryResult` fields

| Field | Type | Description |
|---|---|---|
| `vertical` | `bool` | Left-right reflection symmetry |
| `horizontal` | `bool` | Top-bottom reflection symmetry |
| `rotation_180` | `bool` | 180° rotational symmetry |
| `rotation_90` | `bool` | 90° rotational symmetry (square targets only) |
| `vertical_score` | `float` | Fraction of cells matching under left-right reflection |
| `horizontal_score` | `float` | Fraction of cells matching under top-bottom reflection |
| `rotation_180_score` | `float` | Fraction of cells matching under 180° rotation |
| `rotation_90_score` | `float` | Fraction of cells matching under 90° rotation |
| `vertical_axis` | `Optional[float]` | Column of vertical axis; `None` if not symmetric |
| `horizontal_axis` | `Optional[float]` | Row of horizontal axis; `None` if not symmetric |
| `rotation_center` | `Optional[Tuple[float,float]]` | `(row, col)` rotation center; `None` if no rotational symmetry |

### `analyze_symmetry` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target` | `Grid` or `Object` | required | What to analyse |
| `threshold` | `float` | `1.0` | Minimum score to declare symmetry present |

Returns `SymmetryResult`.

### Standalone helpers

| Function | Returns | Description |
|---|---|---|
| `analyze_grid_symmetry(grid, threshold)` | `SymmetryResult` | Grid-only entry point |
| `analyze_object_symmetry(obj, threshold)` | `SymmetryResult` | Object-only entry point |

---

---

# patterns.py

**File:** `perception/patterns.py`

Tests whether a `Grid` tiles with a repeating 2D pattern. Tries every tile size from 2×2 up to half the grid dimensions and returns the smallest tile whose repetition reconstructs the grid, along with a match score.

---

## Why patterns.py exists

Some ARC tasks present a grid that is clearly a tiled repetition of a smaller motif — the task might ask you to extract that motif, count how many times it repeats, or identify where the tiling breaks. Before a solver can do any of that it needs to know: "is this grid periodic, and if so what is the tile?"

---

## The `PeriodicityResult` dataclass

```python
@dataclass(frozen=True)
class PeriodicityResult:
    found:        bool
    tile:         Optional[Grid]
    period_rows:  Optional[int]
    period_cols:  Optional[int]
    score:        float
```

### `found`

`True` if a tile was found whose repetition achieves the required `threshold`. `False` otherwise.

### `tile`

The minimal repeating `Grid` tile — the top-left `period_rows × period_cols` subgrid of the input. `None` when `found` is `False`.

```python
pr.tile  # Grid object you can inspect, pass to other functions, or compare
```

### `period_rows` / `period_cols`

The height and width of the tile. Together they define the repetition period in both dimensions.

```python
pr.period_rows  # 3  → pattern repeats every 3 rows
pr.period_cols  # 2  → pattern repeats every 2 columns
```

### `score`

Fraction of grid cells (0.0–1.0) that are consistent with the detected tiling. When `found` is `True` this equals (or exceeds) `threshold`. When `found` is `False` this is the **best score seen across all tested tile sizes** — useful for detecting near-periodic grids even when no tile reaches `threshold`.

```python
pr.score  # 0.94 → 94 % of cells match the best tile found
```

---

## `detect_periodicity`

```python
def detect_periodicity(
    grid: Grid,
    threshold: float = 1.0,
) -> PeriodicityResult:
```

The single entry point.

### How it works

For every tile height `th` from `2` to `rows // 2` and every tile width `tw` from `2` to `cols // 2`:

1. Take the top-left `th × tw` subgrid as the candidate tile.
2. Tile it to cover the full grid dimensions (cropping the last partial repetition).
3. Count what fraction of cells match (`score`).
4. If `score ≥ threshold`, record this tile as a candidate.

After all sizes are tested, the candidate with the **smallest area** (`th × tw`) is returned. Ties in area are broken by smaller `th` then smaller `tw`.

### Parameters

**`grid`**

The source grid to test.

**`threshold`**

Minimum match fraction to declare periodicity found. Default `1.0` (perfect tiling only).

```python
# Perfect tiling only (default)
pr = detect_periodicity(grid)

# Allow up to 10 % noise or corruption
pr = detect_periodicity(grid, threshold=0.9)
```

### Return value

`PeriodicityResult`.

### Minimum grid size

The grid must be at least `4 × 4` to have any candidate tile size (the minimum tile is 2×2 inside a 4×4 grid). Smaller grids immediately return `found=False, score=0.0`.

---

## Score when `found` is `False`

When no tile reaches `threshold`, `score` still tells you something:

```python
pr = detect_periodicity(grid)
if not pr.found and pr.score > 0.9:
    # The grid is almost periodic — one or two cells deviate.
    # Lower the threshold or look for the corrupted cell.
    pr2 = detect_periodicity(grid, threshold=0.9)
```

---

## Common patterns

### Check if the grid is a tiled repetition

```python
pr = detect_periodicity(grid)
if pr.found:
    print(f"Tile: {pr.period_rows}×{pr.period_cols}")
    print(pr.tile)
```

### Extract the tile for further analysis

```python
pr = detect_periodicity(grid)
if pr.found:
    objects_in_tile = extract_objects(pr.tile)
    sym = analyze_symmetry(pr.tile)
```

### Count how many times the tile repeats

```python
pr = detect_periodicity(grid)
if pr.found:
    reps_r = grid.rows // pr.period_rows
    reps_c = grid.cols // pr.period_cols
    total  = reps_r * reps_c
```

### Detect near-periodic grids (noisy or broken tiling)

```python
pr = detect_periodicity(grid, threshold=0.0)  # score only, no threshold
if pr.score > 0.85:
    # Mostly periodic — likely a tiling task with deliberate anomalies
    pass
```

---

## Full API reference

### `PeriodicityResult` fields

| Field | Type | Description |
|---|---|---|
| `found` | `bool` | True if a repeating tile was found at or above `threshold` |
| `tile` | `Optional[Grid]` | The minimal repeating tile; `None` if not found |
| `period_rows` | `Optional[int]` | Tile height (vertical period); `None` if not found |
| `period_cols` | `Optional[int]` | Tile width (horizontal period); `None` if not found |
| `score` | `float` | Best match fraction seen; equals threshold when found, otherwise the closest miss |

### `detect_periodicity` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `Grid` | required | Source grid to test |
| `threshold` | `float` | `1.0` | Minimum cell-match fraction to declare periodicity found |

Returns `PeriodicityResult`.

---

---

# topology.py

**File:** `perception/topology.py`

Computes pairwise spatial relationships between objects. For every pair it answers four questions: where is one relative to the other, how far apart are they, do they touch, and does one enclose the other?

---

## Why topology.py exists

Most ARC tasks are relational — "move the object that is above the red one", "find the enclosed object", "connect the two touching shapes". Raw object lists give no relational structure. `topology.py` builds that structure as a flat list of `ObjectRelation` records that a solver can filter, sort, or group without re-reading the grid.

---

## The `ObjectRelation` dataclass

```python
@dataclass(frozen=True)
class ObjectRelation:
    a:           Object
    b:           Object
    direction:   Direction
    distance:    float
    touches:     bool
    a_encloses_b: bool
    b_encloses_a: bool
```

`ObjectRelation` is frozen. Relations are ordered: `(a, b)` and `(b, a)` are separate records with mirrored directions.

### `direction`

Where `b` sits relative to `a`, based on the vector between centroids.

```
"above" | "below" | "left" | "right"
"above-left" | "above-right" | "below-left" | "below-right"
"same"
```

A direction is **cardinal** when the primary axis is at least 2× the secondary axis. Otherwise it is **diagonal**. `"same"` only when centroids are identical (e.g. overlapping objects).

```python
rel.direction  # "above-right"
```

### `distance`

Euclidean distance between the two centroids, in grid cells.

```python
rel.distance  # 5.385
```

### `touches`

`True` if any pixel of `a` is 4-adjacent (cardinal neighbor) to any pixel of `b`. Objects that share a diagonal corner do **not** touch.

```python
rel.touches  # True
```

### `a_encloses_b` / `b_encloses_a`

`True` if one object forms a closed wall around the other.

Enclosure requires **both**:
1. The outer object's bounding box strictly contains the inner object's bounding box.
2. A flood fill of background cells starting from outside the grid cannot reach any background cell inside the inner object's bounding box — meaning the outer object forms a sealed border with no gaps.

```python
rel.a_encloses_b  # True  → a is a frame/ring around b
rel.b_encloses_a  # False
```

---

## Direction in detail

The 2:1 threshold prevents centroid noise from flipping a clearly-vertical pair into a diagonal.

```
b is at dr=+4, dc=+1 relative to a
→ |dr| = 4 ≥ 2×|dc| = 2  → "below"  (not "below-right")

b is at dr=+3, dc=+2 relative to a
→ neither axis dominates  → "below-right"
```

---

## Enclosure in detail

A bounding-box containment check alone is not enough — a C-shaped object contains a bounding box but has a gap. The flood fill closes that gap:

- Seed from any background cell on the grid border.
- Flood through background cells (skipping outer-object pixels).
- If the flood reaches a background cell inside the inner object's bounding box, the wall has a gap → **not enclosed**.
- If the flood is completely blocked → truly enclosed.

This correctly handles frames, rings, and L-shapes (L-shape has a gap → not enclosed).

---

## `relate`

```python
def relate(a: Object, b: Object, grid: Grid, background: int = 0) -> ObjectRelation:
```

Compute the relation for a single ordered pair. Use this when you already know the two objects you care about.

```python
rel = relate(obj_a, obj_b, grid)
```

---

## `relate_all`

```python
def relate_all(
    objects: List[Object],
    grid: Grid,
    background: int = 0,
) -> List[ObjectRelation]:
```

Compute relations for every ordered pair. Returns `n × (n-1)` records for `n` objects. Both `(a, b)` and `(b, a)` are included, so filtering by `rel.a` gives all relations from a given object's point of view.

```python
relations = relate_all(objects, grid)
```

---

## Common patterns

### Find all objects directly above a given object

```python
relations = relate_all(objects, grid)
above_target = [r.a for r in relations if r.b is target and r.direction == "above"]
```

### Find the closest object to a given object

```python
relations = relate_all(objects, grid)
from_target = [r for r in relations if r.a is target]
nearest = min(from_target, key=lambda r: r.distance)
```

### Find all touching pairs (undirected)

```python
relations = relate_all(objects, grid)
# (a,b) and (b,a) both appear; keep one direction to avoid duplicates
touching = [(r.a, r.b) for r in relations if r.touches and id(r.a) < id(r.b)]
```

### Find enclosed objects

```python
relations = relate_all(objects, grid)
enclosed = [(r.a, r.b) for r in relations if r.a_encloses_b]
# enclosed[i] = (frame_object, inner_object)
```

### Check if two specific objects touch

```python
rel = relate(obj_a, obj_b, grid)
rel.touches  # True / False
```

### Sort objects by distance from a reference

```python
relations = relate_all(objects, grid)
from_ref = sorted(
    [r for r in relations if r.a is reference],
    key=lambda r: r.distance
)
```

---

## Full API reference

### `ObjectRelation` fields

| Field | Type | Description |
|---|---|---|
| `a` | `Object` | Reference object |
| `b` | `Object` | Comparison object |
| `direction` | `Direction` | Position of `b` relative to `a` |
| `distance` | `float` | Euclidean centroid-to-centroid distance |
| `touches` | `bool` | True if any pixel of `a` is 4-adjacent to any pixel of `b` |
| `a_encloses_b` | `bool` | True if `a` forms a sealed border around `b` |
| `b_encloses_a` | `bool` | True if `b` forms a sealed border around `a` |

### `Direction` values

`"above"`, `"below"`, `"left"`, `"right"`, `"above-left"`, `"above-right"`, `"below-left"`, `"below-right"`, `"same"`

### `relate` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `a`, `b` | `Object` | required | The two objects to relate |
| `grid` | `Grid` | required | Source grid (used for enclosure flood fill) |
| `background` | `int` | `0` | Background color |

Returns `ObjectRelation`.

### `relate_all` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objects` | `List[Object]` | required | All objects to pair up |
| `grid` | `Grid` | required | Source grid |
| `background` | `int` | `0` | Background color |

Returns `List[ObjectRelation]` with `n × (n-1)` entries.

---

---

# lines.py

**File:** `perception/lines.py`

Detects horizontal, vertical, and diagonal runs of the same color in a `Grid`. Returns every maximal run of length ≥ `min_length` as a `Line` dataclass with start, end, direction, color, and the full ordered cell sequence.

---

## Why lines.py exists

Many ARC tasks hinge on lines — a dividing bar between two regions, a row that acts as a separator, a diagonal streak that must be extended, a cross pattern. Object extraction won't reliably isolate these because a long row of cells can look like a thin rectangle rather than a line. `lines.py` scans all four directions explicitly and gives the solver clean `Line` objects it can reason about directly.

---

## The `Line` dataclass

```python
@dataclass(frozen=True)
class Line:
    start:     Tuple[int, int]
    end:       Tuple[int, int]
    direction: LineDirection
    color:     int
    length:    int
    cells:     Tuple[Tuple[int, int], ...]
```

`Line` is frozen. `cells` is an ordered tuple from `start` to `end`, always in the scan direction.

### `start` / `end`

`(row, col)` of the first and last cell in the run.

```python
line.start  # (2, 0)
line.end    # (2, 7)
```

### `direction`

One of four values:

| Value | Meaning | Step |
|---|---|---|
| `"horizontal"` | same row, cols increase | → |
| `"vertical"` | same col, rows increase | ↓ |
| `"diagonal-down"` | row and col both increase | ↘ |
| `"diagonal-up"` | row decreases, col increases | ↗ |

### `color`

ARC color (0–9) of every cell in the run.

### `length`

Number of cells. Always ≥ `min_length`.

### `cells`

Ordered `Tuple` of `(row, col)` from `start` to `end`. Use this to iterate over the line's cells, convert to a set for membership tests, or pass to other perception functions.

```python
line.cells  # ((2,0), (2,1), (2,2), (2,3))
cell_set = set(line.cells)
```

---

## Adaptive minimum length

A fixed threshold of 3 causes the detector to fire on nearly every ARC grid — three same-color pixels in a row appear constantly as incidental noise, not as structural lines. `find_lines` therefore computes the threshold adaptively based on grid size:

```
min_length = max(3, max(rows, cols) // 3)
```

| Grid size | Adaptive min_length |
|---|---|
| 5×5 | 3 |
| 9×9 | 3 |
| 12×12 | 4 |
| 15×15 | 5 |
| 20×20 | 6 |
| 30×30 | 10 |

The standalone helper `adaptive_min_length(grid)` returns this value without running the full scan.

```python
from perception.lines import adaptive_min_length
threshold = adaptive_min_length(grid)  # e.g. 7 for a 20×30 grid
```

Pass an explicit integer to `min_length` to override the adaptive threshold when you need a specific value.

---

## `find_lines`

```python
def find_lines(
    grid: Grid,
    min_length: Optional[int] = None,
    background: int = 0,
    include_background: bool = False,
) -> List[Line]:
```

The single entry point. Scans all four directions and returns every qualifying run.

### Parameters

**`grid`**

The source grid.

**`min_length`**

Minimum run length to report. When `None` (default), computed adaptively as `max(3, max(rows, cols) // 3)`. Pass an explicit integer to override.

**`background`**

Color treated as background. Runs of this color are excluded unless `include_background` is `True`. Default `0`.

**`include_background`**

When `True`, background-colored runs are also returned. Useful when you need to reason about the shape of background regions.

### Return order

Horizontal lines (row 0 → last row), then vertical (col 0 → last col), then diagonal-down, then diagonal-up. Within each direction, lines appear in scan order.

### Raises

`ValueError` if the effective `min_length < 2`.

---

## What counts as a line

A line is a **maximal** same-color run in one direction. "Maximal" means it cannot be extended: the cell before `start` and the cell after `end` (if they exist) are a different color. Two adjacent cells of the same color in the same direction are always part of the same line, never separate lines.

```
Row: 0 0 3 3 3 3 0 0
          ↑           ↑
        start        end  → one Line(color=3, length=4)
```

---

## Common patterns

### Find all lines of a specific color

```python
lines = find_lines(grid)
red_lines = [l for l in lines if l.color == 2]
```

### Find horizontal lines that span the full grid width

```python
lines = find_lines(grid)
full_width = [l for l in lines if l.direction == "horizontal" and l.length == grid.cols]
```

### Find lines that pass through a specific cell

```python
target = (3, 5)
lines = find_lines(grid)
through = [l for l in lines if target in set(l.cells)]
```

### Check if two objects are separated by a line

```python
lines = find_lines(grid)
separators = [l for l in lines if l.direction == "vertical" and l.length >= grid.rows]
```

### Find the longest line

```python
lines = find_lines(grid)
longest = max(lines, key=lambda l: l.length)
```

### Group lines by direction

```python
from collections import defaultdict
lines = find_lines(grid)
by_dir = defaultdict(list)
for l in lines:
    by_dir[l.direction].append(l)
```

### Find lines that cross (share a cell)

```python
lines = find_lines(grid)
cell_sets = [set(l.cells) for l in lines]
for i, a in enumerate(lines):
    for b in lines[i+1:]:
        if set(a.cells) & set(b.cells):
            print(f"{a.direction} and {b.direction} cross")
```

---

## Full API reference

### `Line` fields

| Field | Type | Description |
|---|---|---|
| `start` | `Tuple[int, int]` | `(row, col)` of the first cell |
| `end` | `Tuple[int, int]` | `(row, col)` of the last cell |
| `direction` | `LineDirection` | `"horizontal"`, `"vertical"`, `"diagonal-down"`, or `"diagonal-up"` |
| `color` | `int` | ARC color (0–9) of every cell in the run |
| `length` | `int` | Number of cells in the run |
| `cells` | `Tuple[Tuple[int,int], ...]` | Ordered cell sequence from `start` to `end` |

### `find_lines` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `Grid` | required | Source grid to scan |
| `min_length` | `Optional[int]` | `None` | Minimum run length; `None` → adaptive `max(3, max(rows,cols)//3)` |
| `background` | `int` | `0` | Color to exclude from results |
| `include_background` | `bool` | `False` | When True, background-colored runs are included |

Returns `List[Line]`.

### Standalone helpers

| Function | Returns | Description |
|---|---|---|
| `adaptive_min_length(grid)` | `int` | `max(3, max(rows,cols)//3)` — the threshold used when `min_length=None` |

---

---

# report.py

**File:** `perception/report.py`

Aggregates all perception modules into a single structured report for a task. Produces a `PerceptionReport` by running every module (objects, colors, symmetry, patterns, lines, topology) on every training pair and collecting both per-grid observations and cross-pair summaries.

---

## Why report.py exists

Individual perception modules each answer one question about one grid. Agents need a unified view across all training pairs — "do all inputs have the same background?", "does any input have enclosure?", "how many objects are in the input vs the output?". `report.py` runs everything once and packages the answers into frozen dataclasses ready to pass directly to an agent.

---

## Dataclasses

### `GridPerception`

All perception module outputs for a single grid.

```python
@dataclass(frozen=True)
class GridPerception:
    # Objects
    objects:       Tuple[Object, ...]
    n_objects:     int
    background:    int

    # Colors
    colors:        ColorAnalysis
    n_colors:      int

    # Symmetry
    symmetry:      SymmetryResult
    has_symmetry:  bool           # True if any symmetry type holds

    # Periodicity
    periodicity:   PeriodicityResult
    is_periodic:   bool

    # Lines
    lines:         Tuple[Line, ...]
    n_lines:       int

    # Topology
    relations:     Tuple[ObjectRelation, ...]
    has_enclosure: bool           # True if any object pair has an enclosure relation
```

### `PairPerception`

Perception of one input/output training pair.

```python
@dataclass(frozen=True)
class PairPerception:
    pair_index: int
    input:      GridPerception
    output:     GridPerception
```

### `PerceptionReport`

Full report for a task, aggregated across all training pairs.

```python
@dataclass(frozen=True)
class PerceptionReport:
    n_pairs:          int
    pair_perceptions: Tuple[PairPerception, ...]

    # Cross-pair summaries (training inputs only)
    all_inputs_have_symmetry:  bool
    any_input_has_symmetry:    bool
    all_inputs_periodic:       bool
    any_input_periodic:        bool
    all_inputs_have_enclosure: bool
    any_input_has_enclosure:   bool
    consistent_background:     Optional[int]  # None if varies across pairs
    consistent_n_objects:      Optional[int]  # None if varies across pairs
```

Cross-pair summaries are derived from **input grids only**. Output grids are available via `pair_perceptions[i].output` but are not summarised at the report level — the agent can inspect them directly per pair.

---

## `perceive`

```python
def perceive(task: Task) -> PerceptionReport:
```

The single entry point. Runs all six perception modules on every training pair and returns a `PerceptionReport`.

```python
from core.task import Task
from perception.report import perceive

tasks = Task.load_all("data/arc-agi_training_challenges.json",
                      "data/arc-agi_training_solutions.json")
report = perceive(tasks["00576224"])
```

---

## Common patterns

### Check cross-pair structure at a glance

```python
report = perceive(task)
print(report.consistent_background)   # e.g. 0 (black) or None if it varies
print(report.consistent_n_objects)    # e.g. 3 or None
print(report.any_input_has_symmetry)  # True / False
print(report.any_input_has_enclosure) # True / False
```

### Inspect per-pair input vs output object counts

```python
for pp in report.pair_perceptions:
    print(f"pair {pp.pair_index}: "
          f"input_objs={pp.input.n_objects}  "
          f"output_objs={pp.output.n_objects}")
```

### Get all objects from the first training input

```python
pp = report.pair_perceptions[0]
for obj in pp.input.objects:
    print(obj.dominant_color, obj.size, obj.centroid)
```

### Check whether the output gains or loses objects

```python
deltas = [pp.output.n_objects - pp.input.n_objects
          for pp in report.pair_perceptions]
# deltas consistent → likely a count-change rule
```

### Feed report to an agent (summary dict)

```python
report = perceive(task)
summary = {
    "n_pairs":            report.n_pairs,
    "background":         report.consistent_background,
    "n_objects":          report.consistent_n_objects,
    "any_symmetry":       report.any_input_has_symmetry,
    "any_periodic":       report.any_input_periodic,
    "any_enclosure":      report.any_input_has_enclosure,
    "pair_object_deltas": [
        pp.output.n_objects - pp.input.n_objects
        for pp in report.pair_perceptions
    ],
}
```

---

## Full API reference

### `GridPerception` fields

| Field | Type | Description |
|---|---|---|
| `objects` | `Tuple[Object, ...]` | Extracted objects (4-connectivity, non-background) |
| `n_objects` | `int` | Number of objects |
| `background` | `int` | Detected background color (frequency method) |
| `colors` | `ColorAnalysis` | Full color analysis including histogram and regions |
| `n_colors` | `int` | Number of distinct colors present |
| `symmetry` | `SymmetryResult` | Full symmetry analysis |
| `has_symmetry` | `bool` | True if any of the four symmetry types holds |
| `periodicity` | `PeriodicityResult` | Full periodicity analysis |
| `is_periodic` | `bool` | True if a repeating tile was found |
| `lines` | `Tuple[Line, ...]` | All detected lines (adaptive min_length) |
| `n_lines` | `int` | Number of lines |
| `relations` | `Tuple[ObjectRelation, ...]` | All pairwise object relations; empty if fewer than 2 objects |
| `has_enclosure` | `bool` | True if any `ObjectRelation` has `a_encloses_b` or `b_encloses_a` |

### `PairPerception` fields

| Field | Type | Description |
|---|---|---|
| `pair_index` | `int` | Zero-based training pair index |
| `input` | `GridPerception` | Perception of the input grid |
| `output` | `GridPerception` | Perception of the output grid |

### `PerceptionReport` fields

| Field | Type | Description |
|---|---|---|
| `n_pairs` | `int` | Number of training pairs |
| `pair_perceptions` | `Tuple[PairPerception, ...]` | Per-pair observations |
| `all_inputs_have_symmetry` | `bool` | Every input grid has at least one symmetry |
| `any_input_has_symmetry` | `bool` | At least one input grid has symmetry |
| `all_inputs_periodic` | `bool` | Every input grid is periodic |
| `any_input_periodic` | `bool` | At least one input grid is periodic |
| `all_inputs_have_enclosure` | `bool` | Every input grid has an enclosure relation |
| `any_input_has_enclosure` | `bool` | At least one input grid has an enclosure relation |
| `consistent_background` | `Optional[int]` | Background color if the same across all inputs; `None` if it varies |
| `consistent_n_objects` | `Optional[int]` | Object count if the same across all inputs; `None` if it varies |

### `perceive` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `Task` | required | Task with at least one training pair |

Returns `PerceptionReport`.
