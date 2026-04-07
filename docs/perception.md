# Perception

The `perception/` package breaks a raw `Grid` into structured observations — objects, colors, symmetries, relations — that a solver can reason about. Each file is one focused analysis step. None of them modify grids; they only read and describe.

---

## Package layout

| File | What it produces |
|---|---|
| `perception/objects.py` | Connected components → list of `Object` dataclasses |
| `perception/colors.py` | Background detection, color histogram, color regions → `ColorAnalysis` |
| `perception/symmetry.py` | Vertical, horizontal, 180°, 90° symmetry tests + scores → `SymmetryResult` |

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
