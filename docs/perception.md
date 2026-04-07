# Perception

The `perception/` package breaks a raw `Grid` into structured observations — objects, colors, symmetries, relations — that a solver can reason about. Each file is one focused analysis step. None of them modify grids; they only read and describe.

---

## Package layout

| File | What it produces |
|---|---|
| `perception/objects.py` | Connected components → list of `Object` dataclasses |

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
