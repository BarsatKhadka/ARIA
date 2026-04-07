# Grid

**File:** `grid.py`

A thin wrapper around a 2D numpy array representing one ARC-AGI grid. Every input, output, and solution in the dataset is loaded as a `Grid`. All perception modules, solvers, and submission code work with `Grid` objects rather than raw lists.

---

## Why Grid exists

ARC grids are just nested Python lists in the JSON files — for example:

```python
[[0, 0, 8], [0, 7, 0], [1, 0, 0]]
```

Working with raw lists means rewriting the same loops everywhere: iterating rows and columns, finding all cells of a color, comparing two grids. `Grid` wraps a numpy array and exposes these operations as clean methods so every other module can stay focused on its own logic.

---

## Construction

### From a raw list (loading from the dataset)

```python
from grid import Grid

raw = [[0, 0, 8], [0, 7, 0], [1, 0, 0]]
g = Grid(raw)
```

This is the primary way grids are created — pass in the nested list directly from the JSON.

### From another list (classmethod alias)

```python
g = Grid.from_list(raw)
```

Identical to `Grid(raw)`. Useful when you want the construction to read more explicitly.

### Internal storage

```python
g.data  # numpy array, dtype=int8
```

The underlying numpy array is always `dtype=int8` (values 0–9 fit in one byte). You can access it directly for numpy operations, but prefer the methods below for everything else.

---

## Dimensions

```python
g.rows   # number of rows (height)
g.cols   # number of columns (width)
g.shape  # (rows, cols) tuple
```

**Example:**

```python
g = Grid([[1, 2, 3], [4, 5, 6]])
g.rows   # 2
g.cols   # 3
g.shape  # (2, 3)
```

ARC grids range from 1×1 to 30×30. Shape is the first thing to check when comparing an input to its output — 65% of tasks have matching shapes, 35% do not.

---

## Cell access

### Reading a cell

```python
g.get(row, col)  # returns int
```

```python
g = Grid([[7, 9],
         [4, 3]])
g.get(0, 0)  # 7
g.get(1, 1)  # 3
```

### Writing a cell

```python
g.set(row, col, value)
```

```python
g.set(0, 0, 5)
g.get(0, 0)  # 5
```

`set` modifies the grid in place. The value must be an integer 0–9.

---

## Color queries

### All cells of a specific color

```python
g.cells_of_color(color)  # returns List[Tuple[int, int]]
```

Returns every `(row, col)` position where the cell equals `color`. Order is top-to-bottom, left-to-right (numpy row-major order).

```python
g = Grid([[0, 8, 0],
          [8, 0, 8],
          [0, 0, 0]])

g.cells_of_color(8)  # [(0, 1), (1, 0), (1, 2)]
g.cells_of_color(0)  # [(0, 0), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]
```

This is heavily used by perception modules — for example, `objects.py` seeds its flood-fill from the result of `cells_of_color`, and `colors.py` counts region sizes using `len(g.cells_of_color(c))`.

### All colors present

```python
g.unique_colors()  # returns List[int], sorted ascending
```

```python
g = Grid([[0, 1, 0], [2, 0, 1]])
g.unique_colors()  # [0, 1, 2]
```

Tells you which colors exist in the grid at all. Useful for the first step of any analysis — if a color is absent, there's no point looking for it.

---

## Comparison

### Exact equality

```python
g.equals(other)  # returns bool
```

Returns `True` only if both grids have the same shape and every cell matches. This is the check used when scoring submissions — a prediction is correct if and only if it exactly equals the solution.

```python
a = Grid([[1, 2], [3, 4]])
b = Grid([[1, 2], [3, 4]])
c = Grid([[1, 2], [3, 5]])

a.equals(b)  # True
a.equals(c)  # False
```

### Diff mask

```python
g.diff_mask(other)  # returns np.ndarray of bool, same shape as g
```

Returns a boolean numpy array where `True` marks cells that differ between `g` and `other`. Both grids must have the same shape — raises `ValueError` if they don't.

```python
a = Grid([[1, 2], [3, 4]])
b = Grid([[1, 9], [3, 4]])

a.diff_mask(b)
# array([[False,  True],
#        [False, False]])
```

Useful for visualizing exactly what a transformation changed, or for checking whether a candidate solution differs from the expected output in only a small number of cells.

### Diff cells

```python
g.diff_cells(other)  # returns List[Tuple[int, int]]
```

The same as `diff_mask` but returns the changed positions as a list of `(row, col)` tuples instead of a boolean array.

```python
a.diff_cells(b)  # [(0, 1)]
```

Both `diff_mask` and `diff_cells` require matching shapes. For tasks where input and output have different shapes (35% of tasks), you cannot diff them directly — you have to reason about the transformation at a higher level.

---

## Serialization

### To nested list (for JSON submission)

```python
g.to_list()  # returns List[List[int]]
```

Converts the grid back to a plain Python nested list. This is what you pass to `json.dump` when building a submission file.

```python
g = Grid([[3, 2], [7, 8]])
g.to_list()  # [[3, 2], [7, 8]]
```

The submission format expects exactly this structure:

```python
submission = {
    "task_id": [
        {
            "attempt_1": grid_1.to_list(),
            "attempt_2": grid_2.to_list(),
        }
    ]
}
```

---

## Display

```python
repr(g)   # or just: print(g)
```

Renders the grid as a human-readable ASCII string. Color `0` (black/background) is shown as `.` to make structure visible. Colors 1–9 are shown as their digit.

```python
g = Grid([[0, 0, 2],
          [0, 7, 0],
          [1, 0, 0]])

print(g)
# . . 2
# . 7 .
# 1 . .
```

---

## Color reference

The dataset contains only integers. These are the conventional names used when talking about tasks:

| Value | Color    |
|-------|----------|
| 0     | black    |
| 1     | blue     |
| 2     | red      |
| 3     | green    |
| 4     | yellow   |
| 5     | grey     |
| 6     | magenta  |
| 7     | orange   |
| 8     | azure    |
| 9     | maroon   |

Color names are not stored anywhere in the dataset — they are purely a human convention for communication.

---

## Full method summary

| Method / Property | Returns | Description |
|---|---|---|
| `Grid(data)` | `Grid` | Construct from nested list |
| `Grid.from_list(data)` | `Grid` | Same as above (classmethod) |
| `.data` | `np.ndarray` | Raw numpy array (int8) |
| `.rows` | `int` | Number of rows |
| `.cols` | `int` | Number of columns |
| `.shape` | `(int, int)` | `(rows, cols)` |
| `.get(r, c)` | `int` | Value at row r, col c |
| `.set(r, c, v)` | `None` | Set cell to value v (in place) |
| `.cells_of_color(color)` | `List[Tuple]` | All `(row, col)` of that color |
| `.unique_colors()` | `List[int]` | Sorted list of colors present |
| `.equals(other)` | `bool` | True if same shape and all cells match |
| `.diff_mask(other)` | `np.ndarray` | Boolean array of differing cells (same shape required) |
| `.diff_cells(other)` | `List[Tuple]` | List of `(row, col)` that differ (same shape required) |
| `.to_list()` | `List[List[int]]` | Nested list for JSON serialization |
| `repr(grid)` | `str` | ASCII render using `.123456789` |
