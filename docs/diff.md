# Differential

The `differential/` package compares the perception outputs of training pairs to find what is **invariant** (true in every pair, therefore likely the rule) versus **incidental** (varies across pairs, therefore likely task-specific data).

None of these modules modify grids. They only read pairs and report structure.

---

## Package layout

| File | What it produces |
|---|---|
| `differential/grid_diff.py` | Per-pair and cross-pair grid analysis → `PairDiff`, `TaskDiff` |
| `differential/feature_classifier.py` | Cross-pair feature classification → `DifferentialReport` |
| `differential/object_alignment.py` | Per-pair object correspondence via Hungarian algorithm → `PairAlignment` |

---

---

# grid_diff.py

**File:** `differential/grid_diff.py`

Analyses every training pair in a task at the grid level — dimensions, colors, cell changes, background — then collapses the per-pair results into a single `TaskDiff` that highlights which properties are consistent across all pairs and which vary.

---

## Why grid_diff.py exists

A solver cannot act on a single pair observation. If the output is twice as wide as the input in pair 0, that might be the rule — or it might be a coincidence specific to that pair's data. `grid_diff.py` answers: *is that true in every pair?* Only consistent properties are worth hypothesising as rules.

---

## The `PairDiff` dataclass

Describes the difference between one input and one output grid.

```python
@dataclass(frozen=True)
class PairDiff:
    # Dimensions
    input_shape:    Tuple[int, int]
    output_shape:   Tuple[int, int]
    shape_changed:  bool
    rows_delta:     int
    cols_delta:     int
    scale_rows:     float
    scale_cols:     float
    # Colors
    input_colors:   FrozenSet[int]
    output_colors:  FrozenSet[int]
    colors_added:   FrozenSet[int]
    colors_removed: FrozenSet[int]
    colors_kept:    FrozenSet[int]
    # Cell change
    change_fraction: Optional[float]
    cells_changed:   Optional[int]
    total_cells:     int
    # Background
    background:      int
```

### Dimension fields

| Field | Meaning |
|---|---|
| `shape_changed` | `True` if input and output have different shapes |
| `rows_delta` | `output_rows - input_rows`; negative means output is shorter |
| `cols_delta` | `output_cols - input_cols`; negative means output is narrower |
| `scale_rows` | `output_rows / input_rows`; `2.0` means output is twice as tall |
| `scale_cols` | `output_cols / input_cols` |

```python
pd.rows_delta   # -2   → output is 2 rows shorter
pd.scale_rows   # 0.5  → output is half the height
```

### Color fields

| Field | Meaning |
|---|---|
| `input_colors` | All ARC colors (0–9) present in the input |
| `output_colors` | All ARC colors present in the output |
| `colors_added` | Colors in output but not in input — new colors introduced by the transformation |
| `colors_removed` | Colors in input but not in output — colors the transformation eliminated |
| `colors_kept` | Colors present in both |

```python
pd.colors_added    # frozenset({2})    → red was introduced
pd.colors_removed  # frozenset({1})    → blue was eliminated
pd.colors_kept     # frozenset({0, 3}) → black and green survived
```

### Cell-change fields

`change_fraction` and `cells_changed` are `None` when `shape_changed` is `True` — you cannot compare cells one-to-one when the grid dimensions differ.

| Field | Meaning |
|---|---|
| `change_fraction` | Fraction of cells (0.0–1.0) whose value differs between input and output |
| `cells_changed` | Raw count of differing cells |
| `total_cells` | `input_rows × input_cols` |

```python
pd.change_fraction  # 0.12 → 12% of cells changed
pd.cells_changed    # 7
```

### `background`

Most frequent color in the input grid, computed by raw pixel count.

---

## The `TaskDiff` dataclass

Collapses all per-pair `PairDiff` results into cross-pair invariants.

```python
@dataclass(frozen=True)
class TaskDiff:
    pairs:   Tuple[PairDiff, ...]
    n_pairs: int
    # Dimension invariants
    shape_always_changes:  bool
    shape_never_changes:   bool
    rows_delta_consistent: bool
    cols_delta_consistent: bool
    scale_rows_consistent: bool
    scale_cols_consistent: bool
    consistent_rows_delta: Optional[int]
    consistent_cols_delta: Optional[int]
    consistent_scale_rows: Optional[float]
    consistent_scale_cols: Optional[float]
    # Color invariants
    colors_always_added:      FrozenSet[int]
    colors_always_removed:    FrozenSet[int]
    colors_always_kept:       FrozenSet[int]
    colors_sometimes_added:   FrozenSet[int]
    colors_sometimes_removed: FrozenSet[int]
    # Change-fraction invariants
    comparable_pairs:           int
    change_fraction_min:        Optional[float]
    change_fraction_max:        Optional[float]
    change_fraction_avg:        Optional[float]
    change_fraction_range:      Optional[float]
    change_fraction_consistent: bool
    # Background
    background_consistent: bool
    background:            Optional[int]
```

### Dimension invariants

**`shape_always_changes`** / **`shape_never_changes`**

Boolean shortcuts. Most tasks fall into one of two categories: the output always has a different shape from the input (expansion/contraction tasks), or it never does (in-place transformation tasks).

```python
td.shape_never_changes   # True  → safe to assume output shape = input shape
td.shape_always_changes  # True  → output will always be a different size
```

**`rows_delta_consistent`** / **`cols_delta_consistent`**

`True` if every pair has the same `rows_delta` / `cols_delta`. If both are consistent, the output shape is fully determined by the input shape via a fixed offset.

```python
td.rows_delta_consistent   # True
td.consistent_rows_delta   # -1 → output is always 1 row shorter than input
```

**`scale_rows_consistent`** / **`scale_cols_consistent`**

`True` if every pair has the same scale factor (within 1e-6). Integer scales (2.0, 3.0) indicate tiling or repetition. Fractional scales (0.5) indicate shrinking.

```python
td.scale_rows_consistent   # True
td.consistent_scale_rows   # 3.0 → output is always 3× taller than input
```

When both scale factors are consistent integers, the output is the input tiled by that factor.

### Color invariants

The key distinction: **always** (invariant) vs **sometimes** (incidental).

| Field | Meaning |
|---|---|
| `colors_always_added` | Added in **every** pair — almost certainly part of the rule |
| `colors_always_removed` | Removed in **every** pair — almost certainly part of the rule |
| `colors_always_kept` | Present in both sides of **every** pair |
| `colors_sometimes_added` | Added in at least one pair — may be data-specific |
| `colors_sometimes_removed` | Removed in at least one pair — may be data-specific |

```python
td.colors_always_added    # frozenset({2}) → red is always introduced
td.colors_sometimes_added # frozenset({2, 4}) → yellow is added in some pairs only
```

A color in `colors_sometimes_added` but not `colors_always_added` is probably present in the input of some pairs and absent in others — it's the task data varying, not the rule.

### Change-fraction invariants

Only computed for **comparable pairs** — pairs where input and output have the same shape.

| Field | Meaning |
|---|---|
| `comparable_pairs` | Number of pairs with matching shapes |
| `change_fraction_min/max/avg` | Statistics over those pairs |
| `change_fraction_range` | `max - min` |
| `change_fraction_consistent` | `range < 0.05` — the fraction changes by less than 5 percentage points |

```python
td.change_fraction_consistent  # True
td.change_fraction_avg         # 0.08 → about 8% of cells change in every pair
```

### Background invariant

```python
td.background_consistent  # True
td.background             # 0 → background is always black
```

---

## `diff_pair`

```python
def diff_pair(inp: Grid, out: Grid) -> PairDiff:
```

Compute the diff for a single training pair.

```python
from differential.grid_diff import diff_pair

pd = diff_pair(task.train_pairs[0][0], task.train_pairs[0][1])
pd.colors_added    # frozenset({2})
pd.change_fraction # 0.14
```

---

## `diff_task`

```python
def diff_task(task: Task) -> TaskDiff:
```

Run `diff_pair` on every training pair and compute all cross-pair invariants.

```python
from differential.grid_diff import diff_task

td = diff_task(task)
td.shape_never_changes        # True
td.colors_always_added        # frozenset({2})
td.change_fraction_consistent # True
```

---

## Reading a `TaskDiff`: decision tree

```
shape_never_changes?
├─ YES → output has same shape as input
│        look at change_fraction_avg for how much changes
│        look at colors_always_added/removed for the color rule
│
└─ NO  → shape_always_changes?
          ├─ YES → scale_rows/cols consistent?
          │         ├─ YES (integer scale) → tiling/repetition rule
          │         ├─ YES (fractional)    → shrinking/cropping rule
          │         └─ NO                 → shape depends on input data
          │
          └─ NO  → some pairs change shape, some don't
                    (unusual — check pairs individually)
```

---

## Common patterns

### Check if this is an in-place recoloring task

```python
td = diff_task(task)
if td.shape_never_changes and td.colors_always_added and td.change_fraction_avg < 0.3:
    print("Likely recoloring:", td.colors_always_added, "→", td.colors_always_removed)
```

### Check if this is a tiling/expansion task

```python
td = diff_task(task)
if (td.shape_always_changes
        and td.scale_rows_consistent
        and td.scale_cols_consistent
        and td.consistent_scale_rows == int(td.consistent_scale_rows)):
    scale = int(td.consistent_scale_rows)
    print(f"Output is always {scale}× the input in both dimensions")
```

### Find the color the rule always introduces

```python
td = diff_task(task)
if td.colors_always_added:
    rule_color = next(iter(td.colors_always_added))
    print(f"Rule always adds color {rule_color}")
```

### Check how much of the grid changes

```python
td = diff_task(task)
if td.change_fraction_consistent:
    print(f"~{td.change_fraction_avg:.0%} of cells change in every pair")
```

### Access individual pair diffs

```python
td = diff_task(task)
for i, pd in enumerate(td.pairs):
    print(f"Pair {i}: {pd.input_shape} → {pd.output_shape}, "
          f"added={pd.colors_added}, changed={pd.change_fraction:.0%}")
```

---

## Full API reference

### `PairDiff` fields

| Field | Type | Description |
|---|---|---|
| `input_shape` | `Tuple[int,int]` | `(rows, cols)` of the input |
| `output_shape` | `Tuple[int,int]` | `(rows, cols)` of the output |
| `shape_changed` | `bool` | True if shapes differ |
| `rows_delta` | `int` | `output_rows - input_rows` |
| `cols_delta` | `int` | `output_cols - input_cols` |
| `scale_rows` | `float` | `output_rows / input_rows` |
| `scale_cols` | `float` | `output_cols / input_cols` |
| `input_colors` | `FrozenSet[int]` | Colors present in input |
| `output_colors` | `FrozenSet[int]` | Colors present in output |
| `colors_added` | `FrozenSet[int]` | In output, not in input |
| `colors_removed` | `FrozenSet[int]` | In input, not in output |
| `colors_kept` | `FrozenSet[int]` | In both |
| `change_fraction` | `Optional[float]` | Fraction of cells changed; `None` if shapes differ |
| `cells_changed` | `Optional[int]` | Count of changed cells; `None` if shapes differ |
| `total_cells` | `int` | `input_rows × input_cols` |
| `background` | `int` | Most frequent color in input |

### `TaskDiff` fields

| Field | Type | Description |
|---|---|---|
| `pairs` | `Tuple[PairDiff,...]` | All per-pair results in order |
| `n_pairs` | `int` | Number of training pairs |
| `shape_always_changes` | `bool` | Every pair changes shape |
| `shape_never_changes` | `bool` | No pair changes shape |
| `rows_delta_consistent` | `bool` | Same `rows_delta` in every pair |
| `cols_delta_consistent` | `bool` | Same `cols_delta` in every pair |
| `scale_rows_consistent` | `bool` | Same `scale_rows` in every pair |
| `scale_cols_consistent` | `bool` | Same `scale_cols` in every pair |
| `consistent_rows_delta` | `Optional[int]` | Shared delta value; `None` if inconsistent |
| `consistent_cols_delta` | `Optional[int]` | Shared delta value; `None` if inconsistent |
| `consistent_scale_rows` | `Optional[float]` | Shared scale; `None` if inconsistent |
| `consistent_scale_cols` | `Optional[float]` | Shared scale; `None` if inconsistent |
| `colors_always_added` | `FrozenSet[int]` | Added in every pair |
| `colors_always_removed` | `FrozenSet[int]` | Removed in every pair |
| `colors_always_kept` | `FrozenSet[int]` | Kept in every pair |
| `colors_sometimes_added` | `FrozenSet[int]` | Added in at least one pair |
| `colors_sometimes_removed` | `FrozenSet[int]` | Removed in at least one pair |
| `comparable_pairs` | `int` | Pairs with matching shapes |
| `change_fraction_min` | `Optional[float]` | Minimum change fraction |
| `change_fraction_max` | `Optional[float]` | Maximum change fraction |
| `change_fraction_avg` | `Optional[float]` | Mean change fraction |
| `change_fraction_range` | `Optional[float]` | `max - min` |
| `change_fraction_consistent` | `bool` | Range < 0.05 |
| `background_consistent` | `bool` | Same background in every pair |
| `background` | `Optional[int]` | Shared background; `None` if inconsistent |

### Functions

| Function | Returns | Description |
|---|---|---|
| `diff_pair(inp, out)` | `PairDiff` | Diff a single input/output pair |
| `diff_task(task)` | `TaskDiff` | Diff all training pairs and compute invariants |

---

---

# feature_classifier.py

**File:** `differential/feature_classifier.py`

The core differential engine. For every computable feature across the aligned training pairs — color mapping, selection criterion, transformation type, spatial pattern — classifies it as `RELEVANT` (invariant across pairs), `INCIDENTAL` (varies), or `AMBIGUOUS` (only one pair). Returns a `DifferentialReport`.

---

## Why feature_classifier.py exists

`grid_diff.py` answers "what changed?" at the grid level. `feature_classifier.py` answers the harder question: "**why** did it change, and will that reason hold for the test input?" It does this by extracting structured features from each pair and checking whether those features are consistent. A feature that is `RELEVANT` is a candidate rule. A feature that is `INCIDENTAL` is task-specific data, not the rule.

---

## `Confidence`

```python
class Confidence(Enum):
    RELEVANT   = "RELEVANT"    # same in every pair — likely part of the rule
    INCIDENTAL = "INCIDENTAL"  # varies across pairs — task-specific data
    AMBIGUOUS  = "AMBIGUOUS"   # only one pair, cannot confirm
```

The three-way classification:

- **RELEVANT** — the feature has the same value in every training pair. It is a candidate rule component. The more pairs that agree, the stronger the evidence.
- **INCIDENTAL** — the feature varies. It reflects the specific data of each pair, not a consistent rule.
- **AMBIGUOUS** — there is only one training pair. Without a second pair to compare, nothing can be confirmed. Treat AMBIGUOUS features with low confidence.

---

## `Feature`

```python
@dataclass(frozen=True)
class Feature:
    name:       str
    confidence: Confidence
    value:      Any                  # shared value when RELEVANT; None otherwise
    per_pair:   Tuple[Any, ...]      # raw observation from each training pair
```

Every classified property is returned as a `Feature`. Check `confidence` first, then read `value` if `RELEVANT`.

```python
f = report.color_mapping
if f.confidence == Confidence.RELEVANT:
    print(f"Always these color transitions: {f.value}")
else:
    print("Color transitions vary — check per_pair:", f.per_pair)
```

---

## `PairAnalysis`

Per-pair computed observations before cross-pair classification.

```python
@dataclass(frozen=True)
class PairAnalysis:
    pair_index: int
    pair_diff:  PairDiff
    alignment:  PairAlignment          # full object-level correspondence

    color_transitions:           FrozenSet[Tuple[int, int]]
    colors_with_changes:         FrozenSet[int]

    selected_object_colors:      Optional[FrozenSet[int]]
    selected_is_largest:         Optional[bool]
    selected_is_smallest:        Optional[bool]
    selected_is_border_touching: Optional[bool]
    selected_is_enclosed:        Optional[bool]

    transform_types:    FrozenSet[str]
    movement_vectors:   FrozenSet[Tuple[float, float]]  # centroid deltas

    n_input_objects:    int
    n_output_objects:   int
    object_count_delta: int
```

`alignment` is the full `PairAlignment` for this pair — all survived/created/destroyed matches are accessible directly from `PairAnalysis`.

Selection fields are `None` when no objects were selected (no recolored, moved, resized, or destroyed objects in the pair).

---

## `DifferentialReport`

The full cross-pair classification result.

```python
@dataclass(frozen=True)
class DifferentialReport:
    n_pairs:       int
    pair_analyses: Tuple[PairAnalysis, ...]

    # Color mapping
    color_mapping:    Feature
    selected_colors:  Feature

    # Selection criterion
    selection_is_largest:         Feature
    selection_is_smallest:        Feature
    selection_is_border_touching: Feature
    selection_is_enclosed:        Feature

    # Transformation type
    transform_type: Feature

    # Spatial pattern
    movement_vector:    Feature
    movement_direction: Feature

    # Object count
    object_count_delta: Feature
```

---

## Feature descriptions

### `color_mapping`

A `FrozenSet` of `(from_color, to_color)` tuples — every `(input_color, output_color)` pair observed across all recolored objects in the alignment. Derived from `PairAlignment.recolored`, so it works for pairs of any shape (not restricted to same-shape grids). `frozenset()` when no objects were recolored.

- `RELEVANT` + non-empty value → the same color transitions happen in every pair. The value is the rule: e.g. `{(1, 2)}` means blue always becomes red.
- `INCIDENTAL` → the transitions vary (different input data has different colors to change).

```python
r.color_mapping.value  # frozenset({(1, 2), (8, 0)})
# blue→red and azure→black in every pair
```

### `selected_colors`

The set of input colors whose cells always change. Complementary to `color_mapping` — this tells you which colors are targeted by the rule.

```python
r.selected_colors.value  # frozenset({1, 8})  → blue and azure are always transformed
```

### `selection_is_largest` / `selection_is_smallest`

Whether the changed objects are always the largest (or smallest) object in the input. `True` means the rule targets objects by size rank.

```python
r.selection_is_largest.confidence  # RELEVANT
r.selection_is_largest.value       # True → always the biggest object that changes
```

### `selection_is_border_touching`

Whether the changed objects always touch the border (outermost row or column) of the grid.

```python
r.selection_is_border_touching.value  # True → rule targets edge objects
```

### `selection_is_enclosed`

Whether the changed objects are always enclosed by another object (surrounded on all sides with no background gap — detected via topology flood-fill).

```python
r.selection_is_enclosed.value  # True → rule targets the inner object of a frame
```

### `transform_type`

A `FrozenSet[str]` of transformation types observed in the pair. Values:

| String | Meaning |
|---|---|
| `"recolor"` | At least one survived object changed color |
| `"move"` | At least one survived object shifted centroid > 0.5 cells |
| `"resize"` | At least one survived object changed pixel count |
| `"delete"` | At least one input object was destroyed (no output match) |
| `"create"` | At least one output object was created (no input match) |
| `"expand"` | Grid grew (different shape, scale > 1) — fallback when no objects detected |
| `"shrink"` | Grid shrank (different shape, scale < 1) — fallback when no objects detected |
| `"none"` | No detectable change |

A pair can have multiple types (e.g. `{"recolor", "delete"}` — some objects recolored, others disappeared).

```python
r.transform_type.value  # frozenset({"recolor", "delete"})
```

### `movement_vector`

A `FrozenSet` of `(dr, dc)` centroid displacements (floats) for objects that moved. Empty when nothing moved. Values come from `ObjectChange.position_delta`, which is a centroid difference so fractional values are possible.

- `RELEVANT` + `frozenset({(0.0, -3.0)})` → every pair moves one object exactly 3 cells to the left.
- `RELEVANT` + `frozenset()` → no movement in any pair.

```python
r.movement_vector.value  # frozenset({(0.0, -3.0)})
```

### `movement_direction`

Human-readable direction derived from `movement_vector`. Values: `"up"`, `"down"`, `"left"`, `"right"`, `"diagonal"`, `"mixed"`, or `None` (no movement).

Uses the same 2:1 primary-axis rule as `topology.py`.

```python
r.movement_direction.value  # "left"
```

### `object_count_delta`

`n_output_objects - n_input_objects`. Negative means objects are deleted; positive means objects are created; zero means the count is preserved.

```python
r.object_count_delta.value  # -1 → one object always disappears
```

---

## `classify`

```python
def classify(task: Task) -> DifferentialReport:
```

The single entry point. Runs all analyses and returns a `DifferentialReport`.

```python
from differential.feature_classifier import classify, Confidence

report = classify(task)
```

---

## Reading a `DifferentialReport`

```
transform_type RELEVANT?
├─ {"recolor"} → in-place color change
│   selected_colors RELEVANT?  → which colors are targeted
│   color_mapping RELEVANT?    → what they become
│   selection_is_largest?      → is it always the biggest object?
│   selection_is_enclosed?     → is it always the inner object?
│
├─ {"move"} → objects shift position
│   movement_direction RELEVANT? → which way do they go?
│   movement_vector RELEVANT?    → by exactly how much?
│
├─ {"expand"} or {"shrink"} → grid resizes
│   check TaskDiff.consistent_scale_rows/cols for the factor
│
├─ {"delete"} → object_count_delta for count
│   selection_is_largest/smallest → which one is removed?
│
└─ {"create"} → new objects appear
    object_count_delta for count
```

---

## Common patterns

### Is there a fixed color rule?

```python
report = classify(task)
if (report.color_mapping.confidence == Confidence.RELEVANT
        and report.color_mapping.value):
    for from_c, to_c in report.color_mapping.value:
        print(f"Color {from_c} always becomes {to_c}")
```

### Which objects are targeted?

```python
if report.selection_is_largest.value is True:
    print("Rule always targets the largest object")
elif report.selection_is_enclosed.value is True:
    print("Rule always targets the enclosed object")
elif report.selected_colors.confidence == Confidence.RELEVANT:
    print(f"Rule always targets colors: {report.selected_colors.value}")
```

### Does the grid always shrink/expand?

```python
from differential.grid_diff import diff_task
td = diff_task(task)
report = classify(task)

if report.transform_type.value == frozenset({"expand"}):
    if td.scale_rows_consistent:
        print(f"Always expands {td.consistent_scale_rows}× vertically")
```

### Does movement have a fixed direction?

```python
if report.movement_direction.value is not None:
    print(f"Objects always move: {report.movement_direction.value}")
    if report.movement_vector.value:
        dr, dc = next(iter(report.movement_vector.value))
        print(f"  by ({dr}, {dc}) cells")
```

### Inspect per-pair raw observations

```python
report = classify(task)
for pa in report.pair_analyses:
    print(f"Pair {pa.pair_index}: transitions={pa.color_transitions}, "
          f"types={pa.transform_types}")
```

---

## Full API reference

### `Confidence` values

| Value | Meaning |
|---|---|
| `RELEVANT` | Same in every pair — candidate rule component |
| `INCIDENTAL` | Varies across pairs — task-specific data |
| `AMBIGUOUS` | Only one pair — cannot confirm |

### `Feature` fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Feature identifier |
| `confidence` | `Confidence` | Classification result |
| `value` | `Any` | Shared value if `RELEVANT`; `None` otherwise |
| `per_pair` | `Tuple[Any,...]` | Raw observation from each training pair |

### `PairAnalysis` fields

| Field | Type | Description |
|---|---|---|
| `pair_index` | `int` | Zero-based index of this pair |
| `pair_diff` | `PairDiff` | Grid-level diff for this pair |
| `alignment` | `PairAlignment` | Full object-level correspondence for this pair |
| `color_transitions` | `FrozenSet[Tuple[int,int]]` | `{(from,to)}` from recolored objects |
| `colors_with_changes` | `FrozenSet[int]` | Input colors of all selected objects |
| `selected_object_colors` | `Optional[FrozenSet[int]]` | Colors of selected objects; `None` if none |
| `selected_is_largest` | `Optional[bool]` | Selected objects all == largest; `None` if none |
| `selected_is_smallest` | `Optional[bool]` | Selected objects all == smallest; `None` if none |
| `selected_is_border_touching` | `Optional[bool]` | Selected objects all touch border; `None` if none |
| `selected_is_enclosed` | `Optional[bool]` | Selected objects all enclosed; `None` if none |
| `transform_types` | `FrozenSet[str]` | Set of transform type strings for this pair |
| `movement_vectors` | `FrozenSet[Tuple[float,float]]` | `{(dr,dc)}` centroid deltas for moved objects |
| `n_input_objects` | `int` | Objects extracted from input |
| `n_output_objects` | `int` | `n_survived + n_created` |
| `object_count_delta` | `int` | `n_created - n_destroyed` |

---

### `DifferentialReport` fields

| Field | Type | Description |
|---|---|---|
| `n_pairs` | `int` | Number of training pairs |
| `pair_analyses` | `Tuple[PairAnalysis,...]` | Per-pair raw observations |
| `color_mapping` | `Feature` | Invariant `{(from,to)}` color transitions |
| `selected_colors` | `Feature` | Input colors always involved in changes |
| `selection_is_largest` | `Feature` | Changed objects always == largest |
| `selection_is_smallest` | `Feature` | Changed objects always == smallest |
| `selection_is_border_touching` | `Feature` | Changed objects always touch border |
| `selection_is_enclosed` | `Feature` | Changed objects always enclosed by another |
| `transform_type` | `Feature` | Invariant set of transform type strings |
| `movement_vector` | `Feature` | Invariant `{(dr,dc)}` float centroid displacement set |
| `movement_direction` | `Feature` | `"up"/"down"/"left"/"right"/"diagonal"/"mixed"` or `None` |
| `object_count_delta` | `Feature` | `n_output - n_input` objects |

### `classify` parameters

| Parameter | Type | Description |
|---|---|---|
| `task` | `Task` | Task with training pairs to analyse |

Returns `DifferentialReport`.

---

---

# object_alignment.py

**File:** `differential/object_alignment.py`

For each training pair, aligns input objects to output objects using the Hungarian algorithm on a composite similarity score. Records which objects survived (and how they changed), which were created, and which were destroyed.

---

## Why object_alignment.py exists

Grid-level diffs tell you *that* cells changed. Object alignment tells you *which objects* changed and *what happened to each one*. Once you know that object A (blue, L-shape, top-left) always becomes object B (red, L-shape, shifted right), you have a concrete transformation rule that can be applied to the test input.

---

## Scoring function

Every candidate (input object, output object) pair receives a composite score in [0, 1]:

```
score = 0.35 × bbox_IoU + 0.40 × shape_Jaccard + 0.25 × color_match
```

| Component | Weight | Description |
|---|---|---|
| **Bounding-box IoU** | 0.35 | Intersection-over-Union of bounding boxes. High when objects overlap spatially. |
| **Shape Jaccard** | 0.40 | `|sig_a ∩ sig_b| / |sig_a ∪ sig_b|` on `shape_signature`. Position-invariant — remains 1.0 even when an object moves. |
| **Color match** | 0.25 | 1.0 if `dominant_color` matches, 0.0 otherwise. |

Shape Jaccard dominates because it is position-invariant: an object that moved but kept its shape still scores 0.65+ against its output counterpart, making the correct assignment clear even with zero spatial overlap.

Pairs with score below **0.10** are discarded and the objects are treated as unmatched.

---

## Hungarian algorithm

`scipy.optimize.linear_sum_assignment` solves the optimal assignment on the cost matrix `1 − score`. This guarantees globally optimal pairing: no other one-to-one assignment has a higher total score.

The matrix is rectangular — if there are more output objects than input objects, the extra outputs become `created`. If there are more input objects, the extras become `destroyed`.

---

## `ObjectChange`

What differed between a matched input and output object.

```python
@dataclass(frozen=True)
class ObjectChange:
    color_changed:  bool
    color_from:     int
    color_to:       int
    position_delta: Tuple[float, float]   # (Δrow, Δcol) of centroids
    size_delta:     int                   # output.size − input.size
    shape_changed:  bool                  # shape_signature differs
    iou:            float                 # bbox IoU of this matched pair
    shape_jaccard:  float                 # shape-sig Jaccard of this matched pair
```

| Field | Meaning |
|---|---|
| `color_changed` | `True` if `dominant_color` differs |
| `color_from/to` | The before and after color values |
| `position_delta` | `(Δrow, Δcol)` centroid shift; `(0,0)` if stationary |
| `size_delta` | Positive = object grew; negative = object shrank |
| `shape_changed` | `True` if `shape_signature` differs (reshaped) |
| `iou` | Bounding-box overlap of this specific matched pair |
| `shape_jaccard` | Shape overlap of this specific matched pair |

---

## `ObjectMatch`

One matched (input → output) pair with its score and change record.

```python
@dataclass(frozen=True)
class ObjectMatch:
    input_obj:  Object
    output_obj: Object
    score:      float
    change:     ObjectChange
```

---

## `PairAlignment`

Complete object-level correspondence for one training pair.

```python
@dataclass(frozen=True)
class PairAlignment:
    pair_index: int
    survived:   Tuple[ObjectMatch, ...]
    created:    Tuple[Object, ...]
    destroyed:  Tuple[Object, ...]
```

### Convenience properties

| Property | Returns | Description |
|---|---|---|
| `n_survived` | `int` | `len(survived)` |
| `n_created` | `int` | `len(created)` |
| `n_destroyed` | `int` | `len(destroyed)` |
| `recolored` | `Tuple[ObjectMatch,...]` | Survived objects whose color changed |
| `moved` | `Tuple[ObjectMatch,...]` | Survived objects whose centroid shifted > 0.5 cells |
| `resized` | `Tuple[ObjectMatch,...]` | Survived objects whose pixel count changed |
| `reshaped` | `Tuple[ObjectMatch,...]` | Survived objects whose shape_signature changed |
| `unchanged` | `Tuple[ObjectMatch,...]` | Survived objects identical in color, position, shape |

---

## `align_pair`

```python
def align_pair(pair_index: int, inp: Grid, out: Grid) -> PairAlignment:
```

Align one input/output grid pair.

```python
from differential.object_alignment import align_pair

pa = align_pair(0, inp_grid, out_grid)
pa.n_survived   # 3
pa.n_destroyed  # 1
pa.n_created    # 0

for m in pa.survived:
    print(m.score, m.change.color_changed, m.change.position_delta)
```

---

## `align_task`

```python
def align_task(task: Task) -> Tuple[PairAlignment, ...]:
```

Align all training pairs in one call.

```python
from differential.object_alignment import align_task

alignments = align_task(task)
for pa in alignments:
    print(f"Pair {pa.pair_index}: {pa.n_survived} survived, "
          f"{pa.n_created} created, {pa.n_destroyed} destroyed")
```

---

## Score interpretation

| Score range | Meaning |
|---|---|
| 1.00 | Perfect: same position, same shape, same color |
| 0.75 | Recolored in place: same position + shape, different color |
| 0.65 | Moved, same shape, same color: IoU=0, shape=1, color=1 |
| 0.40 | Same shape only: moved + recolored |
| < 0.10 | No meaningful correspondence — treated as unmatched |

---

## Common patterns

### Find objects that always get recolored

```python
alignments = align_task(task)
for pa in alignments:
    for m in pa.recolored:
        print(f"Pair {pa.pair_index}: {m.change.color_from} → {m.change.color_to}")
```

### Find consistently destroyed objects

```python
alignments = align_task(task)
destroyed_colors = [
    {o.dominant_color for o in pa.destroyed}
    for pa in alignments
]
always_destroyed = set.intersection(*destroyed_colors) if destroyed_colors else set()
```

### Find objects that always move in the same direction

```python
alignments = align_task(task)
all_deltas = [
    {m.change.position_delta for m in pa.moved}
    for pa in alignments if pa.moved
]
if all_deltas and all(d == all_deltas[0] for d in all_deltas):
    print(f"Objects always move by {all_deltas[0]}")
```

### Check if all pairs have the same survival count

```python
alignments = align_task(task)
counts = [pa.n_survived for pa in alignments]
if len(set(counts)) == 1:
    print(f"Always {counts[0]} objects survive")
```

---

## Full API reference

### `ObjectChange` fields

| Field | Type | Description |
|---|---|---|
| `color_changed` | `bool` | True if dominant color differs |
| `color_from` | `int` | Input object's dominant color |
| `color_to` | `int` | Output object's dominant color |
| `position_delta` | `Tuple[float,float]` | `(Δrow, Δcol)` centroid shift |
| `size_delta` | `int` | `output.size − input.size` |
| `shape_changed` | `bool` | True if shape_signature differs |
| `iou` | `float` | Bounding-box IoU of this pair |
| `shape_jaccard` | `float` | Shape-signature Jaccard of this pair |

### `ObjectMatch` fields

| Field | Type | Description |
|---|---|---|
| `input_obj` | `Object` | The input-side object |
| `output_obj` | `Object` | The output-side object |
| `score` | `float` | Composite match score (0–1) |
| `change` | `ObjectChange` | What changed between the two |

### `PairAlignment` fields

| Field | Type | Description |
|---|---|---|
| `pair_index` | `int` | Position within the task |
| `survived` | `Tuple[ObjectMatch,...]` | Matched pairs |
| `created` | `Tuple[Object,...]` | Output objects with no input match |
| `destroyed` | `Tuple[Object,...]` | Input objects with no output match |

### Functions

| Function | Returns | Description |
|---|---|---|
| `align_pair(index, inp, out)` | `PairAlignment` | Align one training pair |
| `align_task(task)` | `Tuple[PairAlignment,...]` | Align all training pairs |
