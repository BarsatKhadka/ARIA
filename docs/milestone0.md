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

---
---

# Task

**File:** `task.py`

A container for one complete ARC-AGI task. It holds the training pairs, test inputs, and optionally the ground-truth solutions. It is the primary object passed around between perception modules, solvers, and the evaluator.

---

## Why Task exists

Each entry in the dataset JSON is a dict with `"train"` and `"test"` keys holding raw nested lists. Without a wrapper, every piece of code that works with tasks has to manually index into these dicts, construct `Grid` objects on the fly, and re-implement evaluation logic. `Task` does all of that once so everything else stays clean.

---

## What it holds

```python
task.task_id        # str  — e.g. "009d5c81"
task.train_pairs    # List[Tuple[Grid, Grid]]  — (input, output) for each demo pair
task.test_inputs    # List[Grid]               — inputs to predict
task.test_solutions # List[Grid] | None        — correct answers, None for blind test
```

`train_pairs` is a list of tuples — each tuple is `(input_grid, output_grid)`. These are the demonstration examples the solver uses to figure out the rule.

`test_solutions` is `None` for the competition test set (`arc-agi_test_challenges.json`) because those answers are not released.

---

## Construction

### From dataset files directly

```python
task = Task.from_json(
    task_id='009d5c81',
    challenges_path='data/arc-agi_training_challenges.json',
    solutions_path='data/arc-agi_training_solutions.json',  # omit for test set
)
```

Opens the files, finds the task by ID, and builds everything. The most convenient way to load a single task for debugging or development.

### From already-loaded dicts

```python
import json

with open('data/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)
with open('data/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

task = Task.from_dict('009d5c81', challenges['009d5c81'], solutions['009d5c81'])
```

Use this when you've already loaded the full JSON (e.g. iterating over all 1000 training tasks) — avoids re-opening files for every task.

### For the blind test set (no solutions)

```python
task = Task.from_json(
    task_id='some_test_id',
    challenges_path='data/arc-agi_test_challenges.json',
    # solutions_path omitted — task.test_solutions will be None
)
```

---

## Convenience properties

```python
task.n_train        # int — number of training pairs (2–10)
task.n_test         # int — number of test inputs (usually 1)
task.has_solutions  # bool — False for blind test tasks
```

---

## Evaluation

### Single prediction

```python
result = task.evaluate(test_index=0, predicted=my_grid)
```

Scores one predicted `Grid` against the ground truth for that test input.

`evaluate` raises `RuntimeError` if the task has no solutions loaded.

**The result object:**

```python
result.exact_match    # bool  — True only if shape matches AND every cell is correct
result.partial_score  # float — fraction of cells correct (0.0–1.0), 0.0 if shape differs
result.correct_cells  # int   — number of matching cells
result.total_cells    # int   — total cells in the ground truth grid
result.shape_match    # bool  — False means shapes differ entirely
```

**Example — correct prediction:**

```python
result = task.evaluate(0, task.test_solutions[0])
# exact_match=True, partial_score=1.0, correct_cells=196, total_cells=196
```

**Example — wrong prediction (all black):**

```python
wrong = Grid([[0]*14 for _ in range(14)])
result = task.evaluate(0, wrong)
# exact_match=False, partial_score=0.8673, correct_cells=170, total_cells=196
```

Note: an all-black prediction can still score ~87% partial score on a task that is mostly background. This is why `exact_match` is the only metric that counts for real scoring — partial score is only useful for measuring how close a solver is getting during development.

**Example — wrong shape:**

```python
wrong = Grid([[0, 0], [0, 0]])
result = task.evaluate(0, wrong)
# exact_match=False, shape_match=False, partial_score=0.0
```

If the predicted shape doesn't match, partial score is immediately 0.0. No credit for getting some cells right inside the wrong-shaped grid.

### All predictions at once

```python
results = task.evaluate_all([pred_grid_0, pred_grid_1])
# returns List[EvalResult], one per test input
```

Raises `ValueError` if the number of predictions doesn't match `task.n_test`.

---

## EvalResult fields

| Field | Type | Description |
|---|---|---|
| `exact_match` | `bool` | True if prediction is pixel-perfect |
| `partial_score` | `float` | Fraction of correct cells (0.0 if shape mismatch) |
| `correct_cells` | `int` | Count of matching cells |
| `total_cells` | `int` | Total cells in ground truth |
| `shape_match` | `bool` | Whether predicted shape equals ground truth shape |

---

## Full method summary

| Method / Property | Returns | Description |
|---|---|---|
| `Task.from_json(id, challenges, solutions)` | `Task` | Load from dataset files by task ID |
| `Task.from_dict(id, task_dict, solution_list)` | `Task` | Build from already-loaded dicts |
| `.task_id` | `str` | The 8-char hex task identifier |
| `.train_pairs` | `List[Tuple[Grid, Grid]]` | Demo (input, output) pairs |
| `.test_inputs` | `List[Grid]` | Inputs to predict |
| `.test_solutions` | `List[Grid] \| None` | Ground truth outputs, or None |
| `.n_train` | `int` | Number of training pairs |
| `.n_test` | `int` | Number of test inputs |
| `.has_solutions` | `bool` | False for blind test tasks |
| `.evaluate(index, predicted)` | `EvalResult` | Score one prediction |
| `.evaluate_all(predictions)` | `List[EvalResult]` | Score all predictions at once |
| `Task.load_all(challenges, solutions)` | `Dict[str, Task]` | Load every task from a dataset file |

---
---

# Evaluator

**File:** `evaluator.py`

The batch evaluation harness. Loads all tasks, runs a solver function over them, scores the results, and writes `submission.json` in the format required by the ARC competition.

---

## Why Evaluator exists

Without it, running a solver end-to-end means manually looping over all tasks, calling the solver, catching exceptions, collecting results, computing accuracy, and serializing predictions. `Evaluator` does all of that in one call so solvers stay focused on the logic of solving, not the plumbing around it.

---

## The solver contract

A solver is any plain Python function with this signature:

```python
def my_solver(task: Task) -> List[Grid]:
    ...
```

It receives a `Task` and returns **one `Grid` per test input** — that's its prediction for each test input in the task. For most tasks there is only one test input, so the list has one element.

If you want to provide two attempts (the competition allows two per test input), return a list of tuples instead:

```python
def my_solver(task: Task) -> List[Tuple[Grid, Grid]]:
    ...
```

The `Evaluator` handles both forms. If only one grid is provided per input, it is used as both `attempt_1` and `attempt_2`.

---

## Basic usage

```python
from evaluator import Evaluator
from task import Task
from grid import Grid

def my_solver(task: Task):
    # predict all-black for every test input
    return [Grid([[0] * inp.cols for _ in range(inp.rows)])
            for inp in task.test_inputs]

ev = Evaluator(
    challenges_path='data/arc-agi_training_challenges.json',
    solutions_path='data/arc-agi_training_solutions.json',
)

summary = ev.run(my_solver)
print(summary)
```

Output:
```
Tasks solved    : 0/1000 (0.0%)
Inputs solved   : 0/1000
Avg partial     : 52.3%
Time            : 1.4s
```

### Run on a subset

```python
summary = ev.run(my_solver, task_ids=['009d5c81', '00576224', '00d62c1b'])
```

Useful during development to test against a handful of specific tasks.

### Suppress per-task logging

```python
ev = Evaluator(..., verbose=False)
```

---

## Saving the submission

```python
ev.save_submission(summary, output_path='submission.json')
```

Writes the predictions in the required format:

```json
{
  "task_id": [
    {"attempt_1": [[0, 1], [2, 0]], "attempt_2": [[0, 1], [2, 0]]}
  ]
}
```

One entry per task. Each entry is a list with one dict per test input. Each dict has `attempt_1` and `attempt_2` as nested integer lists (from `Grid.to_list()`).

---

## Reading the summary

```python
summary.accuracy            # float — fraction of tasks fully solved
summary.solved_tasks        # int
summary.total_tasks         # int
summary.solved_test_inputs  # int — inputs where attempt_1 was exact
summary.total_test_inputs   # int
summary.avg_partial_score   # float — average partial score across all inputs
summary.total_elapsed_sec   # float

# Per-task results
result = summary.task_results['009d5c81']
result.solved               # bool — all test inputs exact-matched
result.best_partial         # float — best partial score across test inputs
result.elapsed_sec          # float — time taken by solver for this task
result.eval_results         # List[EvalResult] — one per test input
result.predictions          # List[Tuple[Grid, Grid]] — (attempt_1, attempt_2)
```

---

## Error handling

If the solver raises an exception on a task, the `Evaluator` catches it, logs the error, and substitutes a `1×1` black grid as the prediction so the run continues. No task failure can crash the whole evaluation.

---

## Full method summary

| Method | Returns | Description |
|---|---|---|
| `Evaluator(challenges, solutions, verbose)` | `Evaluator` | Load all tasks from files |
| `.run(solver, task_ids)` | `EvalSummary` | Run solver, score results |
| `.save_submission(summary, output_path)` | `None` | Write `submission.json` |

### EvalSummary fields

| Field | Type | Description |
|---|---|---|
| `accuracy` | `float` | Fraction of tasks fully solved |
| `solved_tasks` | `int` | Tasks where all test inputs were exact-matched |
| `total_tasks` | `int` | Tasks that had solutions (excludes blind test) |
| `solved_test_inputs` | `int` | Individual test inputs exact-matched |
| `avg_partial_score` | `float` | Mean partial score across all test inputs |
| `total_elapsed_sec` | `float` | Wall time for entire run |
| `task_results` | `Dict[str, TaskResult]` | Per-task breakdown |

### TaskResult fields

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Task identifier |
| `solved` | `bool` | True if every test input was exact-matched |
| `best_partial` | `float` | Highest partial score across test inputs |
| `elapsed_sec` | `float` | Solver time for this task |
| `eval_results` | `List[EvalResult]` | Per-input scores |
| `predictions` | `List[Tuple[Grid, Grid]]` | (attempt_1, attempt_2) per test input |

---
---

# Solver

**File:** `solver.py`

The plug point for all solving logic. Every milestone replaces what is inside `solve()`. Nothing else in the codebase changes.

---

## Why solver.py exists as its own file

The `Evaluator` accepts any function with the right signature — you could pass a lambda. But keeping the solver in its own file gives one clear place to find and replace the logic, and makes it easy to swap between solver versions by just changing the import.

---

## The stub (Milestone 0)

```python
def solve(task: Task) -> List[Grid]:
    return [
        Grid([[0] * inp.cols for _ in range(inp.rows)])
        for inp in task.test_inputs
    ]
```

Returns an all-zero (all-black) grid matching the shape of each test input. No reasoning, no perception — just the correct shape filled with zeros.

---

## Baseline results (Milestone 0)

Run on all 1000 training tasks:

```
Tasks solved    : 0/1000  (0.0%)
Inputs solved   : 0/1076
Avg partial     : 30.9%
Format errors   : 0
```

The 30.9% avg partial score is the all-zeros floor — it reflects how much of the average ARC grid is black background. It is not a sign the solver is doing anything useful. Every future milestone must beat this number.

The 1076 total inputs (not 1000) is because some tasks have more than one test input.

---

## How to plug in a new solver

The only requirement is the function signature:

```python
def solve(task: Task) -> List[Grid]:
    ...
```

Return one `Grid` per test input. The `Evaluator` handles the rest — scoring, submission formatting, logging.

To run and measure:

```python
from evaluator import Evaluator
from solver import solve

ev = Evaluator(
    'data/arc-agi_training_challenges.json',
    'data/arc-agi_training_solutions.json',
)
summary = ev.run(solve)
print(summary)
ev.save_submission(summary, 'submission.json')
```

That's the full loop. Change `solve`, re-run, read the number. Every milestone improvement shows up immediately as a higher accuracy.

---

## Solver signature reference

| Form | When to use |
|---|---|
| `List[Grid]` | One prediction per test input — used as both attempt_1 and attempt_2 |
| `List[Tuple[Grid, Grid]]` | Two distinct attempts per test input — the competition scores correct if either matches |
