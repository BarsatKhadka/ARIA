"""
pipeline.py — End-to-end walkthrough of Milestone 0.

Shows exactly what happens at each step:
  raw JSON -> Grid -> Task -> Solver -> Evaluator -> submission.json

Run from the ARIA root:
    python demonstrate/pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core import Grid, Task, Evaluator
from solver import solve


# ── Helpers ───────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def show_grid(g: Grid, label: str = "", indent: int = 4):
    pad = " " * indent
    if label:
        print(f"{pad}{label}  {g.shape}")
    for line in repr(g).splitlines():
        print(f"{pad}  {line}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Raw JSON → Grid
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 1 -- Raw JSON -> Grid")

raw = [[0, 0, 8, 0],
       [0, 7, 7, 0],
       [0, 0, 8, 0],
       [0, 0, 0, 0]]

g = Grid(raw)

print(f"  raw list   : {raw[0]} ...")
print(f"  g.shape    : {g.shape}")
print(f"  g.unique_colors : {g.unique_colors()}")
print(f"  g.cells_of_color(7) : {g.cells_of_color(7)}")
print(f"  g.cells_of_color(8) : {g.cells_of_color(8)}")
print(f"  g.get(1, 1) : {g.get(1, 1)}")
print()
show_grid(g, "visual:")

# immutability — update returns a new grid
g2 = g.update(1, 1, 3)
print(f"\n  after g.update(1,1,3):")
print(f"  original g[1,1] : {g.get(1,1)}  (unchanged)")
print(f"  new     g2[1,1] : {g2.get(1,1)} (new grid)")
print(f"  hash(g) == hash(g2) : {hash(g) == hash(g2)}  (different grids, different hashes)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Grid → Task  (loading a real ARC task)
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 2 — Loading a Task")

TASK_ID = "009d5c81"
task = Task.from_json(
    TASK_ID,
    "data/arc-agi_training_challenges.json",
    "data/arc-agi_training_solutions.json",
)

print(f"  {task}")
print(f"  task.n_train       : {task.n_train}")
print(f"  task.n_test        : {task.n_test}")
print(f"  task.has_solutions : {task.has_solutions}")

print(f"\n  Training pairs (what the solver learns from):")
for i, (inp, out) in enumerate(task.train_pairs):
    same = "same shape" if inp.shape == out.shape else f"{inp.shape} → {out.shape}"
    colors_in  = inp.unique_colors()
    colors_out = out.unique_colors()
    print(f"    pair {i}: {same}  |  colors in={colors_in}  out={colors_out}")

print(f"\n  First train pair — input:")
show_grid(task.train_pairs[0][0])
print(f"\n  First train pair — output:")
show_grid(task.train_pairs[0][1])

print(f"\n  Test input (what we must predict):")
show_grid(task.test_inputs[0])

print(f"\n  Correct answer (ground truth):")
show_grid(task.test_solutions[0])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Task → Solver → Prediction
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 3 — Solver makes a prediction")

predictions = solve(task)
pred = predictions[0]

print(f"  solver returned {len(predictions)} prediction(s)")
print(f"\n  Prediction (all zeros — stub solver):")
show_grid(pred)
print(f"\n  Correct answer:")
show_grid(task.test_solutions[0])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate the prediction
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 4 — Evaluation")

result = task.evaluate(0, pred)

print(f"  exact_match    : {result.exact_match}")
print(f"  partial_score  : {result.partial_score:.1%}")
print(f"  correct_cells  : {result.correct_cells} / {result.total_cells}")
print(f"  shape_match    : {result.shape_match}")

# Show diff — which cells are wrong
diff = pred.diff_cells(task.test_solutions[0])
print(f"\n  {len(diff)} cells wrong (showing first 5): {diff[:5]}...")

# Two-attempt evaluation
correct = task.test_solutions[0]
result2 = task.evaluate(0, (pred, correct))   # attempt_2 is correct
print(f"\n  Two-attempt eval (bad, correct): exact_match = {result2.exact_match}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Batch evaluation over 20 tasks
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 5 — Batch Evaluator (20 tasks)")

ev = Evaluator(
    "data/arc-agi_training_challenges.json",
    "data/arc-agi_training_solutions.json",
    verbose=True,
)

task_ids = list(ev.tasks.keys())[:20]
summary = ev.run(solve, task_ids=task_ids)

print(f"\n  {'-'*40}")
print(f"  {summary}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Submission file
# ══════════════════════════════════════════════════════════════════════════════

header("STEP 6 — submission.json")

import json
ev.save_submission(summary, "submission.json")

with open("submission.json") as f:
    sub = json.load(f)

first_id = task_ids[0]
entry = sub[first_id][0]

print(f"  tasks in file       : {len(sub)}")
print(f"  keys per entry      : {list(entry.keys())}")
print(f"  attempt_1 (row 0)   : {entry['attempt_1'][0]}")
print(f"  attempt_2 (row 0)   : {entry['attempt_2'][0]}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

header("MILESTONE 0 — WHAT WE HAVE")

print("""
  core/grid.py       Grid — wraps a numpy array, immutable, hashable
  core/task.py       Task — holds train pairs + test inputs + solutions
  core/evaluator.py  Evaluator — batch runner, scorer, submission writer
  solver.py          solve() — stub, returns all-zeros

  Pipeline:
    raw JSON -> Grid -> Task -> solve() -> evaluate() -> submission.json

  Baseline (all-zeros solver on 1000 tasks):
    Solved        : 0 / 1000  (0.0%)
    Avg partial   : ~30.9%    ← the floor to beat
    Format errors : 0         ← submission.json is valid
""")
