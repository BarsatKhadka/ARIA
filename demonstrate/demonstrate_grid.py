import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from core.grid import Grid

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

with open(os.path.join(data_dir, 'arc-agi_training_challenges.json')) as f:
    tasks = json.load(f)
with open(os.path.join(data_dir, 'arc-agi_training_solutions.json')) as f:
    solutions = json.load(f)


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ── Pick 3 tasks that show different behaviors ─────────────────────────────
# 009d5c81 : same shape, recoloring (8 -> 7, 1 disappears)
# 00576224 : input smaller than output (tiling)
# 6150a2bd : output smaller than input (shrinking)

TASKS = {
    "009d5c81": "same shape  — recoloring",
    "00576224": "output bigger — tiling/expansion",
}

# find a shrinking task
for tid, task in tasks.items():
    pair = task['train'][0]
    ir, ic = len(pair['input']), len(pair['input'][0])
    or_, oc = len(pair['output']), len(pair['output'][0])
    if or_ < ir and oc < ic and ir >= 6:
        TASKS[tid] = "output smaller — shrinking/reduction"
        break


for task_id, description in TASKS.items():
    task   = tasks[task_id]
    sol    = solutions[task_id]
    n_train = len(task['train'])
    n_test  = len(task['test'])

    section(f"TASK {task_id}  |  {description}")
    print(f"  train pairs: {n_train}   test inputs: {n_test}\n")

    # ── Show all train pairs ───────────────────────────────────────────────
    for i, pair in enumerate(task['train']):
        inp = Grid(pair['input'])
        out = Grid(pair['output'])

        print(f"  --- Train pair {i} ---")
        print(f"  Input  {inp.shape}:")
        for line in repr(inp).splitlines():
            print(f"    {line}")
        print(f"  Output {out.shape}:")
        for line in repr(out).splitlines():
            print(f"    {line}")

        # .shape
        print(f"\n  inp.shape          -> {inp.shape}")
        print(f"  out.shape          -> {out.shape}")

        # .unique_colors
        print(f"  inp.unique_colors  -> {inp.unique_colors()}")
        print(f"  out.unique_colors  -> {out.unique_colors()}")

        # .cells_of_color  (show for each non-zero color in input)
        for color in inp.unique_colors():
            if color != 0:
                cells = inp.cells_of_color(color)
                print(f"  inp.cells_of_color({color}) -> {len(cells)} cells  e.g. {cells[:3]}{'...' if len(cells) > 3 else ''}")

        # .get / .set  (only meaningful demo on same-shape tasks)
        r, c = 0, 0
        print(f"  inp.get(0,0)       -> {inp.get(r, c)}")

        # .equals
        print(f"  inp.equals(out)    -> {inp.equals(out)}")

        # .diff_cells / .diff_mask  (only if shapes match)
        if inp.shape == out.shape:
            changed = inp.diff_cells(out)
            mask    = inp.diff_mask(out)
            print(f"  inp.diff_cells(out)-> {len(changed)} cells changed  e.g. {changed[:3]}{'...' if len(changed) > 3 else ''}")
            print(f"  inp.diff_mask(out) ->")
            for row in mask:
                print(f"    {['X' if v else '.' for v in row]}")
        else:
            print(f"  diff_cells/mask    -> N/A (shapes differ: {inp.shape} vs {out.shape})")

        # .to_list
        as_list = inp.to_list()
        print(f"  inp.to_list()      -> {as_list[0]}  (first row, ready for JSON)")
        print()

    # ── Show test input and solution ──────────────────────────────────────
    for i, test_pair in enumerate(task['test']):
        test_inp = Grid(test_pair['input'])
        test_sol = Grid(sol[i])

        print(f"  --- Test input {i}  (what we must predict) ---")
        print(f"  Input  {test_inp.shape}:")
        for line in repr(test_inp).splitlines():
            print(f"    {line}")
        print(f"  Correct answer {test_sol.shape}:")
        for line in repr(test_sol).splitlines():
            print(f"    {line}")
        print()
