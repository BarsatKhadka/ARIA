"""
perception_audit.py — Integration test of all perception modules on the full ARC training set.

Runs every perception module on every training input and output grid.
Logs per-task findings and prints a summary that counts how many tasks
have "clean perception" — meaning the modules capture structure that
clearly distinguishes input from output.

Run from the ARIA root:
    python demonstrate/perception_audit.py
"""

import sys, os, json, traceback
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core.grid    import Grid
from core.task    import Task
from perception.objects  import extract_objects
from perception.colors   import analyze_colors
from perception.symmetry import analyze_symmetry
from perception.patterns import detect_periodicity
from perception.lines    import find_lines
from perception.topology import relate_all


# ── Clean-perception criteria ─────────────────────────────────────────────────
#
# A task has "clean perception" if, across its training inputs, at least one
# of the following is true:
#
#   OBJECTS     — every training input has ≥ 2 non-background objects
#   SYMMETRY    — at least one training input has a detected symmetry
#   LINES       — at least one training input has ≥ 2 detected lines
#   PERIODICITY — at least one training input is periodic
#   ENCLOSURE   — at least one training input has an enclosed object pair
#
# Each criterion is tracked independently so we can see which modules
# fire most often.


# ── Helpers ───────────────────────────────────────────────────────────────────

def _analyze_grid(grid: Grid) -> dict:
    """Run all modules on one grid. Return a summary dict. Never raises."""
    result = {
        "errors": [],
        "n_objects_4": 0,
        "n_objects_8": 0,
        "n_colors": 0,
        "background_freq": 0,
        "background_border": 0,
        "sym_vertical": False,
        "sym_horizontal": False,
        "sym_rot180": False,
        "sym_rot90": False,
        "sym_any": False,
        "periodic": False,
        "period": None,
        "n_lines": 0,
        "has_enclosure": False,
    }

    try:
        objs4 = extract_objects(grid, connectivity=4)
        objs8 = extract_objects(grid, connectivity=8)
        result["n_objects_4"] = len(objs4)
        result["n_objects_8"] = len(objs8)
    except Exception as e:
        result["errors"].append(f"objects: {e}")
        objs4 = []

    try:
        ca_freq   = analyze_colors(grid, background_method="frequency")
        ca_border = analyze_colors(grid, background_method="border")
        result["n_colors"]         = len(ca_freq.histogram)
        result["background_freq"]  = ca_freq.background
        result["background_border"]= ca_border.background
    except Exception as e:
        result["errors"].append(f"colors: {e}")

    try:
        sr = analyze_symmetry(grid)
        result["sym_vertical"]   = sr.vertical
        result["sym_horizontal"] = sr.horizontal
        result["sym_rot180"]     = sr.rotation_180
        result["sym_rot90"]      = sr.rotation_90
        result["sym_any"]        = sr.vertical or sr.horizontal or sr.rotation_180 or sr.rotation_90
    except Exception as e:
        result["errors"].append(f"symmetry: {e}")

    try:
        pr = detect_periodicity(grid)
        result["periodic"] = pr.found
        if pr.found:
            result["period"] = (pr.period_rows, pr.period_cols)
    except Exception as e:
        result["errors"].append(f"patterns: {e}")

    try:
        lines = find_lines(grid)  # adaptive min_length
        result["n_lines"] = len(lines)
    except Exception as e:
        result["errors"].append(f"lines: {e}")

    try:
        if len(objs4) >= 2:
            rels = relate_all(objs4, grid)
            result["has_enclosure"] = any(r.a_encloses_b or r.b_encloses_a for r in rels)
    except Exception as e:
        result["errors"].append(f"topology: {e}")

    return result


def _task_clean_flags(train_summaries: list[dict]) -> dict:
    """Derive clean-perception flags from a list of per-grid summaries."""
    all_have_2plus_objects = all(s["n_objects_4"] >= 2 for s in train_summaries)
    any_symmetry     = any(s["sym_any"]        for s in train_summaries)
    any_2plus_lines  = any(s["n_lines"] >= 2   for s in train_summaries)
    any_periodic     = any(s["periodic"]        for s in train_summaries)
    any_enclosure    = any(s["has_enclosure"]   for s in train_summaries)

    return {
        "objects":     all_have_2plus_objects,
        "symmetry":    any_symmetry,
        "lines":       any_2plus_lines,
        "periodicity": any_periodic,
        "enclosure":   any_enclosure,
        "any":         any([all_have_2plus_objects, any_symmetry,
                            any_2plus_lines, any_periodic, any_enclosure]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    tasks = Task.load_all(
        os.path.join(data_dir, 'arc-agi_training_challenges.json'),
        os.path.join(data_dir, 'arc-agi_training_solutions.json'),
    )

    total = len(tasks)
    print(f"Loaded {total} training tasks.\n")

    # Aggregate counters
    clean_counts    = defaultdict(int)   # criterion → count of tasks that pass
    error_tasks     = []
    sym_breakdown   = defaultdict(int)
    period_sizes    = defaultdict(int)
    obj_dist        = defaultdict(int)   # n_objects → count of grids
    line_dist       = defaultdict(int)   # n_lines   → count of grids
    structureless   = []                 # (task_id, task) for tasks with no structure

    task_logs = {}   # task_id → brief summary string (for verbose dump)

    for task_id, task in tasks.items():
        input_summaries  = [_analyze_grid(inp) for inp, _ in task.train_pairs]
        output_summaries = [_analyze_grid(out) for _, out in task.train_pairs]

        # Collect any errors
        all_errors = [e for s in input_summaries + output_summaries for e in s["errors"]]
        if all_errors:
            error_tasks.append((task_id, all_errors))

        flags = _task_clean_flags(input_summaries)

        for criterion, passed in flags.items():
            if passed:
                clean_counts[criterion] += 1

        if not flags["any"]:
            structureless.append((task_id, task))

        # Per-grid distributions
        for s in input_summaries:
            obj_dist[s["n_objects_4"]] += 1
            line_dist[s["n_lines"]] += 1

            if s["sym_vertical"]:   sym_breakdown["vertical"]   += 1
            if s["sym_horizontal"]: sym_breakdown["horizontal"]  += 1
            if s["sym_rot180"]:     sym_breakdown["rot180"]      += 1
            if s["sym_rot90"]:      sym_breakdown["rot90"]       += 1

            if s["periodic"] and s["period"]:
                period_sizes[s["period"]] += 1

        # Brief per-task log
        n_obj_avg = sum(s["n_objects_4"] for s in input_summaries) / len(input_summaries)
        sym_any   = flags["symmetry"]
        task_logs[task_id] = (
            f"objs≈{n_obj_avg:.1f}  sym={'Y' if sym_any else 'N'}  "
            f"lines={sum(s['n_lines'] for s in input_summaries)}  "
            f"periodic={'Y' if flags['periodicity'] else 'N'}  "
            f"enclosure={'Y' if flags['enclosure'] else 'N'}"
        )

    # ── Report ────────────────────────────────────────────────────────────────

    W = 64
    bar = "═" * W

    print(f"\n{'═'*W}")
    print(f"  PERCEPTION AUDIT  —  {total} training tasks")
    print(f"{'═'*W}\n")

    print(f"  {'CLEAN-PERCEPTION COUNTS':40s}  {'tasks':>6}  {'%':>6}")
    print(f"  {'-'*54}")
    order = ["any", "objects", "symmetry", "lines", "periodicity", "enclosure"]
    for crit in order:
        n = clean_counts[crit]
        pct = 100 * n / total
        bar_fill = "█" * int(pct / 2)
        print(f"  {crit:40s}  {n:6d}  {pct:5.1f}%  {bar_fill}")

    print(f"\n  {'SYMMETRY BREAKDOWN (training inputs)':40s}  {'grids':>6}")
    print(f"  {'-'*48}")
    for sym_type, count in sorted(sym_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {sym_type:40s}  {count:6d}")

    print(f"\n  {'OBJECT COUNT DISTRIBUTION (training inputs)':40s}  {'grids':>6}")
    print(f"  {'-'*48}")
    for n_obj in sorted(obj_dist):
        if n_obj <= 15:
            print(f"  {n_obj:2d} objects  {'█'*min(40, obj_dist[n_obj]//5):40s}  {obj_dist[n_obj]:6d}")

    print(f"\n  {'TOP TILE SIZES':40s}  {'tasks':>6}")
    print(f"  {'-'*48}")
    for size, count in sorted(period_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {str(size):40s}  {count:6d}")

    print(f"\n  {'LINE COUNT DISTRIBUTION (training inputs)':40s}  {'grids':>6}")
    print(f"  {'-'*48}")
    for n_lines in sorted(line_dist):
        if n_lines <= 20:
            print(f"  {n_lines:2d} lines    {'█'*min(40, line_dist[n_lines]//5):40s}  {line_dist[n_lines]:6d}")

    if error_tasks:
        print(f"\n  ERRORS IN {len(error_tasks)} TASKS:")
        for tid, errs in error_tasks[:10]:
            print(f"    {tid}: {errs}")
    else:
        print(f"\n  No errors across all {total} tasks.")

    print(f"\n{'═'*W}")
    print(f"  SUMMARY")
    print(f"{'═'*W}")
    clean_any = clean_counts["any"]
    print(f"  Tasks with clean perception (any criterion): "
          f"{clean_any} / {total}  ({100*clean_any/total:.1f}%)")
    print(f"  Tasks with NO detected structure:            "
          f"{total - clean_any} / {total}  ({100*(total-clean_any)/total:.1f}%)")
    print()

    # ── Structureless tasks ───────────────────────────────────────────────────
    COLOR_NAME = {
        0:"black", 1:"blue", 2:"red",  3:"green",  4:"yellow",
        5:"grey",  6:"fuchsia", 7:"orange", 8:"azure", 9:"maroon",
    }
    ANSI = {
        0:"\033[90m", 1:"\033[94m", 2:"\033[91m", 3:"\033[92m", 4:"\033[93m",
        5:"\033[37m", 6:"\033[95m", 7:"\033[33m", 8:"\033[96m", 9:"\033[31m",
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    CHARS = ".123456789"

    def _render_grid(grid, indent=6):
        pad = " " * indent
        for r in range(grid.rows):
            print(pad + " ".join(
                f"{BOLD if grid.get(r,c) != 0 else ''}{ANSI[grid.get(r,c)]}{CHARS[grid.get(r,c)]}{RESET}"
                for c in range(grid.cols)
            ))

    print(f"\n{'═'*W}")
    print(f"  STRUCTURELESS TASKS  ({len(structureless)} tasks — no criterion fired)")
    print(f"{'═'*W}")
    print(f"  These tasks have: <2 objects in at least one train input,")
    print(f"  no symmetry, no structural lines, no periodicity, no enclosure.\n")

    for task_id, task in structureless:
        print(f"  {'─'*60}")
        print(f"  Task: {task_id}   ({task.n_train} train pairs, {task.n_test} test inputs)")
        for i, (inp, out) in enumerate(task.train_pairs):
            ca = analyze_colors(inp, background_method="frequency")
            objs = extract_objects(inp)
            colors_str = ", ".join(
                f"{COLOR_NAME[c]}({ca.histogram[c]})" for c in sorted(ca.histogram) if c != 0
            ) or "none (all background)"
            print(f"\n    Train {i}  input {inp.rows}×{inp.cols}  "
                  f"objects={len(objs)}  colors={colors_str}")
            _render_grid(inp)
            print(f"    Train {i}  output {out.rows}×{out.cols}")
            _render_grid(out)

        print(f"\n    Test input {task.test_inputs[0].rows}×{task.test_inputs[0].cols}:")
        _render_grid(task.test_inputs[0])
        if task.test_solutions:
            print(f"    Test output (solution):")
            _render_grid(task.test_solutions[0])
        print()

    # ── Verbose per-task dump (optional) ─────────────────────────────────────
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print(f"\n{'─'*W}")
        print("  PER-TASK LOG")
        print(f"{'─'*W}")
        for task_id, log in task_logs.items():
            print(f"  {task_id}  {log}")


if __name__ == "__main__":
    main()
