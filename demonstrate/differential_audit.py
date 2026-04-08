"""
differential_audit.py — Validation of the differential engine on 50 ARC training tasks.

For each task:
  - Runs classify() to build a DifferentialReport
  - Prints the report with ANSI colour coding (RELEVANT=green, INCIDENTAL=yellow, AMBIGUOUS=grey)
  - Renders the first training pair as a mini-grid for manual context
  - Auto-scores whether the RELEVANT features meaningfully capture the rule

Auto-score heuristic
--------------------
A task is "captured" when ALL of:
  (a) transform_type is RELEVANT  — we know what kind of change happens
  (b) at least one substantive feature is RELEVANT with a non-trivial value:
        color_mapping non-empty,  OR
        movement_direction non-None,  OR
        selected_colors non-empty,  OR
        object_count_delta non-zero

This intentionally biases toward tasks the engine can describe fully.
Tasks where the rule is purely spatial / shape-morphing without object-level
features will score as INCIDENTAL and lower the accuracy — correctly, since the
engine cannot yet explain them.

Run from the ARIA root:
    python demonstrate/differential_audit.py
    python demonstrate/differential_audit.py --n 100
    python demonstrate/differential_audit.py --verbose
"""

import sys, os, json, traceback, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from core.task   import Task
from core.grid   import Grid
from differential.feature_classifier import (
    classify, DifferentialReport, Confidence, Feature
)


# ── ANSI palette ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

GREEN  = "\033[92m"   # RELEVANT
YELLOW = "\033[93m"   # INCIDENTAL
GREY   = "\033[90m"   # AMBIGUOUS
RED    = "\033[91m"   # errors / not captured

# Grid colours (ARC 10-colour palette)
_CELL_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[91m", 3: "\033[92m", 4: "\033[93m",
    5: "\033[37m", 6: "\033[95m", 7: "\033[33m", 8: "\033[96m", 9: "\033[31m",
}
_CELL_CHAR = ".123456789"


def _conf_color(c: Confidence) -> str:
    return {
        Confidence.RELEVANT:   GREEN,
        Confidence.INCIDENTAL: YELLOW,
        Confidence.AMBIGUOUS:  GREY,
    }[c]


def _conf_tag(c: Confidence) -> str:
    return {
        Confidence.RELEVANT:   "RELEVANT  ",
        Confidence.INCIDENTAL: "INCIDENTAL",
        Confidence.AMBIGUOUS:  "AMBIGUOUS ",
    }[c]


# ── Grid renderer ─────────────────────────────────────────────────────────────

def _render_grid(grid: Grid, indent: int = 4) -> None:
    pad = " " * indent
    for r in range(grid.rows):
        row = " ".join(
            f"{BOLD if grid.get(r, c) != 0 else ''}"
            f"{_CELL_ANSI[grid.get(r, c)]}"
            f"{_CELL_CHAR[grid.get(r, c)]}"
            f"{RESET}"
            for c in range(grid.cols)
        )
        print(pad + row)


# ── Feature printer ───────────────────────────────────────────────────────────

def _fmt_value(v) -> str:
    """Compact string for a feature value."""
    if v is None:
        return "–"
    if isinstance(v, frozenset):
        if not v:
            return "{}"
        items = sorted(str(x) for x in v)
        return "{" + ", ".join(items) + "}"
    return str(v)


def _print_feature(label: str, f: Feature, indent: int = 2) -> None:
    pad    = " " * indent
    color  = _conf_color(f.confidence)
    tag    = _conf_tag(f.confidence)
    val    = _fmt_value(f.value)
    pp_str = ""
    if f.confidence != Confidence.RELEVANT and len(f.per_pair) <= 6:
        pp_str = f"  per_pair={f.per_pair}"
    print(f"{pad}{color}{tag}{RESET}  {BOLD}{label:<30s}{RESET}  {val}{DIM}{pp_str}{RESET}")


# ── Auto-scorer ───────────────────────────────────────────────────────────────

def _score_task(r: DifferentialReport) -> tuple[bool, str]:
    """
    Returns (captured: bool, reason: str).

    A task is "captured" when at least one of the following holds:

    A) color_mapping is RELEVANT and non-empty
       → the colour-transformation rule is fully identified

    B) movement_direction is RELEVANT and non-None
       → the spatial-movement rule is identified

    C) transform_type is RELEVANT (and non-trivial) AND at least one
       selection / count feature is RELEVANT with a meaningful value:
         - selected_colors non-empty
         - selection_is_largest/smallest/border_touching/enclosed is True
         - object_count_delta non-zero
       → we know WHAT changes and can distinguish WHICH objects

    Rationale: transform_type being INCIDENTAL (alignment noise across pairs)
    does NOT disqualify a task when the colour rule or direction is invariant —
    those are the actionable signals.  Conversely, transform_type being
    RELEVANT but selecting no objects is not enough on its own.
    """
    reasons: list[str] = []

    # ── Criterion A: invariant colour mapping ─────────────────────────────────
    if (r.color_mapping.confidence == Confidence.RELEVANT
            and r.color_mapping.value):
        reasons.append(f"color_mapping={_fmt_value(r.color_mapping.value)}")

    # ── Criterion B: invariant movement direction ─────────────────────────────
    if (r.movement_direction.confidence == Confidence.RELEVANT
            and r.movement_direction.value is not None):
        reasons.append(f"direction={r.movement_direction.value}")

    if reasons:
        return True, " | ".join(reasons)

    # ── Criterion C: transform type + selection ───────────────────────────────
    tt = r.transform_type.value or frozenset()
    tt_ok = (r.transform_type.confidence == Confidence.RELEVANT
             and tt not in (frozenset(), frozenset({"none"})))

    if not tt_ok:
        return False, "transform_type not RELEVANT"

    selection_reasons: list[str] = []

    if r.selected_colors.confidence == Confidence.RELEVANT and r.selected_colors.value:
        selection_reasons.append(f"selected_colors={_fmt_value(r.selected_colors.value)}")

    for feat, label in [
        (r.selection_is_largest,         "is_largest"),
        (r.selection_is_smallest,        "is_smallest"),
        (r.selection_is_border_touching, "is_border"),
        (r.selection_is_enclosed,        "is_enclosed"),
    ]:
        if feat.confidence == Confidence.RELEVANT and feat.value is True:
            selection_reasons.append(label)

    if (r.object_count_delta.confidence == Confidence.RELEVANT
            and r.object_count_delta.value is not None
            and r.object_count_delta.value != 0):
        selection_reasons.append(f"count_delta={r.object_count_delta.value}")

    if selection_reasons:
        return True, f"transform={_fmt_value(tt)} | " + " | ".join(selection_reasons)

    return False, f"transform_type={_fmt_value(tt)} but no substantive RELEVANT features"


# ── Per-task report printer ───────────────────────────────────────────────────

def _print_task(task_id: str, task: Task, report: DifferentialReport,
                captured: bool, reason: str, verbose: bool) -> None:
    W = 72
    verdict_color = GREEN if captured else RED
    verdict       = "CAPTURED" if captured else "MISSED  "

    print(f"\n{'─'*W}")
    print(f"  {BOLD}{task_id}{RESET}   "
          f"{task.n_train} train pairs   "
          f"{verdict_color}{BOLD}{verdict}{RESET}  {DIM}{reason}{RESET}")
    print(f"{'─'*W}")

    # ── Mini-grid of first training pair ─────────────────────────────────────
    inp, out = task.train_pairs[0]
    print(f"  First pair  input {inp.rows}×{inp.cols}  →  output {out.rows}×{out.cols}")
    max_rows = max(inp.rows, out.rows)
    # side-by-side if both fit in ~40 cols total, else stacked
    side_by_side = inp.cols + out.cols + 4 <= 40 and max_rows <= 15
    if side_by_side:
        inp_lines = []
        for r in range(inp.rows):
            inp_lines.append("  ".join(
                f"{BOLD if inp.get(r, c) != 0 else ''}"
                f"{_CELL_ANSI[inp.get(r, c)]}{_CELL_CHAR[inp.get(r, c)]}{RESET}"
                for c in range(inp.cols)
            ))
        out_lines = []
        for r in range(out.rows):
            out_lines.append("  ".join(
                f"{BOLD if out.get(r, c) != 0 else ''}"
                f"{_CELL_ANSI[out.get(r, c)]}{_CELL_CHAR[out.get(r, c)]}{RESET}"
                for c in range(out.cols)
            ))
        n = max(len(inp_lines), len(out_lines))
        inp_lines  += [""] * (n - len(inp_lines))
        out_lines  += [""] * (n - len(out_lines))
        gap_cols   = inp.cols * 2 + 2   # approximate ANSI-stripped width
        for il, ol in zip(inp_lines, out_lines):
            # pad with spaces for alignment (rough — ANSI codes inflate string length)
            print(f"    {il}    →    {ol}")
    else:
        print(f"  input:")
        _render_grid(inp, indent=4)
        print(f"  output:")
        _render_grid(out, indent=4)

    # ── Feature table ─────────────────────────────────────────────────────────
    print()
    _print_feature("transform_type",             report.transform_type)
    _print_feature("color_mapping",              report.color_mapping)
    _print_feature("selected_colors",            report.selected_colors)
    _print_feature("selection_is_largest",       report.selection_is_largest)
    _print_feature("selection_is_smallest",      report.selection_is_smallest)
    _print_feature("selection_is_border_touch",  report.selection_is_border_touching)
    _print_feature("selection_is_enclosed",      report.selection_is_enclosed)
    _print_feature("movement_direction",         report.movement_direction)
    _print_feature("movement_vector",            report.movement_vector)
    _print_feature("object_count_delta",         report.object_count_delta)

    if verbose:
        print(f"\n  {DIM}Per-pair object counts:{RESET}")
        for pa in report.pair_analyses:
            print(f"    pair {pa.pair_index}:  "
                  f"inp_objs={pa.n_input_objects}  "
                  f"out_objs={pa.n_output_objects}  "
                  f"transforms={set(pa.transform_types)}  "
                  f"transitions={pa.color_transitions}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=50,
                        help="Number of tasks to audit (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-pair breakdown for every task")
    parser.add_argument("--missed",  action="store_true",
                        help="Only print tasks that were NOT captured")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    tasks = Task.load_all(
        os.path.join(data_dir, 'arc-agi_training_challenges.json'),
        os.path.join(data_dir, 'arc-agi_training_solutions.json'),
    )

    task_ids  = list(tasks.keys())[:args.n]
    n_total   = len(task_ids)

    print(f"\n{'═'*72}")
    print(f"  DIFFERENTIAL AUDIT  —  first {n_total} training tasks")
    print(f"{'═'*72}")
    print(f"  {GREEN}green{RESET}  = RELEVANT (invariant across all pairs — candidate rule)")
    print(f"  {YELLOW}yellow{RESET} = INCIDENTAL (varies — task-specific data)")
    print(f"  {GREY}grey{RESET}   = AMBIGUOUS (only one pair — cannot confirm)")

    captured_ids   = []
    missed_ids     = []
    error_ids      = []

    for task_id in task_ids:
        task = tasks[task_id]
        try:
            report = classify(task)
            ok, reason = _score_task(report)
            if ok:
                captured_ids.append(task_id)
            else:
                missed_ids.append(task_id)

            if not args.missed or not ok:
                _print_task(task_id, task, report, ok, reason, args.verbose)

        except Exception as e:
            error_ids.append((task_id, str(e)))
            print(f"\n  {RED}ERROR{RESET} {task_id}: {e}")
            if args.verbose:
                traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────

    n_cap   = len(captured_ids)
    n_err   = len(error_ids)
    n_scored = n_total - n_err
    pct     = 100 * n_cap / n_scored if n_scored else 0.0

    W = 72
    print(f"\n{'═'*W}")
    print(f"  DIFFERENTIAL ACCURACY SUMMARY")
    print(f"{'═'*W}")
    print(f"  Tasks audited:   {n_total}")
    print(f"  Errors:          {n_err}")
    print(f"  Captured:        {GREEN}{BOLD}{n_cap}{RESET} / {n_scored}   "
          f"({GREEN}{BOLD}{pct:.1f}%{RESET})")
    print(f"  Missed:          {len(missed_ids)} / {n_scored}")

    # bar
    bar_len = 48
    filled  = int(bar_len * n_cap / n_scored) if n_scored else 0
    bar_color = GREEN if pct >= 60 else (YELLOW if pct >= 40 else RED)
    print(f"\n  {bar_color}{'█'*filled}{GREY}{'░'*(bar_len-filled)}{RESET}  {bar_color}{pct:.1f}%{RESET}")

    if pct >= 60:
        verdict = f"{GREEN}PASS — cross-example differencing extracts sufficient signal{RESET}"
    elif pct >= 40:
        verdict = f"{YELLOW}MARGINAL — some signal, but rule extraction needs strengthening{RESET}"
    else:
        verdict = f"{RED}FAIL — differential approach needs rethinking{RESET}"
    print(f"\n  {verdict}\n")

    # missed breakdown
    if missed_ids:
        print(f"  Missed tasks ({len(missed_ids)}):")
        for tid in missed_ids:
            print(f"    {GREY}{tid}{RESET}")

    if error_ids:
        print(f"\n  Error tasks ({n_err}):")
        for tid, msg in error_ids:
            print(f"    {RED}{tid}{RESET}: {msg}")

    print()


if __name__ == "__main__":
    main()
