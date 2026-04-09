from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.grid import Grid
from core.task import Task
from perception.objects import Object, extract_objects
from perception.colors import ColorAnalysis, analyze_colors
from perception.symmetry import SymmetryResult, analyze_symmetry
from perception.patterns import PeriodicityResult, detect_periodicity
from perception.lines import Line, find_lines
from perception.topology import ObjectRelation, relate_all


# ── GridPerception ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridPerception:
    """All perception module outputs for a single grid."""

    # Objects
    objects:    Tuple[Object, ...]
    n_objects:  int
    background: int

    # Colors
    colors:   ColorAnalysis
    n_colors: int

    # Symmetry
    symmetry:     SymmetryResult
    has_symmetry: bool          # True if any of the four symmetry types holds

    # Periodicity
    periodicity: PeriodicityResult
    is_periodic: bool

    # Lines
    lines:   Tuple[Line, ...]
    n_lines: int

    # Topology
    relations:    Tuple[ObjectRelation, ...]
    has_enclosure: bool         # True if any object pair has an enclosure relation


# ── PairPerception ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairPerception:
    """Perception of one input/output training pair."""
    pair_index: int
    input:      GridPerception
    output:     GridPerception


# ── PerceptionReport ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PerceptionReport:
    """
    Full perception report for a task, aggregated across all training pairs.

    ``pair_perceptions`` contains per-pair raw observations.
    The remaining fields are cross-pair summaries derived from the input grids only.
    """
    n_pairs:          int
    pair_perceptions: Tuple[PairPerception, ...]

    # Cross-pair summaries (training inputs only)
    all_inputs_have_symmetry:  bool
    any_input_has_symmetry:    bool
    all_inputs_periodic:       bool
    any_input_periodic:        bool
    all_inputs_have_enclosure: bool
    any_input_has_enclosure:   bool
    consistent_background:     Optional[int]  # same bg across all inputs; None if varies
    consistent_n_objects:      Optional[int]  # same object count in all inputs; None if varies


# ── Internal helpers ──────────────────────────────────────────────────────────

def _perceive_grid(grid: Grid) -> GridPerception:
    colors = analyze_colors(grid, background_method="frequency")
    bg     = colors.background

    objects   = tuple(extract_objects(grid, connectivity=4, background=bg))
    symmetry  = analyze_symmetry(grid)
    periodicity = detect_periodicity(grid)
    lines     = tuple(find_lines(grid, background=bg))
    relations = (
        tuple(relate_all(list(objects), grid, background=bg))
        if len(objects) >= 2
        else ()
    )
    has_enclosure = any(r.a_encloses_b or r.b_encloses_a for r in relations)

    return GridPerception(
        objects=objects,
        n_objects=len(objects),
        background=bg,
        colors=colors,
        n_colors=len(colors.histogram),
        symmetry=symmetry,
        has_symmetry=(
            symmetry.vertical or symmetry.horizontal
            or symmetry.rotation_180 or symmetry.rotation_90
        ),
        periodicity=periodicity,
        is_periodic=periodicity.found,
        lines=lines,
        n_lines=len(lines),
        relations=relations,
        has_enclosure=has_enclosure,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def perceive(task: Task) -> PerceptionReport:
    """
    Run all perception modules on every training pair of *task*.

    Each training pair produces a :class:`PairPerception` with a
    :class:`GridPerception` for both the input and the output grid.
    Cross-pair summaries (``consistent_background``, symmetry flags, etc.)
    are derived from the input grids only and stored at the report level.

    Parameters
    ----------
    task:
        A :class:`~core.task.Task` with at least one training pair.

    Returns
    -------
    :class:`PerceptionReport`
    """
    pair_perceptions: List[PairPerception] = []
    for i, (inp, out) in enumerate(task.train_pairs):
        pair_perceptions.append(PairPerception(
            pair_index=i,
            input=_perceive_grid(inp),
            output=_perceive_grid(out),
        ))

    inputs = [pp.input for pp in pair_perceptions]

    backgrounds    = [g.background for g in inputs]
    n_objects_list = [g.n_objects  for g in inputs]

    return PerceptionReport(
        n_pairs=len(pair_perceptions),
        pair_perceptions=tuple(pair_perceptions),
        all_inputs_have_symmetry=all(g.has_symmetry  for g in inputs),
        any_input_has_symmetry=any(g.has_symmetry    for g in inputs),
        all_inputs_periodic=all(g.is_periodic        for g in inputs),
        any_input_periodic=any(g.is_periodic         for g in inputs),
        all_inputs_have_enclosure=all(g.has_enclosure for g in inputs),
        any_input_has_enclosure=any(g.has_enclosure   for g in inputs),
        consistent_background=(
            backgrounds[0] if len(set(backgrounds)) == 1 else None
        ),
        consistent_n_objects=(
            n_objects_list[0] if len(set(n_objects_list)) == 1 else None
        ),
    )
