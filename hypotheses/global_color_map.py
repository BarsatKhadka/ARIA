"""
global_color_map.py — Hypothesis: apply a consistent color mapping to every cell.

Signal used: DifferentialReport.color_mapping is RELEVANT and non-empty.

Example rule it captures:
  "Every blue cell becomes red, every green cell becomes yellow."
  This covers ~5-8% of ARC tasks.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.grid import Grid
from differential.feature_classifier import DifferentialReport, Confidence
from hypotheses.base import Hypothesis
from perception.report import PerceptionReport


def generate(
    perception: PerceptionReport,
    differential: DifferentialReport,
) -> List[Hypothesis]:
    """
    Return a single global-color-map hypothesis if the differential shows a
    consistent color mapping across all training pairs, else return [].
    """
    cm = differential.color_mapping
    if cm.confidence != Confidence.RELEVANT or not cm.value:
        return []

    mapping: Dict[int, int] = {int(f): int(t) for f, t in cm.value}
    if not mapping:
        return []

    def apply(grid: Grid, _m: Dict[int, int] = mapping) -> Grid:
        new_data = grid.data.copy()
        for from_c, to_c in _m.items():
            new_data[grid.data == from_c] = to_c
        return Grid(new_data.tolist())

    return [Hypothesis(
        name=f"global_color_map({mapping})",
        apply=apply,
        complexity=1,
    )]
