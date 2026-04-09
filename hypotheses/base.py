from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.grid import Grid


@dataclass
class Hypothesis:
    """
    A candidate rule that maps an input Grid to a predicted output Grid.

    Attributes
    ----------
    name:
        Human-readable description of the rule (used in logs/debugging).
    apply:
        Callable that takes an input Grid and returns the predicted output Grid.
        Must be a pure function — it should not mutate the input.
    complexity:
        Integer complexity score for MDL tie-breaking.  Lower = simpler = preferred
        when multiple hypotheses pass all training pairs.
    """
    name:       str
    apply:      Callable[[Grid], Grid]
    complexity: int = field(default=1)
