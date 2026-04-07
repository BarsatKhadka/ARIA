from typing import List
from core.grid import Grid
from core.task import Task


def solve(task: Task) -> List[Grid]:
    """
    Stub solver. Returns an all-zero grid for every test input.
    This is the baseline — 0% accuracy, but the full pipeline runs end to end.
    Every future milestone replaces the logic inside this function.
    """
    return [
        Grid([[0] * inp.cols for _ in range(inp.rows)])
        for inp in task.test_inputs
    ]
