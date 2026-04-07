from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from core.grid import Grid


@dataclass
class EvalResult:
    exact_match: bool
    partial_score: float        # fraction of cells that are correct (0.0 – 1.0)
    correct_cells: int
    total_cells: int
    shape_match: bool


class Task:
    def __init__(
        self,
        task_id: str,
        train_pairs: List[Tuple[Grid, Grid]],
        test_inputs: List[Grid],
        test_solutions: Optional[List[Grid]] = None,
    ):
        self.task_id = task_id
        self.train_pairs = train_pairs          # [(input_grid, output_grid), ...]
        self.test_inputs = test_inputs          # [input_grid, ...]
        self.test_solutions = test_solutions    # [output_grid, ...] or None

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        task_id: str,
        task_dict: dict,
        solution_list: Optional[list] = None,
    ) -> "Task":
        """
        Build a Task from the raw dicts loaded from JSON.

        task_dict   — one entry from arc-agi_training/evaluation/test_challenges.json
        solution_list — the corresponding entry from the solutions file (a list of
                        grids, one per test input), or None for blind test tasks.
        """
        train_pairs = [
            (Grid(pair["input"]), Grid(pair["output"]))
            for pair in task_dict["train"]
        ]
        test_inputs = [Grid(t["input"]) for t in task_dict["test"]]
        test_solutions = (
            [Grid(sol) for sol in solution_list]
            if solution_list is not None
            else None
        )
        return cls(task_id, train_pairs, test_inputs, test_solutions)

    @classmethod
    def from_json(
        cls,
        task_id: str,
        challenges_path: str,
        solutions_path: Optional[str] = None,
    ) -> "Task":
        """Load a single task by ID directly from the dataset files."""
        with open(challenges_path) as f:
            challenges = json.load(f)
        if task_id not in challenges:
            raise KeyError(f"Task '{task_id}' not found in {challenges_path}")

        solution_list = None
        if solutions_path is not None:
            with open(solutions_path) as f:
                solutions = json.load(f)
            solution_list = solutions.get(task_id)

        return cls.from_dict(task_id, challenges[task_id], solution_list)

    @classmethod
    def load_all(
        cls,
        challenges_path: str,
        solutions_path: Optional[str] = None,
    ) -> Dict[str, "Task"]:
        """
        Load every task from a challenges file.
        Returns a dict of {task_id: Task}.

        Use for training/evaluation sets. For the blind test set,
        omit solutions_path — every task will have has_solutions=False.
        """
        with open(challenges_path) as f:
            challenges = json.load(f)

        solutions = {}
        if solutions_path is not None:
            with open(solutions_path) as f:
                solutions = json.load(f)

        return {
            task_id: cls.from_dict(task_id, task_dict, solutions.get(task_id))
            for task_id, task_dict in challenges.items()
        }

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_train(self) -> int:
        return len(self.train_pairs)

    @property
    def n_test(self) -> int:
        return len(self.test_inputs)

    @property
    def has_solutions(self) -> bool:
        return self.test_solutions is not None

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_index: int,
        predicted: Union[Grid, Tuple[Grid, Grid]],
    ) -> EvalResult:
        """
        Score a prediction for one test input.

        predicted may be:
          - a single Grid       — scored as both attempt_1 and attempt_2
          - Tuple[Grid, Grid]   — attempt_1 and attempt_2 scored separately;
                                  the better result is returned.

        Exact match   — shapes match AND every cell is identical.
        Partial score — fraction of cells correct; 0.0 if shapes differ.
        """
        if not self.has_solutions:
            raise RuntimeError(f"Task {self.task_id} has no solutions loaded.")

        if isinstance(predicted, tuple):
            r1 = self._score_single(test_index, predicted[0])
            r2 = self._score_single(test_index, predicted[1])
            # exact match wins outright; otherwise take higher partial score
            if r1.exact_match:
                return r1
            if r2.exact_match:
                return r2
            return r1 if r1.partial_score >= r2.partial_score else r2

        return self._score_single(test_index, predicted)

    def _score_single(self, test_index: int, predicted: Grid) -> EvalResult:
        ground_truth = self.test_solutions[test_index]
        shape_match = predicted.shape == ground_truth.shape

        if not shape_match:
            return EvalResult(
                exact_match=False,
                partial_score=0.0,
                correct_cells=0,
                total_cells=ground_truth.rows * ground_truth.cols,
                shape_match=False,
            )

        correct_cells = int((predicted.data == ground_truth.data).sum())
        total_cells = ground_truth.rows * ground_truth.cols

        return EvalResult(
            exact_match=correct_cells == total_cells,
            partial_score=correct_cells / total_cells,
            correct_cells=correct_cells,
            total_cells=total_cells,
            shape_match=True,
        )

    def evaluate_all(
        self,
        predictions: List[Union[Grid, Tuple[Grid, Grid]]],
    ) -> List[EvalResult]:
        """Score predictions for all test inputs at once."""
        if len(predictions) != self.n_test:
            raise ValueError(
                f"Expected {self.n_test} predictions, got {len(predictions)}"
            )
        return [self.evaluate(i, pred) for i, pred in enumerate(predictions)]

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        sol = "with solutions" if self.has_solutions else "no solutions"
        return (
            f"Task(id={self.task_id}, "
            f"train={self.n_train}, "
            f"test={self.n_test}, "
            f"{sol})"
        )
