from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from core.grid import Grid
from core.task import Task, EvalResult


# A solver is any function that takes a Task and returns one Grid per test input
SolverFn = Callable[[Task], List[Grid]]


@dataclass
class TaskResult:
    task_id: str
    eval_results: List[EvalResult]          # one per test input
    predictions: List[Tuple[Grid, Grid]]    # [(attempt_1, attempt_2), ...]
    elapsed_sec: float

    @property
    def solved(self) -> bool:
        """True if every test input was exactly matched."""
        return all(r.exact_match for r in self.eval_results)

    @property
    def best_partial(self) -> float:
        """Highest partial score across all test inputs."""
        if not self.eval_results:
            return 0.0
        return max(r.partial_score for r in self.eval_results)


@dataclass
class EvalSummary:
    total_tasks: int
    solved_tasks: int
    total_test_inputs: int
    solved_test_inputs: int
    avg_partial_score: float
    total_elapsed_sec: float
    task_results: Dict[str, TaskResult] = field(repr=False)

    @property
    def accuracy(self) -> float:
        """Fraction of tasks fully solved (all test inputs exact-matched)."""
        if self.total_tasks == 0:
            return 0.0
        return self.solved_tasks / self.total_tasks

    def __str__(self) -> str:
        return (
            f"Tasks solved    : {self.solved_tasks}/{self.total_tasks} "
            f"({self.accuracy:.1%})\n"
            f"Inputs solved   : {self.solved_test_inputs}/{self.total_test_inputs}\n"
            f"Avg partial     : {self.avg_partial_score:.1%}\n"
            f"Time            : {self.total_elapsed_sec:.1f}s"
        )


class Evaluator:
    def __init__(
        self,
        challenges_path: str,
        solutions_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._log(f"Loading tasks from {challenges_path} ...")
        self.tasks: Dict[str, Task] = Task.load_all(challenges_path, solutions_path)
        self._log(f"Loaded {len(self.tasks)} tasks.")

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        solver: SolverFn,
        task_ids: Optional[List[str]] = None,
    ) -> EvalSummary:
        """
        Run solver over all (or a subset of) tasks.

        solver     — function(Task) -> List[Grid], one Grid per test input.
                     For submission format, two attempts per test input are
                     needed; if the solver returns only one grid per input,
                     the second attempt is a copy of the first.
        task_ids   — optional list of IDs to restrict evaluation to a subset.
        """
        ids = task_ids if task_ids is not None else list(self.tasks.keys())
        task_results: Dict[str, TaskResult] = {}

        for i, task_id in enumerate(ids):
            task = self.tasks[task_id]
            t0 = time.perf_counter()

            try:
                raw_preds = solver(task)
            except Exception as e:
                self._log(f"  [{i+1}/{len(ids)}] {task_id} ERROR: {e}")
                raw_preds = [Grid([[0]]) for _ in range(task.n_test)]

            elapsed = time.perf_counter() - t0

            # Normalise to [(attempt_1, attempt_2), ...] — one pair per test input
            predictions = self._normalise_predictions(raw_preds, task.n_test)

            # Evaluate attempt_1 against ground truth (if solutions available)
            eval_results: List[EvalResult] = []
            if task.has_solutions:
                eval_results = [
                    task.evaluate(j, pred_pair[0])
                    for j, pred_pair in enumerate(predictions)
                ]

            task_result = TaskResult(
                task_id=task_id,
                eval_results=eval_results,
                predictions=predictions,
                elapsed_sec=elapsed,
            )
            task_results[task_id] = task_result

            if self.verbose and task.has_solutions:
                status = "SOLVED" if task_result.solved else f"{task_result.best_partial:.0%}"
                self._log(f"  [{i+1}/{len(ids)}] {task_id}  {status}  ({elapsed:.2f}s)")

        return self._summarise(task_results)

    # ── Submission export ─────────────────────────────────────────────────────

    def save_submission(
        self,
        summary: EvalSummary,
        output_path: str = "submission.json",
    ):
        """
        Write predictions to a submission.json in the required ARC format:

        {
          "task_id": [
            {"attempt_1": [[...]], "attempt_2": [[...]]},
            ...   (one entry per test input)
          ],
          ...
        }
        """
        submission = {}
        for task_id, task_result in summary.task_results.items():
            submission[task_id] = [
                {
                    "attempt_1": attempt_1.to_list(),
                    "attempt_2": attempt_2.to_list(),
                }
                for attempt_1, attempt_2 in task_result.predictions
            ]

        with open(output_path, "w") as f:
            json.dump(submission, f)

        self._log(f"Submission saved to {output_path}  ({len(submission)} tasks)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalise_predictions(
        self,
        raw: List[Grid],
        n_test: int,
    ) -> List[Tuple[Grid, Grid]]:
        """
        Solvers return either:
          - List[Grid]            — one grid per test input (attempt_2 = copy of attempt_1)
          - List[Tuple[Grid,Grid]]— two attempts already provided

        This normalises either form into List[Tuple[Grid, Grid]].
        If the solver returned too few predictions, pad with a 1x1 black grid.
        """
        fallback = Grid([[0]])

        # Pad if the solver returned fewer grids than needed
        while len(raw) < n_test:
            raw.append(fallback)

        pairs = []
        for item in raw[:n_test]:
            if isinstance(item, tuple):
                a1, a2 = item
            else:
                a1 = item
                a2 = item   # duplicate as second attempt
            pairs.append((a1, a2))

        return pairs

    def _summarise(self, task_results: Dict[str, TaskResult]) -> EvalSummary:
        tasks_with_solutions = [
            tr for tr in task_results.values() if tr.eval_results
        ]
        solved_tasks = sum(1 for tr in tasks_with_solutions if tr.solved)
        total_inputs = sum(len(tr.eval_results) for tr in tasks_with_solutions)
        solved_inputs = sum(
            sum(1 for r in tr.eval_results if r.exact_match)
            for tr in tasks_with_solutions
        )
        all_partial = [
            r.partial_score
            for tr in tasks_with_solutions
            for r in tr.eval_results
        ]
        avg_partial = sum(all_partial) / len(all_partial) if all_partial else 0.0
        total_time = sum(tr.elapsed_sec for tr in task_results.values())

        return EvalSummary(
            total_tasks=len(tasks_with_solutions),
            solved_tasks=solved_tasks,
            total_test_inputs=total_inputs,
            solved_test_inputs=solved_inputs,
            avg_partial_score=avg_partial,
            total_elapsed_sec=total_time,
            task_results=task_results,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
