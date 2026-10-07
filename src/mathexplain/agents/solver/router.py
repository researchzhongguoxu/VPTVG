"""Solver route selection."""

from __future__ import annotations

from mathexplain.agents.solver.algebra_solver import AlgebraSolver
from mathexplain.schemas.classification import ClassificationResult


class SolverRouter:
    """Select a concrete solver implementation from classification output."""

    def __init__(self, algebra_solver: AlgebraSolver | None = None) -> None:
        self.algebra_solver = algebra_solver or AlgebraSolver()

    def route(self, classification_result: ClassificationResult) -> AlgebraSolver | None:
        """Return a solver for supported routes."""

        if (
            classification_result.solver_route == "algebra_solver"
            and classification_result.task_type == "equation_solving"
        ):
            return self.algebra_solver
        return None
