"""Solver component."""

from __future__ import annotations

from mathexplain.agents.solver.algebra_solver import AlgebraSolver
from mathexplain.agents.solver.planner import DeepSeekPlanner
from mathexplain.agents.solver.router import SolverRouter
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR


class Solver:
    """Solver v1 that returns a single main SolutionChain."""

    def __init__(
        self,
        router: SolverRouter | None = None,
        planner: DeepSeekPlanner | None = None,
        use_planner: bool = True,
    ) -> None:
        self.router = router or SolverRouter()
        self.planner = planner
        self.use_planner = use_planner

    def solve(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None = None,
    ) -> SolutionChain:
        """Solve an EMR according to the classification route."""

        concrete_solver = self.router.route(classification_result)
        if concrete_solver is not None and self._should_solve_from_emr_first(emr, classification_result):
            direct_chain = concrete_solver.solve(emr, classification_result, spr=spr)
            if direct_chain.final_answer is not None:
                return direct_chain

        planner_draft = self._planning_draft(emr, classification_result, spr, concrete_solver is None)
        if concrete_solver is None:
            if planner_draft is not None:
                return self.router.algebra_solver.solve_tool_request(
                    emr,
                    classification_result,
                    planner_draft,
                    spr,
                )
            return AlgebraSolver.unsupported_chain(
                emr,
                classification_result,
                self._unsupported_reason(classification_result),
                planner_draft,
            )
        if planner_draft and planner_draft.get("planner_tool_calls"):
            return concrete_solver.solve_tool_request(emr, classification_result, planner_draft, spr)
        return concrete_solver.solve(emr, classification_result, planner_draft, spr=spr)

    @staticmethod
    def _should_solve_from_emr_first(
        emr: EMR,
        classification_result: ClassificationResult,
    ) -> bool:
        return (
            classification_result.solver_route == "algebra_solver"
            and emr.representation_type == "equation_system"
            and bool(emr.equations)
        )

    def _planning_draft(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None,
        route_requires_planner: bool,
    ) -> dict[str, object] | None:
        if not self.use_planner or self.planner is None:
            if not self.use_planner:
                return None
            if spr is None or not route_requires_planner:
                return None
            self.planner = DeepSeekPlanner()
        try:
            return self.planner.plan(emr, classification_result, spr)
        except Exception as exc:  # pragma: no cover - provider errors vary.
            return {
                "planner_provider": "unknown",
                "planner_model": "unknown",
                "planner_role": "planning_draft",
                "planner_raw_output": "",
                "planner_output_summary": "",
                "planner_tool_calls": [],
                "planner_parse_error": None,
                "planner_error": str(exc),
            }

    @staticmethod
    def _unsupported_reason(classification_result: ClassificationResult) -> str:
        return (
            "Solver v1 only supports solver_route='algebra_solver' "
            "with task_type='equation_solving'. "
            f"Got solver_route={classification_result.solver_route}, "
            f"task_type={classification_result.task_type}."
        )
