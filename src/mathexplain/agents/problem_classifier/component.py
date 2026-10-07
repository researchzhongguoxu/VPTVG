"""Rule-based Problem Classifier component."""

from __future__ import annotations

from typing import Any

from mathexplain.schemas.classification import ClassificationResult, validate_classification_result_dict
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.reports import ParsingReport
from mathexplain.schemas.spr import SPR


class ProblemClassifier:
    """Lightweight classification and solver routing agent."""

    def classify(
        self,
        spr: SPR,
        emr: EMR,
        parsing_report: ParsingReport | None = None,
    ) -> ClassificationResult:
        """Classify a parsed problem and produce solver routing hints."""

        warnings: list[str] = []
        spr_type = self._problem_type_value(spr.problem_type)
        emr_type = self._problem_type_value(emr.source_problem_type)
        primary_type = self._primary_type(spr_type, emr_type)

        if spr_type != "unknown" and emr_type != "unknown" and spr_type != emr_type:
            warnings.append(
                f"SPR/EMR problem type conflict: spr={spr_type}, emr={emr_type}; using EMR."
            )

        task_type, solver_route, route_warning = self._route(primary_type, spr, emr)
        if route_warning:
            warnings.append(route_warning)

        difficulty = self._difficulty(primary_type, task_type, spr, emr)
        confidence = self._confidence(primary_type, task_type, warnings)
        requires_solver = solver_route not in {"unknown", "unsupported"}
        requires_verifier = requires_solver
        knowledge_units = self._knowledge_units(spr, task_type)

        result = {
            "problem_id": emr.problem_id or spr.metadata.get("problem_id") or "unknown_problem",
            "primary_type": primary_type,
            "task_type": task_type,
            "difficulty": difficulty,
            "knowledge_units": knowledge_units,
            "solver_route": solver_route,
            "solver_config": self._solver_config(solver_route, task_type, emr),
            "requires_solver": requires_solver,
            "requires_verifier": requires_verifier,
            "confidence": confidence,
            "warnings": warnings,
            "metadata": {
                "classifier_mode": "rule_based",
                "source_component": "ProblemClassifier",
                "spr_problem_type": spr_type,
                "emr_problem_type": emr_type,
                "emr_representation_type": emr.representation_type,
                "has_parsing_report": parsing_report is not None,
            },
        }
        return validate_classification_result_dict(result)

    @staticmethod
    def _problem_type_value(value: Any) -> str:
        return getattr(value, "value", str(value or "unknown"))

    @staticmethod
    def _primary_type(spr_type: str, emr_type: str) -> str:
        if emr_type != "unknown":
            return emr_type
        if spr_type != "unknown":
            return spr_type
        return "unknown"

    def _route(self, primary_type: str, spr: SPR, emr: EMR) -> tuple[str, str, str | None]:
        if primary_type == "algebra":
            if self._is_optimization_task(spr, emr):
                return "optimization", "algebra_optimization_solver", None
            if emr.representation_type == "equation_system" and emr.equations:
                return "equation_solving", "algebra_solver", None
            return (
                "unsupported",
                "unsupported",
                "Algebra problem lacks a supported equation or optimization structure.",
            )

        if primary_type in {"geometry", "proof"}:
            if self._has_proof_intent(spr):
                return "proof", "geometry_stub", "Geometry proof solver is not implemented yet."
            if self._is_coordinate_midpoint_algebra_task(spr, emr):
                return "equation_solving", "algebra_solver", None
            return (
                "geometry_reasoning",
                "geometry_stub",
                "Geometry reasoning solver is not implemented yet.",
            )

        if primary_type == "word_problem" and self._is_linear_word_problem(spr):
            return "equation_solving", "algebra_word_problem_solver", None

        return "unknown", "unknown", "Insufficient problem classification information."

    @staticmethod
    def _is_optimization_task(spr: SPR, emr: EMR) -> bool:
        text = ProblemClassifier._combined_text(spr, emr).lower()
        keywords = {
            "最小值",
            "最大值",
            "minimum",
            "maximum",
            "optimization",
            "minimum_value",
            "maximum_value",
        }
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _has_proof_intent(spr: SPR) -> bool:
        text = ProblemClassifier._combined_text(spr, None).lower()
        return any(keyword in text for keyword in ("prove", "proof", "求证"))

    @staticmethod
    def _is_coordinate_midpoint_algebra_task(spr: SPR, emr: EMR) -> bool:
        text = ProblemClassifier._combined_text(spr, emr).lower()
        return (
            "midpoint" in text
            and "coordinate" in text
            and not any(keyword in text for keyword in ("diagram", "figure", "shown below"))
        )

    @staticmethod
    def _combined_text(spr: SPR, emr: EMR | None) -> str:
        parts = [spr.problem_text or "", spr.problem_stem or ""]
        parts.extend(spr.knowledge_units)
        parts.extend(question.text for question in spr.questions)
        parts.extend(target.text for target in spr.targets)
        if emr is not None:
            parts.extend(goal.description or "" for goal in emr.goals)
            parts.extend(goal.goal_type for goal in emr.goals)
            parts.extend(str(item) for item in emr.metadata.get("choice_options", []))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _is_linear_word_problem(spr: SPR) -> bool:
        text = ProblemClassifier._combined_text(spr, None).lower()
        keywords = {
            "linear equations",
            "linear_equations",
            "linear equation",
            "arithmetic operations",
            "algebraic_modeling",
            "algebraic expression",
            "algebraic_expression",
            "working_backwards_strategy",
            "reverse_calculation",
            "fractions_and_halves",
            "fraction",
            "fractions",
            "half",
            "halves",
            "一半",
            "剩",
            "还剩",
            "几箱",
            "多少千克",
            "相距",
            "相遇",
            "速度比",
            "每小时",
            "平均每小时",
            "speed ratio",
            "meet",
            "distance",
            "per hour",
        }
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _difficulty(primary_type: str, task_type: str, spr: SPR, emr: EMR) -> str:
        if task_type in {"unknown", "unsupported"} or primary_type == "unknown":
            return "unknown"
        if primary_type == "algebra" and task_type == "equation_solving":
            if len(emr.variables) == 1 and len(emr.equations) == 1:
                return "elementary"
            return "intermediate"
        if primary_type == "algebra" and task_type == "optimization":
            return "intermediate"
        if primary_type == "word_problem" and task_type == "equation_solving":
            return "elementary"
        if task_type == "proof":
            if len(spr.targets) > 1 or len(spr.knowledge_units) >= 4:
                return "advanced"
            return "intermediate"
        if primary_type == "geometry":
            return "intermediate"
        return "unknown"

    @staticmethod
    def _confidence(primary_type: str, task_type: str, warnings: list[str]) -> float:
        if task_type in {"unknown", "unsupported"} or primary_type == "unknown":
            return 0.2
        base = 0.85 if not warnings else 0.7
        if "stub" in task_type:
            return 0.6
        return base

    @staticmethod
    def _knowledge_units(spr: SPR, task_type: str) -> list[str]:
        units = list(dict.fromkeys(spr.knowledge_units))
        inferred = {
            "equation_solving": "linear_equation",
            "optimization": "optimization",
            "proof": "geometry_proof",
            "geometry_reasoning": "geometry_reasoning",
        }.get(task_type)
        if inferred and inferred not in units:
            units.append(inferred)
        return units

    @staticmethod
    def _solver_config(solver_route: str, task_type: str, emr: EMR) -> dict[str, str]:
        solver_name_by_route = {
            "algebra_solver": "AlgebraSolver",
            "algebra_word_problem_solver": "AlgebraSolver",
            "algebra_optimization_solver": "AlgebraOptimizationSolver",
            "geometry_stub": "GeometrySolverStub",
            "unknown": "UnknownSolver",
            "unsupported": "UnsupportedSolver",
        }
        strategy_by_route = {
            "algebra_solver": "sympy_equation_solving",
            "algebra_word_problem_solver": "planner_toolcall_algebra_word_problem",
            "algebra_optimization_solver": "sympy_optimization_planned",
            "geometry_stub": "not_implemented_stub",
            "unknown": "no_route",
            "unsupported": "unsupported_route",
        }
        return {
            "solver_name": solver_name_by_route.get(solver_route, "UnknownSolver"),
            "expected_emr_type": emr.representation_type,
            "task_type": task_type,
            "strategy": strategy_by_route.get(solver_route, "no_route"),
        }
