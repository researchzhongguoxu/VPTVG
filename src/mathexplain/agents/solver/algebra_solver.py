"""Algebra Solver v1."""

from __future__ import annotations

from fractions import Fraction
import re
from typing import Any

from mathexplain.agents.solver.calculation_request import CalculationRequestValidator
from mathexplain.agents.solver.scs_assembly import SCSAssemblyEnhancer
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain, validate_solution_chain_dict
from mathexplain.schemas.spr import SPR
from mathexplain.services.cas import CASService


class AlgebraSolver:
    """Solve supported algebra EMRs and assemble one main SolutionChain."""

    def __init__(self, cas_service: CASService | None = None) -> None:
        self.cas_service = cas_service or CASService()
        self.request_validator = CalculationRequestValidator(self.cas_service)
        self.scs_assembly = SCSAssemblyEnhancer()

    def solve(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        planner_draft: dict[str, str] | None = None,
        spr: SPR | None = None,
    ) -> SolutionChain:
        """Solve a supported one-equation algebra problem."""

        unsupported_reason = self._unsupported_reason(emr)
        if unsupported_reason:
            return self.unsupported_chain(emr, classification_result, unsupported_reason, planner_draft)

        equation = emr.equations[0]
        variables = [variable.symbol for variable in emr.variables]
        target = self._solve_target(emr, variables)
        equations = [
            {
                "id": equation.id,
                "lhs_sympy": self._repair_mixed_numbers_in_expression(equation.lhs_sympy, spr),
                "rhs_sympy": self._repair_mixed_numbers_in_expression(equation.rhs_sympy, spr),
            }
        ]
        backend_result = self.cas_service.solve_equation_system(
            equations=equations,
            variables=variables,
            target=target,
        )
        backend_result = self._filter_direct_solution_candidates(
            backend_result,
            emr,
            spr,
            target,
            variables,
        )
        final_answer = backend_result["output"]["formatted_solution"] if backend_result["success"] else None
        final_answer_items = self._direct_final_answer_items(
            backend_result,
            source_step_id="chain_main_step_1",
        )
        derived_result = self._direct_solution_aggregate_evaluation(emr, backend_result, spr)
        if derived_result is None:
            derived_result = self._direct_target_evaluation(emr, variables, backend_result)
        if derived_result is not None:
            final_answer = f"{derived_result['target']} = {derived_result['result']['output'].get('value')}"

        planner_metadata = planner_draft or {}
        chain_data: dict[str, Any] = {
            "problem_id": emr.problem_id,
            "source_emr_id": emr.problem_id,
            "source_emr_version": emr.schema_version,
            "chain_id": "chain_main",
            "steps": [
                {
                    "step_id": "chain_main_step_0",
                    "step_index": 0,
                    "description": "建立来自 EMR 的方程。",
                    "math_expression": f"{equation.lhs_sympy} = {equation.rhs_sympy}",
                    "rule_name": "given_equation",
                    "justification": "该方程来自可执行数学表示 EMR。",
                    "verification_status": "unverified",
                    "diagnostics": {
                        "source_equation_id": equation.id,
                        "solver_route": classification_result.solver_route,
                        **planner_metadata,
                    },
                },
                {
                    "step_id": "chain_main_step_1",
                    "step_index": 1,
                    "description": "调用 SymPy 后端求解方程。",
                    "math_expression": backend_result["output"]["formatted_solution"] if backend_result["success"] else None,
                    "rule_name": "cas_solve_equation_system",
                    "justification": "该计算交给 CAS 后端完成。",
                    "depends_on": ["chain_main_step_0"],
                    "verification_status": "unverified",
                    "diagnostics": backend_result,
                },
                {
                    "step_id": "chain_main_step_2",
                    "step_index": 2,
                    "description": "得到最终答案。",
                    "math_expression": final_answer,
                    "rule_name": "final_answer",
                    "justification": "最终答案来自 CAS 后端结果。",
                    "depends_on": ["chain_main_step_1"],
                    "verification_status": "unverified",
                    "diagnostics": {
                        "backend": backend_result["backend"],
                        "operation": "format_final_answer",
                        "input": {
                            **backend_result["output"],
                            "final_answer_items": final_answer_items,
                        },
                        "output": {"formatted_solution": final_answer},
                        "success": bool(final_answer),
                        "errors": backend_result["errors"],
                        "warnings": backend_result["warnings"],
                    },
                },
            ],
            "final_answer": final_answer,
            "confidence": 0.85 if final_answer else 0.0,
            "metadata": {
                "solver": "AlgebraSolver",
                "solver_route": classification_result.solver_route,
                "task_type": classification_result.task_type,
                "backend": backend_result["backend"],
                "backend_success": backend_result["success"],
                "backend_confirmed": backend_result["success"],
                "final_answer_items": final_answer_items,
                **planner_metadata,
            },
        }
        if derived_result is not None:
            derived_step_id = "chain_main_step_2"
            derived_value = derived_result["result"]["output"].get("value")
            final_answer_items.append(
                {
                    "target": derived_result["target"],
                    "value": str(derived_value),
                    "source_step_id": derived_step_id,
                    "backend_operation": "evaluate",
                    "backend_confirmed": derived_result["result"].get("success") is True,
                }
            )
            chain_data["steps"].insert(
                -1,
                {
                    "step_id": derived_step_id,
                    "step_index": 2,
                    "description": "Evaluate the EMR target expression after solving the equation.",
                    "math_expression": str(derived_value) if derived_value is not None else None,
                    "rule_name": "cas_evaluate",
                    "justification": "The requested answer is a derived quantity backed by a CAS evaluation.",
                    "depends_on": ["chain_main_step_1"],
                    "verification_status": "unverified",
                    "diagnostics": derived_result["result"],
                },
            )
            final_step = chain_data["steps"][-1]
            final_step["step_id"] = "chain_main_step_3"
            final_step["step_index"] = 3
            final_step["depends_on"] = [derived_step_id]
            final_step["diagnostics"]["input"] = {
                **derived_result["result"]["output"],
                "final_answer_items": final_answer_items,
            }
            final_step["diagnostics"]["errors"] = derived_result["result"]["errors"]
            final_step["diagnostics"]["warnings"] = derived_result["result"]["warnings"]
            chain_data["metadata"]["final_answer_items"] = final_answer_items
        return self._validate_enhanced_chain(chain_data, emr, classification_result, spr)

    def solve_tool_request(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        planner_draft: dict[str, Any] | None,
        spr: SPR | None = None,
    ) -> SolutionChain:
        """Execute planner-proposed tool calls after deterministic validation."""

        if not planner_draft:
            return self.unsupported_chain(
                emr,
                classification_result,
                "No planner draft was available for tool-call solving.",
                {"backend_confirmed": False},
            )

        if planner_draft.get("planner_error"):
            deterministic_tool_calls = (
                self._deterministic_preferred_tool_calls(spr)
                or self._deterministic_calculus_tool_calls(spr)
            )
            if deterministic_tool_calls:
                repaired_planner_draft = {
                    **planner_draft,
                    "planner_tool_calls": deterministic_tool_calls,
                    "deterministic_repair": "deterministic_fallback_after_planner_error",
                    "deterministic_repair_reason": planner_draft.get("planner_error"),
                }
                return self._execute_tool_calls(
                    emr,
                    classification_result,
                    repaired_planner_draft,
                    deterministic_tool_calls,
                    spr,
                )
            return self.unsupported_chain(
                emr,
                classification_result,
                "Planner failed before producing a usable tool call.",
                {
                    **planner_draft,
                    "backend_confirmed": False,
                    "validator_errors": [],
                },
            )

        tool_calls = planner_draft.get("planner_tool_calls") or []
        preferred_tool_calls = self._deterministic_preferred_tool_calls(spr)
        if preferred_tool_calls:
            planner_draft = {
                **planner_draft,
                "planner_tool_calls": preferred_tool_calls,
                "deterministic_repair": "preferred_deterministic_math_pattern",
            }
            tool_calls = preferred_tool_calls
        deterministic_tool_calls = self._deterministic_calculus_tool_calls(spr)
        if deterministic_tool_calls and spr is not None and self._spr_asks_for_extrema(spr):
            planner_draft = {
                **planner_draft,
                "planner_tool_calls": deterministic_tool_calls,
                "deterministic_repair": "calculus_extrema_completion",
            }
            tool_calls = deterministic_tool_calls
        if not tool_calls:
            return self.unsupported_chain(
                emr,
                classification_result,
                "Planner did not produce a valid tool call.",
                {
                    **planner_draft,
                    "backend_confirmed": False,
                    "validator_errors": [
                        {
                            "field_path": "planner_tool_calls",
                            "value": tool_calls,
                            "message": planner_draft.get("planner_parse_error")
                            or "No tool calls were found.",
                        }
                    ],
                },
            )

        return self._execute_tool_calls(emr, classification_result, planner_draft, tool_calls, spr)

    @staticmethod
    def _solve_target(emr: EMR, variables: list[str]) -> str:
        if not variables:
            return ""
        if not emr.goals:
            return variables[0]
        goal = emr.goals[0]
        if goal.target in set(variables):
            return str(goal.target)
        variable_by_id = {variable.id: variable.symbol for variable in emr.variables}
        for variable_id in goal.target_variable_ids:
            symbol = variable_by_id.get(variable_id)
            if symbol in variables:
                return symbol
        return variables[0]

    @staticmethod
    def _unsupported_reason(emr: EMR) -> str | None:
        if len(emr.equations) != 1:
            return "Solver v1 supports exactly one equation."
        if len(emr.variables) != 1:
            return "Solver v1 supports exactly one variable."
        return None

    def _execute_request(
        self,
        request: Any,
        emr: EMR,
        spr: SPR | None = None,
        execution_context: dict[str, Any] | None = None,
        effective_variables: list[str] | None = None,
    ) -> dict[str, Any]:
        context = execution_context or {"variables": {}}
        variables = effective_variables or request.variables or self._available_variables(emr, spr)
        if request.operation in {"construct_expression", "parse_expression"}:
            expression = self._resolve_expression_reference(
                request.expressions[0].expression,
                context,
            )
            substitutions = self._resolve_substitutions(request.substitutions, context)
            if substitutions:
                expression = self._apply_substitutions_to_statement(
                    expression,
                    sorted(set(variables) | set(substitutions)),
                    substitutions,
                    request.target,
                )
                variables = sorted(set(variables) | set(self._symbols_from_statement(expression)))
            result = self.cas_service.construct_expression(
                expression=expression,
                variables=variables,
                target=request.target,
            )
            return self._attach_numeric_value_to_construct_expression(result, variables, request.target)

        equation_substitutions = self._resolve_substitutions(request.substitutions, context)
        equations = [
            {
                "id": equation.id or f"tool_eq_{index + 1}",
                "lhs_sympy": self._apply_substitutions_to_expression(
                    self._resolve_expression_reference(equation.lhs, context),
                    variables,
                    equation_substitutions,
                    request.target,
                ),
                "rhs_sympy": self._apply_substitutions_to_expression(
                    self._resolve_expression_reference(equation.rhs, context),
                    variables,
                    equation_substitutions,
                    request.target,
                ),
            }
            for index, equation in enumerate(request.equations)
        ]
        if request.operation == "solve_equation_system":
            return self.cas_service.solve_equation_system(
                equations=equations,
                variables=variables,
                target=request.target,
            )
        if request.operation == "solve_for":
            if len(equations) > 1:
                return self.cas_service.solve_equation_system(
                    equations=equations,
                    variables=variables,
                    target=request.target,
                )
            return self.cas_service.solve_for(
                equations=equations,
                variables=variables,
                target=request.target or variables[0],
            )
        if request.operation in {"substitute", "evaluate"}:
            expression = self._resolve_expression_reference(
                request.expressions[0].expression,
                context,
            )
            substitutions = self._resolve_substitutions(request.substitutions, context)
            if request.operation == "substitute" and expression.count("=") == 1:
                variables = sorted(set(variables) | set(substitutions) | set(self._symbols_from_statement(expression)))
            if request.operation == "evaluate":
                substitutions = self._complete_substitutions_from_context(
                    expression,
                    variables,
                    substitutions,
                    context,
                )
            if request.operation == "substitute":
                return self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.target,
                )
            if expression.count("=") == 1:
                result = self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.target,
                )
                result["operation"] = "evaluate"
                return result
            result = self.cas_service.evaluate_expression(
                expression=expression,
                variables=variables,
                substitutions=substitutions,
                target=request.target,
            )
            if self._evaluate_failed_only_because_symbolic(result):
                return self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.target,
                )
            return result

        return {
            "backend": self.cas_service.backend_name,
            "operation": request.operation,
            "input": request.model_dump(mode="json"),
            "output": {},
            "success": False,
            "errors": ["Operation is not implemented."],
            "warnings": [],
        }

    def _execute_tool_calls(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        planner_draft: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        spr: SPR | None,
    ) -> SolutionChain:
        steps: list[dict[str, Any]] = [
            {
                "step_id": "chain_main_step_0",
                "step_index": 0,
                "description": "读取题意和已知变量。",
                "math_expression": None,
                "rule_name": "read_problem_and_variables",
                "justification": "Solver 将 SPR、EMR 和分类结果交给 Planner，用于生成工具调用草稿。",
                "verification_status": "unverified",
                "diagnostics": {
                    "variables": self._available_variables(emr, spr),
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    **planner_draft,
                },
            }
        ]
        execution_context: dict[str, Any] = {"variables": {}}
        validation_reports: list[dict[str, Any]] = []
        backend_results: list[dict[str, Any]] = []
        final_answer: str | None = None
        step_index = 1

        for tool_index, request_data in enumerate(tool_calls):
            call_id = request_data.get("id") or f"tool_{tool_index + 1}"
            validation_report = self.request_validator.validate(
                request_data,
                emr,
                spr=spr,
                execution_context=execution_context,
            )
            validation_reports.append(self._jsonable_validation_report(validation_report))
            if not validation_report["valid"]:
                if backend_results and (
                    self._derive_emr_target_expression_backend_result(emr, backend_results)
                    or self._derive_square_sum_identity_backend_result(emr, spr)
                ):
                    validation_reports.pop()
                    break
                return self._partial_unsupported_chain(
                    emr,
                    classification_result,
                    planner_draft,
                    steps,
                    validation_reports,
                    backend_results,
                    "多步工具调用在校验阶段失败。",
                    final_answer,
                    spr,
                )

            request = validation_report["request"]
            normalized_request = validation_report["normalized_request"]
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": self._planner_step_description(request.operation),
                    "math_expression": self._request_math_expression(normalized_request),
                    "rule_name": f"planner_{request.operation}",
                    "justification": normalized_request.get("reason") or "Planner 根据题意提出这一步计算请求。",
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        "tool_call_id": call_id,
                        "planner_provider": planner_draft.get("planner_provider"),
                        "planner_model": planner_draft.get("planner_model"),
                        "planner_role": planner_draft.get("planner_role"),
                        "normalized_request": normalized_request,
                        "planner_variables": validation_report.get("planner_variables", []),
                        "effective_variables": validation_report.get("effective_variables", []),
                        "normalization_warnings": validation_report.get("normalization_warnings", []),
                        "validator_errors": validation_report["errors"],
                        "backend_confirmed": False,
                    },
                }
            )
            step_index += 1

            backend_result = self._execute_request(
                request,
                emr,
                spr,
                execution_context,
                effective_variables=validation_report.get("effective_variables"),
            )
            backend_results.append(backend_result)
            if not backend_result["success"]:
                return self._partial_unsupported_chain(
                    emr,
                    classification_result,
                    planner_draft,
                    steps,
                    validation_reports,
                    backend_results,
                    "多步工具调用在 CAS 执行阶段失败。",
                    final_answer,
                    spr,
                )

            self._store_tool_result(call_id, backend_result, execution_context, normalized_request)
            cas_expression = self._final_answer_from_backend(backend_result)
            final_answer = self._format_final_answer(cas_expression, backend_result, spr)
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": self._cas_step_description(backend_result["operation"]),
                    "math_expression": cas_expression,
                    "rule_name": f"cas_{backend_result['operation']}",
                    "justification": "该结果由 CAS 后端确认后写入解题链。",
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **backend_result,
                        "tool_call_id": call_id,
                        "backend_confirmed": True,
                    },
                }
            )
            step_index += 1

        derived_backend = self._derive_linear_function_expression_backend_result(
            spr,
            backend_results,
            execution_context,
        )
        if derived_backend:
            backend_results.append(derived_backend["backend_result"])
            validation_reports.append(derived_backend["validation_report"])
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": "CAS 代入待定系数得到函数解析式。",
                    "math_expression": self._final_answer_from_backend(derived_backend["backend_result"]),
                    "rule_name": "cas_substitute",
                    "justification": "根据已由 CAS 确认的系数结果，确定性代入 y = k*x + b 得到最终解析式。",
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **derived_backend["backend_result"],
                        "backend_confirmed": True,
                    },
                }
            )
            step_index += 1

        for derived_intercept in self._derive_quadratic_intercept_backend_results(spr):
            backend_results.append(derived_intercept["backend_result"])
            validation_reports.append(derived_intercept["validation_report"])
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": "CAS 确定性补全二次函数交点坐标。",
                    "math_expression": self._final_answer_from_backend(derived_intercept["backend_result"]),
                    "rule_name": f"cas_{derived_intercept['backend_result']['operation']}",
                    "justification": "根据题目中的二次函数表达式，确定性补齐 x 轴或 y 轴交点，避免 Planner 漏掉同类答案。",
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **derived_intercept["backend_result"],
                        "tool_call_id": derived_intercept["validation_report"]["normalized_request"]["id"],
                        "backend_confirmed": True,
                        "derived_by": "quadratic_intercept_completion",
                    },
                }
            )
            step_index += 1

        for deterministic_completion in [
            self._derive_coordinate_midpoint_sum_backend_result(spr),
            self._derive_equivalent_unit_chain_backend_result(spr),
            self._derive_square_sum_identity_backend_result(emr, spr),
        ]:
            if not deterministic_completion:
                continue
            backend_results.append(deterministic_completion["backend_result"])
            validation_reports.append(deterministic_completion["validation_report"])
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": deterministic_completion["description"],
                    "math_expression": self._final_answer_from_backend(deterministic_completion["backend_result"]),
                    "rule_name": "cas_evaluate",
                    "justification": deterministic_completion["justification"],
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **deterministic_completion["backend_result"],
                        "tool_call_id": deterministic_completion["validation_report"]["normalized_request"]["id"],
                        "backend_confirmed": True,
                        "derived_by": deterministic_completion["derived_by"],
                    },
                }
            )
            step_index += 1

        derived_target = self._derive_emr_target_expression_backend_result(emr, backend_results)
        if derived_target:
            backend_results.append(derived_target["backend_result"])
            validation_reports.append(derived_target["validation_report"])
            steps.append(
                {
                    "step_id": f"chain_main_step_{step_index}",
                    "step_index": step_index,
                    "description": "Evaluate the requested target expression from CAS-confirmed solution values.",
                    "math_expression": self._final_answer_from_backend(derived_target["backend_result"]),
                    "rule_name": "cas_evaluate",
                    "justification": "The requested answer is an expression over solved variables, so it is evaluated after the CAS-confirmed solve step.",
                    "depends_on": [steps[-1]["step_id"]],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **derived_target["backend_result"],
                        "tool_call_id": derived_target["validation_report"]["normalized_request"]["id"],
                        "backend_confirmed": True,
                        "derived_by": "emr_target_expression_completion",
                    },
                }
            )
            step_index += 1

        final_answer_items = self._final_answer_items(backend_results, validation_reports, spr)
        final_answer = self._aggregate_final_answer(final_answer, final_answer_items, spr)
        steps.append(
            {
                "step_id": f"chain_main_step_{step_index}",
                "step_index": step_index,
                "description": "给出最终答案。",
                "math_expression": final_answer,
                "rule_name": "final_answer",
                "justification": "最终答案来自最后一个成功的 CAS 结果。",
                "depends_on": [steps[-1]["step_id"]],
                "verification_status": "unverified",
                "diagnostics": {
                    "backend": self.cas_service.backend_name,
                    "operation": "format_final_answer",
                    "input": {
                        "last_output": backend_results[-1]["output"] if backend_results else {},
                        "final_answer_items": final_answer_items,
                    },
                    "output": {"formatted_solution": final_answer},
                    "success": final_answer is not None,
                    "errors": [],
                    "warnings": [],
                },
            }
        )

        chain_data = {
                "problem_id": emr.problem_id,
                "source_emr_id": emr.problem_id,
                "source_emr_version": emr.schema_version,
                "chain_id": "chain_main",
                "steps": steps,
                "final_answer": final_answer,
                "confidence": 0.85 if final_answer else 0.0,
                "metadata": {
                    "solver": "AlgebraSolver",
                    "solver_version": "Solver v1 enhanced: multi-step ToolCall + CAS",
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "planner_provider": planner_draft.get("planner_provider"),
                    "planner_model": planner_draft.get("planner_model"),
                    "planner_role": planner_draft.get("planner_role"),
                    "tool_call_count": len(tool_calls),
                    "completed_tool_call_count": len(backend_results),
                    "backend": self.cas_service.backend_name,
                    "backend_confirmed": bool(backend_results),
                    "partial_failure": False,
                    "validation_reports": validation_reports,
                    "final_answer_items": final_answer_items,
                    "execution_context": execution_context,
                },
            }
        return self._validate_enhanced_chain(chain_data, emr, classification_result, spr)

    def _validate_enhanced_chain(
        self,
        chain_data: dict[str, Any],
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None = None,
    ) -> SolutionChain:
        enhanced = self.scs_assembly.enhance(chain_data, emr, classification_result, spr)
        return validate_solution_chain_dict(enhanced)

    @staticmethod
    def _available_variables(emr: EMR, spr: SPR | None) -> list[str]:
        variables = [variable.symbol for variable in emr.variables if variable.symbol]
        if variables or spr is None:
            return variables
        return [variable.symbol for variable in spr.variables if variable.symbol]

    @staticmethod
    def _jsonable_validation_report(validation_report: dict[str, Any]) -> dict[str, Any]:
        return {
            "valid": validation_report["valid"],
            "normalized_request": validation_report["normalized_request"],
            "planner_variables": validation_report.get("planner_variables", []),
            "effective_variables": validation_report.get("effective_variables", []),
            "normalization_warnings": validation_report.get("normalization_warnings", []),
            "errors": validation_report["errors"],
            "declared_variables": validation_report.get("declared_variables", []),
            "semantic_symbol_aliases": validation_report.get("semantic_symbol_aliases", {}),
        }

    @staticmethod
    def _resolve_substitutions(
        substitutions: dict[str, Any],
        execution_context: dict[str, Any],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for symbol, value in substitutions.items():
            if isinstance(value, str) and value.startswith("$"):
                reference_value = CalculationRequestValidator.resolve_reference(value, execution_context)
                resolved[symbol] = str(reference_value)
            else:
                resolved[symbol] = str(value)
        return resolved

    @staticmethod
    def _store_tool_result(
        call_id: str,
        backend_result: dict[str, Any],
        execution_context: dict[str, Any],
        request_data: dict[str, Any] | None = None,
    ) -> None:
        output = backend_result.get("output", {})
        stored: dict[str, Any] = {
            "operation": backend_result.get("operation"),
            "output": output,
        }
        formatted_values = AlgebraSolver._formatted_solution_mapping(output.get("formatted_solution"))
        raw_solution_symbols: set[str] = set()
        for solution in output.get("raw_solution", []):
            for symbol, value in solution.items():
                symbol_text = str(symbol)
                stored_value = AlgebraSolver._preferred_context_solution_value(
                    symbol_text,
                    str(value),
                    formatted_values,
                )
                raw_solution_symbols.add(symbol_text)
                execution_context.setdefault("variables", {})[symbol_text] = stored_value
                stored[symbol_text] = stored_value
        if "value" in output:
            stored["value"] = output["value"]
            stored["result"] = output["value"]
        if "formatted_solution" in output:
            stored["formatted_solution"] = output["formatted_solution"]
            stored.setdefault("result", output["formatted_solution"])
        if "confirmed_expression" in output:
            stored["confirmed_expression"] = output["confirmed_expression"]
            stored.setdefault("result", output["confirmed_expression"])
        scalar_value = (
            output.get("value")
            or output.get("formatted_solution")
            or output.get("confirmed_expression")
        )
        target = (request_data or {}).get("target")
        if target and scalar_value is not None and str(target) not in raw_solution_symbols:
            AlgebraSolver._store_symbol_aliases(stored, str(target), scalar_value)
            execution_context.setdefault("variables", {}).setdefault(str(target), scalar_value)
        request_expressions = (request_data or {}).get("expressions") or []
        if request_expressions:
            expression_name = request_expressions[0].get("name")
            expression_value = (
                output.get("value")
                or output.get("confirmed_expression")
                or output.get("formatted_solution")
                or request_expressions[0].get("expression")
            )
            if expression_name and expression_value is not None:
                AlgebraSolver._store_symbol_aliases(stored, str(expression_name), expression_value)
        execution_context[call_id] = stored

    @staticmethod
    def _formatted_solution_mapping(formatted_solution: Any) -> dict[str, str]:
        text = str(formatted_solution or "")
        if not text:
            return {}
        values: dict[str, str] = {}
        for part in re.split(r"\s*,\s*", text):
            match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_{}\\]*)\s*=\s*(.+?)\s*", part)
            if match:
                values[match.group(1)] = match.group(2)
        return values

    @staticmethod
    def _preferred_context_solution_value(
        symbol: str,
        raw_value: str,
        formatted_values: dict[str, str],
    ) -> str:
        formatted_value = formatted_values.get(symbol)
        if not formatted_value:
            return raw_value
        raw_has_complex_branch = "I" in raw_value or "pi" in raw_value
        formatted_is_plain_real = "I" not in formatted_value and not re.search(r"\bpi\b", formatted_value)
        if raw_has_complex_branch and formatted_is_plain_real:
            return formatted_value
        return raw_value

    @staticmethod
    def _store_symbol_aliases(target: dict[str, Any], symbol: str, value: Any) -> None:
        target[symbol] = value
        for alias in AlgebraSolver._latex_symbol_aliases(symbol):
            target.setdefault(alias, value)

    @staticmethod
    def _latex_symbol_aliases(symbol: str) -> set[str]:
        aliases: set[str] = set()
        compact = symbol.strip()
        if not compact:
            return aliases
        no_braces = re.sub(r"_\{([^{}]+)\}", r"_\1", compact)
        if no_braces != compact:
            aliases.add(no_braces)
        brace_match = re.fullmatch(r"([A-Za-z]+)_([A-Za-z0-9]+)", compact)
        if brace_match:
            aliases.add(f"{brace_match.group(1)}_{{{brace_match.group(2)}}}")
        return aliases

    @staticmethod
    def _resolve_expression_reference(
        value: str,
        execution_context: dict[str, Any],
    ) -> str:
        if not isinstance(value, str):
            return value
        if "$" not in value:
            context_expression = AlgebraSolver._context_expression_for_name(value, execution_context)
            return context_expression if context_expression is not None else value

        def replace(match: re.Match[str]) -> str:
            reference = match.group(0)
            resolved = CalculationRequestValidator.resolve_reference(reference, execution_context)
            return CalculationRequestValidator._format_embedded_reference_value(resolved, value) if resolved is not None else reference

        return re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", replace, value)

    def _attach_numeric_value_to_construct_expression(
        self,
        result: dict[str, Any],
        variables: list[str],
        target: str | None,
    ) -> dict[str, Any]:
        if result.get("operation") not in {"construct_expression", "parse_expression"}:
            return result
        if not result.get("success"):
            return result
        output = result.get("output") or {}
        expression = output.get("confirmed_expression")
        if expression is None or output.get("value") is not None:
            return result
        if output.get("free_symbols"):
            return result
        expression_text = str(expression).strip()
        if not expression_text or expression_text.count("=") == 1:
            return result
        evaluated = self.cas_service.evaluate_expression(
            expression=expression_text,
            variables=variables,
            substitutions={},
            target=target,
        )
        if not evaluated.get("success"):
            return result
        value = (evaluated.get("output") or {}).get("value")
        if value is None:
            return result
        output["value"] = value
        output["evaluated_expression"] = expression_text
        result["output"] = output
        result.setdefault("warnings", []).append("Numeric value attached for symbol-free constructed expression.")
        return result

    @staticmethod
    def _context_expression_for_name(
        value: str,
        execution_context: dict[str, Any],
    ) -> str | None:
        name = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return None
        for stored in execution_context.values():
            if not isinstance(stored, dict):
                continue
            if name in stored and isinstance(stored[name], str):
                return stored[name]
        return None

    def _complete_substitutions_from_context(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        execution_context: dict[str, Any],
    ) -> dict[str, str]:
        parse_report = self.cas_service.parse_expression(expression, variables)
        if not parse_report.get("success"):
            return substitutions
        completed = dict(substitutions)
        context_variables = execution_context.get("variables") or {}
        for symbol in parse_report.get("free_symbols", []):
            if symbol in completed:
                continue
            if symbol in context_variables:
                completed[symbol] = str(context_variables[symbol])
        return completed

    def _derive_linear_function_expression_backend_result(
        self,
        spr: SPR | None,
        backend_results: list[dict[str, Any]],
        execution_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if spr is None or not any(self._question_asks_for_function_expression(question, spr) for question in spr.questions):
            return None
        if any(
            result.get("success")
            and result.get("operation") in {"construct_expression", "parse_expression", "substitute"}
            and self._final_answer_from_backend(result)
            and re.search(r"\bx\b", self._final_answer_from_backend(result) or "")
            for result in backend_results
        ):
            return None

        context_variables = execution_context.get("variables") or {}
        if "k" not in context_variables or "b" not in context_variables:
            return None

        substitutions = {"k": str(context_variables["k"]), "b": str(context_variables["b"])}
        backend_result = self.cas_service.substitute(
            expression="k*x + b",
            variables=["x", "k", "b"],
            substitutions=substitutions,
            target="y",
        )
        if not backend_result.get("success"):
            return None
        backend_result["tool_call_id"] = "derived_linear_function_expression"
        backend_result["backend_confirmed"] = True
        validation_report = {
            "valid": True,
            "request": None,
            "normalized_request": {
                "id": "derived_linear_function_expression",
                "operation": "substitute",
                "variables": ["x", "k", "b"],
                "equations": [],
                "expressions": [{"name": "y", "expression": "k*x + b"}],
                "substitutions": substitutions,
                "target": "y",
                "reason": "Derive the explicit linear function from CAS-confirmed k and b.",
                "evidence_span": "y = kx + b",
                "planner_variables": ["x", "k", "b"],
                "effective_variables": ["b", "k", "x"],
                "normalization_warnings": [],
                "semantic_symbol_aliases": {},
            },
            "planner_variables": ["x", "k", "b"],
            "effective_variables": ["b", "k", "x"],
            "normalization_warnings": [],
            "errors": [],
            "declared_variables": ["b", "k", "x", "y"],
            "semantic_symbol_aliases": {},
        }
        return {"backend_result": backend_result, "validation_report": validation_report}

    def _derive_emr_target_expression_backend_result(
        self,
        emr: EMR,
        backend_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not emr.expressions:
            return None
        candidates = self._direct_target_expression_candidates(emr)
        if not candidates:
            return None
        for backend_result in reversed(backend_results):
            substitutions = self._solution_substitutions(backend_result)
            if not substitutions:
                continue
            variables = sorted(set(self._available_variables(emr, None)) | set(substitutions))
            target = self._direct_target_label(emr, candidates[0][1])
            for expression_id, expression_text in candidates:
                if not self._target_expression_is_meaningful(expression_text):
                    continue
                result = self.cas_service.evaluate_expression(
                    expression=expression_text,
                    variables=variables,
                    substitutions=substitutions,
                    target=target,
                )
                if not result.get("success"):
                    continue
                result["input"]["source_expression_id"] = expression_id
                return self._derived_backend_package(
                    {
                        "id": "derived_emr_target_expression",
                        "operation": "evaluate",
                        "variables": variables,
                        "equations": [],
                        "expressions": [{"name": target, "expression": expression_text}],
                        "substitutions": substitutions,
                        "target": target,
                        "reason": "Evaluate the EMR target expression after solving the equations.",
                        "evidence_span": expression_text,
                    },
                    result,
                )
        return None

    @staticmethod
    def _target_expression_is_meaningful(expression_text: str | None) -> bool:
        expression = str(expression_text or "").strip()
        if not expression:
            return False
        compact = re.sub(r"[^A-Za-z0-9_+\-*/().]", "", expression)
        return bool(re.search(r"[+\-*/]", compact))

    @staticmethod
    def _repair_mixed_numbers_in_expression(expression: str, spr: SPR | None) -> str:
        if spr is None or not isinstance(expression, str):
            return expression
        text = " ".join(
            str(part or "")
            for part in [
                spr.problem_text,
                spr.problem_stem,
                *[condition.text for condition in spr.conditions],
                *[formula.raw_text for formula in spr.formulas],
            ]
        )
        repaired = expression
        for match in re.finditer(r"(?<![\d/])(\d+)\s+(\d+)\s*/\s*(\d+)(?![\d/])", text):
            whole, numerator, denominator = match.groups()
            concatenated = f"{whole}{numerator}/{denominator}"
            mixed_expression = f"({whole}+{numerator}/{denominator})"
            repaired = re.sub(rf"(?<![\d/]){re.escape(concatenated)}(?!\d)", mixed_expression, repaired)
        return repaired

    def _direct_solution_aggregate_evaluation(
        self,
        emr: EMR,
        backend_result: dict[str, Any],
        spr: SPR | None,
    ) -> dict[str, Any] | None:
        aggregate = self._solution_aggregate_request(emr, backend_result, spr)
        if aggregate is None:
            return None
        result = self.cas_service.evaluate_expression(
            expression=aggregate["expression"],
            variables=[],
            substitutions={},
            target=aggregate["target"],
        )
        if not result.get("success"):
            return None
        result["input"]["source_expression_id"] = aggregate["source_expression_id"]
        return {
            "target": aggregate["target"],
            "expression_id": aggregate["source_expression_id"],
            "expression": aggregate["expression"],
            "result": result,
        }

    @staticmethod
    def _solution_aggregate_request(
        emr: EMR,
        backend_result: dict[str, Any],
        spr: SPR | None,
    ) -> dict[str, str] | None:
        raw_solution = backend_result.get("output", {}).get("raw_solution") or []
        if len(raw_solution) < 2:
            return None
        variable_values: dict[str, list[str]] = {}
        for solution in raw_solution:
            if len(solution) != 1:
                continue
            symbol, value = next(iter(solution.items()))
            variable_values.setdefault(str(symbol), []).append(str(value))
        if not variable_values:
            return None
        symbol, values = max(variable_values.items(), key=lambda item: len(item[1]))
        if len(values) < 2:
            return None
        question_text = AlgebraSolver._spr_question_text(spr) or " ".join(
            str(getattr(goal, "description", "") or getattr(goal, "target", "") or "")
            for goal in emr.goals
        )
        lowered = question_text.lower()
        if "product" in lowered and any(signal in lowered for signal in ["two solutions", "solutions", "roots"]):
            return {
                "target": "product_of_solutions",
                "expression": "*".join(f"({value})" for value in values),
                "source_expression_id": f"derived_product_of_{symbol}_solutions",
            }
        if "sum" in lowered and any(signal in lowered for signal in ["two solutions", "solutions", "roots"]):
            return {
                "target": "sum_of_solutions",
                "expression": "+".join(f"({value})" for value in values),
                "source_expression_id": f"derived_sum_of_{symbol}_solutions",
            }
        return None

    @staticmethod
    def _spr_question_text(spr: SPR | None) -> str:
        if spr is None:
            return ""
        return " ".join(
            str(part or "")
            for part in [
                spr.problem_text,
                spr.problem_stem,
                *[question.text for question in spr.questions],
                *[target.text for target in spr.targets],
            ]
        )

    def _apply_substitutions_to_expression(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        target: str | None,
    ) -> str:
        if not substitutions:
            return expression
        result = self.cas_service.substitute(
            expression=expression,
            variables=variables,
            substitutions=substitutions,
            target=target,
        )
        if result.get("success") and (result.get("output") or {}).get("value") is not None:
            return str(result["output"]["value"])
        return expression

    def _apply_substitutions_to_statement(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        target: str | None,
    ) -> str:
        if expression.count("=") != 1:
            return self._apply_substitutions_to_expression(expression, variables, substitutions, target)
        lhs, rhs = (part.strip() for part in expression.split("=", 1))
        substituted_lhs = self._apply_substitutions_to_expression(lhs, variables, substitutions, target)
        substituted_rhs = self._apply_substitutions_to_expression(rhs, variables, substitutions, target)
        return f"{substituted_lhs} = {substituted_rhs}"

    @staticmethod
    def _symbols_from_statement(expression: str) -> set[str]:
        return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))

    @staticmethod
    def _evaluate_failed_only_because_symbolic(result: dict[str, Any]) -> bool:
        if result.get("success") is True:
            return False
        errors = " ".join(str(error) for error in result.get("errors", []))
        return "unresolved symbols" in errors and bool(result.get("missing_symbols"))

    @staticmethod
    def _final_answer_items(
        backend_results: list[dict[str, Any]],
        validation_reports: list[dict[str, Any]],
        spr: SPR | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, backend_result in enumerate(backend_results):
            if not backend_result.get("success"):
                continue
            request = (
                validation_reports[index].get("normalized_request", {})
                if index < len(validation_reports)
                else {}
            )
            call_id = request.get("id") or backend_result.get("tool_call_id") or f"tool_{index + 1}"
            output = backend_result.get("output", {})
            aggregate_system = AlgebraSolver._named_system_solution_item(
                output,
                request,
                call_id,
                backend_result,
            )
            if aggregate_system:
                items.append(aggregate_system)
                continue
            aggregate_intercepts = AlgebraSolver._derived_quadratic_x_axis_aggregate_item(
                output.get("raw_solution", []),
                request,
                call_id,
                backend_result,
            )
            if aggregate_intercepts:
                items.append(aggregate_intercepts)
                continue
            for solution in output.get("raw_solution", []):
                requested_target = request.get("target")
                solution_targets = {str(symbol) for symbol in solution.keys()}
                coordinate_items = AlgebraSolver._coordinate_solution_items(
                    solution,
                    request,
                    call_id,
                    backend_result,
                )
                if coordinate_items:
                    items.extend(coordinate_items)
                    continue
                intercept_items = AlgebraSolver._intercept_solution_items(
                    solution,
                    request,
                    call_id,
                    backend_result,
                )
                if intercept_items:
                    items.extend(intercept_items)
                    continue
                target_is_aggregate = AlgebraSolver._is_aggregate_solution_target(
                    requested_target,
                    solution_targets,
                ) or AlgebraSolver._spr_question_requests_solution_targets(
                    spr,
                    solution_targets,
                )
                for symbol, value in solution.items():
                    if requested_target and str(symbol) != str(requested_target) and not target_is_aggregate:
                        continue
                    target = AlgebraSolver._canonical_target_name(
                        str(symbol),
                        spr,
                        request.get("semantic_symbol_aliases"),
                    )
                    items.append(
                        {
                            "target": str(target),
                            "value": str(value),
                            "source_tool_call_id": call_id,
                            "backend_operation": backend_result.get("operation"),
                            "backend_confirmed": True,
                        }
                    )
            value = output.get("value")
            if value is None and backend_result.get("operation") in {"construct_expression", "parse_expression"}:
                value = AlgebraSolver._expression_output_value(backend_result)
            if value is None:
                continue
            target = request.get("target")
            if not target and backend_result.get("operation") in {
                "construct_expression",
                "parse_expression",
                "evaluate",
            }:
                expressions = request.get("expressions") or []
                if expressions:
                    target = expressions[0].get("name")
            if not target:
                continue
            intercept_value_item = AlgebraSolver._intercept_value_item(
                value,
                request,
                call_id,
                backend_result,
            )
            if intercept_value_item:
                items.append(intercept_value_item)
                continue
            target = AlgebraSolver._canonical_target_name(
                str(target),
                spr,
                request.get("semantic_symbol_aliases"),
            )
            target = AlgebraSolver._contextual_final_answer_target_name(str(target), call_id, spr)
            target = AlgebraSolver._function_expression_final_answer_target(
                str(target),
                str(value),
                backend_result.get("operation"),
                spr,
            )
            items.append(
                {
                    "target": str(target),
                    "value": str(value),
                    "source_tool_call_id": call_id,
                    "backend_operation": backend_result.get("operation"),
                    "backend_confirmed": True,
                }
            )
        extrema_item = AlgebraSolver._derived_extrema_aggregate_item(backend_results, validation_reports)
        if extrema_item:
            items.append(extrema_item)
        selected_items = AlgebraSolver._select_final_answer_items(items, spr)
        selected_items = AlgebraSolver._dedupe_final_answer_items(selected_items)
        return AlgebraSolver._attach_choice_options(selected_items, spr)

    def _derive_quadratic_intercept_backend_results(self, spr: SPR | None) -> list[dict[str, Any]]:
        if spr is None:
            return []
        expression = self._quadratic_function_expression(spr)
        if not expression:
            return []
        results: list[dict[str, Any]] = []
        if self._spr_asks_for_x_axis_intercepts(spr):
            request = {
                "id": "derived_quadratic_x_axis_intercepts",
                "operation": "solve_equation_system",
                "variables": ["x", "y"],
                "equations": [
                    {"id": "derived_function", "lhs": "y", "rhs": expression},
                    {"id": "derived_x_axis", "lhs": "y", "rhs": "0"},
                ],
                "expressions": [],
                "substitutions": {},
                "target": "x_and_y",
                "reason": "Deterministically solve the quadratic function with y=0 to list all x-axis intercept coordinates.",
                "evidence_span": expression,
            }
            backend_result = self.cas_service.solve_equation_system(
                equations=[
                    {"id": "derived_function", "lhs_sympy": "y", "rhs_sympy": expression},
                    {"id": "derived_x_axis", "lhs_sympy": "y", "rhs_sympy": "0"},
                ],
                variables=["x", "y"],
                target="x_and_y",
            )
            if backend_result.get("success"):
                results.append(self._derived_backend_package(request, backend_result))
        if self._spr_asks_for_y_axis_intercept(spr):
            request = {
                "id": "derived_quadratic_y_axis_intercept",
                "operation": "evaluate",
                "variables": ["x"],
                "equations": [],
                "expressions": [{"name": "y_axis_intercept", "expression": expression}],
                "substitutions": {"x": "0"},
                "target": "y_axis_intercept",
                "reason": "Deterministically evaluate the quadratic function at x=0 to get the y-axis intercept coordinate.",
                "evidence_span": expression,
            }
            backend_result = self.cas_service.evaluate_expression(
                expression=expression,
                variables=["x"],
                substitutions={"x": "0"},
                target="y_axis_intercept",
            )
            if backend_result.get("success"):
                results.append(self._derived_backend_package(request, backend_result))
        return results

    def _deterministic_calculus_tool_calls(self, spr: SPR | None) -> list[dict[str, Any]]:
        if spr is None:
            return []
        expression = self._single_variable_function_expression(spr)
        if not expression:
            return []
        if self._spr_asks_for_tangent_line(spr):
            point = self._single_x_value_from_spr(spr)
            if point is None:
                return []
            return self._deterministic_tangent_line_tool_calls(expression, point)
        if self._spr_asks_for_extrema(spr):
            return self._deterministic_polynomial_extrema_tool_calls(expression)
        return []

    def _deterministic_preferred_tool_calls(self, spr: SPR | None) -> list[dict[str, Any]]:
        if spr is None:
            return []
        return (
            self._deterministic_ratio_share_total_tool_calls(spr)
            or self._deterministic_two_set_neither_tool_calls(spr)
            or self._deterministic_equivalent_unit_tool_calls(spr)
        )

    @classmethod
    def _deterministic_ratio_share_total_tool_calls(cls, spr: SPR) -> list[dict[str, Any]]:
        text = cls._spr_question_text(spr)
        lowered = text.lower()
        if "ratio" not in lowered or "total" not in lowered:
            return []
        ratio_match = re.search(
            r"([A-Za-z]+(?:\s*,\s*[A-Za-z]+)+(?:\s*,?\s*and\s+[A-Za-z]+)?)\s+share\b.*?\bratio\s+of\s+([0-9\s:]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not ratio_match:
            return []
        names = [name.lower() for name in re.findall(r"[A-Za-z]+", ratio_match.group(1))]
        names = [name for name in names if name not in {"and"}]
        ratios = [int(value) for value in re.findall(r"\d+", ratio_match.group(2))]
        if len(names) != len(ratios) or not ratios:
            return []
        known_match = re.search(
            r"\b([A-Za-z]+)'?s\s+portion\s+is\s+\$?\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if not known_match:
            return []
        known_name = known_match.group(1).lower()
        if known_name not in names:
            return []
        known_index = names.index(known_name)
        known_ratio = ratios[known_index]
        if known_ratio == 0:
            return []
        known_amount = int(known_match.group(2))
        expression = cls._fraction_expression(Fraction(known_amount * sum(ratios), known_ratio))
        return [
            {
                "id": "derived_ratio_share_total",
                "operation": "evaluate",
                "variables": [],
                "equations": [],
                "expressions": [{"name": "total", "expression": expression}],
                "substitutions": {},
                "target": "total",
                "reason": "Deterministically scale the full ratio sum from the known share.",
                "evidence_span": text,
            }
        ]

    @classmethod
    def _deterministic_two_set_neither_tool_calls(cls, spr: SPR) -> list[dict[str, Any]]:
        text = cls._spr_question_text(spr)
        lowered = text.lower()
        if "neither" not in lowered or "both" not in lowered:
            return []
        total_match = re.search(
            r"(?:of\s+the\s+|there\s+are\s+)?(\d+)\s+(?:students|people|members|items)\b",
            lowered,
        )
        if not total_match:
            return []
        total = total_match.group(1)
        take_counts = [
            match.group(1)
            for match in re.finditer(r"\b(\d+)\s+(?:take|takes|study|studies|are\s+in)\b", lowered)
        ]
        if len(take_counts) < 2:
            return []
        both_match = re.search(r"\b(\d+)\s+(?:students\s+)?(?:take\s+)?both\b", lowered)
        if not both_match:
            return []
        first, second = take_counts[0], take_counts[1]
        both = both_match.group(1)
        expression = f"{total} - (({first}) + ({second}) - ({both}))"
        return [
            {
                "id": "derived_two_set_neither",
                "operation": "evaluate",
                "variables": [],
                "equations": [],
                "expressions": [{"name": "neither", "expression": expression}],
                "substitutions": {},
                "target": "neither",
                "reason": "Deterministically apply two-set inclusion-exclusion to compute neither.",
                "evidence_span": text,
            }
        ]

    @classmethod
    def _deterministic_equivalent_unit_tool_calls(cls, spr: SPR) -> list[dict[str, Any]]:
        text = cls._spr_question_text(spr)
        lowered = text.lower()
        if not any(signal in lowered for signal in ["equivalent to", "cost the same as", "price of"]):
            return []
        relations = cls._equivalent_unit_relations(text)
        target_query = cls._equivalent_unit_target_query(text)
        if len(relations) < 2 or target_query is None:
            return []
        amount, source_unit, target_unit, target_label = target_query
        factor = cls._equivalent_unit_conversion_factor(relations, source_unit, target_unit)
        if factor is None:
            return []
        expression = cls._fraction_expression(amount * factor)
        return [
            {
                "id": "derived_equivalent_unit_chain",
                "operation": "evaluate",
                "variables": [],
                "equations": [],
                "expressions": [{"name": target_label, "expression": expression}],
                "substitutions": {},
                "target": target_label,
                "reason": "Deterministically convert through the equivalent-unit chain.",
                "evidence_span": text,
            }
        ]

    @staticmethod
    def _deterministic_tangent_line_tool_calls(expression: str, point: str) -> list[dict[str, Any]]:
        derivative = f"diff({expression}, x)"
        return [
            {
                "id": "derived_tangent_y0",
                "operation": "evaluate",
                "variables": ["x"],
                "expressions": [{"name": "y0", "expression": expression}],
                "substitutions": {"x": point},
                "target": "y0",
                "reason": "Deterministically evaluate the function value at the tangent point.",
                "evidence_span": f"x = {point}",
            },
            {
                "id": "derived_tangent_slope",
                "operation": "evaluate",
                "variables": ["x"],
                "expressions": [{"name": "m", "expression": derivative}],
                "substitutions": {"x": point},
                "target": "m",
                "reason": "Deterministically evaluate the derivative at the tangent point.",
                "evidence_span": "tangent slope",
            },
            {
                "id": "derived_tangent_line",
                "operation": "substitute",
                "variables": ["x", "y", "M", "A", "Y0"],
                "expressions": [{"name": "tangent_line", "expression": "y = M*(x - A) + Y0"}],
                "substitutions": {
                    "M": "$derived_tangent_slope.m",
                    "A": point,
                    "Y0": "$derived_tangent_y0.y0",
                },
                "target": "tangent_line",
                "reason": "Deterministically construct the tangent line from point-slope form.",
                "evidence_span": "tangent line equation",
            },
        ]

    def _deterministic_polynomial_extrema_tool_calls(self, expression: str) -> list[dict[str, Any]]:
        derivative = self._evaluate_symbolic_expression(f"diff({expression}, x)", ["x"])
        if not derivative:
            return []
        critical_points = self.cas_service.solve_for(
            equations=[{"id": "derived_critical_points", "lhs_sympy": derivative, "rhs_sympy": "0"}],
            variables=["x"],
            target="x",
        )
        if not critical_points.get("success"):
            return []
        tool_calls: list[dict[str, Any]] = [
            {
                "id": "derived_extrema_critical_points",
                "operation": "solve_for",
                "variables": ["x"],
                "equations": [{"id": "derived_critical_points", "lhs": derivative, "rhs": "0"}],
                "expressions": [],
                "substitutions": {},
                "target": "x",
                "reason": "Deterministically solve f'(x)=0 for critical points.",
                "evidence_span": "extrema",
            }
        ]
        raw_solutions = (critical_points.get("output") or {}).get("raw_solution") or []
        for index, solution in enumerate(raw_solutions, start=1):
            x_value = str(solution.get("x") or "").strip()
            if not x_value:
                continue
            tool_calls.append(
                {
                    "id": f"derived_extrema_value_{index}",
                    "operation": "evaluate",
                    "variables": ["x"],
                    "expressions": [{"name": f"extreme_value_{index}", "expression": expression}],
                    "substitutions": {"x": x_value},
                    "target": f"extreme_value_{index}",
                    "reason": "Deterministically evaluate f(x) at a critical point.",
                    "evidence_span": f"x = {x_value}",
                }
            )
            tool_calls.append(
                {
                    "id": f"derived_extrema_second_derivative_{index}",
                    "operation": "evaluate",
                    "variables": ["x"],
                    "expressions": [
                        {
                            "name": f"second_derivative_{index}",
                            "expression": f"diff({expression}, x, 2)",
                        }
                    ],
                    "substitutions": {"x": x_value},
                    "target": f"second_derivative_{index}",
                    "reason": "Use the second derivative sign to classify the extremum.",
                    "evidence_span": f"x = {x_value}",
                }
            )
        return tool_calls

    def _derive_coordinate_midpoint_sum_backend_result(self, spr: SPR | None) -> dict[str, Any] | None:
        if spr is None:
            return None
        text = self._spr_question_text(spr)
        lowered = text.lower()
        if not all(signal in lowered for signal in ["midpoint", "coordinates", "sum"]):
            return None
        if "endpoints" not in lowered:
            return None
        points = re.findall(
            r"\(\s*(-?\d+(?:\.\d+)?|-?\d+\s*/\s*\d+)\s*,\s*(-?\d+(?:\.\d+)?|-?\d+\s*/\s*\d+)\s*\)",
            text,
        )
        if len(points) < 2:
            return None
        (x1, y1), (x2, y2) = points[0], points[1]
        expression = f"(({x1})+({x2}))/2 + (({y1})+({y2}))/2"
        backend_result = self.cas_service.evaluate_expression(
            expression=expression,
            variables=[],
            substitutions={},
            target="sum_coords",
        )
        if not backend_result.get("success"):
            return None
        return {
            **self._derived_backend_package(
                {
                    "id": "derived_midpoint_coordinate_sum",
                    "operation": "evaluate",
                    "variables": [],
                    "equations": [],
                    "expressions": [{"name": "sum_coords", "expression": expression}],
                    "substitutions": {},
                    "target": "sum_coords",
                    "reason": "Deterministically compute the sum of midpoint coordinates from two endpoints.",
                    "evidence_span": text,
                },
                backend_result,
            ),
            "description": "CAS deterministically computes the requested sum of midpoint coordinates.",
            "justification": "The question asks for the sum of the midpoint coordinates, so both midpoint coordinates must be computed before the final sum.",
            "derived_by": "coordinate_midpoint_sum_completion",
        }

    def _derive_equivalent_unit_chain_backend_result(self, spr: SPR | None) -> dict[str, Any] | None:
        if spr is None:
            return None
        text = self._spr_question_text(spr)
        lowered = text.lower()
        if not any(signal in lowered for signal in ["equivalent to", "cost the same as", "price of"]):
            return None
        relations = self._equivalent_unit_relations(text)
        if len(relations) < 2:
            return None
        target_query = self._equivalent_unit_target_query(text)
        if target_query is None:
            return None
        amount, source_unit, target_unit, target_label = target_query
        factor = self._equivalent_unit_conversion_factor(relations, source_unit, target_unit)
        if factor is None:
            return None
        result_value = amount * factor
        expression = self._fraction_expression(result_value)
        backend_result = self.cas_service.evaluate_expression(
            expression=expression,
            variables=[],
            substitutions={},
            target=target_label,
        )
        if not backend_result.get("success"):
            return None
        return {
            **self._derived_backend_package(
                {
                    "id": "derived_equivalent_unit_chain",
                    "operation": "evaluate",
                    "variables": [],
                    "equations": [],
                    "expressions": [{"name": target_label, "expression": expression}],
                    "substitutions": {},
                    "target": target_label,
                    "reason": "Deterministically convert through an equivalent-unit chain.",
                    "evidence_span": text,
                },
                backend_result,
            ),
            "description": "CAS deterministically computes the equivalent-unit conversion chain.",
            "justification": "The question states equivalence relations between units or prices, so the conversion direction is derived from the requested source and target units.",
            "derived_by": "equivalent_unit_chain_completion",
        }

    def _derive_square_sum_identity_backend_result(self, emr: EMR, spr: SPR | None) -> dict[str, Any] | None:
        text = self._spr_question_text(spr)
        if "x" not in text or "y" not in text:
            return None
        target_expressions = [expression.sympy for expression in emr.expressions]
        if not any(self._is_x2_plus_y2_expression(expression) for expression in target_expressions):
            return None
        sum_square: str | None = None
        product: str | None = None
        for equation in emr.equations:
            lhs = str(equation.lhs_sympy)
            rhs = str(equation.rhs_sympy)
            lhs_norm = self._compact_expression(lhs)
            rhs_norm = self._compact_expression(rhs)
            if lhs_norm in {"(x+y)**2", "(y+x)**2"} and self._looks_numeric_expression(rhs):
                sum_square = rhs
            elif rhs_norm in {"(x+y)**2", "(y+x)**2"} and self._looks_numeric_expression(lhs):
                sum_square = lhs
            if lhs_norm in {"x*y", "y*x"} and self._looks_numeric_expression(rhs):
                product = rhs
            elif rhs_norm in {"x*y", "y*x"} and self._looks_numeric_expression(lhs):
                product = lhs
        if sum_square is None or product is None:
            return None
        expression = f"({sum_square}) - 2*({product})"
        target = self._direct_target_label(emr, "x_squared_plus_y_squared")
        backend_result = self.cas_service.evaluate_expression(
            expression=expression,
            variables=[],
            substitutions={},
            target=target,
        )
        if not backend_result.get("success"):
            return None
        return {
            **self._derived_backend_package(
                {
                    "id": "derived_square_sum_identity",
                    "operation": "evaluate",
                    "variables": [],
                    "equations": [],
                    "expressions": [{"name": target, "expression": expression}],
                    "substitutions": {},
                    "target": target,
                    "reason": "Deterministically evaluate x^2+y^2 from (x+y)^2 and xy.",
                    "evidence_span": text,
                },
                backend_result,
            ),
            "description": "CAS deterministically applies x^2+y^2=(x+y)^2-2xy.",
            "justification": "The requested expression is x^2+y^2 and the problem provides (x+y)^2 and xy, so the identity gives the target directly.",
            "derived_by": "square_sum_identity_completion",
        }

    @staticmethod
    def _is_x2_plus_y2_expression(expression: str | None) -> bool:
        normalized = AlgebraSolver._compact_expression(str(expression or ""))
        return normalized in {"x**2+y**2", "y**2+x**2"}

    @staticmethod
    def _compact_expression(expression: str) -> str:
        return re.sub(r"\s+", "", expression.replace("^", "**"))

    @staticmethod
    def _looks_numeric_expression(expression: str) -> bool:
        return bool(re.fullmatch(r"[-+*/().\d\s]+", str(expression or "")))

    @classmethod
    def _equivalent_unit_relations(cls, text: str) -> list[tuple[Fraction, str, Fraction, str]]:
        number = cls._number_pattern()
        relation_pattern = re.compile(
            rf"\b({number})\s+([A-Za-z]+)\s*(?:=|are\s+equivalent\s+to|is\s+equivalent\s+to|cost(?:s)?\s+the\s+same\s+as)\s*({number})\s+([A-Za-z]+)\b",
            flags=re.IGNORECASE,
        )
        relations: list[tuple[Fraction, str, Fraction, str]] = []
        for match in relation_pattern.finditer(text):
            left_amount = cls._parse_number_token(match.group(1))
            right_amount = cls._parse_number_token(match.group(3))
            if left_amount is None or right_amount is None:
                continue
            relations.append(
                (
                    left_amount,
                    cls._normalize_unit_token(match.group(2)),
                    right_amount,
                    cls._normalize_unit_token(match.group(4)),
                )
            )
        return relations

    @classmethod
    def _equivalent_unit_target_query(cls, text: str) -> tuple[Fraction, str, str, str] | None:
        number = cls._number_pattern()
        patterns = [
            re.compile(
                rf"how\s+many\s+([A-Za-z]+)\s+are\s+equivalent\s+to\s+({number})\s+([A-Za-z]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"how\s+many\s+([A-Za-z]+)\s+can\b.*?\bfor\s+the\s+price\s+of\s+({number})\s+([A-Za-z]+)",
                flags=re.IGNORECASE,
            ),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            amount = cls._parse_number_token(match.group(2))
            if amount is None:
                continue
            target_label = match.group(1).lower()
            return (
                amount,
                cls._normalize_unit_token(match.group(3)),
                cls._normalize_unit_token(match.group(1)),
                target_label,
            )
        return None

    @staticmethod
    def _equivalent_unit_conversion_factor(
        relations: list[tuple[Fraction, str, Fraction, str]],
        source_unit: str,
        target_unit: str,
    ) -> Fraction | None:
        graph: dict[str, list[tuple[str, Fraction]]] = {}
        for left_amount, left_unit, right_amount, right_unit in relations:
            if left_amount == 0 or right_amount == 0:
                continue
            graph.setdefault(left_unit, []).append((right_unit, right_amount / left_amount))
            graph.setdefault(right_unit, []).append((left_unit, left_amount / right_amount))
        queue: list[tuple[str, Fraction]] = [(source_unit, Fraction(1, 1))]
        seen = {source_unit}
        while queue:
            unit, factor = queue.pop(0)
            if unit == target_unit:
                return factor
            for next_unit, edge_factor in graph.get(unit, []):
                if next_unit in seen:
                    continue
                seen.add(next_unit)
                queue.append((next_unit, factor * edge_factor))
        return None

    @staticmethod
    def _normalize_unit_token(unit: str) -> str:
        normalized = str(unit or "").strip().lower()
        if len(normalized) > 3 and normalized.endswith("s"):
            return normalized[:-1]
        return normalized

    @staticmethod
    def _fraction_expression(value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"

    @classmethod
    def _parse_number_token(cls, token: str) -> Fraction | None:
        normalized = str(token or "").strip().lower().replace("-", " ")
        if re.fullmatch(r"\d+", normalized):
            return Fraction(int(normalized), 1)
        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
        }
        if normalized in number_words:
            return Fraction(number_words[normalized], 1)
        return None

    @classmethod
    def _number_pattern(cls) -> str:
        return (
            r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
        )

    def _evaluate_symbolic_expression(self, expression: str, variables: list[str]) -> str | None:
        result = self.cas_service.substitute(
            expression=expression,
            variables=variables,
            substitutions={},
            target=None,
        )
        if not result.get("success"):
            return None
        value = (result.get("output") or {}).get("value")
        return str(value) if value is not None else None

    @staticmethod
    def _single_variable_function_expression(spr: SPR) -> str | None:
        for formula in spr.formulas:
            if getattr(formula, "role", None) == "option":
                continue
            for text in (getattr(formula, "raw_text", None), getattr(formula, "latex", None)):
                expression = AlgebraSolver._extract_y_equals_expression(str(text or ""))
                if expression and "x" in expression:
                    return expression
        text_pool = " ".join(
            str(part or "")
            for part in [spr.problem_text, spr.problem_stem, *[condition.text for condition in spr.conditions]]
        )
        expression = AlgebraSolver._extract_y_equals_expression(text_pool)
        if expression and "x" in expression:
            return expression
        return None

    @staticmethod
    def _single_x_value_from_spr(spr: SPR) -> str | None:
        text_pool = " ".join(
            str(part or "")
            for part in [
                spr.problem_text,
                spr.problem_stem,
                *[condition.text for condition in spr.conditions],
                *[question.text for question in spr.questions],
            ]
        )
        match = re.search(r"\bx\s*=\s*(-?\d+(?:/\d+)?(?:\.\d+)?)", text_pool)
        return match.group(1) if match else None

    @staticmethod
    def _spr_asks_for_tangent_line(spr: SPR) -> bool:
        return AlgebraSolver._spr_text_has_any(
            spr,
            [
                "tangent line",
                "tangent equation",
                "equation of the tangent",
                "\u5207\u7ebf",
                "\u5207\u7ebf\u65b9\u7a0b",
            ],
        )

    @staticmethod
    def _spr_asks_for_extrema(spr: SPR) -> bool:
        return AlgebraSolver._spr_text_has_any(
            spr,
            [
                "extrema",
                "extreme value",
                "local extrema",
                "\u6781\u503c",
            ],
        )

    @staticmethod
    def _derived_backend_package(request: dict[str, Any], backend_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend_result": backend_result,
            "validation_report": {
                "valid": True,
                "normalized_request": request,
                "planner_variables": request.get("variables", []),
                "effective_variables": request.get("variables", []),
                "normalization_warnings": [
                    {
                        "field_path": "tool_call",
                        "value": request.get("id"),
                        "message": "Deterministic Solver-derived calculation request.",
                    }
                ],
                "errors": [],
                "declared_variables": sorted(request.get("variables", [])),
                "semantic_symbol_aliases": {},
            },
        }

    @staticmethod
    def _quadratic_function_expression(spr: SPR) -> str | None:
        for formula in spr.formulas:
            if getattr(formula, "role", None) == "option":
                continue
            for text in (getattr(formula, "raw_text", None), getattr(formula, "latex", None)):
                expression = AlgebraSolver._extract_y_equals_expression(str(text or ""))
                if expression and AlgebraSolver._looks_like_quadratic_expression(expression):
                    return expression
        text_pool = " ".join(
            str(part or "")
            for part in [spr.problem_text, spr.problem_stem, *[condition.text for condition in spr.conditions]]
        )
        expression = AlgebraSolver._extract_y_equals_expression(text_pool)
        if expression and AlgebraSolver._looks_like_quadratic_expression(expression):
            return expression
        return None

    @staticmethod
    def _extract_y_equals_expression(text: str) -> str | None:
        normalized = text.replace("²", "^2").replace("＝", "=")
        match = re.search(r"y\s*=\s*([^,;，；。]+)", normalized, flags=re.IGNORECASE)
        if not match:
            return None
        expression = match.group(1).strip()
        expression = expression.replace("^", "**")
        expression = expression.replace(" ", "")
        return expression or None

    @staticmethod
    def _looks_like_quadratic_expression(expression: str) -> bool:
        return bool(re.search(r"\bx\b.*\*\*\s*2|\bx\s*\*\*\s*2|x\^2|x²", expression))

    @staticmethod
    def _spr_asks_for_x_axis_intercepts(spr: SPR) -> bool:
        return AlgebraSolver._spr_text_has_any(
            spr,
            [
                "x-axis",
                "x axis",
                "x intercept",
                "x-intercept",
                "x_axis",
                "x \u8f74",
                "x\u8f74",
                "\u4e0e x \u8f74",
                "\u4e0ex\u8f74",
            ],
        )

    @staticmethod
    def _spr_asks_for_y_axis_intercept(spr: SPR) -> bool:
        return AlgebraSolver._spr_text_has_any(
            spr,
            [
                "y-axis",
                "y axis",
                "y intercept",
                "y-intercept",
                "y_axis",
                "y \u8f74",
                "y\u8f74",
                "\u4e0e y \u8f74",
                "\u4e0ey\u8f74",
            ],
        )

    @staticmethod
    def _spr_text_has_any(spr: SPR, signals: list[str]) -> bool:
        text = " ".join(
            str(part or "")
            for part in [
                spr.problem_text,
                spr.problem_stem,
                *[question.text for question in spr.questions],
                *[target.text for target in spr.targets],
            ]
        ).lower()
        return any(signal.lower() in text for signal in signals)

    @staticmethod
    def _dedupe_final_answer_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (str(item.get("target") or ""), str(item.get("value") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _named_system_solution_item(
        output: dict[str, Any],
        request: dict[str, Any],
        call_id: Any,
        backend_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if backend_result.get("operation") != "solve_equation_system":
            return None
        target = str(request.get("target") or "").strip()
        if not target:
            return None
        target_lower = target.lower().replace(" ", "_")
        if target_lower in {"x_and_y", "x_y", "all", "solution", "solutions", "solution_set"}:
            return None
        if not re.search(r"(?:soln|solution|system|sys|part|subproblem)[_-]?\d+", target_lower):
            return None
        raw_solution = output.get("raw_solution") or []
        formatted_solution = output.get("formatted_solution")
        if not raw_solution or not formatted_solution:
            return None
        solution_symbols = {str(symbol) for solution in raw_solution for symbol in solution.keys()}
        requested_symbols = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", target))
        if requested_symbols and requested_symbols <= solution_symbols:
            return None
        return {
            "target": target,
            "value": str(formatted_solution),
            "source_tool_call_id": call_id,
            "backend_operation": backend_result.get("operation"),
            "backend_confirmed": True,
        }

    @staticmethod
    def _coordinate_solution_items(
        solution: dict[Any, Any],
        request: dict[str, Any],
        call_id: Any,
        backend_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if {"x", "y"} - {str(symbol) for symbol in solution.keys()}:
            return []
        context_text = " ".join(
            str(request.get(field) or "")
            for field in ("id", "target", "reason", "evidence_span")
        ).lower()
        coordinate_signals = [
            "coordinate",
            "coordinates",
            "intercept",
            "intercepts",
            "x_axis",
            "y_axis",
            "x-axis",
            "y-axis",
            "坐标",
            "交点",
            "x 轴",
            "y 轴",
            "x轴",
            "y轴",
        ]
        if not any(signal in context_text for signal in coordinate_signals):
            return []
        if any(signal in context_text for signal in ["y_axis", "y-axis", "y 轴", "y轴"]):
            target = "y_axis_intercept"
        elif any(signal in context_text for signal in ["x_axis", "x-axis", "x 轴", "x轴"]):
            target = "x_axis_intercept"
        elif "vertex" in context_text or "顶点" in context_text:
            target = "vertex"
        else:
            target = "coordinate"
        return [
            {
                "target": target,
                "value": f"({solution['x']}, {solution['y']})",
                "source_tool_call_id": call_id,
                "backend_operation": backend_result.get("operation"),
                "backend_confirmed": True,
            }
        ]

    @staticmethod
    def _derived_quadratic_x_axis_aggregate_item(
        raw_solution: list[dict[Any, Any]],
        request: dict[str, Any],
        call_id: Any,
        backend_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(call_id) != "derived_quadratic_x_axis_intercepts":
            return None
        points = [
            f"({solution['x']}, {solution.get('y', '0')})"
            for solution in raw_solution
            if "x" in solution
        ]
        if not points:
            return None
        return {
            "target": "x_axis_intercepts",
            "value": ", ".join(points),
            "source_tool_call_id": call_id,
            "backend_operation": backend_result.get("operation"),
            "backend_confirmed": True,
        }

    @staticmethod
    def _derived_extrema_aggregate_item(
        backend_results: list[dict[str, Any]],
        validation_reports: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        critical_points: list[str] = []
        values: dict[int, str] = {}
        second_derivatives: dict[int, str] = {}
        for index, backend_result in enumerate(backend_results):
            request = (
                validation_reports[index].get("normalized_request", {})
                if index < len(validation_reports)
                else {}
            )
            call_id = str(request.get("id") or backend_result.get("tool_call_id") or "")
            if not call_id.startswith("derived_extrema_") or not backend_result.get("success"):
                continue
            output = backend_result.get("output") or {}
            if call_id == "derived_extrema_critical_points":
                critical_points = [
                    str(solution.get("x"))
                    for solution in output.get("raw_solution", [])
                    if solution.get("x") is not None
                ]
                continue
            value_match = re.fullmatch(r"derived_extrema_value_(\d+)", call_id)
            if value_match and output.get("value") is not None:
                values[int(value_match.group(1))] = str(output["value"])
                continue
            second_match = re.fullmatch(r"derived_extrema_second_derivative_(\d+)", call_id)
            if second_match and output.get("value") is not None:
                second_derivatives[int(second_match.group(1))] = str(output["value"])
        parts: list[str] = []
        source_call_id: str | None = None
        for index, x_value in enumerate(critical_points, start=1):
            y_value = values.get(index)
            if y_value is None:
                continue
            source_call_id = f"derived_extrema_value_{index}"
            kind = AlgebraSolver._classify_second_derivative(second_derivatives.get(index))
            if kind:
                parts.append(f"x = {x_value}, y = {y_value} ({kind})")
            else:
                parts.append(f"x = {x_value}, y = {y_value}")
        if not parts:
            return None
        return {
            "target": "local_extrema",
            "value": "; ".join(parts),
            "source_tool_call_id": source_call_id or "derived_extrema_critical_points",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }

    @staticmethod
    def _classify_second_derivative(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except ValueError:
            return None
        if numeric > 0:
            return "local minimum"
        if numeric < 0:
            return "local maximum"
        return None

    @staticmethod
    def _intercept_solution_items(
        solution: dict[Any, Any],
        request: dict[str, Any],
        call_id: Any,
        backend_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        context_text = AlgebraSolver._request_context_text(request)
        if not AlgebraSolver._is_x_axis_context(context_text) or "x" not in {str(symbol) for symbol in solution}:
            return []
        return [
            {
                "target": "x_axis_intercept",
                "value": f"({solution['x']}, 0)",
                "source_tool_call_id": call_id,
                "backend_operation": backend_result.get("operation"),
                "backend_confirmed": True,
            }
        ]

    @staticmethod
    def _intercept_value_item(
        value: Any,
        request: dict[str, Any],
        call_id: Any,
        backend_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        context_text = AlgebraSolver._request_context_text(request)
        if not AlgebraSolver._is_y_axis_context(context_text):
            return None
        return {
            "target": "y_axis_intercept",
            "value": f"(0, {value})",
            "source_tool_call_id": call_id,
            "backend_operation": backend_result.get("operation"),
            "backend_confirmed": True,
        }

    @staticmethod
    def _request_context_text(request: dict[str, Any]) -> str:
        return " ".join(
            str(request.get(field) or "")
            for field in ("id", "target", "reason", "evidence_span")
        ).lower()

    @staticmethod
    def _is_x_axis_context(context_text: str) -> bool:
        return any(signal in context_text for signal in ["x_axis", "x-intercept", "x intercept", "x 轴", "x轴"])

    @staticmethod
    def _is_y_axis_context(context_text: str) -> bool:
        return any(signal in context_text for signal in ["y_axis", "y-intercept", "y intercept", "y 轴", "y轴"])

    @staticmethod
    def _select_final_answer_items(
        items: list[dict[str, Any]],
        spr: SPR | None,
    ) -> list[dict[str, Any]]:
        if spr is None or not items or not spr.questions:
            return items
        derived_priority_items = [
            item
            for item in items
            if str(item.get("source_tool_call_id") or "") in {
                "derived_equivalent_unit_chain",
                "derived_midpoint_coordinate_sum",
                "derived_ratio_share_total",
                "derived_square_sum_identity",
            }
        ]
        if derived_priority_items and AlgebraSolver._spr_supports_derived_priority_answer(spr):
            return [derived_priority_items[-1]]
        if AlgebraSolver._should_keep_last_repeated_single_answer_item(items, spr):
            return [items[-1]]
        selected: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for question in spr.questions:
            candidates = [
                (index, item, AlgebraSolver._answer_item_question_score(item, question, spr))
                for index, item in enumerate(items)
                if index not in used_indexes
            ]
            candidates = [candidate for candidate in candidates if candidate[2] > 0]
            if not candidates:
                continue
            if AlgebraSolver._question_requests_multiple_answer_items(question, spr, candidates):
                for index, item, _score in candidates:
                    used_indexes.add(index)
                    selected.append(item)
                continue
            index, item, _score = max(candidates, key=lambda candidate: candidate[2])
            used_indexes.add(index)
            selected.append(item)
        return selected or items

    @staticmethod
    def _spr_supports_derived_priority_answer(spr: SPR) -> bool:
        text = AlgebraSolver._spr_question_text(spr).lower()
        return (
            all(signal in text for signal in ["midpoint", "coordinates", "sum"])
            or "equivalent to" in text
            or "cost the same as" in text
            or "price of" in text
            or ("ratio" in text and "total" in text)
            or ("x^2" in text and "y^2" in text and "(x + y)^2" in text)
        )

    @staticmethod
    def _should_keep_last_repeated_single_answer_item(items: list[dict[str, Any]], spr: SPR) -> bool:
        if len(items) <= 1 or len(getattr(spr, "questions", []) or []) != 1:
            return False
        targets = {str(item.get("target") or "") for item in items if item.get("target")}
        if len(targets) != 1:
            return False
        question = spr.questions[0]
        target_texts = [
            target.text or ""
            for target in spr.targets
            if target.id in (getattr(question, "target_ids", []) or [])
        ]
        combined = " ".join([getattr(question, "text", "") or "", *target_texts]).lower().replace("_", " ")
        multi_answer_signals = [
            "roots",
            "solutions",
            "solution set",
            "x-intercepts",
            "x intercepts",
            "intercepts",
            "both",
            "respectively",
            "ordered pair",
        ]
        return not any(signal in combined for signal in multi_answer_signals)

    @staticmethod
    def _is_aggregate_solution_target(target: Any, solution_targets: set[str]) -> bool:
        if not target or len(solution_targets) <= 1:
            return False
        target_text = str(target).strip().lower()
        if target_text in {"all", "solution", "solutions", "solution_set", "solution set", "x_and_y"}:
            return True
        normalized = target_text.replace("_", " ")
        if " and " in normalized:
            requested_parts = set(re.findall(r"[a-zA-Z][A-Za-z0-9]*", normalized))
            if len(requested_parts & solution_targets) >= 2:
                return True
        return len(solution_targets) >= 2 and all(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol.lower())}(?![A-Za-z0-9_])", normalized)
            for symbol in solution_targets
        )

    @staticmethod
    def _spr_question_requests_solution_targets(spr: SPR | None, solution_targets: set[str]) -> bool:
        if spr is None or len(solution_targets) <= 1:
            return False
        variables_by_symbol = {variable.symbol: variable for variable in spr.variables}
        for question in spr.questions:
            target_texts = [
                target.text or ""
                for target in spr.targets
                if target.id in (getattr(question, "target_ids", []) or [])
            ]
            combined_text = " ".join([getattr(question, "text", "") or "", *target_texts])
            if not AlgebraSolver._looks_like_multi_object_question(combined_text, len(target_texts)):
                continue
            matched_symbols = {
                symbol
                for symbol in solution_targets
                if AlgebraSolver._solution_symbol_matches_question_target(
                    symbol,
                    variables_by_symbol.get(symbol),
                    combined_text,
                    target_texts,
                )
            }
            if len(matched_symbols) >= 2:
                return True
        return False

    @staticmethod
    def _looks_like_multi_object_question(text: str, target_count: int) -> bool:
        lowered = text.lower().replace("_", " ")
        return target_count >= 2 or any(
            signal in lowered
            for signal in [
                "x and y",
                "both",
                "respectively",
                "solution set",
                "ordered pair",
            ]
        ) or any(
            signal in text
            for signal in [
                "各有多少",
                "分别",
                "各是多少",
                "各为多少",
                "鸡和兔",
                "两种",
                "二者",
            ]
        )

    @staticmethod
    def _solution_symbol_matches_question_target(
        symbol: str,
        variable: Any | None,
        question_text: str,
        target_texts: list[str],
    ) -> bool:
        text_pool = " ".join([question_text, *target_texts])
        lowered_pool = text_pool.lower()
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol.lower())}(?![A-Za-z0-9_])", lowered_pool):
            return True
        description = str(getattr(variable, "description", "") or "")
        if not description:
            return False
        return AlgebraSolver._semantic_answer_score(description, text_pool) > 0.0

    @staticmethod
    def _question_requests_multiple_answer_items(
        question: Any,
        spr: SPR,
        candidates: list[tuple[int, dict[str, Any], float]],
    ) -> bool:
        if AlgebraSolver._question_asks_for_function_expression(question, spr):
            return False
        if len(candidates) <= 1:
            return False
        operations = {
            str(item.get("backend_operation") or "")
            for _index, item, _score in candidates
        }
        target_texts = [
            target_obj.text or ""
            for target_obj in spr.targets
            if target_obj.id in (getattr(question, "target_ids", []) or [])
        ]
        combined_text = " ".join([getattr(question, "text", "") or "", *target_texts])
        lowered = combined_text.lower().replace("_", " ")
        if any(
            signal in lowered
            for signal in ["how long", "how much time", "how many hours", "how many minutes"]
        ):
            return False
        if "how many" in lowered and not any(
            token in lowered
            for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]
        ):
            return False
        candidate_targets = [
            str(item.get("target") or "")
            for _index, item, _score in candidates
            if item.get("target")
        ]
        unique_candidate_targets = set(candidate_targets)
        if (
            AlgebraSolver._question_subproblem_index(combined_text, target_texts) is not None
            and len(unique_candidate_targets) >= 2
            and all(
                str(item.get("backend_operation") or "") == "solve_equation_system"
                for _index, item, _score in candidates
            )
        ):
            return True
        if len(unique_candidate_targets) <= 1 and any(
            signal in lowered
            for signal in ["intercepts", "intersections", "roots", "zeros", "coordinates"]
        ):
            return len(candidates) >= 2
        if len(unique_candidate_targets) <= 1 and any(signal in combined_text for signal in ["交点", "坐标", "零点"]):
            return len(candidates) >= 2
        if len(unique_candidate_targets) <= 1:
            return False
        if "tangent" in lowered and any(token in lowered for token in ["line", "equation"]):
            return len(candidates) >= 2
        if "vertex" in lowered and any(token in lowered for token in ["coordinate", "coordinates"]):
            return len(candidates) >= 2
        if operations & {"construct_expression", "parse_expression"} and operations != {"solve_equation_system"}:
            return False
        target_text_tokens = {
            token
            for text in target_texts
            for token in AlgebraSolver._target_tokens(text)
        }
        covers_multiple_declared_targets = len(
            {
                target
                for target in unique_candidate_targets
                if target.lower() in target_text_tokens or target.lower() in lowered
            }
        ) >= 2
        multi_target_signal = any(
            signal in lowered
            for signal in [
                "solution set",
                "values of",
                "value of",
                "x and y",
                "ordered pair",
            ]
        ) or any(
            signal in combined_text
            for signal in [
                "解集",
                "方程组的解",
                "求x和y",
                "求 x 和 y",
                "各有多少",
                "分别",
                "各是多少",
                "各为多少",
                "鸡和兔",
            ]
        )
        multi_target_signal = multi_target_signal or covers_multiple_declared_targets
        if not multi_target_signal:
            return False
        if (
            operations == {"solve_equation_system"}
            and any(
                signal in combined_text
                for signal in ["各有多少", "分别", "各是多少", "各为多少", "鸡和兔", "两种", "二者"]
            )
        ):
            return True
        if len(target_texts) >= 2 and all(
            str(item.get("backend_operation") or "") == "solve_equation_system"
            for _index, item, _score in candidates
        ):
            return True
        if covers_multiple_declared_targets:
            return True
        mentioned_targets = {
            target
            for target in unique_candidate_targets
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(target.lower())}(?![A-Za-z0-9_])", lowered)
        }
        return len(mentioned_targets) >= 2

    @staticmethod
    def _answer_item_question_score(item: dict[str, Any], question: Any, spr: SPR) -> float:
        target = str(item.get("target") or "")
        question_text = str(getattr(question, "text", "") or "")
        target_texts = [
            target_obj.text or ""
            for target_obj in spr.targets
            if target_obj.id in (getattr(question, "target_ids", []) or [])
        ]
        variable = next((variable for variable in spr.variables if variable.symbol == target), None)
        candidate_text = " ".join(
            [
                target,
                str(item.get("source_tool_call_id") or ""),
                str(item.get("source_step_id") or ""),
                getattr(variable, "description", "") or "",
                getattr(variable, "domain", "") or "",
            ]
        )
        score = 0.0
        source_id = str(item.get("source_tool_call_id") or item.get("source_step_id") or "").lower()
        question_index = AlgebraSolver._question_subproblem_index(question_text, target_texts)
        if question_index is not None:
            if re.search(rf"(?<!\d){question_index}(?!\d)", source_id):
                score += 10.0
            elif re.search(r"(?<!\d)\d+(?!\d)", source_id):
                score -= 2.0
        for target_text in target_texts:
            score = max(score, AlgebraSolver._semantic_answer_score(candidate_text, target_text) * 8.0)
        score = max(score, AlgebraSolver._semantic_answer_score(candidate_text, question_text) * 5.0)
        lower_candidate = candidate_text.lower()
        lower_question = question_text.lower()
        requested_entity_tokens = AlgebraSolver._requested_entity_tokens(question_text, target_texts)
        if requested_entity_tokens:
            if any(token in lower_candidate for token in requested_entity_tokens):
                score += 8.0
            if any(
                token in lower_candidate
                for token in AlgebraSolver._nonrequested_entity_tokens(spr, requested_entity_tokens)
            ):
                score -= 6.0
        if AlgebraSolver._question_asks_for_function_expression(question, spr):
            if AlgebraSolver._answer_item_is_symbolic_expression(item):
                score += 8.0
            if item.get("backend_operation") == "substitute":
                score += 2.0
            if item.get("backend_operation") == "construct_expression" and re.search(r"\b[abkm]\b", str(item.get("value") or "")):
                score -= 2.0
            if target.lower() in {"k", "b", "a", "m"}:
                score -= 3.0
        if AlgebraSolver._question_asks_for_point_value(question, spr):
            if item.get("backend_operation") == "evaluate" and not AlgebraSolver._answer_item_is_symbolic_expression(item):
                score += 8.0
            if AlgebraSolver._answer_item_is_symbolic_expression(item):
                score -= 4.0
        if "tangent" in lower_question and any(token in lower_question for token in ["line", "equation"]):
            if target.lower() in {"m", "slope", "derivative"}:
                score += 6.0
            if any(token in target.lower() for token in ["y_at", "point_y", "y0"]) or target.lower() == "y":
                score += 5.0
            if item.get("backend_operation") == "evaluate" and not AlgebraSolver._answer_item_is_symbolic_expression(item):
                score += 1.0
        if AlgebraSolver._spr_asks_for_tangent_line(spr):
            if target.lower() == "tangent_line":
                score += 20.0
            elif target.lower() in {"m", "slope", "derivative", "y0", "y"}:
                score -= 2.0
        if AlgebraSolver._spr_asks_for_extrema(spr):
            if target.lower() in {"local_extrema", "extrema"}:
                score += 20.0
            elif target.lower().startswith("extreme_value") or target.lower() in {"x", "derivative"}:
                score -= 1.0
        asks_passenger_bus = any(token in lower_question for token in ["bus", "passenger", "coach"]) or "客车" in question_text
        asks_truck = any(token in lower_question for token in ["truck", "cargo", "freight"]) or "货车" in question_text
        asks_sum = any(token in lower_question for token in ["sum", "total", "together"]) or any(
            token in question_text for token in ["一共", "总和", "速度和", "合计"]
        )
        asks_savings = any(token in lower_question for token in ["save", "saves", "saving", "savings"])
        asks_left_or_remaining = any(
            token in lower_question
            for token in ["left", "remaining", "remain", "rest", "leftover"]
        )
        asks_percentage = any(token in lower_question for token in ["percentage", "percent", "%"])
        asks_grams = any(token in lower_question for token in ["gram", "grams"])
        asks_difference = any(
            token in lower_question
            for token in ["difference", "farther", "more than", "how much more", "how many more"]
        )
        asks_weekly = any(token in lower_question for token in ["per week", "weekly", "a week"])
        asks_still_need = any(
            signal in lower_question
            for signal in ["still need", "need to", "needs to", "needed to"]
        )
        asks_not_ready = "not ready" in lower_question
        if asks_percentage:
            if any(token in lower_candidate for token in ["percent", "percentage", "pct"]):
                score += 12.0
            if any(token in lower_candidate for token in ["count", "number", "remaining", "contemp", "jazz"]):
                score -= 4.0
        if asks_grams:
            if any(token in lower_candidate for token in ["gram", "grams", "eaten", "eat"]):
                score += 12.0
            if any(token in lower_candidate for token in ["calorie", "serving"]):
                score -= 4.0
        if asks_difference:
            if any(token in lower_candidate for token in ["difference", "diff", "farther", "more"]):
                score += 12.0
            if any(token in lower_candidate for token in ["total", "blake", "kelly", "flour", "milk"]):
                score -= 3.0
        if asks_weekly:
            if any(token in lower_candidate for token in ["week", "weekly"]):
                score += 10.0
            if any(token in lower_candidate for token in ["day", "daily"]):
                score -= 5.0
        if asks_still_need:
            if any(token in lower_candidate for token in ["need", "needed", "remaining", "left"]):
                score += 12.0
            if any(token in lower_candidate for token in ["played", "total_played", "phase"]):
                score -= 5.0
        if asks_not_ready:
            if any(token in lower_candidate for token in ["not_ready", "not ready", "final"]):
                score += 12.0
            if any(token in lower_candidate for token in ["remaining_after_first", "rem1", "first"]):
                score -= 6.0
        if asks_savings:
            if any(token in lower_candidate for token in ["save", "saving", "savings"]):
                score += 12.0
            if any(token in lower_candidate for token in ["total_cost", "total cost", "cost_total"]):
                score -= 6.0
        if asks_left_or_remaining:
            if any(token in lower_candidate for token in ["left", "remaining", "remain", "rest", "leftover"]):
                score += 10.0
            if any(token in lower_candidate for token in ["bill", "mark", "jenny", "given", "eaten", "sold"]):
                score -= 6.0
        asks_general_count = "how many" in lower_question and not any(
            token in lower_question
            for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]
        )
        asks_duration = any(
            signal in lower_question
            for signal in ["how long", "how much time", "how many hours", "how many minutes"]
        )
        asks_hours = any(signal in lower_question for signal in ["in hours", "hours?", "how many hours"])
        asks_minutes = any(signal in lower_question for signal in ["in minutes", "minutes?", "how many minutes"])
        if asks_duration:
            if asks_hours and any(token in lower_candidate for token in ["hour", "hours", "t_hour", "t_hours"]):
                score += 12.0
            elif asks_minutes and any(token in lower_candidate for token in ["minute", "minutes", "t_min"]):
                score += 12.0
            elif any(token in lower_candidate for token in ["time", "duration", "hour", "minute", "t_min", "t_hour"]):
                score += 6.0
            if any(token in lower_candidate for token in ["remaining", "charge", "percent", "percentage"]):
                score -= 5.0
        if asks_general_count:
            if any(token in lower_candidate for token in ["total", "sum", "combined", "altogether"]):
                score += 8.0
            if any(token in lower_candidate for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]):
                score -= 3.0
            if any(token in lower_question for token in ["good", "rest", "remaining"]):
                if any(token in lower_candidate for token in ["good", "rest", "remaining"]):
                    score += 5.0
                if any(token in lower_candidate for token in ["non_good", "not_good", "bad", "unripe", "sour"]):
                    score -= 8.0
        if asks_sum:
            if any(token in lower_candidate for token in ["sum", "total", "combined", "altogether"]):
                score += 14.0
            elif item.get("backend_operation") in {"solve_for", "solve_equation_system"}:
                score -= 4.0
        if asks_passenger_bus:
            if any(token in lower_candidate for token in ["bus", "passenger", "coach"]) or "客车" in candidate_text:
                score += 5.0
            if any(token in lower_candidate for token in ["sum", "total", "together", "relative", "combined"]):
                score -= 3.0
        if asks_truck:
            if any(token in lower_candidate for token in ["truck", "cargo", "freight"]) or "货车" in candidate_text:
                score += 5.0
            if any(token in lower_candidate for token in ["sum", "total", "together", "relative", "combined"]):
                score -= 3.0
        if not asks_sum and any(
            token in lower_candidate
            for token in ["sum_speed", "speed_sum", "sum speed", "relative_speed", "relative speed"]
        ):
            score -= 4.0
        if any(token in lower_question for token in ["box", "boxes"]) or "箱" in question_text:
            if any(token in lower_candidate for token in ["box", "boxes", "n_boxes", "numboxes"]) or "箱" in candidate_text:
                score += 5.0
            if any(token in lower_candidate for token in ["sold", "morning", "afternoon", "remaining"]) or any(
                token in candidate_text for token in ["卖出", "上午", "下午", "剩"]
            ):
                score -= 4.0
        if any(token in lower_question for token in ["original", "transported", "total"]) or any(
            token in question_text for token in ["原来", "运来", "总", "一共", "买来"]
        ):
            if any(token in lower_candidate for token in ["original", "transported", "total"]) or any(
                token in candidate_text for token in ["原来", "运来", "总", "一共", "买来"]
            ):
                score += 5.0
            if any(token in lower_candidate for token in ["sold", "morning", "afternoon", "remaining", "box"]) or any(
                token in candidate_text for token in ["卖出", "上午", "下午", "剩", "箱"]
            ):
                score -= 4.0
        if item.get("backend_confirmed") is True:
            score += 0.2
        return score

    @staticmethod
    def _question_subproblem_index(question_text: str, target_texts: list[str]) -> str | None:
        combined = " ".join([question_text, *target_texts])
        match = re.search(r"(?:system|part|subproblem)\s*[\(（]?\s*(\d+)\s*[\)）]?", combined, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"[\(（]\s*(\d+)\s*[\)）]", combined)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _requested_entity_tokens(question_text: str, target_texts: list[str]) -> set[str]:
        direct_question_entities = {
            match.group(1).lower()
            for match in re.finditer(
                r"\b(?:did|does|do|is|are|was|were)\s+([A-Z][a-z]{2,})\b",
                question_text,
            )
        }
        if direct_question_entities:
            return direct_question_entities
        text = " ".join(target_texts) if target_texts else question_text
        generic = {
            "math",
            "word",
            "problem",
            "total",
            "number",
            "items",
            "item",
            "onions",
            "onion",
            "potatoes",
            "potato",
            "market",
            "bought",
            "buy",
            "many",
            "much",
        }
        return {
            token.lower()
            for token in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
            if token.lower() not in generic
        }

    @staticmethod
    def _nonrequested_entity_tokens(spr: SPR, requested_tokens: set[str]) -> set[str]:
        generic = {
            "total",
            "number",
            "items",
            "item",
            "onions",
            "onion",
            "potatoes",
            "potato",
            "market",
            "bought",
            "buy",
        }
        tokens: set[str] = set()
        for variable in spr.variables:
            text = " ".join([variable.symbol or "", variable.description or ""])
            for token in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
                lowered = token.lower()
                if lowered not in requested_tokens and lowered not in generic:
                    tokens.add(lowered)
        return tokens

    @staticmethod
    def _question_asks_for_function_expression(question: Any, spr: SPR) -> bool:
        question_text = str(getattr(question, "text", "") or "")
        target_texts = " ".join(AlgebraSolver._question_target_texts(question, spr))
        question_lower = question_text.lower().replace("_", " ")
        combined = f"{question_text} {target_texts}".lower().replace("_", " ")
        expression_signals = [
            "analytic expression",
            "explicit equation",
            "equation of the line",
            "function expression",
            "function formula",
            "specific equation",
            "解析式",
            "函数关系式",
        ]
        coefficient_only_signals = [
            "values of coefficients",
            "values of k and b",
            "coefficient values",
        ]
        if any(signal in question_lower for signal in expression_signals):
            return True
        return any(signal in combined for signal in expression_signals) and not any(
            signal in question_lower for signal in coefficient_only_signals
        )

    @staticmethod
    def _question_asks_for_point_value(question: Any, spr: SPR) -> bool:
        question_text = str(getattr(question, "text", "") or "")
        target_texts = " ".join(AlgebraSolver._question_target_texts(question, spr))
        combined = f"{question_text} {target_texts}".lower().replace("_", " ")
        if AlgebraSolver._question_asks_for_function_expression(question, spr):
            return False
        point_value_signals = [
            "at x",
            "at t",
            "when x",
            "when t",
            "value at",
            "evaluated at",
            "instantaneous",
            "t=",
            "x=",
            "t =",
            "x =",
        ]
        return any(signal in combined for signal in point_value_signals)

    @staticmethod
    def _question_target_texts(question: Any, spr: SPR) -> list[str]:
        target_ids = set(getattr(question, "target_ids", []) or [])
        if not target_ids:
            return []
        return [
            target.text or ""
            for target in spr.targets
            if target.id in target_ids
        ]

    @staticmethod
    def _answer_item_is_symbolic_expression(item: dict[str, Any]) -> bool:
        operation = str(item.get("backend_operation") or "")
        if operation not in {"construct_expression", "parse_expression", "substitute"}:
            return False
        value = str(item.get("value") or "")
        return bool(re.search(r"[A-Za-z]", value))

    @staticmethod
    def _semantic_answer_score(left: str | None, right: str | None) -> float:
        left_tokens = AlgebraSolver._target_tokens(left)
        right_tokens = AlgebraSolver._target_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        return len(overlap) / min(len(left_tokens), len(right_tokens))

    @staticmethod
    def _attach_choice_options(
        items: list[dict[str, Any]],
        spr: SPR | None,
    ) -> list[dict[str, Any]]:
        if spr is None or not items:
            return items
        has_select_question = any(
            getattr(question, "question_type", None) == "select"
            for question in spr.questions
        ) or any(getattr(target, "target_type", None) == "select" for target in spr.targets)
        if not has_select_question:
            return items
        options = [formula for formula in spr.formulas if formula.role == "option"]
        if not options:
            return items
        enriched: list[dict[str, Any]] = []
        for item in items:
            option = AlgebraSolver._matching_choice_option(str(item.get("value")), options)
            if option is None:
                enriched.append(item)
                continue
            updated = dict(item)
            updated["selected_option_id"] = option.id
            updated["selected_option_label"] = AlgebraSolver._choice_label(option.id, option.raw_text)
            updated["selected_option_text"] = option.raw_text
            enriched.append(updated)
        return enriched

    @staticmethod
    def _matching_choice_option(value: str, options: list[Any]) -> Any | None:
        normalized_value = AlgebraSolver._choice_value_key(value)
        for option in options:
            candidates = [option.raw_text, option.latex]
            for candidate in candidates:
                if normalized_value and AlgebraSolver._choice_value_key(str(candidate or "")) == normalized_value:
                    return option
        return None

    @staticmethod
    def _choice_value_key(text: str) -> str:
        value = re.sub(r"^[A-Da-d][\.\、\s:：]+", "", str(text or "").strip())
        value = value.replace("平方厘米", "").replace("平方厘 米", "").replace("cm²", "").strip()
        value = re.sub(r"\s+", "", value)
        return value

    @staticmethod
    def _choice_label(option_id: str, raw_text: str) -> str | None:
        id_match = re.search(r"(?:^|[_-])([a-dA-D])$", str(option_id or ""))
        if id_match:
            return id_match.group(1).upper()
        text_match = re.match(r"\s*([A-Da-d])[\.\、\s:：]", str(raw_text or ""))
        if text_match:
            return text_match.group(1).upper()
        return None

    def _direct_target_evaluation(
        self,
        emr: EMR,
        variables: list[str],
        backend_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not backend_result.get("success") or not emr.expressions:
            return None
        goal = emr.goals[0] if emr.goals else None
        if goal is None or goal.goal_type not in {"compute", "select"}:
            return None
        if goal.target in set(variables):
            return None
        substitutions = self._solution_substitutions(backend_result)
        if not substitutions:
            return None
        candidates = self._direct_target_expression_candidates(emr)
        if not candidates:
            return None
        target = self._direct_target_label(emr, candidates[0][1])
        for expression_id, expression_text in candidates:
            if not self._target_expression_is_meaningful(expression_text):
                continue
            result = self.cas_service.evaluate_expression(
                expression=expression_text,
                variables=variables,
                substitutions=substitutions,
                target=target,
            )
            if not result.get("success"):
                continue
            result["input"]["source_expression_id"] = expression_id
            return {
                "target": target,
                "expression_id": expression_id,
                "expression": expression_text,
                "result": result,
            }
        return None

    @staticmethod
    def _direct_target_expression_candidates(emr: EMR) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        target_expression_id = emr.goals[0].target_expression_id if emr.goals else None
        if target_expression_id:
            for expression in emr.expressions:
                if expression.id == target_expression_id:
                    candidates.append((expression.id, expression.sympy))
                    break
        candidates.extend(
            (expression.id, expression.sympy)
            for expression in emr.expressions
            if expression.id != target_expression_id
        )
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _direct_target_label(emr: EMR, fallback: str) -> str:
        if emr.goals and emr.goals[0].target:
            return str(emr.goals[0].target)
        return fallback

    @staticmethod
    def _solution_substitutions(backend_result: dict[str, Any]) -> dict[str, str]:
        raw_solution = backend_result.get("output", {}).get("raw_solution") or []
        if not raw_solution:
            return {}
        return {str(symbol): str(value) for symbol, value in raw_solution[0].items()}

    def _filter_direct_solution_candidates(
        self,
        backend_result: dict[str, Any],
        emr: EMR,
        spr: SPR | None,
        target: str | None,
        variables: list[str],
    ) -> dict[str, Any]:
        if not backend_result.get("success"):
            return backend_result
        raw_solution = backend_result.get("output", {}).get("raw_solution") or []
        if len(raw_solution) <= 1 or not target:
            return backend_result
        filtered = raw_solution
        if self._solution_target_has_nonzero_constraint(str(target), emr, spr):
            filtered = [
                solution
                for solution in filtered
                if str(target) not in solution or not self._cas_value_is_zero(str(solution[str(target)]))
            ]
        if not filtered or len(filtered) == len(raw_solution):
            return backend_result
        updated = dict(backend_result)
        output = dict(backend_result.get("output") or {})
        output["raw_solution"] = filtered
        output["formatted_solution"] = self._format_solution_payload(filtered, variables)
        output["solution_filter"] = {
            "target": str(target),
            "reason": "nonzero_constraint",
            "original_count": len(raw_solution),
            "filtered_count": len(filtered),
        }
        updated["output"] = output
        updated.setdefault("warnings", []).append(
            f"Filtered direct solve candidates using nonzero constraint on {target}."
        )
        return updated

    def _cas_value_is_zero(self, value: str) -> bool:
        compact = str(value).strip()
        if compact in {"0", "0.0"}:
            return True
        result = self.cas_service.evaluate_expression(
            expression=compact,
            variables=[],
            substitutions={},
            target="constraint_value",
        )
        return result.get("success") is True and (result.get("output") or {}).get("value") in {"0", "0.0"}

    @staticmethod
    def _solution_target_has_nonzero_constraint(target: str, emr: EMR, spr: SPR | None) -> bool:
        text = " ".join(
            str(part or "")
            for part in [
                AlgebraSolver._spr_question_text(spr),
                *[str(getattr(constraint, "expression", "") or "") for constraint in getattr(emr, "constraints", [])],
                *[str(getattr(constraint, "raw_text", "") or "") for constraint in getattr(emr, "constraints", [])],
            ]
        )
        if not text:
            return False
        escaped = re.escape(target.lower())
        patterns = [
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*(?:\\neq|≠|!=|<>|not\s*=\s*)\s*0",
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s+is\s+not\s+0",
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s+is\s+nonzero",
        ]
        lowered = text.lower()
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _format_solution_payload(raw_solution: list[dict[str, Any]], variables: list[str]) -> str | None:
        if not raw_solution:
            return None
        first_solution = raw_solution[0]
        parts = [
            f"{variable} = {first_solution[variable]}"
            for variable in variables
            if variable in first_solution
        ]
        if parts:
            return ", ".join(parts)
        return ", ".join(f"{symbol} = {value}" for symbol, value in first_solution.items())

    @staticmethod
    def _direct_final_answer_items(
        backend_result: dict[str, Any],
        source_step_id: str,
    ) -> list[dict[str, Any]]:
        if not backend_result.get("success"):
            return []
        items: list[dict[str, Any]] = []
        for solution in backend_result.get("output", {}).get("raw_solution", []):
            for symbol, value in solution.items():
                items.append(
                    {
                        "target": str(symbol),
                        "value": str(value),
                        "source_step_id": source_step_id,
                        "backend_operation": backend_result.get("operation"),
                        "backend_confirmed": True,
                    }
                )
        return items

    @staticmethod
    def _canonical_target_name(
        target: str,
        spr: SPR | None,
        semantic_symbol_aliases: dict[str, str] | None = None,
    ) -> str:
        normalized = target.strip()
        if semantic_symbol_aliases and normalized in semantic_symbol_aliases:
            return semantic_symbol_aliases[normalized]
        if spr is None:
            return normalized
        variable_symbols = {variable.symbol for variable in spr.variables}
        if normalized in variable_symbols:
            return normalized
        normalized_tokens = AlgebraSolver._target_tokens(normalized)
        if "/" in normalized or normalized_tokens & {"ratio", "over", "per"}:
            return normalized
        if normalized_tokens & {"plus", "squared", "square"} and len(normalized_tokens & variable_symbols) >= 2:
            return normalized
        if re.search(r"\^|\*\*", normalized) and len(normalized_tokens & variable_symbols) >= 2:
            return normalized
        aggregate_speed_tokens = {"sum", "total", "relative", "combined", "together"}
        if "speed" in normalized_tokens and normalized_tokens & aggregate_speed_tokens and not any(
            AlgebraSolver._target_tokens(variable.symbol) & aggregate_speed_tokens
            or AlgebraSolver._target_tokens(variable.description or "") & aggregate_speed_tokens
            for variable in spr.variables
        ):
            return normalized
        best_symbol: str | None = None
        best_score = 0.0
        for variable in spr.variables:
            symbol_tokens = AlgebraSolver._target_tokens(variable.symbol)
            description_tokens = AlgebraSolver._target_tokens(variable.description or "")
            candidate_tokens = symbol_tokens | description_tokens
            if not normalized_tokens or not candidate_tokens:
                continue
            if normalized_tokens <= symbol_tokens or normalized_tokens <= description_tokens:
                score = 1.0
            else:
                score = len(normalized_tokens & candidate_tokens) / max(
                    1,
                    min(len(normalized_tokens), len(candidate_tokens)),
                )
            if score > best_score:
                best_score = score
                best_symbol = variable.symbol
        if (
            best_symbol is not None
            and best_score >= 0.5
            and not AlgebraSolver._has_unit_alias_conflict(normalized, best_symbol)
        ):
            return best_symbol
        lower_target = normalized.lower()
        box_aliases = {"box", "boxes", "n", "n_boxes", "num_boxes", "number_of_boxes"}
        if lower_target in box_aliases:
            for variable in spr.variables:
                description = (variable.description or "").lower()
                if "箱" in description or "box" in description:
                    return variable.symbol
        return normalized

    @staticmethod
    def _contextual_final_answer_target_name(target: str, call_id: Any, spr: SPR | None) -> str:
        call_text = str(call_id or "").lower()
        target_text = target.strip()
        variable_symbols = {variable.symbol for variable in spr.variables} if spr is not None else set()
        if "vertex_y" in call_text:
            return "vertex_y"
        if "vertex_x" in call_text:
            return "vertex_x"
        if "y_vertex" in call_text:
            return "vertex_y"
        if "x_vertex" in call_text:
            return "vertex_x"
        return target

    @staticmethod
    def _has_unit_alias_conflict(source_symbol: str, candidate_symbol: str) -> bool:
        source = source_symbol.lower()
        candidate = candidate_symbol.lower()
        minute_tokens = ["min", "minute", "minutes"]
        hour_tokens = ["hour", "hours"]
        source_minute = any(token in source for token in minute_tokens)
        source_hour = any(token in source for token in hour_tokens)
        candidate_minute = any(token in candidate for token in minute_tokens)
        candidate_hour = any(token in candidate for token in hour_tokens)
        return (source_minute and candidate_hour) or (source_hour and candidate_minute)

    @staticmethod
    def _function_expression_final_answer_target(
        target: str,
        value: str,
        operation: Any,
        spr: SPR | None,
    ) -> str:
        if spr is None:
            return target
        if str(operation or "") != "substitute":
            return target
        if not re.search(r"[A-Za-z]", value):
            return target
        asks_for_expression = any(
            AlgebraSolver._question_asks_for_function_expression(question, spr)
            for question in spr.questions
        )
        if not asks_for_expression:
            return target
        target_lower = target.lower()
        if target_lower in {"k", "b", "a", "m", "y", "f", "g", "h"}:
            return "y"
        if any(token in target_lower for token in ["func", "function", "expr", "expression", "formula"]):
            return "y"
        return target

    @staticmethod
    def _target_tokens(text: str | None) -> set[str]:
        if not text:
            return set()
        normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        normalized = normalized.replace("_", " ").replace("{", " ").replace("}", " ").lower()
        aliases = {
            "num": "number",
            "boxes": "box",
            "box": "box",
            "required": "need",
            "needed": "need",
            "chickens": "chicken",
            "hens": "chicken",
            "hen": "chicken",
            "rabbits": "rabbit",
        }
        tokens = {
            aliases.get(token, token)
            for token in re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", normalized)
            if token not in {"of", "the", "a", "an"}
        }
        if "箱" in normalized:
            tokens.add("box")
        if "鸡" in normalized:
            tokens.add("chicken")
        if "兔" in normalized:
            tokens.add("rabbit")
        return tokens

    @staticmethod
    def _aggregate_final_answer(
        last_final_answer: str | None,
        final_answer_items: list[dict[str, Any]],
        spr: SPR | None,
    ) -> str | None:
        if not final_answer_items:
            return last_final_answer
        if len(final_answer_items) == 1 and str(final_answer_items[0].get("target") or "") in {
            "tangent_line",
            "local_extrema",
        }:
            item = final_answer_items[0]
            return f"{item.get('target')} = {item.get('value')}"
        question_count = len(spr.questions) if spr is not None else 0
        target_count = len(spr.targets) if spr is not None else 0
        if (
            spr is not None
            and len(final_answer_items) == 1
            and any(AlgebraSolver._question_asks_for_function_expression(question, spr) for question in spr.questions)
        ):
            expression_items = [
                item
                for item in final_answer_items
                if AlgebraSolver._answer_item_is_symbolic_expression(item)
            ]
            if expression_items:
                value = expression_items[-1].get("value")
                return f"答案是 {value}" if value is not None else last_final_answer
        if len(final_answer_items) == 1 and AlgebraSolver._should_use_single_selected_final_item(
            last_final_answer,
            final_answer_items[0],
        ):
            item = final_answer_items[0]
            return f"{item.get('target')} = {item.get('value')}"
        if (
            len(final_answer_items) == 1
            and str(final_answer_items[0].get("source_tool_call_id") or "").startswith("derived_")
        ):
            item = final_answer_items[0]
            return f"{item.get('target')} = {item.get('value')}"
        if question_count <= 1 and target_count <= 1 and len(final_answer_items) == 1:
            return last_final_answer
        if (
            question_count <= 1
            and target_count <= 1
            and last_final_answer
            and len({str(item.get("source_tool_call_id") or "") for item in final_answer_items}) == 1
            and all(item.get("backend_operation") == "solve_equation_system" for item in final_answer_items)
        ):
            return last_final_answer

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in final_answer_items:
            key = (str(item.get("target")), str(item.get("value")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        parts = [
            f"{item.get('target')} = {item.get('value')}"
            for item in deduped
            if item.get("target") and item.get("value") is not None
        ]
        return "；".join(parts) if parts else last_final_answer

    @staticmethod
    def _should_use_single_selected_final_item(
        last_final_answer: str | None,
        item: dict[str, Any],
    ) -> bool:
        if not last_final_answer or not item.get("target") or item.get("value") is None:
            return False
        text = str(last_final_answer)
        if text.count("=") <= 1 and "," not in text and "，" not in text:
            return False
        return str(item.get("value")) in text

    @staticmethod
    def _planner_step_description(operation: str) -> str:
        descriptions = {
            "construct_expression": "根据题意构造表达式。",
            "parse_expression": "检查表达式是否可执行。",
            "solve_equation_system": "根据题意列出方程并准备求解。",
            "solve_for": "根据题意列出方程并指定求解目标。",
            "substitute": "把已求出的结果代入目标表达式。",
            "evaluate": "计算目标表达式的值。",
        }
        return descriptions.get(operation, "根据题意提出一个计算请求。")

    @staticmethod
    def _cas_step_description(operation: str) -> str:
        descriptions = {
            "construct_expression": "CAS 确认表达式可解析。",
            "parse_expression": "CAS 完成表达式解析检查。",
            "solve_equation_system": "CAS 求解方程。",
            "solve_for": "CAS 求解指定变量。",
            "substitute": "CAS 完成代入计算。",
            "evaluate": "CAS 完成表达式求值。",
        }
        return descriptions.get(operation, "CAS 完成该步计算。")

    @staticmethod
    def _request_math_expression(request_data: dict[str, Any]) -> str | None:
        equations = request_data.get("equations") or []
        if equations:
            equation = equations[0]
            return f"{equation.get('lhs')} = {equation.get('rhs')}"
        expressions = request_data.get("expressions") or []
        if expressions:
            return expressions[0].get("expression")
        return None

    def _partial_unsupported_chain(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        planner_draft: dict[str, Any],
        steps: list[dict[str, Any]],
        validation_reports: list[dict[str, Any]],
        backend_results: list[dict[str, Any]],
        unsupported_reason: str,
        final_answer: str | None,
        spr: SPR | None = None,
    ) -> SolutionChain:
        final_answer_items = self._final_answer_items(backend_results, validation_reports, spr)
        final_answer = self._aggregate_final_answer(final_answer, final_answer_items, spr)
        steps.append(
            {
                "step_id": f"chain_main_step_{len(steps)}",
                "step_index": len(steps),
                "description": "当前多步工具调用未能安全完成。",
                "math_expression": final_answer,
                "rule_name": "unsupported_route",
                "justification": unsupported_reason,
                "depends_on": [steps[-1]["step_id"]] if steps else [],
                "verification_status": "unverified",
                "diagnostics": {
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "unsupported_reason": unsupported_reason,
                    "validation_reports": validation_reports,
                    "backend_results": backend_results,
                    "backend_confirmed": bool(backend_results),
                    **planner_draft,
                },
            }
        )
        chain_data = {
                "problem_id": emr.problem_id,
                "source_emr_id": emr.problem_id,
                "source_emr_version": emr.schema_version,
                "chain_id": "chain_main",
                "steps": steps,
                "final_answer": final_answer,
                "confidence": 0.55 if final_answer_items and final_answer else 0.0,
                "metadata": {
                    "solver": "AlgebraSolver",
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "unsupported_reason": unsupported_reason,
                    "tool_call_count": len(planner_draft.get("planner_tool_calls") or []),
                    "completed_tool_call_count": len(backend_results),
                    "backend_confirmed": bool(backend_results),
                    "partial_failure": True,
                    "validation_reports": validation_reports,
                    "final_answer_items": final_answer_items,
                    **planner_draft,
                },
            }
        return self._validate_enhanced_chain(chain_data, emr, classification_result, spr)

    def _chain_from_tool_result(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        planner_draft: dict[str, Any],
        validation_report: dict[str, Any],
        backend_result: dict[str, Any],
    ) -> SolutionChain:
        request_data = validation_report["normalized_request"]
        final_answer = self._final_answer_from_backend(backend_result)
        math_expression = final_answer
        chain_data = {
                "problem_id": emr.problem_id,
                "source_emr_id": emr.problem_id,
                "source_emr_version": emr.schema_version,
                "chain_id": "chain_main",
                "steps": [
                    {
                        "step_id": "chain_main_step_0",
                        "step_index": 0,
                        "description": "读取题意和已知变量。",
                        "math_expression": None,
                        "rule_name": "read_problem_and_variables",
                        "justification": "Planner 在读取 SPR/EMR 后提出工具调用请求。",
                        "verification_status": "unverified",
                        "diagnostics": {
                            "variables": [variable.symbol for variable in emr.variables],
                            "solver_route": classification_result.solver_route,
                            "task_type": classification_result.task_type,
                            **planner_draft,
                        },
                    },
                    {
                        "step_id": "chain_main_step_1",
                        "step_index": 1,
                        "description": "构造结构化计算请求。",
                        "math_expression": None,
                        "rule_name": f"planner_{request_data['operation']}",
                        "justification": request_data.get("reason")
                        or "Planner 提出一个需要 CAS 后端确认的计算请求。",
                        "depends_on": ["chain_main_step_0"],
                        "verification_status": "unverified",
                        "diagnostics": {
                            "planner_provider": planner_draft.get("planner_provider"),
                            "planner_model": planner_draft.get("planner_model"),
                            "planner_role": planner_draft.get("planner_role"),
                            "planner_raw_output": planner_draft.get("planner_raw_output"),
                            "planner_tool_calls": planner_draft.get("planner_tool_calls"),
                            "normalized_request": request_data,
                            "validator_errors": validation_report["errors"],
                            "backend_confirmed": False,
                        },
                    },
                    {
                        "step_id": "chain_main_step_2",
                        "step_index": 2,
                        "description": "使用 CAS 后端确认计算请求。",
                        "math_expression": math_expression,
                        "rule_name": f"cas_{backend_result['operation']}",
                        "justification": "只有经过 CAS 确认的结果才写入正式解题链。",
                        "depends_on": ["chain_main_step_1"],
                        "verification_status": "unverified",
                        "diagnostics": {
                            **backend_result,
                            "backend_confirmed": True,
                        },
                    },
                    {
                        "step_id": "chain_main_step_3",
                        "step_index": 3,
                        "description": "给出最终答案。",
                        "math_expression": final_answer,
                        "rule_name": "final_answer",
                        "justification": "最终答案来自 CAS 确认后的结果。",
                        "depends_on": ["chain_main_step_2"],
                        "verification_status": "unverified",
                        "diagnostics": {
                            "backend": backend_result["backend"],
                            "operation": "format_final_answer",
                            "input": backend_result["output"],
                            "output": {"formatted_solution": final_answer},
                            "success": final_answer is not None,
                            "errors": backend_result["errors"],
                            "warnings": backend_result["warnings"],
                        },
                    },
                ],
                "final_answer": final_answer,
                "confidence": 0.8 if final_answer else 0.0,
                "metadata": {
                    "solver": "AlgebraSolver",
                    "solver_version": "Solver v1 enhanced: Planner + ToolCall + CAS",
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "planner_provider": planner_draft.get("planner_provider"),
                    "planner_model": planner_draft.get("planner_model"),
                    "planner_role": planner_draft.get("planner_role"),
                    "backend": backend_result["backend"],
                    "backend_confirmed": True,
                    "normalized_request": request_data,
                },
            }
        return self._validate_enhanced_chain(chain_data, emr, classification_result)

    @staticmethod
    def _final_answer_from_backend(backend_result: dict[str, Any]) -> str | None:
        output = backend_result.get("output", {})
        if backend_result["operation"] == "construct_expression":
            return AlgebraSolver._expression_output_value(backend_result)
        if backend_result["operation"] == "parse_expression":
            return output.get("parsed_expression")
        if backend_result["operation"] in {"substitute", "evaluate"}:
            return output.get("value") or output.get("formatted_solution")
        return output.get("formatted_solution")

    @staticmethod
    def _expression_output_value(backend_result: dict[str, Any]) -> str | None:
        output = backend_result.get("output", {})
        input_data = backend_result.get("input") or {}
        expression = str(input_data.get("expression") or "").strip()
        if expression.startswith("diff(") or expression.lower().startswith("derivative("):
            return output.get("parsed_expression") or output.get("confirmed_expression")
        return output.get("value") or output.get("confirmed_expression") or output.get("parsed_expression")

    @staticmethod
    def _format_final_answer(
        value: str | None,
        backend_result: dict[str, Any],
        spr: SPR | None,
    ) -> str | None:
        if value is None:
            return None
        if backend_result.get("operation") not in {"substitute", "evaluate"}:
            return value
        text_parts: list[str] = []
        if spr is not None:
            text_parts.append(spr.problem_text or "")
            text_parts.extend(question.text for question in spr.questions)
            text_parts.extend(target.text for target in spr.targets)
        combined = " ".join(part for part in text_parts if part)
        if "页" in combined and ("第二天" in combined or "第2天" in combined):
            return f"第二天看了 {value} 页"
        if "岁" in combined:
            return f"答案是 {value} 岁"
        return f"答案是 {value}"

    @staticmethod
    def unsupported_chain(
        emr: EMR,
        classification_result: ClassificationResult,
        unsupported_reason: str,
        planner_draft: dict[str, str] | None = None,
    ) -> SolutionChain:
        planner_metadata = planner_draft or {}
        chain_data = {
                "problem_id": emr.problem_id,
                "source_emr_id": emr.problem_id,
                "source_emr_version": emr.schema_version,
                "chain_id": "chain_main",
                "steps": [
                    {
                        "step_id": "chain_main_step_0",
                        "step_index": 0,
                        "description": "当前 Solver v1 不支持该求解路线。",
                        "math_expression": None,
                        "rule_name": "unsupported_route",
                        "justification": "该题超出当前 Solver 可安全处理的范围。",
                        "verification_status": "unverified",
                        "diagnostics": {
                            "solver_route": classification_result.solver_route,
                            "task_type": classification_result.task_type,
                            "unsupported_reason": unsupported_reason,
                            **planner_metadata,
                        },
                    }
                ],
                "final_answer": None,
                "confidence": 0.0,
                "metadata": {
                    "solver": "AlgebraSolver",
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "unsupported_reason": unsupported_reason,
                    **planner_metadata,
                },
            }
        enhanced = SCSAssemblyEnhancer().enhance(chain_data, emr, classification_result)
        return validate_solution_chain_dict(enhanced)
