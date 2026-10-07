"""Verifier: CAS replay, provenance checking, and downstream validation support."""

from __future__ import annotations

import re
from typing import Any

from sympy import Symbol, simplify
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from mathexplain.agents.solver.calculation_request import CalculationRequestValidator
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain, validate_solution_chain_dict
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport
from mathexplain.services.cas import CASService


class Verifier:
    """Verify one Solver-produced SCS through deterministic CAS replay."""

    _TRANSFORMATIONS = standard_transformations + (convert_xor,)

    def __init__(self, cas_service: CASService | None = None) -> None:
        self.cas_service = cas_service or CASService()

    def verify(
        self,
        scs: SolutionChain,
        spr: SPR | None = None,
        emr: EMR | None = None,
        classification_result: ClassificationResult | None = None,
    ) -> VerificationReport:
        """Verify a single solution chain.

        Verifier treats Solver ToolCall traces as executable mathematical
        representations when EMR is unknown or empty. It does not prove that the
        planner's equations fully match the original natural-language problem.
        """

        errors: list[str] = []
        warnings: list[str] = []
        checked_steps: list[dict[str, Any]] = []

        scs_schema_valid = self._validate_scs_schema(scs, errors)
        mode = self._verification_mode(scs, emr, classification_result)
        if mode == "unsupported":
            reason = self._unsupported_reason(scs, classification_result)
            errors.append(reason)
            return self._report(
                scs,
                mode,
                False,
                "unsupported",
                scs_schema_valid,
                False,
                False,
                checked_steps,
                errors,
                warnings,
                "当前 Verifier v1 只支持 CAS-backed 单链验证；该求解链未进入可安全验证路径。",
                metadata=self._metadata(scs, spr, emr, classification_result, repair_performed=False),
            )

        backend_steps_valid = True
        final_answer_valid = True
        if mode == "tool_call_backed":
            replay_report = self._verify_tool_call_trace(scs, spr)
            checked_steps.extend(replay_report["checked_steps"])
            warnings.extend(replay_report["warnings"])
            errors.extend(replay_report["errors"])
            backend_steps_valid = replay_report["valid"]
        else:
            replay_report = self._verify_emr_backed_chain(scs)
            checked_steps.extend(replay_report["checked_steps"])
            errors.extend(replay_report["errors"])
            warnings.extend(replay_report["warnings"])
            backend_steps_valid = replay_report["valid"]

        explanatory_report = self._verify_explanatory_steps(scs, spr)
        checked_steps.extend(explanatory_report["checked_steps"])
        warnings.extend(explanatory_report["warnings"])
        errors.extend(explanatory_report["errors"])

        final_report = self._verify_final_answer_items(scs)
        checked_steps.extend(final_report["checked_steps"])
        errors.extend(final_report["errors"])
        warnings.extend(final_report["warnings"])
        final_answer_valid = final_report["valid"]

        if not scs.final_answer:
            final_answer_valid = False
            errors.append("SCS final_answer is missing.")

        explanatory_steps_valid = explanatory_report["valid"]
        valid = scs_schema_valid and backend_steps_valid and final_answer_valid and explanatory_steps_valid
        status = "passed" if valid else "failed"
        summary = (
            "验证通过：给定 Solver 生成的 ToolCall 表示，CAS 回放结果与 SCS 记录一致，"
            "最终答案可追溯到后端确认结果。"
            if valid
            else "验证未通过：Verifier v1 在 SCS 结构、CAS 回放或最终答案来源中发现问题。"
        )
        return self._report(
            scs,
            mode,
            valid,
            status,
            scs_schema_valid,
            backend_steps_valid,
            final_answer_valid,
            checked_steps,
            errors,
            warnings,
            summary,
            metadata={
                **self._metadata(scs, spr, emr, classification_result, repair_performed=False),
                "explanatory_steps_checked": explanatory_report["checked_count"],
            },
        )

    @staticmethod
    def _validate_scs_schema(scs: SolutionChain, errors: list[str]) -> bool:
        try:
            validate_solution_chain_dict(scs.model_dump(mode="json"))
        except Exception as exc:  # pragma: no cover - pydantic error shape can vary.
            errors.append(f"SCS schema validation failed: {exc}")
            return False
        if not scs.steps:
            errors.append("SCS contains no solution steps.")
            return False
        return True

    @staticmethod
    def _verification_mode(
        scs: SolutionChain,
        emr: EMR | None,
        classification_result: ClassificationResult | None,
    ) -> str:
        if any(step.rule_name and step.rule_name.startswith("planner_") for step in scs.steps):
            return "tool_call_backed"
        if any(step.rule_name and step.rule_name.startswith("cas_") for step in scs.steps):
            return "emr_backed"
        route = classification_result.solver_route if classification_result is not None else scs.metadata.get("solver_route")
        if route in {"algebra_solver", "algebra_word_problem_solver"} and emr is not None and emr.equations:
            return "emr_backed"
        return "unsupported"

    @staticmethod
    def _unsupported_reason(
        scs: SolutionChain,
        classification_result: ClassificationResult | None,
    ) -> str:
        if any(step.rule_name == "unsupported_route" for step in scs.steps):
            return "SCS is an unsupported_route chain."
        route = classification_result.solver_route if classification_result is not None else scs.metadata.get("solver_route")
        return f"Verifier v1 does not support solver_route={route!r} without CAS-backed steps."

    def _verify_tool_call_trace(
        self,
        scs: SolutionChain,
        spr: SPR | None,
    ) -> dict[str, Any]:
        checked_steps: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        valid = True
        context: dict[str, Any] = {"variables": {}}
        cas_steps_by_call_id = self._cas_steps_by_tool_call_id(scs)

        for planner_step in self._planner_steps(scs):
            diagnostics = planner_step.diagnostics
            request = diagnostics.get("normalized_request") or {}
            call_id = diagnostics.get("tool_call_id") or request.get("id")
            operation = request.get("operation")
            if not call_id:
                valid = False
                message = f"Planner step {planner_step.step_id} is missing tool_call_id."
                errors.append(message)
                checked_steps.append(self._checked_step(planner_step.step_id, None, operation, "failed", False, message))
                continue
            cas_step = cas_steps_by_call_id.get(call_id)
            if cas_step is None:
                valid = False
                message = f"No CAS step was found for tool_call_id={call_id}."
                errors.append(message)
                checked_steps.append(self._checked_step(planner_step.step_id, call_id, operation, "failed", False, message))
                continue

            spr_warnings = self._spr_constant_warnings(request, spr)
            warnings.extend(spr_warnings)
            replay_result = self._execute_request(request, context)
            step_valid, mismatch_errors = self._compare_backend_result(replay_result, cas_step.diagnostics)
            if replay_result.get("success"):
                self._store_tool_result(call_id, replay_result, context, request)

            if not step_valid:
                valid = False
                errors.extend(mismatch_errors)
            checked_steps.append(
                self._checked_step(
                    cas_step.step_id,
                    call_id,
                    operation,
                    "passed" if step_valid else "failed",
                    True,
                    "CAS replay matched recorded result." if step_valid else "CAS replay did not match recorded result.",
                    errors=mismatch_errors,
                    warnings=spr_warnings,
                    expected=self._comparison_view(cas_step.diagnostics),
                    observed=self._comparison_view(replay_result),
                    metadata={"planner_step_id": planner_step.step_id},
                )
            )
            substitution_report = self._verify_equation_back_substitution(cas_step.diagnostics)
            checked_steps.extend(substitution_report["checked_steps"])
            warnings.extend(substitution_report["warnings"])
            if not substitution_report["valid"]:
                valid = False
                errors.extend(substitution_report["errors"])

        if not checked_steps:
            valid = False
            errors.append("No planner/CAS ToolCall trace was available for replay.")
        return {"valid": valid, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}

    def _verify_emr_backed_chain(self, scs: SolutionChain) -> dict[str, Any]:
        checked_steps: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        valid = True
        for step in scs.steps:
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            diagnostics = step.diagnostics
            replay_result = self._execute_backend_input(diagnostics)
            step_valid, mismatch_errors = self._compare_backend_result(replay_result, diagnostics)
            if not step_valid:
                valid = False
                errors.extend(mismatch_errors)
            checked_steps.append(
                self._checked_step(
                    step.step_id,
                    diagnostics.get("tool_call_id"),
                    diagnostics.get("operation"),
                    "passed" if step_valid else "failed",
                    True,
                    "CAS replay matched recorded result." if step_valid else "CAS replay did not match recorded result.",
                    errors=mismatch_errors,
                    expected=self._comparison_view(diagnostics),
                    observed=self._comparison_view(replay_result),
                )
            )
            substitution_report = self._verify_equation_back_substitution(diagnostics)
            checked_steps.extend(substitution_report["checked_steps"])
            warnings.extend(substitution_report["warnings"])
            if not substitution_report["valid"]:
                valid = False
                errors.extend(substitution_report["errors"])
        if not checked_steps:
            valid = False
            errors.append("No CAS-backed step was available for EMR-backed replay.")
        return {"valid": valid, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}

    def _verify_equation_back_substitution(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        operation = diagnostics.get("operation")
        if operation not in {"solve_equation_system", "solve_for"}:
            return {"valid": True, "checked_steps": [], "errors": [], "warnings": []}
        if diagnostics.get("success") is not True:
            return {"valid": True, "checked_steps": [], "errors": [], "warnings": []}

        input_data = diagnostics.get("input") or {}
        output = diagnostics.get("output") or {}
        equations = input_data.get("equations") or []
        variables = input_data.get("variables") or []
        raw_solution = output.get("raw_solution") or []
        checked_steps: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        if not equations or not raw_solution:
            return {"valid": True, "checked_steps": [], "errors": [], "warnings": []}

        valid = True
        for solution_index, solution in enumerate(raw_solution):
            for equation_index, equation in enumerate(equations):
                lhs = str(equation.get("lhs_sympy") or equation.get("lhs") or "")
                rhs = str(equation.get("rhs_sympy") or equation.get("rhs") or "")
                difference = f"({lhs}) - ({rhs})"
                result = self.cas_service.evaluate_expression(
                    expression=difference,
                    variables=variables,
                    substitutions={str(symbol): str(value) for symbol, value in solution.items()},
                    target="equation_residual",
                )
                value = (result.get("output") or {}).get("value")
                passed = result.get("success") is True and self._is_zero_residual(value, variables)
                step_errors: list[str] = []
                if not passed:
                    valid = False
                    message = (
                        f"Equation back-substitution failed for equation {equation_index + 1} "
                        f"and solution {solution_index + 1}."
                    )
                    step_errors.append(message)
                    if result.get("errors"):
                        step_errors.extend(str(error) for error in result.get("errors") or [])
                    errors.extend(step_errors)
                checked_steps.append(
                    self._checked_step(
                        f"equation_back_substitution_{solution_index}_{equation_index}",
                        diagnostics.get("tool_call_id"),
                        "equation_back_substitution",
                        "passed" if passed else "failed",
                        True,
                        "CAS back-substitution confirmed the solution satisfies the equation."
                        if passed
                        else "CAS back-substitution found the solution does not satisfy the equation.",
                        errors=step_errors,
                        expected={"residual": "0", "equation": equation, "solution": solution},
                        observed={"residual": value, "cas_result": result},
                        metadata={
                            "source_operation": operation,
                            "source_step_id": diagnostics.get("step_id"),
                            "solution_index": solution_index,
                            "equation_index": equation_index,
                        },
                    )
                )
        return {"valid": valid, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}

    def _is_zero_residual(self, value: Any, variables: list[str]) -> bool:
        if value is None:
            return False
        if str(value).strip() == "0":
            return True
        result = self.cas_service.evaluate_expression(
            expression=str(value),
            variables=variables,
            substitutions={},
            target="equation_residual",
        )
        simplified = (result.get("output") or {}).get("value")
        if result.get("success") is True and str(simplified).strip() == "0":
            return True
        try:
            local_dict = {symbol: Symbol(symbol) for symbol in variables}
            parsed = parse_expr(
                str(value),
                local_dict=local_dict,
                transformations=self._TRANSFORMATIONS,
                evaluate=True,
            )
            return simplify(parsed) == 0
        except Exception:
            return False

    def _verify_explanatory_steps(
        self,
        scs: SolutionChain,
        spr: SPR | None,
    ) -> dict[str, Any]:
        checked_steps: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        valid = True
        variable_symbols = self._scs_variable_symbols(scs, spr)
        for step in scs.steps:
            diagnostics = step.diagnostics or {}
            math_role = diagnostics.get("math_role")
            if math_role not in {"modeling", "reasoning_step", "variable_definition", "answer_interpretation"}:
                continue
            step_warnings: list[str] = []
            if diagnostics.get("explainer_ready") is True and not diagnostics.get("source"):
                step_warnings.append(f"Explainer-ready step {step.step_id} is missing diagnostics.source.")
            if math_role in {"modeling", "reasoning_step"}:
                if diagnostics.get("source") == "planner_tool_call" and not diagnostics.get("tool_call_id"):
                    step_warnings.append(f"Modeling step {step.step_id} is missing tool_call_id.")
                parse_warnings = self._modeling_expression_warnings(step, variable_symbols)
                step_warnings.extend(parse_warnings)
                request = diagnostics.get("normalized_request") or {}
                if request:
                    step_warnings.extend(self._spr_constant_warnings(request, spr))
            warnings.extend(step_warnings)
            checked_steps.append(
                self._checked_step(
                    step.step_id,
                    diagnostics.get("tool_call_id"),
                    "explanatory_structure",
                    "passed",
                    False,
                    "Explanatory step has traceable structure; this is not a full semantic proof.",
                    warnings=step_warnings,
                    expected={
                        "math_role": math_role,
                        "source": diagnostics.get("source"),
                        "explainer_ready": diagnostics.get("explainer_ready"),
                    },
                )
            )
        return {
            "valid": valid,
            "checked_steps": checked_steps,
            "errors": errors,
            "warnings": warnings,
            "checked_count": len(checked_steps),
        }

    def _verify_final_answer_items(self, scs: SolutionChain) -> dict[str, Any]:
        checked_steps: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        final_answer_items = scs.metadata.get("final_answer_items") or []
        cas_steps_by_call_id = self._cas_steps_by_tool_call_id(scs)
        cas_steps_by_step_id = self._cas_steps_by_step_id(scs)
        if not final_answer_items:
            has_planner_trace = any(
                step.rule_name and step.rule_name.startswith("planner_")
                for step in scs.steps
            )
            direct_cas_steps = [
                step
                for step in scs.steps
                if step.rule_name
                and step.rule_name.startswith("cas_")
                and step.diagnostics.get("success") is True
            ]
            if not has_planner_trace and direct_cas_steps and scs.final_answer:
                checked_steps.append(
                    self._checked_step(
                        "final_answer",
                        None,
                        "final_answer",
                        "passed",
                        False,
                        "Final answer is backed by a successful direct CAS step.",
                        expected={"final_answer": scs.final_answer},
                        metadata={"source_step_id": direct_cas_steps[-1].step_id},
                    )
                )
                return {"valid": True, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}
            message = "metadata.final_answer_items is missing or empty."
            errors.append(message)
            checked_steps.append(self._checked_step("final_answer", None, "final_answer", "failed", False, message))
            return {"valid": False, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}

        valid = True
        for index, item in enumerate(final_answer_items):
            item_errors: list[str] = []
            call_id = item.get("source_tool_call_id")
            source_step_id = item.get("source_step_id")
            if not item.get("target"):
                item_errors.append("final_answer_item.target is missing.")
            if item.get("value") is None:
                item_errors.append("final_answer_item.value is missing.")
            elif (
                not self._final_answer_item_allows_symbolic_value(item)
                and self._final_answer_value_has_unresolved_symbols(str(item.get("value")), scs)
            ):
                item_errors.append("final_answer_item.value contains unresolved symbols.")
            if not call_id and not source_step_id:
                item_errors.append("final_answer_item source must include source_tool_call_id or source_step_id.")
            if item.get("backend_confirmed") is not True:
                item_errors.append("final_answer_item.backend_confirmed must be true.")
            if call_id and call_id not in cas_steps_by_call_id:
                item_errors.append(f"No successful CAS step found for source_tool_call_id={call_id}.")
            if source_step_id and source_step_id not in cas_steps_by_step_id:
                item_errors.append(f"No successful CAS step found for source_step_id={source_step_id}.")
            if call_id and call_id in cas_steps_by_call_id:
                cas_step = cas_steps_by_call_id[call_id]
                cas_operation = cas_step.diagnostics.get("operation")
                if item.get("backend_operation") and item.get("backend_operation") != cas_operation:
                    item_errors.append(
                        "final_answer_item.backend_operation does not match source CAS step operation."
                    )
                if cas_step.diagnostics.get("success") is not True:
                    item_errors.append(f"Source CAS step {cas_step.step_id} was not successful.")
            if source_step_id and source_step_id in cas_steps_by_step_id:
                cas_step = cas_steps_by_step_id[source_step_id]
                cas_operation = cas_step.diagnostics.get("operation")
                if item.get("backend_operation") and item.get("backend_operation") != cas_operation:
                    item_errors.append(
                        "final_answer_item.backend_operation does not match source CAS step operation."
                    )
                if cas_step.diagnostics.get("success") is not True:
                    item_errors.append(f"Source CAS step {cas_step.step_id} was not successful.")

            if item_errors:
                valid = False
                errors.extend(item_errors)
            checked_steps.append(
                self._checked_step(
                    f"final_answer_item_{index}",
                    call_id,
                    item.get("backend_operation"),
                    "passed" if not item_errors else "failed",
                    False,
                    "Final answer item is traceable to a successful CAS step."
                    if not item_errors
                    else "Final answer item provenance is invalid.",
                    errors=item_errors,
                    expected={"final_answer_item": item},
                    metadata={"source_step_id": source_step_id} if source_step_id else {},
                )
            )
        return {"valid": valid, "checked_steps": checked_steps, "errors": errors, "warnings": warnings}

    def _execute_request(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        variables = request.get("effective_variables") or request.get("variables") or []
        if operation in {"construct_expression", "parse_expression"}:
            expression = self._resolve_expression_reference(
                (request.get("expressions") or [{}])[0].get("expression", ""),
                context,
            )
            substitutions = self._resolve_substitutions(request.get("substitutions") or {}, context)
            if substitutions:
                expression = self._apply_substitutions_to_statement(
                    expression,
                    sorted(set(variables) | set(substitutions)),
                    substitutions,
                    request.get("target"),
                )
                variables = sorted(set(variables) | set(self._symbols_from_statement(expression)))
            result = self.cas_service.construct_expression(
                expression=expression,
                variables=variables,
                target=request.get("target"),
            )
            return self._attach_numeric_value_to_construct_expression(
                result,
                variables,
                request.get("target"),
            )
        if operation in {"solve_equation_system", "solve_for"}:
            equation_substitutions = self._resolve_substitutions(request.get("substitutions") or {}, context)
            equations = [
                {
                    "id": equation.get("id") or f"tool_eq_{index + 1}",
                    "lhs_sympy": self._apply_substitutions_to_expression(
                        self._resolve_expression_reference(equation.get("lhs", ""), context),
                        variables,
                        equation_substitutions,
                        request.get("target"),
                    ),
                    "rhs_sympy": self._apply_substitutions_to_expression(
                        self._resolve_expression_reference(equation.get("rhs", ""), context),
                        variables,
                        equation_substitutions,
                        request.get("target"),
                    ),
                }
                for index, equation in enumerate(request.get("equations") or [])
            ]
            if operation == "solve_for":
                if len(equations) > 1:
                    return self.cas_service.solve_equation_system(
                        equations=equations,
                        variables=variables,
                        target=request.get("target"),
                    )
                return self.cas_service.solve_for(
                    equations=equations,
                    variables=variables,
                    target=request.get("target") or (variables[0] if variables else ""),
                )
            return self.cas_service.solve_equation_system(
                equations=equations,
                variables=variables,
                target=request.get("target"),
            )
        if operation in {"substitute", "evaluate"}:
            expression = self._resolve_expression_reference(
                (request.get("expressions") or [{}])[0].get("expression", ""),
                context,
            )
            substitutions = self._resolve_substitutions(request.get("substitutions") or {}, context)
            if operation == "substitute" and expression.count("=") == 1:
                variables = sorted(set(variables) | set(substitutions) | set(self._symbols_from_statement(expression)))
            if operation == "evaluate":
                substitutions = self._complete_substitutions_from_context(
                    expression,
                    variables,
                    substitutions,
                    context,
                )
            if operation == "substitute":
                return self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.get("target"),
                )
            if expression.count("=") == 1:
                result = self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.get("target"),
                )
                result["operation"] = "evaluate"
                return result
            result = self.cas_service.evaluate_expression(
                expression=expression,
                variables=variables,
                substitutions=substitutions,
                target=request.get("target"),
            )
            if self._evaluate_failed_only_because_symbolic(result):
                return self.cas_service.substitute(
                    expression=expression,
                    variables=variables,
                    substitutions=substitutions,
                    target=request.get("target"),
                )
            return result
        return {
            "backend": self.cas_service.backend_name,
            "operation": operation,
            "input": request,
            "output": {},
            "success": False,
            "errors": [f"Operation {operation!r} is not supported by Verifier v1."],
            "warnings": [],
        }

    def _complete_substitutions_from_context(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        context: dict[str, Any],
    ) -> dict[str, str]:
        parse_report = self.cas_service.parse_expression(expression, variables)
        if not parse_report.get("success"):
            return substitutions
        completed = dict(substitutions)
        context_variables = context.get("variables") or {}
        for symbol in parse_report.get("free_symbols", []):
            if symbol in completed:
                continue
            if symbol in context_variables:
                completed[symbol] = str(context_variables[symbol])
        return completed

    def _final_answer_value_has_unresolved_symbols(self, value: str, scs: SolutionChain) -> bool:
        variable_symbols = self._scs_variable_symbols(scs, None)
        if variable_symbols:
            parse_report = self.cas_service.parse_expression(value, variable_symbols)
            if parse_report.get("success"):
                free_symbols = set(parse_report.get("free_symbols") or [])
                return bool(free_symbols - self._allowed_symbolic_constants())
        unresolved = set(re.findall(r"\b[A-Za-z_]\w*\b", value))
        return bool(unresolved - self._allowed_symbolic_constants())

    @staticmethod
    def _allowed_symbolic_constants() -> set[str]:
        return {"e", "E", "pi", "Pi"}

    @staticmethod
    def _final_answer_item_allows_symbolic_value(item: dict[str, Any]) -> bool:
        operation = str(item.get("backend_operation") or "")
        if operation in {"construct_expression", "parse_expression"}:
            return True
        target = str(item.get("target") or "").lower()
        if target in {"tangent_line", "local_extrema"}:
            return True
        if operation == "substitute" and target in {"y", "f", "g", "h"}:
            return True
        if operation in {"substitute", "evaluate", "construct_expression", "parse_expression"} and target in {
            "axis",
            "axis_of_symmetry",
            "line",
            "equation",
        }:
            return True
        return any(
            token in target
            for token in (
                "axis",
                "expr",
                "expression",
                "formula",
                "function",
                "equation",
                "prime",
                "double",
                "second",
                "derivative",
                "解析式",
            )
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

    def _execute_backend_input(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        operation = diagnostics.get("operation")
        input_data = diagnostics.get("input") or {}
        if operation in {"construct_expression", "parse_expression"}:
            result = self.cas_service.construct_expression(
                expression=input_data.get("expression", ""),
                variables=input_data.get("variables") or [],
                target=input_data.get("target"),
            )
            return self._attach_numeric_value_to_construct_expression(
                result,
                input_data.get("variables") or [],
                input_data.get("target"),
            )
        if operation == "solve_for":
            return self.cas_service.solve_for(
                equations=input_data.get("equations") or [],
                variables=input_data.get("variables") or [],
                target=input_data.get("target") or "",
            )
        if operation == "solve_equation_system":
            return self.cas_service.solve_equation_system(
                equations=input_data.get("equations") or [],
                variables=input_data.get("variables") or [],
                target=input_data.get("target"),
            )
        if operation == "substitute":
            return self.cas_service.substitute(
                expression=input_data.get("expression", ""),
                variables=input_data.get("variables") or [],
                substitutions=input_data.get("substitutions") or {},
                target=input_data.get("target"),
            )
        if operation == "evaluate":
            return self.cas_service.evaluate_expression(
                expression=input_data.get("expression", ""),
                variables=input_data.get("variables") or [],
                substitutions=input_data.get("substitutions") or {},
                target=input_data.get("target"),
            )
        return {
            "backend": self.cas_service.backend_name,
            "operation": operation,
            "input": input_data,
            "output": {},
            "success": False,
            "errors": [f"Operation {operation!r} is not supported by Verifier v1."],
            "warnings": [],
        }

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
        if expression is None or output.get("value") is not None or output.get("free_symbols"):
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
    def _compare_backend_result(observed: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if observed.get("success") != expected.get("success"):
            errors.append("Replay success flag does not match recorded CAS step.")
        expected_view = Verifier._comparison_view(expected)
        observed_view = Verifier._comparison_view(observed)
        if expected_view.get("operation") != observed_view.get("operation"):
            errors.append("Replay operation does not match recorded CAS step.")
        if expected_view.get("output") != observed_view.get("output") and not Verifier._expected_is_filtered_solution_subset(
            expected,
            observed,
        ):
            errors.append("Replay output does not match recorded CAS step.")
        return not errors, errors

    @staticmethod
    def _expected_is_filtered_solution_subset(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
        if expected.get("operation") not in {"solve_for", "solve_equation_system"}:
            return False
        expected_output = expected.get("output") or {}
        observed_output = observed.get("output") or {}
        if not expected_output.get("solution_filter"):
            return False
        expected_solutions = expected_output.get("raw_solution") or []
        observed_solutions = observed_output.get("raw_solution") or []
        if not expected_solutions or not observed_solutions:
            return False
        observed_keys = {Verifier._solution_key(solution) for solution in observed_solutions}
        return all(Verifier._solution_key(solution) in observed_keys for solution in expected_solutions)

    @staticmethod
    def _solution_key(solution: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(symbol), str(value)) for symbol, value in solution.items()))

    @staticmethod
    def _comparison_view(result: dict[str, Any]) -> dict[str, Any]:
        output = result.get("output") or {}
        operation = result.get("operation")
        if operation in {"solve_for", "solve_equation_system"}:
            comparable_output = {
                "formatted_solution": output.get("formatted_solution"),
                "raw_solution": output.get("raw_solution"),
            }
        elif operation in {"substitute", "evaluate"}:
            comparable_output = {
                "value": output.get("value"),
                "formatted_solution": output.get("formatted_solution"),
            }
        elif operation in {"construct_expression", "parse_expression"}:
            comparable_output = {
                "confirmed_expression": output.get("confirmed_expression"),
                "parsed_expression": output.get("parsed_expression"),
                "free_symbols": output.get("free_symbols"),
            }
        else:
            comparable_output = output
        return {
            "operation": operation,
            "success": result.get("success"),
            "output": comparable_output,
        }

    def _modeling_expression_warnings(self, step: Any, variable_symbols: list[str]) -> list[str]:
        expression = step.math_expression
        if not expression or "$" in expression:
            return []
        warnings: list[str] = []
        for index, part in enumerate(self._expression_parts(expression)):
            parse_report = self.cas_service.parse_expression(part, variable_symbols)
            if not parse_report.get("success"):
                warnings.append(
                    f"Modeling step {step.step_id} expression part {index + 1} could not be parsed: {part}."
                )
        return warnings

    @staticmethod
    def _expression_parts(expression: str) -> list[str]:
        parts: list[str] = []
        for segment in expression.split(";"):
            segment = segment.strip()
            if not segment:
                continue
            if "=" in segment:
                lhs, rhs = segment.split("=", 1)
                parts.extend([lhs.strip(), rhs.strip()])
            else:
                parts.append(segment)
        return [part for part in parts if part]

    @staticmethod
    def _scs_variable_symbols(scs: SolutionChain, spr: SPR | None) -> list[str]:
        symbols = set((scs.metadata.get("variable_bindings") or {}).keys())
        if spr is not None:
            symbols.update(variable.symbol for variable in spr.variables if variable.symbol)
        for step in scs.steps:
            diagnostics = step.diagnostics or {}
            for key in ("effective_variables", "planner_variables", "variables"):
                values = diagnostics.get(key) or []
                if isinstance(values, list):
                    symbols.update(str(value) for value in values)
            request = diagnostics.get("normalized_request") or {}
            values = request.get("effective_variables") or request.get("variables") or []
            if isinstance(values, list):
                symbols.update(str(value) for value in values)
        return sorted(symbols)

    @staticmethod
    def _planner_steps(scs: SolutionChain) -> list[Any]:
        return [step for step in scs.steps if step.rule_name and step.rule_name.startswith("planner_")]

    @staticmethod
    def _cas_steps_by_tool_call_id(scs: SolutionChain) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for step in scs.steps:
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            call_id = step.diagnostics.get("tool_call_id")
            if call_id and step.diagnostics.get("success") is True:
                result[str(call_id)] = step
        return result

    @staticmethod
    def _cas_steps_by_step_id(scs: SolutionChain) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for step in scs.steps:
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            if step.diagnostics.get("success") is True:
                result[step.step_id] = step
        return result

    @staticmethod
    def _resolve_substitutions(substitutions: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for symbol, value in substitutions.items():
            if isinstance(value, str) and value.startswith("$"):
                reference_value = CalculationRequestValidator.resolve_reference(value, context)
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
        formatted_values = Verifier._formatted_solution_mapping(output.get("formatted_solution"))
        raw_solution_symbols: set[str] = set()
        for solution in output.get("raw_solution", []):
            for symbol, value in solution.items():
                symbol_text = str(symbol)
                stored_value = Verifier._preferred_context_solution_value(
                    symbol_text,
                    str(value),
                    formatted_values,
                )
                raw_solution_symbols.add(symbol_text)
                execution_context.setdefault("variables", {})[symbol_text] = stored_value
                stored[symbol_text] = stored_value
        if "value" in output:
            stored["value"] = output["value"]
        if "formatted_solution" in output:
            stored["formatted_solution"] = output["formatted_solution"]
        if "confirmed_expression" in output:
            stored["confirmed_expression"] = output["confirmed_expression"]
        scalar_value = (
            output.get("value")
            or output.get("formatted_solution")
            or output.get("confirmed_expression")
        )
        target = (request_data or {}).get("target")
        if target and scalar_value is not None and str(target) not in raw_solution_symbols:
            Verifier._store_symbol_aliases(stored, str(target), scalar_value)
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
                Verifier._store_symbol_aliases(stored, str(expression_name), expression_value)
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
        for alias in Verifier._latex_symbol_aliases(symbol):
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
            context_expression = Verifier._context_expression_for_name(value, execution_context)
            return context_expression if context_expression is not None else value

        def replace(match: re.Match[str]) -> str:
            reference = match.group(0)
            resolved = CalculationRequestValidator.resolve_reference(reference, execution_context)
            return CalculationRequestValidator._format_embedded_reference_value(resolved, value) if resolved is not None else reference

        return re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", replace, value)

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

    @staticmethod
    def _spr_constant_warnings(request: dict[str, Any], spr: SPR | None) -> list[str]:
        if spr is None or not spr.problem_text:
            return []
        text_numbers = set(re.findall(r"\d+", spr.problem_text))
        if not text_numbers:
            return []
        request_text = " ".join(
            [
                *(equation.get("lhs", "") + " " + equation.get("rhs", "") for equation in request.get("equations") or []),
                *(expression.get("expression", "") for expression in request.get("expressions") or []),
            ]
        )
        request_numbers = {number for number in re.findall(r"\d+", request_text) if number not in {"0", "1"}}
        missing = sorted(request_numbers - text_numbers)
        if not missing:
            return []
        return [
            "ToolCall uses constants not found verbatim in SPR problem_text: "
            + ", ".join(missing)
            + ". This is a lightweight warning, not a full semantic failure."
        ]

    @staticmethod
    def _checked_step(
        step_id: str,
        tool_call_id: str | None,
        operation: str | None,
        status: str,
        backend_replayed: bool,
        message: str,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        expected: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "step_id": step_id,
            "tool_call_id": tool_call_id,
            "operation": operation,
            "status": status,
            "backend_replayed": backend_replayed,
            "message": message,
            "errors": errors or [],
            "warnings": warnings or [],
            "expected": expected or {},
            "observed": observed or {},
            "metadata": metadata or {},
        }

    def _report(
        self,
        scs: SolutionChain,
        mode: str,
        valid: bool,
        status: str,
        scs_schema_valid: bool,
        backend_steps_valid: bool,
        final_answer_valid: bool,
        checked_steps: list[dict[str, Any]],
        errors: list[str],
        warnings: list[str],
        summary: str,
        metadata: dict[str, Any],
    ) -> VerificationReport:
        return VerificationReport.model_validate(
            {
                "problem_id": scs.problem_id,
                "chain_id": scs.chain_id,
                "valid": valid,
                "status": status,
                "verification_mode": mode,
                "scs_schema_valid": scs_schema_valid,
                "backend_steps_valid": backend_steps_valid,
                "final_answer_valid": final_answer_valid,
                "checked_steps": checked_steps,
                "errors": errors,
                "warnings": warnings,
                "summary": summary,
                "metadata": metadata,
            }
        )

    @staticmethod
    def _metadata(
        scs: SolutionChain,
        spr: SPR | None,
        emr: EMR | None,
        classification_result: ClassificationResult | None,
        repair_performed: bool,
    ) -> dict[str, Any]:
        return {
            "emr_required": False,
            "emr_used": bool(emr is not None and emr.equations),
            "tool_call_trace_used": any(
                step.rule_name and step.rule_name.startswith("planner_") for step in scs.steps
            ),
            "self_consistency": "not_applicable_single_chain",
            "repair_performed": repair_performed,
            "spr_problem_type": str(spr.problem_type) if spr is not None else None,
            "emr_representation_type": emr.representation_type if emr is not None else None,
            "solver_route": (
                classification_result.solver_route
                if classification_result is not None
                else scs.metadata.get("solver_route")
            ),
        }
