"""Verifier-guided constrained replanning for CAS-backed solver chains."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from mathexplain.agents.solver.calculation_request import CalculationRequestValidator
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain, SolutionStep
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport
from mathexplain.services.cas import CASService
from mathexplain.services.deepseek import DEFAULT_DEEPSEEK_MODEL, DeepSeekChatClient


class ReplanningClient(Protocol):
    """Text client used by the verifier-guided replanner."""

    model: str
    provider: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a constrained replanning JSON object."""


@dataclass(frozen=True)
class ReplanResult:
    """Replanning result containing a chain and inspectable report."""

    chain: SolutionChain
    report: dict[str, Any]


class VerifierGuidedReplanner:
    """Ask an LLM to repair only by proposing additional CAS tool calls."""

    SYSTEM_PROMPT = (
        "You are the verifier-guided replanner for MathExplainAgent Solver. "
        "You must not solve the problem by directly giving a final numeric answer. "
        "Your only allowed repair is to propose additional CAS tool_calls that use "
        "the given problem, verifier errors, and existing CAS context. "
        "Every new computational conclusion must be represented as a tool call. "
        "Return only a JSON object with the requested schema."
    )

    REPORT_VERSION = "VERIFIER_GUIDED_REPLAN-1.0"

    def __init__(
        self,
        client: ReplanningClient | None = None,
        cas_service: CASService | None = None,
        *,
        max_rounds: int = 1,
    ) -> None:
        self.client = client
        self.cas_service = cas_service or CASService()
        self.request_validator = CalculationRequestValidator(self.cas_service)
        self.max_rounds = max(0, max_rounds)

    def should_run(
        self,
        scs: SolutionChain,
        verification_report: VerificationReport,
        *,
        spr: SPR | None = None,
        repair_report: dict[str, Any] | None = None,
        semantic_report: dict[str, Any] | None = None,
    ) -> bool:
        """Return whether constrained replanning is worth trying."""

        if self.max_rounds <= 0:
            return False
        if not scs.final_answer:
            return True
        if scs.metadata.get("partial_failure"):
            return True
        if verification_report.valid is not True or verification_report.status != "passed":
            return True
        if self._current_answer_has_target_contract_risk(scs, spr):
            return True
        semantic_report = semantic_report or {}
        if semantic_report.get("decision") == "request_missing_cas_computation" and not semantic_report.get("repair_performed"):
            return True
        return False

    def replan(
        self,
        spr: SPR,
        emr: EMR,
        classification: ClassificationResult,
        scs: SolutionChain,
        verification_report: VerificationReport,
        *,
        repair_report: dict[str, Any] | None = None,
        semantic_report: dict[str, Any] | None = None,
        mode: str = "risk",
    ) -> ReplanResult:
        """Return a chain with appended CAS repair steps when safe."""

        report = self._base_report(scs, mode)
        report["initial_should_run"] = self.should_run(
            scs,
            verification_report,
            spr=spr,
            repair_report=repair_report,
            semantic_report=semantic_report,
        )
        if mode == "off":
            report["reason"] = "Verifier-guided replanning is disabled."
            return ReplanResult(scs, report)
        if mode == "risk" and not report["initial_should_run"]:
            report["reason"] = "No replanning trigger fired."
            return ReplanResult(scs, report)

        working_chain = scs
        for round_index in range(1, self.max_rounds + 1):
            round_report = self._run_one_round(
                spr,
                emr,
                classification,
                working_chain,
                verification_report,
                repair_report=repair_report,
                semantic_report=semantic_report,
                round_index=round_index,
            )
            report["rounds"].append(round_report)
            report["triggered"] = True
            if round_report.get("accepted"):
                working_chain = round_report.pop("_chain")
                report["repair_performed"] = True
                report["accepted"] = True
                report["reason"] = round_report.get("reason") or "Replanner appended CAS-backed final computation."
                return ReplanResult(working_chain, report)
            if round_report.get("fatal"):
                break
        report["reason"] = report["rounds"][-1].get("reason") if report["rounds"] else "No replanning round was run."
        return ReplanResult(scs, report)

    def _run_one_round(
        self,
        spr: SPR,
        emr: EMR,
        classification: ClassificationResult,
        scs: SolutionChain,
        verification_report: VerificationReport,
        *,
        repair_report: dict[str, Any] | None,
        semantic_report: dict[str, Any] | None,
        round_index: int,
    ) -> dict[str, Any]:
        round_report: dict[str, Any] = {
            "round": round_index,
            "accepted": False,
            "tool_call_count": 0,
            "executed_tool_call_count": 0,
            "target_contract": None,
            "llm_decision": None,
            "reason": "",
        }
        try:
            client = self.client or DeepSeekChatClient()
            raw = client.generate_text(
                self.SYSTEM_PROMPT,
                self._user_prompt(spr, emr, classification, scs, verification_report, repair_report, semantic_report),
            )
        except Exception as exc:
            round_report["fatal"] = True
            round_report["reason"] = f"Replanner client failed: {exc}"
            round_report["llm_error"] = str(exc)
            return round_report

        round_report["provider"] = getattr(client, "provider", "deepseek")
        round_report["model"] = getattr(client, "model", DEFAULT_DEEPSEEK_MODEL)
        round_report["usage"] = dict(getattr(client, "last_usage", {}) or {})
        round_report["raw_output"] = raw
        decision = self._parse_json_object(raw)
        round_report["llm_decision"] = decision
        if decision.get("parse_error"):
            round_report["reason"] = decision["parse_error"]
            return round_report
        round_report["target_contract"] = decision.get("target_contract")
        if self._contains_direct_final_answer(decision):
            round_report["reason"] = "Replanner output contained a direct final answer; rejected."
            return round_report
        if decision.get("replan_needed") is False:
            round_report["reason"] = str(decision.get("reason") or "LLM decided replanning is unnecessary.")
            return round_report

        tool_calls = decision.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            round_report["reason"] = "Replanner did not provide any CAS tool_calls."
            return round_report
        round_report["tool_call_count"] = len(tool_calls)
        appended = self._append_tool_calls(
            scs,
            emr,
            classification,
            spr,
            tool_calls,
            round_index=round_index,
            round_report=round_report,
            decision_reason=str(decision.get("reason") or ""),
        )
        if appended is None:
            return round_report
        round_report["accepted"] = True
        round_report["_chain"] = appended
        round_report["reason"] = str(decision.get("reason") or "Appended constrained CAS repair tool calls.")
        return round_report

    def _append_tool_calls(
        self,
        scs: SolutionChain,
        emr: EMR,
        classification: ClassificationResult,
        spr: SPR,
        tool_calls: list[dict[str, Any]],
        *,
        round_index: int,
        round_report: dict[str, Any],
        decision_reason: str,
    ) -> SolutionChain | None:
        repaired = scs.model_copy(deep=True)
        metadata = dict(repaired.metadata or {})
        execution_context = self._execution_context_from_scs(repaired)
        validation_reports = list(metadata.get("validation_reports") or [])
        final_item: dict[str, Any] | None = None
        depends_on = [repaired.steps[-1].step_id] if repaired.steps else []

        for tool_index, request_data in enumerate(tool_calls):
            shape_error = self._request_shape_error(request_data)
            if shape_error:
                round_report["reason"] = shape_error
                return None
            raw_call_id = str(request_data.get("id") or f"replan_{round_index}_{tool_index + 1}")
            call_id = raw_call_id
            if call_id in execution_context:
                call_id = f"replan_{round_index}_{call_id}"
            validation = self.request_validator.validate(
                request_data,
                emr,
                spr=spr,
                execution_context=execution_context,
            )
            jsonable_validation = self._jsonable_validation(validation)
            validation_reports.append(jsonable_validation)
            round_report.setdefault("validation_reports", []).append(jsonable_validation)
            if validation.get("valid") is not True:
                round_report["reason"] = "Replanner CAS request did not pass deterministic validation."
                return None
            normalized = validation.get("normalized_request") or {}
            normalized["id"] = call_id
            planner_step_id = f"replan_round_{round_index}_planner_{tool_index + 1}"
            planner_step = SolutionStep.model_validate(
                {
                    "step_id": planner_step_id,
                    "step_index": len(repaired.steps),
                    "description": "Verifier-guided replanner proposed a constrained CAS request.",
                    "math_expression": self._request_math_expression(normalized),
                    "rule_name": f"planner_{normalized.get('operation')}",
                    "justification": normalized.get("reason") or decision_reason or "Repair request generated from verifier feedback.",
                    "depends_on": depends_on,
                    "verification_status": "unverified",
                    "diagnostics": {
                        "tool_call_id": call_id,
                        "planner_role": "verifier_guided_replan",
                        "normalized_request": normalized,
                        "planner_variables": validation.get("planner_variables", []),
                        "effective_variables": validation.get("effective_variables", []),
                        "normalization_warnings": validation.get("normalization_warnings", []),
                        "validator_errors": validation.get("errors", []),
                        "backend_confirmed": False,
                    },
                }
            )
            repaired.steps.append(planner_step)

            cas_result = self._execute_request(normalized, execution_context)
            round_report.setdefault("backend_results", []).append(cas_result)
            if cas_result.get("success") is not True:
                round_report["reason"] = "Replanner CAS request failed during CAS execution."
                return None
            self._store_context_result(execution_context, call_id, cas_result, normalized)
            if raw_call_id != call_id:
                execution_context[raw_call_id] = execution_context.get(call_id, {})
            cas_step_id = f"replan_round_{round_index}_cas_{tool_index + 1}"
            cas_step = SolutionStep.model_validate(
                {
                    "step_id": cas_step_id,
                    "step_index": len(repaired.steps),
                    "description": "CAS executed verifier-guided repair request.",
                    "math_expression": self._final_answer_from_backend(cas_result),
                    "rule_name": f"cas_{cas_result.get('operation')}",
                    "justification": "The repair result is backend-confirmed by CAS.",
                    "depends_on": [planner_step.step_id],
                    "verification_status": "unverified",
                    "diagnostics": {
                        **cas_result,
                        "tool_call_id": call_id,
                        "backend_confirmed": True,
                        "source": "verifier_guided_replanner",
                    },
                }
            )
            repaired.steps.append(cas_step)
            round_report["executed_tool_call_count"] += 1
            depends_on = [cas_step.step_id]
            final_item = self._item_from_backend_result(normalized, cas_result, cas_step_id)

        if final_item is None:
            round_report["reason"] = "Replanner CAS calls did not produce a final-answer item."
            return None
        final_answer = self._format_final_answer(final_item)
        repaired.final_answer = final_answer
        repaired.confidence = max(repaired.confidence, 0.82)
        repaired.steps.append(
            SolutionStep.model_validate(
                {
                    "step_id": f"replan_round_{round_index}_final_answer",
                    "step_index": len(repaired.steps),
                    "description": "Use the verifier-guided CAS repair result as the final answer.",
                    "math_expression": final_answer,
                    "rule_name": "final_answer",
                    "justification": "Final answer is selected from the last verifier-guided CAS-backed result.",
                    "depends_on": depends_on,
                    "verification_status": "repaired",
                    "diagnostics": {
                        "backend": self.cas_service.backend_name,
                        "operation": "format_final_answer",
                        "input": {"final_answer_item": final_item},
                        "output": {"formatted_solution": final_answer},
                        "success": True,
                        "errors": [],
                        "warnings": [],
                    },
                }
            )
        )
        metadata["partial_failure"] = False
        metadata["backend_confirmed"] = True
        metadata["completed_tool_call_count"] = int(metadata.get("completed_tool_call_count") or 0) + round_report["executed_tool_call_count"]
        metadata["tool_call_count"] = int(metadata.get("tool_call_count") or 0) + len(tool_calls)
        metadata["validation_reports"] = validation_reports
        metadata["final_answer_items"] = [final_item]
        metadata["execution_context"] = execution_context
        metadata.setdefault("replanner", {})
        metadata["replanner"] = {
            "schema_version": self.REPORT_VERSION,
            "round_index": round_index,
            "tool_call_count": len(tool_calls),
            "executed_tool_call_count": round_report["executed_tool_call_count"],
        }
        repaired.metadata = metadata
        return repaired

    def _user_prompt(
        self,
        spr: SPR,
        emr: EMR,
        classification: ClassificationResult,
        scs: SolutionChain,
        verification_report: VerificationReport,
        repair_report: dict[str, Any] | None,
        semantic_report: dict[str, Any] | None,
    ) -> str:
        payload = {
            "problem": {
                "problem_id": scs.problem_id,
                "problem_text": spr.problem_text,
                "problem_stem": spr.problem_stem,
                "questions": [question.model_dump(mode="json") for question in spr.questions],
                "targets": [target.model_dump(mode="json") for target in spr.targets],
            },
            "classification": classification.model_dump(mode="json"),
            "emr": emr.model_dump(mode="json"),
            "current_solution": {
                "final_answer": scs.final_answer,
                "final_answer_items": scs.metadata.get("final_answer_items") or [],
                "partial_failure": bool(scs.metadata.get("partial_failure")),
                "cas_context": self._cas_context_summary(scs),
            },
            "verification": {
                "valid": verification_report.valid,
                "status": verification_report.status,
                "final_answer_valid": verification_report.final_answer_valid,
                "errors": verification_report.errors,
                "warnings": verification_report.warnings,
                "checked_steps": verification_report.metadata.get("checked_steps", [])
                if isinstance(verification_report.metadata, dict)
                else [],
            },
            "rule_repair_report": repair_report or {},
            "semantic_binding_report": semantic_report or {},
        }
        return (
            "Inspect the final question, verifier errors, and the current CAS-backed chain. "
            "If the current chain stops at an intermediate quantity, has a bad CAS request, "
            "or lacks the final computation requested by the question, propose only the missing or corrected CAS tool_calls.\n"
            "Return exactly this JSON shape, with no markdown:\n"
            "{\n"
            '  "target_contract": {\n'
            '    "requested_quantity": "what the final question asks for",\n'
            '    "target_type": "final_total | remaining | difference | converted_unit | money_amount | rate | percentage | count | other",\n'
            '    "unit": "requested unit or empty string",\n'
            '    "why_current_is_insufficient": "brief reason or empty string"\n'
            "  },\n"
            '  "replan_needed": true,\n'
            '  "tool_calls": [\n'
            "    {\n"
            '      "id": "short_repair_step_id",\n'
            '      "operation": "construct_expression | parse_expression | solve_equation_system | solve_for | substitute | evaluate",\n'
            '      "variables": ["x"],\n'
            '      "expressions": [{"name": "final_answer", "expression": "$some_prior_step.result + 3"}],\n'
            '      "equations": [{"lhs": "2*x", "rhs": "10"}],\n'
            '      "substitutions": {"x": "$some_prior_step.x"},\n'
            '      "target": "final_answer",\n'
            '      "reason": "why this CAS request completes the final question",\n'
            '      "evidence_span": "problem text supporting this computation"\n'
            "    }\n"
            "  ],\n"
            '  "reason": "short verifier-grounded explanation"\n'
            "}\n"
            "Rules:\n"
            "- Do not include any key named final_answer outside expressions.name/target.\n"
            "- Do not output a direct numeric answer. Numbers may appear only inside CAS expressions, equations, or substitutions.\n"
            "- Prefer evaluate or substitute when combining existing CAS results with remaining constants.\n"
            "- Every evaluate or substitute call must include expressions with the exact expression to execute; "
            "do not leave expressions empty and do not rely on target as the expression.\n"
            "- To evaluate a prior constructed expression, use an expression reference such as "
            '[{"name": "final_answer", "expression": "$prior_construct_call.result"}].\n'
            "- Use references exactly like $call_id.result, $call_id.value, $call_id.variable, or $call_id.target. "
            "The current CAS context lists available call ids and symbols.\n"
            "- If no safe CAS repair is possible, return replan_needed=true and an empty tool_calls list with a reason.\n"
            "Input JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @classmethod
    def _cas_context_summary(cls, scs: SolutionChain) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for step in scs.steps:
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            diagnostics = step.diagnostics or {}
            if diagnostics.get("success") is not True:
                continue
            request = diagnostics.get("input") or diagnostics.get("normalized_request") or {}
            output = diagnostics.get("output") or {}
            context.append(
                {
                    "step_id": step.step_id,
                    "tool_call_id": diagnostics.get("tool_call_id") or step.step_id,
                    "operation": diagnostics.get("operation"),
                    "target": request.get("target"),
                    "expression_names": [
                        expression.get("name")
                        for expression in request.get("expressions") or []
                        if isinstance(expression, dict)
                    ],
                    "output": output,
                    "description": step.description,
                    "justification": step.justification,
                }
            )
        return context

    @classmethod
    def _execution_context_from_scs(cls, scs: SolutionChain) -> dict[str, Any]:
        context: dict[str, Any] = {"variables": {}}
        for step in scs.steps:
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            diagnostics = step.diagnostics or {}
            if diagnostics.get("success") is not True:
                continue
            call_id = str(diagnostics.get("tool_call_id") or step.step_id)
            request = diagnostics.get("input") or diagnostics.get("normalized_request") or {}
            cls._store_context_result(context, call_id, diagnostics, request)
        return context

    @classmethod
    def _store_context_result(
        cls,
        context: dict[str, Any],
        call_id: str,
        cas_result: dict[str, Any],
        request: dict[str, Any],
    ) -> None:
        output = cas_result.get("output") or {}
        stored: dict[str, Any] = {"operation": cas_result.get("operation"), "output": output}
        raw_solution = output.get("raw_solution") or []
        if isinstance(raw_solution, dict):
            raw_solution = [raw_solution]
        for solution in raw_solution:
            if not isinstance(solution, dict):
                continue
            for symbol, value in solution.items():
                cls._store_symbol_aliases(stored, str(symbol), value)
                context.setdefault("variables", {})[str(symbol)] = value
        scalar = output.get("value") or output.get("formatted_solution") or output.get("confirmed_expression")
        if scalar is not None:
            stored["value"] = scalar
            stored["result"] = scalar
            stored.setdefault("formatted_solution", scalar)
        if output.get("confirmed_expression") is not None:
            stored["confirmed_expression"] = output["confirmed_expression"]
        target = request.get("target")
        if target and scalar is not None:
            cls._store_symbol_aliases(stored, str(target), scalar)
            context.setdefault("variables", {})[str(target)] = scalar
        for expression in request.get("expressions") or []:
            if isinstance(expression, dict) and expression.get("name") and scalar is not None:
                cls._store_symbol_aliases(stored, str(expression["name"]), scalar)
                context.setdefault("variables", {})[str(expression["name"])] = scalar
        context[call_id] = stored

    @staticmethod
    def _store_symbol_aliases(stored: dict[str, Any], symbol: str, value: Any) -> None:
        stored[symbol] = value
        stored[symbol.replace("{", "").replace("}", "")] = value
        stored["target"] = value

    def _execute_request(self, request: dict[str, Any], execution_context: dict[str, Any]) -> dict[str, Any]:
        request = self._resolve_request_references(request, execution_context)
        operation = request.get("operation")
        variables = request.get("effective_variables") or request.get("variables") or []
        if operation == "evaluate":
            expression = (request.get("expressions") or [{}])[0].get("expression", "")
            return self.cas_service.evaluate_expression(
                expression=expression,
                variables=variables,
                substitutions=request.get("substitutions") or {},
                target=request.get("target"),
            )
        if operation == "substitute":
            expression = (request.get("expressions") or [{}])[0].get("expression", "")
            return self.cas_service.substitute(
                expression=expression,
                variables=variables,
                substitutions=request.get("substitutions") or {},
                target=request.get("target"),
            )
        if operation == "solve_for":
            return self.cas_service.solve_for(
                equations=self._cas_equations(request),
                variables=variables,
                target=request.get("target") or "",
            )
        if operation == "solve_equation_system":
            return self.cas_service.solve_equation_system(
                equations=self._cas_equations(request),
                variables=variables,
                target=request.get("target"),
            )
        if operation in {"construct_expression", "parse_expression"}:
            expression = (request.get("expressions") or [{}])[0].get("expression", "")
            return self.cas_service.construct_expression(
                expression=expression,
                variables=variables,
                target=request.get("target"),
            )
        return {
            "backend": self.cas_service.backend_name,
            "operation": operation,
            "input": request,
            "output": {},
            "success": False,
            "errors": [f"Unsupported replanner CAS operation: {operation!r}"],
            "warnings": [],
        }

    @classmethod
    def _resolve_request_references(cls, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        resolved = deepcopy(request)
        for expression in resolved.get("expressions") or []:
            if isinstance(expression, dict) and isinstance(expression.get("expression"), str):
                expression["expression"] = cls._resolve_text_references(expression["expression"], context)
        for equation in resolved.get("equations") or []:
            if isinstance(equation, dict):
                if isinstance(equation.get("lhs"), str):
                    equation["lhs"] = cls._resolve_text_references(equation["lhs"], context)
                if isinstance(equation.get("rhs"), str):
                    equation["rhs"] = cls._resolve_text_references(equation["rhs"], context)
        substitutions = resolved.get("substitutions") or {}
        if isinstance(substitutions, dict):
            resolved["substitutions"] = {
                key: cls._resolve_reference_value(value, context)
                for key, value in substitutions.items()
            }
        return resolved

    @classmethod
    def _resolve_text_references(cls, text: str, context: dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            value = cls._resolve_reference_value(match.group(0), context)
            return CalculationRequestValidator._format_embedded_reference_value(value, text)

        return re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", replace, text)

    @staticmethod
    def _resolve_reference_value(value: Any, context: dict[str, Any]) -> Any:
        if not isinstance(value, str) or not value.startswith("$"):
            return value
        path = value[1:].split(".")
        current: Any = context.get(path[0])
        for part in path[1:]:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = None
            if current is None:
                break
        return value if current is None else current

    @staticmethod
    def _cas_equations(request: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "id": equation.get("id") or f"replan_eq_{index + 1}",
                "lhs_sympy": equation.get("lhs", ""),
                "rhs_sympy": equation.get("rhs", ""),
            }
            for index, equation in enumerate(request.get("equations") or [])
            if isinstance(equation, dict)
        ]

    @staticmethod
    def _current_answer_has_target_contract_risk(scs: SolutionChain, spr: SPR | None) -> bool:
        if spr is None:
            return False
        question_text = " ".join(question.text or "" for question in spr.questions).lower()
        if not question_text:
            question_text = (spr.problem_text or spr.problem_stem or "").lower()
        asks_for_positive_gap = any(
            signal in question_text
            for signal in [
                "win by",
                "faster",
                "how many more",
                "how much more",
                "difference",
                "gap",
                "ahead",
            ]
        )
        if not asks_for_positive_gap:
            return False
        for item in scs.metadata.get("final_answer_items") or []:
            value = str(item.get("value") or "").strip()
            if re.fullmatch(r"-\d+(?:\.\d+)?(?:0+)?", value):
                return True
        return bool(re.search(r"=\s*-\d", str(scs.final_answer or "")))

    @staticmethod
    def _request_shape_error(request_data: dict[str, Any]) -> str | None:
        operation = request_data.get("operation")
        if operation in {"evaluate", "substitute"} and not request_data.get("expressions"):
            return "Replanner evaluate/substitute request must include an explicit expressions list."
        return None

    @staticmethod
    def _request_math_expression(request: dict[str, Any]) -> str | None:
        equations = request.get("equations") or []
        if equations:
            equation = equations[0]
            return f"{equation.get('lhs')} = {equation.get('rhs')}"
        expressions = request.get("expressions") or []
        if expressions:
            return expressions[0].get("expression")
        return None

    @staticmethod
    def _first_expression_name(request: dict[str, Any]) -> str | None:
        for expression in request.get("expressions") or []:
            if isinstance(expression, dict) and expression.get("name"):
                return str(expression["name"])
        return None

    @classmethod
    def _item_from_backend_result(
        cls,
        request: dict[str, Any],
        cas_result: dict[str, Any],
        source_step_id: str,
    ) -> dict[str, Any] | None:
        output = cas_result.get("output") or {}
        target = str(request.get("target") or cls._first_expression_name(request) or request.get("id") or "final_answer")
        value = None
        raw_solution = output.get("raw_solution")
        if isinstance(raw_solution, list) and raw_solution and isinstance(raw_solution[0], dict):
            solution = raw_solution[0]
            value = solution.get(target)
            if value is None and len(solution) == 1:
                value = next(iter(solution.values()))
        elif isinstance(raw_solution, dict):
            value = raw_solution.get(target)
            if value is None and len(raw_solution) == 1:
                value = next(iter(raw_solution.values()))
        if value is None:
            value = output.get("value") or output.get("formatted_solution") or output.get("confirmed_expression")
        if value is None:
            return None
        return {
            "target": target,
            "value": str(value),
            "source_tool_call_id": request.get("id"),
            "source_step_id": source_step_id,
            "backend_operation": cas_result.get("operation"),
            "backend_confirmed": True,
            "source": "VerifierGuidedReplanner",
        }

    @staticmethod
    def _final_answer_from_backend(cas_result: dict[str, Any]) -> str | None:
        output = cas_result.get("output") or {}
        return output.get("formatted_solution") or output.get("value") or output.get("confirmed_expression")

    @staticmethod
    def _format_final_answer(item: dict[str, Any]) -> str:
        target = str(item.get("target") or "")
        value = item.get("value")
        if target and target not in {"answer", "final_answer", "requested_quantity"}:
            return f"{target} = {value}"
        return f"答案是 {value}"

    @staticmethod
    def _contains_direct_final_answer(decision: dict[str, Any]) -> bool:
        for key in decision:
            if key.lower() == "final_answer":
                return True
        return False

    @staticmethod
    def _parse_json_object(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        candidates = [text]
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            candidates.append(text[first : last + 1])
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return {"parse_error": "Replanner output was not valid JSON."}

    @staticmethod
    def _jsonable_validation(validation: dict[str, Any]) -> dict[str, Any]:
        data = dict(validation)
        request = data.get("request")
        if request is not None and hasattr(request, "model_dump"):
            data["request"] = request.model_dump(mode="json")
        return data

    @classmethod
    def _base_report(cls, scs: SolutionChain, mode: str) -> dict[str, Any]:
        return {
            "schema_version": cls.REPORT_VERSION,
            "problem_id": scs.problem_id,
            "chain_id": scs.chain_id,
            "mode": mode,
            "triggered": False,
            "accepted": False,
            "repair_performed": False,
            "initial_final_answer": scs.final_answer,
            "rounds": [],
            "reason": "No replanning was needed.",
        }
