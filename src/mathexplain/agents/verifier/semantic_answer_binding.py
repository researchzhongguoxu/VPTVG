"""Semantic final-answer binding verification with constrained repair."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from mathexplain.agents.solver.calculation_request import CalculationRequestValidator
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain, SolutionStep
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport
from mathexplain.services.cas import CASService
from mathexplain.services.deepseek import DEFAULT_DEEPSEEK_MODEL, DeepSeekChatClient


class SemanticBindingClient(Protocol):
    """Text client used by the semantic binding verifier."""

    model: str
    provider: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON decision for semantic answer binding."""


@dataclass(frozen=True)
class BindingCandidate:
    """One CAS-confirmed candidate answer available to the verifier."""

    candidate_id: str
    target: str
    value: str
    source_tool_call_id: str | None
    source_step_id: str | None
    backend_operation: str | None
    context: str
    is_current: bool
    step_index: int


class SemanticAnswerBindingVerifier:
    """Use an LLM only to select or request CAS-backed final-answer bindings."""

    SYSTEM_PROMPT = (
        "You are the semantic answer binding verifier for MathExplainAgent. "
        "Your job is to judge whether the current final answer answers the problem question. "
        "You must not solve the problem from scratch and must not invent a numeric answer. "
        "You may only choose from provided CAS-confirmed candidates, or request one missing CAS computation. "
        "Return only a JSON object with the requested schema."
    )
    ALLOWED_ACTIONS = {
        "accept_current",
        "rebind_existing_cas_candidate",
        "normalize_unit_or_format",
        "request_missing_cas_computation",
        "needs_review",
    }
    MIN_CONFIDENCE = 0.75

    def __init__(
        self,
        client: SemanticBindingClient | None = None,
        cas_service: CASService | None = None,
    ) -> None:
        self.client = client
        self.cas_service = cas_service or CASService()
        self.request_validator = CalculationRequestValidator(self.cas_service)

    def should_run(
        self,
        scs: SolutionChain,
        verification_report: VerificationReport,
        repair_report: dict[str, Any] | None = None,
        spr: SPR | None = None,
    ) -> bool:
        """Return whether semantic binding verification is useful for this chain."""

        if verification_report.final_answer_valid is not True:
            return True
        repair_report = repair_report or {}
        if repair_report.get("binding_decision") == "unrepairable":
            return True
        if repair_report.get("repair_rolled_back"):
            return True
        candidates = self._cas_candidates(scs)
        if len(candidates) >= 3:
            current = [candidate for candidate in candidates if candidate.is_current]
            if not current:
                return True
            current_index = max(candidate.step_index for candidate in current)
            later = [candidate for candidate in candidates if candidate.step_index > current_index]
            if later:
                return True
        if self._current_answer_uses_generic_symbol_for_semantic_quantity(scs, spr):
            return True
        return False

    def verify_and_repair(
        self,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        *,
        emr: EMR | None = None,
        repair_report: dict[str, Any] | None = None,
        mode: str = "risk",
    ) -> tuple[SolutionChain, dict[str, Any]]:
        """Return a possibly repaired chain plus a structured semantic report."""

        report = self._base_report(scs, mode)
        candidates = self._cas_candidates(scs)
        report["candidate_count"] = len(candidates)
        report["candidates"] = [candidate.__dict__ for candidate in candidates]
        if mode == "off":
            report["reason"] = "Semantic verifier is disabled."
            return scs, report
        if not candidates:
            report["reason"] = "No CAS-confirmed candidates are available."
            return scs, report
        if mode == "risk" and not self.should_run(scs, verification_report, repair_report, spr):
            report["reason"] = "No semantic binding risk trigger fired."
            report["decision"] = "accept_current"
            return scs, report
        report["triggered"] = True

        try:
            client = self.client or DeepSeekChatClient()
            raw = client.generate_text(
                self.SYSTEM_PROMPT,
                self._user_prompt(spr, scs, verification_report, candidates, repair_report),
            )
        except Exception as exc:
            report["reason"] = f"Semantic verifier client failed: {exc}"
            report["llm_error"] = str(exc)
            return scs, report

        report["provider"] = getattr(client, "provider", "deepseek")
        report["model"] = getattr(client, "model", DEFAULT_DEEPSEEK_MODEL)
        report["usage"] = dict(getattr(client, "last_usage", {}) or {})
        report["raw_output"] = raw
        decision = self._parse_decision(raw)
        report["llm_decision"] = decision
        if decision.get("parse_error"):
            report["reason"] = decision["parse_error"]
            return scs, report

        action = str(decision.get("repair_action") or decision.get("action") or "").strip()
        confidence = self._confidence(decision.get("confidence"))
        report["decision"] = action or "needs_review"
        report["confidence"] = confidence
        if action not in self.ALLOWED_ACTIONS:
            report["reason"] = f"LLM returned unsupported repair_action={action!r}."
            return scs, report
        if action in {"accept_current", "needs_review"}:
            report["accepted"] = action == "accept_current"
            report["reason"] = str(decision.get("reason") or f"LLM decision: {action}.")
            return scs, report
        if confidence < self.MIN_CONFIDENCE:
            report["reason"] = "LLM confidence is below the conservative acceptance threshold."
            return scs, report
        if action in {"rebind_existing_cas_candidate", "normalize_unit_or_format"}:
            return self._repair_from_existing_candidate(scs, candidates, decision, report, action)
        if action == "request_missing_cas_computation":
            return self._repair_from_missing_cas_request(spr, emr, scs, decision, report)
        return scs, report

    def _repair_from_existing_candidate(
        self,
        scs: SolutionChain,
        candidates: list[BindingCandidate],
        decision: dict[str, Any],
        report: dict[str, Any],
        action: str,
    ) -> tuple[SolutionChain, dict[str, Any]]:
        candidate = self._match_candidate(candidates, decision.get("preferred_candidate") or decision)
        if candidate is None:
            report["reason"] = "LLM preferred candidate was not found among CAS-confirmed candidates."
            return scs, report
        item = {
            "target": candidate.target,
            "value": candidate.value,
            "source_tool_call_id": candidate.source_tool_call_id,
            "source_step_id": candidate.source_step_id,
            "backend_operation": candidate.backend_operation,
            "backend_confirmed": True,
        }
        normalized = self._normalize_candidate_item(item, decision, action)
        if normalized is None:
            report["reason"] = "Requested normalization was not allowed by deterministic constraints."
            return scs, report
        return self._apply_repair(
            scs,
            normalized,
            report,
            repair_type=f"semantic_{action}",
            reason=str(decision.get("reason") or "Semantic verifier selected a CAS-confirmed final answer."),
        )

    def _repair_from_missing_cas_request(
        self,
        spr: SPR,
        emr: EMR | None,
        scs: SolutionChain,
        decision: dict[str, Any],
        report: dict[str, Any],
    ) -> tuple[SolutionChain, dict[str, Any]]:
        request_data = decision.get("cas_request") or decision.get("missing_cas_request")
        if not isinstance(request_data, dict):
            report["reason"] = "Missing CAS request repair did not include a request object."
            return scs, report
        if emr is None:
            report["reason"] = "Missing CAS request repair requires EMR for validation."
            return scs, report
        execution_context = self._execution_context_from_scs(scs)
        validation = self.request_validator.validate(request_data, emr, spr=spr, execution_context=execution_context)
        report["cas_request_validation"] = self._jsonable_validation(validation)
        if validation.get("valid") is not True:
            report["reason"] = "Proposed CAS request did not pass deterministic validation."
            return scs, report
        normalized = validation.get("normalized_request") or {}
        cas_result = self._execute_request(normalized, execution_context)
        report["cas_request_result"] = cas_result
        if cas_result.get("success") is not True:
            report["reason"] = "Proposed CAS request did not execute successfully."
            return scs, report
        item = self._item_from_backend_result(normalized, cas_result)
        if item is None:
            report["reason"] = "Successful CAS request did not produce a final-answer item."
            return scs, report

        repaired = scs.model_copy(deep=True)
        step_index = len(repaired.steps)
        step_id = f"semantic_repair_step_{step_index}"
        repaired.steps.append(
            SolutionStep.model_validate(
                {
                "step_id": step_id,
                "step_index": step_index,
                "description": str(normalized.get("reason") or "Semantic verifier requested a missing CAS computation."),
                "math_expression": self._format_final_answer(item),
                "rule_name": f"cas_{normalized.get('operation') or cas_result.get('operation')}",
                "justification": str(decision.get("reason") or "Constrained semantic repair."),
                "verification_status": "repaired",
                "diagnostics": {
                    **cas_result,
                    "tool_call_id": normalized.get("id") or "semantic_missing_cas_request",
                    "backend_confirmed": True,
                    "math_role": "cas_result",
                    "source": "semantic_answer_binding_verifier",
                    "input": cas_result.get("input") or normalized,
                },
            }
            )
        )
        return self._apply_repair(
            repaired,
            item,
            report,
            repair_type="semantic_missing_cas_request",
            reason=str(decision.get("reason") or "Semantic verifier requested and verified a missing CAS computation."),
        )

    @classmethod
    def _cas_candidates(cls, scs: SolutionChain) -> list[BindingCandidate]:
        current_items = scs.metadata.get("final_answer_items") or []
        candidates: list[BindingCandidate] = []
        seen: set[tuple[str, str, str | None]] = set()
        for step_index, step in enumerate(scs.steps):
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            diagnostics = step.diagnostics or {}
            if diagnostics.get("success") is not True:
                continue
            output = diagnostics.get("output") or {}
            request = diagnostics.get("input") or diagnostics.get("normalized_request") or {}
            target = str(
                request.get("target")
                or cls._first_expression_name(request)
                or diagnostics.get("tool_call_id")
                or step.step_id
            )
            values = cls._candidate_values(output, target)
            for value_target, value in values:
                candidate_target = str(value_target or target)
                candidate_value = str(value)
                call_id = diagnostics.get("tool_call_id")
                key = (candidate_target, candidate_value, call_id)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    BindingCandidate(
                        candidate_id=cls._candidate_id(call_id, step.step_id, candidate_target, candidate_value),
                        target=candidate_target,
                        value=candidate_value,
                        source_tool_call_id=call_id,
                        source_step_id=step.step_id,
                        backend_operation=diagnostics.get("operation"),
                        context=" ".join(
                            str(part or "")
                            for part in [
                                candidate_target,
                                diagnostics.get("tool_call_id"),
                                diagnostics.get("operation"),
                                step.description,
                                step.justification,
                                request.get("reason"),
                                request.get("evidence_span"),
                            ]
                        ),
                        is_current=cls._matches_current_answer(candidate_target, candidate_value, call_id, step.step_id, current_items, scs.final_answer),
                        step_index=step_index,
                    )
                )
        return candidates

    @staticmethod
    def _current_answer_uses_generic_symbol_for_semantic_quantity(
        scs: SolutionChain,
        spr: SPR | None,
    ) -> bool:
        if spr is None:
            return False
        items = scs.metadata.get("final_answer_items") or []
        if len(items) != 1:
            return False
        target = str(items[0].get("target") or "").strip().lower()
        if target not in {"x", "y", "z", "n", "m", "k"}:
            return False
        final_answer = str(scs.final_answer or "")
        if not re.fullmatch(r"\s*[A-Za-z]\s*=\s*[-+]?\d+(?:\.\d+)?\s*", final_answer):
            return False
        question_text = " ".join(
            [
                *(question.text or "" for question in spr.questions),
                *(target_obj.text or "" for target_obj in spr.targets),
            ]
        ).lower()
        return any(
            signal in question_text
            for signal in [
                "currently have",
                "current number",
                "current total",
                "combined total",
                "all raise in total",
            ]
        )

    @staticmethod
    def _candidate_values(output: dict[str, Any], target: str) -> list[tuple[str, Any]]:
        values: list[tuple[str, Any]] = []
        raw_solution = output.get("raw_solution")
        if isinstance(raw_solution, list):
            for solution in raw_solution:
                if not isinstance(solution, dict):
                    continue
                if target in solution:
                    values.append((target, solution[target]))
                else:
                    values.extend((str(symbol), value) for symbol, value in solution.items())
        elif isinstance(raw_solution, dict):
            if target in raw_solution:
                values.append((target, raw_solution[target]))
            else:
                values.extend((str(symbol), value) for symbol, value in raw_solution.items())
        value = output.get("value")
        if value is not None:
            values.append((target, value))
        if not values and output.get("formatted_solution") is not None:
            values.append((target, output["formatted_solution"]))
        if not values and output.get("confirmed_expression") is not None:
            values.append((target, output["confirmed_expression"]))
        if not values and output.get("parsed_expression") is not None:
            values.append((target, output["parsed_expression"]))
        return values

    @staticmethod
    def _candidate_id(call_id: str | None, step_id: str, target: str, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", f"{call_id or step_id}_{target}_{value}")[:120]
        return safe.strip("_") or step_id

    @classmethod
    def _matches_current_answer(
        cls,
        target: str,
        value: str,
        call_id: str | None,
        step_id: str | None,
        current_items: list[dict[str, Any]],
        final_answer: str | None,
    ) -> bool:
        for item in current_items:
            if call_id and item.get("source_tool_call_id") == call_id:
                return True
            if step_id and item.get("source_step_id") == step_id:
                return True
            if str(item.get("target") or "") == target and cls._numbers_equal(item.get("value"), value):
                return True
        return cls._numbers_equal(final_answer, value)

    @staticmethod
    def _first_expression_name(request: dict[str, Any]) -> str | None:
        for expression in request.get("expressions") or []:
            if isinstance(expression, dict) and expression.get("name"):
                return str(expression["name"])
        return None

    def _user_prompt(
        self,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        candidates: list[BindingCandidate],
        repair_report: dict[str, Any] | None,
    ) -> str:
        payload = {
            "problem_id": scs.problem_id,
            "problem_text": spr.problem_text,
            "questions": [question.model_dump(mode="json") for question in spr.questions],
            "targets": [target.model_dump(mode="json") for target in spr.targets],
            "current_final_answer": scs.final_answer,
            "current_final_answer_items": scs.metadata.get("final_answer_items") or [],
            "verification": {
                "valid": verification_report.valid,
                "status": verification_report.status,
                "final_answer_valid": verification_report.final_answer_valid,
                "errors": verification_report.errors,
                "warnings": verification_report.warnings,
            },
            "rule_repair_report": repair_report or {},
            "cas_confirmed_candidates": [candidate.__dict__ for candidate in candidates],
        }
        return (
            "Decide whether the current final answer semantically answers the problem question.\n"
            "Use only the provided CAS-confirmed candidates unless a missing CAS request is necessary.\n"
            "Return exactly this JSON shape, with no markdown:\n"
            "{\n"
            '  "question_intent": "brief semantic target of the final question",\n'
            '  "current_answer_matches_intent": true,\n'
            '  "binding_error_type": "none | selected_intermediate_total | selected_wrong_variable | unit_mismatch | missing_candidate | other",\n'
            '  "repair_action": "accept_current | rebind_existing_cas_candidate | normalize_unit_or_format | request_missing_cas_computation | needs_review",\n'
            '  "preferred_candidate": {"candidate_id": "...", "source_tool_call_id": "...", "target": "...", "value": "..."},\n'
            '  "cas_request": null,\n'
            '  "confidence": 0.0,\n'
            '  "reason": "short reason grounded in the question and candidates"\n'
            "}\n"
            "If choosing rebind_existing_cas_candidate, preferred_candidate must exactly identify one provided candidate. "
            "If requesting missing CAS computation, cas_request must be a planner-style CAS request and must not include a final answer value. "
            "Input JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_decision(raw_text: str) -> dict[str, Any]:
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
        return {"parse_error": "Semantic verifier output was not valid JSON."}

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _match_candidate(cls, candidates: list[BindingCandidate], data: Any) -> BindingCandidate | None:
        if not isinstance(data, dict):
            return None
        candidate_id = str(data.get("candidate_id") or "")
        source_tool_call_id = str(data.get("source_tool_call_id") or "")
        target = str(data.get("target") or "")
        value = data.get("value")
        for candidate in candidates:
            if candidate_id and candidate.candidate_id == candidate_id:
                return candidate
        matches = [
            candidate
            for candidate in candidates
            if (
                (not source_tool_call_id or candidate.source_tool_call_id == source_tool_call_id)
                and (not target or candidate.target == target)
                and (value is None or cls._numbers_equal(candidate.value, value))
            )
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _normalize_candidate_item(
        cls,
        item: dict[str, Any],
        decision: dict[str, Any],
        action: str,
    ) -> dict[str, Any] | None:
        if action != "normalize_unit_or_format":
            return item
        conversion = str(decision.get("normalization") or decision.get("unit_conversion") or "").lower()
        if conversion not in {"cents_to_dollars", "cent_to_dollar"}:
            return None
        value = cls._decimal(item.get("value"))
        if value is None or abs(value) < Decimal("100"):
            return None
        converted = dict(item)
        converted["value"] = cls._format_decimal(value / Decimal("100"))
        converted["semantic_normalization"] = conversion
        return converted

    def _execute_request(
        self,
        request: dict[str, Any],
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = execution_context or {}
        request = self._resolve_request_references(request, context)
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
            "errors": [f"Unsupported semantic repair CAS operation: {operation!r}"],
            "warnings": [],
        }

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
        stored: dict[str, Any] = {
            "operation": cas_result.get("operation"),
            "output": output,
        }
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
            context.setdefault("variables", {}).setdefault(str(target), scalar)
        expressions = request.get("expressions") or []
        if expressions and isinstance(expressions[0], dict):
            name = expressions[0].get("name")
            expression_value = output.get("value") or output.get("confirmed_expression") or output.get("formatted_solution")
            if name and expression_value is not None:
                cls._store_symbol_aliases(stored, str(name), expression_value)
        context[call_id] = stored

    @staticmethod
    def _store_symbol_aliases(target: dict[str, Any], symbol: str, value: Any) -> None:
        target[symbol] = value
        no_braces = re.sub(r"_\{([^{}]+)\}", r"_\1", symbol.strip())
        if no_braces != symbol:
            target.setdefault(no_braces, value)
        brace_match = re.fullmatch(r"([A-Za-z]+)_([A-Za-z0-9]+)", symbol.strip())
        if brace_match:
            target.setdefault(f"{brace_match.group(1)}_{{{brace_match.group(2)}}}", value)

    @staticmethod
    def _resolve_request_references(request: dict[str, Any], execution_context: dict[str, Any]) -> dict[str, Any]:
        resolved = deepcopy(request)
        substitutions = resolved.get("substitutions") or {}
        resolved["substitutions"] = {
            symbol: (
                str(CalculationRequestValidator.resolve_reference(value, execution_context))
                if isinstance(value, str)
                and value.startswith("$")
                and CalculationRequestValidator.resolve_reference(value, execution_context) is not None
                else value
            )
            for symbol, value in substitutions.items()
        }
        for expression in resolved.get("expressions") or []:
            if not isinstance(expression, dict):
                continue
            value = expression.get("expression")
            if isinstance(value, str):
                expression["expression"] = CalculationRequestValidator._resolve_expression_reference(
                    value,
                    execution_context=execution_context,
                )
        for equation in resolved.get("equations") or []:
            if not isinstance(equation, dict):
                continue
            for field in ("lhs", "rhs", "lhs_sympy", "rhs_sympy"):
                value = equation.get(field)
                if isinstance(value, str):
                    equation[field] = CalculationRequestValidator._resolve_expression_reference(
                        value,
                        execution_context=execution_context,
                    )
        return resolved

    @staticmethod
    def _cas_equations(request: dict[str, Any]) -> list[dict[str, str]]:
        equations: list[dict[str, str]] = []
        for index, equation in enumerate(request.get("equations") or []):
            equations.append(
                {
                    "id": equation.get("id") or f"semantic_eq_{index + 1}",
                    "lhs_sympy": equation.get("lhs_sympy") or equation.get("lhs") or "",
                    "rhs_sympy": equation.get("rhs_sympy") or equation.get("rhs") or "",
                }
            )
        return equations

    @classmethod
    def _item_from_backend_result(cls, request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        output = result.get("output") or {}
        target = str(request.get("target") or cls._first_expression_name(request) or "semantic_result")
        value = None
        raw_solution = output.get("raw_solution")
        if isinstance(raw_solution, list) and raw_solution and isinstance(raw_solution[0], dict):
            value = raw_solution[0].get(target)
        elif isinstance(raw_solution, dict):
            value = raw_solution.get(target)
        if value is None:
            value = output.get("value") or output.get("formatted_solution")
        if value is None:
            return None
        return {
            "target": target,
            "value": str(value),
            "source_tool_call_id": request.get("id") or "semantic_missing_cas_request",
            "backend_operation": result.get("operation") or request.get("operation"),
            "backend_confirmed": True,
        }

    def _apply_repair(
        self,
        scs: SolutionChain,
        item: dict[str, Any],
        report: dict[str, Any],
        *,
        repair_type: str,
        reason: str,
    ) -> tuple[SolutionChain, dict[str, Any]]:
        repaired = scs.model_copy(deep=True)
        original_answer = repaired.final_answer
        new_answer = self._format_final_answer(item)
        repaired.final_answer = new_answer
        metadata = deepcopy(repaired.metadata)
        metadata["final_answer_items"] = [item]
        metadata["semantic_answer_binding"] = {
            "repair_performed": True,
            "repair_type": repair_type,
            "original_final_answer": original_answer,
            "repaired_final_answer": new_answer,
            "reason": reason,
        }
        repaired.metadata = metadata
        self._rewrite_final_answer_step(repaired, new_answer, item)
        report.update(
            {
                "accepted": True,
                "repair_performed": True,
                "repair_type": repair_type,
                "original_final_answer": original_answer,
                "repaired_final_answer": new_answer,
                "selected_target": item.get("target"),
                "selected_source_tool_call_id": item.get("source_tool_call_id"),
                "reason": reason,
            }
        )
        return repaired, report

    @classmethod
    def _rewrite_final_answer_step(cls, scs: SolutionChain, final_answer: str, item: dict[str, Any]) -> None:
        for step in reversed(scs.steps):
            if step.rule_name == "final_answer":
                step.math_expression = final_answer
                step.verification_status = "repaired"
                step.diagnostics = {
                    **(step.diagnostics or {}),
                    "input": {"final_answer_items": [item]},
                    "output": {"formatted_solution": final_answer},
                    "source": "semantic_answer_binding_verifier",
                }
                return

    @staticmethod
    def _format_final_answer(item: dict[str, Any]) -> str:
        target = str(item.get("target") or "").strip()
        value = str(item.get("value") or "").strip()
        if not target:
            return value
        return f"{target} = {value}"

    @staticmethod
    def _base_report(scs: SolutionChain, mode: str) -> dict[str, Any]:
        return {
            "schema_version": "SEMANTIC_ANSWER_BINDING-1.0",
            "problem_id": scs.problem_id,
            "chain_id": scs.chain_id,
            "mode": mode,
            "triggered": False,
            "accepted": False,
            "repair_performed": False,
            "repair_type": None,
            "decision": "skip",
            "confidence": 0.0,
            "candidate_count": 0,
            "candidates": [],
            "reason": "No semantic repair was performed.",
        }

    @staticmethod
    def _jsonable_validation(validation: dict[str, Any]) -> dict[str, Any]:
        result = dict(validation)
        if result.get("request") is not None and hasattr(result["request"], "model_dump"):
            result["request"] = result["request"].model_dump(mode="json")
        return result

    @staticmethod
    def _first_numeric_token(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if "=" in text:
            text = text.rsplit("=", 1)[-1]
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
        return match.group(0) if match else None

    @classmethod
    def _numbers_equal(cls, left: Any, right: Any) -> bool:
        left_decimal = cls._decimal(cls._first_numeric_token(left))
        right_decimal = cls._decimal(cls._first_numeric_token(right))
        return left_decimal is not None and right_decimal is not None and left_decimal == right_decimal

    @staticmethod
    def _decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        try:
            return Decimal(str(value).strip().replace(",", ""))
        except (InvalidOperation, ValueError):
            return None

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.quantize(Decimal("1")))
        return format(normalized, "f")
