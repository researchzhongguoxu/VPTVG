"""SCS assembly helpers for explainer-ready solution chains."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.spr import SPR


class SCSAssemblyEnhancer:
    """Add general explanatory structure to Solver-produced SCS dictionaries.

    The enhancer does not solve new math. It only organizes existing SPR/EMR,
    planner, and CAS facts into a more explainer-ready SCS shape.
    """

    def enhance(
        self,
        chain_data: dict[str, Any],
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None = None,
    ) -> dict[str, Any]:
        """Return an enhanced copy of a SolutionChain dictionary."""

        enhanced = deepcopy(chain_data)
        original_steps = enhanced.get("steps") or []
        new_steps: list[dict[str, Any]] = []
        old_to_new: dict[str, str] = {}
        pending_variable_steps = self._variable_definition_steps(spr, emr, enhanced)

        for step in original_steps:
            rule_name = step.get("rule_name") or ""
            if rule_name in {"given_equation", "unsupported_route"}:
                for variable_step in pending_variable_steps:
                    self._append_generated_step(new_steps, variable_step, old_to_new)
                pending_variable_steps = []

            if rule_name.startswith("planner_"):
                self._append_generated_step(
                    new_steps,
                    self._modeling_step_from_planner(step, spr),
                    old_to_new,
                )

            copied = deepcopy(step)
            self._augment_existing_step(copied, spr)
            self._append_existing_step(new_steps, copied, old_to_new)

            if rule_name == "read_problem_and_variables":
                for variable_step in pending_variable_steps:
                    self._append_generated_step(new_steps, variable_step, old_to_new)
                pending_variable_steps = []

        if pending_variable_steps:
            for variable_step in pending_variable_steps:
                self._append_generated_step(new_steps, variable_step, old_to_new)

        self._rewrite_dependencies(new_steps, old_to_new)
        enhanced["steps"] = new_steps
        self._rewrite_step_references(enhanced, old_to_new)
        self._augment_metadata(enhanced, emr, classification_result, spr)
        return enhanced

    @staticmethod
    def _append_existing_step(
        steps: list[dict[str, Any]],
        step: dict[str, Any],
        old_to_new: dict[str, str],
    ) -> None:
        old_id = str(step.get("step_id") or f"original_step_{len(old_to_new)}")
        new_id = f"chain_main_step_{len(steps)}"
        old_to_new[old_id] = new_id
        step["step_id"] = new_id
        step["step_index"] = len(steps)
        if not step.get("depends_on") and steps:
            step["depends_on"] = [steps[-1]["step_id"]]
        steps.append(step)

    @staticmethod
    def _append_generated_step(
        steps: list[dict[str, Any]],
        step: dict[str, Any] | None,
        old_to_new: dict[str, str],
    ) -> None:
        if step is None:
            return
        step = deepcopy(step)
        step["step_id"] = f"chain_main_step_{len(steps)}"
        step["step_index"] = len(steps)
        step["depends_on"] = [steps[-1]["step_id"]] if steps else []
        steps.append(step)

    @staticmethod
    def _rewrite_dependencies(steps: list[dict[str, Any]], old_to_new: dict[str, str]) -> None:
        for index, step in enumerate(steps):
            dependencies = step.get("depends_on") or []
            rewritten = [old_to_new.get(dep, dep) for dep in dependencies]
            if not rewritten and index > 0 and step["diagnostics"].get("generated_by") == "scs_assembly":
                rewritten = [steps[index - 1]["step_id"]]
            step["depends_on"] = rewritten

    def _augment_existing_step(self, step: dict[str, Any], spr: SPR | None) -> None:
        diagnostics = step.setdefault("diagnostics", {})
        rule_name = step.get("rule_name") or ""
        if rule_name == "read_problem_and_variables":
            diagnostics.setdefault("math_role", "problem_understanding")
            diagnostics.setdefault("source", "spr")
            diagnostics.setdefault("explainer_ready", True)
            diagnostics.setdefault("referenced_condition_ids", self._condition_ids(spr))
            diagnostics.setdefault("referenced_question_ids", self._question_ids(spr))
        elif rule_name == "given_equation":
            diagnostics.setdefault("math_role", "modeling")
            diagnostics.setdefault("source", "emr")
            diagnostics.setdefault("explainer_ready", True)
        elif rule_name.startswith("planner_"):
            request = diagnostics.get("normalized_request") or {}
            diagnostics.setdefault("math_role", "calculation_request")
            diagnostics.setdefault("source", "planner_tool_call")
            diagnostics.setdefault("evidence_span", request.get("evidence_span"))
            diagnostics.setdefault("explainer_ready", False)
            diagnostics.setdefault("referenced_condition_ids", self._matching_condition_ids(spr, request.get("evidence_span")))
            diagnostics.setdefault("referenced_question_ids", self._matching_question_ids(spr, request.get("evidence_span")))
        elif rule_name.startswith("cas_"):
            diagnostics.setdefault("math_role", "cas_result")
            diagnostics.setdefault("source", "cas")
            diagnostics.setdefault("explainer_ready", True)
        elif rule_name == "final_answer":
            diagnostics.setdefault("math_role", "answer_interpretation")
            diagnostics.setdefault("source", "cas")
            diagnostics.setdefault("explainer_ready", True)
            diagnostics.setdefault("referenced_question_ids", self._question_ids(spr))
        elif rule_name == "unsupported_route":
            diagnostics.setdefault("math_role", "unsupported")
            diagnostics.setdefault("source", "rule")
            diagnostics.setdefault("explainer_ready", True)

    def _variable_definition_steps(
        self,
        spr: SPR | None,
        emr: EMR,
        chain_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        bindings = self._variable_bindings(spr, emr, chain_data)
        steps: list[dict[str, Any]] = []
        for symbol, binding in bindings.items():
            description = binding.get("description") or symbol
            steps.append(
                {
                    "step_id": "",
                    "step_index": 0,
                    "description": f"Define {symbol} as {description}.",
                    "math_expression": symbol,
                    "rule_name": "define_variable",
                    "justification": "The variable meaning is taken from SPR/EMR before computation.",
                    "depends_on": [],
                    "verification_status": "unverified",
                    "diagnostics": {
                        "math_role": "variable_definition",
                        "source": binding.get("source") or "spr",
                        "symbol": symbol,
                        "description": description,
                        "domain": binding.get("domain"),
                        "unit": binding.get("unit"),
                        "variable_role": binding.get("variable_role"),
                        "used_in_computation": binding.get("used_in_computation"),
                        "explainer_priority": binding.get("explainer_priority"),
                        "explainer_ready": True,
                        "generated_by": "scs_assembly",
                    },
                }
            )
        return steps

    def _modeling_step_from_planner(self, planner_step: dict[str, Any], spr: SPR | None) -> dict[str, Any] | None:
        diagnostics = planner_step.get("diagnostics") or {}
        request = diagnostics.get("normalized_request") or {}
        operation = request.get("operation")
        if not operation:
            return None
        expression = self._request_math_expression(request)
        reason = request.get("reason") or planner_step.get("justification") or "Prepare a CAS-backed computation."
        evidence = request.get("evidence_span")
        return {
            "step_id": "",
            "step_index": 0,
            "description": self._modeling_description(operation),
            "math_expression": expression,
            "rule_name": "modeling_from_tool_call",
            "justification": reason,
            "depends_on": [],
            "verification_status": "unverified",
            "diagnostics": {
                "math_role": "modeling",
                "source": "planner_tool_call",
                "tool_call_id": diagnostics.get("tool_call_id") or request.get("id"),
                "operation": operation,
                "evidence_span": evidence,
                "referenced_condition_ids": self._matching_condition_ids(spr, evidence),
                "referenced_question_ids": self._matching_question_ids(spr, evidence),
                "explainer_ready": True,
                "generated_by": "scs_assembly",
                "normalized_request": request,
            },
        }

    @staticmethod
    def _modeling_description(operation: str) -> str:
        descriptions = {
            "construct_expression": "Build an executable expression from the problem statement.",
            "parse_expression": "Check that the planned expression is executable.",
            "solve_equation_system": "Translate the problem conditions into an equation system.",
            "solve_for": "Translate the problem conditions into an equation for the target variable.",
            "substitute": "Substitute a confirmed result into the requested expression.",
            "evaluate": "Prepare the requested expression for evaluation.",
        }
        return descriptions.get(operation, "Prepare a CAS-backed reasoning step.")

    @staticmethod
    def _request_math_expression(request: dict[str, Any]) -> str | None:
        equations = request.get("equations") or []
        if equations:
            return " ; ".join(
                f"{equation.get('lhs')} = {equation.get('rhs')}"
                for equation in equations
            )
        expressions = request.get("expressions") or []
        if expressions:
            return " ; ".join(
                str(expression.get("expression"))
                for expression in expressions
                if expression.get("expression") is not None
            )
        return None

    def _augment_metadata(
        self,
        chain_data: dict[str, Any],
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None,
    ) -> None:
        metadata = chain_data.setdefault("metadata", {})
        metadata.setdefault("scs_assembly_version", "SCSAssembly-1.0")
        metadata.setdefault("explainer_ready", True)
        metadata["variable_bindings"] = self._variable_bindings(spr, emr, chain_data)
        metadata["target_bindings"] = self._target_bindings(spr, emr)
        self._add_semantic_answer_variable_bindings(metadata, spr)
        metadata["answer_bindings"] = self._answer_bindings(metadata.get("final_answer_items") or [], metadata, spr)
        metadata["solution_outline"] = self._solution_outline(chain_data.get("steps") or [])
        metadata.setdefault("solver_route", classification_result.solver_route)
        metadata.setdefault("task_type", classification_result.task_type)

    def _add_semantic_answer_variable_bindings(self, metadata: dict[str, Any], spr: SPR | None) -> None:
        variable_bindings = metadata.setdefault("variable_bindings", {})
        for item in metadata.get("final_answer_items") or []:
            target = str(item.get("target") or "")
            if not target or target in variable_bindings or not self._is_generic_solver_symbol(target):
                continue
            semantic_target = self._semantic_target_for_answer_item(item, metadata, spr)
            if not semantic_target:
                continue
            semantic_binding = variable_bindings.get(semantic_target) or {}
            description = str(semantic_binding.get("description") or "").strip()
            if not description:
                description = self._display_name_for_target(semantic_target, variable_bindings, metadata.get("target_bindings") or {}, spr)
            unit = semantic_binding.get("unit") or self._unit_from_question_or_domain(description, spr, item)
            variable_bindings[target] = {
                "symbol": target,
                "description": description or target,
                "domain": semantic_binding.get("domain"),
                "unit": unit,
                "source": "semantic_answer_alias",
                "semantic_target": semantic_target,
                "variable_role": "primary_unknown",
                "used_in_computation": True,
                "is_final_answer_target": True,
                "explainer_priority": self._explainer_priority("primary_unknown"),
            }

    def _variable_bindings(
        self,
        spr: SPR | None,
        emr: EMR,
        chain_data: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        bindings: dict[str, dict[str, Any]] = {}
        if spr is not None:
            for variable in spr.variables:
                if not variable.symbol:
                    continue
                bindings[variable.symbol] = {
                    "symbol": variable.symbol,
                    "description": variable.description or variable.symbol,
                    "domain": variable.domain,
                    "unit": self._infer_unit(" ".join([variable.description or "", variable.domain or ""])),
                    "source": "spr",
                }
        for variable in emr.variables:
            if not variable.symbol or variable.symbol in bindings:
                continue
            bindings[variable.symbol] = {
                "symbol": variable.symbol,
                "description": variable.description or variable.symbol,
                "domain": variable.domain,
                "unit": self._infer_unit(" ".join([variable.description or "", variable.domain or ""])),
                "source": "emr",
            }
        usage = self._variable_usage(bindings, chain_data)
        for symbol, binding in bindings.items():
            role = self._variable_role(symbol, binding, usage, spr)
            binding["variable_role"] = role
            binding["used_in_computation"] = symbol in usage["used_symbols"]
            binding["is_final_answer_target"] = symbol in usage["final_targets"]
            binding["explainer_priority"] = self._explainer_priority(role)
        return bindings

    def _target_bindings(self, spr: SPR | None, emr: EMR) -> dict[str, dict[str, Any]]:
        bindings: dict[str, dict[str, Any]] = {}
        if spr is not None:
            question_by_target: dict[str, str] = {}
            for question in spr.questions:
                for target_id in question.target_ids:
                    question_by_target[target_id] = question.id
            for target in spr.targets:
                bindings[target.id] = {
                    "id": target.id,
                    "text": target.text,
                    "target_type": target.target_type,
                    "question_id": question_by_target.get(target.id),
                    "unit": self._infer_unit(target.text),
                    "source": "spr",
                }
        for goal in emr.goals:
            if goal.target and goal.target not in bindings:
                question_id = self._question_id_for_goal(goal, spr)
                bindings[str(goal.target)] = {
                    "id": goal.id,
                    "text": goal.description or str(goal.target),
                    "target_type": goal.goal_type,
                    "question_id": question_id,
                    "unit": self._infer_unit(" ".join([goal.description or "", str(goal.target)])),
                    "source": "emr",
                }
        return bindings

    def _answer_bindings(
        self,
        final_answer_items: list[dict[str, Any]],
        metadata: dict[str, Any],
        spr: SPR | None,
    ) -> list[dict[str, Any]]:
        variable_bindings = metadata.get("variable_bindings") or {}
        target_bindings = metadata.get("target_bindings") or {}
        result: list[dict[str, Any]] = []
        for item in final_answer_items:
            target = str(item.get("target") or "")
            semantic_target = self._semantic_target_for_answer_item(item, metadata, spr)
            display_name = self._display_name_for_target(
                semantic_target or target,
                variable_bindings,
                target_bindings,
                spr,
            )
            unit = self._unit_for_target(
                semantic_target or target,
                display_name,
                variable_bindings,
                target_bindings,
            )
            if unit is None:
                unit = self._unit_from_question_or_domain(display_name, spr, item)
            result.append(
                {
                    "target": target,
                    "display_name": display_name,
                    "semantic_target": semantic_target,
                    "value": item.get("value"),
                    "selected_option_id": item.get("selected_option_id"),
                    "selected_option_label": item.get("selected_option_label"),
                    "selected_option_text": item.get("selected_option_text"),
                    "unit": unit,
                    "source_tool_call_id": item.get("source_tool_call_id"),
                    "source_step_id": item.get("source_step_id"),
                    "backend_operation": item.get("backend_operation"),
                    "backend_confirmed": item.get("backend_confirmed") is True,
                    "question_id": self._question_id_for_target(
                        semantic_target or target,
                        display_name,
                        target_bindings,
                        spr,
                    ),
                }
            )
        return result

    @staticmethod
    def _semantic_target_for_answer_item(
        item: dict[str, Any],
        metadata: dict[str, Any],
        spr: SPR | None,
    ) -> str | None:
        target = str(item.get("target") or "")
        source_tool_call_id = str(item.get("source_tool_call_id") or "")
        for report in metadata.get("validation_reports") or []:
            request = report.get("normalized_request") or {}
            if source_tool_call_id and request.get("id") != source_tool_call_id:
                continue
            aliases = request.get("semantic_symbol_aliases") or report.get("semantic_symbol_aliases") or {}
            if target in aliases:
                return str(aliases[target])
        if spr is None or not SCSAssemblyEnhancer._is_generic_solver_symbol(target):
            return None
        if len(metadata.get("final_answer_items") or []) != 1:
            return None
        return SCSAssemblyEnhancer._single_final_target_symbol(spr)

    @staticmethod
    def _is_generic_solver_symbol(symbol: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z]", str(symbol or ""))) or str(symbol or "").lower() in {
            "answer",
            "result",
            "unknown",
        }

    @staticmethod
    def _single_final_target_symbol(spr: SPR) -> str | None:
        if not spr.questions:
            return None
        target_ids = list(getattr(spr.questions[-1], "target_ids", []) or [])
        if len(target_ids) == 1:
            target_id = target_ids[0]
            target_obj = next((target for target in spr.targets if target.id == target_id), None)
            if target_obj is not None:
                best_symbol: str | None = None
                best_score = 0.0
                for variable in spr.variables:
                    symbol = variable.symbol
                    if not symbol:
                        continue
                    score = SCSAssemblyEnhancer._semantic_match_score(
                        " ".join([symbol, variable.description or ""]),
                        target_obj.text,
                    )
                    if score > best_score:
                        best_score = score
                        best_symbol = symbol
                if best_symbol and best_score >= 0.35:
                    return best_symbol
        candidates: list[tuple[float, str]] = []
        question_text = str(spr.questions[-1].text or "")
        for variable in spr.variables:
            symbol = variable.symbol
            if not symbol:
                continue
            text = " ".join([symbol, variable.description or ""])
            score = SCSAssemblyEnhancer._semantic_match_score(text, question_text)
            if any(token in text.lower() for token in ["unknown", "cost", "price", "answer", "requested"]):
                score += 0.25
            candidates.append((score, symbol))
        if not candidates:
            return None
        best_score, best_symbol = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        return best_symbol if best_score >= 0.35 else None

    @staticmethod
    def _unit_from_question_or_domain(
        display_name: str,
        spr: SPR | None,
        item: dict[str, Any],
    ) -> str | None:
        text = " ".join(
            str(part or "")
            for part in [
                display_name,
                item.get("target"),
                item.get("semantic_target"),
                getattr(spr.questions[-1], "text", "") if spr and spr.questions else "",
                getattr(spr, "problem_text", "") if spr is not None else "",
            ]
        ).lower()
        if any(token in text for token in ["$", "dollar", "cost", "price", "spent", "paid", "money"]):
            return "dollars"
        return None

    @staticmethod
    def _solution_outline(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups = [
            ("understand", "理解题意", {"problem_understanding", "variable_definition"}),
            ("model", "建模", {"modeling", "calculation_request", "reasoning_step"}),
            ("solve", "求解", {"cas_result"}),
            ("answer", "回答", {"answer_interpretation"}),
        ]
        outline: list[dict[str, Any]] = []
        for group_id, title, roles in groups:
            step_ids = [
                step["step_id"]
                for step in steps
                if (step.get("diagnostics") or {}).get("math_role") in roles
            ]
            if step_ids:
                outline.append({"id": group_id, "title": title, "step_ids": step_ids})
        return outline

    @staticmethod
    def _rewrite_step_references(chain_data: dict[str, Any], old_to_new: dict[str, str]) -> None:
        def rewrite_item(item: dict[str, Any]) -> None:
            source_step_id = item.get("source_step_id")
            if source_step_id in old_to_new:
                item["source_step_id"] = old_to_new[source_step_id]

        metadata = chain_data.get("metadata") or {}
        for item in metadata.get("final_answer_items") or []:
            rewrite_item(item)
        for step in chain_data.get("steps") or []:
            diagnostics = step.get("diagnostics") or {}
            final_items = (diagnostics.get("input") or {}).get("final_answer_items") or []
            for item in final_items:
                rewrite_item(item)

    @staticmethod
    def _display_name_for_target(
        target: str,
        variable_bindings: dict[str, Any],
        target_bindings: dict[str, Any],
        spr: SPR | None,
    ) -> str:
        if target in target_bindings:
            return target_bindings[target].get("text") or target
        if target in variable_bindings:
            return variable_bindings[target].get("description") or target
        lower_target = target.lower()
        for binding in target_bindings.values():
            text = str(binding.get("text") or "")
            if lower_target in text.lower():
                return text
        if spr is not None:
            for target_obj in spr.targets:
                text = target_obj.text or ""
                if lower_target in text.lower():
                    return text
        return target

    @staticmethod
    def _unit_for_target(
        target: str,
        display_name: str,
        variable_bindings: dict[str, Any],
        target_bindings: dict[str, Any],
    ) -> str | None:
        if target in variable_bindings and variable_bindings[target].get("unit"):
            return variable_bindings[target]["unit"]
        if target in target_bindings and target_bindings[target].get("unit"):
            return target_bindings[target]["unit"]
        return SCSAssemblyEnhancer._infer_unit(" ".join([target, display_name]))

    @staticmethod
    def _question_id_for_target(
        target: str,
        display_name: str,
        target_bindings: dict[str, Any],
        spr: SPR | None,
    ) -> str | None:
        if target in target_bindings:
            return target_bindings[target].get("question_id")
        if spr is None:
            return None
        lower_target = target.lower()
        lower_display_name = display_name.lower()
        best_target_id: str | None = None
        best_score = 0.0
        for spr_target in spr.targets:
            target_text = spr_target.text or ""
            target_text_lower = target_text.lower()
            score = max(
                SCSAssemblyEnhancer._semantic_match_score(target, target_text),
                SCSAssemblyEnhancer._semantic_match_score(display_name, target_text),
            )
            if (len(lower_target) > 1 and lower_target in target_text_lower) or (
                lower_display_name and lower_display_name in target_text_lower
            ):
                score = max(score, 1.0)
            if score > best_score:
                best_score = score
                best_target_id = spr_target.id
        if best_target_id and best_score >= 0.75:
            for question in spr.questions:
                if best_target_id in question.target_ids:
                    return question.id
        return None

    @staticmethod
    def _question_id_for_goal(goal: Any, spr: SPR | None) -> str | None:
        if spr is None:
            return None
        goal_text = " ".join(
            str(part or "")
            for part in [
                getattr(goal, "target", None),
                getattr(goal, "description", None),
            ]
        )
        best_question_id: str | None = None
        best_score = 0.0
        for question in spr.questions:
            question_text = getattr(question, "text", "") or ""
            score = SCSAssemblyEnhancer._semantic_match_score(goal_text, question_text)
            if goal.description and question_text:
                goal_description = str(goal.description)
                if goal_description in question_text or question_text in goal_description:
                    score = max(score, 1.0)
            for target_id in getattr(question, "target_ids", []) or []:
                target_obj = next((target for target in spr.targets if target.id == target_id), None)
                if target_obj is not None:
                    score = max(
                        score,
                        SCSAssemblyEnhancer._semantic_match_score(goal_text, target_obj.text),
                    )
            if score > best_score:
                best_score = score
                best_question_id = question.id
        return best_question_id if best_score >= 0.45 else None

    @staticmethod
    def _variable_usage(
        bindings: dict[str, dict[str, Any]],
        chain_data: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        symbols = set(bindings.keys())
        final_targets = {
            str(item.get("target"))
            for item in (chain_data or {}).get("metadata", {}).get("final_answer_items", [])
            if item.get("target") is not None
        }
        used_symbols: set[str] = set(final_targets & symbols)
        solved_symbols: set[str] = set()
        expression_symbols: set[str] = set()
        for item in (chain_data or {}).get("metadata", {}).get("final_answer_items", []):
            target = str(item.get("target") or "")
            if (
                target in symbols
                and item.get("backend_operation") in {"solve_for", "solve_equation_system"}
            ):
                solved_symbols.add(target)
        for step in (chain_data or {}).get("steps") or []:
            diagnostics = step.get("diagnostics") or {}
            request = diagnostics.get("normalized_request") or {}
            backend_operation = diagnostics.get("operation")
            if request:
                request_text = SCSAssemblyEnhancer._request_usage_text(request)
                expression_symbols.update(
                    symbol for symbol in symbols if SCSAssemblyEnhancer._contains_symbol(request_text, symbol)
                )
                target = request.get("target")
                if target in symbols:
                    used_symbols.add(str(target))
                    if request.get("operation") in {"solve_for", "solve_equation_system"}:
                        solved_symbols.add(str(target))
                used_symbols.update(expression_symbols)
            output = diagnostics.get("output") or {}
            for solution in output.get("raw_solution") or []:
                for symbol in solution.keys():
                    if symbol in symbols:
                        used_symbols.add(str(symbol))
                        if backend_operation in {"solve_for", "solve_equation_system"}:
                            solved_symbols.add(str(symbol))
        return {
            "used_symbols": used_symbols,
            "final_targets": final_targets,
            "solved_symbols": solved_symbols,
            "expression_symbols": expression_symbols,
        }

    @staticmethod
    def _request_usage_text(request: dict[str, Any]) -> str:
        chunks: list[str] = []
        for equation in request.get("equations") or []:
            chunks.extend([str(equation.get("lhs") or ""), str(equation.get("rhs") or "")])
        for expression in request.get("expressions") or []:
            chunks.append(str(expression.get("expression") or ""))
            chunks.append(str(expression.get("name") or ""))
        chunks.extend(str(value) for value in (request.get("substitutions") or {}).keys())
        chunks.append(str(request.get("target") or ""))
        return " ".join(chunks)

    @staticmethod
    def _contains_symbol(text: str, symbol: str) -> bool:
        if not symbol:
            return False
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
        return re.search(pattern, text) is not None

    @staticmethod
    def _variable_role(
        symbol: str,
        binding: dict[str, Any],
        usage: dict[str, set[str]],
        spr: SPR | None,
    ) -> str:
        if symbol in usage["final_targets"]:
            if symbol in usage["solved_symbols"]:
                return "primary_unknown"
            return "final_target"
        description = str(binding.get("description") or "")
        if spr is not None:
            for target in spr.targets:
                if SCSAssemblyEnhancer._semantic_match_score(description, target.text or "") >= 0.45:
                    return "final_target"
        if symbol in usage["solved_symbols"]:
            return "primary_unknown"
        if symbol in usage["used_symbols"]:
            return "computational_auxiliary"
        return "auxiliary_concept"

    @staticmethod
    def _explainer_priority(role: str) -> int:
        priorities = {
            "primary_unknown": 10,
            "final_target": 9,
            "computational_auxiliary": 6,
            "auxiliary_concept": 3,
        }
        return priorities.get(role, 1)

    @staticmethod
    def _semantic_match_score(left: str | None, right: str | None) -> float:
        left_tokens = SCSAssemblyEnhancer._semantic_tokens(left)
        right_tokens = SCSAssemblyEnhancer._semantic_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        containment = len(overlap) / min(len(left_tokens), len(right_tokens))
        jaccard = len(overlap) / len(left_tokens | right_tokens)
        return max(containment, jaccard)

    @staticmethod
    def _semantic_tokens(text: str | None) -> set[str]:
        if not text:
            return set()
        normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        normalized = normalized.replace("_", " ").replace("-", " ").lower()
        aliases = {
            "num": "number",
            "boxes": "box",
            "box": "box",
            "kg": "weight",
            "kilogram": "weight",
            "kilograms": "weight",
            "needed": "need",
            "required": "need",
            "total": "total",
            "weight": "weight",
            "original": "original",
            "transported": "transport",
            "apples": "apple",
            "apple": "apple",
        }
        stopwords = {"of", "the", "a", "an", "to", "in", "if", "each"}
        tokens = set()
        for token in re.findall(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]+", normalized):
            if token in stopwords:
                continue
            tokens.add(aliases.get(token, token))
        if "箱" in normalized:
            tokens.add("box")
        if "千克" in normalized or "重量" in normalized:
            tokens.add("weight")
        if "总" in normalized or "一共" in normalized:
            tokens.add("total")
        return tokens

    @staticmethod
    def _condition_ids(spr: SPR | None) -> list[str]:
        return [condition.id for condition in spr.conditions] if spr is not None else []

    @staticmethod
    def _question_ids(spr: SPR | None) -> list[str]:
        return [question.id for question in spr.questions] if spr is not None else []

    @staticmethod
    def _matching_condition_ids(spr: SPR | None, evidence: str | None) -> list[str]:
        if spr is None or not evidence:
            return []
        evidence_text = evidence.lower()
        return [
            condition.id
            for condition in spr.conditions
            if condition.text and (condition.text.lower() in evidence_text or evidence_text in condition.text.lower())
        ]

    @staticmethod
    def _matching_question_ids(spr: SPR | None, evidence: str | None) -> list[str]:
        if spr is None or not evidence:
            return []
        evidence_text = evidence.lower()
        return [
            question.id
            for question in spr.questions
            if question.text and (question.text.lower() in evidence_text or evidence_text in question.text.lower())
        ]

    @staticmethod
    def _infer_unit(text: str | None) -> str | None:
        if not text:
            return None
        lowered = text.lower()
        if "箱" in text or "box" in lowered:
            return "box"
        if "千克" in text or "kg" in lowered or "weight" in lowered:
            return "kg"
        if "页" in text or "page" in lowered:
            return "page"
        if "支" in text or "pencil" in lowered:
            return "piece"
        age_match = re.search(r"岁|age|year", lowered)
        if age_match:
            return "year"
        return None
