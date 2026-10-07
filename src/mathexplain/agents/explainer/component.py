"""Explainer v1: verified-SCS grounded pedagogical script generation."""

from __future__ import annotations

import re
from typing import Any

from mathexplain.agents.explainer.teaching_expansion import StudentMathFormatter
from mathexplain.agents.explainer.teaching_planner import RuleBasedTeachingPlanner
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.explanation import ExplanationScript
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.teaching_plan import TeachingPlan
from mathexplain.schemas.verification import VerificationReport


class Explainer:
    """Generate a structured pedagogical script without solving new math."""

    _DISPLAY_SYMBOLS = ["x", "y", "z", "t", "a", "b", "c", "m", "n"]
    _UNIT_MAP = {
        "kg": "千克",
        "kilogram": "千克",
        "kilograms": "千克",
        "box": "箱",
        "boxes": "箱",
        "page": "页",
        "pages": "页",
        "piece": "个",
        "pieces": "个",
        "year": "岁",
        "years": "岁",
        "m": "米",
        "meter": "米",
        "meters": "米",
        "km/h": "千米/小时",
        "cm": "厘米",
        "centimeter": "厘米",
        "centimeters": "厘米",
        "yuan": "元",
    }
    _ALLOWED_ENGLISH_IN_NARRATION: set[str] = set()

    def __init__(self, teaching_planner: RuleBasedTeachingPlanner | None = None) -> None:
        self.teaching_planner = teaching_planner or RuleBasedTeachingPlanner()
        self.last_teaching_plan: TeachingPlan | None = None

    def explain(
        self,
        spr: SPR,
        classification_result: ClassificationResult,
        scs: SolutionChain,
        verification_report: VerificationReport,
        audience_level: str = "standard",
        teaching_plan: TeachingPlan | None = None,
        output_language: str = "zh-CN",
        language_metadata: dict[str, Any] | None = None,
    ) -> ExplanationScript:
        """Generate an explanation script from a verified SolutionChain."""

        if verification_report.valid is not True:
            self.last_teaching_plan = None
            return self._unsupported_script(spr, scs, verification_report, audience_level)

        presentation = self._presentation_context(spr, scs, output_language=output_language)
        teaching_plan = teaching_plan or self.plan_teaching(
            spr,
            classification_result,
            scs,
            verification_report,
            presentation=presentation,
            output_language=output_language,
        )
        self.last_teaching_plan = teaching_plan
        problem_segments = self._problem_segments(spr, scs, output_language=output_language)
        variable_segments: list[dict[str, Any]] = []
        if not self._provider_plan_has_variable_moves(teaching_plan):
            if self._is_english(output_language):
                variable_segments.extend(self._english_variable_segments(scs, presentation))
            else:
                variable_segments.extend(self._variable_segments(scs, presentation))
        step_segments = self._step_segments(spr, scs, presentation, teaching_plan, output_language=output_language)
        answer_segments: list[dict[str, Any]] = []
        if not self._provider_plan_has_answer_moves(teaching_plan):
            if self._is_english(output_language):
                answer_segments.extend(self._english_answer_segments(scs, presentation))
            else:
                answer_segments.extend(self._answer_segments(scs, presentation))
        segments, symbol_consistency = self._enforce_symbol_consistency(
            [*problem_segments, *variable_segments, *step_segments, *answer_segments]
        )
        teaching_plan_summary = self._teaching_plan_summary(teaching_plan)
        anchors = self._visual_anchors(scs, presentation, segments)

        report = self._consistency_report(segments, anchors, spr, scs, presentation)
        return ExplanationScript.model_validate(
            {
                "problem_id": scs.problem_id,
                "source_chain_id": scs.chain_id,
                "status": "passed" if report["valid"] else "failed",
                "audience_level": audience_level,
                "segments": segments,
                "visual_anchors": anchors,
                "consistency_report": report,
                "metadata": {
                    "explainer": "Explainer",
                    "explainer_version": "Explainer v2 rule-based teaching expansion",
                    "source_verification_status": verification_report.status,
                    "source_verification_mode": verification_report.verification_mode,
                    "solver_route": classification_result.solver_route,
                    "task_type": classification_result.task_type,
                    "detected_source_language": (language_metadata or {}).get("detected_source_language"),
                    "output_language": output_language,
                    "language_policy": (language_metadata or {}).get("language_policy"),
                    "language": language_metadata or {"output_language": output_language},
                    "does_not_generate_eds": True,
                    "presentation": {
                        "symbols": presentation["symbols"],
                        "selected_answer_targets": [
                            item.get("target") for item in presentation["answers"]
                        ],
                    },
                    "symbol_consistency": symbol_consistency,
                    "teaching_expansion": {
                        "enabled": True,
                        "strategy": teaching_plan.metadata.get("strategy"),
                        "strategies": teaching_plan.quality_report.strategy_names,
                        "strategy_count": len(teaching_plan.quality_report.strategy_names),
                        "expanded_step_ids": teaching_plan.quality_report.expanded_step_ids,
                        "fallback_step_ids": teaching_plan.quality_report.fallback_step_ids,
                        "warnings": teaching_plan.quality_report.warnings,
                    },
                    "teaching_plan": teaching_plan_summary,
                },
            }
        )

    def plan_teaching(
        self,
        spr: SPR,
        classification_result: ClassificationResult,
        scs: SolutionChain,
        verification_report: VerificationReport,
        *,
        presentation: dict[str, Any] | None = None,
        output_language: str = "zh-CN",
    ) -> TeachingPlan:
        """Create a TeachingPlan before converting it to ExplanationScript."""

        del classification_result
        presentation = presentation or self._presentation_context(spr, scs, output_language=output_language)
        teaching_plan = self.teaching_planner.plan(
            spr,
            scs,
            verification_report,
            presentation,
            output_language=output_language,
        )
        self.last_teaching_plan = teaching_plan
        return teaching_plan

    @staticmethod
    def _unsupported_script(
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        audience_level: str,
    ) -> ExplanationScript:
        report = {
            "valid": False,
            "status": "unsupported",
            "checked_segment_count": 1,
            "errors": ["Explanation generation requires a passed VerificationReport."],
            "warnings": verification_report.warnings,
            "metadata": {"source_verification_status": verification_report.status},
        }
        return ExplanationScript.model_validate(
            {
                "problem_id": scs.problem_id or spr.metadata.get("problem_id", "unknown_problem"),
                "source_chain_id": scs.chain_id,
                "status": "unsupported",
                "audience_level": audience_level,
                "segments": [
                    {
                        "segment_id": "seg_unsupported",
                        "segment_type": "unsupported",
                        "linked_step_ids": [],
                        "narration": "当前解题链没有通过验证，因此不会生成正式教学讲解。",
                        "math_expressions": [],
                        "visual_refs": [],
                        "pedagogical_actions": ["SHOW_UNSUPPORTED_NOTICE"],
                        "pacing": "normal",
                        "estimated_duration_ms": 3500,
                        "pause_after_ms": 0,
                        "tts_hints": {"emphasis_terms": ["没有通过验证"]},
                        "layout_hints": {"focus_area": "status_notice"},
                        "metadata": {"source": "verification_report"},
                    }
                ],
                "visual_anchors": [],
                "consistency_report": report,
                "metadata": {
                    "explainer": "Explainer",
                    "explainer_version": "Explainer v1 rule-based",
                    "source_verification_status": verification_report.status,
                    "does_not_generate_eds": True,
                },
            }
        )

    def _presentation_context(
        self,
        spr: SPR,
        scs: SolutionChain,
        *,
        output_language: str = "zh-CN",
    ) -> dict[str, Any]:
        bindings = scs.metadata.get("variable_bindings") or {}
        answer_items = self._select_answer_bindings(spr, scs)
        answer_by_target = {
            str(item.get("target")): item
            for item in answer_items
            if item.get("target") is not None
        }
        selected_answer_targets = {
            str(item.get("target")) for item in answer_items if item.get("target") is not None
        }
        expression_symbols = self._symbols_in_explainer_ready_expressions(scs, bindings)
        expression_symbols.update(
            target
            for target in selected_answer_targets
            if self._symbol_appears_in_scs_expression(target, scs)
        )
        define_symbols = self._select_variable_symbols_for_explanation(
            bindings,
            expression_symbols,
            selected_answer_targets,
        )
        define_symbols.update(
            target
            for target, item in answer_by_target.items()
            if target in expression_symbols
            and self._is_student_friendly_symbol(target)
            and not self._is_box_answer(item)
        )

        symbol_map: dict[str, dict[str, Any]] = {}
        used_display_symbols: set[str] = set()
        for internal in self._ordered_symbols(bindings, define_symbols | expression_symbols | selected_answer_targets):
            binding = bindings.get(internal, {})
            display_symbol: str | None = None
            if internal in define_symbols or internal in expression_symbols:
                if self._is_student_friendly_symbol(internal) and internal not in used_display_symbols:
                    display_symbol = internal
                else:
                    display_symbol = self._next_display_symbol(used_display_symbols)
                used_display_symbols.add(display_symbol)
            if self._is_english(output_language):
                label = self._english_student_label_for_symbol(
                    internal,
                    binding,
                    spr,
                    answer_by_target.get(internal),
                )
            else:
                label = self._student_label_for_symbol(internal, binding, spr)
                if label == "这个未知量" and internal in answer_by_target:
                    label = self._student_label_for_answer(answer_by_target[internal], spr)
            symbol_map[internal] = {
                "internal_symbol": internal,
                "display_symbol": display_symbol,
                "student_label": label,
                "unit": self._localize_unit(
                    binding.get("unit")
                    or (answer_by_target.get(internal) or {}).get("unit"),
                    output_language=output_language,
                ),
                "variable_role": binding.get("variable_role"),
                "source_description": binding.get("description"),
            }

        answers: list[dict[str, Any]] = []
        for item in answer_items:
            target = str(item.get("target") or "")
            symbol_label = (symbol_map.get(target) or {}).get("student_label")
            display_value = self._display_value(item.get("value"))
            if self._is_english(output_language):
                answer_label = (
                    symbol_label
                    if symbol_label
                    and item.get("source") != "scs.final_answer"
                    and not self._contains_chinese(str(symbol_label))
                    else self._english_student_label_for_answer(item, spr)
                )
            else:
                answer_label = symbol_label or self._student_label_for_answer(item, spr)
            answers.append(
                {
                    **item,
                    "value": display_value,
                    "student_label": answer_label,
                    "unit": self._localize_unit(
                        self._unit_for_answer(item, spr) or item.get("unit"),
                        output_language=output_language,
                    ),
                    "question_label": self._question_label(
                        item.get("question_id"),
                        spr,
                        output_language=output_language,
                    ),
                }
            )

        return {
            "symbols": symbol_map,
            "answers": answers,
            "expression_aliases": self._expression_aliases(bindings, symbol_map, answers, output_language=output_language),
            "output_language": output_language,
        }

    @classmethod
    def _display_value(cls, value: Any) -> Any:
        if value is None:
            return value
        formatted = StudentMathFormatter.format_number_text(str(value))
        return formatted if formatted is not None else value

    @classmethod
    def _expression_aliases(
        cls,
        bindings: dict[str, Any],
        symbol_map: dict[str, dict[str, Any]],
        answers: list[dict[str, Any]],
        *,
        output_language: str,
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for internal, binding in bindings.items():
            compact = cls._compact_expression_label(
                internal,
                binding,
                (symbol_map.get(internal) or {}).get("student_label"),
                output_language=output_language,
            )
            if compact:
                aliases[str(internal)] = compact
                aliases[cls._brace_symbol_to_underscore(str(internal))] = compact
        for item in answers:
            target = str(item.get("target") or "")
            if not target:
                continue
            compact = cls._compact_expression_label(
                target,
                {},
                item.get("student_label"),
                output_language=output_language,
            )
            if compact:
                aliases[target] = compact
                aliases[cls._brace_symbol_to_underscore(target)] = compact
        return aliases

    @classmethod
    def _compact_expression_label(
        cls,
        symbol: str,
        binding: dict[str, Any],
        label: Any,
        *,
        output_language: str,
    ) -> str | None:
        if not cls._is_english(output_language):
            return None
        text = " ".join(
            str(part or "")
            for part in [
                label,
                binding.get("description"),
                symbol,
            ]
        ).lower()
        for color in ["yellow", "purple", "green", "red", "blue", "white", "black"]:
            if color in text:
                return color
        for token in ["total", "sum", "flowers", "flower"]:
            if token in text and ("total" in text or "sum" in text):
                return "total"
        match = re.fullmatch(r"[A-Za-z]", symbol.strip())
        if match:
            return symbol.strip()
        cleaned = re.sub(r"^N_\{?([A-Za-z0-9]+)\}?$", r"\1", symbol.strip())
        cleaned = cleaned.replace("_", " ").strip()
        return cleaned or None

    @staticmethod
    def _brace_symbol_to_underscore(symbol: str) -> str:
        return re.sub(r"_\{([^{}]+)\}", r"_\1", symbol)

    def _problem_segments(
        self,
        spr: SPR,
        scs: SolutionChain,
        *,
        output_language: str = "zh-CN",
    ) -> list[dict[str, Any]]:
        step_ids = self._outline_step_ids(scs, "understand")
        question_count = len(spr.questions)
        problem_text = self._problem_text_for_narration(spr, output_language=output_language)
        narration = "我们先读题。"
        if problem_text:
            narration += f"题目是：{problem_text}"
        narration += "先找出已知条件和要求的问题。"
        if question_count:
            narration += " 后面会按照题目中的问题逐一回答。"
        if self._is_english(output_language):
            narration = "Let's read the problem first. "
            if problem_text:
                narration += f"The problem says: {problem_text} "
            narration += "We will identify the given information and the question we need to answer."
            if question_count:
                narration += " Then we will answer the problem step by step."
        duration_ms = max(7600, min(15000, 4200 + len(narration) * 110))
        return [
            self._segment(
                "seg_problem_overview",
                "problem_overview",
                step_ids[:1],
                narration,
                [],
                ["problem_text"],
                ["SHOW_PROBLEM_TEXT", "HIGHLIGHT_QUESTION"],
                "normal",
                duration_ms,
                900,
                {"focus_area": "problem_context"},
                {"source": "spr", "question_count": question_count, "reads_problem_text": bool(problem_text)},
            )
        ]

    @staticmethod
    def _problem_text_for_narration(spr: SPR, *, output_language: str = "zh-CN") -> str:
        text = Explainer._clean_problem_text(spr.problem_text or spr.problem_stem or "")
        if not text:
            return ""
        if Explainer._is_english(output_language):
            return text if text.endswith((".", "?", "!")) else f"{text}."
        return text if text.endswith(("。", "？", "?", "！", "!")) else f"{text}。"

    @staticmethod
    def _clean_problem_text(text: str) -> str:
        """Remove dataset wrapper lines before narration or rendering."""

        filtered_lines: list[str] = []
        metadata_key_pattern = re.compile(
            r"^(?:sample[_\s-]*id|source[_\s-]*id|problem[_\s-]*id|dataset|split|subset)\s*:",
            re.IGNORECASE,
        )
        wrapper_titles = {
            "gsm8k math word problem",
            "math word problem",
        }
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = re.sub(r"\s+", " ", line).strip().lower()
            if normalized in wrapper_titles:
                continue
            if metadata_key_pattern.match(line):
                continue
            filtered_lines.append(line)
        return " ".join(filtered_lines).strip()

    def _variable_segments(self, scs: SolutionChain, presentation: dict[str, Any]) -> list[dict[str, Any]]:
        symbols = presentation["symbols"]
        steps_by_symbol = {
            step.diagnostics.get("symbol"): step
            for step in scs.steps
            if step.rule_name == "define_variable"
        }
        ordered = [
            item
            for item in symbols.values()
            if item.get("display_symbol") is not None
        ]
        ordered.sort(key=lambda item: self._DISPLAY_SYMBOLS.index(item["display_symbol"]) if item["display_symbol"] in self._DISPLAY_SYMBOLS else 99)

        segments: list[dict[str, Any]] = []
        for item in ordered:
            internal = item["internal_symbol"]
            display_symbol = item["display_symbol"]
            label = item["student_label"]
            unit = item.get("unit")
            unit_text = f"，单位是{unit}" if unit else ""
            step = steps_by_symbol.get(internal) or self._first_step_referencing_symbol(scs, internal)
            narration = f"设 {display_symbol} 表示{label}{unit_text}。"
            segments.append(
                self._segment(
                    f"seg_define_{self._safe_id(str(internal))}",
                    "variable_definition",
                    [step.step_id] if step is not None else [],
                    narration,
                    [str(display_symbol)],
                    [f"var_{self._safe_id(str(internal))}"],
                    ["DEFINE_VARIABLE", "HIGHLIGHT_TERMS"],
                    "normal",
                    4300,
                    300,
                    {"focus_area": "main_formula"},
                    {
                        "source": "presentation_normalizer",
                        "internal_symbol": internal,
                        "display_symbol": display_symbol,
                        "student_label": label,
                        "variable_role": item.get("variable_role"),
                    },
                    tts_hints={"emphasis_terms": [str(display_symbol), str(label)]},
                )
            )
        return segments

    def _english_variable_segments(self, scs: SolutionChain, presentation: dict[str, Any]) -> list[dict[str, Any]]:
        symbols = presentation["symbols"]
        steps_by_symbol = {
            step.diagnostics.get("symbol"): step
            for step in scs.steps
            if step.rule_name == "define_variable"
        }
        ordered = [
            item
            for item in symbols.values()
            if item.get("display_symbol") is not None
        ]
        ordered.sort(
            key=lambda item: (
                self._DISPLAY_SYMBOLS.index(item["display_symbol"])
                if item["display_symbol"] in self._DISPLAY_SYMBOLS
                else 99
            )
        )

        segments: list[dict[str, Any]] = []
        for item in ordered:
            internal = item["internal_symbol"]
            display_symbol = item["display_symbol"]
            label = item["student_label"]
            if self._is_vacuous_variable_label(display_symbol, label):
                continue
            unit = item.get("unit")
            unit_text = f", measured in {unit}" if unit else ""
            step = steps_by_symbol.get(internal) or self._first_step_referencing_symbol(scs, internal)
            narration = f"Let {display_symbol} represent {label}{unit_text}."
            segments.append(
                self._segment(
                    f"seg_define_{self._safe_id(str(internal))}",
                    "variable_definition",
                    [step.step_id] if step is not None else [],
                    narration,
                    [str(display_symbol)],
                    [f"var_{self._safe_id(str(internal))}"],
                    ["DEFINE_VARIABLE", "HIGHLIGHT_TERMS"],
                    "normal",
                    4300,
                    300,
                    {"focus_area": "main_formula"},
                    {
                        "source": "presentation_normalizer",
                        "internal_symbol": internal,
                        "display_symbol": display_symbol,
                        "student_label": label,
                        "variable_role": item.get("variable_role"),
                        "output_language": "en-US",
                    },
                    tts_hints={"emphasis_terms": [str(display_symbol), str(label)]},
                )
            )
        return segments

    def _step_segments(
        self,
        spr: SPR,
        scs: SolutionChain,
        presentation: dict[str, Any],
        teaching_plan: TeachingPlan,
        *,
        output_language: str = "zh-CN",
    ) -> list[dict[str, Any]]:
        if self._uses_provider_teaching_plan(teaching_plan):
            return self._segments_from_teaching_moves(list(teaching_plan.moves), presentation, scs=scs)

        segments: list[dict[str, Any]] = []
        moves_by_step_id: dict[str, list[Any]] = {}
        for move in teaching_plan.moves:
            for step_id in move.source_step_ids:
                moves_by_step_id.setdefault(step_id, []).append(move)
        for step in scs.steps:
            role = step.diagnostics.get("math_role")
            if role == "modeling" and step.diagnostics.get("explainer_ready") is True:
                planned_moves = moves_by_step_id.get(step.step_id, [])
                if planned_moves:
                    segments.extend(self._segments_from_teaching_moves(planned_moves, presentation))
                    continue
                expression = self._display_expression(step.math_expression, presentation)
                narration = self._modeling_narration(step, expression, output_language=output_language)
                segments.append(
                    self._segment(
                        f"seg_model_{step.step_id}",
                        "modeling",
                        [step.step_id],
                        narration,
                        [expression] if expression else [],
                        [f"expr_{step.step_id}"] if expression else [],
                        ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS", "QUESTION_PAUSE"],
                        "slow",
                        self._duration_for_text(narration, 6200),
                        600,
                        {"focus_area": "main_formula", "keep_previous_context": True},
                        {
                            "source": step.diagnostics.get("source"),
                            "evidence_span": step.diagnostics.get("evidence_span"),
                            "presentation_normalized": True,
                        },
                        tts_hints={"pause_points": ["对应的式子是"], "emphasis_terms": [expression] if expression else []},
                    )
                )
            elif (
                role == "cas_result"
                and step.diagnostics.get("explainer_ready") is True
                and not self._uses_provider_teaching_plan(teaching_plan)
            ):
                expression = self._cas_step_display_expression(step, presentation) or self._display_expression(
                    step.math_expression,
                    presentation,
                )
                operation = step.diagnostics.get("operation") or "calculate"
                narration = self._cas_narration(operation, expression, step, presentation, output_language=output_language)
                segments.append(
                    self._segment(
                        f"seg_calc_{step.step_id}",
                        "calculation",
                        [step.step_id],
                        narration,
                        [expression] if expression else [],
                        [f"expr_{step.step_id}"] if expression else [],
                        ["SHOW_RESULT", "HIGHLIGHT_RESULT"],
                        "normal",
                        self._duration_for_text(narration, 4200),
                        300,
                        {"focus_area": "main_formula", "keep_previous_context": True},
                        {"source": "cas", "operation": operation, "presentation_normalized": True},
                        tts_hints={"emphasis_terms": [expression] if expression else []},
                    )
                )
        return segments

    @classmethod
    def _cas_step_display_expression(cls, step: Any, presentation: dict[str, Any]) -> str | None:
        diagnostics = getattr(step, "diagnostics", {}) or {}
        operation = diagnostics.get("operation")
        if operation not in {"evaluate", "substitute"}:
            return None
        input_payload = diagnostics.get("input") or {}
        output_payload = diagnostics.get("output") or {}
        expression = str(input_payload.get("expression") or "").strip()
        result = cls._clean_display_value(output_payload.get("formatted_solution") or output_payload.get("value"))
        if not expression or not result:
            return None
        display_expression = StudentMathFormatter.display(expression, presentation) or expression
        if cls._last_expression_number(display_expression) == str(result):
            return display_expression
        return f"{display_expression} = {result}"

    def _enforce_symbol_consistency(self, segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Remove variable-definition scenes whose symbols are not reused later."""

        variable_segments = [segment for segment in segments if segment.get("segment_type") == "variable_definition"]
        if not variable_segments:
            return segments, {
                "checked": False,
                "removed_variable_definition_count": 0,
                "removed_symbols": [],
                "kept_symbols": [],
            }

        non_variable_text = "\n".join(
            self._segment_symbol_search_text(segment)
            for segment in segments
            if segment.get("segment_type") not in {"variable_definition", "answer", "recap"}
        )
        kept_ids: set[str] = set()
        removed: list[dict[str, Any]] = []
        kept_symbols: list[str] = []
        for segment in variable_segments:
            symbols = self._symbols_from_variable_segment(segment)
            used_symbols = [
                symbol for symbol in symbols
                if self._symbol_token_appears(symbol, non_variable_text)
            ]
            if used_symbols:
                kept_ids.add(str(segment.get("segment_id")))
                kept_symbols.extend(used_symbols)
                segment.setdefault("metadata", {})["symbol_consistency"] = "kept_reused_later"
                continue
            removed.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "symbols": symbols,
                    "reason": "defined_symbol_not_reused_in_later_teaching_segments",
                }
            )

        filtered = [
            segment for segment in segments
            if segment.get("segment_type") != "variable_definition"
            or str(segment.get("segment_id")) in kept_ids
        ]
        return filtered, {
            "checked": True,
            "removed_variable_definition_count": len(removed),
            "removed_symbols": removed,
            "kept_symbols": sorted(set(kept_symbols)),
        }

    @staticmethod
    def _segment_symbol_search_text(segment: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.extend(str(expression) for expression in segment.get("math_expressions") or [])
        metadata = segment.get("metadata") or {}
        for key in ("display_expression", "source_expression"):
            if metadata.get(key):
                parts.append(str(metadata[key]))
        return "\n".join(parts)

    @staticmethod
    def _symbols_from_variable_segment(segment: dict[str, Any]) -> list[str]:
        metadata = segment.get("metadata") or {}
        candidates: list[str] = []
        display_symbol = metadata.get("display_symbol")
        if display_symbol:
            candidates.append(str(display_symbol))
        for expression in segment.get("math_expressions") or []:
            text = str(expression).strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
                candidates.append(text)
        narration = str(segment.get("narration") or "")
        for pattern in (
            r"\bLet\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:represent|be|stand)",
            r"\b([A-Za-z][A-Za-z0-9_]*)\s+represent",
        ):
            candidates.extend(match.group(1) for match in re.finditer(pattern, narration, flags=re.IGNORECASE))
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _symbol_token_appears(symbol: str, text: str) -> bool:
        if not symbol:
            return False
        return re.search(rf"(?<![A-Za-z_]){re.escape(symbol)}(?![A-Za-z_])", text) is not None

    @staticmethod
    def _uses_provider_teaching_plan(teaching_plan: TeachingPlan) -> bool:
        return teaching_plan.metadata.get("provider_used") == "deepseek" or teaching_plan.planner_mode == "deepseek"

    @staticmethod
    def _strip_unverified_result_tail(narration: str, display_expression: Any) -> str:
        text = str(narration or "").strip()
        expression = str(display_expression or "")
        if not text or not expression:
            return text
        match = re.search(
            r"(?i)(?:\s+)?(?:This|That)\s+gives\s+([-+]?\d+(?:\.\d+)?)\s*\.?$",
            text,
        )
        if not match:
            return text
        result = match.group(1)
        expression_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", expression))
        if result in expression_numbers:
            return text
        return text[: match.start()].rstrip(" .") + "."

    def _segments_from_teaching_moves(
        self,
        moves: list[Any],
        presentation: dict[str, Any] | None = None,
        *,
        scs: SolutionChain | None = None,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        presentation = presentation or {}
        cas_display_by_source_step = self._cas_display_by_source_step(scs, presentation) if scs is not None else {}
        source_step_counts: dict[str, int] = {}
        for move in moves:
            step_id = move.source_step_ids[0] if move.source_step_ids else "unknown_step"
            source_step_counts[step_id] = source_step_counts.get(step_id, 0) + 1
        source_step_seen: dict[str, int] = {}
        answer_move_seen = 0
        for move in moves:
            step_id = move.source_step_ids[0] if move.source_step_ids else "unknown_step"
            source_step_seen[step_id] = source_step_seen.get(step_id, 0) + 1
            segment_id = move.metadata.get("source_segment_id") or f"seg_model_{self._safe_id(move.move_id)}"
            segment_type = self._segment_type_for_teaching_move(move)
            enhanced = cas_display_by_source_step.get(step_id)
            apply_enhanced = self._should_apply_cas_enhancement(
                move,
                enhanced,
                source_step_seen.get(step_id, 1),
                source_step_counts.get(step_id, 1),
            )
            display_expression = (
                enhanced.get("display_expression")
                if apply_enhanced and enhanced and not enhanced.get("preserve_move_expression")
                else StudentMathFormatter.display(move.display_expression, presentation)
            )
            narration = StudentMathFormatter.format_number_text(
                StudentMathFormatter.replace_symbols(move.narration, presentation) or move.narration
            ) or move.narration
            narration = self._strip_unverified_result_tail(narration, display_expression)
            if apply_enhanced and enhanced:
                narration = self._substitute_expression_for_display(
                    narration,
                    enhanced.get("display_substitutions") or {},
                )
                if self._should_append_result_for_expression(display_expression, enhanced.get("result")):
                    narration = self._append_result_to_narration(
                        narration,
                        enhanced.get("result"),
                        output_language=str(presentation.get("output_language") or ""),
                    )
            meaning = (
                StudentMathFormatter.format_number_text(
                    StudentMathFormatter.replace_symbols(move.meaning, presentation) or move.meaning
                )
                if move.meaning
                else move.meaning
            )
            source_sentence = (
                StudentMathFormatter.format_number_text(move.source_sentence)
                if move.source_sentence
                else move.source_sentence
            )
            answer_item: dict[str, Any] | None = None
            if segment_type == "answer" and (presentation.get("answers") or []):
                answers = list(presentation.get("answers") or [])
                answer_item = answers[min(answer_move_seen, len(answers) - 1)]
                answer_move_seen += 1
                output_language = str(presentation.get("output_language") or "")
                label = (
                    self._polish_english_label(answer_item.get("student_label") or "the requested quantity")
                    if self._is_english(output_language)
                    else str(answer_item.get("student_label") or "要求的数量")
                )
                value = answer_item.get("value")
                unit = answer_item.get("unit")
                unit_text = f" {unit}" if unit else ""
                display_expression = f"{label} = {value}{unit_text}".strip()
                if self._is_english(output_language):
                    narration = self._english_final_answer_narration(
                        label=label,
                        value=value,
                        unit_text=unit_text,
                        question_label=None,
                        option_label=answer_item.get("selected_option_label"),
                    )
                else:
                    narration = f"{label}是 {value}{unit_text}。"
            visual_refs = move.metadata.get("visual_refs")
            if visual_refs is None:
                visual_refs = [f"expr_{self._safe_id(move.move_id)}"] if display_expression else []
            elif not display_expression:
                visual_refs = [ref for ref in visual_refs if not str(ref).startswith("expr_")]
            layout_hints = move.metadata.get("layout_hints") or {
                "focus_area": "main_formula",
                "keep_previous_context": True,
            }
            tts_hints = move.metadata.get("tts_hints") or {
                "emphasis_terms": [display_expression] if display_expression else [],
            }
            metadata = {
                **dict(move.metadata),
                "source": move.metadata.get("source") or "teaching_plan",
                "source_step_id": step_id,
                "move_id": move.move_id,
                "move_type": move.move_type,
                "source_expression": move.raw_expression,
                "display_expression": display_expression,
                "substituted_expression": enhanced.get("substituted_expression") if apply_enhanced and enhanced else None,
                "cas_result": enhanced.get("result") if apply_enhanced and enhanced else None,
                "source_sentence": source_sentence,
                "meaning": meaning,
                "from_teaching_plan": True,
            }
            if answer_item is not None:
                metadata.update(
                    {
                        "source": "answer_bindings",
                        "internal_symbol": answer_item.get("target"),
                        "student_label": answer_item.get("student_label"),
                        "question_id": answer_item.get("question_id"),
                        "backend_confirmed": answer_item.get("backend_confirmed"),
                    }
                )
            linked_step_ids = list(move.source_step_ids)
            if segment_type == "answer" and not linked_step_ids:
                linked_step_ids = self._final_answer_linked_step_ids(scs, answer_item)
            if self._is_low_value_identity_segment(display_expression, narration):
                continue
            segments.append(
                self._segment(
                    segment_id,
                    segment_type,
                    linked_step_ids,
                    narration,
                    [display_expression] if display_expression else [],
                    list(visual_refs),
                    list(move.pedagogical_actions),
                    "slow",
                    move.estimated_duration_ms,
                    move.pause_after_ms,
                    dict(layout_hints),
                    metadata,
                    tts_hints=dict(tts_hints),
                )
            )
            solve_expression = enhanced.get("solve_expression") if apply_enhanced and enhanced else None
            if solve_expression:
                solve_narration = self._solve_narration(
                    solve_expression,
                    enhanced.get("result"),
                    metadata.get("meaning"),
                    output_language=str((presentation.get("language") or {}).get("output_language") or ""),
                )
                solve_segment_id = f"{segment_id}_solve"
                segments.append(
                    self._segment(
                        solve_segment_id,
                        "calculation",
                        list(move.source_step_ids),
                        solve_narration,
                        [solve_expression],
                        [f"expr_{self._safe_id(solve_segment_id)}"],
                        ["RENDER_EXPRESSION", "HIGHLIGHT_RESULT"],
                        "normal",
                        self._duration_for_text(solve_narration, 5200),
                        400,
                        dict(layout_hints),
                        {
                            **metadata,
                            "display_expression": solve_expression,
                            "source_expression": solve_expression,
                            "cas_result": enhanced.get("result"),
                            "synthetic_solve_step": True,
                        },
                        tts_hints={"emphasis_terms": [solve_expression]},
                    )
                )
        return segments

    @staticmethod
    def _should_apply_cas_enhancement(
        move: Any,
        enhanced: dict[str, str] | None,
        occurrence_index: int,
        occurrence_count: int,
    ) -> bool:
        if not enhanced:
            return False
        if occurrence_count <= 1:
            return True
        move_type = str(getattr(move, "move_type", "") or "").lower()
        if move_type == "answer":
            return True
        if occurrence_index >= occurrence_count:
            return True
        raw_expression = str(getattr(move, "display_expression", "") or "").strip()
        enhanced_expression = str(enhanced.get("display_expression") or "").strip()
        if not raw_expression or not enhanced_expression:
            return False
        return raw_expression == enhanced_expression

    @staticmethod
    def _is_low_value_identity_segment(expression: str | None, narration: str | None) -> bool:
        text = str(expression or "").strip()
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*=\s*\1", text)
        if not match:
            return False
        lowered = str(narration or "").lower()
        return any(word in lowered for word in ["from the problem", "we know", "speed is", "rate is"])

    @staticmethod
    def _solve_narration(
        solve_expression: str,
        result: str | None,
        meaning: str | None,
        *,
        output_language: str = "",
    ) -> str:
        if Explainer._is_english(output_language) or re.search(r"[A-Za-z]", str(meaning or "")):
            return f"Now solve for the requested amount: {solve_expression}."
        return f"继续求出要求的数量：{solve_expression}。"

    @classmethod
    def _cas_display_by_source_step(
        cls,
        scs: SolutionChain,
        presentation: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        by_step_id = {step.step_id: step for step in scs.steps}
        cas_by_tool_call: dict[str, Any] = {}
        for step in scs.steps:
            diagnostics = getattr(step, "diagnostics", {}) or {}
            tool_call_id = diagnostics.get("tool_call_id")
            output = diagnostics.get("output") or {}
            if (
                tool_call_id
                and step.rule_name in {
                    "cas_evaluate",
                    "cas_substitute",
                    "cas_solve",
                    "cas_solve_for",
                    "cas_solve_equation",
                }
                and output.get("formatted_solution") is not None
            ):
                cas_by_tool_call[str(tool_call_id)] = step

        display_by_step: dict[str, dict[str, str]] = {}
        for step in scs.steps:
            diagnostics = getattr(step, "diagnostics", {}) or {}
            if step.rule_name not in {"modeling_from_tool_call", "planner_evaluate", "planner_substitute", "planner_solve_for"}:
                continue
            tool_call_id = diagnostics.get("tool_call_id")
            if not tool_call_id:
                continue
            cas_step = cas_by_tool_call.get(str(tool_call_id))
            if cas_step is None:
                continue
            cas_diagnostics = getattr(cas_step, "diagnostics", {}) or {}
            cas_input = cas_diagnostics.get("input") or {}
            cas_output = cas_diagnostics.get("output") or {}
            target = str(cas_input.get("target") or (diagnostics.get("normalized_request") or {}).get("target") or "")
            result = cls._cas_result_value(cas_output, target)
            if not result:
                continue
            expression = str(cas_input.get("expression") or step.math_expression or "").strip()
            if not expression and cas_input.get("equations"):
                equation = (cas_input.get("equations") or [{}])[0]
                expression = f"{equation.get('lhs_sympy') or equation.get('lhs')} = {equation.get('rhs_sympy') or equation.get('rhs')}"
            substitutions = {
                str(symbol): cls._clean_display_value(value)
                for symbol, value in (cas_input.get("substitutions") or {}).items()
                if cls._clean_display_value(value) is not None
            }
            display_substitutions = cls._display_substitutions(substitutions, presentation)
            substituted = cls._substitute_expression_for_display(expression, substitutions)
            display_expression = StudentMathFormatter.display(substituted, presentation) or substituted
            result_display = StudentMathFormatter.format_number_text(result) or result
            solve_expression = cls._solve_expression_from_cas(cas_step, str(result_display), presentation)
            if cas_step.rule_name in {"cas_solve", "cas_solve_for", "cas_solve_equation"}:
                pass
            elif display_expression and cls._last_expression_number(display_expression) != str(result_display):
                display_expression = f"{display_expression} = {result_display}"
            elif display_expression and "=" not in display_expression:
                display_expression = f"{display_expression} = {result_display}"
            payload = {
                "display_expression": display_expression,
                "substituted_expression": substituted,
                "result": str(result_display),
                "display_substitutions": display_substitutions,
            }
            if solve_expression and solve_expression != display_expression:
                payload["solve_expression"] = solve_expression
                payload["preserve_move_expression"] = "true"
            display_by_step[step.step_id] = payload
            for dep in getattr(cas_step, "depends_on", []) or []:
                if dep in by_step_id:
                    display_by_step.setdefault(dep, payload)
        return display_by_step

    @classmethod
    def _cas_result_value(cls, cas_output: dict[str, Any], target: str) -> str | None:
        raw_solution = cas_output.get("raw_solution")
        if isinstance(raw_solution, list) and raw_solution:
            first = raw_solution[0]
            if isinstance(first, dict):
                if target and first.get(target) is not None:
                    return cls._clean_display_value(first.get(target))
                for value in first.values():
                    cleaned = cls._clean_display_value(value)
                    if cleaned is not None:
                        return cleaned
        return cls._clean_display_value(cas_output.get("formatted_solution") or cas_output.get("value"))

    @classmethod
    def _solve_expression_from_cas(
        cls,
        cas_step: Any,
        result: str,
        presentation: dict[str, Any],
    ) -> str | None:
        diagnostics = getattr(cas_step, "diagnostics", {}) or {}
        if getattr(cas_step, "rule_name", "") not in {"cas_solve", "cas_solve_for", "cas_solve_equation"}:
            return None
        cas_input = diagnostics.get("input") or {}
        target = str(cas_input.get("target") or "")
        equations = cas_input.get("equations") or []
        if not target or not equations:
            return None
        equation = equations[0]
        lhs = str(equation.get("lhs_sympy") or equation.get("lhs") or "")
        rhs = str(equation.get("rhs_sympy") or equation.get("rhs") or "")
        expression = cls._division_from_linear_product(lhs, rhs, target, result)
        if expression is None:
            expression = cls._division_from_linear_product(rhs, lhs, target, result)
        return StudentMathFormatter.display(expression, presentation) if expression else None

    @staticmethod
    def _division_from_linear_product(lhs: str, rhs: str, target: str, result: str) -> str | None:
        lhs_compact = re.sub(r"\s+", "", str(lhs or ""))
        rhs_compact = re.sub(r"\s+", "", str(rhs or ""))
        escaped = re.escape(target)
        match = re.fullmatch(rf"([-+]?\d+(?:\.\d+)?)\*{escaped}|{escaped}\*([-+]?\d+(?:\.\d+)?)", lhs_compact)
        if not match:
            return None
        factor = match.group(1) or match.group(2)
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs_compact):
            return None
        return f"{rhs_compact} / {factor} = {result}"

    @staticmethod
    def _display_substitutions(
        substitutions: dict[str, str],
        presentation: dict[str, Any],
    ) -> dict[str, str]:
        display: dict[str, str] = {}
        symbols = presentation.get("symbols") or {}
        for internal, value in substitutions.items():
            display_symbol = (symbols.get(internal) or {}).get("display_symbol")
            if display_symbol:
                display[str(display_symbol)] = str(value)
            display[str(internal)] = str(value)
        return display

    @staticmethod
    def _substitute_expression_for_display(expression: str, substitutions: dict[str, str]) -> str:
        substituted = str(expression or "")
        for symbol in sorted(substitutions, key=len, reverse=True):
            value = substitutions[symbol]
            substituted = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                str(value),
                substituted,
            )
        return substituted

    @staticmethod
    def _clean_display_value(value: Any) -> str | None:
        if value is None:
            return None
        text = StudentMathFormatter.format_number_text(str(value))
        return str(text).strip() if text is not None and str(text).strip() else None

    @staticmethod
    def _last_expression_number(expression: str) -> str | None:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(expression or ""))
        if not matches:
            return None
        value = matches[-1]
        return value.rstrip("0").rstrip(".") if "." in value else value

    @staticmethod
    def _append_result_to_narration(narration: str, result: str | None, *, output_language: str = "") -> str:
        if not result or result in re.findall(r"[-+]?\d+(?:\.\d+)?", narration):
            return narration
        if Explainer._is_english(output_language) or re.search(r"[A-Za-z]", narration):
            return f"{narration.rstrip()} This gives {result}."
        return f"{narration.rstrip()} 得到 {result}。"

    @staticmethod
    def _should_append_result_for_expression(display_expression: Any, result: Any) -> bool:
        expression = str(display_expression or "")
        result_text = str(result or "").strip()
        if not expression or not result_text:
            return False
        expression_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", expression))
        if result_text in expression_numbers:
            return True
        if "=" in expression and re.search(r"[A-Za-z]", expression):
            return False
        return True

    @staticmethod
    def _segment_type_for_teaching_move(move: Any) -> str:
        move_type = str(move.move_type or "").strip().lower()
        if move_type in {"answer", "calculation", "recap"}:
            return move_type
        return "modeling"

    @staticmethod
    def _provider_plan_has_answer_moves(teaching_plan: TeachingPlan) -> bool:
        return Explainer._uses_provider_teaching_plan(teaching_plan) and any(
            str(move.move_type or "").strip().lower() == "answer" for move in teaching_plan.moves
        )

    @staticmethod
    def _provider_plan_has_variable_moves(teaching_plan: TeachingPlan) -> bool:
        if not Explainer._uses_provider_teaching_plan(teaching_plan):
            return False
        for move in teaching_plan.moves:
            move_type = str(move.move_type or "").strip().lower()
            text = f"{move.meaning or ''} {move.narration or ''} {move.display_expression or ''}"
            if move_type in {"variable_definition", "define_variable"}:
                return True
            if "设" in text and "表示" in text:
                return True
        return False

    @staticmethod
    def _final_answer_linked_step_ids(
        scs: SolutionChain,
        answer_item: dict[str, Any] | None = None,
    ) -> list[str]:
        final_step = next((step for step in reversed(scs.steps) if step.rule_name == "final_answer"), None)
        if final_step is not None:
            return [final_step.step_id]
        source_step_id = str((answer_item or {}).get("source_step_id") or "").strip()
        if source_step_id and any(step.step_id == source_step_id for step in scs.steps):
            return [source_step_id]
        source_tool_call_id = str((answer_item or {}).get("source_tool_call_id") or "").strip()
        if source_tool_call_id:
            for step in reversed(scs.steps):
                diagnostics = getattr(step, "diagnostics", {}) or {}
                if str(diagnostics.get("tool_call_id") or "") == source_tool_call_id:
                    return [step.step_id]
        if scs.steps:
            return [scs.steps[-1].step_id]
        return []

    def _answer_segments(self, scs: SolutionChain, presentation: dict[str, Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for index, item in enumerate(presentation["answers"]):
            target = item.get("target")
            label = item.get("student_label") or "要求的数量"
            value = item.get("value")
            unit = item.get("unit")
            unit_text = f" {unit}" if unit else ""
            question_label = item.get("question_label")
            prefix = f"{question_label}，" if question_label else ""
            narration = f"{prefix}{label}是 {value}{unit_text}。"
            option_label = item.get("selected_option_label")
            option_text = item.get("selected_option_text")
            if option_label:
                narration = f"{prefix}计算结果是 {value}{unit_text}，对应选项 {option_label}。"
            display_symbol = (presentation["symbols"].get(str(target)) or {}).get("display_symbol")
            expression_target = label or display_symbol
            expression = f"{expression_target} = {value}{unit_text}".strip()
            if option_label:
                expression = f"{option_label}. {option_text or value}".strip()
            segments.append(
                self._segment(
                    f"seg_answer_{index + 1}",
                    "answer",
                    self._final_answer_linked_step_ids(scs, item),
                    narration,
                    [expression],
                    [f"answer_{index + 1}"],
                    ["FINAL_ANSWER_REVEAL", "HIGHLIGHT_RESULT"],
                    "normal",
                    self._duration_for_text(narration, 4300),
                    500,
                    {"focus_area": "final_answer"},
                    {
                        "source": "answer_bindings",
                        "internal_symbol": target,
                        "display_symbol": display_symbol,
                        "student_label": label,
                        "selected_option_id": item.get("selected_option_id"),
                        "selected_option_label": option_label,
                        "selected_option_text": option_text,
                        "question_id": item.get("question_id"),
                        "backend_confirmed": item.get("backend_confirmed"),
                    },
                    tts_hints={"emphasis_terms": [str(value), unit] if unit else [str(value)]},
                )
            )
        return segments

    def _recap_segment(self, scs: SolutionChain, presentation: dict[str, Any]) -> dict[str, Any]:
        recap_text = self._recap_takeaway_text(output_language="zh-CN")
        return self._segment(
            "seg_recap",
            "recap",
            [scs.steps[-1].step_id] if scs.steps else [],
            recap_text,
            [],
            [],
            ["RECAP_FAST"],
            "fast",
            min(self._duration_for_text(recap_text, 2200), 2600),
            0,
            {"focus_area": "recap"},
            {
                "source": "presentation_normalizer",
                "source_final_answer": scs.final_answer,
                "does_not_repeat_final_answer": True,
            },
            tts_hints={"emphasis_terms": ["验证", "单位"]},
        )

    def _english_answer_segments(self, scs: SolutionChain, presentation: dict[str, Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        include_question_prefix = len(presentation["answers"]) > 1
        for index, item in enumerate(presentation["answers"]):
            target = item.get("target")
            label = self._polish_english_label(item.get("student_label") or "the requested quantity")
            value = item.get("value")
            unit = item.get("unit")
            unit_text = f" {unit}" if unit else ""
            question_label = item.get("question_label")
            option_label = item.get("selected_option_label")
            option_text = item.get("selected_option_text")
            narration = self._english_final_answer_narration(
                label=label,
                value=value,
                unit_text=unit_text,
                question_label=question_label if include_question_prefix else None,
                option_label=option_label,
            )
            display_symbol = (presentation["symbols"].get(str(target)) or {}).get("display_symbol")
            expression_target = label or display_symbol
            expression = f"{expression_target} = {value}{unit_text}".strip()
            if option_label:
                expression = f"{option_label}. {option_text or value}".strip()
            segments.append(
                self._segment(
                    f"seg_answer_{index + 1}",
                    "answer",
                    self._final_answer_linked_step_ids(scs, item),
                    narration,
                    [expression],
                    [f"answer_{index + 1}"],
                    ["FINAL_ANSWER_REVEAL", "HIGHLIGHT_RESULT"],
                    "normal",
                    self._duration_for_text(narration, 4300),
                    500,
                    {"focus_area": "final_answer"},
                    {
                        "source": "answer_bindings",
                        "internal_symbol": target,
                        "display_symbol": display_symbol,
                        "student_label": label,
                        "selected_option_id": item.get("selected_option_id"),
                        "selected_option_label": option_label,
                        "selected_option_text": option_text,
                        "question_id": item.get("question_id"),
                        "backend_confirmed": item.get("backend_confirmed"),
                        "output_language": "en-US",
                    },
                    tts_hints={"emphasis_terms": [str(value), unit] if unit else [str(value)]},
                )
            )
        return segments

    def _english_recap_segment(self, scs: SolutionChain, presentation: dict[str, Any]) -> dict[str, Any]:
        recap_text = self._recap_takeaway_text(output_language="en-US")
        return self._segment(
            "seg_recap",
            "recap",
            [scs.steps[-1].step_id] if scs.steps else [],
            recap_text,
            [],
            [],
            ["RECAP_FAST"],
            "fast",
            min(self._duration_for_text(recap_text, 2200), 2600),
            0,
            {"focus_area": "recap"},
            {
                "source": "presentation_normalizer",
                "source_final_answer": scs.final_answer,
                "output_language": "en-US",
                "does_not_repeat_final_answer": True,
            },
            tts_hints={"emphasis_terms": ["verified", "units"]},
        )

    @staticmethod
    def _english_final_answer_narration(
        *,
        label: str,
        value: Any,
        unit_text: str,
        question_label: str | None,
        option_label: str | None,
    ) -> str:
        prefix = ""
        if question_label:
            prefix = f"For {str(question_label).strip().lower()}, "
        if option_label:
            return f"{prefix}the computed result is {value}{unit_text}, so the matching choice is {option_label}."
        clean_label = str(label or "").strip()
        leading = "the final answer" if prefix else "The final answer"
        if (
            clean_label
            and clean_label.lower() != "the requested quantity"
            and not Explainer._is_generic_english_answer_label(clean_label)
            and Explainer._is_safe_english_final_answer_label(clean_label)
        ):
            spoken_label = Explainer._english_spoken_label(clean_label)
            return f"{prefix}{leading} is {value}{unit_text}. This is {spoken_label}."
        return f"{prefix}{leading} is {value}{unit_text}."

    @staticmethod
    def _english_spoken_label(label: str) -> str:
        normalized = re.sub(r"\s+", " ", str(label or "").strip())
        if not normalized:
            return "the requested quantity"
        if re.match(r"(?i)^(the|a|an)\b", normalized):
            return normalized
        return f"the {normalized}"

    @staticmethod
    def _is_generic_english_answer_label(label: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(label or "").strip().lower())
        return normalized in {
            "answer",
            "result",
            "the answer",
            "the result",
            "the total",
            "total",
            "requested quantity",
            "the requested quantity",
        }

    @staticmethod
    def _is_safe_english_final_answer_label(label: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(label or "").strip().lower())
        unsafe_tokens = {
            "speed",
            "rate",
            "per blouse",
            "per skirt",
            "per pair",
            "cost per",
            "price per",
        }
        return not any(token in normalized for token in unsafe_tokens)

    @staticmethod
    def _polish_english_label(label: str) -> str:
        polished = str(label or "")
        for name in ["Sophia", "Rose", "Emily", "Mike", "Castle"]:
            polished = re.sub(rf"\b{name.lower()}\b", name, polished, flags=re.IGNORECASE)
        return polished

    @classmethod
    def _recap_takeaway_text(cls, *, output_language: str = "zh-CN") -> str:
        if cls._is_english(output_language):
            return "Done. The checked final answer was shown above."
        return "完成。上一页已经给出验证后的最终答案。"

    def _visual_anchors(
        self,
        scs: SolutionChain,
        presentation: dict[str, Any],
        segments: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = [
            {
                "anchor_id": "problem_text",
                "kind": "problem_text",
                "label": "题干",
                "metadata": {"source": "spr"},
            }
        ]
        for internal, item in presentation["symbols"].items():
            display_symbol = item.get("display_symbol")
            if display_symbol is None:
                continue
            anchors.append(
                {
                    "anchor_id": f"var_{self._safe_id(str(internal))}",
                    "kind": "variable",
                    "label": str(display_symbol),
                    "expression": str(display_symbol),
                    "metadata": {
                        "internal_symbol": internal,
                        "display_symbol": display_symbol,
                        "student_label": item.get("student_label"),
                        "source_description": item.get("source_description"),
                        "variable_role": item.get("variable_role"),
                    },
                }
            )
        used_visual_refs = {
            ref
            for segment in (segments or [])
            for ref in segment.get("visual_refs", [])
        }
        for step in scs.steps:
            if not step.math_expression:
                continue
            if step.diagnostics.get("math_role") not in {"modeling", "cas_result"}:
                continue
            if step.diagnostics.get("explainer_ready") is not True:
                continue
            if segments is not None and f"expr_{step.step_id}" not in used_visual_refs:
                continue
            display_expression = self._display_expression(step.math_expression, presentation)
            anchors.append(
                {
                    "anchor_id": f"expr_{step.step_id}",
                    "kind": "expression",
                    "label": display_expression,
                    "source_step_id": step.step_id,
                    "expression": display_expression,
                    "metadata": {
                        "math_role": step.diagnostics.get("math_role"),
                        "source_expression": step.math_expression,
                    },
                }
            )
        anchors.extend(self._teaching_visual_anchors(segments or []))
        for index, item in enumerate(presentation["answers"]):
            target = str(item.get("target") or "")
            display_symbol = (presentation["symbols"].get(target) or {}).get("display_symbol")
            label = display_symbol or item.get("student_label") or target
            unit_text = f" {item['unit']}" if item.get("unit") else ""
            expression = f"{label} = {item.get('value')}{unit_text}".strip()
            anchors.append(
                {
                    "anchor_id": f"answer_{index + 1}",
                    "kind": "answer",
                    "label": expression,
                    "expression": expression,
                    "metadata": {
                        "internal_symbol": target,
                        "display_symbol": display_symbol,
                        "question_id": item.get("question_id"),
                        "unit": item.get("unit"),
                    },
                }
            )
        anchors.append(
            {
                "anchor_id": "final_answer",
                "kind": "answer",
                "label": self._friendly_final_answer_text(
                    scs,
                    presentation,
                    output_language=str(presentation.get("output_language") or "zh-CN"),
                ),
                "expression": self._friendly_final_answer_text(
                    scs,
                    presentation,
                    output_language=str(presentation.get("output_language") or "zh-CN"),
                ),
                "metadata": {"source": "scs.final_answer"},
            }
        )
        return anchors

    def _teaching_visual_anchors(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for segment in segments:
            if not (
                segment.get("metadata", {}).get("from_teaching_plan")
                or segment.get("metadata", {}).get("source") == "teaching_expansion_rule"
            ):
                continue
            expressions = list(segment.get("math_expressions") or [])
            visual_refs = list(segment.get("visual_refs") or [])
            for index, expression in enumerate(expressions):
                if index >= len(visual_refs):
                    continue
                anchor_id = visual_refs[index]
                anchors.append(
                    {
                        "anchor_id": anchor_id,
                        "kind": "expression",
                        "label": expression,
                        "source_step_id": (
                            segment.get("metadata", {}).get("source_step_id")
                            or (segment.get("linked_step_ids") or [None])[0]
                        ),
                        "expression": expression,
                        "metadata": {
                            "math_role": "teaching_expansion",
                            "source": "teaching_expansion_rule",
                            "strategy": segment.get("metadata", {}).get("strategy"),
                            "move_id": segment.get("metadata", {}).get("move_id"),
                            "source_expression": segment.get("metadata", {}).get("source_expression"),
                            "source_sentence": segment.get("metadata", {}).get("source_sentence"),
                            "meaning": segment.get("metadata", {}).get("meaning"),
                        },
                    }
                )
        return anchors

    @staticmethod
    def _teaching_plan_summary(teaching_plan: TeachingPlan) -> dict[str, Any]:
        return {
            "schema_version": teaching_plan.schema_version,
            "planner_mode": teaching_plan.planner_mode,
            "move_count": len(teaching_plan.moves),
            "valid": teaching_plan.quality_report.valid,
            "status": teaching_plan.quality_report.status,
            "expanded_step_ids": teaching_plan.quality_report.expanded_step_ids,
            "fallback_step_ids": teaching_plan.quality_report.fallback_step_ids,
            "strategy_names": teaching_plan.quality_report.strategy_names,
        }

    def _friendly_final_answer_text(
        self,
        scs: SolutionChain,
        presentation: dict[str, Any],
        output_language: str = "zh-CN",
    ) -> str:
        if self._is_english(output_language) and presentation.get("answers"):
            choice_parts = [
                f"choice {item.get('selected_option_label')}, with result {item.get('value')}"
                f"{(' ' + item['unit']) if item.get('unit') else ''}"
                for item in presentation["answers"]
                if item.get("selected_option_label")
            ]
            if choice_parts:
                return "; ".join(choice_parts)
            return "; ".join(
                f"{item.get('student_label') or 'the answer'} is {item.get('value')}"
                f"{(' ' + item['unit']) if item.get('unit') else ''}"
                for item in presentation["answers"]
            )
        if presentation.get("answers"):
            choice_parts = [
                f"答案选 {item.get('selected_option_label')}，结果是 {item.get('value')}"
                f"{(' ' + item['unit']) if item.get('unit') else ''}"
                for item in presentation["answers"]
                if item.get("selected_option_label")
            ]
            if choice_parts:
                return "；".join(choice_parts)
            return "；".join(
                f"{item.get('student_label') or '答案'}是 {item.get('value')}"
                f"{(' ' + item['unit']) if item.get('unit') else ''}"
                for item in presentation["answers"]
            )
        return self._display_expression(scs.final_answer or "最终答案", presentation) or "最终答案"

    @classmethod
    def _english_student_label_for_answer(cls, item: dict[str, Any], spr: SPR) -> str:
        display_name = str(item.get("display_name") or "").strip()
        prefer_question_label = item.get("source") == "scs.final_answer"
        if display_name and not prefer_question_label and not cls._is_generic_symbol_text(display_name):
            cleaned_display = cls._clean_english_label(display_name)
            if cleaned_display:
                return cleaned_display
        for question in spr.questions:
            if item.get("question_id") and question.id != item.get("question_id"):
                continue
            text = str(question.text or "").strip()
            lower = text.lower()
            cost_match = re.search(r"(?i)\bhow\s+much\s+did\s+(.+?)\s+cost\b", text)
            if cost_match:
                entity = re.sub(r"(?i)\b(the|a|an)\b", "", cost_match.group(1))
                entity = re.sub(r"\s+", " ", entity).strip(" ?.!$")
                if entity:
                    return f"the cost of the {entity}"
            spend_match = re.search(r"(?i)\bhow\s+much\s+(?:does|do|did|will|would)\s+(.+?)\s+spend\s+on\s+(.+?)\?", text)
            if spend_match:
                subject = re.sub(r"\s+", " ", spend_match.group(1)).strip(" ?.!$")
                object_text = re.sub(r"\s+", " ", spend_match.group(2)).strip(" ?.!$")
                if object_text:
                    return f"the total amount {subject} spends on {object_text}"
            if re.search(r"(?i)\bhow\s+long\b", text):
                if "return home" in lower:
                    return "the total time to return home"
                return "the total time"
            if re.search(r"(?i)\bhow\s+much\s+more\s+money\b", text):
                return "the difference in money"
            if "net pay" in lower and any(token in lower for token in ["more", "difference", "compared"]):
                return "the difference in net pay"
            if "saving" in lower or re.search(r"\bsave\b", lower):
                return "the total money saved"
            if re.search(r"(?i)\bhow\s+much\s+money\b", text):
                return "the amount of money"
            distance_match = re.search(r"(?i)\bwhat\s+is\s+the\s+distance(?:\s*,?\s*in\s+[^,?]+,?)?\s+(.+?)\?", text)
            if distance_match:
                relation = re.sub(r"\s+", " ", distance_match.group(1)).strip(" ?.!$")
                if relation:
                    return f"the distance {relation}"
            if "how many" in lower:
                remainder = re.sub(r"(?i)^.*?how many\s+", "", text)
                remainder = re.sub(
                    r"(?i)\s+(does|do|did|are|is|were|was|will|would|can|could|should)\b.*$",
                    "",
                    remainder,
                )
                remainder = remainder.strip(" ?.!") 
                if remainder:
                    return f"the number of {remainder}"
            if "total" in lower or "altogether" in lower:
                return "the total"
        if display_name and not cls._is_generic_symbol_text(display_name):
            cleaned_display = cls._clean_english_label(display_name)
            if cleaned_display:
                return cleaned_display
        target = str(item.get("target") or "").strip()
        if target:
            cleaned = re.sub(r"[_{}\\]+", " ", target)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return cleaned
        return "the requested quantity"

    @classmethod
    def _english_student_label_for_symbol(
        cls,
        symbol: str,
        binding: dict[str, Any],
        spr: SPR,
        answer_item: dict[str, Any] | None = None,
    ) -> str:
        raw = str(binding.get("description") or "").strip()
        cleaned = ""
        lower = ""
        if raw and not cls._contains_chinese(raw):
            cleaned = re.sub(r"\s+", " ", raw).strip(" .")
            lower = cleaned.lower()
            if lower == str(symbol or "").strip().lower():
                cleaned = ""
                lower = ""
        if raw and cleaned:
            if lower.startswith("number of "):
                return f"the {lower}"
            if lower.startswith("total number of "):
                return f"the {lower}"
            if lower.startswith(("amount of ", "value of ", "total ")):
                return f"the {lower}"
            return cleaned[0].lower() + cleaned[1:] if cleaned else "the unknown quantity"
        if answer_item:
            return cls._english_student_label_for_answer(answer_item, spr)
        cleaned_symbol = re.sub(r"[_{}\\]+", " ", str(symbol)).strip()
        cleaned_symbol = re.sub(r"\s+", " ", cleaned_symbol)
        return cleaned_symbol or "the unknown quantity"

    @staticmethod
    def _is_generic_symbol_text(text: str) -> bool:
        cleaned = re.sub(r"[_{}\\]+", " ", str(text or "")).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return bool(re.fullmatch(r"[a-z]\d*", cleaned)) or cleaned in {"x", "y", "z", "answer", "result"}

    @staticmethod
    def _clean_english_label(text: str) -> str:
        cleaned = re.sub(r"[_{}\\]+", " ", str(text or "")).strip(" .")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned.lower().startswith("calculate "):
            cleaned = cleaned[len("calculate ") :]
        if cleaned.lower().startswith(("the ", "a ", "an ")):
            return cleaned[0].lower() + cleaned[1:]
        return cleaned[0].lower() + cleaned[1:] if cleaned else ""

    @staticmethod
    def _is_vacuous_variable_label(display_symbol: Any, label: Any) -> bool:
        symbol = re.sub(r"\s+", " ", str(display_symbol or "").strip().lower())
        text = re.sub(r"\s+", " ", str(label or "").strip().lower())
        text = re.sub(r"^(the\s+)?", "", text)
        return bool(symbol and text and symbol == text)

    @staticmethod
    def _segment(
        segment_id: str,
        segment_type: str,
        linked_step_ids: list[str],
        narration: str,
        math_expressions: list[str],
        visual_refs: list[str],
        actions: list[str],
        pacing: str,
        duration_ms: int,
        pause_after_ms: int,
        layout_hints: dict[str, Any],
        metadata: dict[str, Any],
        tts_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "segment_id": segment_id,
            "segment_type": segment_type,
            "linked_step_ids": linked_step_ids,
            "narration": narration,
            "math_expressions": [expr for expr in math_expressions if expr],
            "visual_refs": visual_refs,
            "pedagogical_actions": actions,
            "pacing": pacing,
            "estimated_duration_ms": duration_ms,
            "pause_after_ms": pause_after_ms,
            "tts_hints": tts_hints or {},
            "layout_hints": layout_hints,
            "metadata": metadata,
        }

    def _consistency_report(
        self,
        segments: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
        spr: SPR,
        scs: SolutionChain,
        presentation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        step_ids = {step.step_id for step in scs.steps}
        anchor_ids = {anchor["anchor_id"] for anchor in anchors}
        traceable_numbers = self._traceable_numbers(spr, scs)
        errors: list[str] = []
        warnings: list[str] = []
        for segment in segments:
            if segment["segment_type"] not in {"problem_overview", "unsupported"} and not segment["linked_step_ids"]:
                errors.append(f"{segment['segment_id']} has no linked_step_ids.")
            for step_id in segment["linked_step_ids"]:
                if step_id not in step_ids:
                    errors.append(f"{segment['segment_id']} references missing step_id={step_id}.")
            for ref in segment["visual_refs"]:
                if ref not in anchor_ids:
                    errors.append(f"{segment['segment_id']} references missing visual anchor {ref}.")
            for number in re.findall(r"\d+", segment["narration"]):
                if number not in traceable_numbers:
                    warnings.append(
                        f"{segment['segment_id']} uses number {number}, which is not traceable to SPR/SCS."
                    )
            for expression in segment.get("math_expressions") or []:
                for number in re.findall(r"\d+", str(expression)):
                    if number not in traceable_numbers:
                        warnings.append(
                            f"{segment['segment_id']} uses formula number {number}, which is not traceable to SPR/SCS."
                        )
            if segment["segment_type"] in {"modeling", "calculation"} and segment.get("math_expressions"):
                metadata = segment.get("metadata") or {}
                if not metadata.get("source_sentence"):
                    warnings.append(f"{segment['segment_id']} has formula(s) but no source_sentence.")
                if not metadata.get("meaning"):
                    warnings.append(f"{segment['segment_id']} has formula(s) but no meaning.")
            if segment["segment_type"] == "recap":
                final_numbers = set(re.findall(r"\d+", scs.final_answer or ""))
                recap_text = " ".join([segment.get("narration") or "", *segment.get("math_expressions", [])])
                repeated = sorted(number for number in final_numbers if number and number in re.findall(r"\d+", recap_text))
                if repeated:
                    errors.append(f"{segment['segment_id']} repeats final answer number(s): {', '.join(repeated)}.")
            warnings.extend(self._language_quality_warnings(segment))

        answer_segments = [segment for segment in segments if segment["segment_type"] == "answer"]
        selected_answers = (presentation or {}).get("answers") if presentation else scs.metadata.get("answer_bindings")
        if selected_answers and not answer_segments:
            errors.append("No answer segment was generated from selected answer_bindings.")
        if len(answer_segments) > len(spr.questions) and spr.questions:
            warnings.append("Answer segment count is greater than the number of SPR questions.")
        asked_question_ids = {question.id for question in spr.questions}
        for segment in answer_segments:
            question_id = segment.get("metadata", {}).get("question_id")
            if question_id and question_id not in asked_question_ids:
                warnings.append(f"{segment['segment_id']} answers a question_id not present in SPR: {question_id}.")
        if presentation:
            warnings.extend(self._duplicate_student_label_warnings(presentation))
        return {
            "valid": not errors,
            "status": "passed" if not errors else "failed",
            "checked_segment_count": len(segments),
            "errors": errors,
            "warnings": warnings,
            "metadata": {
                "traceable_number_count": len(traceable_numbers),
                "answer_segment_count": len(answer_segments),
            },
        }

    @staticmethod
    def _duplicate_student_label_warnings(presentation: dict[str, Any]) -> list[str]:
        label_to_symbols: dict[str, list[str]] = {}
        for symbol, item in (presentation.get("symbols") or {}).items():
            display_symbol = item.get("display_symbol")
            label = str(item.get("student_label") or "").strip()
            if not display_symbol or not label:
                continue
            label_to_symbols.setdefault(label, []).append(str(symbol))
        warnings: list[str] = []
        for label, symbols in sorted(label_to_symbols.items()):
            unique_symbols = sorted(set(symbols))
            if len(unique_symbols) > 1:
                warnings.append(
                    f"Multiple displayed variables share student_label={label}: {', '.join(unique_symbols)}."
                )
        return warnings

    @classmethod
    def _language_quality_warnings(cls, segment: dict[str, Any]) -> list[str]:
        narration = segment["narration"]
        warnings: list[str] = []
        if re.search(r"\bq_?\d+\b", narration, flags=re.IGNORECASE):
            warnings.append(f"{segment['segment_id']} contains raw question id in narration.")
        if re.search(r"\b(?:kg|box|boxes|kilograms?)\b", narration, flags=re.IGNORECASE):
            warnings.append(f"{segment['segment_id']} contains an unlocalized unit in narration.")
        english_words = [
            word
            for word in re.findall(r"[A-Za-z]{2,}", narration)
            if word.upper() not in cls._ALLOWED_ENGLISH_IN_NARRATION
        ]
        if english_words:
            warnings.append(f"{segment['segment_id']} contains English words in narration: {', '.join(sorted(set(english_words)))}.")
        return warnings

    @staticmethod
    def _traceable_numbers(spr: SPR, scs: SolutionChain) -> set[str]:
        chunks = [
            spr.problem_text or "",
            spr.problem_stem or "",
            scs.final_answer or "",
        ]
        chunks.extend(condition.text for condition in spr.conditions)
        chunks.extend(question.text for question in spr.questions)
        chunks.extend(str(step.math_expression or "") for step in scs.steps)
        for item in scs.metadata.get("answer_bindings") or []:
            chunks.extend(str(item.get(key) or "") for key in ("value", "unit", "display_name"))
        return set(re.findall(r"\d+", " ".join(chunks)))

    @staticmethod
    def _outline_step_ids(scs: SolutionChain, group_id: str) -> list[str]:
        for group in scs.metadata.get("solution_outline") or []:
            if group.get("id") == group_id:
                return list(group.get("step_ids") or [])
        return []

    @classmethod
    def _select_answer_bindings(cls, spr: SPR, scs: SolutionChain) -> list[dict[str, Any]]:
        answer_bindings = list(scs.metadata.get("answer_bindings") or [])
        final_answer_bindings = cls._answer_bindings_from_final_answer(spr, scs, answer_bindings)
        if final_answer_bindings:
            return final_answer_bindings
        if not answer_bindings:
            return []
        if not spr.questions:
            return answer_bindings

        selected: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for question in spr.questions:
            candidates = [
                (index, item, cls._answer_question_score(item, question, scs))
                for index, item in enumerate(answer_bindings)
                if index not in used_indexes
            ]
            candidates = [candidate for candidate in candidates if candidate[2] > 0]
            if not candidates:
                continue
            if cls._question_requests_multiple_answer_bindings(question, spr, candidates):
                for index, item, _score in candidates:
                    used_indexes.add(index)
                    selected_item = dict(item)
                    selected_item.setdefault("question_id", getattr(question, "id", None))
                    if not selected_item.get("question_id"):
                        selected_item["question_id"] = getattr(question, "id", None)
                    selected.append(selected_item)
                continue
            index, item, _score = max(candidates, key=lambda candidate: candidate[2])
            used_indexes.add(index)
            selected_item = dict(item)
            selected_item.setdefault("question_id", getattr(question, "id", None))
            if not selected_item.get("question_id"):
                selected_item["question_id"] = getattr(question, "id", None)
            selected.append(selected_item)
        return selected or answer_bindings

    @classmethod
    def _answer_bindings_from_final_answer(
        cls,
        spr: SPR,
        scs: SolutionChain,
        answer_bindings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        final_answer = str(getattr(scs, "final_answer", "") or "").strip()
        if not final_answer:
            return []
        pair_matches = re.findall(
            r"([A-Za-z][A-Za-z0-9_{}\\]*)\s*=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
            final_answer,
        )
        if len(pair_matches) >= 2:
            parts = [f"{target} = {value}" for target, value in pair_matches]
        else:
            parts = [part.strip() for part in re.split(r"[;；]\s*", final_answer) if part.strip()]
        parsed: list[dict[str, str]] = []
        for part in parts:
            target: str | None = None
            value_text = part
            if "=" in part:
                left, right = part.split("=", 1)
                target = left.strip()
                value_text = right.strip()
            value = cls._first_numeric_text(value_text) or value_text.strip()
            if not value:
                continue
            if target and cls._is_generic_answer_target(target):
                target = None
            parsed.append({"target": target or "", "value": value})
        parsed = cls._collapse_repeated_final_answer_targets(parsed, spr)
        if not parsed:
            value = cls._first_numeric_text(final_answer)
            if not value:
                return []
            parsed.append({"target": "", "value": value})

        bindings_by_target = {
            str(item.get("target") or ""): item
            for item in answer_bindings
            if item.get("target") is not None
        }
        default_question_id = getattr(spr.questions[-1], "id", None) if spr.questions else None
        inferred_target = cls._infer_answer_target_from_question(spr, answer_bindings)
        final_items: list[dict[str, Any]] = []
        for parsed_item in parsed:
            target = parsed_item["target"] or inferred_target
            source_binding = bindings_by_target.get(target) or cls._find_answer_binding_for_value(
                answer_bindings,
                parsed_item["value"],
            )
            if parsed_item["target"] == "" and source_binding.get("target"):
                target = str(source_binding.get("target"))
            final_item = dict(source_binding)
            final_item["target"] = target or final_item.get("target") or "final_answer"
            final_item["value"] = parsed_item["value"]
            final_item.setdefault("display_name", final_item["target"])
            if default_question_id and not final_item.get("question_id"):
                final_item["question_id"] = default_question_id
            final_item["source"] = "scs.final_answer"
            final_items.append(final_item)
        return final_items

    @classmethod
    def _collapse_repeated_final_answer_targets(
        cls,
        parsed: list[dict[str, str]],
        spr: SPR,
    ) -> list[dict[str, str]]:
        if len(parsed) <= 1:
            return parsed
        question_count = len(getattr(spr, "questions", []) or [])
        explicit_targets = [str(item.get("target") or "") for item in parsed if item.get("target")]
        if question_count <= 1 and explicit_targets and len(set(explicit_targets)) == 1:
            return [parsed[-1]]
        return parsed

    @classmethod
    def _find_answer_binding_for_value(
        cls,
        answer_bindings: list[dict[str, Any]],
        value: str,
    ) -> dict[str, Any]:
        normalized_value = cls._normalize_numeric_text(value)
        for item in answer_bindings:
            item_value = cls._first_numeric_text(str(item.get("value") or ""))
            if item_value is not None and cls._normalize_numeric_text(item_value) == normalized_value:
                return item
        return {}

    @staticmethod
    def _normalize_numeric_text(value: str) -> str:
        text = str(value or "").strip()
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return str(int(number))
        return f"{number:.12g}"

    @staticmethod
    def _first_numeric_text(text: str) -> str | None:
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _is_generic_answer_target(target: str) -> bool:
        cleaned = re.sub(r"\s+", " ", str(target or "").strip().lower())
        return cleaned in {"answer", "final answer", "result", "答案", "答案是", "最终答案"}

    @classmethod
    def _infer_answer_target_from_question(cls, spr: SPR, answer_bindings: list[dict[str, Any]]) -> str:
        if spr.questions:
            target_ids = list(getattr(spr.questions[-1], "target_ids", []) or [])
            if target_ids:
                return str(target_ids[-1])
        if len(answer_bindings) == 1 and answer_bindings[0].get("target"):
            return str(answer_bindings[0].get("target"))
        return "final_answer"

    @classmethod
    def _answer_question_score(cls, item: dict[str, Any], question: Any, scs: SolutionChain) -> float:
        target = str(item.get("target") or "")
        display_name = str(item.get("display_name") or "")
        question_text = str(getattr(question, "text", "") or "")
        text = " ".join([target, display_name]).lower()
        q_text = question_text.lower()
        score = 0.0
        if item.get("question_id") == getattr(question, "id", None):
            score += 1.0
        asks_general_count = "how many" in q_text and not any(
            token in q_text
            for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]
        )
        if asks_general_count:
            if any(token in text for token in ["total", "sum", "combined", "altogether"]):
                score += 8.0
            if any(token in text for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]):
                score -= 3.0
        target_bindings = scs.metadata.get("target_bindings") or {}
        direct_target_binding = target_bindings.get(target) or {}
        if direct_target_binding.get("question_id") == getattr(question, "id", None):
            score += 8.0
        direct_target_text = str(direct_target_binding.get("text") or "")
        if direct_target_text:
            target_question_score = cls._token_overlap_score(direct_target_text, question_text)
            if target_question_score >= 0.6:
                score += 6.0
            elif target_question_score >= 0.35:
                score += 3.0
        if any(token in q_text for token in ["box", "boxes", "箱", "几箱"]):
            if any(token in text for token in ["box", "boxes", "箱"]):
                score += 5.0
            if any(token in text for token in ["sold", "morning", "afternoon", "remaining"]):
                score -= 2.0
        if any(token in q_text for token in ["original", "transported", "total", "原来", "运来", "总", "一共", "买来"]):
            if any(token in text for token in ["total", "original", "transported", "原来", "运来", "总", "一共", "买来"]):
                score += 5.0
            if any(token in text for token in ["sold", "morning", "afternoon", "remaining", "per class", "class", "box", "上午", "下午", "卖出", "剩余", "剩下", "每班", "箱"]):
                score -= 4.0
        for target_id in getattr(question, "target_ids", []) or []:
            target_text = str((target_bindings.get(target_id) or {}).get("text") or "").lower()
            if target_text and cls._token_overlap_score(text, target_text) >= 0.35:
                score += 3.0
        variable_role = (scs.metadata.get("variable_bindings") or {}).get(target, {}).get("variable_role")
        if variable_role == "primary_unknown":
            score += 0.7
        elif variable_role == "final_target":
            score += 0.5
        if item.get("backend_confirmed") is True:
            score += 0.3
        return score

    @classmethod
    def _question_requests_multiple_answer_bindings(
        cls,
        question: Any,
        spr: SPR,
        candidates: list[tuple[int, dict[str, Any], float]],
    ) -> bool:
        if len(candidates) <= 1:
            return False
        candidate_targets = {
            str(item.get("target") or "")
            for _index, item, _score in candidates
            if item.get("target")
        }
        if len(candidate_targets) <= 1:
            return False
        target_texts = [
            target.text or ""
            for target in spr.targets
            if target.id in (getattr(question, "target_ids", []) or [])
        ]
        combined_text = " ".join([getattr(question, "text", "") or "", *target_texts])
        lowered = combined_text.lower().replace("_", " ")
        if "how many" in lowered and not any(
            token in lowered
            for token in ["yellow", "purple", "green", "red", "blue", "white", "black"]
        ):
            return False
        if len(target_texts) >= 2:
            return True
        return any(
            signal in lowered
            for signal in ["x and y", "both", "respectively", "solution set", "ordered pair"]
        ) or any(
            signal in combined_text
            for signal in ["各有多少", "分别", "各是多少", "各为多少", "鸡和兔", "两种", "二者"]
        )

    @staticmethod
    def _token_overlap_score(left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+|\d+", left.lower()))
        right_tokens = set(re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+|\d+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))

    @classmethod
    def _symbols_in_explainer_ready_expressions(
        cls,
        scs: SolutionChain,
        bindings: dict[str, Any],
    ) -> set[str]:
        symbols: set[str] = set()
        for step in scs.steps:
            role = step.diagnostics.get("math_role")
            if role not in {"modeling", "cas_result"} or step.diagnostics.get("explainer_ready") is not True:
                continue
            expression = step.math_expression or ""
            for symbol in bindings:
                if cls._contains_symbol(expression, symbol):
                    symbols.add(symbol)
        return symbols

    @classmethod
    def _symbol_appears_in_scs_expression(cls, symbol: str, scs: SolutionChain) -> bool:
        return any(
            cls._contains_symbol(step.math_expression or "", symbol)
            for step in scs.steps
            if step.math_expression
            and step.diagnostics.get("math_role") in {"modeling", "cas_result"}
        )

    @classmethod
    def _is_box_answer(cls, item: dict[str, Any]) -> bool:
        text = " ".join(str(item.get(key) or "") for key in ("target", "display_name", "unit")).lower()
        return "box" in text or "箱" in text

    @classmethod
    def _first_step_referencing_symbol(cls, scs: SolutionChain, symbol: str) -> Any | None:
        return next(
            (
                step
                for step in scs.steps
                if step.math_expression and cls._contains_symbol(step.math_expression, symbol)
            ),
            None,
        )

    @classmethod
    def _select_variable_symbols_for_explanation(
        cls,
        bindings: dict[str, Any],
        expression_symbols: set[str],
        selected_answer_targets: set[str],
    ) -> set[str]:
        selected: set[str] = set()
        for symbol, binding in bindings.items():
            role = binding.get("variable_role")
            if symbol in expression_symbols and role in {"primary_unknown", "computational_auxiliary"}:
                selected.add(symbol)
            elif symbol in selected_answer_targets and role == "primary_unknown":
                selected.add(symbol)
        return selected

    @staticmethod
    def _ordered_symbols(bindings: dict[str, Any], symbols: set[str]) -> list[str]:
        def sort_key(symbol: str) -> tuple[int, str]:
            role = (bindings.get(symbol) or {}).get("variable_role")
            priority = {"primary_unknown": 0, "computational_auxiliary": 1, "final_target": 2}.get(role, 3)
            return priority, symbol

        return sorted(symbols, key=sort_key)

    @classmethod
    def _next_display_symbol(cls, used_symbols: set[str]) -> str:
        for candidate in cls._DISPLAY_SYMBOLS:
            if candidate not in used_symbols:
                return candidate
        index = len(used_symbols) + 1
        while True:
            candidate = f"u{index}"
            if candidate not in used_symbols:
                return candidate
            index += 1

    @staticmethod
    def _is_student_friendly_symbol(symbol: str) -> bool:
        return bool(re.fullmatch(r"[a-z]", symbol))

    @classmethod
    def _student_label_for_symbol(cls, symbol: str, binding: dict[str, Any], spr: SPR) -> str:
        raw = " ".join(
            str(part or "")
            for part in [
                symbol,
                binding.get("description"),
                binding.get("domain"),
            ]
        )
        raw_label = cls._student_label_from_text(raw, fallback="这个未知量")
        if raw_label != "这个未知量" and cls._is_specific_symbol_label(raw_label):
            return raw_label
        for question in spr.questions:
            if cls._text_mentions_symbol_target(str(question.text or ""), symbol, binding):
                return cls._student_label_from_text(question.text, fallback="这个未知量")
        return raw_label

    @classmethod
    def _student_label_for_answer(cls, item: dict[str, Any], spr: SPR) -> str:
        question = cls._question_by_id(spr, item.get("question_id"))
        if question is not None:
            return cls._student_label_from_text(question.text, fallback="要求的数量")
        raw = " ".join(
            str(part or "")
            for part in [
                item.get("target"),
                item.get("display_name"),
            ]
        )
        return cls._student_label_from_text(raw, fallback="要求的数量")

    @classmethod
    def _unit_for_answer(cls, item: dict[str, Any], spr: SPR) -> str | None:
        question = cls._question_by_id(spr, item.get("question_id"))
        question_unit = cls._unit_from_problem_text(item, spr, question=question)
        if question_unit:
            return question_unit
        question_text = str(getattr(question, "text", "") or "")
        direct_question_unit = cls._unit_from_text(question_text)
        if direct_question_unit:
            return direct_question_unit
        inherited_unit = cls._single_measurement_unit_from_problem(spr)
        if inherited_unit and str(item.get("unit") or "").strip().lower() != inherited_unit:
            return inherited_unit
        text = " ".join(
            str(part or "")
            for part in [
                item.get("target"),
                item.get("display_name"),
                question.text if question is not None else "",
            ]
        )
        lower = text.lower()
        if any(token in lower for token in ["speed", "per hour"]) or any(
            token in text for token in ["速度", "每小时"]
        ):
            return "km/h"
        if "box" in lower or "箱" in text:
            return "box"
        if "sticker" in lower:
            return "stickers"
        if "second" in lower or "time" in lower:
            return "seconds"
        if "kg" in lower or "kilogram" in lower or "千克" in text:
            return "kg"
        if "meter" in lower or "米" in text or "长度" in text or "彩带" in text:
            return "m"
        return None

    @classmethod
    def _unit_from_problem_text(cls, item: dict[str, Any], spr: SPR, *, question: Any | None = None) -> str | None:
        chunks: list[str] = []
        if question is not None:
            chunks.append(str(getattr(question, "text", "") or ""))
            linked_target_ids = set(getattr(question, "target_ids", []) or [])
            chunks.extend(str(target.text or "") for target in spr.targets if target.id in linked_target_ids)
        chunks.extend(
            str(part or "")
            for part in [
                item.get("target"),
                item.get("display_name"),
            ]
        )
        return cls._unit_from_text(" ".join(chunks))

    @staticmethod
    def _unit_from_text(text: str) -> str | None:
        lower = str(text or "").lower()
        if "$" in str(text or "") or any(
            token in lower
            for token in [
                "cost",
                "price",
                "spent",
                "spend",
                "paid",
                "money",
                "pay",
                "salary",
                "dollar",
                "save",
                "saving",
            ]
        ):
            return "dollars"
        if re.search(r"\b(pounds?|lbs?)\b", lower):
            return "pounds"
        if re.search(r"\b(bandages?)\b", lower):
            return "bandages"
        if re.search(r"\b(stamps?)\b", lower):
            return "stamps"
        if re.search(r"\b(stickers?)\b", lower):
            return "stickers"
        if re.search(r"\b(seconds?|secs?)\b", lower):
            return "seconds"
        if re.search(r"\b(minutes?|mins?)\b", lower):
            return "minutes"
        if re.search(r"\b(hours?|hrs?)\b", lower):
            return "hours"
        if re.search(r"\b(oranges?)\b", lower):
            return "oranges"
        if re.search(r"\b(cookies?)\b", lower):
            return "cookies"
        if re.search(r"\b(dollars?)\b", lower):
            return "dollars"
        if re.search(r"\b(cents?)\b", lower):
            return "cents"
        if re.search(r"\b(box|boxes)\b", lower):
            return "box"
        if re.search(r"\b(miles?\s+per\s+hour|mph)\b", lower):
            return "mph"
        if re.search(r"\b(miles?)\b", lower):
            return "miles"
        if re.search(r"\b(kilometers?\s+per\s+hour|km/h|kph)\b", lower):
            return "km/h"
        if re.search(r"\b(kilograms?|kg)\b", lower):
            return "kg"
        if re.search(r"\b(meters?|metres?)\b", lower):
            return "m"
        return None

    @staticmethod
    def _single_measurement_unit_from_problem(spr: SPR) -> str | None:
        chunks = [
            str(getattr(spr, "problem_text", "") or ""),
            str(getattr(spr, "problem_stem", "") or ""),
        ]
        chunks.extend(str(condition.text or "") for condition in getattr(spr, "conditions", []) or [])
        text = " ".join(chunks).lower()
        unit_patterns = {
            "pounds": r"\b(pounds?|lbs?)\b",
            "kg": r"\b(kilograms?|kg)\b",
            "dollars": r"\b(dollars?)\b",
            "cents": r"\b(cents?)\b",
            "seconds": r"\b(seconds?|secs?)\b",
            "minutes": r"\b(minutes?|mins?)\b",
            "hours": r"\b(hours?|hrs?)\b",
            "mph": r"\b(miles?\s+per\s+hour|mph)\b",
            "miles": r"\b(miles?)\b",
            "km/h": r"\b(kilometers?\s+per\s+hour|km/h|kph)\b",
            "m": r"\b(meters?|metres?)\b",
        }
        present = [unit for unit, pattern in unit_patterns.items() if re.search(pattern, text)]
        return present[0] if len(present) == 1 else None

    @staticmethod
    def _question_by_id(spr: SPR, question_id: str | None) -> Any | None:
        if question_id is None:
            return None
        return next((question for question in spr.questions if question.id == question_id), None)

    @classmethod
    def _student_label_from_text(cls, text: str, fallback: str) -> str:
        text = cls._clean_student_label_text(text)
        lower = text.lower()
        has_apple = "apple" in lower or "苹果" in text
        has_remaining = "remaining" in lower or "剩" in text
        if any(token in lower for token in ["box", "boxes"]) or "箱" in text or "几箱" in text:
            return "可以装的箱数"
        if (any(token in lower for token in ["pencil", "pencils"]) or "铅笔" in text) and any(
            token in lower for token in ["6 class", "6 classes", "6class", "6classes", "six class", "six classes"]
        ):
            return "平均分给 6 个班时每班分到的铅笔数量"
        if (any(token in lower for token in ["pencil", "pencils"]) or "铅笔" in text) and any(
            token in lower for token in ["8 class", "8 classes", "8class", "8classes", "eight class", "eight classes"]
        ):
            return "平均分给 8 个班时每班分到的铅笔数量"
        if (any(token in lower for token in ["pencil", "pencils"]) or "铅笔" in text) and any(
            token in lower for token in ["per class", "each class", "class"]
        ):
            return "每班分到的铅笔数量"
        if any(token in lower for token in ["truck", "cargo"]) or "\u8d27\u8f66" in text:
            return "\u8d27\u8f66\u5e73\u5747\u901f\u5ea6"
        if any(token in lower for token in ["passenger", "bus"]) or "\u5ba2\u8f66" in text:
            return "\u5ba2\u8f66\u5e73\u5747\u901f\u5ea6"
        if any(token in lower for token in ["chicken", "chickens", "hen", "hens"]) or "鸡" in text:
            return "鸡的数量"
        if any(token in lower for token in ["rabbit", "rabbits"]) or "兔" in text:
            return "兔的数量"
        if any(token in lower for token in ["speed"]) or any(
            token in text for token in ["客车", "速度", "每小时"]
        ):
            return "客车平均速度"
        if "彩带" in text or ("ribbon" in lower and any(token in lower for token in ["length", "original"])):
            return "彩带原来的长度"
        if ("morning" in lower or "上午" in text) and has_apple:
            if has_remaining:
                return "上午卖出后剩下的苹果重量"
            return "上午卖出的苹果重量"
        if ("afternoon" in lower or "下午" in text) and has_apple:
            if has_remaining:
                return "下午卖出后剩下的苹果重量"
            return "下午卖出的苹果重量"
        if has_remaining and has_apple:
            return "剩下的苹果重量"
        if any(token in lower for token in ["original", "transported", "total", "weight"]) or any(
            token in text for token in ["原来", "运来", "总", "重量", "多少千克", "一共"]
        ):
            if has_apple:
                return "原来运来的苹果重量"
            if "铅笔" in text:
                return "买来的铅笔总数"
            return "要求的总量"
        if "page" in lower or "页" in text:
            return "要求的页数"
        if "year" in lower or "岁" in text or "年龄" in text:
            return "要求的年龄"
        if cls._contains_chinese(text):
            cleaned = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", "", text)
            cleaned = re.sub(r"\s+", "", cleaned)
            cleaned = cls._strip_question_numbering(cleaned)
            if 2 <= len(cleaned) <= 16:
                return cleaned
        return fallback

    @staticmethod
    def _is_specific_symbol_label(label: str) -> bool:
        return any(
            token in label
            for token in [
                "上午",
                "下午",
                "剩下",
                "剩余",
                "每班",
                "可以装",
                "卖出",
            ]
        )

    @classmethod
    def _clean_student_label_text(cls, text: Any) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"\([^()\u4e00-\u9fff]*[A-Za-z][^()\u4e00-\u9fff]*\)", "", cleaned)
        cleaned = re.sub(r"（[^（）\u4e00-\u9fff]*[A-Za-z][^（）\u4e00-\u9fff]*）", "", cleaned)
        cleaned = cls._strip_question_numbering(cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    @staticmethod
    def _strip_question_numbering(text: str) -> str:
        cleaned = str(text or "").strip()
        patterns = [
            r"^[（(]\s*\d+\s*[)）]",
            r"^\d+\s*[.．、)]",
            r"^[一二三四五六七八九十]+\s*[、.]",
        ]
        changed = True
        while changed:
            changed = False
            for pattern in patterns:
                updated = re.sub(pattern, "", cleaned).strip()
                if updated != cleaned:
                    cleaned = updated
                    changed = True
        return cleaned

    @staticmethod
    def _text_mentions_symbol_target(text: str, symbol: str, binding: dict[str, Any]) -> bool:
        combined = " ".join([symbol, str(binding.get("description") or "")]).lower()
        lower_text = text.lower()
        if any(token in combined for token in ["morning", "afternoon", "sold", "remaining", "上午", "下午", "卖出", "剩下", "剩余"]):
            if any(token in lower_text for token in ["original", "transported", "total", "原来", "运来", "总"]):
                return False
        if any(token in combined for token in ["per class", "each class", "class", "每班", "每个班"]):
            return False
        if "铅笔" in text and any(token in combined for token in ["pencil", "铅笔"]):
            return any(token in lower_text for token in ["total", "一共", "总", "买来"])
        if "苹果" in text and any(token in combined for token in ["apple", "苹果"]):
            return any(token in lower_text for token in ["total", "original", "原来", "运来", "总"])
        if "彩带" in text and any(token in combined for token in ["ribbon", "length", "彩带", "长度"]):
            return any(token in lower_text for token in ["原来", "长", "长度"])
        return False

    @classmethod
    def _localize_unit(cls, unit: Any, *, output_language: str = "zh-CN") -> str | None:
        if unit is None:
            return None
        text = str(unit).strip()
        if not text:
            return None
        if cls._is_english(output_language):
            return text
        return cls._UNIT_MAP.get(text.lower(), text)

    @classmethod
    def _question_label(
        cls,
        question_id: str | None,
        spr: SPR,
        *,
        output_language: str = "zh-CN",
    ) -> str | None:
        question_ids = [question.id for question in spr.questions]
        if question_id not in question_ids:
            return None
        index = question_ids.index(question_id)
        if cls._is_english(output_language):
            return f"Question {index + 1}"
        labels = ["第一个问题", "第二个问题", "第三个问题", "第四个问题", "第五个问题"]
        return labels[index] if index < len(labels) else f"第{index + 1}个问题"

    @classmethod
    def _display_expression(cls, expression: str | None, presentation: dict[str, Any]) -> str | None:
        return StudentMathFormatter.display(expression, presentation)

    @staticmethod
    def _is_english(output_language: str | None) -> bool:
        return str(output_language or "").lower().startswith("en")

    @staticmethod
    def _modeling_narration(
        step: Any,
        expression: str | None,
        *,
        output_language: str = "zh-CN",
    ) -> str:
        if Explainer._is_english(output_language):
            operation = step.diagnostics.get("operation")
            if operation in {"solve_for", "solve_equation_system"}:
                narration = "Set up an equation from the information in the problem."
            elif operation in {"evaluate", "substitute"}:
                narration = "Substitute the known result into the next calculation."
            else:
                narration = "Translate the problem information into a mathematical relationship."
            if expression:
                narration += f" The expression is {expression}."
            return narration
        operation = step.diagnostics.get("operation")
        if operation in {"solve_for", "solve_equation_system"}:
            narration = "根据题意建立方程。"
        elif operation in {"evaluate", "substitute"}:
            narration = "把已经求出的结果代入后面的计算。"
        else:
            narration = "根据题意整理出一个可以计算的数学关系。"
        if expression:
            narration += f" 对应的式子是 {expression}。"
        return narration

    @staticmethod
    def _cas_narration(
        operation: str,
        expression: str | None,
        step: Any | None = None,
        presentation: dict[str, Any] | None = None,
        *,
        output_language: str = "zh-CN",
    ) -> str:
        if Explainer._is_english(output_language):
            if operation in {"solve_for", "solve_equation_system"}:
                return f"Solve this equation to get {expression}."
            if operation in {"evaluate", "substitute"}:
                answer = Explainer._answer_for_step_target(step, presentation)
                if answer is not None and expression:
                    label = answer.get("student_label") or "the result"
                    unit = answer.get("unit")
                    unit_text = f" {unit}" if unit else ""
                    spoken_result = Explainer._spoken_result_from_expression(expression)
                    return f"Compute {label}, which gives {spoken_result}{unit_text}."
                return f"Substitute the known values and compute {expression}."
            if operation in {"construct_expression", "parse_expression"}:
                return f"First simplify the expression: {expression}."
            return f"Complete this calculation to get {expression}."
        if operation in {"solve_for", "solve_equation_system"}:
            return f"解这个方程，得到 {expression}。"
        if operation in {"evaluate", "substitute"}:
            answer = Explainer._answer_for_step_target(step, presentation)
            if answer is not None and expression:
                label = answer.get("student_label") or "结果"
                unit = answer.get("unit")
                unit_text = f" {unit}" if unit else ""
                spoken_result = Explainer._spoken_result_from_expression(expression)
                return f"计算{label}，得到 {spoken_result}{unit_text}。"
            return f"把已经求出的结果代入计算，得到 {expression}。"
        if operation in {"construct_expression", "parse_expression"}:
            return f"先整理这个表达式：{expression}。"
        return f"完成这一步计算，得到 {expression}。"

    @staticmethod
    def _spoken_result_from_expression(expression: str | None) -> str:
        text = str(expression or "").strip()
        if "=" in text:
            right = text.rsplit("=", 1)[-1].strip()
            if right:
                return right
        return text

    @staticmethod
    def _answer_for_step_target(
        step: Any | None,
        presentation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if step is None or presentation is None:
            return None
        diagnostics = getattr(step, "diagnostics", {}) or {}
        target = (
            (diagnostics.get("input") or {}).get("target")
            or (diagnostics.get("normalized_request") or {}).get("target")
        )
        tool_call_id = diagnostics.get("tool_call_id")
        for item in presentation.get("answers") or []:
            if target and str(item.get("target") or "") == str(target):
                return item
            if tool_call_id and str(item.get("source_tool_call_id") or "") == str(tool_call_id):
                return item
        return None

    @staticmethod
    def _duration_for_text(text: str, minimum: int) -> int:
        return max(minimum, min(12000, len(text) * 180))

    @staticmethod
    def _safe_id(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
        return safe or "item"

    @staticmethod
    def _contains_symbol(text: str, symbol: str) -> bool:
        if not symbol:
            return False
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", text) is not None

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return re.search(r"[\u4e00-\u9fff]", text) is not None
