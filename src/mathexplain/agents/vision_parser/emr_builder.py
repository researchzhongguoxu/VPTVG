"""EMR builder for SGVP."""

from __future__ import annotations

import re
from typing import Any

from mathexplain.schemas.emr import EMR, validate_emr_dict
from mathexplain.schemas.spr import Formula, SPR


class EMRBuilder:
    """Build EMR through the stable SGVP EMRBuilder entry point."""

    MOCK_CONDITION_TEXT = "2*x + 3 = 7"
    MOCK_VARIABLE = "x"
    _SUPERSCRIPT_TRANSLATION = str.maketrans(
        {
            "²": "2",
            "³": "3",
            "（": "(",
            "）": ")",
            "，": ",",
            "＋": "+",
            "－": "-",
            "＝": "=",
            "·": "*",
        }
    )

    def build(self, problem_id: str, spr: SPR) -> EMR:
        """Build an EMR from SPR while preserving the deterministic mock path."""

        if spr.metadata.get("parser_mode") == "deterministic_mock":
            return self._build_deterministic_mock(problem_id, spr)
        if self._should_attempt_algebra_build(spr):
            return self._build_algebra(problem_id, spr)
        return self._build_unknown(
            problem_id,
            spr,
            "Only algebra SPR inputs are supported by the first AlgebraEMRBuilder.",
        )

    def _build_deterministic_mock(self, problem_id: str, spr: SPR) -> EMR:
        """Build the fixed algebra EMR used by the first VisionParser skeleton."""

        data = {
            "problem_id": problem_id,
            "source_spr_version": spr.schema_version,
            "source_spr_id": problem_id,
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [
                {
                    "id": "var_x",
                    "symbol": self.MOCK_VARIABLE,
                    "domain": "real",
                    "description": "unknown variable",
                }
            ],
            "expressions": [
                {
                    "id": "expr_1",
                    "sympy": "2*x + 3",
                    "raw_text": self.MOCK_CONDITION_TEXT,
                    "latex": "2x + 3",
                    "source_formula_id": "formula_1",
                }
            ],
            "equations": [
                {
                    "id": "eq_1",
                    "lhs_sympy": "2*x + 3",
                    "rhs_sympy": "7",
                    "relation": "eq",
                    "source_condition_ids": ["condition_1"],
                    "source_formula_ids": ["formula_1"],
                }
            ],
            "goals": [
                {
                    "id": "goal_1",
                    "goal_type": "solve",
                    "target": self.MOCK_VARIABLE,
                    "target_variable_ids": ["var_x"],
                    "description": "solve for x",
                }
            ],
            "metadata": {
                "builder_mode": "deterministic_mock",
                "source_component": "SGVP",
            },
        }
        return validate_emr_dict(data)

    def _build_algebra(self, problem_id: str, spr: SPR) -> EMR:
        variables = [
            {
                "id": self._variable_id(variable.symbol),
                "symbol": variable.symbol,
                "domain": self._normalize_domain(variable.domain, variable.description),
                "description": variable.description,
            }
            for variable in spr.variables
            if variable.symbol
        ]
        variable_symbols = [variable["symbol"] for variable in variables]
        condition_ids_by_formula_id = self._condition_ids_by_formula_id(spr)
        warnings: list[str] = []

        equations = []
        equation_formula_ids: set[str] = set()
        for formula in spr.formulas:
            if formula.role != "condition" and not self._looks_like_explicit_equation(formula.raw_text):
                continue
            if not self._looks_like_explicit_equation(formula.raw_text):
                continue
            equation = self._equation_from_text(
                formula.raw_text,
                variable_symbols,
                source_formula_id=formula.id,
                source_condition_ids=condition_ids_by_formula_id.get(formula.id, []),
                index=len(equations) + 1,
            )
            if equation is None:
                warnings.append(f"Skipped condition formula {formula.id}: not a simple equation.")
                continue
            equations.append(equation)
            equation_formula_ids.add(formula.id)

        if not equations:
            return self._build_unknown(
                problem_id,
                spr,
                "No reliable algebra equation could be extracted from SPR conditions.",
            )

        expressions = []
        goal_candidate_formulas = self._goal_candidate_formulas(spr, equation_formula_ids)
        for formula in goal_candidate_formulas:
            sympy_text = self._to_sympy_like(formula.raw_text, variable_symbols)
            if not sympy_text or "=" in sympy_text:
                warnings.append(f"Skipped target formula {formula.id}: not a simple expression.")
                continue
            expressions.append(
                {
                    "id": f"expr_{len(expressions) + 1}",
                    "sympy": sympy_text,
                    "raw_text": formula.raw_text,
                    "latex": formula.latex,
                    "source_formula_id": formula.id,
                }
            )

        goals = [
            {
                "id": "goal_1",
                "goal_type": self._infer_goal_type(spr),
                "target": self._goal_target_text(spr, expressions),
                "target_variable_ids": [variable["id"] for variable in variables],
                "target_expression_id": self._goal_target_expression_id(expressions, goal_candidate_formulas),
                "description": self._goal_description(spr),
            }
        ]

        metadata: dict[str, Any] = {
            "builder_mode": "algebra_minimal",
            "source_component": "SGVP",
            "builder_version": "algebra-emr-builder-v1",
            "source_spr_problem_type": self._problem_type_value(spr),
        }
        choice_options = self._choice_options(spr)
        if choice_options:
            metadata["choice_options"] = choice_options
        if warnings:
            metadata["warnings"] = warnings

        return validate_emr_dict(
            {
                "problem_id": problem_id,
                "source_spr_version": spr.schema_version,
                "source_spr_id": problem_id,
                "source_problem_type": "algebra",
                "representation_type": "equation_system",
                "variables": variables,
                "expressions": expressions,
                "equations": equations,
                "goals": goals,
                "metadata": metadata,
            }
        )

    def _build_unknown(self, problem_id: str, spr: SPR, reason: str) -> EMR:
        return validate_emr_dict(
            {
                "problem_id": problem_id,
                "source_spr_version": spr.schema_version,
                "source_spr_id": problem_id,
                "source_problem_type": self._problem_type_value(spr),
                "representation_type": "unknown",
                "metadata": {
                    "builder_mode": "fallback_unknown",
                    "source_component": "SGVP",
                    "warnings": [reason],
                },
            }
        )

    @staticmethod
    def _problem_type_value(spr: SPR) -> str:
        value = spr.problem_type
        return getattr(value, "value", str(value))

    def _should_attempt_algebra_build(self, spr: SPR) -> bool:
        problem_type = self._problem_type_value(spr)
        if problem_type == "algebra":
            return True
        if problem_type not in {"word_problem", "unknown", "mixed"}:
            return False
        return any(self._looks_like_explicit_equation(formula.raw_text) for formula in spr.formulas)

    @staticmethod
    def _looks_like_explicit_equation(text: str) -> bool:
        if text.count("=") != 1:
            return False
        lhs, rhs = (part.strip() for part in text.split("=", 1))
        if not lhs or not rhs:
            return False
        return bool(re.search(r"[A-Za-z]", lhs + rhs) and re.search(r"\d", lhs + rhs))

    @staticmethod
    def _variable_id(symbol: str) -> str:
        normalized = re.sub(r"\W+", "_", symbol, flags=re.ASCII).strip("_")
        return f"var_{normalized or 'unknown'}"

    @staticmethod
    def _normalize_domain(domain: str | None, description: str | None) -> str:
        text = f"{domain or ''} {description or ''}".lower()
        if "正实数" in text or "positive real" in text or "(0, \\infty)" in text:
            return "positive_real"
        if "integer" in text or "整数" in text:
            return "integer"
        if "rational" in text or "有理" in text:
            return "rational"
        if "complex" in text or "复数" in text:
            return "complex"
        if "natural" in text or "自然" in text:
            return "natural"
        if "real" in text or "实数" in text:
            return "real"
        return "unknown"

    @staticmethod
    def _condition_ids_by_formula_id(spr: SPR) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for condition in spr.conditions:
            for formula_id in condition.linked_formula_ids:
                mapping.setdefault(formula_id, []).append(condition.id)
        return mapping

    def _equation_from_text(
        self,
        text: str,
        variable_symbols: list[str],
        source_formula_id: str,
        source_condition_ids: list[str],
        index: int,
    ) -> dict[str, Any] | None:
        if text.count("=") != 1:
            return None
        lhs, rhs = (part.strip() for part in text.split("=", 1))
        lhs_sympy = self._to_sympy_like(lhs, variable_symbols)
        rhs_sympy = self._to_sympy_like(rhs, variable_symbols)
        if not lhs_sympy or not rhs_sympy:
            return None
        return {
            "id": f"eq_{index}",
            "lhs_sympy": lhs_sympy,
            "rhs_sympy": rhs_sympy,
            "relation": "eq",
            "source_condition_ids": source_condition_ids,
            "source_formula_ids": [source_formula_id],
        }

    def _goal_candidate_formulas(self, spr: SPR, equation_formula_ids: set[str]) -> list[Formula]:
        goal_formulas = [
            formula
            for formula in spr.formulas
            if formula.role == "goal" and formula.id not in equation_formula_ids
        ]
        if goal_formulas:
            return goal_formulas
        candidates = [
            formula
            for formula in spr.formulas
            if formula.role in {"condition", "unknown", "derived_candidate"}
            and formula.id not in equation_formula_ids
            and "=" not in formula.raw_text
        ]
        if self._has_explicit_text_goal(spr):
            variable_symbols = [variable.symbol for variable in spr.variables]
            return [
                formula
                for formula in candidates
                if self._formula_mentions_variable(formula.raw_text, variable_symbols)
            ]
        return candidates

    @staticmethod
    def _has_explicit_text_goal(spr: SPR) -> bool:
        return any((target.text or "").strip() for target in spr.targets) or any(
            (question.text or "").strip() for question in spr.questions
        )

    @staticmethod
    def _formula_mentions_variable(text: str, variable_symbols: list[str]) -> bool:
        return any(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", text)
            for symbol in variable_symbols
            if symbol
        )

    def _to_sympy_like(self, text: str, variable_symbols: list[str]) -> str:
        normalized = text.translate(self._SUPERSCRIPT_TRANSLATION)
        normalized = normalized.replace("^", "**")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = self._insert_implicit_multiplication(normalized, variable_symbols)
        return normalized

    @staticmethod
    def _insert_implicit_multiplication(text: str, variable_symbols: list[str]) -> str:
        lowercase_variables = sorted(
            {symbol for symbol in variable_symbols if re.fullmatch(r"[a-z]", symbol)},
            key=len,
            reverse=True,
        )
        if not lowercase_variables:
            return text

        variable_class = "".join(re.escape(symbol) for symbol in lowercase_variables)
        text = re.sub(rf"(\d)([{variable_class}])", r"\1*\2", text)
        text = re.sub(rf"([{variable_class}])([{variable_class}])", r"\1*\2", text)
        return text

    @staticmethod
    def _infer_goal_type(spr: SPR) -> str:
        text = EMRBuilder._combined_goal_text(spr).lower()
        if any(keyword in text for keyword in ("最小值", "最大值", "minimum", "maximum")):
            return "compute"
        for question in spr.questions:
            if question.question_type != "unknown":
                return question.question_type
        for target in spr.targets:
            if target.target_type != "unknown":
                return target.target_type
        return "unknown"

    @staticmethod
    def _goal_target_text(spr: SPR, expressions: list[dict[str, Any]]) -> str | None:
        for target in spr.targets:
            if target.text:
                return target.text
        if expressions:
            return expressions[0]["sympy"]
        return None

    @staticmethod
    def _goal_target_expression_id(
        expressions: list[dict[str, Any]],
        source_formulas: list[Formula],
    ) -> str | None:
        if not expressions or not source_formulas:
            return None
        if source_formulas[0].role == "goal":
            return expressions[0]["id"]
        return expressions[0]["id"]

    @staticmethod
    def _goal_description(spr: SPR) -> str | None:
        for question in spr.questions:
            if question.text:
                return question.text
        for target in spr.targets:
            if target.text:
                return target.text
        return None

    @staticmethod
    def _combined_goal_text(spr: SPR) -> str:
        parts = [spr.problem_text or "", spr.problem_stem or ""]
        parts.extend(question.text for question in spr.questions)
        parts.extend(target.text for target in spr.targets)
        return "\n".join(parts)

    @staticmethod
    def _choice_options(spr: SPR) -> list[dict[str, str]]:
        options = []
        for formula in spr.formulas:
            if formula.role != "option":
                continue
            label, value = EMRBuilder._split_option_text(formula.raw_text)
            option = {
                "label": label,
                "value": value,
                "source_formula_id": formula.id,
            }
            if formula.latex:
                option["latex"] = formula.latex
            options.append(option)
        return options

    @staticmethod
    def _split_option_text(raw_text: str) -> tuple[str, str]:
        match = re.match(r"^\s*([A-Da-d])\s*[.．、:：]\s*(.+?)\s*$", raw_text)
        if match:
            return match.group(1).upper(), match.group(2)
        return "", raw_text.strip()
