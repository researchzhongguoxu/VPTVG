"""Internal calculation request protocol for Solver tool calls."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mathexplain.schemas.emr import EMR
from mathexplain.schemas.spr import SPR
from mathexplain.services.cas import CASService


SupportedOperation = Literal[
    "construct_expression",
    "parse_expression",
    "solve_equation_system",
    "solve_for",
    "substitute",
    "simplify",
    "evaluate",
    "compare_options",
]


IMPLEMENTED_OPERATIONS = {
    "construct_expression",
    "parse_expression",
    "solve_equation_system",
    "solve_for",
    "substitute",
    "evaluate",
}
SUPPORTED_OPERATIONS = {
    "construct_expression",
    "parse_expression",
    "solve_equation_system",
    "solve_for",
    "substitute",
    "simplify",
    "evaluate",
    "compare_options",
}


class CalculationRequestModel(BaseModel):
    """Base model for internal solver tool-call requests."""

    model_config = ConfigDict(extra="forbid")


class ToolExpression(CalculationRequestModel):
    """Expression proposed by a planner tool call."""

    expression: str
    name: str | None = None

    @field_validator("expression")
    @classmethod
    def expression_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty expression")
        return value


class ToolEquation(CalculationRequestModel):
    """Equation proposed by a planner tool call."""

    lhs: str
    rhs: str
    id: str | None = None

    @field_validator("lhs", "rhs")
    @classmethod
    def side_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty expression")
        return value


class CalculationRequest(CalculationRequestModel):
    """Internal, non-public tool call request emitted by the planner."""

    id: str | None = None
    operation: SupportedOperation
    variables: list[str] = Field(default_factory=list)
    equations: list[ToolEquation] = Field(default_factory=list)
    expressions: list[ToolExpression] = Field(default_factory=list)
    substitutions: dict[str, Any] = Field(default_factory=dict)
    target: str | None = None
    reason: str | None = None
    evidence_span: str | None = None


class CalculationRequestValidator:
    """Deterministically validate planner tool-call requests before CAS execution."""

    def __init__(self, cas_service: CASService | None = None) -> None:
        self.cas_service = cas_service or CASService()

    def validate(
        self,
        request_data: dict[str, Any],
        emr: EMR,
        spr: SPR | None = None,
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate one request and return a JSON-serializable report."""

        original_request_data = dict(request_data)
        context = execution_context or {}
        request_data = self._normalize_request_data(request_data, context)
        try:
            request = CalculationRequest.model_validate(request_data)
        except ValidationError as exc:
            return {
                "valid": False,
                "request": None,
                "normalized_request": request_data,
                "errors": [
                    {
                        "field_path": ".".join(str(item) for item in error["loc"]),
                        "value": error.get("input"),
                        "message": error["msg"],
                    }
                    for error in exc.errors()
                ],
            }

        errors: list[dict[str, Any]] = []
        normalization_warnings: list[dict[str, Any]] = []
        target_list_warning = self._target_list_normalization_warning(original_request_data, request_data)
        if target_list_warning:
            normalization_warnings.append(target_list_warning)
        schema_declared_variables = self._declared_variables(emr, spr)
        base_declared_variables = schema_declared_variables | self._context_symbols(context)
        if not schema_declared_variables and self._allows_planner_local_symbols(emr, spr):
            local_symbols = self._request_declared_symbols(request)
            if local_symbols:
                base_declared_variables.update(local_symbols)
                normalization_warnings.append(
                    {
                        "field_path": "variables",
                        "value": sorted(local_symbols),
                        "message": (
                            "SPR/EMR did not declare variables; accepting planner-local "
                            "symbols for this word-problem ToolCall."
                        ),
                    }
                )
        if schema_declared_variables and self._allows_planner_local_symbols(emr, spr):
            local_symbols = self._planner_temporary_symbols(request, base_declared_variables)
            if local_symbols:
                base_declared_variables.update(local_symbols)
                normalization_warnings.append(
                    {
                        "field_path": "variables",
                        "value": sorted(local_symbols),
                        "message": (
                            "Accepting planner-local temporary symbols for this "
                            "word-problem ToolCall."
                        ),
                    }
                )
            semantic_symbol_aliases = self._semantic_symbol_aliases(request, base_declared_variables, spr)
            if semantic_symbol_aliases:
                base_declared_variables.update(semantic_symbol_aliases.keys())
                normalization_warnings.append(
                    {
                        "field_path": "variables",
                        "value": request.variables,
                        "semantic_symbol_aliases": semantic_symbol_aliases,
                        "message": (
                            "Planner used a conventional temporary symbol that was "
                            "semantically aligned with an SPR variable."
                        ),
                    }
                )
            derived_local_symbols = self._planner_local_derived_symbols(request, base_declared_variables)
            if derived_local_symbols:
                base_declared_variables.update(derived_local_symbols)
                normalization_warnings.append(
                    {
                        "field_path": "variables",
                        "value": sorted(derived_local_symbols),
                        "message": (
                            "Accepting planner-local derived symbols for this "
                            "word-problem ToolCall."
                        ),
                    }
                )
        else:
            semantic_symbol_aliases = self._semantic_symbol_aliases(request, base_declared_variables, spr)
        derived_result_symbols = self._derived_result_symbols(request)
        new_derived_symbols = derived_result_symbols - base_declared_variables
        if new_derived_symbols:
            base_declared_variables.update(new_derived_symbols)
            normalization_warnings.append(
                {
                    "field_path": "target",
                    "value": sorted(new_derived_symbols),
                    "message": (
                        "Accepting ToolCall target or expression name as a derived "
                        "result symbol for this request."
                    ),
                }
            )
        substitution_local_symbols = self._resolvable_substitution_symbols(request, base_declared_variables, context)
        if substitution_local_symbols:
            base_declared_variables.update(substitution_local_symbols)
            normalization_warnings.append(
                {
                    "field_path": "substitutions",
                    "value": sorted(substitution_local_symbols),
                    "message": (
                        "Accepting planner-local symbols that are fully defined by "
                        "resolvable prior-result substitutions."
                    ),
                }
            )
        constant_substitution_symbols = self._concrete_substitution_symbols(request, base_declared_variables)
        if constant_substitution_symbols:
            base_declared_variables.update(constant_substitution_symbols)
            normalization_warnings.append(
                {
                    "field_path": "substitutions",
                    "value": sorted(constant_substitution_symbols),
                    "message": (
                        "Accepting planner-local symbols that are fully defined "
                        "by concrete constant substitutions."
                    ),
                }
            )
        declared_variables = base_declared_variables | set(semantic_symbol_aliases.keys())

        if request.operation not in SUPPORTED_OPERATIONS:
            errors.append(
                {
                    "field_path": "operation",
                    "value": request.operation,
                    "message": "Unsupported operation.",
                }
            )
        elif request.operation not in IMPLEMENTED_OPERATIONS:
            errors.append(
                {
                    "field_path": "operation",
                    "value": request.operation,
                    "message": "Operation is recognized but not implemented in Solver v1 enhanced.",
                }
            )

        undeclared_request_variables = sorted(set(request.variables) - declared_variables)
        if undeclared_request_variables:
            errors.append(
                {
                    "field_path": "variables",
                    "value": request.variables,
                    "undeclared_symbols": undeclared_request_variables,
                    "message": "Request variables must be declared in EMR variables.",
                }
            )

        variable_symbols = sorted(declared_variables)
        used_symbols: set[str] = set()
        self._validate_required_fields(request, errors)
        used_symbols.update(
            self._validate_expressions(
                request,
                variable_symbols,
                declared_variables,
                context,
                errors,
            )
        )
        used_symbols.update(
            self._validate_equations(
                request,
                variable_symbols,
                declared_variables,
                context,
                errors,
            )
        )
        used_symbols.update(
            self._validate_substitutions(request, declared_variables, context, errors)
        )
        self._validate_numeric_word_problem_request_is_grounded(
            request,
            spr,
            context,
            declared_variables,
            errors,
        )

        if request.target and request.target in declared_variables:
            used_symbols.add(request.target)

        planner_variables = list(request.variables)
        effective_variables = sorted(set(planner_variables) | used_symbols)
        missing_from_planner = sorted(used_symbols - set(planner_variables))
        if missing_from_planner:
            normalization_warnings.append(
                {
                    "field_path": "variables",
                    "value": planner_variables,
                    "added_symbols": missing_from_planner,
                    "message": (
                        "Request variables did not include every symbol required "
                        "by expressions, equations, substitutions, or target."
                    ),
                }
            )

        normalized_request = request.model_dump(mode="json")
        normalized_request["planner_variables"] = planner_variables
        normalized_request["effective_variables"] = effective_variables
        normalized_request["normalization_warnings"] = normalization_warnings
        normalized_request["semantic_symbol_aliases"] = semantic_symbol_aliases

        return {
            "valid": not errors,
            "request": request,
            "normalized_request": normalized_request,
            "planner_variables": planner_variables,
            "effective_variables": effective_variables,
            "normalization_warnings": normalization_warnings,
            "errors": errors,
            "declared_variables": sorted(base_declared_variables),
            "semantic_symbol_aliases": semantic_symbol_aliases,
        }

    def _validate_numeric_word_problem_request_is_grounded(
        self,
        request: CalculationRequest,
        spr: SPR | None,
        execution_context: dict[str, Any],
        declared_variables: set[str],
        errors: list[dict[str, Any]],
    ) -> None:
        if spr is None or request.operation not in {"evaluate", "substitute"}:
            return
        if not self._spr_expects_numeric_final_quantity(spr):
            return
        if self._request_allows_symbolic_result(request):
            return
        expression_symbols: set[str] = set()
        for index, expression in enumerate(request.expressions):
            expression_value = self._resolve_expression_reference(
                expression.expression,
                execution_context=execution_context,
            )
            parse_report = self.cas_service.parse_expression(expression_value, sorted(declared_variables))
            if parse_report.get("success"):
                expression_symbols.update(parse_report.get("free_symbols") or [])
        if not expression_symbols:
            return
        grounded = self._grounded_symbols(request, execution_context)
        unresolved = sorted(symbol for symbol in expression_symbols if symbol not in grounded)
        if unresolved:
            errors.append(
                {
                    "field_path": "expressions",
                    "value": [expression.model_dump(mode="json") for expression in request.expressions],
                    "undeclared_symbols": unresolved,
                    "message": (
                        "Numeric word-problem evaluate/substitute requests must ground every "
                        "expression symbol through a concrete substitution or prior CAS result."
                    ),
                }
            )

    @staticmethod
    def _spr_expects_numeric_final_quantity(spr: SPR) -> bool:
        text = " ".join(
            [
                spr.problem_text or "",
                spr.problem_stem or "",
                *(question.text or "" for question in spr.questions),
                *(target.text or "" for target in spr.targets),
            ]
        ).lower()
        if any(signal in text for signal in ["function", "equation of", "expression", "formula", "解析式"]):
            return False
        return any(
            signal in text
            for signal in [
                "how many",
                "how much",
                "total",
                "combined",
                "current",
                "left",
                "remaining",
                "profit",
                "cost",
                "receive",
                "pay",
            ]
        )

    @staticmethod
    def _request_allows_symbolic_result(request: CalculationRequest) -> bool:
        if request.target is None:
            return True
        target_text = " ".join(
            [
                request.target or "",
                *(expression.name or "" for expression in request.expressions),
                request.reason or "",
            ]
        ).lower()
        return any(
            signal in target_text
            for signal in ["expression", "expr", "formula", "function", "equation", "line", "axis"]
        )

    @staticmethod
    def _grounded_symbols(
        request: CalculationRequest,
        execution_context: dict[str, Any],
    ) -> set[str]:
        grounded: set[str] = set()
        context_variables = execution_context.get("variables") or {}
        for symbol in context_variables:
            grounded.add(symbol)
        for symbol, value in request.substitutions.items():
            if isinstance(value, str) and value.startswith("$"):
                if CalculationRequestValidator.resolve_reference(value, execution_context) is not None:
                    grounded.add(symbol)
            elif str(value).strip():
                grounded.add(symbol)
        return grounded

    @staticmethod
    def _validate_required_fields(
        request: CalculationRequest,
        errors: list[dict[str, Any]],
    ) -> None:
        if request.operation in {"construct_expression", "parse_expression"} and not request.expressions:
            errors.append(
                {
                    "field_path": "expressions",
                    "value": [],
                    "message": "This operation requires at least one expression.",
                }
            )
        if request.operation in {"substitute", "evaluate"} and not request.expressions:
            errors.append(
                {
                    "field_path": "expressions",
                    "value": [],
                    "message": "This operation requires at least one expression.",
                }
            )
        if request.operation == "substitute" and not request.substitutions:
            errors.append(
                {
                    "field_path": "substitutions",
                    "value": {},
                    "message": "substitute requires substitutions or resolved prior results.",
                }
            )
        if request.operation in {"solve_equation_system", "solve_for"} and not request.equations:
            errors.append(
                {
                    "field_path": "equations",
                    "value": [],
                    "message": "This operation requires at least one equation.",
                }
            )
        if request.operation == "solve_for" and not request.target:
            errors.append(
                {
                    "field_path": "target",
                    "value": None,
                    "message": "solve_for requires a target variable.",
                }
            )

    def _validate_expressions(
        self,
        request: CalculationRequest,
        variable_symbols: list[str],
        declared_variables: set[str],
        execution_context: dict[str, Any],
        errors: list[dict[str, Any]],
    ) -> set[str]:
        used_symbols: set[str] = set()
        for index, expression in enumerate(request.expressions):
            field_path = f"expressions[{index}].expression"
            expression_value = self._resolve_expression_reference(
                expression.expression,
                execution_context=execution_context,
            )
            used_symbols.update(
                self._parse_and_check_symbols(
                    expression_value,
                    field_path,
                    variable_symbols,
                    declared_variables,
                    errors,
                )
            )
        return used_symbols

    def _validate_equations(
        self,
        request: CalculationRequest,
        variable_symbols: list[str],
        declared_variables: set[str],
        execution_context: dict[str, Any],
        errors: list[dict[str, Any]],
    ) -> set[str]:
        used_symbols: set[str] = set()
        for index, equation in enumerate(request.equations):
            lhs = self._resolve_expression_reference(
                equation.lhs,
                execution_context=execution_context,
            )
            rhs = self._resolve_expression_reference(
                equation.rhs,
                execution_context=execution_context,
            )
            used_symbols.update(
                self._parse_and_check_symbols(
                    lhs,
                    f"equations[{index}].lhs",
                    variable_symbols,
                    declared_variables,
                    errors,
                )
            )
            used_symbols.update(
                self._parse_and_check_symbols(
                    rhs,
                    f"equations[{index}].rhs",
                    variable_symbols,
                    declared_variables,
                    errors,
                )
            )
        return used_symbols

    def _parse_and_check_symbols(
        self,
        value: str,
        field_path: str,
        variable_symbols: list[str],
        declared_variables: set[str],
        errors: list[dict[str, Any]],
    ) -> set[str]:
        parse_report = self.cas_service.parse_expression(value, variable_symbols)
        if not parse_report["success"]:
            errors.append(
                {
                    "field_path": field_path,
                    "value": value,
                    "message": "Expression failed SymPy parse-only validation.",
                    "parse_errors": parse_report["errors"],
                }
            )
            return set()

        undeclared_symbols = sorted(set(parse_report["free_symbols"]) - declared_variables)
        if undeclared_symbols:
            errors.append(
                {
                    "field_path": field_path,
                    "value": value,
                    "undeclared_symbols": undeclared_symbols,
                    "message": "Expression uses symbols not declared in EMR variables.",
                }
            )
        return set(parse_report["free_symbols"])

    @staticmethod
    def _declared_variables(emr: EMR, spr: SPR | None) -> set[str]:
        declared = {variable.symbol for variable in emr.variables if variable.symbol}
        if declared or spr is None:
            return declared
        return {variable.symbol for variable in spr.variables if variable.symbol}

    @staticmethod
    def _allows_planner_local_symbols(emr: EMR, spr: SPR | None) -> bool:
        emr_type = getattr(emr.source_problem_type, "value", str(emr.source_problem_type or "unknown"))
        spr_type = getattr(spr.problem_type, "value", str(spr.problem_type or "unknown")) if spr is not None else "unknown"
        return emr_type in {"word_problem", "algebra", "calculus"} or spr_type in {
            "word_problem",
            "algebra",
            "calculus",
        } or CalculationRequestValidator._is_coordinate_algebra_geometry_task(emr, spr)

    @staticmethod
    def _is_coordinate_algebra_geometry_task(emr: EMR, spr: SPR | None) -> bool:
        if spr is None:
            return False
        emr_type = getattr(emr.source_problem_type, "value", str(emr.source_problem_type or "unknown"))
        spr_type = getattr(spr.problem_type, "value", str(spr.problem_type or "unknown"))
        if emr_type != "geometry" and spr_type != "geometry":
            return False
        text_parts = [
            spr.problem_text or "",
            spr.problem_stem or "",
            *[question.text for question in spr.questions],
            *[target.text for target in spr.targets],
            *spr.knowledge_units,
        ]
        text = " ".join(str(part or "") for part in text_parts).lower()
        if any(keyword in text for keyword in ("diagram", "figure", "shown below", "prove", "proof")):
            return False
        return "midpoint" in text and ("coordinate" in text or "(" in text)

    @staticmethod
    def _request_declared_symbols(request: CalculationRequest) -> set[str]:
        symbols = {
            symbol
            for symbol in request.variables
            if CalculationRequestValidator._is_plain_symbol(symbol)
        }
        if request.target and CalculationRequestValidator._is_plain_symbol(request.target):
            symbols.add(request.target)
        symbols.update(
            expression.name
            for expression in request.expressions
            if expression.name and CalculationRequestValidator._is_plain_symbol(expression.name)
        )
        symbols.update(
            symbol
            for symbol in request.substitutions
            if CalculationRequestValidator._is_plain_symbol(symbol)
        )
        return symbols

    @staticmethod
    def _derived_result_symbols(request: CalculationRequest) -> set[str]:
        if request.operation not in {"construct_expression", "parse_expression", "substitute", "evaluate"}:
            return set()
        symbols: set[str] = set()
        if request.target and CalculationRequestValidator._is_plain_symbol(request.target):
            symbols.add(request.target)
        symbols.update(
            expression.name
            for expression in request.expressions
            if expression.name and CalculationRequestValidator._is_plain_symbol(expression.name)
        )
        return symbols

    @staticmethod
    def _resolvable_substitution_symbols(
        request: CalculationRequest,
        declared_variables: set[str],
        execution_context: dict[str, Any],
    ) -> set[str]:
        symbols: set[str] = set()
        for symbol, value in request.substitutions.items():
            if symbol in declared_variables or not CalculationRequestValidator._is_plain_symbol(symbol):
                continue
            if isinstance(value, str) and value.startswith("$"):
                resolved = CalculationRequestValidator.resolve_reference(value, execution_context)
                if resolved is not None:
                    symbols.add(symbol)
        return symbols

    @staticmethod
    def _planner_temporary_symbols(
        request: CalculationRequest,
        declared_variables: set[str],
    ) -> set[str]:
        symbols = CalculationRequestValidator._request_declared_symbols(request)
        undeclared = symbols - declared_variables
        allowed = {"k", "m"}
        if request.operation in {"solve_equation_system", "solve_for"} and request.equations:
            allowed.update({"a", "b"})
        if CalculationRequestValidator._is_conventional_local_equation_system(request):
            allowed.update({"x", "y", "z"})
        return {
            symbol for symbol in undeclared
            if symbol.lower() in allowed
        }

    @staticmethod
    def _planner_local_derived_symbols(
        request: CalculationRequest,
        declared_variables: set[str],
    ) -> set[str]:
        if request.operation in {"solve_equation_system", "solve_for"}:
            return {
                symbol
                for symbol in CalculationRequestValidator._request_declared_symbols(request) - declared_variables
                if CalculationRequestValidator._is_plain_symbol(symbol)
            }
        if request.operation not in {"construct_expression", "parse_expression", "substitute", "evaluate"}:
            return set()
        symbols = CalculationRequestValidator._request_declared_symbols(request)
        return {
            symbol
            for symbol in symbols - declared_variables
            if CalculationRequestValidator._is_plain_symbol(symbol)
        }

    @staticmethod
    def _is_conventional_local_equation_system(request: CalculationRequest) -> bool:
        if request.operation not in {"solve_equation_system", "solve_for"}:
            return False
        if not request.equations:
            return False
        variable_set = {symbol.lower() for symbol in request.variables if symbol}
        if len(variable_set) < 2:
            return False
        return variable_set.issubset({"x", "y", "z"})

    @staticmethod
    def _semantic_symbol_aliases(
        request: CalculationRequest,
        declared_variables: set[str],
        spr: SPR | None,
    ) -> dict[str, str]:
        if spr is None or not declared_variables:
            return {}
        request_symbols = set(request.variables)
        request_symbols.update(request.substitutions.keys())
        if request.target:
            request_symbols.add(request.target)
        undeclared = {
            symbol
            for symbol in request_symbols
            if symbol not in declared_variables and CalculationRequestValidator._is_plain_symbol(symbol)
        }
        aliases: dict[str, str] = {}
        primary = CalculationRequestValidator._primary_spr_symbol(spr)
        remaining_symbol = CalculationRequestValidator._remaining_spr_symbol(spr)
        box_symbol = CalculationRequestValidator._box_spr_symbol(spr)
        latex_symbol_aliases = CalculationRequestValidator._latex_plain_symbol_aliases(declared_variables)
        role_symbol_aliases = CalculationRequestValidator._role_based_symbol_aliases(spr)
        for symbol in sorted(undeclared):
            lowered = symbol.lower()
            if symbol in latex_symbol_aliases:
                aliases[symbol] = latex_symbol_aliases[symbol]
            elif symbol in role_symbol_aliases:
                aliases[symbol] = role_symbol_aliases[symbol]
            elif lowered in {"x", "t", "total", "w"} and primary:
                aliases[symbol] = primary
            elif (
                lowered in {"r", "rem", "remain", "remaining", "left", "leftover", "total_remaining"}
                or lowered.endswith("_remaining")
                or lowered.endswith("_remain")
            ) and remaining_symbol:
                aliases[symbol] = remaining_symbol
            elif lowered in {"n", "n_boxes", "num_boxes", "boxes", "box", "nboxes"} and box_symbol:
                aliases[symbol] = box_symbol
        return aliases

    @staticmethod
    def _role_based_symbol_aliases(spr: SPR) -> dict[str, str]:
        aliases: dict[str, str] = {}
        passenger_symbol: str | None = None
        truck_symbol: str | None = None
        semantic_tokens = {
            "yellow": ("yellow",),
            "purple": ("purple",),
            "green": ("green",),
            "red": ("red",),
            "blue": ("blue",),
            "total": ("total", "altogether", "combined"),
        }
        for variable in spr.variables:
            symbol = variable.symbol
            if not symbol:
                continue
            text = " ".join([symbol, variable.description or ""]).lower()
            for alias, tokens in semantic_tokens.items():
                if any(token in text for token in tokens):
                    aliases.setdefault(alias, symbol)
                    aliases.setdefault(f"{alias}_count", symbol)
                    aliases.setdefault(f"{alias}_value", symbol)
            if any(token in text for token in ["passenger", "bus", "coach"]) or "客车" in text:
                passenger_symbol = symbol
            if any(token in text for token in ["truck", "cargo", "freight"]) or "货车" in text:
                truck_symbol = symbol
        if passenger_symbol:
            for alias in ("v_p", "v_b", "v_bus", "v_passenger", "bus_speed", "passenger_speed"):
                aliases[alias] = passenger_symbol
        if truck_symbol:
            for alias in ("v_t", "v_f", "v_truck", "v_cargo", "truck_speed", "cargo_speed"):
                aliases[alias] = truck_symbol
        return aliases

    @staticmethod
    def _latex_plain_symbol_aliases(declared_variables: set[str]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for declared in declared_variables:
            match = re.fullmatch(r"([A-Za-z]+)_\{([A-Za-z0-9]+)\}", declared or "")
            if not match:
                continue
            plain = f"{match.group(1)}_{match.group(2)}"
            if CalculationRequestValidator._is_plain_symbol(plain):
                aliases[plain] = declared
        return aliases

    @staticmethod
    def _is_plain_symbol(symbol: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol or ""))

    @staticmethod
    def _primary_spr_symbol(spr: SPR) -> str | None:
        candidates: list[tuple[int, str]] = []
        for variable in spr.variables:
            symbol = variable.symbol
            if not symbol:
                continue
            description = (variable.description or "").lower()
            if "box" in description or "箱" in description:
                continue
            score = 0
            if "total" in description or "总" in description or "原来" in description:
                score += 2
            if "weight" in description or "重量" in description or "千克" in description:
                score += 1
            candidates.append((score, symbol))
        if candidates:
            return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
        return spr.variables[0].symbol if spr.variables else None

    @staticmethod
    def _box_spr_symbol(spr: SPR) -> str | None:
        for variable in spr.variables:
            description = (variable.description or "").lower()
            symbol_text = (variable.symbol or "").lower()
            if "box" in description or "箱" in description or "box" in symbol_text:
                return variable.symbol
        return None

    @staticmethod
    def _remaining_spr_symbol(spr: SPR) -> str | None:
        for variable in spr.variables:
            description = (variable.description or "").lower()
            symbol_text = (variable.symbol or "").lower()
            if any(token in symbol_text for token in ["remaining", "remain", "rem"]):
                return variable.symbol
            if any(token in description for token in ["remaining", "leftover", "left", "剩", "余"]):
                return variable.symbol
        return None

    @staticmethod
    def _context_symbols(execution_context: dict[str, Any]) -> set[str]:
        reserved = {
            "variables",
            "operation",
            "output",
            "formatted_solution",
            "confirmed_expression",
            "value",
        }
        symbols = set(execution_context.get("variables", {}).keys())
        for value in execution_context.values():
            if not isinstance(value, dict):
                continue
            symbols.update(
                key for key in value.keys()
                if key not in reserved and not key.startswith("_")
            )
        return symbols

    @staticmethod
    def _validate_substitutions(
        request: CalculationRequest,
        declared_variables: set[str],
        execution_context: dict[str, Any],
        errors: list[dict[str, Any]],
    ) -> set[str]:
        used_symbols: set[str] = set()
        for symbol, value in request.substitutions.items():
            used_symbols.add(symbol)
            if symbol not in declared_variables:
                errors.append(
                    {
                        "field_path": f"substitutions.{symbol}",
                        "value": value,
                        "undeclared_symbols": [symbol],
                        "message": "Substitution symbol is not declared in EMR or SPR variables.",
                    }
                )
            if isinstance(value, str) and value.startswith("$"):
                resolved = CalculationRequestValidator.resolve_reference(value, execution_context)
                if resolved is None:
                    errors.append(
                        {
                            "field_path": f"substitutions.{symbol}",
                            "value": value,
                            "message": "Substitution reference could not be resolved.",
                        }
                    )
        return used_symbols

    @staticmethod
    def resolve_reference(reference: str, execution_context: dict[str, Any]) -> Any:
        """Resolve references like $solve_x.x from prior tool results."""

        path = reference[1:].split(".")
        current: Any = execution_context
        for part in path:
            if isinstance(current, dict):
                if part == "result":
                    return CalculationRequestValidator._scalar_reference_value(current)
                if part in current:
                    current = current[part]
                    continue
                aliased_key = CalculationRequestValidator._matching_reference_key(part, current)
                if aliased_key is not None:
                    current = current[aliased_key]
                    continue
            return None
        return CalculationRequestValidator._scalar_reference_value(current)

    @staticmethod
    def _scalar_reference_value(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key in ("value", "formatted_solution", "confirmed_expression"):
            if value.get(key) is not None:
                return value[key]
        output = value.get("output")
        if isinstance(output, dict):
            for key in ("value", "formatted_solution", "confirmed_expression"):
                if output.get(key) is not None:
                    return output[key]
        return value

    @staticmethod
    def _matching_reference_key(part: str, current: dict[str, Any]) -> str | None:
        target_key = CalculationRequestValidator._reference_key(part)
        for key in current:
            if CalculationRequestValidator._reference_key(str(key)) == target_key:
                return str(key)
        return None

    @staticmethod
    def _reference_key(symbol: str) -> str:
        return re.sub(r"_\{([^{}]+)\}", r"_\1", symbol.strip())

    @staticmethod
    def _normalize_request_data(
        request_data: dict[str, Any],
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Coerce common LLM nulls into the schema's empty containers."""

        normalized = dict(request_data)
        for field in ("variables", "equations", "expressions"):
            if normalized.get(field) is None:
                normalized[field] = []
        if normalized.get("substitutions") is None:
            normalized["substitutions"] = {}
        normalized["substitutions"] = CalculationRequestValidator._normalize_substitutions(
            normalized.get("substitutions") or {}
        )
        if normalized.get("substitutions"):
            normalized["expressions"] = [
                CalculationRequestValidator._normalize_short_references_in_expression(
                    expression,
                    normalized["substitutions"],
                )
                for expression in normalized.get("expressions") or []
            ]
        if (
            normalized.get("operation") == "evaluate"
            and not normalized.get("substitutions")
            and CalculationRequestValidator._request_uses_symbolic_diff(normalized)
        ):
            normalized["operation"] = "construct_expression"
        target = normalized.get("target")
        if isinstance(target, list):
            target_items = [str(item).strip() for item in target if str(item).strip()]
            if len(target_items) == 1:
                normalized["target"] = target_items[0]
            elif target_items:
                normalized["target"] = "_and_".join(target_items)
        elif isinstance(target, str) and not target.strip():
            normalized["target"] = None
        if (
            normalized.get("operation") in {"substitute", "evaluate"}
            and not normalized.get("expressions")
            and isinstance(normalized.get("target"), str)
            and CalculationRequestValidator._looks_like_expression_target(str(normalized["target"]))
        ):
            expression = str(normalized["target"]).strip()
            result_name = CalculationRequestValidator._derived_name_from_request(normalized)
            normalized["expressions"] = [
                {"name": result_name, "expression": expression}
            ]
            normalized["target"] = result_name
        if normalized.get("operation") == "solve_equation_system":
            normalized["target"] = CalculationRequestValidator._normalized_system_target(normalized)
            if CalculationRequestValidator._should_treat_system_as_solve_for(normalized):
                normalized["operation"] = "solve_for"
        if (
            normalized.get("operation") in {"substitute", "evaluate"}
            and not normalized.get("expressions")
            and normalized.get("target")
        ):
            expression = CalculationRequestValidator._context_expression_for_target(
                str(normalized["target"]),
                execution_context or {},
            )
            if expression:
                normalized["expressions"] = [
                    {"name": str(normalized["target"]), "expression": expression}
                ]
        return normalized

    @staticmethod
    def _looks_like_expression_target(target: str) -> bool:
        stripped = target.strip()
        if not stripped:
            return False
        if CalculationRequestValidator._is_plain_symbol(stripped):
            return False
        return bool(
            re.search(r"[\d$+\-*/().]", stripped)
            or any(token in stripped for token in ["sqrt", "frac", "^"])
        )

    @staticmethod
    def _derived_name_from_request(request_data: dict[str, Any]) -> str:
        for key in ("id", "name"):
            value = str(request_data.get(key) or "").strip()
            if CalculationRequestValidator._is_plain_symbol(value):
                return value
        return "result"

    @staticmethod
    def _normalized_system_target(request_data: dict[str, Any]) -> str | None:
        target = request_data.get("target")
        variables = [str(variable) for variable in request_data.get("variables") or [] if str(variable).strip()]
        if not variables:
            return target
        if not target and len(variables) > 1:
            return "_and_".join(variables)
        target_text = str(target or "").strip()
        if not target_text:
            return None
        if {"x", "y"} <= set(variables) and target_text in {"x", "y"}:
            context_text = " ".join(
                str(request_data.get(field) or "")
                for field in ("reason", "evidence_span")
            ).lower()
            coordinate_signals = [
                "coordinate",
                "coordinates",
                "intercept",
                "intercepts",
                "x-axis",
                "y-axis",
                "坐标",
                "交点",
                "x 轴",
                "y 轴",
                "x轴",
                "y轴",
            ]
            if any(signal in context_text for signal in coordinate_signals):
                return "x_and_y"
        return target_text

    @staticmethod
    def _should_treat_system_as_solve_for(request_data: dict[str, Any]) -> bool:
        target = str(request_data.get("target") or "").strip()
        if not target or not CalculationRequestValidator._is_plain_symbol(target):
            return False
        variables = {str(variable) for variable in request_data.get("variables") or []}
        if target not in variables:
            return False
        equations = request_data.get("equations") or []
        if len(equations) != 1:
            return False
        if "_and_" in target or target in {"all", "solution", "solutions", "solution_set"}:
            return False
        context_text = " ".join(
            str(request_data.get(field) or "")
            for field in ("reason", "evidence_span")
        ).lower()
        calculus_signals = [
            "derivative",
            "monotonic",
            "critical",
            "extrema",
            "extremum",
            "导数",
            "单调",
            "临界",
            "极值",
        ]
        return any(signal in context_text for signal in calculus_signals)

    @staticmethod
    def _request_uses_symbolic_diff(request_data: dict[str, Any]) -> bool:
        expressions = request_data.get("expressions") or []
        if not expressions:
            return False
        expression = str(expressions[0].get("expression") or "").strip()
        return expression.startswith("diff(") or expression.lower().startswith("derivative(")

    @staticmethod
    def _context_expression_for_target(
        target: str,
        execution_context: dict[str, Any],
    ) -> str | None:
        for value in execution_context.values():
            if not isinstance(value, dict):
                continue
            if target in value and isinstance(value[target], str):
                return value[target]
            if value.get("operation") in {"construct_expression", "parse_expression"}:
                output = value.get("output") or {}
                expression = output.get("confirmed_expression") or output.get("parsed_expression")
                if isinstance(expression, str):
                    return expression
        return None

    @staticmethod
    def _target_list_normalization_warning(
        original_request_data: dict[str, Any],
        normalized_request_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        target = original_request_data.get("target")
        if not isinstance(target, list):
            return None
        return {
            "field_path": "target",
            "value": target,
            "normalized_value": normalized_request_data.get("target"),
            "message": "Normalized a list target into an aggregate string target.",
        }

    @staticmethod
    def _normalize_substitutions(substitutions: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for symbol, value in substitutions.items():
            symbol_text = str(symbol)
            if symbol_text.startswith("$"):
                continue
            normalized[symbol_text] = value
        return normalized

    @staticmethod
    def _normalize_short_references_in_expression(
        expression_data: Any,
        substitutions: dict[str, Any],
    ) -> Any:
        if not isinstance(expression_data, dict):
            return expression_data
        expression = expression_data.get("expression")
        if not isinstance(expression, str) or "$" not in expression:
            return expression_data
        normalized = dict(expression_data)

        def replace(match: re.Match[str]) -> str:
            symbol = match.group(1)
            return symbol if symbol in substitutions else match.group(0)

        normalized["expression"] = re.sub(
            r"\$([A-Za-z_][A-Za-z0-9_]*)(?!\.)",
            replace,
            expression,
        )
        return normalized

    @staticmethod
    def _concrete_substitution_symbols(
        request: CalculationRequest,
        declared_variables: set[str],
    ) -> set[str]:
        if request.operation not in {"substitute", "evaluate"}:
            return set()
        symbols: set[str] = set()
        for symbol, value in request.substitutions.items():
            if symbol in declared_variables:
                continue
            if not CalculationRequestValidator._is_plain_symbol(symbol):
                continue
            if isinstance(value, str) and value.startswith("$"):
                continue
            if str(value).strip():
                symbols.add(symbol)
        return symbols

    @staticmethod
    def _resolve_expression_reference(
        value: str,
        execution_context: dict[str, Any],
    ) -> str:
        if not isinstance(value, str) or "$" not in value:
            return value

        def replace(match: re.Match[str]) -> str:
            reference = match.group(0)
            resolved = CalculationRequestValidator.resolve_reference(reference, execution_context)
            return CalculationRequestValidator._format_embedded_reference_value(resolved, value) if resolved is not None else reference

        return re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", replace, value)

    @staticmethod
    def _format_embedded_reference_value(resolved: Any, expression: str) -> str:
        text = str(resolved).strip()
        if not text:
            return text
        expression_text = str(expression).strip()
        if expression_text == text or re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", expression_text):
            return text
        if CalculationRequestValidator._needs_reference_parentheses(text):
            return f"({text})"
        return text

    @staticmethod
    def _needs_reference_parentheses(value: str) -> bool:
        text = str(value or "").strip()
        if not text or "=" in text:
            return False
        if text.startswith("(") and text.endswith(")"):
            return False
        return bool(re.search(r"(?<!^)[+\-]", text))
