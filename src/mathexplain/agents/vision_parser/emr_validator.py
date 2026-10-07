"""EMR validation adapter for SGVP."""

from __future__ import annotations

from typing import Any

from mathexplain.schemas.emr import EMR, validate_emr_dict
from sympy import Symbol
from sympy.core.basic import Basic
from sympy.parsing.sympy_parser import parse_expr


class EMRValidator:
    """Run EMR schema validation and lightweight algebra parse checks."""

    def validate(self, emr: EMR) -> dict:
        """Validate an EMR object and return a compact validation report."""

        validate_emr_dict(emr.model_dump(mode="json"))
        report: dict[str, Any] = {
            "schema": "EMR-1.0",
            "valid": True,
            "validator": "pydantic+sympy_parse",
            "schema_valid": True,
            "sympy_parse_valid": True,
            "sympy_parse_skipped": False,
            "checked_fields": [],
            "errors": [],
        }

        if not self._should_check_sympy(emr):
            report["sympy_parse_skipped"] = True
            report["sympy_parse_skip_reason"] = "EMR is not algebra-related."
            return report

        local_dict = self._local_dict(emr)
        declared_symbols = set(local_dict)
        errors = []

        for field_path, value in self._sympy_fields(emr):
            report["checked_fields"].append(field_path)
            try:
                parsed_expr = parse_expr(value, local_dict=local_dict, evaluate=False)
            except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
                errors.append(
                    {
                        "field_path": field_path,
                        "value": value,
                        "message": f"SymPy parse failed: {exc}",
                    }
                )
                continue

            used_symbols = self._free_symbol_names(parsed_expr)
            if isinstance(parsed_expr, tuple | list):
                errors.append(
                    {
                        "field_path": field_path,
                        "value": value,
                        "message": "SymPy parsed a tuple/list; expected one scalar expression.",
                    }
                )
                continue
            undeclared_symbols = sorted(used_symbols - declared_symbols)
            if undeclared_symbols:
                errors.append(
                    {
                        "field_path": field_path,
                        "value": value,
                        "undeclared_symbols": undeclared_symbols,
                        "message": "Expression uses symbols not declared in emr.variables.",
                    }
                )

        if errors:
            report["valid"] = False
            report["sympy_parse_valid"] = False
            report["errors"] = errors

        return report

    @staticmethod
    def _should_check_sympy(emr: EMR) -> bool:
        source_problem_type = getattr(emr.source_problem_type, "value", emr.source_problem_type)
        return source_problem_type == "algebra" or emr.representation_type in {
            "equation_system",
            "expression",
        }

    @staticmethod
    def _local_dict(emr: EMR) -> dict[str, Symbol]:
        return {variable.symbol: Symbol(variable.symbol) for variable in emr.variables}

    @staticmethod
    def _sympy_fields(emr: EMR) -> list[tuple[str, str]]:
        fields = []
        for index, expression in enumerate(emr.expressions):
            fields.append((f"expressions[{index}].sympy", expression.sympy))
        for index, equation in enumerate(emr.equations):
            fields.append((f"equations[{index}].lhs_sympy", equation.lhs_sympy))
            fields.append((f"equations[{index}].rhs_sympy", equation.rhs_sympy))
        for index, constraint in enumerate(emr.constraints):
            fields.append((f"constraints[{index}].expression_sympy", constraint.expression_sympy))
        return fields

    @classmethod
    def _free_symbol_names(cls, parsed_expr: Any) -> set[str]:
        if isinstance(parsed_expr, Basic):
            return {str(symbol) for symbol in parsed_expr.free_symbols}
        if isinstance(parsed_expr, tuple | list):
            symbols: set[str] = set()
            for item in parsed_expr:
                symbols.update(cls._free_symbol_names(item))
            return symbols
        return set()
