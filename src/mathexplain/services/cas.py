"""Computer algebra service adapters."""

from __future__ import annotations

import re
from typing import Any

from sympy import Abs, Eq, Symbol, Tuple, ceiling, floor, solve
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class SymPyBackend:
    """Small JSON-serializable wrapper around SymPy operations."""

    backend_name = "sympy"
    _TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)

    def parse_expression(
        self,
        expression: str,
        variable_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse an expression and return a JSON-serializable report."""

        expression = self._normalize_expression(expression)
        variables = variable_symbols or []
        local_dict = self._local_dict(variable_symbols or [])
        try:
            parsed = self._parse_expr(expression, local_dict=local_dict, evaluate=False)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            return {
                "backend": self.backend_name,
                "operation": "parse_expression",
                "input": {
                    "expression": expression,
                    "variables": variables,
                },
                "output": {
                    "parsed_expression": None,
                    "free_symbols": [],
                },
                "success": False,
                "expression": expression,
                "parsed_expression": None,
                "free_symbols": [],
                "errors": [str(exc)],
                "warnings": [],
            }
        if not hasattr(parsed, "free_symbols"):
            return {
                "backend": self.backend_name,
                "operation": "parse_expression",
                "input": {
                    "expression": expression,
                    "variables": variables,
                },
                "output": {
                    "parsed_expression": str(parsed),
                    "free_symbols": [],
                },
                "success": False,
                "expression": expression,
                "parsed_expression": str(parsed),
                "free_symbols": [],
                "errors": ["SymPy parsed a tuple/list; expected one scalar expression."],
                "warnings": [],
            }
        free_symbols = self._free_symbols(parsed)
        return {
            "backend": self.backend_name,
            "operation": "parse_expression",
            "input": {
                "expression": expression,
                "variables": variables,
            },
            "output": {
                "parsed_expression": str(parsed),
                "free_symbols": free_symbols,
            },
            "success": True,
            "expression": expression,
            "parsed_expression": str(parsed),
            "free_symbols": free_symbols,
            "errors": [],
            "warnings": [],
        }

    def construct_expression(
        self,
        expression: str,
        variables: list[str],
        target: str | None = None,
    ) -> dict[str, Any]:
        """Confirm a constructed expression using parse-only validation."""

        parse_report = self.parse_statement(expression, variables)
        return {
            "backend": self.backend_name,
            "operation": "construct_expression",
            "input": {
                "expression": expression,
                "variables": variables,
                "target": target,
            },
            "output": {
                "confirmed_expression": expression if parse_report["success"] else None,
                "parsed_expression": parse_report["parsed_expression"],
                "free_symbols": parse_report["free_symbols"],
            },
            "success": parse_report["success"],
            "errors": parse_report["errors"],
            "warnings": parse_report["warnings"],
        }

    def parse_statement(
        self,
        expression: str,
        variable_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse either a scalar expression or a simple equality statement."""

        expression = self._normalize_expression(expression)
        if expression.count("=") != 1:
            return self.parse_expression(expression, variable_symbols)

        variables = variable_symbols or []
        local_dict = self._local_dict(variables)
        lhs_text, rhs_text = (part.strip() for part in expression.split("=", 1))
        try:
            lhs = self._parse_expr(lhs_text, local_dict=local_dict, evaluate=False)
            rhs = self._parse_expr(rhs_text, local_dict=local_dict, evaluate=False)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            return {
                "backend": self.backend_name,
                "operation": "parse_expression",
                "input": {
                    "expression": expression,
                    "variables": variables,
                },
                "output": {
                    "parsed_expression": None,
                    "free_symbols": [],
                },
                "success": False,
                "expression": expression,
                "parsed_expression": None,
                "free_symbols": [],
                "errors": [str(exc)],
                "warnings": [],
            }
        if not hasattr(lhs, "free_symbols") or not hasattr(rhs, "free_symbols"):
            return {
                "backend": self.backend_name,
                "operation": "parse_expression",
                "input": {
                    "expression": expression,
                    "variables": variables,
                },
                "output": {
                    "parsed_expression": None,
                    "free_symbols": [],
                },
                "success": False,
                "expression": expression,
                "parsed_expression": None,
                "free_symbols": [],
                "errors": ["SymPy parsed a tuple/list; expected one scalar expression on each side."],
                "warnings": [],
            }
        free_symbols = sorted(set(self._free_symbols(lhs)) | set(self._free_symbols(rhs)))
        parsed_expression = f"{lhs} = {rhs}"
        return {
            "backend": self.backend_name,
            "operation": "parse_expression",
            "input": {
                "expression": expression,
                "variables": variables,
            },
            "output": {
                "parsed_expression": parsed_expression,
                "free_symbols": free_symbols,
            },
            "success": True,
            "expression": expression,
            "parsed_expression": parsed_expression,
            "free_symbols": free_symbols,
            "errors": [],
            "warnings": [],
        }

    def solve_equation_system(
        self,
        equations: list[dict[str, str]],
        variables: list[str],
        target: str | None = None,
    ) -> dict[str, Any]:
        """Solve an equation system using SymPy and return a JSON-safe result."""

        local_dict = self._local_dict(variables)
        errors: list[str] = []
        parsed_equations = []
        for equation in equations:
            try:
                lhs = self._parse_expr(equation["lhs_sympy"], local_dict=local_dict, evaluate=False)
                rhs = self._parse_expr(equation["rhs_sympy"], local_dict=local_dict, evaluate=False)
                parsed_equations.append(Eq(lhs, rhs))
            except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
                errors.append(f"Failed to parse equation {equation.get('id', 'unknown')}: {exc}")

        if errors:
            return self._solve_result(False, equations, variables, target, [], errors)
        if not parsed_equations:
            return self._solve_result(False, equations, variables, target, [], ["No equations provided."])
        if not variables:
            return self._solve_result(False, equations, variables, target, [], ["No variables provided."])

        try:
            symbols = [local_dict[symbol] for symbol in variables]
            raw_solution = solve(parsed_equations, symbols, dict=True)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            return self._solve_result(False, equations, variables, target, [], [str(exc)])

        formatted_solution = self._format_solution(raw_solution, variables)
        return self._solve_result(
            bool(raw_solution),
            equations,
            variables,
            target,
            raw_solution,
            [] if raw_solution else ["SymPy returned no solution."],
            formatted_solution,
        )

    def solve_for(
        self,
        equations: list[dict[str, str]],
        variables: list[str],
        target: str,
    ) -> dict[str, Any]:
        """Solve equations for a requested target variable."""

        if target not in variables:
            return {
                "backend": self.backend_name,
                "operation": "solve_for",
                "input": {
                    "variables": variables,
                    "equations": equations,
                    "target": target,
                },
                "output": {
                    "raw_solution": [],
                    "formatted_solution": None,
                },
                "success": False,
                "errors": [f"Target variable {target!r} is not declared in variables."],
                "warnings": [],
            }

        local_dict = self._local_dict(variables)
        errors: list[str] = []
        parsed_equations = []
        for equation in equations:
            try:
                lhs = self._parse_expr(equation["lhs_sympy"], local_dict=local_dict, evaluate=False)
                rhs = self._parse_expr(equation["rhs_sympy"], local_dict=local_dict, evaluate=False)
                parsed_equations.append(Eq(lhs, rhs))
            except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
                errors.append(f"Failed to parse equation {equation.get('id', 'unknown')}: {exc}")

        if errors:
            result = self._solve_result(False, equations, variables, target, [], errors)
            result["operation"] = "solve_for"
            return result
        if not parsed_equations:
            result = self._solve_result(False, equations, variables, target, [], ["No equations provided."])
            result["operation"] = "solve_for"
            return result

        try:
            raw_solution = solve(parsed_equations, [local_dict[target]], dict=True)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            result = self._solve_result(False, equations, variables, target, [], [str(exc)])
            result["operation"] = "solve_for"
            return result

        formatted_solution = self._format_solution(raw_solution, [target])
        result = self._solve_result(
            bool(raw_solution),
            equations,
            variables,
            target,
            raw_solution,
            [] if raw_solution else ["SymPy returned no solution."],
            formatted_solution,
        )
        result["operation"] = "solve_for"
        return result

    def substitute(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        target: str | None = None,
    ) -> dict[str, Any]:
        """Substitute known values into an expression."""

        local_dict = self._local_dict(variables)
        normalized_expression = self._normalize_expression(expression)
        try:
            if normalized_expression.count("=") == 1:
                return self._substitute_statement(
                    normalized_expression,
                    variables,
                    substitutions,
                    target,
                )
            parsed_expression = self._parse_expr(normalized_expression, local_dict=local_dict, evaluate=True)
            if not hasattr(parsed_expression, "free_symbols"):
                return self._substitute_result(
                    False,
                    "substitute",
                    normalized_expression,
                    variables,
                    substitutions,
                    target,
                    None,
                    ["SymPy parsed a tuple/list; expected one scalar expression."],
                )
            missing_expression_symbols = sorted(
                str(symbol)
                for symbol in self._free_symbol_objects(parsed_expression)
                if str(symbol) not in local_dict
            )
            missing_substitution_symbols = sorted(
                symbol for symbol in substitutions if symbol not in local_dict
            )
            missing_symbols = sorted(set(missing_expression_symbols) | set(missing_substitution_symbols))
            if missing_symbols:
                return self._substitute_result(
                    False,
                    "substitute",
                    normalized_expression,
                    variables,
                    substitutions,
                    target,
                    None,
                    [
                        "Missing symbols in CAS variable scope: "
                        + ", ".join(missing_symbols)
                    ],
                    missing_symbols=missing_symbols,
                )
            parsed_substitutions = {
                local_dict[symbol]: self._parse_expr(str(value), local_dict=local_dict, evaluate=True)
                for symbol, value in substitutions.items()
            }
            substituted = parsed_expression.subs(parsed_substitutions)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            return self._substitute_result(
                False,
                "substitute",
                    normalized_expression,
                variables,
                substitutions,
                target,
                None,
                [str(exc)],
            )

        return self._substitute_result(
            True,
            "substitute",
            normalized_expression,
            variables,
            substitutions,
            target,
            self._format_value(substituted),
            [],
        )

    def evaluate_expression(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str] | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate an expression, optionally with substitutions."""

        result = self.substitute(expression, variables, substitutions or {}, target)
        result["operation"] = "evaluate"
        if not result.get("success"):
            return result

        value = (result.get("output") or {}).get("value")
        if value is None:
            return result
        local_dict = self._local_dict(variables)
        try:
            parsed_value = self._parse_expr(str(value), local_dict=local_dict, evaluate=True)
        except Exception as exc:  # pragma: no cover - exact SymPy exception types vary.
            result["success"] = False
            result.setdefault("errors", []).append(str(exc))
            return result
        if not hasattr(parsed_value, "free_symbols"):
            result["success"] = False
            result.setdefault("errors", []).append(
                "SymPy parsed a tuple/list; expected one scalar expression."
            )
            return result
        unresolved_symbols = self._free_symbols(parsed_value)
        if unresolved_symbols:
            result["success"] = False
            result.setdefault("errors", []).append(
                "Evaluation result still contains unresolved symbols: "
                + ", ".join(unresolved_symbols)
            )
            result["missing_symbols"] = unresolved_symbols
        return result

    @staticmethod
    def _local_dict(variable_symbols: list[str]) -> dict[str, Symbol]:
        local_dict: dict[str, Any] = {symbol: Symbol(symbol) for symbol in variable_symbols}
        local_dict.update({"Abs": Abs, "Eq": Eq, "Tuple": Tuple, "ceil": ceiling, "ceiling": ceiling, "floor": floor})
        return local_dict

    @staticmethod
    def _normalize_expression(expression: str) -> str:
        if not isinstance(expression, str):
            return expression
        normalized = re.sub(r"\bderivative\s*\(", "diff(", expression, flags=re.IGNORECASE)
        normalized = re.sub(r"\|([^|]+)\|", r"Abs(\1)", normalized)
        return normalized

    def _substitute_statement(
        self,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        target: str | None,
    ) -> dict[str, Any]:
        lhs, rhs = (part.strip() for part in expression.split("=", 1))
        statement_variables = sorted(set(variables) | set(substitutions))
        lhs_result = self.substitute(lhs, statement_variables, substitutions, target)
        rhs_result = self.substitute(rhs, statement_variables, substitutions, target)
        if not lhs_result.get("success") or not rhs_result.get("success"):
            errors = list(lhs_result.get("errors") or []) + list(rhs_result.get("errors") or [])
            missing = sorted(set(lhs_result.get("missing_symbols") or []) | set(rhs_result.get("missing_symbols") or []))
            return self._substitute_result(
                False,
                "substitute",
                expression,
                statement_variables,
                substitutions,
                target,
                None,
                errors,
                missing_symbols=missing,
            )
        lhs_value = (lhs_result.get("output") or {}).get("value")
        rhs_value = (rhs_result.get("output") or {}).get("value")
        return self._substitute_result(
            True,
            "substitute",
            expression,
            statement_variables,
            substitutions,
            target,
            f"{lhs_value} = {rhs_value}",
            [],
        )

    @classmethod
    def _parse_expr(cls, expression: str, *, local_dict: dict[str, Symbol], evaluate: bool):
        return parse_expr(
            expression,
            local_dict=local_dict,
            transformations=cls._TRANSFORMATIONS,
            evaluate=evaluate,
        )

    @staticmethod
    def _free_symbol_objects(parsed: Any) -> set[Any]:
        if not hasattr(parsed, "free_symbols"):
            return set()
        return set(parsed.free_symbols)

    @classmethod
    def _free_symbols(cls, parsed: Any) -> list[str]:
        return sorted(str(symbol) for symbol in cls._free_symbol_objects(parsed))

    @staticmethod
    def _format_value(value: Any) -> str:
        if getattr(value, "func", None) is Eq and hasattr(value, "lhs") and hasattr(value, "rhs"):
            return f"{value.lhs} = {value.rhs}"
        return str(value)

    @staticmethod
    def _format_solution(raw_solution: list[dict[Any, Any]], variables: list[str]) -> str | None:
        if not raw_solution:
            return None
        first_solution = raw_solution[0]
        parts = []
        for variable in variables:
            symbol = Symbol(variable)
            if symbol in first_solution:
                parts.append(f"{variable} = {first_solution[symbol]}")
        return ", ".join(parts) if parts else str(first_solution)

    def _solve_result(
        self,
        success: bool,
        equations: list[dict[str, str]],
        variables: list[str],
        target: str | None,
        raw_solution: list[dict[Any, Any]],
        errors: list[str],
        formatted_solution: str | None = None,
    ) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "operation": "solve_equation_system",
            "input": {
                "variables": variables,
                "equations": equations,
                "target": target,
            },
            "output": {
                "raw_solution": [
                    {str(symbol): str(value) for symbol, value in solution.items()}
                    for solution in raw_solution
                ],
                "formatted_solution": formatted_solution,
            },
            "success": success,
            "errors": errors,
            "warnings": [],
        }

    def _substitute_result(
        self,
        success: bool,
        operation: str,
        expression: str,
        variables: list[str],
        substitutions: dict[str, str],
        target: str | None,
        value: str | None,
        errors: list[str],
        missing_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "operation": operation,
            "input": {
                "expression": expression,
                "variables": variables,
                "substitutions": substitutions,
                "target": target,
            },
            "output": {
                "value": value,
                "formatted_solution": value,
            },
            "success": success,
            "errors": errors,
            "missing_symbols": missing_symbols or [],
            "warnings": [],
        }


CASService = SymPyBackend
