"""Optional DeepSeek planner for Solver."""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Protocol

from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.spr import SPR
from mathexplain.services.deepseek import DEFAULT_DEEPSEEK_MODEL, DeepSeekChatClient


class PlanningClient(Protocol):
    """Protocol for fake and real planning clients."""

    model: str
    provider: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a planning draft."""


class DeepSeekPlanner:
    """Generate planning drafts without producing trusted final answers."""

    SYSTEM_PROMPT = (
        "You are a planning assistant for MathExplainAgent Solver. "
        "Return only a JSON object with tool_calls for CAS-backed math work. "
        "Do not provide a trusted final answer. "
        "Every computational conclusion must be represented as a tool call draft. "
        "The ordered tool calls must continue until the last tool call directly computes "
        "the quantity asked by the final question, not just an intermediate value."
    )

    def __init__(self, client: PlanningClient | None = None) -> None:
        self.client = client

    def plan(
        self,
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable planning draft."""

        client = self.client or DeepSeekChatClient()
        user_prompt = self._user_prompt(emr, classification_result, spr)
        draft = client.generate_text(self.SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_tool_call_json(draft)
        return {
            "planner_provider": getattr(client, "provider", "deepseek"),
            "planner_model": getattr(client, "model", DEFAULT_DEEPSEEK_MODEL),
            "planner_role": "planning_draft",
            "planner_usage": dict(getattr(client, "last_usage", {}) or {}),
            "planner_raw_output": draft,
            "planner_output_summary": draft[:1000],
            "planner_tool_calls": parsed.get("tool_calls", []),
            "planner_parse_error": parsed.get("parse_error"),
        }

    @staticmethod
    def _user_prompt(
        emr: EMR,
        classification_result: ClassificationResult,
        spr: SPR | None,
    ) -> str:
        payload: dict[str, Any] = {
            "problem_id": emr.problem_id,
            "solver_route": classification_result.solver_route,
            "task_type": classification_result.task_type,
            "emr": emr.model_dump(mode="json"),
        }
        if spr is not None:
            payload["spr"] = spr.model_dump(mode="json")
        return (
            "Create an ordered list of tool call drafts for all safe CAS-backed computations needed to solve the problem.\n"
            "Return exactly this JSON shape, with no markdown:\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "id": "short_step_id",\n'
            '      "operation": "construct_expression | parse_expression | solve_equation_system | solve_for | substitute | evaluate",\n'
            '      "variables": ["x"],\n'
            '      "expressions": [{"name": "target_name", "expression": "3*x + 4"}],\n'
            '      "equations": [{"lhs": "2*x + 3", "rhs": "7"}],\n'
            '      "substitutions": {"x": "$previous_step_id.x"},\n'
            '      "target": "x or target_name",\n'
            '      "reason": "brief reason grounded in the problem statement",\n'
            '      "evidence_span": "source text span that supports the request"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Use multiple tool calls when solving needs multiple computations. "
            "For example, first solve an equation for x, then substitute x into the requested expression. "
            "Use references like $solve_x.x to reuse a prior CAS result.\n"
            "Critical final-target rules:\n"
            "- Identify what the final question asks for, then make the last tool call compute exactly that quantity.\n"
            "- The last tool call id or target should use a semantic name such as final_answer, total_distance, "
            "remaining_count, final_cost, profit, difference, change, monthly_payment, or requested_quantity.\n"
            "- Do not stop after computing an intermediate quantity such as remaining calories, savings, total time, "
            "one person's amount, a rate, a percent, or one team's time when the question asks for a converted, "
            "combined, remaining, difference, or final total value.\n"
            "- If a previous tool call gives an intermediate result, the final tool call should combine or convert it "
            "with the remaining given constants to answer the question.\n"
            "- Prefer references like $previous_step.target or $previous_step.result when reusing prior CAS output.\n"
            "For each tool call, variables must list every symbol needed by that CAS call: "
            "all symbols in equations, expressions, substitutions, and the target when it is a symbol. "
            "Do not use variables only for the output target; it is the CAS variable scope.\n"
            "Write reason and evidence_span in Chinese when the problem is Chinese.\n"
            "Use explicit multiplication, for example 3*x + 4, not 3x + 4.\n"
            "If the problem asks for an expression such as a father's age in terms of x, "
            "prefer operation=construct_expression.\n"
            "Input artifact JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_tool_call_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        candidates = [text]
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(text[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                repaired_candidate = DeepSeekPlanner._escape_unescaped_string_quotes(candidate)
                if repaired_candidate == candidate:
                    continue
                try:
                    data = json.loads(repaired_candidate)
                except json.JSONDecodeError:
                    continue
            if isinstance(data, dict):
                tool_calls = data.get("tool_calls")
                if isinstance(tool_calls, list):
                    return {"tool_calls": tool_calls, "parse_error": None}
                return {"tool_calls": [data], "parse_error": None}

        return {
            "tool_calls": [],
            "parse_error": "Planner output was not valid JSON with tool_calls.",
        }

    @staticmethod
    def _escape_unescaped_string_quotes(text: str) -> str:
        """Repair a common LLM JSON slip: unescaped quotes inside string values."""

        repaired: list[str] = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if escaped:
                repaired.append(char)
                escaped = False
                continue
            if char == "\\" and in_string:
                repaired.append(char)
                escaped = True
                continue
            if char == '"':
                if not in_string:
                    in_string = True
                    repaired.append(char)
                    continue
                next_non_space = DeepSeekPlanner._next_non_space(text, index + 1)
                if next_non_space in {":", ",", "}", "]", ""}:
                    in_string = False
                    repaired.append(char)
                else:
                    repaired.append('\\"')
                continue
            repaired.append(char)
        return "".join(repaired)

    @staticmethod
    def _next_non_space(text: str, start: int) -> str:
        for char in text[start:]:
            if not char.isspace():
                return char
        return ""
