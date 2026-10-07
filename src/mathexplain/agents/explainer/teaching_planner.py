"""TeachingPlan generation for Explainer."""

from __future__ import annotations

import json
import re
import queue
import threading
from collections.abc import Callable
from typing import Any
from typing import Protocol

from mathexplain.agents.explainer.teaching_expansion import StudentMathFormatter
from mathexplain.agents.explainer.teaching_expansion import TeachingExpansionEngine
from mathexplain.schemas.scs import SolutionChain, SolutionStep
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.teaching_plan import TeachingPlan
from mathexplain.schemas.verification import VerificationReport


def _is_english(output_language: str | None) -> bool:
    return str(output_language or "").lower().startswith("en")


class TeachingPlannerProviderError(ValueError):
    """Provider failure with optional raw response for debugging."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class TeachingPlannerProvider(Protocol):
    """Optional provider that can produce a TeachingPlan from verified artifacts."""

    def plan(
        self,
        *,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        rule_plan: TeachingPlan | None = None,
        output_language: str = "zh-CN",
    ) -> TeachingPlan:
        """Return a valid TeachingPlan."""


class TextPlanningClient(Protocol):
    """Minimal chat client interface used by DeepSeekTeachingPlannerProvider."""

    provider: str
    model: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response."""


class TeachingPlanQualityGate:
    """Detect plans that are structurally valid but poor teaching inputs."""

    BAD_PHRASES = (
        "根据题意整理出一个可以计算的数学关系",
        "根据题意得到的数学关系",
    )
    INTERNAL_PATTERNS = (
        r"\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        r"\bremaining_length\b",
        r"\btotal_length\b",
        r"\bsum_speed\b",
    )

    @classmethod
    def assess(cls, plan: TeachingPlan) -> dict[str, Any]:
        reasons: list[str] = []
        if plan.quality_report.fallback_step_ids:
            reasons.append("rule_plan_has_fallback")
        if not plan.moves:
            reasons.append("plan_has_no_moves")
        for move in plan.moves:
            text = " ".join(
                str(part or "")
                for part in [
                    move.narration,
                    move.display_expression,
                    move.source_sentence,
                    move.meaning,
                ]
            )
            if any(phrase in text for phrase in cls.BAD_PHRASES):
                reasons.append("plan_has_generic_narration")
            if cls._contains_internal_reference(text):
                reasons.append("plan_leaks_internal_reference")
            if not move.source_sentence:
                reasons.append("move_missing_source_sentence")
            if not move.meaning:
                reasons.append("move_missing_meaning")
            if move.display_expression and cls._looks_like_long_direct_equation(move.display_expression):
                reasons.append("move_directly_displays_long_equation")
        deduped = list(dict.fromkeys(reasons))
        return {
            "status": "passed" if not deduped else "needs_provider",
            "valid": not deduped,
            "reasons": deduped,
        }

    @classmethod
    def validate_provider_plan(cls, plan: TeachingPlan, scs: SolutionChain) -> list[str]:
        errors: list[str] = []
        final_values = {
            str(item.get("value"))
            for item in (scs.metadata.get("final_answer_items") or [])
            if item.get("value") is not None
        }
        for move in plan.moves:
            text = " ".join(
                str(part or "")
                for part in [
                    move.narration,
                    move.display_expression,
                    move.source_sentence,
                    move.meaning,
                ]
            )
            if cls._contains_internal_reference(text):
                errors.append(f"{move.move_id} leaks internal references.")
            if any(phrase in text for phrase in cls.BAD_PHRASES):
                errors.append(f"{move.move_id} uses generic fallback narration.")
            if not move.source_sentence:
                errors.append(f"{move.move_id} is missing source_sentence.")
            if not move.meaning:
                errors.append(f"{move.move_id} is missing meaning.")
            if move.move_type == "answer" and final_values:
                numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
                if numbers and not (numbers & final_values):
                    errors.append(f"{move.move_id} answer value is not backed by final_answer_items.")
        return errors

    @classmethod
    def _contains_internal_reference(cls, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in cls.INTERNAL_PATTERNS)

    @staticmethod
    def _looks_like_long_direct_equation(expression: str) -> bool:
        return (
            "=" in expression
            and len(expression) >= 28
            and (expression.count("÷") + expression.count("/") + expression.count("(")) >= 3
        )


    @classmethod
    def _should_reject_long_equation(cls, move: Any, moves: list[Any]) -> bool:
        if not move.display_expression or not cls._looks_like_long_direct_equation(move.display_expression):
            return False
        if len(moves) <= 2:
            return True
        return moves.index(move) == 0


class DeepSeekTeachingPlannerProvider:
    """DeepSeek-backed TeachingPlan provider for pedagogical planning only."""

    SYSTEM_PROMPT = """你是 MathExplainAgent 的教学规划器，只负责把已验证的解题链改写成老师式讲解计划。
必须只输出 JSON，不要输出 Markdown。
禁止重新求解，禁止修改最终答案，禁止把未经验证的新方程当作结论。
输出必须符合 TeachingPlan v1：problem_id, source_chain_id, planner_mode, moves, quality_report, metadata。
planner_mode 必须是 deepseek。
每个 move 必须有 move_id、move_type、source_step_ids、source_sentence、meaning、raw_expression、display_expression、narration、pedagogical_actions、estimated_duration_ms、pause_after_ms、metadata。
讲解要面向中国中小学生：先说题目依据，再说数量含义，再给局部表达式。
不要出现 $step1.xxx、remaining_length、sum_speed、total_length 等内部变量或引用。
不要使用“根据题意整理出一个可以计算的数学关系”这类空泛句。
"""
    ENGLISH_SYSTEM_PROMPT = """You are the Teaching Planner for MathExplainAgent. Convert an already verified solution chain into a teacher-like TeachingPlan for an English-speaking middle or high school learner. Output JSON only, with no Markdown. Do not solve the problem again, do not change the final answer, and do not introduce unverified equations as conclusions. The output must follow TeachingPlan v1 with: problem_id, source_chain_id, planner_mode, moves, quality_report, metadata. planner_mode must be deepseek. Every move must include move_id, move_type, source_step_ids, source_sentence, meaning, raw_expression, display_expression, narration, pedagogical_actions, estimated_duration_ms, pause_after_ms, metadata. Explain one local teaching action per move: cite the problem information, state the quantity meaning, then show the local expression. Use natural English narration. Do not expose internal references such as $step1.xxx, remaining_length, sum_speed, or total_length."""
    PROMPT_VERSION = "deepseek-teaching-planner-v1"

    def __init__(
        self,
        client: TextPlanningClient | None = None,
        client_factory: Callable[[], TextPlanningClient] | None = None,
    ) -> None:
        self.client = client
        self.client_factory = client_factory
        self.last_raw_response: str | None = None
        self.last_normalized_data: dict[str, Any] | None = None
        self.last_usage: dict[str, Any] = {}

    def plan(
        self,
        *,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        rule_plan: TeachingPlan | None = None,
        output_language: str = "zh-CN",
    ) -> TeachingPlan:
        client = self._client()
        system_prompt = self.ENGLISH_SYSTEM_PROMPT if _is_english(output_language) else self.SYSTEM_PROMPT
        raw = self._generate_text_with_timeout(
            client,
            system_prompt,
            self._user_prompt(spr, scs, verification_report, presentation, rule_plan, output_language),
        )
        self.last_raw_response = raw
        self.last_usage = dict(getattr(client, "last_usage", {}) or {})
        data = self._extract_json(raw)
        data = self._normalize_plan_data(data, spr, scs, verification_report, presentation, client, rule_plan, output_language)
        self.last_normalized_data = data
        plan = TeachingPlan.model_validate(data)
        errors = TeachingPlanQualityGate.validate_provider_plan(plan, scs)
        if errors:
            raise TeachingPlannerProviderError("; ".join(errors), raw_response=raw)
        return plan

    @staticmethod
    def _default_client() -> TextPlanningClient:
        from mathexplain.services.deepseek import DeepSeekChatClient

        return DeepSeekChatClient()

    def _client(self) -> TextPlanningClient:
        if self.client is None:
            self.client = self.client_factory() if self.client_factory else self._default_client()
        return self.client

    @staticmethod
    def _generate_text_with_timeout(
        client: TextPlanningClient,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        timeout_seconds = float(getattr(client, "timeout_seconds", 60) or 60)
        result_queue: queue.Queue[tuple[str, str | BaseException]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(("ok", client.generate_text(system_prompt, user_prompt)))
            except BaseException as exc:  # pragma: no cover - exercised by integration behavior
                result_queue.put(("error", exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(f"DeepSeek TeachingPlanner timed out after {timeout_seconds:g} seconds.")
        status, payload = result_queue.get_nowait()
        if status == "error":
            raise payload
        return str(payload)

    @classmethod
    def _extract_json(cls, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise ValueError("DeepSeek TeachingPlanner output was not valid JSON.")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek TeachingPlanner output must be a JSON object.")
        return parsed

    def _normalize_plan_data(
        self,
        data: dict[str, Any],
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        client: TextPlanningClient,
        rule_plan: TeachingPlan | None,
        output_language: str = "zh-CN",
    ) -> dict[str, Any]:
        if "moves" not in data:
            raise ValueError("DeepSeek TeachingPlanner output must include moves.")
        moves = []
        normalization_warnings: list[str] = []
        default_step_id = self._first_modeling_step_id(scs)
        default_source_sentence = self._default_source_sentence(spr, scs)
        for index, raw_move in enumerate(data.get("moves") or []):
            if not isinstance(raw_move, dict):
                raise ValueError("Every TeachingPlan move must be an object.")
            raw_expression = self._coerce_optional_text(
                raw_move.get("raw_expression"),
                f"moves.{index}.raw_expression",
                normalization_warnings,
            )
            display_expression = self._coerce_optional_text(
                raw_move.get("display_expression"),
                f"moves.{index}.display_expression",
                normalization_warnings,
                separator="\n",
            )
            if raw_expression and not display_expression:
                display_expression = StudentMathFormatter.display(raw_expression, presentation)
            elif display_expression:
                display_expression = StudentMathFormatter.display(display_expression, presentation)
            narration = self._coerce_required_text(
                raw_move.get("narration"),
                fallback=raw_move.get("meaning") or display_expression or "这一步先说明题目中的数量关系。",
                field_path=f"moves.{index}.narration",
                warnings=normalization_warnings,
            )
            narration = StudentMathFormatter.format_number_text(
                StudentMathFormatter.replace_symbols(narration, presentation) or narration
            ) or narration
            meaning = self._coerce_optional_text(
                raw_move.get("meaning"),
                f"moves.{index}.meaning",
                normalization_warnings,
            ) or narration
            meaning = StudentMathFormatter.format_number_text(
                StudentMathFormatter.replace_symbols(meaning, presentation) or meaning
            ) or meaning
            moves.append(
                {
                    "move_id": raw_move.get("move_id") or f"deepseek_move_{index + 1}",
                    "move_type": self._normalize_move_type(raw_move.get("move_type")),
                    "source_step_ids": self._coerce_string_list(
                        raw_move.get("source_step_ids"),
                        f"moves.{index}.source_step_ids",
                        normalization_warnings,
                    )
                    or ([default_step_id] if default_step_id else []),
                    "source_sentence": self._coerce_optional_text(
                        raw_move.get("source_sentence"),
                        f"moves.{index}.source_sentence",
                        normalization_warnings,
                    )
                    or default_source_sentence,
                    "meaning": meaning,
                    "raw_expression": raw_expression,
                    "display_expression": display_expression,
                    "narration": narration,
                    "pedagogical_actions": self._coerce_string_list(
                        raw_move.get("pedagogical_actions"),
                        f"moves.{index}.pedagogical_actions",
                        normalization_warnings,
                    ),
                    "estimated_duration_ms": self._coerce_nonnegative_int(
                        raw_move.get("estimated_duration_ms"),
                        default=6200,
                        field_path=f"moves.{index}.estimated_duration_ms",
                        warnings=normalization_warnings,
                    ),
                    "pause_after_ms": self._coerce_nonnegative_int(
                        raw_move.get("pause_after_ms"),
                        default=600,
                        field_path=f"moves.{index}.pause_after_ms",
                        warnings=normalization_warnings,
                    ),
                    "metadata": {
                        **self._coerce_dict(raw_move.get("metadata"), f"moves.{index}.metadata", normalization_warnings),
                        "source": "deepseek_teaching_planner",
                        "output_language": output_language,
                    },
                }
            )
            if not moves[-1]["pedagogical_actions"]:
                moves[-1]["pedagogical_actions"] = ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS", "QUESTION_PAUSE"]
        provider_metadata = {
            "planner": "DeepSeekTeachingPlannerProvider",
            "planner_mode": "deepseek",
            "provider": getattr(client, "provider", "deepseek"),
            "model": getattr(client, "model", "unknown"),
            "timeout_seconds": getattr(client, "timeout_seconds", None),
            "max_retries": getattr(client, "max_retries", None),
            "usage": dict(getattr(client, "last_usage", {}) or {}),
            "prompt_version": self.PROMPT_VERSION,
            "output_language": output_language,
            "rule_plan_quality": TeachingPlanQualityGate.assess(rule_plan) if rule_plan else None,
            "provider_used": "deepseek",
            "provider_attempted": True,
            "quality_gate_status": "passed",
            "normalization_warnings": normalization_warnings,
        }
        return {
            "schema_version": data.get("schema_version") or "TEACHING_PLAN-1.0",
            "problem_id": data.get("problem_id") or scs.problem_id or spr.metadata.get("problem_id", "unknown_problem"),
            "source_chain_id": data.get("source_chain_id") or scs.chain_id,
            "planner_mode": "deepseek",
            "moves": moves,
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_move_count": len(moves),
                "expanded_step_ids": list(
                    (data.get("quality_report") or {}).get("expanded_step_ids")
                    or [step_id for move in moves for step_id in move["source_step_ids"]]
                ),
                "fallback_step_ids": [],
                "strategy_names": ["deepseek_teaching_planner"],
                "errors": [],
                "warnings": list((data.get("quality_report") or {}).get("warnings") or []) + normalization_warnings,
                "metadata": {
                    "source_verification_status": verification_report.status,
                    "provider": "deepseek",
                },
            },
            "metadata": {
                **provider_metadata,
                **dict(data.get("metadata") or {}),
            },
        }

    @staticmethod
    def _coerce_optional_text(
        value: Any,
        field_path: str,
        warnings: list[str],
        *,
        separator: str = "；",
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, list):
            warnings.append(f"{field_path} was a list and was joined into a string.")
            parts = [str(item).strip() for item in value if str(item).strip()]
            return separator.join(parts) or None
        if isinstance(value, dict):
            warnings.append(f"{field_path} was an object and was serialized into a string.")
            return json.dumps(value, ensure_ascii=False)
        warnings.append(f"{field_path} was coerced into a string.")
        return str(value).strip() or None

    @classmethod
    def _coerce_required_text(
        cls,
        value: Any,
        *,
        fallback: Any,
        field_path: str,
        warnings: list[str],
    ) -> str:
        text = cls._coerce_optional_text(value, field_path, warnings)
        if text:
            return text
        warnings.append(f"{field_path} was missing and was filled from fallback context.")
        return str(fallback or "这一步先说明题目中的数量关系。").strip()

    @staticmethod
    def _coerce_string_list(value: Any, field_path: str, warnings: list[str]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            warnings.append(f"{field_path} was a string and was wrapped into a list.")
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        warnings.append(f"{field_path} was coerced into a one-item list.")
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _coerce_dict(value: Any, field_path: str, warnings: list[str]) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        warnings.append(f"{field_path} was not an object and was ignored.")
        return {}

    @staticmethod
    def _coerce_nonnegative_int(value: Any, *, default: int, field_path: str, warnings: list[str]) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            warnings.append(f"{field_path} was not an integer and was replaced with {default}.")
            return default
        if parsed < 0:
            warnings.append(f"{field_path} was negative and was replaced with {default}.")
            return default
        return parsed

    @staticmethod
    def _normalize_move_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"modeling", "calculation", "answer", "recap", "fallback"}:
            return text
        if any(token in text for token in ["answer", "conclude", "summary", "result"]):
            return "answer"
        if any(token in text for token in ["solve", "calculate", "compute", "substitute"]):
            return "calculation"
        if "recap" in text or "review" in text:
            return "recap"
        return "modeling"

    @staticmethod
    def _first_modeling_step_id(scs: SolutionChain) -> str | None:
        for step in scs.steps:
            if step.diagnostics.get("math_role") == "modeling":
                return step.step_id
        return None

    @staticmethod
    def _default_source_sentence(spr: SPR, scs: SolutionChain) -> str | None:
        for step in scs.steps:
            sentence = step.diagnostics.get("evidence_span")
            if sentence:
                return str(sentence)
        return spr.problem_stem or spr.problem_text or None

    @staticmethod
    def _user_prompt(
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        rule_plan: TeachingPlan | None,
        output_language: str = "zh-CN",
    ) -> str:
        modeling_steps = [
            {
                "step_id": step.step_id,
                "rule_name": step.rule_name,
                "math_expression": step.math_expression,
                "justification": step.justification,
                "evidence_span": step.diagnostics.get("evidence_span"),
                "normalized_request": step.diagnostics.get("normalized_request"),
            }
            for step in scs.steps
            if step.diagnostics.get("math_role") == "modeling"
        ]
        payload = {
            "problem": {
                "problem_id": scs.problem_id or spr.metadata.get("problem_id"),
                "text": spr.problem_text,
                "conditions": [condition.model_dump(mode="json") for condition in spr.conditions],
                "questions": [question.model_dump(mode="json") for question in spr.questions],
            },
            "verified_solution": {
                "chain_id": scs.chain_id,
                "final_answer": scs.final_answer,
                "final_answer_items": scs.metadata.get("final_answer_items") or [],
                "modeling_steps": modeling_steps,
            },
            "verification": {
                "status": verification_report.status,
                "valid": verification_report.valid,
                "summary": verification_report.summary,
            },
            "presentation": presentation,
            "output_language": output_language,
            "rule_plan_summary": (
                {
                    "planner_mode": rule_plan.planner_mode,
                    "moves": [move.model_dump(mode="json") for move in rule_plan.moves],
                    "quality_report": rule_plan.quality_report.model_dump(mode="json"),
                }
                if rule_plan is not None
                else None
            ),
            "requirements": [
                "输出 TeachingPlan JSON 对象。",
                "每个 move 解释一个局部教学动作，不要一开始直接抛长方程。",
                "display_expression 使用 × 和 ÷。",
                "不要出现内部引用或英文内部变量名。",
            ],
        }
        if _is_english(output_language):
            payload["requirements"] = [
                "Output one TeachingPlan JSON object.",
                "Each move should explain one local teaching action.",
                "Use natural English narration for an English-speaking learner.",
                "Use readable math symbols such as × and ÷ when helpful.",
                "Do not expose internal references or internal variable names.",
            ]
        return json.dumps(payload, ensure_ascii=False, indent=2)


class RuleBasedTeachingPlanner:
    """Generate TeachingPlan moves from verified SCS modeling steps."""

    planner_mode = "rule_based"

    def __init__(
        self,
        *,
        provider_mode: str = "rule",
        provider: TeachingPlannerProvider | None = None,
    ) -> None:
        if provider_mode not in {"rule", "auto", "deepseek"}:
            raise ValueError("provider_mode must be one of: rule, auto, deepseek")
        self.provider_mode = provider_mode
        self.provider = provider

    def plan(
        self,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        output_language: str = "zh-CN",
    ) -> TeachingPlan:
        rule_plan = self._rule_plan(spr, scs, verification_report, presentation, output_language=output_language)
        rule_quality = TeachingPlanQualityGate.assess(rule_plan)
        if self.provider_mode == "rule":
            return self._with_provider_metadata(
                rule_plan,
                provider_used="rule",
                provider_attempted=False,
                quality_gate=rule_quality,
            )
        if self.provider_mode == "auto" and rule_quality["valid"]:
            return self._with_provider_metadata(
                rule_plan,
                provider_used="rule",
                provider_attempted=False,
                quality_gate=rule_quality,
            )

        provider = self.provider or DeepSeekTeachingPlannerProvider()
        try:
            try:
                provider_plan = provider.plan(
                    spr=spr,
                    scs=scs,
                    verification_report=verification_report,
                    presentation=presentation,
                    rule_plan=rule_plan,
                    output_language=output_language,
                )
            except TypeError as exc:
                if "output_language" not in str(exc):
                    raise
                provider_plan = provider.plan(
                    spr=spr,
                    scs=scs,
                    verification_report=verification_report,
                    presentation=presentation,
                    rule_plan=rule_plan,
                )
        except Exception as exc:
            raw_response = getattr(provider, "last_raw_response", None)
            return self._with_provider_metadata(
                rule_plan,
                provider_used="rule",
                provider_attempted=True,
                quality_gate=rule_quality,
                fallback_reason=str(exc),
                provider_raw_response=raw_response,
            )
        return self._with_provider_metadata(
            provider_plan,
            provider_used="deepseek",
            provider_attempted=True,
            quality_gate={"status": "passed", "valid": True, "reasons": []},
        )

    def _rule_plan(
        self,
        spr: SPR,
        scs: SolutionChain,
        verification_report: VerificationReport,
        presentation: dict[str, Any],
        *,
        output_language: str = "zh-CN",
    ) -> TeachingPlan:
        engine = TeachingExpansionEngine()
        moves: list[dict[str, Any]] = []

        for step in scs.steps:
            if not self._is_modeling_step(step):
                continue
            expanded_segments = engine.expand_step(spr, scs, step, presentation)
            if expanded_segments:
                moves.extend(self._move_from_segment(segment) for segment in expanded_segments)
            else:
                moves.append(self._fallback_move(step, presentation, output_language=output_language))

        report = engine.report()
        errors: list[str] = []
        warnings = list(report.get("warnings") or [])
        if verification_report.valid is not True:
            errors.append("TeachingPlan requires a passed VerificationReport.")
        if not moves:
            warnings.append("No modeling moves were generated.")

        return TeachingPlan.model_validate(
            {
                "problem_id": scs.problem_id or spr.metadata.get("problem_id", "unknown_problem"),
                "source_chain_id": scs.chain_id,
                "planner_mode": self.planner_mode,
                "moves": moves,
                "quality_report": {
                    "valid": not errors,
                    "status": "passed" if not errors else "failed",
                    "checked_move_count": len(moves),
                    "expanded_step_ids": report.get("expanded_step_ids", []),
                    "fallback_step_ids": report.get("fallback_step_ids", []),
                    "strategy_names": report.get("strategies", []),
                    "errors": errors,
                    "warnings": warnings,
                    "metadata": {"source_verification_status": verification_report.status},
                },
                "metadata": {
                    "planner": "RuleBasedTeachingPlanner",
                    "planner_mode": self.planner_mode,
                    "strategy": report.get("strategy"),
                    "strategy_count": report.get("strategy_count", 0),
                    "provider_used": "rule",
                    "provider_attempted": False,
                    "provider_fallback_reason": None,
                    "quality_gate_status": "not_checked",
                    "output_language": output_language,
                },
            }
        )

    @staticmethod
    def _with_provider_metadata(
        plan: TeachingPlan,
        *,
        provider_used: str,
        provider_attempted: bool,
        quality_gate: dict[str, Any],
        fallback_reason: str | None = None,
        provider_raw_response: str | None = None,
    ) -> TeachingPlan:
        data = plan.model_dump(mode="json")
        data["metadata"] = {
            **data.get("metadata", {}),
            "provider_used": provider_used,
            "provider_attempted": provider_attempted,
            "provider_fallback_reason": fallback_reason,
            "quality_gate_status": quality_gate.get("status"),
            "quality_gate_reasons": quality_gate.get("reasons", []),
        }
        if provider_raw_response:
            data["metadata"]["provider_raw_response_preview"] = provider_raw_response[:4000]
        if fallback_reason:
            data["quality_report"]["warnings"] = list(data["quality_report"].get("warnings") or [])
            data["quality_report"]["warnings"].append(f"DeepSeek TeachingPlanner fallback: {fallback_reason}")
        return TeachingPlan.model_validate(data)

    @staticmethod
    def _is_modeling_step(step: SolutionStep) -> bool:
        return (
            step.diagnostics.get("math_role") == "modeling"
            and step.diagnostics.get("explainer_ready") is True
            and bool(step.math_expression)
        )

    @staticmethod
    def _move_from_segment(segment: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(segment.get("metadata") or {})
        return {
            "move_id": segment["segment_id"].replace("seg_model_", "move_model_", 1),
            "move_type": "modeling",
            "source_step_ids": list(segment.get("linked_step_ids") or []),
            "source_sentence": metadata.get("source_sentence"),
            "meaning": metadata.get("meaning"),
            "raw_expression": metadata.get("source_expression"),
            "display_expression": (segment.get("math_expressions") or [None])[0],
            "narration": segment["narration"],
            "pedagogical_actions": list(segment.get("pedagogical_actions") or []),
            "estimated_duration_ms": segment["estimated_duration_ms"],
            "pause_after_ms": segment.get("pause_after_ms", 0),
            "metadata": {
                **metadata,
                "source_segment_id": segment.get("segment_id"),
                "layout_hints": dict(segment.get("layout_hints") or {}),
                "tts_hints": dict(segment.get("tts_hints") or {}),
                "visual_refs": list(segment.get("visual_refs") or []),
            },
        }

    @staticmethod
    def _fallback_move(
        step: SolutionStep,
        presentation: dict[str, Any],
        *,
        output_language: str = "zh-CN",
    ) -> dict[str, Any]:
        from mathexplain.agents.explainer.teaching_expansion import StudentMathFormatter

        display_expression = StudentMathFormatter.display(step.math_expression, presentation)
        if str(output_language or "").lower().startswith("en"):
            narration = "Translate the problem information into a mathematical relationship."
            if display_expression:
                narration += f" The expression is {display_expression}."
            meaning = "A mathematical relationship from the problem statement."
        else:
            narration = "根据题意整理出一个可以计算的数学关系。"
            if display_expression:
                narration += f" 对应的式子是 {display_expression}。"
            meaning = "根据题意得到的数学关系"
        return {
            "move_id": f"move_model_{step.step_id}_fallback",
            "move_type": "fallback",
            "source_step_ids": [step.step_id],
            "source_sentence": step.diagnostics.get("evidence_span"),
            "meaning": meaning,
            "raw_expression": step.math_expression,
            "display_expression": display_expression,
            "narration": narration,
            "pedagogical_actions": ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS", "QUESTION_PAUSE"],
            "estimated_duration_ms": max(6200, len(narration) * 80),
            "pause_after_ms": 600,
            "metadata": {
                "source": step.diagnostics.get("source"),
                "source_step_id": step.step_id,
                "source_expression": step.math_expression,
                "display_expression": display_expression,
                "evidence_span": step.diagnostics.get("evidence_span"),
                "presentation_normalized": True,
                "fallback": True,
                "independent_verification": False,
                "output_language": output_language,
            },
        }
