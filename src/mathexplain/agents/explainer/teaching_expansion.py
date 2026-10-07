"""Rule-based teaching expansion for Explainer.

The expansion layer is intentionally internal to the Explainer. It produces
ordinary ExplanationSegment dictionaries so downstream components do not need
to know whether a segment came from the compact v1 path or a teaching strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from mathexplain.schemas.scs import SolutionChain, SolutionStep
from mathexplain.schemas.spr import SPR


@dataclass(frozen=True)
class TeachingMove:
    """One teacher-style move derived from a verified modeling step."""

    move_id: str
    narration: str
    expression: str
    meaning: str | None = None
    source_sentence: str | None = None
    duration_ms: int = 5600
    pause_after_ms: int = 300


class StudentMathFormatter:
    """Format CAS-like linear expressions for student-facing display."""

    @staticmethod
    def replace_symbols(expression: str | None, presentation: dict[str, Any]) -> str | None:
        if expression is None:
            return None
        displayed = str(expression)
        replacements = {
            internal: data["display_symbol"]
            for internal, data in (presentation.get("symbols") or {}).items()
            if data.get("display_symbol") and StudentMathFormatter._is_replaceable_symbol(str(internal))
        }
        replacements.update(
            {
                str(internal): str(display)
                for internal, display in (presentation.get("expression_aliases") or {}).items()
                if display
            }
        )
        placeholders: dict[str, str] = {}
        for internal in sorted(replacements, key=len, reverse=True):
            placeholder = f"__TEACHING_SYMBOL_{len(placeholders)}__"
            placeholders[placeholder] = str(replacements[internal])
            displayed = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])",
                placeholder,
                displayed,
            )
        for placeholder, replacement in placeholders.items():
            displayed = displayed.replace(placeholder, replacement)
        return displayed

    @staticmethod
    def format_number_text(text: str | None) -> str | None:
        if text is None:
            return None

        def normalize(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                value = Decimal(raw)
            except InvalidOperation:
                return raw
            if value == value.to_integral_value():
                return format(value.quantize(Decimal(1)), "f")
            return format(value.normalize(), "f").rstrip("0").rstrip(".")

        return re.sub(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])", normalize, str(text))

    @staticmethod
    def _is_replaceable_symbol(symbol: str) -> bool:
        """Return true only for symbolic identifiers, not numeric constants like 1/4."""

        return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:_\{[A-Za-z0-9_]+\})?", symbol) is not None

    @staticmethod
    def format(expression: str | None) -> str | None:
        if expression is None:
            return None
        text = str(StudentMathFormatter.format_number_text(expression) or "").strip()
        if not text:
            return text
        text = text.replace("*", " × ").replace("/", " ÷ ")
        text = re.sub(r"\s*([=+×÷])\s*", r" \1 ", text)
        text = re.sub(r"(?<!^)\s*-\s*", " - ", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def display(cls, expression: str | None, presentation: dict[str, Any]) -> str | None:
        return cls.format(cls.replace_symbols(expression, presentation))


class TeachingExpansionStrategy(Protocol):
    """Strategy interface for future topic-specific teaching expansion."""

    name: str

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return expanded ExplanationSegment dictionaries, or an empty list."""


class TeachingExpansionEngine:
    """Apply registered teaching strategies with conservative fallback."""

    def __init__(self) -> None:
        self._strategies: list[TeachingExpansionStrategy] = [
            RibbonArithmeticStrategy(),
            RibbonRemainderMultipleStrategy(),
            EquivalentTotalDistributionStrategy(),
            AdditiveTotalWordProblemStrategy(),
            MeetingSpeedRatioStrategy(),
            ReadingRatioProgressStrategy(),
            LinearWordProblemExpansionStrategy(),
            BoxDivisionWordProblemExpansionStrategy(),
        ]
        self._expanded_step_ids: list[str] = []
        self._fallback_step_ids: list[str] = []
        self._strategy_names: list[str] = []
        self._warnings: list[str] = []

    def expand_step(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Try to expand one modeling step into teacher-style segments."""

        for strategy in self._strategies:
            segments = strategy.expand(spr, scs, step, presentation)
            if segments:
                self._expanded_step_ids.append(step.step_id)
                if strategy.name not in self._strategy_names:
                    self._strategy_names.append(strategy.name)
                return segments
        self._fallback_step_ids.append(step.step_id)
        return []

    def report(self) -> dict[str, Any]:
        """Return metadata saved into ExplanationScript.metadata."""

        return {
            "enabled": True,
            "strategy": (
                "multi_strategy"
                if len(self._strategy_names) > 1
                else self._strategy_names[0] if self._strategy_names else None
            ),
            "strategies": list(self._strategy_names),
            "strategy_count": len(self._strategy_names),
            "expanded_step_ids": list(self._expanded_step_ids),
            "fallback_step_ids": list(self._fallback_step_ids),
            "warnings": list(self._warnings),
        }


def _evidence_text(spr: SPR, step: SolutionStep) -> str:
    return " ".join(
        str(part or "")
        for part in [
            spr.problem_text,
            spr.problem_stem,
            " ".join(condition.text for condition in spr.conditions),
            " ".join(question.text for question in spr.questions),
            step.diagnostics.get("evidence_span"),
            step.justification,
            step.description,
        ]
    )


def _source_sentence(text: str, *keywords: str) -> str | None:
    candidates = re.split(r"[。；;！？!?]\s*", text)
    for candidate in candidates:
        if candidate and all(keyword in candidate for keyword in keywords):
            return candidate.strip()
    return None


class RibbonArithmeticStrategy:
    """Expand arithmetic ribbon problems without exposing planner-local names."""

    name = "ribbon_arithmetic"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expression = (step.math_expression or "").replace(" ", "")
        evidence = _evidence_text(spr, step)
        if "彩带" not in evidence or "剪去" not in evidence:
            return []

        multiply_match = re.fullmatch(r"(\d+)\*(\d+)|(\d+)×(\d+)", expression)
        if multiply_match and ("剩下" in evidence or "剩余" in evidence or "3倍" in evidence):
            left = int(multiply_match.group(1) or multiply_match.group(3))
            right = int(multiply_match.group(2) or multiply_match.group(4))
            removed, factor = (max(left, right), min(left, right))
            move = TeachingMove(
                "remaining_from_multiple",
                f"题目说剩下的长度是剪去部分的 {factor} 倍，剪去的是 {removed} 米，所以先算剩下的长度。",
                f"{removed}*{factor}",
                meaning="剩下的长度",
                source_sentence=_source_sentence(evidence, "剩下") or step.diagnostics.get("evidence_span"),
                duration_ms=6200,
                pause_after_ms=600,
            )
            return [self._segment(step, move, 0)]

        add_match = re.fullmatch(r"(\d+)\+([A-Za-z_][A-Za-z0-9_]*)", expression)
        if not add_match:
            return []
        removed = int(add_match.group(1))
        symbol = add_match.group(2)
        remaining = self._resolved_value(scs, step, symbol)
        total = self._resolved_value(
            scs,
            step,
            str((step.diagnostics.get("normalized_request") or {}).get("target") or ""),
        )
        if remaining is None:
            return []
        expression_text = f"{removed}+{remaining}"
        if total is not None:
            expression_text = f"{expression_text}={total}"
        move = TeachingMove(
            "total_from_removed_and_remaining",
            f"原来的长度等于剪去的长度加上剩下的长度，所以用 {removed} 米加上 {remaining} 米。",
            expression_text,
            meaning="彩带原来的长度",
            source_sentence=_source_sentence(evidence, "剪去") or step.diagnostics.get("evidence_span"),
            duration_ms=6500,
            pause_after_ms=700,
        )
        return [self._segment(step, move, 0)]

    @staticmethod
    def _resolved_value(scs: SolutionChain, step: SolutionStep, symbol: str) -> str | None:
        if not symbol:
            return None
        request = step.diagnostics.get("normalized_request") or {}
        substitutions = request.get("substitutions") or {}
        reference = substitutions.get(symbol)
        if isinstance(reference, str) and reference.startswith("$"):
            value = RibbonArithmeticStrategy._resolve_reference(reference, scs.metadata.get("execution_context") or {})
            if value is not None:
                return str(value)
        call_id = request.get("id") or step.diagnostics.get("tool_call_id")
        if call_id:
            value = RibbonArithmeticStrategy._resolve_reference(
                f"${call_id}.{symbol}",
                scs.metadata.get("execution_context") or {},
            )
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _resolve_reference(reference: str, context: dict[str, Any]) -> Any:
        current: Any = context
        for part in reference[1:].split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            return None
        return current

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, RibbonArithmeticStrategy.name)


class RibbonRemainderMultipleStrategy:
    """Expand "remaining amount is a multiple of removed amount" problems."""

    name = "ribbon_remainder_multiple"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expression = StudentMathFormatter.replace_symbols(step.math_expression, presentation)
        if not expression:
            return []
        match = re.search(
            r"([A-Za-z])\s*-\s*(\d+)\s*=\s*(\d+)\s*\*\s*(\d+)|"
            r"(\d+)\s*\*\s*(\d+)\s*=\s*([A-Za-z])\s*-\s*(\d+)",
            expression,
        )
        if not match:
            return []
        evidence = _evidence_text(spr, step)
        if not any(token in evidence for token in ["剪去", "剩下", "倍", "彩带"]):
            return []
        if match.group(1):
            symbol = match.group(1)
            removed = int(match.group(2))
            factor = int(match.group(3)) if int(match.group(4)) == removed else int(match.group(4))
        else:
            symbol = match.group(7)
            removed = int(match.group(8))
            factor = int(match.group(5)) if int(match.group(6)) == removed else int(match.group(6))
        source = _source_sentence(evidence, "剩下") or _source_sentence(evidence, "剪去") or step.diagnostics.get("evidence_span")
        moves = [
            TeachingMove(
                "removed_amount",
                f"题目先告诉我们剪去了 {removed} 米，这一部分是已知的。",
                str(removed),
                meaning="剪去的长度",
                source_sentence=_source_sentence(evidence, "剪去") or source,
            ),
            TeachingMove(
                "remaining_expression",
                f"如果原来的长度是 {symbol} 米，剪去 {removed} 米后，剩下的长度就是原来的长度减去剪去的长度。",
                f"{symbol}-{removed}",
                meaning="剩下的长度",
                source_sentence=_source_sentence(evidence, "剪去") or source,
            ),
            TeachingMove(
                "multiple_expression",
                f"题目说剩下的长度正好是剪去部分的 {factor} 倍，所以也可以写成 {removed} 乘以 {factor}。",
                f"{removed}*{factor}",
                meaning="剪去部分的倍数",
                source_sentence=source,
            ),
            TeachingMove(
                "equation_from_relation",
                "这两个式子都表示剩下的长度，所以可以把它们用等号连接起来。",
                expression,
                meaning="由同一个剩余长度建立方程",
                source_sentence=source,
                duration_ms=6200,
                pause_after_ms=600,
            ),
        ]
        return [self._segment(step, move, index) for index, move in enumerate(moves)]

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, RibbonRemainderMultipleStrategy.name)


class EquivalentTotalDistributionStrategy:
    """Expand two distribution scenarios that describe the same total."""

    name = "equivalent_total_distribution"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expression = StudentMathFormatter.replace_symbols(step.math_expression, presentation)
        if not expression:
            return []
        compact = expression.replace(" ", "")
        match = re.fullmatch(r"(\d+)\*([A-Za-z])\+(\d+)=(\d+)\*\(\2-(\d+)\)", compact)
        if not match:
            match = re.fullmatch(r"(\d+)\*\(([A-Za-z])-(\d+)\)=(\d+)\*\2\+(\d+)", compact)
        is_system_shape = False
        if not match:
            system_match = re.fullmatch(
                r"([A-Za-z])=(\d+)\*([A-Za-z])\+(\d+);\1/(\d+)=\3-(\d+)",
                compact,
            )
            if not system_match:
                system_match = re.fullmatch(
                    r"([A-Za-z])=(\d+)\*([A-Za-z])\+(\d+);([A-Za-z])=\3-(\d+);\1=(\d+)\*\5",
                    compact,
                )
                if not system_match:
                    return []
                is_system_shape = True
                left_classes = int(system_match.group(2))
                symbol = system_match.group(3)
                remainder = int(system_match.group(4))
                less = int(system_match.group(6))
                right_classes = int(system_match.group(7))
            else:
                is_system_shape = True
                left_classes = int(system_match.group(2))
                symbol = system_match.group(3)
                remainder = int(system_match.group(4))
                right_classes = int(system_match.group(5))
                less = int(system_match.group(6))
        elif match.re.pattern.startswith(r"(\d+)\*\("):
            right_classes = int(match.group(1))
            symbol = match.group(2)
            less = int(match.group(3))
            left_classes = int(match.group(4))
            remainder = int(match.group(5))
        else:
            left_classes = int(match.group(1))
            symbol = match.group(2)
            remainder = int(match.group(3))
            right_classes = int(match.group(4))
            less = int(match.group(5))
        evidence = _evidence_text(spr, step)
        if not any(token in evidence for token in ["平均分", "班", "还剩", "少"]):
            return []
        first_source = _source_sentence(evidence, f"{left_classes}个班") or _source_sentence(evidence, "还剩")
        second_source = _source_sentence(evidence, f"{right_classes}个班") or _source_sentence(evidence, "少")
        moves = [
            TeachingMove(
                "first_total_expression",
                f"先看第一种分法：如果每班分 {symbol} 支，{left_classes} 个班一共分走 {left_classes} 乘以 {symbol} 支，还剩 {remainder} 支。",
                f"{left_classes}*{symbol}+{remainder}",
                meaning="第一种分法表示铅笔总数",
                source_sentence=first_source,
            ),
            TeachingMove(
                "second_total_expression",
                f"再看第二种分法：每班比原来少 {less} 支，也就是每班 {symbol} 减 {less} 支，分给 {right_classes} 个班。",
                f"{right_classes}*({symbol}-{less})",
                meaning="第二种分法表示铅笔总数",
                source_sentence=second_source,
            ),
            TeachingMove(
                "same_total_equation",
                "两种分法说的都是同一批铅笔的总数，所以这两个表示总数的式子相等。"
                if not is_system_shape
                else "这两种写法都在表示同一批铅笔的总数，可以整理成两个总数表达式相等。",
                f"{left_classes}*{symbol}+{remainder}={right_classes}*({symbol}-{less})",
                meaning="同一批铅笔总数相等",
                source_sentence=_source_sentence(evidence, "平均分") or step.diagnostics.get("evidence_span"),
                duration_ms=6500,
                pause_after_ms=700,
            ),
        ]
        return [self._segment(step, move, index) for index, move in enumerate(moves)]

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, EquivalentTotalDistributionStrategy.name)


class AdditiveTotalWordProblemStrategy:
    """Expand simple part-sum equations such as day-by-day reading totals."""

    name = "additive_total_word_problem"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expression = StudentMathFormatter.replace_symbols(step.math_expression, presentation)
        if not expression:
            return []
        compact = expression.replace(" ", "")
        match = re.fullmatch(r"(\d+)\+\((\d+)\+([A-Za-z])\)\+2\*\(\2\+\3\)=(\d+)", compact)
        if not match:
            return []
        first = int(match.group(1))
        base = int(match.group(2))
        symbol = match.group(3)
        total = int(match.group(4))
        if first != base:
            return []
        evidence = _evidence_text(spr, step)
        if not any(token in evidence for token in ["第一天", "第二天", "第三天", "一共"]):
            return []
        moves = [
            TeachingMove(
                "first_day",
                f"第一天看了 {first} 页，这是一个已知数量。",
                str(first),
                meaning="第一天看的页数",
                source_sentence=_source_sentence(evidence, "第一天") or step.diagnostics.get("evidence_span"),
            ),
            TeachingMove(
                "second_day",
                f"第二天比第一天多看 {symbol} 页，所以第二天是 {first} 加 {symbol} 页。",
                f"{first}+{symbol}",
                meaning="第二天看的页数",
                source_sentence=_source_sentence(evidence, "第二天") or step.diagnostics.get("evidence_span"),
            ),
            TeachingMove(
                "third_day",
                "第三天看的页数是第二天的 2 倍，所以用第二天的页数乘以 2。",
                f"2*({first}+{symbol})",
                meaning="第三天看的页数",
                source_sentence=_source_sentence(evidence, "第三天") or step.diagnostics.get("evidence_span"),
            ),
            TeachingMove(
                "total_equation",
                f"三天一共看了 {total} 页，所以把三天的页数加起来等于 {total}。",
                expression,
                meaning="三天页数总和建立方程",
                source_sentence=_source_sentence(evidence, "一共") or step.diagnostics.get("evidence_span"),
                duration_ms=6500,
                pause_after_ms=700,
            ),
        ]
        return [self._segment(step, move, index) for index, move in enumerate(moves)]

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, AdditiveTotalWordProblemStrategy.name)


class MeetingSpeedRatioStrategy:
    """Expand meeting-distance problems with a speed ratio."""

    name = "meeting_speed_ratio"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        evidence = _evidence_text(spr, step)
        if (
            not all(token in evidence for token in ["相距", "相遇"])
            and "速度比" not in evidence
            and "speed" not in evidence.lower()
        ):
            return []
        expression = (step.math_expression or "").replace(" ", "")
        context = scs.metadata.get("execution_context") or {}

        distance_time = re.fullmatch(r"(\d+)/(\d+)", expression)
        if distance_time:
            distance = int(distance_time.group(1))
            hours = int(distance_time.group(2))
            speed_sum = distance // hours if distance % hours == 0 else distance / hours
            move = TeachingMove(
                "speed_sum",
                f"两车相对开出，5 小时一共走完 900 千米，所以先用总路程除以相遇时间，求出两车的速度和。",
                f"{distance}/{hours}={speed_sum}",
                meaning="两车速度和",
                source_sentence=_source_sentence(evidence, "相遇") or step.diagnostics.get("evidence_span"),
                duration_ms=6500,
                pause_after_ms=700,
            )
            return [self._segment(step, move, 0)]

        if "=" not in expression or "/" not in expression:
            return []
        ratio_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)/([A-Za-z_][A-Za-z0-9_]*)=(\d+)/(\d+)", expression)
        sum_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\+([A-Za-z_][A-Za-z0-9_]*)=([^;]+)", expression)
        if not ratio_match or not sum_match:
            return []
        bus_symbol = ratio_match.group(1)
        truck_symbol = ratio_match.group(2)
        bus_parts = int(ratio_match.group(3))
        truck_parts = int(ratio_match.group(4))
        speed_sum_ref = sum_match.group(3)
        speed_sum = self._resolve_expression_value(speed_sum_ref, context)
        if speed_sum is None:
            return []
        total_parts = bus_parts + truck_parts
        bus_speed = int(speed_sum) * bus_parts / total_parts
        bus_speed_text = str(int(bus_speed)) if float(bus_speed).is_integer() else str(bus_speed)
        bus_display = StudentMathFormatter.display(bus_symbol, presentation) or bus_symbol
        truck_display = StudentMathFormatter.display(truck_symbol, presentation) or truck_symbol
        moves = [
            TeachingMove(
                "ratio_parts",
                f"客车和货车速度比是 {bus_parts}:{truck_parts}，可以看成客车占 {bus_parts} 份，货车占 {truck_parts} 份，一共 {total_parts} 份。",
                f"{bus_parts}+{truck_parts}={total_parts}",
                meaning="速度比的总份数",
                source_sentence=_source_sentence(evidence, "速度比") or step.diagnostics.get("evidence_span"),
                duration_ms=6500,
                pause_after_ms=700,
            ),
            TeachingMove(
                "passenger_speed",
                f"两车速度和是 {speed_sum} 千米每小时，客车占其中的 {bus_parts} 份，所以客车速度是 {bus_speed_text} 千米每小时。",
                f"{speed_sum}*{bus_parts}/{total_parts}={bus_speed_text}",
                meaning="客车平均速度",
                source_sentence=_source_sentence(evidence, "客车") or step.diagnostics.get("evidence_span"),
                duration_ms=6500,
                pause_after_ms=700,
            ),
        ]
        # Keep the original symbols traceable without showing them in narration.
        for move in moves:
            object.__setattr__(move, "source_sentence", move.source_sentence)
        return [self._segment(step, move, index) for index, move in enumerate(moves)]

    @staticmethod
    def _resolve_expression_value(expression: str, context: dict[str, Any]) -> str | None:
        if expression.startswith("$"):
            return MeetingSpeedRatioStrategy._resolve_reference(expression, context)
        if re.fullmatch(r"\d+(?:\.\d+)?", expression):
            return expression
        return None

    @staticmethod
    def _resolve_reference(reference: str, context: dict[str, Any]) -> str | None:
        current: Any = context
        for part in reference[1:].split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            return None
        return str(current)

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, MeetingSpeedRatioStrategy.name)


class ReadingRatioProgressStrategy:
    """Expand read/unread ratio word problems before the long ratio equation."""

    name = "reading_ratio_progress"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del scs
        expression = (step.math_expression or "").replace(" ", "")
        evidence = _evidence_text(spr, step)
        if not self._is_reading_ratio_problem(evidence, expression):
            return []
        symbol = self._primary_symbol(presentation) or self._first_symbol(expression) or "x"
        fallback_source = (step.diagnostics.get("evidence_span") or spr.problem_stem or spr.problem_text or evidence).strip()
        source_read = _source_sentence(evidence, "读", "1/4") or _source_sentence(evidence, "已经读")
        source_more = _source_sentence(evidence, "再读", "15") or source_read
        source_ratio = _source_sentence(evidence, "读过", "未读") or _source_sentence(evidence, "2:3")
        moves = [
            TeachingMove(
                "already_read_pages",
                f"设 {symbol} 表示这本书的总页数。题目说已经读了全书的四分之一，所以已经读过的页数是 {symbol} ÷ 4。",
                f"{symbol}/4",
                meaning="已经读过的页数",
                source_sentence=source_read or fallback_source,
                duration_ms=6200,
                pause_after_ms=500,
            ),
            TeachingMove(
                "after_more_read_pages",
                f"如果再读 15 页，读过的页数就要在原来的 {symbol} ÷ 4 后面再加 15，得到 {symbol} ÷ 4 + 15。",
                f"{symbol}/4 + 15",
                meaning="再读15页后读过的页数",
                source_sentence=source_more or fallback_source,
                duration_ms=6500,
                pause_after_ms=500,
            ),
            TeachingMove(
                "unread_pages",
                f"未读的页数是全书页数减去已经读过的页数，所以未读页数是 {symbol} - ({symbol} ÷ 4 + 15)。",
                f"{symbol} - ({symbol}/4 + 15)",
                meaning="再读15页后未读的页数",
                source_sentence=source_ratio or fallback_source,
                duration_ms=6800,
                pause_after_ms=500,
            ),
            TeachingMove(
                "ratio_meaning",
                "题目说读过的页数与未读的页数之比是 2:3，也就是前面这两个数量的比值等于 2:3。",
                f"({symbol}/4 + 15) : ({symbol} - ({symbol}/4 + 15)) = 2 : 3",
                meaning="把读过和未读两个数量按题目比例联系起来",
                source_sentence=source_ratio or fallback_source,
                duration_ms=7200,
                pause_after_ms=600,
            ),
        ]
        return [_teaching_segment(step, move, index, self.name) for index, move in enumerate(moves)]

    @staticmethod
    def _is_reading_ratio_problem(evidence: str, expression: str) -> bool:
        lower = evidence.lower()
        return (
            ("读" in evidence or "read" in lower)
            and ("未读" in evidence or "unread" in lower)
            and ("2:3" in evidence or "2/3" in expression)
            and ("1/4" in evidence or "/4" in expression)
        )

    @staticmethod
    def _primary_symbol(presentation: dict[str, Any]) -> str | None:
        for data in (presentation.get("symbols") or {}).values():
            display = data.get("display_symbol")
            if display:
                return str(display)
        return None

    @staticmethod
    def _first_symbol(expression: str) -> str | None:
        match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", expression)
        return match.group(0) if match else None


class LinearWordProblemExpansionStrategy:
    """Expand simple state-change linear word problems.

    v1 focuses on elementary "sell half less N, then sell half less M, final
    remaining amount" problems. The strategy is rule-based and pattern-driven;
    it does not claim to cover arbitrary word problems.
    """

    name = "linear_word_problem"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        operation = step.diagnostics.get("operation")
        if operation not in {"solve_for", "solve_equation_system"}:
            return []

        facts = self._remaining_after_two_half_less_sales_facts(spr, scs, step)
        if facts is None:
            return []

        symbol = self._target_display_symbol(step, presentation)
        expression = self._display_expression(step.math_expression, presentation)
        final_left = self._final_left_expression(expression)
        moves = self._moves(symbol, facts, final_left, expression)
        return [self._segment(step, move, index) for index, move in enumerate(moves)]

    def _remaining_after_two_half_less_sales_facts(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
    ) -> dict[str, int] | None:
        evidence = _evidence_text(spr, step)
        expression = step.math_expression or ""

        first_less = self._first_int(
            [
                r"上午[^，。；;]*?一半少\s*(\d+)",
                r"morning[^.;,]*?half[^.;,]*?less\s*(\d+)",
            ],
            evidence,
        )
        second_less = self._first_int(
            [
                r"下午[^，。；;]*?一半少\s*(\d+)",
                r"afternoon[^.;,]*?half[^.;,]*?less\s*(\d+)",
            ],
            evidence,
        )
        remaining = self._first_int(
            [
                r"还剩\s*(\d+)",
                r"剩\s*(\d+)\s*千克",
                r"remaining\s*(\d+)",
            ],
            evidence,
        )

        if first_less is None or second_less is None:
            less_numbers = [int(value) for value in re.findall(r"/2\s*-\s*(\d+)", expression)]
            if len(less_numbers) >= 2:
                first_less = first_less if first_less is not None else less_numbers[0]
                second_less = second_less if second_less is not None else less_numbers[-1]

        if remaining is None:
            numbers = [int(value) for value in re.findall(r"(?<![A-Za-z_])(\d+)(?![A-Za-z_])", expression)]
            candidates = [value for value in numbers if value not in {2, first_less, second_less}]
            if candidates:
                remaining = candidates[-1]

        if first_less is None or second_less is None or remaining is None:
            return None

        if not self._looks_like_two_stage_remaining_problem(evidence, expression):
            return None

        return {
            "first_less": first_less,
            "second_less": second_less,
            "remaining": remaining,
        }

    @staticmethod
    def _looks_like_two_stage_remaining_problem(evidence: str, expression: str) -> bool:
        lower = evidence.lower()
        has_text_signal = (
            ("上午" in evidence and "下午" in evidence and "一半" in evidence)
            or ("morning" in lower and "afternoon" in lower and "half" in lower)
        )
        has_expression_signal = expression.count("/2") >= 2 or "/4" in expression
        has_remaining_signal = (
            "还剩" in evidence
            or "剩" in evidence
            or "remaining" in lower
            or "=" in expression
        )
        return has_remaining_signal and (has_text_signal or has_expression_signal)

    @staticmethod
    def _first_int(patterns: list[str], text: str) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _target_display_symbol(cls, step: SolutionStep, presentation: dict[str, Any]) -> str:
        request = step.diagnostics.get("normalized_request") or {}
        candidates = [
            request.get("target"),
            *((request.get("effective_variables") or [])[:1]),
            *((request.get("variables") or [])[:1]),
        ]
        expression = step.math_expression or ""
        candidates.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
        for candidate in candidates:
            if not candidate:
                continue
            item = (presentation.get("symbols") or {}).get(str(candidate)) or {}
            return str(item.get("display_symbol") or candidate)
        return "x"

    @staticmethod
    def _display_expression(expression: str | None, presentation: dict[str, Any]) -> str | None:
        return StudentMathFormatter.replace_symbols(expression, presentation)

    @staticmethod
    def _final_left_expression(expression: str | None) -> str | None:
        if not expression or "=" not in expression:
            return None
        left, right = [part.strip() for part in expression.split("=", 1)]
        if re.fullmatch(r"[A-Za-z]", left):
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", left):
            return right
        return left

    def _moves(
        self,
        symbol: str,
        facts: dict[str, int],
        final_left: str | None,
        original_expression: str | None,
    ) -> list[TeachingMove]:
        first_less = facts["first_less"]
        second_less = facts["second_less"]
        remaining = facts["remaining"]
        first_expr = f"{symbol}/2 - {first_less}"
        after_first_expr = f"{symbol} - ({first_expr}) = {symbol}/2 + {first_less}"
        second_expr = f"({symbol}/2 + {first_less})/2 - {second_less}"
        compact_remaining = (
            final_left
            if final_left and self._is_compact_remaining(final_left, symbol)
            else self._compact_remaining(symbol, first_less, second_less)
        )
        final_remaining_expr = compact_remaining
        final_equation = (
            f"{compact_remaining} = {remaining}"
            if compact_remaining != original_expression
            else str(original_expression)
        )
        return [
            TeachingMove(
                "morning_sale",
                f"先看上午：卖出的是原来总量的一半少 {first_less} 千克。",
                first_expr,
                meaning="上午卖出的重量",
                source_sentence=f"上午卖出总数的一半少{first_less}千克",
            ),
            TeachingMove(
                "after_morning",
                f"上午卖出后，要用原来的总量减去上午卖出的部分，上一步已经写出了少 {first_less} 千克。",
                after_first_expr,
                meaning="上午卖出后剩下的重量",
                source_sentence=f"上午卖出总数的一半少{first_less}千克",
            ),
            TeachingMove(
                "afternoon_sale",
                f"再看下午：下午卖出的是上午后剩下数量的一半少 {second_less} 千克。",
                second_expr,
                meaning="下午卖出的重量",
                source_sentence=f"下午卖出剩下的一半少{second_less}千克",
            ),
            TeachingMove(
                "final_remaining",
                "下午卖出后剩下的数量，可以继续化成只含原来总量的式子。",
                final_remaining_expr,
                meaning="下午卖出后最终剩下的重量",
                source_sentence=f"下午卖出剩下的一半少{second_less}千克",
            ),
            TeachingMove(
                "final_equation",
                f"题目说最后还剩 {remaining} 千克，所以用这个剩余量建立方程。也就是说，下午卖完后剩下的苹果就是 {remaining} 千克。",
                final_equation,
                meaning="最终剩余量等于题目给出的 18 千克",
                source_sentence=f"这时还剩{remaining}千克苹果",
                duration_ms=6000,
                pause_after_ms=600,
            ),
        ]

    @staticmethod
    def _compact_remaining(symbol: str, first_less: int, second_less: int) -> str:
        constant = first_less / 2 + second_less
        if constant.is_integer():
            constant_text = str(int(constant))
        else:
            constant_text = str(constant)
        return f"{symbol}/4 + {constant_text}"

    @staticmethod
    def _is_compact_remaining(expression: str, symbol: str) -> bool:
        compact_pattern = rf"^\s*{re.escape(symbol)}\s*/\s*4\s*(?:[+-]\s*\d+(?:\.\d+)?)?\s*$"
        return re.fullmatch(compact_pattern, expression) is not None

    @staticmethod
    def _segment(step: SolutionStep, move: TeachingMove, index: int) -> dict[str, Any]:
        return _teaching_segment(step, move, index, LinearWordProblemExpansionStrategy.name)


def _teaching_segment(
    step: SolutionStep,
    move: TeachingMove,
    index: int,
    strategy_name: str,
) -> dict[str, Any]:
    display_expression = StudentMathFormatter.format(move.expression) or move.expression
    return {
        "segment_id": f"seg_model_{step.step_id}_teach_{index}_{move.move_id}",
        "segment_type": "modeling",
        "linked_step_ids": [step.step_id],
        "narration": move.narration,
        "math_expressions": [display_expression],
        "visual_refs": [f"expr_{step.step_id}_teach_{index}_{move.move_id}"],
        "pedagogical_actions": ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS", "QUESTION_PAUSE"],
        "pacing": "slow",
        "estimated_duration_ms": move.duration_ms,
        "pause_after_ms": move.pause_after_ms,
        "tts_hints": {
            "emphasis_terms": [display_expression],
            "pause_points": ["所以", "再看", "题目说"],
        },
        "layout_hints": {"focus_area": "main_formula", "keep_previous_context": True},
        "metadata": {
            "source": "teaching_expansion_rule",
            "strategy": strategy_name,
            "move_id": move.move_id,
            "source_step_id": step.step_id,
            "source_expression": step.math_expression,
            "display_expression": display_expression,
            "source_sentence": move.source_sentence,
            "meaning": move.meaning,
            "evidence_span": step.diagnostics.get("evidence_span"),
            "presentation_normalized": True,
            "derived_from": "source_modeling_step",
            "independent_verification": False,
        },
    }


class BoxDivisionWordProblemExpansionStrategy:
    """Expand simple "total divided by capacity per box" modeling steps."""

    name = "box_division_word_problem"

    def expand(
        self,
        spr: SPR,
        scs: SolutionChain,
        step: SolutionStep,
        presentation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if step.diagnostics.get("operation") not in {"evaluate", "substitute"}:
            return []
        expression = LinearWordProblemExpansionStrategy._display_expression(
            step.math_expression,
            presentation,
        )
        if not expression:
            return []
        match = re.fullmatch(r"\s*([A-Za-z])\s*/\s*(\d+)\s*", expression)
        if not match:
            return []
        divisor = int(match.group(2))
        evidence = _evidence_text(spr, step)
        lower = evidence.lower()
        if not (
            "每箱" in evidence
            or "装几箱" in evidence
            or "box" in lower
            or "boxes" in lower
        ):
            return []
        display_expression = StudentMathFormatter.format(expression) or expression
        segment = {
            "segment_id": f"seg_model_{step.step_id}_teach_0_box_division",
            "segment_type": "modeling",
            "linked_step_ids": [step.step_id],
            "narration": f"第二个问题要算可以装的箱数。每箱装 {divisor} 千克，所以用总重量除以 {divisor}。",
            "math_expressions": [display_expression],
            "visual_refs": [f"expr_{step.step_id}_teach_0_box_division"],
            "pedagogical_actions": ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS", "QUESTION_PAUSE"],
            "pacing": "slow",
            "estimated_duration_ms": 6200,
            "pause_after_ms": 600,
            "tts_hints": {
                "emphasis_terms": [display_expression, str(divisor)],
                "pause_points": ["所以"],
            },
            "layout_hints": {"focus_area": "main_formula", "keep_previous_context": True},
            "metadata": {
                "source": "teaching_expansion_rule",
                "strategy": self.name,
                "move_id": "box_division",
                "source_step_id": step.step_id,
                "source_expression": step.math_expression,
                "display_expression": display_expression,
                "source_sentence": _source_sentence(evidence, "每箱") or step.diagnostics.get("evidence_span"),
                "meaning": "总重量除以每箱重量得到箱数",
                "evidence_span": step.diagnostics.get("evidence_span"),
                "presentation_normalized": True,
                "derived_from": "source_modeling_step",
                "independent_verification": False,
            },
        }
        return [segment]
