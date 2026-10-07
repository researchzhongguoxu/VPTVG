"""Output-language policy for pipeline narration and rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from mathexplain.schemas.spr import SPR


OutputLanguage = Literal["auto", "en-US", "zh-CN"]


@dataclass(frozen=True)
class LanguageDecision:
    """Resolved source/output language metadata for a pipeline run."""

    detected_source_language: str
    output_language: str
    language_policy: str
    language_confidence: float
    chinese_char_count: int
    latin_word_count: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "detected_source_language": self.detected_source_language,
            "output_language": self.output_language,
            "language_policy": self.language_policy,
            "language_confidence": self.language_confidence,
            "chinese_char_count": self.chinese_char_count,
            "latin_word_count": self.latin_word_count,
        }


def resolve_output_language(spr: SPR, requested: OutputLanguage = "auto") -> LanguageDecision:
    """Resolve the language used downstream by Explainer, EDS, TTS, and Renderer.

    Policy:
    - Explicit en-US/zh-CN wins.
    - Auto uses dominant natural language in the SPR text.
    - Formula-only or ambiguous text defaults to en-US for paper-facing runs.
    """

    if requested not in {"auto", "en-US", "zh-CN"}:
        raise ValueError("output_language must be one of: auto, en-US, zh-CN")

    text = _spr_language_text(spr)
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = re.findall(r"[A-Za-z]{2,}", text)
    latin_count = len(latin_words)
    detected, confidence = _detect_language(zh_count, latin_count)

    if requested == "auto":
        output_language = detected if detected in {"en-US", "zh-CN"} else "en-US"
        policy = "auto_detected" if detected in {"en-US", "zh-CN"} else "auto_formula_default_en"
    else:
        output_language = requested
        policy = "explicit"

    return LanguageDecision(
        detected_source_language=detected,
        output_language=output_language,
        language_policy=policy,
        language_confidence=confidence,
        chinese_char_count=zh_count,
        latin_word_count=latin_count,
    )


def is_english_language(language: str | None) -> bool:
    """Return True when the requested/render language should use English UI text."""

    return str(language or "").lower().startswith("en")


def _detect_language(chinese_char_count: int, latin_word_count: int) -> tuple[str, float]:
    if chinese_char_count == 0 and latin_word_count == 0:
        return "formula-only", 0.5
    weighted_zh = chinese_char_count
    weighted_en = latin_word_count * 2
    total = weighted_zh + weighted_en
    if total <= 0:
        return "formula-only", 0.5
    if weighted_zh > weighted_en * 1.2:
        return "zh-CN", min(1.0, weighted_zh / total)
    if weighted_en > weighted_zh * 1.2:
        return "en-US", min(1.0, weighted_en / total)
    return "mixed", max(weighted_zh, weighted_en) / total


def _spr_language_text(spr: SPR) -> str:
    chunks: list[str] = [
        spr.problem_text or "",
        spr.problem_stem or "",
    ]
    chunks.extend(condition.text for condition in spr.conditions)
    chunks.extend(question.text for question in spr.questions)
    chunks.extend(formula.raw_text for formula in spr.formulas)
    chunks.extend(target.text for target in spr.targets)
    return "\n".join(chunk for chunk in chunks if chunk)
