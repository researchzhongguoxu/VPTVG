"""Plain-text cleanup helpers for baseline video rendering."""

from __future__ import annotations

import re


def normalize_plain_math_text(value: object) -> str:
    """Convert common lightweight LaTeX fragments into readable plain text."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("¡Á", "×").replace("¡Â", "÷")
    text = text.replace("กม", "×").replace("กย", "÷").replace("กฃ", ".")
    text = text.replace("\u2212", "-")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$", "")
    text = _replace_latex_frac(text)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", text)
    replacements = {
        r"\times": "×",
        r"\cdot": "×",
        r"\div": "÷",
        r"\pm": "±",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\quad": " ",
        r"\qquad": " ",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\\([A-Za-z]+)\s*\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        updated = pattern.sub(r"\1/\2", text)
        if updated == text:
            return text
        text = updated
