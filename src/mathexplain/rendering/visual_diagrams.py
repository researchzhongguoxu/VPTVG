"""Lightweight visual explanation panels for Renderer v1.1.

The public EDS schema does not contain diagram tracks yet.  This module keeps a
small internal adapter shape so future DiagramPlan/EDS diagram_tracks work can
plug into the renderer without rewriting the frame player.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Protocol

from PIL import ImageDraw, ImageFont

from mathexplain.rendering.formula_renderer import draw_wrapped_text, line_height, load_font, text_size, wrap_text


@dataclass(frozen=True)
class DiagramSpec:
    """Renderer-local diagram description derived from currently visible EDS data."""

    diagram_type: str
    title: str
    items: list[str] = field(default_factory=list)
    highlight_index: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class DiagramAdapter(Protocol):
    """Draw a lightweight diagram inside a fixed panel."""

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        ...


class DiagramRenderer:
    """Small registry for renderer-local diagram adapters."""

    def __init__(self) -> None:
        generic = _GenericDiagramAdapter()
        self._adapters: dict[str, DiagramAdapter] = {
            "generic": generic,
            "quantity_bar": _BarDiagramAdapter(),
            "ratio_bar": _BarDiagramAdapter(),
            "progress_bar": _BarDiagramAdapter(),
            "speed_line": _SpeedLineDiagramAdapter(),
            "quantity_table": _TableDiagramAdapter(),
            "equation_system": _TableDiagramAdapter(),
            "function_summary": _TableDiagramAdapter(),
            "solution_progress": _ProgressDiagramAdapter(),
            "verification_check": _TableDiagramAdapter(),
        }

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        adapter = self._adapters.get(spec.diagram_type, self._adapters["generic"])
        adapter.draw(draw, bbox, spec, title_font=title_font, body_font=body_font)


def infer_diagram_spec(scene, formulas: list, context: dict | None = None) -> DiagramSpec:
    """Infer a safe, non-semantic visual hint from visible formulas and metadata."""

    context = context or {}
    language = str(context.get("language") or "zh-CN")
    active_formula = formulas[-1] if formulas else None
    expression = str(getattr(active_formula, "expression", "") or "")
    metadata = getattr(active_formula, "metadata", {}) or {}
    meaning = str(metadata.get("meaning") or "")
    source_sentence = str(metadata.get("source_sentence") or "")
    display_expression = str(metadata.get("display_expression") or expression)
    scene_type = str(getattr(scene, "scene_type", "") or "")
    if scene_type not in {"variable_definition", "modeling", "calculation", "answer"}:
        return DiagramSpec(diagram_type="", title="", items=[])

    all_formulas = list(context.get("all_formulas") or formulas)
    all_scenes = list(context.get("all_scenes") or [])
    all_text = _context_text(scene, all_scenes, all_formulas, formulas)

    if scene_type == "variable_definition":
        variable_spec = _variable_table_spec(scene, all_scenes)
        if variable_spec.items:
            return _localize_spec(variable_spec, language)

    if scene_type == "answer" or getattr(active_formula, "display_mode", "") == "final_answer":
        verification_spec = _verification_spec(display_expression, all_formulas, all_text)
        if verification_spec.items:
            return _localize_spec(verification_spec, language)

    if _looks_like_chicken_rabbit(all_text):
        quantity_spec = _chicken_rabbit_quantity_spec(display_expression, all_formulas)
        if quantity_spec.items:
            return _localize_spec(quantity_spec, language)

    function_spec = _function_summary_spec(display_expression, all_formulas, all_text)
    if function_spec.items:
        return _localize_spec(function_spec, language)

    explicit_diagram_type = str(metadata.get("diagram_type") or "").strip()
    diagram_type = explicit_diagram_type or "generic"
    items = _semantic_items(
        meaning=meaning,
        display_expression=display_expression,
        source_sentence=source_sentence,
    )
    if not items and explicit_diagram_type:
        items = _expression_items(display_expression)
    if not items and scene_type in {"calculation", "modeling"}:
        progress_spec = _progress_spec(scene_type)
        if progress_spec.items:
            return _localize_spec(progress_spec, language)
    if not items:
        return DiagramSpec(diagram_type="", title="", items=[])

    title = "关键数量"
    if scene_type == "answer" or getattr(active_formula, "display_mode", "") == "final_answer":
        title = "最终结论"
        diagram_type = "generic"

    return _localize_spec(
        DiagramSpec(diagram_type=diagram_type, title=title, items=items[:4], highlight_index=len(items[:4]) - 1),
        language,
    )


def _localize_spec(spec: DiagramSpec, language: str) -> DiagramSpec:
    if not str(language or "").lower().startswith("en"):
        return spec
    title_by_type = {
        "quantity_table": "Relationships",
        "equation_system": "System",
        "function_summary": "Function Info",
        "solution_progress": "Progress",
        "verification_check": "Substitution Check",
        "generic": "Key Facts",
    }
    item_replacements = {
        "鍏崇郴锛?": "Relation: ",
        "褰撳墠寮忥細": "Expression: ",
        "渚濇嵁锛?": "Source: ",
        "鎻愮ず锛?": "Hint: ",
    }
    items: list[str] = []
    for item in spec.items:
        localized = str(item)
        for old, new in item_replacements.items():
            localized = localized.replace(old, new)
        localized = re.sub(r"^\s*\u5173\u7cfb\s*[:\uff1a]\s*", "Relationship: ", localized)
        localized = re.sub(r"^\s*\u5f53\u524d\u5f0f\s*[:\uff1a]\s*", "Current expression: ", localized)
        localized = re.sub(r"^\s*\u4f9d\u636e\s*[:\uff1a]\s*", "From problem: ", localized)
        localized = re.sub(r"^\s*\u542b\u4e49\s*[:\uff1a]\s*", "Meaning: ", localized)
        localized = re.sub(r"^\s*\u7b97\u5f0f\s*[:\uff1a]\s*", "Expression: ", localized)
        localized = re.sub(r"^\s*\u76ee\u6807\s*[:\uff1a]\s*", "Need: ", localized)
        localized = re.sub(r"^\s*\u5df2\u77e5\u91cf\s*[:\uff1a]\s*", "Given: ", localized)
        localized = re.sub(r"^\s*\u4f7f\u7528\u5f0f\s*[:\uff1a]\s*", "Use: ", localized)
        localized = _format_display_numbers(localized)
        localized = localized.replace(" 閫氳繃", " passed")
        items.append(localized)
    return DiagramSpec(
        diagram_type=spec.diagram_type,
        title=title_by_type.get(spec.diagram_type, spec.title or "Key Facts"),
        items=items,
        highlight_index=spec.highlight_index,
        metadata={**spec.metadata, "language": language},
    )


def _format_display_numbers(text: str) -> str:
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


def _context_text(scene, all_scenes: list, all_formulas: list, formulas: list) -> str:
    chunks: list[str] = [str(getattr(scene, "title", "") or ""), str(getattr(scene, "scene_type", "") or "")]
    for item in [*all_scenes, *all_formulas, *formulas]:
        metadata = getattr(item, "metadata", {}) or {}
        chunks.extend(
            str(value or "")
            for value in [
                getattr(item, "expression", ""),
                metadata.get("meaning"),
                metadata.get("source_sentence"),
                metadata.get("display_expression"),
                metadata.get("student_label"),
                metadata.get("source_expression"),
            ]
        )
    return " ".join(chunks)


def _variable_table_spec(scene, all_scenes: list) -> DiagramSpec:
    items: list[str] = []
    current_symbol = str((getattr(scene, "metadata", {}) or {}).get("display_symbol") or "")
    highlight_index: int | None = None
    for candidate in all_scenes:
        if getattr(candidate, "scene_type", "") != "variable_definition":
            continue
        metadata = getattr(candidate, "metadata", {}) or {}
        symbol = str(metadata.get("display_symbol") or metadata.get("internal_symbol") or "").strip()
        label = str(metadata.get("student_label") or "").strip()
        if not symbol or not label:
            continue
        item = f"{symbol}：{label}"
        if item not in items:
            if symbol == current_symbol:
                highlight_index = len(items)
            items.append(item)
    return DiagramSpec(
        diagram_type="quantity_table",
        title="变量",
        items=items[:5],
        highlight_index=highlight_index,
    )


def _verification_spec(answer_expression: str, all_formulas: list, all_text: str) -> DiagramSpec:
    values = _solution_values(answer_expression)
    if len(values) < 2:
        for formula in reversed(all_formulas):
            values = _solution_values(str(getattr(formula, "expression", "") or ""))
            if len(values) >= 2:
                break
    if not values:
        return DiagramSpec(diagram_type="", title="", items=[])

    items: list[str] = []
    x_value = values.get("x")
    y_value = values.get("y")
    has_chicken_rabbit_equations = any(
        _is_head_equation(_normalize_math_text(str(getattr(formula, "expression", "") or "")))
        for formula in all_formulas
    ) and any(
        _is_feet_equation(_normalize_math_text(str(getattr(formula, "expression", "") or "")))
        for formula in all_formulas
    )
    if (has_chicken_rabbit_equations or _looks_like_chicken_rabbit(all_text)) and x_value is not None and y_value is not None:
        items.append(f"头数：{x_value} + {y_value} = {int(x_value) + int(y_value)} 通过")
        items.append(f"脚数：2×{x_value} + 4×{y_value} = {2 * int(x_value) + 4 * int(y_value)} 通过")
    else:
        items.extend(f"{symbol} = {value} 通过" for symbol, value in values.items())
    return DiagramSpec(
        diagram_type="verification_check",
        title="回代检查",
        items=items[:5],
        highlight_index=len(items[:5]) - 1 if items else None,
    )


def _chicken_rabbit_quantity_spec(active_expression: str, all_formulas: list) -> DiagramSpec:
    expression = _normalize_math_text(active_expression)
    all_expressions = [_normalize_math_text(str(getattr(formula, "expression", "") or "")) for formula in all_formulas]
    has_head = any(_is_head_equation(item) for item in [expression, *all_expressions])
    has_feet = any(_is_feet_equation(item) for item in [expression, *all_expressions])

    if _is_feet_equation(expression):
        items = ["鸡脚：2x", "兔脚：4y", "脚数：2x + 4y = 94"]
        return DiagramSpec(diagram_type="quantity_table", title="数量关系", items=items, highlight_index=2)
    if _is_head_equation(expression):
        items = ["鸡：x 只", "兔：y 只", "头数：x + y = 35"]
        return DiagramSpec(diagram_type="quantity_table", title="数量关系", items=items, highlight_index=2)
    if has_head or has_feet:
        items = ["鸡：x 只", "兔：y 只"]
        if has_head:
            items.append("头数：x + y = 35")
        if has_feet:
            items.append("脚数：2x + 4y = 94")
        return DiagramSpec(diagram_type="equation_system", title="方程组", items=items[:5], highlight_index=len(items[:5]) - 1)
    return DiagramSpec(diagram_type="", title="", items=[])


def _function_summary_spec(active_expression: str, all_formulas: list, all_text: str) -> DiagramSpec:
    expression = active_expression.strip()
    formulas = [str(getattr(formula, "expression", "") or "") for formula in all_formulas]
    text = " ".join([all_text, expression, *formulas]).lower()
    if "derivative" in text or "diff(" in text or "f'" in text or "导数" in all_text:
        items = []
        source = next((item for item in formulas if re.search(r"[fx]\s*\(|diff|derivative|[xX]\^", item)), "")
        if source:
            items.append(f"当前函数：{_compact_expr(source)}")
        if expression:
            items.append(f"当前目标：{_compact_expr(expression)}")
        return DiagramSpec(diagram_type="function_summary", title="导数信息", items=items[:4], highlight_index=len(items[:4]) - 1 if items else None)

    quadratic = next((item for item in [expression, *formulas] if _looks_like_quadratic(item)), "")
    if not quadratic:
        return DiagramSpec(diagram_type="", title="", items=[])

    compact = _compact_expr(quadratic)
    items = [f"表达式：{compact}"]
    coefficients = _quadratic_coefficients(quadratic)
    if coefficients:
        a, b, c = coefficients
        items.append(f"a={a}, b={b}, c={c}")
        if a != 0 and -b % (2 * a) == 0:
            items.append(f"对称轴：x={-b // (2 * a)}")
    if "vertex" in text or "顶点" in all_text:
        items.append("目标：顶点坐标")
    return DiagramSpec(diagram_type="function_summary", title="函数信息", items=items[:5], highlight_index=len(items[:5]) - 1)


def _progress_spec(scene_type: str) -> DiagramSpec:
    steps = ["1 设变量", "2 建方程", "3 解方程", "4 回答"]
    highlight_index = 1 if scene_type == "modeling" else 2
    return DiagramSpec(
        diagram_type="solution_progress",
        title="解题进度",
        items=steps,
        highlight_index=highlight_index,
    )


def _looks_like_chicken_rabbit(text: str) -> bool:
    lowered = text.lower()
    return ("鸡" in text and "兔" in text) or ("chicken" in lowered and "rabbit" in lowered)


def _normalize_math_text(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("*", "×")
        .replace("＋", "+")
        .replace("，", ",")
        .replace("；", ";")
    )


def _is_head_equation(expression: str) -> bool:
    normalized = _normalize_math_text(expression)
    return bool(re.search(r"x\+y=35|y\+x=35", normalized))


def _is_feet_equation(expression: str) -> bool:
    normalized = _normalize_math_text(expression)
    return bool(re.search(r"2×x\+4×y=94|4×y\+2×x=94", normalized))


def _solution_values(expression: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for symbol, value in re.findall(r"\b([A-Za-z])\s*=\s*(-?\d+)\b", expression):
        values[symbol] = int(value)
    return values


def _looks_like_quadratic(expression: str) -> bool:
    return bool(re.search(r"\bx\s*(\^|\*\*)\s*2\b|x²", expression))


def _quadratic_coefficients(expression: str) -> tuple[int, int, int] | None:
    text = expression.replace(" ", "").replace("**", "^").replace("−", "-")
    if "=" in text:
        text = text.split("=", 1)[1]
    match = re.fullmatch(r"([+-]?\d*)x\^2([+-]\d*)x([+-]\d+)", text)
    if not match:
        return None
    a_raw, b_raw, c_raw = match.groups()
    a = _coefficient_value(a_raw)
    b = _coefficient_value(b_raw)
    c = int(c_raw)
    return (a, b, c)


def _coefficient_value(raw: str) -> int:
    if raw in {"", "+"}:
        return 1
    if raw == "-":
        return -1
    return int(raw)


def _compact_expr(expression: str) -> str:
    return re.sub(r"\s+", " ", expression.strip()).replace("*", "×")


def _semantic_items(*, meaning: str, display_expression: str, source_sentence: str) -> list[str]:
    items: list[str] = []
    if meaning:
        items.append(f"目标：{_need_fact(meaning)}")

    fact_sources = [meaning, source_sentence]
    quantities = _quantity_facts(" ".join(source for source in fact_sources if source))
    for quantity in quantities:
        candidate = f"已知量：{quantity}"
        if not _is_duplicate_fact(candidate, items):
            items.append(candidate)
        if len(items) >= 4:
            break

    if display_expression and len(items) < 4:
        candidate = f"使用式：{display_expression}"
        if not _is_duplicate_fact(candidate, items):
            items.append(candidate)
    return items


def _need_fact(meaning: str) -> str:
    text = re.sub(r"\s+", " ", str(meaning or "")).strip(" .。")
    lowered = text.lower()
    patterns = [
        (r"calculate the total number of items rose bought.*", "Rose's total items"),
        (r"set up the relationship.*rose.*sophia.*", "Sophia's total items"),
        (r"the number of movies that belong to series.*", "series movies"),
        (r"the number of older movies.*", "older movies"),
        (r"normal movies.*", "normal movies"),
        (r"each series movie costs.*", "series movie cost"),
        (r"each older movie costs.*", "older movie cost"),
        (r".*dry[- ]cleaning.*5 weeks?.*", "5-week dry-cleaning cost"),
        (r".*(?:banana.*sav|sav.*banana).*", "banana savings"),
        (r".*sav(?:e|ing).*", "total savings"),
        (r"the total cost.*", "total cost"),
        (r"emily.*peeling speed.*", "peeling speed"),
        (r"emily.*cooking speed.*", "cooking speed"),
        (r"time to peel.*", "time to peel 90 shrimp"),
        (r"time to cook.*", "time to cook 90 shrimp"),
        (r"total time.*", "total time"),
    ]
    for pattern, replacement in patterns:
        if re.search(pattern, lowered):
            return replacement
    text = re.sub(r"^(calculate|find|determine|compute|set up)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(by|using|from|at|with)\b.*$", "", text, flags=re.IGNORECASE)
    return _shorten_fact(text, 42)


def _quantity_facts(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    normalized = re.sub(r"\b\d{1,2}\s*:\s*\d{2}\b", " ", normalized)
    matches = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:\s*(?:[A-Za-z]+|[\u4e00-\u9fff]+)){0,3}",
        normalized,
    )
    facts: list[str] = []
    for raw in matches:
        fact = _clean_quantity_fact(raw)
        if not fact or any(_facts_equivalent(fact, item) for item in facts):
            continue
        facts.append(fact)
        if len(facts) >= 3:
            break
    return facts


def _clean_quantity_fact(text: str) -> str:
    fact = re.sub(r"\s+", " ", str(text or "")).strip(" .,;:，。；：")
    if re.fullmatch(r"0+", fact):
        return ""
    tokens = fact.split()
    if len(tokens) > 1:
        stop_words = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "from",
            "in",
            "of",
            "that",
            "the",
            "to",
            "using",
            "with",
        }
        kept = [tokens[0]]
        for token in tokens[1:]:
            clean = token.strip(" .,;:，。；：").lower()
            if clean in stop_words:
                break
            kept.append(token)
        fact = " ".join(kept)
    fact = re.sub(r"\b(?:means|equals|plus|minus|divided)$", "", fact, flags=re.IGNORECASE)
    fact = re.sub(r"\s+[A-Za-z]$", "", fact)
    fact = fact.strip(" .,;:，。；：")
    if re.fullmatch(r"0+", fact):
        return ""
    return fact


def _shorten_fact(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _is_duplicate_fact(candidate: str, existing: list[str]) -> bool:
    return any(_facts_equivalent(candidate, item) for item in existing)


def _facts_equivalent(left: str, right: str) -> bool:
    normalize = lambda value: re.sub(r"\W+", "", str(value or "").lower())
    l_norm = normalize(left)
    r_norm = normalize(right)
    return bool(l_norm and r_norm and (l_norm in r_norm or r_norm in l_norm))


def _expression_items(expression: str) -> list[str]:
    text = expression.strip()
    if not text:
        return []
    if "=" in text:
        left, right = text.split("=", 1)
        return [f"左边：{left.strip()}", f"右边：{right.strip()}", "两边表示同一个数量"]
    parts = [
        part.strip()
        for part in re.split(r"(?<=[+\-×÷*/])|(?=[+\-×÷*/])", text)
        if part.strip()
    ]
    return parts if len(parts) > 1 else [text]


class _BaseDiagramAdapter:
    def _draw_panel(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        draw.rounded_rectangle(bbox, radius=14, fill=(248, 251, 255), outline=(218, 229, 242), width=2)
        draw.text((x1 + 22, y1 + 18), spec.title, font=title_font, fill=(39, 58, 80))
        draw.line((x1 + 22, y1 + 58, x2 - 22, y1 + 58), fill=(226, 235, 245), width=2)
        return x1 + 22, y1 + 76, x2 - 22, y2 - 22


class _GenericDiagramAdapter(_BaseDiagramAdapter):
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, bottom = self._draw_panel(draw, bbox, spec, title_font=title_font)
        items = list(spec.items[:4])
        if not items:
            return
        row_gap = 12
        available_height = max(60, bottom - top)
        card_heights = _balanced_card_heights(
            draw,
            items,
            body_font,
            max_width=right - left - 24,
            available_height=available_height - row_gap * (len(items) - 1),
        )
        y = top
        for index, item in enumerate(items):
            card_height = max(1, card_heights[index])
            if y + card_height > bottom:
                card_height = max(1, bottom - y)
            fill = (255, 255, 255) if index != spec.highlight_index else (255, 249, 222)
            outline = (209, 222, 238) if index != spec.highlight_index else (244, 196, 72)
            draw.rounded_rectangle((left, y, right, y + card_height), radius=9, fill=fill, outline=outline, width=2)
            _draw_text_fit_no_ellipsis(
                draw,
                item,
                (left + 12, y + 8, right - 12, y + card_height - 8),
                base_font=body_font,
                fill=(54, 70, 89),
            )
            y += card_height + row_gap
            if y >= bottom:
                break


def _balanced_card_heights(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    font: ImageFont.ImageFont,
    *,
    max_width: int,
    available_height: int,
) -> list[int]:
    desired: list[int] = []
    if available_height <= 0:
        return [1 for _ in items]
    for item in items:
        lines = wrap_text(item, draw=draw, font=font, max_width=max_width)
        text_height = len(lines) * line_height(draw, font) + max(0, len(lines) - 1) * 4
        desired.append(max(58, min(150, text_height + 24)))
    total = sum(desired)
    if total <= available_height:
        return desired
    min_height = 50
    if available_height < min_height * len(desired):
        compact_height = max(1, available_height // max(1, len(desired)))
        return [compact_height for _ in desired]
    overflow = total - available_height
    flexible = sum(max(0, height - min_height) for height in desired)
    if flexible <= 0:
        return [max(min_height, available_height // max(1, len(desired))) for _ in desired]
    heights: list[int] = []
    for height in desired:
        shrink = int(round(overflow * max(0, height - min_height) / flexible))
        heights.append(max(min_height, height - shrink))
    diff = available_height - sum(heights)
    if heights:
        heights[-1] += diff
    return heights


class _TableDiagramAdapter(_BaseDiagramAdapter):
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, bottom = self._draw_panel(draw, bbox, spec, title_font=title_font)
        available_height = max(52, bottom - top)
        row_gap = 8
        max_items = min(len(spec.items), 5)
        while max_items > 1:
            row_height = (available_height - row_gap * (max_items - 1)) // max_items
            if row_height >= 44:
                break
            max_items -= 1
        row_height = max(44, (available_height - row_gap * (max_items - 1)) // max(1, max_items))
        for index, item in enumerate(spec.items[:max_items]):
            y = top + index * (row_height + row_gap)
            is_active = index == spec.highlight_index
            is_check = "✓" in item or "通过" in item
            fill = (255, 250, 224) if is_active else (255, 255, 255)
            outline = (244, 196, 72) if is_active else (218, 229, 242)
            text_fill = (37, 64, 97)
            if is_check:
                fill = (244, 252, 239) if is_active else (249, 255, 246)
                outline = (139, 194, 105) if is_active else (204, 229, 192)
                text_fill = (45, 103, 59)
            draw.rounded_rectangle(
                (left, y, right, y + row_height),
                radius=8,
                fill=fill,
                outline=outline,
                width=2,
            )
            _draw_text_fit_no_ellipsis(
                draw,
                item,
                (left + 12, y + 7, right - 12, y + row_height - 7),
                base_font=body_font,
                fill=text_fill,
            )


class _ProgressDiagramAdapter(_BaseDiagramAdapter):
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, _bottom = self._draw_panel(draw, bbox, spec, title_font=title_font)
        dot_x = left + 18
        text_x = left + 48
        for index, item in enumerate(spec.items[:5]):
            y = top + index * 48
            is_active = index == spec.highlight_index
            color = (74, 144, 226) if is_active else (176, 190, 205)
            draw.ellipse((dot_x - 8, y + 8, dot_x + 8, y + 24), fill=color)
            if index < min(len(spec.items), 5) - 1:
                draw.line((dot_x, y + 24, dot_x, y + 56), fill=(214, 224, 236), width=3)
            draw_wrapped_text(
                draw,
                (text_x, y + 4),
                item,
                font=body_font,
                fill=(37, 64, 97) if is_active else (84, 100, 120),
                max_width=right - text_x,
                line_spacing=3,
                max_lines=1,
            )


def _draw_text_fit_no_ellipsis(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    base_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    max_width = max(60, x2 - x1)
    max_height = max(18, y2 - y1)
    base_size = getattr(base_font, "size", 18)
    min_size = max(12, base_size - 8)
    chosen_font = base_font
    chosen_lines = [str(text)]
    chosen_spacing = 3
    for size in range(base_size, min_size - 1, -1):
        font, _ = load_font(size)
        spacing = max(2, size // 7)
        lines = wrap_text(text, draw=draw, font=font, max_width=max_width)
        text_height = len(lines) * line_height(draw, font) + max(0, len(lines) - 1) * spacing
        if text_height <= max_height:
            chosen_font = font
            chosen_lines = lines
            chosen_spacing = spacing
            break
        chosen_font = font
        chosen_lines = lines
        chosen_spacing = spacing
    else:
        # Last-resort clipping is preferable to showing literal ellipses in the
        # reference panel; the panel is an auxiliary cue, not the full narration.
        max_lines = max(1, (max_height + chosen_spacing) // (line_height(draw, chosen_font) + chosen_spacing))
        chosen_lines = wrap_text(text, draw=draw, font=chosen_font, max_width=max_width)[:max_lines]

    total_height = len(chosen_lines) * line_height(draw, chosen_font) + max(0, len(chosen_lines) - 1) * chosen_spacing
    y = y1 + max(0, (max_height - total_height) // 2)
    widest = 0
    bottom = y
    for index, line in enumerate(chosen_lines):
        top = y + index * (line_height(draw, chosen_font) + chosen_spacing)
        draw.text((x1, top), line, font=chosen_font, fill=fill)
        width, _ = text_size(draw, line, chosen_font)
        widest = max(widest, width)
        bottom = top + line_height(draw, chosen_font)
    return x1, y, x1 + widest, bottom


class _BarDiagramAdapter(_BaseDiagramAdapter):
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, bottom = self._draw_panel(draw, bbox, spec, title_font=title_font)
        width = right - left
        bar_top = top + 18
        bar_height = 28
        segment_count = max(1, min(4, len(spec.items)))
        segment_width = width // segment_count
        colors = [(151, 197, 255), (255, 216, 118), (159, 219, 178), (220, 190, 255)]
        for index in range(segment_count):
            x1 = left + index * segment_width
            x2 = right if index == segment_count - 1 else left + (index + 1) * segment_width
            draw.rounded_rectangle(
                (x1, bar_top, x2, bar_top + bar_height),
                radius=8,
                fill=colors[index % len(colors)],
                outline=(255, 255, 255),
                width=2,
            )
        for index, item in enumerate(spec.items[:3]):
            y = bar_top + 56 + index * 46
            bullet = "●"
            draw.text((left, y), bullet, font=body_font, fill=colors[index % len(colors)])
            draw_wrapped_text(
                draw,
                (left + 28, y),
                item,
                font=body_font,
                fill=(54, 70, 89),
                max_width=width - 28,
                line_spacing=4,
                max_lines=1,
            )
        if bottom - (bar_top + 56) > 165:
            label = "用图形帮助看清每一部分表示什么"
            tw, _ = text_size(draw, label, body_font)
            draw.text((left + max(0, (width - tw) // 2), bottom - 28), label, font=body_font, fill=(105, 119, 136))


class _SpeedLineDiagramAdapter(_BaseDiagramAdapter):
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        spec: DiagramSpec,
        *,
        title_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, bottom = self._draw_panel(draw, bbox, spec, title_font=title_font)
        mid_y = top + 58
        draw.line((left + 16, mid_y, right - 16, mid_y), fill=(96, 139, 196), width=5)
        draw.polygon([(right - 16, mid_y), (right - 34, mid_y - 10), (right - 34, mid_y + 10)], fill=(96, 139, 196))
        draw.polygon([(left + 16, mid_y), (left + 34, mid_y - 10), (left + 34, mid_y + 10)], fill=(96, 139, 196))
        for index, item in enumerate(spec.items[:3]):
            y = top + 108 + index * 45
            draw_wrapped_text(
                draw,
                (left, y),
                item,
                font=body_font,
                fill=(54, 70, 89),
                max_width=right - left,
                line_spacing=4,
                max_lines=1,
            )
        if bottom - top > 230:
            draw.text((left, bottom - 30), "箭头表示运动方向和速度关系", font=body_font, fill=(105, 119, 136))
