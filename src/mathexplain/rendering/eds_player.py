"""Frame renderer for EDS playback."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mathexplain.rendering.formula_renderer import (
    draw_wrapped_text,
    line_height,
    load_font,
    text_size,
    wrap_text,
)
from mathexplain.rendering.plain_text import normalize_plain_math_text
from mathexplain.rendering.visual_diagrams import DiagramRenderer, infer_diagram_spec
from mathexplain.schemas.eds import ExecutableDirectorScript


@dataclass(frozen=True)
class _FormulaBox:
    formula_id: str
    expression: str
    bbox: tuple[int, int, int, int]
    display_mode: str


class EDSFrameRenderer:
    """Render individual video frames from EDS timing state."""

    def __init__(self, eds: ExecutableDirectorScript) -> None:
        self.eds = eds
        self.width = eds.render_hints.canvas_width
        self.height = eds.render_hints.canvas_height
        self.assets_by_id = {asset.asset_id: asset for asset in eds.assets}
        self.scenes_by_id = {scene.scene_id: scene for scene in eds.scenes}
        self.formulas_by_scene: dict[str, list] = {}
        for formula in eds.formula_tracks:
            self.formulas_by_scene.setdefault(formula.scene_id, []).append(formula)
        self.actions_by_scene: dict[str, list] = {}
        for action in eds.visual_actions:
            self.actions_by_scene.setdefault(action.scene_id, []).append(action)
        self.narration_by_scene = {track.scene_id: track for track in eds.narration_tracks}
        self.diagram_renderer = DiagramRenderer()
        self.warnings: list[str] = []
        self.language = eds.render_hints.language
        self._collect_quality_warnings()

        self.title_font, title_path = load_font(max(18, self.width // 42), bold=True)
        self.body_font, body_path = load_font(max(16, self.width // 64))
        self.small_font, small_path = load_font(max(14, self.width // 76))
        self.formula_font, formula_path = load_font(max(20, self.width // 42), bold=True)
        self.subtitle_font, subtitle_path = load_font(max(16, self.width // 58), bold=True)
        if not any([title_path, body_path, small_path, formula_path, subtitle_path]):
            self.warnings.append("No system TrueType font found; using Pillow default font.")

    def _collect_quality_warnings(self) -> None:
        scene_types = {scene.scene_id: scene.scene_type for scene in self.eds.scenes}
        for formula in self.eds.formula_tracks:
            if scene_types.get(formula.scene_id) not in {"modeling", "calculation"}:
                continue
            metadata = getattr(formula, "metadata", {}) or {}
            if not metadata.get("source_sentence"):
                self.warnings.append(f"{formula.formula_id} is missing source_sentence; using verified-step fallback.")
            if not metadata.get("meaning"):
                self.warnings.append(f"{formula.formula_id} is missing meaning; source-to-equation panel may be less specific.")

    def render_frame(self, time_ms: int) -> np.ndarray:
        scene = self._scene_at(time_ms)
        image = Image.new("RGB", (self.width, self.height), (247, 250, 252))
        draw = ImageDraw.Draw(image)
        self._draw_background(draw)
        if scene is None:
            self._draw_center_text(draw, "No scene", fill=(128, 128, 128))
            return np.asarray(image, dtype=np.uint8)

        self._draw_header(draw, scene, time_ms)
        if self._is_plain_simple_theme():
            self._draw_plain_simple_slide(draw, scene)
            self._draw_subtitle(draw, scene.scene_id)
            self._draw_progress(draw, time_ms)
            return np.asarray(image, dtype=np.uint8)

        visible_formulas = self._visible_formulas(scene.scene_id, time_ms)
        if scene.scene_type == "answer":
            answer_text = self._draw_answer_card(draw, scene, visible_formulas, time_ms)
            self._draw_highlights(draw, scene.scene_id, time_ms, [])
            self._draw_subtitle(draw, scene.scene_id, text_override=answer_text)
            self._draw_progress(draw, time_ms)
            return np.asarray(image, dtype=np.uint8)
        if scene.scene_type == "recap":
            self._draw_recap_slide(draw, scene)
            self._draw_progress(draw, time_ms)
            return np.asarray(image, dtype=np.uint8)

        self._draw_problem_context(draw, scene)
        self._draw_teaching_hint(draw, visible_formulas)
        self._draw_relation_panel(draw, scene, visible_formulas)
        formula_boxes = self._draw_formulas(draw, visible_formulas, time_ms)
        self._draw_highlights(draw, scene.scene_id, time_ms, formula_boxes)
        self._draw_subtitle(draw, scene.scene_id)
        self._draw_progress(draw, time_ms)
        return np.asarray(image, dtype=np.uint8)

    def _scene_at(self, time_ms: int):
        for scene in self.eds.scenes:
            if scene.start_ms <= time_ms < scene.start_ms + scene.duration_ms:
                return scene
        if self.eds.scenes and time_ms >= self.eds.scenes[-1].start_ms:
            return self.eds.scenes[-1]
        return self.eds.scenes[0] if self.eds.scenes else None

    def _draw_background(self, draw: ImageDraw.ImageDraw) -> None:
        margin = max(24, self.width // 60)
        draw.rounded_rectangle(
            (margin, margin, self.width - margin, self.height - margin),
            radius=18,
            fill=(255, 255, 255),
            outline=(219, 228, 238),
            width=2,
        )

    def _draw_header(self, draw: ImageDraw.ImageDraw, scene, time_ms: int) -> None:
        label = scene.title or self._scene_type_label(scene.scene_type, self.language)
        x = max(44, self.width // 34)
        y = max(34, self.height // 32)
        draw.text((x, y), label, font=self.title_font, fill=(29, 44, 62))
        scene_index = self.eds.scenes.index(scene) + 1
        counter = f"{scene_index}/{len(self.eds.scenes)}"
        counter_width, _ = text_size(draw, counter, self.body_font)
        draw.text((self.width - x - counter_width, y + 6), counter, font=self.body_font, fill=(85, 100, 118))
        draw.line((x, y + 58, self.width - x, y + 58), fill=(229, 236, 244), width=2)

    @staticmethod
    def _scene_type_label(scene_type: str, language: str = "zh-CN") -> str:
        if str(language or "").lower().startswith("en"):
            labels = {
                "problem_overview": "Understand",
                "variable_definition": "Define Variables",
                "modeling": "Set Up",
                "calculation": "Solve",
                "answer": "Final Answer",
                "recap": "Recap",
                "unsupported": "Not Supported",
            }
            return labels.get(scene_type, scene_type)
        labels = {
            "problem_overview": "读题",
            "variable_definition": "定义变量",
            "modeling": "建立关系",
            "calculation": "计算",
            "answer": "回答问题",
            "recap": "回顾",
            "unsupported": "暂不支持",
        }
        return labels.get(scene_type, scene_type)

    def _draw_problem_context(self, draw: ImageDraw.ImageDraw, scene) -> None:
        if scene.scene_type not in {"problem_overview", "variable_definition"}:
            return
        problem_text = self._problem_text()
        if not problem_text:
            return
        x = max(56, self.width // 28)
        y = max(112, self.height // 9)
        max_width = int(self.width * 0.86)
        draw_wrapped_text(
            draw,
            (x, y),
            problem_text,
            font=self.body_font,
            fill=(56, 68, 82),
            max_width=max_width,
            line_spacing=14,
            max_lines=7,
        )

    def _draw_plain_simple_slide(self, draw: ImageDraw.ImageDraw, scene) -> None:
        metadata = getattr(scene, "metadata", {}) or {}
        display_text = normalize_plain_math_text(metadata.get("plain_display_text") or metadata.get("plain_formula_text") or "")
        if not display_text:
            display_text = self._problem_text() or ""
        formula_text = normalize_plain_math_text(metadata.get("plain_formula_text") or "")

        left = int(self.width * 0.12)
        right = int(self.width * 0.88)
        top = int(self.height * 0.19)
        bottom = int(self.height * 0.66)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=10,
            fill=(255, 255, 255),
            outline=(222, 230, 240),
            width=2,
        )

        content = display_text
        if formula_text and formula_text != display_text:
            content = f"{display_text}\n\n{formula_text}"
        draw_wrapped_text(
            draw,
            (left + 36, top + 34),
            content,
            font=self.formula_font if len(content) < 180 else self.body_font,
            fill=(31, 42, 56),
            max_width=right - left - 72,
            line_spacing=14,
            max_lines=8,
        )

    def _visible_formulas(self, scene_id: str, time_ms: int) -> list:
        return [
            formula for formula in self.formulas_by_scene.get(scene_id, [])
            if time_ms >= formula.start_ms
        ]

    def _draw_teaching_hint(self, draw: ImageDraw.ImageDraw, formulas: list) -> None:
        if not formulas:
            return
        meaning, source_sentence = self._formula_context_parts(formulas[-1])
        if not meaning and not source_sentence:
            return

        left, top, right, bottom, limits = self._teaching_hint_layout(draw, formulas)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=14,
            fill=(246, 250, 255),
            outline=(211, 224, 240),
            width=2,
        )
        y = top + 18
        step_label = "This step: " if self._is_english() else "杩欎竴姝ワ細"
        source_label = "From the problem: " if self._is_english() else "鍘熼骞蹭緷鎹細"
        if self._is_english():
            if meaning:
                y = self._draw_labeled_text(
                    draw,
                    step_label,
                    str(meaning),
                    left + 22,
                    y,
                    right - left - 44,
                    max_lines=limits["meaning"],
                ) + 8
            if source_sentence:
                self._draw_labeled_text(
                    draw,
                    source_label,
                    str(source_sentence),
                    left + 22,
                    y,
                    right - left - 44,
                    max_lines=limits["source"],
                )
            return
        if meaning:
            y = self._draw_labeled_text(
                draw,
                "这一步：",
                str(meaning),
                left + 22,
                y,
                right - left - 44,
                max_lines=limits["meaning"],
            ) + 8
        if source_sentence:
            self._draw_labeled_text(
                draw,
                "原题干依据：",
                str(source_sentence),
                left + 22,
                y,
                right - left - 44,
                max_lines=limits["source"],
            )

    def _teaching_hint_layout(
        self,
        draw: ImageDraw.ImageDraw,
        formulas: list,
    ) -> tuple[int, int, int, int, dict[str, int]]:
        left = int(self.width * 0.07)
        right = int(self.width * 0.61)
        top = int(self.height * 0.17)
        min_bottom = int(self.height * 0.31)
        max_bottom = int(self.height * (0.39 if self._is_english() else 0.36))
        meaning, source_sentence = self._formula_context_parts(formulas[-1]) if formulas else (None, None)
        limits = {"meaning": 4 if self._is_english() else 3, "source": 3 if self._is_english() else 2}
        content_width = right - left - 44
        total_height = 36
        if meaning:
            total_height += self._labeled_text_height(
                draw,
                "This step: " if self._is_english() else "这一步：",
                str(meaning),
                content_width,
                max_lines=limits["meaning"],
            )
            if source_sentence:
                total_height += 8
        if source_sentence:
            total_height += self._labeled_text_height(
                draw,
                "From the problem: " if self._is_english() else "原题干依据：",
                str(source_sentence),
                content_width,
                max_lines=limits["source"],
            )
        bottom = min(max_bottom, max(min_bottom, top + total_height))
        return left, top, right, bottom, limits

    def _labeled_text_height(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        text: str,
        max_width: int,
        *,
        max_lines: int,
    ) -> int:
        label_width, _ = text_size(draw, label, self.body_font)
        lines = wrap_text(
            text,
            draw=draw,
            font=self.body_font,
            max_width=max(80, max_width - label_width),
            max_lines=max_lines,
        )
        return len(lines) * line_height(draw, self.body_font) + max(0, len(lines) - 1) * 5

    def _draw_labeled_text(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        text: str,
        x: int,
        y: int,
        max_width: int,
        *,
        max_lines: int,
    ) -> int:
        label_width, _ = text_size(draw, label, self.body_font)
        draw.text((x, y), label, font=self.body_font, fill=(37, 64, 97))
        bbox = draw_wrapped_text(
            draw,
            (x + label_width, y),
            text,
            font=self.body_font,
            fill=(59, 75, 95),
            max_width=max(80, max_width - label_width),
            line_spacing=5,
            max_lines=max_lines,
        )
        return bbox[3]

    def _draw_relation_panel(self, draw: ImageDraw.ImageDraw, scene, formulas: list) -> None:
        if scene.scene_type not in {"variable_definition", "modeling", "calculation", "answer"}:
            return
        if not formulas and scene.scene_type not in {"variable_definition", "answer"}:
            return
        left = int(self.width * 0.64)
        right = int(self.width * 0.91)
        top = int(self.height * 0.17)
        bottom = int(self.height * 0.72)
        spec = infer_diagram_spec(
            scene,
            formulas,
            context={
                "all_formulas": list(self.eds.formula_tracks),
                "all_scenes": list(self.eds.scenes),
                "language": self.language,
            },
        )
        if not spec.items:
            return
        self.diagram_renderer.draw(
            draw,
            (left, top, right, bottom),
            spec,
            title_font=self.body_font,
            body_font=self.small_font,
        )

    def _draw_answer_card(self, draw: ImageDraw.ImageDraw, scene, formulas: list, time_ms: int) -> str:
        expression = self._formula_display_text(formulas[-1], time_ms) if formulas else ""
        if not expression:
            track = self.narration_by_scene.get(scene.scene_id)
            expression = str(track.text if track is not None else "").strip()
        display_expression = self._friendly_answer_expression(expression)
        label = "Verified Final Answer" if self._is_english() else "已验证最终答案"
        verification = self._answer_verification_line(expression, formulas[-1] if formulas else None)
        note = "Verification" if self._is_english() else "验证依据"
        left = int(self.width * 0.16)
        right = int(self.width * 0.84)
        top = int(self.height * 0.22)
        bottom = int(self.height * 0.66)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=22,
            fill=(248, 255, 243),
            outline=(112, 174, 88),
            width=4,
        )
        badge_w, badge_h = text_size(draw, label, self.body_font)
        badge = (left + 34, top + 28, left + 66 + badge_w, top + 62 + badge_h)
        draw.rounded_rectangle(badge, radius=12, fill=(221, 246, 212), outline=(126, 184, 96), width=2)
        draw.text((badge[0] + 16, badge[1] + 10), label, font=self.body_font, fill=(40, 92, 50))
        self._draw_fitted_text(
            draw,
            display_expression,
            (left + 58, top + 104, right - 58, bottom - 92),
            base_font=self.formula_font,
            fill=(24, 44, 32),
            bold=True,
            min_size=max(26, self.width // 46),
            line_spacing=max(8, self.height // 120),
        )
        verification_text = verification or (
            "matches the verified solution chain"
            if self._is_english()
            else "与验证通过的解题链一致"
        )
        note_text = f"{note}: {verification_text}"
        verification_font, _ = load_font(max(18, self.width // 58), bold=True)
        note_w, _ = text_size(draw, note_text, verification_font)
        if note_w <= right - left - 80:
            draw.text(((self.width - note_w) // 2, bottom - 66), note_text, font=verification_font, fill=(73, 101, 78))
        else:
            draw_wrapped_text(
                draw,
                (left + 40, bottom - 88),
                note_text,
                font=verification_font,
                fill=(73, 101, 78),
                max_width=right - left - 80,
                line_spacing=6,
                max_lines=2,
            )
        return display_expression

    def _draw_recap_slide(self, draw: ImageDraw.ImageDraw, scene) -> None:
        title = "Wrap-up" if self._is_english() else "收束"
        line = (
            "The answer above is the final checked result."
            if self._is_english()
            else "上一页给出的就是验证后的最终结果。"
        )
        left = int(self.width * 0.25)
        right = int(self.width * 0.75)
        top = int(self.height * 0.28)
        bottom = int(self.height * 0.58)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=18,
            fill=(250, 253, 255),
            outline=(202, 218, 236),
            width=3,
        )
        title_w, _ = text_size(draw, title, self.title_font)
        draw.text(((self.width - title_w) // 2, top + 42), title, font=self.title_font, fill=(29, 44, 62))
        draw_wrapped_text(
            draw,
            (left + 42, top + 126),
            line,
            font=self.body_font,
            fill=(43, 60, 78),
            max_width=right - left - 84,
            line_spacing=8,
            max_lines=2,
        )

    def _draw_formulas(self, draw: ImageDraw.ImageDraw, formulas: list, time_ms: int) -> list[_FormulaBox]:
        if not formulas:
            return []

        boxes: list[_FormulaBox] = []
        area_left = int(self.width * 0.07)
        area_right = int(self.width * 0.61)
        hint_bottom = self._teaching_hint_layout(draw, formulas)[3] if formulas else int(self.height * 0.31)
        start_y = max(int(self.height * 0.34), hint_bottom + max(22, self.height // 36))
        row_gap = max(18, self.height // 45)
        area_bottom = int(self.height * 0.72)
        available_height = max(96, area_bottom - start_y)
        max_rows = 3
        visible = formulas[-min(3, max_rows):]
        formula_height = max(
            88,
            min(
                int(self.height * 0.22),
                (available_height - row_gap * (len(visible) - 1)) // max(1, len(visible)),
            ),
        )
        for index, formula in enumerate(visible):
            y = start_y + index * (formula_height + row_gap)
            fill = (255, 255, 255)
            outline = (195, 210, 230)
            if formula.display_mode == "final_answer":
                fill = (248, 255, 240)
                outline = (120, 184, 91)
            bbox = (area_left, y, area_right, y + formula_height)
            draw.rounded_rectangle(bbox, radius=12, fill=fill, outline=outline, width=3)
            text_bbox = self._draw_fitted_text(
                draw,
                self._formula_display_text(formula, time_ms),
                (area_left + 28, y + 16, area_right - 28, y + formula_height - 16),
                base_font=self.formula_font,
                fill=(28, 38, 52),
                bold=True,
                min_size=max(17, self.width // 72),
                line_spacing=max(5, self.height // 150),
            )
            boxes.append(_FormulaBox(formula.formula_id, formula.expression, bbox, formula.display_mode))
            if formula.display_mode == "final_answer":
                draw.rectangle(
                    (text_bbox[0] - 8, text_bbox[1] - 4, text_bbox[2] + 8, text_bbox[3] + 4),
                    outline=(107, 160, 70),
                    width=2,
                )
        return boxes

    @staticmethod
    def _formula_display_text(formula, time_ms: int) -> str:
        metadata = getattr(formula, "metadata", {}) or {}
        parts = [str(part).strip() for part in metadata.get("buildup_parts") or [] if str(part).strip()]
        expression = str(getattr(formula, "expression", "") or "")
        if metadata.get("buildup_mode") != "sequential_terms" or len(parts) < 3:
            return expression
        if not EDSFrameRenderer._valid_buildup_parts(expression, parts):
            return expression
        elapsed = max(0, time_ms - int(getattr(formula, "start_ms", 0) or 0))
        duration = int(getattr(formula, "duration_ms", 0) or 0)
        interval = max(120, int(metadata.get("buildup_interval_ms") or 450))
        if duration > 1200 and elapsed >= duration - 1000:
            return expression
        visible_count = min(len(parts), max(1, elapsed // interval + 1))
        return parts[visible_count - 1]

    @staticmethod
    def _valid_buildup_parts(expression: str, parts: list[str]) -> bool:
        compact_expression = re.sub(r"\s+", "", str(expression or ""))
        if not compact_expression:
            return False
        previous = ""
        for part in parts:
            compact_part = re.sub(r"\s+", "", part)
            if not compact_part or compact_part not in compact_expression:
                return False
            if previous and not compact_part.startswith(previous):
                return False
            previous = compact_part
        return previous == compact_expression

    def _answer_verification_line(self, answer_expression: str, answer_formula: Any | None = None) -> str | None:
        metadata = getattr(answer_formula, "metadata", {}) or {}
        explicit = str(metadata.get("verification_expression") or "").strip()
        if explicit and explicit != str(answer_expression or "").strip():
            return explicit
        answer_value = self._last_number(answer_expression)
        if not answer_value:
            return None
        answer_scene_ids = {scene.scene_id for scene in self.eds.scenes if scene.scene_type == "answer"}
        prior = [
            formula for formula in self.eds.formula_tracks
            if formula.scene_id not in answer_scene_ids and self._last_number(getattr(formula, "expression", ""))
        ]
        if not prior:
            return None
        for formula in reversed(prior):
            expression = str(getattr(formula, "expression", "") or "")
            if self._last_number(expression) == answer_value:
                return expression if "=" in expression else f"{expression} = {answer_value}"
        return None

    def _friendly_answer_expression(self, expression: str) -> str:
        text = re.sub(r"\s+", " ", str(expression or "")).strip()
        value = self._last_number(text)
        if not value:
            return text
        generic_variable = re.fullmatch(r"[A-Za-z]\d*\s*=\s*[-+]?\d+(?:\.\d+)?", text)
        inferred_unit = self._infer_answer_unit()
        if generic_variable and inferred_unit:
            return f"{value} {self._pluralize_unit(inferred_unit, value)}"
        text = re.sub(r"\b1\s+boxes\b", "1 box", text)
        if value != "1":
            text = re.sub(rf"=\s*{re.escape(value)}\s+box\b", f"= {value} boxes", text)
            text = re.sub(rf"\bis\s+{re.escape(value)}\s+box\b", f"is {value} boxes", text)
        return text

    def _infer_answer_unit(self) -> str | None:
        problem = (self._problem_text() or "").lower()
        if "$" in (self._problem_text() or "") or any(token in problem for token in ["cost", "price", "spent", "paid"]):
            return "dollars"
        if ("how long" in problem or "time" in problem) and ("minute" in problem or "min" in problem):
            return "minutes"
        if ("how long" in problem or "time" in problem) and "hour" in problem:
            return "hours"
        if "how many more boxes" in problem or "boxes of" in problem:
            return "boxes"
        return None

    @staticmethod
    def _pluralize_unit(unit: str, value: str) -> str:
        if value == "1":
            return unit[:-1] if unit.endswith("s") else unit
        if unit.endswith("s"):
            return unit
        return f"{unit}s"

    @staticmethod
    def _last_number(text: str) -> str | None:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
        if not matches:
            return None
        value = matches[-1]
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        return value

    def _draw_fitted_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        bbox: tuple[int, int, int, int],
        *,
        base_font,
        fill: tuple[int, int, int],
        bold: bool = False,
        min_size: int = 16,
        line_spacing: int = 8,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        max_width = max(80, x2 - x1)
        max_height = max(24, y2 - y1)
        base_size = getattr(base_font, "size", max(20, self.width // 42))
        chosen_font = base_font
        chosen_lines: list[str] = [str(text)]
        chosen_spacing = line_spacing
        for size in range(base_size, min_size - 1, -2):
            font, _ = load_font(size, bold=bold)
            spacing = max(3, min(line_spacing, size // 4))
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

    @staticmethod
    def _formula_context_text(formula) -> str | None:
        meaning, source_sentence = EDSFrameRenderer._formula_context_parts(formula)
        if meaning and source_sentence:
            return f"这一步：{meaning}｜原题干依据：{source_sentence}"
        if meaning:
            return f"这一步：{meaning}"
        if source_sentence:
            return f"原题干依据：{source_sentence}"
        return None

    def _is_english(self) -> bool:
        return str(self.language or "").lower().startswith("en")

    def _is_plain_simple_theme(self) -> bool:
        return self.eds.render_hints.theme == "plain_simple_baseline"

    @staticmethod
    def _formula_context_parts(formula) -> tuple[str | None, str | None]:
        metadata = getattr(formula, "metadata", {}) or {}
        meaning = metadata.get("meaning")
        source_sentence = metadata.get("source_sentence")
        if not source_sentence and getattr(formula, "source_step_ids", None):
            source_sentence = (
                "Derived from the verified solution step"
                if str(metadata.get("language") or "").lower().startswith("en")
                else "来自已验证的解题步骤"
            )
        return (str(meaning) if meaning else None, str(source_sentence) if source_sentence else None)

    def _draw_highlights(
        self,
        draw: ImageDraw.ImageDraw,
        scene_id: str,
        time_ms: int,
        formula_boxes: list[_FormulaBox],
    ) -> None:
        boxes_by_id = {box.formula_id: box for box in formula_boxes}
        active_actions = [
            action for action in self.actions_by_scene.get(scene_id, [])
            if action.start_ms <= time_ms < action.start_ms + max(1, action.duration_ms)
            and action.action_type in {"highlight", "emphasize", "final_answer_reveal"}
        ]
        for action in active_actions:
            box = boxes_by_id.get(action.target_id)
            if box is None and formula_boxes:
                box = formula_boxes[-1]
            if box is None:
                continue
            color = (250, 196, 55) if action.action_type == "highlight" else (74, 144, 226)
            width = 6 if action.action_type == "final_answer_reveal" else 4
            expand = 10
            x1, y1, x2, y2 = box.bbox
            draw.rounded_rectangle(
                (x1 - expand, y1 - expand, x2 + expand, y2 + expand),
                radius=16,
                outline=color,
                width=width,
            )

    def _draw_subtitle(self, draw: ImageDraw.ImageDraw, scene_id: str, *, text_override: str | None = None) -> None:
        track = self.narration_by_scene.get(scene_id)
        if track is None and text_override is None:
            return
        text = str(text_override if text_override is not None else track.text)
        box_left = int(self.width * 0.08)
        box_right = int(self.width * 0.92)
        box_bottom = int(self.height * 0.93)
        font, lines, line_gap = self._subtitle_layout(draw, text, box_right - box_left - 48)
        text_height = len(lines) * line_height(draw, font) + max(0, len(lines) - 1) * line_gap
        box_height = max(110, text_height + 40)
        box_top = max(int(self.height * 0.73), box_bottom - box_height)
        draw.rounded_rectangle(
            (box_left, box_top, box_right, box_bottom),
            radius=14,
            fill=(29, 44, 62),
        )
        y = box_top + max(14, (box_bottom - box_top - text_height) // 2)
        for index, line in enumerate(lines):
            top = y + index * (line_height(draw, font) + line_gap)
            draw.text((box_left + 24, top), line, font=font, fill=(255, 255, 255))

    def _subtitle_layout(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
    ):
        max_lines = 5
        line_gap = max(6, self.height // 90)
        base_size = getattr(self.subtitle_font, "size", max(18, self.width // 58))
        for size in range(base_size, max(13, base_size - 10), -2):
            font, _ = load_font(size, bold=True)
            lines = wrap_text(text, draw=draw, font=font, max_width=max_width, max_lines=max_lines)
            available_height = int(self.height * 0.20)
            text_height = len(lines) * line_height(draw, font) + max(0, len(lines) - 1) * line_gap
            if text_height + 40 <= available_height or size <= max(14, base_size - 8):
                return font, lines, line_gap
        return self.subtitle_font, wrap_text(text, draw=draw, font=self.subtitle_font, max_width=max_width, max_lines=max_lines), line_gap

    def _draw_progress(self, draw: ImageDraw.ImageDraw, time_ms: int) -> None:
        total = max(1, self.eds.metadata.get("total_duration_ms") or self._total_duration())
        x1 = int(self.width * 0.08)
        x2 = int(self.width * 0.92)
        y = int(self.height * 0.965)
        draw.line((x1, y, x2, y), fill=(217, 226, 236), width=6)
        progress_x = x1 + int((x2 - x1) * min(1.0, max(0.0, time_ms / total)))
        draw.line((x1, y, progress_x, y), fill=(74, 144, 226), width=6)

    def _draw_center_text(self, draw: ImageDraw.ImageDraw, text: str, *, fill: tuple[int, int, int]) -> None:
        w, h = text_size(draw, text, self.title_font)
        draw.text(((self.width - w) // 2, (self.height - h) // 2), text, font=self.title_font, fill=fill)

    def _problem_text(self) -> str | None:
        for asset in self.eds.assets:
            if asset.asset_type == "problem_text":
                value = asset.metadata.get("problem_text") or asset.metadata.get("label")
                if value:
                    return str(value)
        return None

    def _total_duration(self) -> int:
        return max((scene.start_ms + scene.duration_ms for scene in self.eds.scenes), default=1)
