"""Script Director v1: deterministic ExplanationScript to EDS conversion."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from mathexplain.schemas.eds import ExecutableDirectorScript
from mathexplain.schemas.explanation import ExplanationScript, ExplanationSegment
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport


class ScriptDirector:
    """Convert a verified explanation blueprint into an executable director script."""

    _FORMULA_OFFSET_MS = 400
    _HIGHLIGHT_DELAY_MS = 400
    _QUESTION_PAUSE_DURATION_MS = 600

    _ACTION_MAP = {
        "SHOW_PROBLEM_TEXT": "show",
        "HIGHLIGHT_QUESTION": "highlight",
        "DEFINE_VARIABLE": "reveal",
        "HIGHLIGHT_TERMS": "highlight",
        "RENDER_EXPRESSION": "reveal",
        "QUESTION_PAUSE": "pause",
        "SHOW_RESULT": "reveal",
        "HIGHLIGHT_RESULT": "highlight",
        "FINAL_ANSWER_REVEAL": "final_answer_reveal",
        "RECAP_FAST": "emphasize",
        "SHOW_UNSUPPORTED_NOTICE": "show",
    }

    def direct(
        self,
        explanation: ExplanationScript,
        *,
        spr: SPR | None = None,
        scs: SolutionChain | None = None,
        verification_report: VerificationReport | None = None,
        output_language: str | None = None,
    ) -> ExecutableDirectorScript:
        """Build EDS-1.0 from an ExplanationScript without invoking models."""

        resolved_language = output_language or str(explanation.metadata.get("output_language") or "zh-CN")
        scenes: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        assets = self._assets(explanation, spr)
        narration_tracks: list[dict[str, Any]] = []
        formula_tracks: list[dict[str, Any]] = []
        visual_actions: list[dict[str, Any]] = []
        sync_anchors: list[dict[str, Any]] = []

        current_ms = 0
        errors = self._precheck_errors(explanation)
        warnings: list[str] = []

        for scene_index, segment in enumerate(explanation.segments):
            segment_safe_id = self._safe_id(segment.segment_id)
            scene_id = f"scene_{scene_index}_{segment_safe_id}"
            narration_id = f"nar_{segment_safe_id}"
            anchor_id = f"anchor_{segment_safe_id}_0"
            narration_duration = segment.estimated_duration_ms
            scene_duration = narration_duration + segment.pause_after_ms

            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_type": segment.segment_type,
                    "source_segment_ids": [segment.segment_id],
                    "title": self._scene_title(segment, resolved_language),
                    "start_ms": current_ms,
                    "duration_ms": scene_duration,
                    "layout": dict(segment.layout_hints),
                    "metadata": {
                        "source_segment_id": segment.segment_id,
                        "linked_step_ids": list(segment.linked_step_ids),
                        "visual_refs": list(segment.visual_refs),
                        "pacing": segment.pacing,
                        **(
                            {
                                "source_final_answer": scs.final_answer if scs is not None else segment.metadata.get("source_final_answer"),
                                "answer_card": True,
                                "verified": True,
                            }
                            if segment.segment_type == "answer"
                            else {}
                        ),
                        **segment.metadata,
                    },
                }
            )

            narration_text = self._sanitize_narration_text(segment.narration)
            narration_tracks.append(
                {
                    "narration_id": narration_id,
                    "scene_id": scene_id,
                    "text": narration_text,
                    "start_ms": current_ms,
                    "duration_ms": narration_duration,
                    "pause_after_ms": segment.pause_after_ms,
                    "tts_hints": dict(segment.tts_hints),
                    "sync_anchor_ids": [anchor_id],
                }
            )
            timeline.append(
                self._timeline_item(
                    "narration",
                    narration_id,
                    scene_id,
                    current_ms,
                    narration_duration,
                    "play",
                )
            )

            formula_ids: list[str] = []
            formula_start = current_ms + min(self._FORMULA_OFFSET_MS, narration_duration)
            formula_duration = max(0, narration_duration - min(self._FORMULA_OFFSET_MS, narration_duration))
            for formula_index, expression in enumerate(segment.math_expressions):
                expression = self._sanitize_narration_text(str(expression))
                formula_id = f"formula_{segment_safe_id}_{formula_index}"
                formula_ids.append(formula_id)
                formula_style = self._formula_style(segment)
                formula_tracks.append(
                    {
                        "formula_id": formula_id,
                        "scene_id": scene_id,
                        "expression": expression,
                        "display_mode": self._display_mode(segment),
                        "source_step_ids": list(segment.linked_step_ids),
                        "start_ms": formula_start,
                        "duration_ms": formula_duration,
                        "layout_slot": segment.layout_hints.get("focus_area"),
                        "style": formula_style,
                        "sync_anchor_ids": [anchor_id],
                        "metadata": self._formula_metadata(
                            segment,
                            expression,
                            scs=scs,
                            formula_style=formula_style,
                            output_language=resolved_language,
                        ),
                    }
                )
                timeline.append(
                    self._timeline_item(
                        "formula",
                        formula_id,
                        scene_id,
                        formula_start,
                        formula_duration,
                        "render",
                    )
                )

            action_ids: list[str] = []
            has_explicit_pause_action = False
            for action_index, pedagogical_action in enumerate(segment.pedagogical_actions):
                action_type = self._map_action(pedagogical_action)
                has_explicit_pause_action = has_explicit_pause_action or action_type == "pause"
                target_id = self._action_target(action_type, formula_ids, segment, narration_id)
                action_id = f"action_{segment_safe_id}_{action_index}"
                action_start, action_duration = self._action_timing(
                    action_type,
                    current_ms,
                    narration_duration,
                    segment.pause_after_ms,
                )
                action_ids.append(action_id)
                visual_actions.append(
                    {
                        "action_id": action_id,
                        "scene_id": scene_id,
                        "action_type": action_type,
                        "target_id": target_id,
                        "start_ms": action_start,
                        "duration_ms": action_duration,
                        "style": self._action_style(action_type, segment),
                        "metadata": {
                            "source_pedagogical_action": pedagogical_action,
                            "source_segment_id": segment.segment_id,
                            "linked_step_ids": list(segment.linked_step_ids),
                            "visual_refs": list(segment.visual_refs),
                        },
                    }
                )
                timeline.append(
                    self._timeline_item(
                        "pause" if action_type == "pause" else "visual",
                        action_id,
                        scene_id,
                        action_start,
                        action_duration,
                        action_type,
                    )
                )

            if segment.pause_after_ms > 0 and not has_explicit_pause_action:
                pause_id = f"action_{segment_safe_id}_pause_after"
                action_ids.append(pause_id)
                visual_actions.append(
                    {
                        "action_id": pause_id,
                        "scene_id": scene_id,
                        "action_type": "pause",
                        "target_id": narration_id,
                        "start_ms": current_ms + narration_duration,
                        "duration_ms": segment.pause_after_ms,
                        "style": {},
                        "metadata": {
                            "source_segment_id": segment.segment_id,
                            "reason": "pause_after_ms",
                        },
                    }
                )
                timeline.append(
                    self._timeline_item(
                        "pause",
                        pause_id,
                        scene_id,
                        current_ms + narration_duration,
                        segment.pause_after_ms,
                        "pause",
                    )
                )

            sync_anchors.append(
                {
                    "anchor_id": anchor_id,
                    "scene_id": scene_id,
                    "source_ref": segment.segment_id,
                    "time_ms": formula_start,
                    "linked_narration_id": narration_id,
                    "linked_formula_ids": formula_ids,
                    "linked_action_ids": action_ids,
                    "metadata": {
                        "source_segment_id": segment.segment_id,
                        "visual_refs": list(segment.visual_refs),
                    },
                }
            )
            current_ms += scene_duration

        structure_errors, structure_warnings = self._eds_structure_issues(
            scenes=scenes,
            timeline=timeline,
            assets=assets,
            narration_tracks=narration_tracks,
            formula_tracks=formula_tracks,
            visual_actions=visual_actions,
            sync_anchors=sync_anchors,
            total_duration_ms=current_ms,
        )
        errors.extend(structure_errors)
        warnings.extend(structure_warnings)
        quality_report = self._quality_report(explanation, errors, warnings, len(scenes))
        return ExecutableDirectorScript.model_validate(
            {
                "problem_id": explanation.problem_id,
                "source_explanation_id": self._source_explanation_id(explanation),
                "source_chain_id": explanation.source_chain_id,
                "scenes": scenes,
                "timeline": timeline,
                "assets": assets,
                "narration_tracks": narration_tracks,
                "formula_tracks": formula_tracks,
                "visual_actions": visual_actions,
                "sync_anchors": sync_anchors,
                "render_hints": {
                    "language": resolved_language,
                },
                "quality_report": quality_report,
                "metadata": {
                    "script_director": "ScriptDirector",
                    "script_director_version": "Script Director v1 rule-based",
                    "source_explanation_status": explanation.status,
                    "source_consistency_valid": explanation.consistency_report.valid,
                    "source_verification_status": (
                        verification_report.status if verification_report is not None else None
                    ),
                    "source_problem_type": str(spr.problem_type) if spr is not None else None,
                    "source_chain_step_count": len(scs.steps) if scs is not None else None,
                    "output_language": resolved_language,
                    "detected_source_language": explanation.metadata.get("detected_source_language"),
                    "language_policy": explanation.metadata.get("language_policy"),
                    "total_duration_ms": current_ms,
                    "does_not_render_video": True,
                },
            }
        )

    @staticmethod
    def _sanitize_narration_text(text: str) -> str:
        replacements = {
            "กฃ": ".",
            "กม": "×",
            "กย": "÷",
            "¡Á": "×",
            "¡Â": "÷",
        }
        cleaned = text
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        return cleaned

    @staticmethod
    def _source_explanation_id(explanation: ExplanationScript) -> str:
        return f"explanation_{explanation.problem_id}_{explanation.source_chain_id}"

    @staticmethod
    def _precheck_errors(explanation: ExplanationScript) -> list[str]:
        errors: list[str] = []
        if not explanation.segments:
            errors.append("ExplanationScript has no segments.")
        for segment in explanation.segments:
            if not segment.narration.strip():
                errors.append(f"{segment.segment_id} has empty narration.")
        if explanation.status == "failed" or not explanation.consistency_report.valid:
            errors.extend(explanation.consistency_report.errors or ["Source explanation did not pass consistency checks."])
        return errors

    @staticmethod
    def _quality_report(
        explanation: ExplanationScript,
        errors: list[str],
        warnings: list[str],
        checked_scene_count: int,
    ) -> dict[str, Any]:
        if explanation.status == "unsupported":
            status = "unsupported"
        elif errors:
            status = "failed"
        else:
            status = "passed"
        return {
            "valid": status == "passed",
            "status": status,
            "checked_scene_count": checked_scene_count,
            "errors": errors,
            "warnings": warnings,
            "metadata": {
                "source_explanation_status": explanation.status,
                "source_consistency_status": explanation.consistency_report.status,
            },
        }

    @staticmethod
    def _assets(explanation: ExplanationScript, spr: SPR | None) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        seen_asset_ids: set[str] = set()
        if spr is not None:
            problem_text = ScriptDirector._clean_problem_text(spr.problem_text)
            source_problem_asset = {
                "asset_id": "asset_source_problem_text",
                "asset_type": "problem_text",
                "metadata": {
                    "problem_text": problem_text,
                    "source": "spr",
                },
            }
            assets.append(source_problem_asset)
            seen_asset_ids.add(source_problem_asset["asset_id"])
        for anchor in explanation.visual_anchors:
            asset_id = f"asset_{ScriptDirector._safe_id(anchor.anchor_id)}"
            if asset_id in seen_asset_ids:
                continue
            seen_asset_ids.add(asset_id)
            assets.append(
                {
                    "asset_id": asset_id,
                    "asset_type": anchor.kind,
                    "metadata": {
                        "anchor_id": anchor.anchor_id,
                        "label": anchor.label,
                        "expression": anchor.expression,
                        "source_step_id": anchor.source_step_id,
                        **anchor.metadata,
                    },
                }
            )
        return assets

    @staticmethod
    def _clean_problem_text(text: str) -> str:
        filtered_lines: list[str] = []
        metadata_key_pattern = re.compile(
            r"^(?:sample[_\s-]*id|source[_\s-]*id|problem[_\s-]*id|dataset|split|subset)\s*:",
            re.IGNORECASE,
        )
        wrapper_titles = {
            "gsm8k math word problem",
            "math word problem",
        }
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = re.sub(r"\s+", " ", line).strip().lower()
            if normalized in wrapper_titles:
                continue
            if metadata_key_pattern.match(line):
                continue
            filtered_lines.append(line)
        return " ".join(filtered_lines).strip()

    @staticmethod
    def _scene_title(segment: ExplanationSegment, output_language: str = "zh-CN") -> str:
        if str(output_language or "").lower().startswith("en"):
            titles = {
                "problem_overview": "Understand",
                "variable_definition": "Define Variables",
                "modeling": "Set Up",
                "calculation": "Solve",
                "answer": "Final Answer",
                "recap": "Recap",
                "unsupported": "Not Supported",
            }
            return titles.get(segment.segment_type, segment.segment_type)
        titles = {
            "problem_overview": "读题",
            "variable_definition": "定义变量",
            "modeling": "建立关系",
            "calculation": "计算",
            "answer": "给出答案",
            "recap": "回顾",
            "unsupported": "无法生成正式讲解",
        }
        return titles.get(segment.segment_type, segment.segment_type)

    @staticmethod
    def _display_mode(segment: ExplanationSegment) -> str:
        if segment.segment_type == "answer":
            return "final_answer"
        if len(segment.math_expressions) == 1 and len(segment.math_expressions[0]) <= 12:
            return "inline"
        return "block"

    @staticmethod
    def _formula_style(segment: ExplanationSegment) -> dict[str, Any]:
        if segment.segment_type == "answer":
            return {"emphasis": "final_answer", "verified": True, "answer_card": True}
        if segment.segment_type == "calculation":
            return {"emphasis": "result"}
        return {}

    @classmethod
    def _formula_metadata(
        cls,
        segment: ExplanationSegment,
        expression: str,
        *,
        scs: SolutionChain | None,
        formula_style: dict[str, Any],
        output_language: str,
    ) -> dict[str, Any]:
        metadata = {
            "source_segment_id": segment.segment_id,
            "visual_refs": list(segment.visual_refs),
            "source_expression": segment.metadata.get("source_expression"),
            "display_expression": segment.metadata.get("display_expression") or expression,
            "source_sentence": segment.metadata.get("source_sentence"),
            "meaning": segment.metadata.get("meaning"),
            "strategy": segment.metadata.get("strategy"),
            "language": output_language,
        }
        if segment.segment_type == "answer":
            metadata["source_final_answer"] = (
                scs.final_answer if scs is not None else segment.metadata.get("source_final_answer")
            )
            metadata["answer_card"] = formula_style.get("answer_card") is True
            metadata["verified"] = formula_style.get("verified") is True
            verification_expression = cls._answer_verification_expression(scs)
            if verification_expression:
                metadata["verification_expression"] = verification_expression
        buildup_parts = cls._buildup_parts(expression)
        if segment.segment_type in {"modeling", "calculation"} and len(buildup_parts) >= 3:
            metadata["buildup_parts"] = buildup_parts
            metadata["buildup_mode"] = "sequential_terms"
            metadata["buildup_interval_ms"] = 450
        return metadata

    @classmethod
    def _answer_verification_expression(cls, scs: SolutionChain | None) -> str | None:
        if scs is None:
            return None
        final_items = scs.metadata.get("final_answer_items") or []
        if len(final_items) != 1:
            return None
        item = final_items[0]
        target = str(item.get("target") or "").strip()
        value = str(item.get("value") or "").strip()
        if not value:
            return None
        source_tool_call_id = str(item.get("source_tool_call_id") or "")
        for step in reversed(scs.steps):
            diagnostics = step.diagnostics or {}
            operation = diagnostics.get("operation")
            if operation not in {"solve_for", "solve_equation_system", "evaluate", "substitute"}:
                continue
            if source_tool_call_id and diagnostics.get("tool_call_id") != source_tool_call_id:
                continue
            input_payload = diagnostics.get("input") or {}
            if operation in {"evaluate", "substitute"}:
                expression = str(input_payload.get("expression") or "").strip()
                if expression:
                    expression = cls._substitute_expression_values(
                        expression,
                        input_payload.get("substitutions") or {},
                    )
                    return cls._display_math(f"{expression} = {value}")
            equations = input_payload.get("equations") or []
            if not equations:
                continue
            if not target:
                continue
            equation = equations[0]
            lhs = equation.get("lhs_sympy") or equation.get("lhs")
            rhs = equation.get("rhs_sympy") or equation.get("rhs")
            if not lhs or rhs is None:
                continue
            substituted_lhs = cls._replace_symbol(str(lhs), target, value)
            substituted_rhs = cls._replace_symbol(str(rhs), target, value)
            expression = f"{substituted_lhs} = {substituted_rhs}"
            return cls._display_math(expression)
        return None

    @staticmethod
    def _replace_symbol(expression: str, symbol: str, value: str) -> str:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
        return re.sub(pattern, value, expression)

    @classmethod
    def _substitute_expression_values(cls, expression: str, substitutions: dict[str, Any]) -> str:
        substituted = str(expression or "")
        for symbol in sorted((str(key) for key in substitutions.keys()), key=len, reverse=True):
            value = substitutions.get(symbol)
            if isinstance(value, (int, float)):
                replacement = cls._format_plain_number(value)
            else:
                replacement = str(value or "").strip()
            if not replacement or "$" in replacement:
                continue
            substituted = cls._replace_symbol(substituted, symbol, replacement)
        return substituted

    @staticmethod
    def _format_plain_number(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.12g}"

    @staticmethod
    def _display_math(expression: str) -> str:
        text = str(expression or "")
        text = ScriptDirector._sanitize_narration_text(text)
        text = text.replace("**", "^")
        text = text.replace("*", " × ")
        text = text.replace("/", " ÷ ")
        text = re.sub(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?", ScriptDirector._format_math_number, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _format_math_number(match: re.Match[str]) -> str:
        text = match.group(0)
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return str(int(number))
        return f"{number:.12g}"

    @staticmethod
    def _buildup_parts(expression: str) -> list[str]:
        text = str(expression or "").strip()
        if len(text) < 10:
            return []
        if any(char in text for char in "()[]{}"):
            return []
        parts = [
            part
            for part in re.split(r"(\s+(?:[=+\-*/÷×])\s+|\s*[；;]\s*)", text)
            if part and part.strip()
        ]
        if len(parts) < 3:
            return []
        merged: list[str] = []
        current = parts[0].strip()
        merged.append(current)
        index = 1
        while index + 1 < len(parts):
            operator = parts[index]
            operand = parts[index + 1].strip()
            if not operand:
                break
            if operator.strip() in {"；", ";"}:
                current = f"{current}; {operand}".strip()
            else:
                current = f"{current}{operator}{operand}".strip()
            merged.append(current)
            index += 2
        return merged if len(merged) >= 3 else []

    @classmethod
    def _map_action(cls, pedagogical_action: str) -> str:
        return cls._ACTION_MAP.get(pedagogical_action, "emphasize")

    @staticmethod
    def _action_target(
        action_type: str,
        formula_ids: list[str],
        segment: ExplanationSegment,
        narration_id: str,
    ) -> str:
        if formula_ids:
            return formula_ids[-1] if action_type == "final_answer_reveal" else formula_ids[0]
        if segment.visual_refs:
            return segment.visual_refs[0]
        return narration_id

    @staticmethod
    def _action_timing(
        action_type: str,
        scene_start_ms: int,
        narration_duration_ms: int,
        pause_after_ms: int,
    ) -> tuple[int, int]:
        offset = min(400, narration_duration_ms)
        if action_type == "pause":
            duration = pause_after_ms if pause_after_ms > 0 else ScriptDirector._QUESTION_PAUSE_DURATION_MS
            return scene_start_ms + narration_duration_ms, duration
        if action_type in {"highlight", "emphasize"}:
            delayed_offset = min(offset + ScriptDirector._HIGHLIGHT_DELAY_MS, narration_duration_ms)
            return scene_start_ms + delayed_offset, min(1200, max(800, narration_duration_ms // 3))
        if action_type == "final_answer_reveal":
            return scene_start_ms + offset, min(1200, max(800, narration_duration_ms // 2))
        return scene_start_ms + offset, min(1000, max(800, narration_duration_ms // 3))

    @classmethod
    def _eds_structure_issues(
        cls,
        *,
        scenes: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
        assets: list[dict[str, Any]],
        narration_tracks: list[dict[str, Any]],
        formula_tracks: list[dict[str, Any]],
        visual_actions: list[dict[str, Any]],
        sync_anchors: list[dict[str, Any]],
        total_duration_ms: int,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        groups = {
            "scene_id": [item["scene_id"] for item in scenes],
            "timeline_id": [item["timeline_id"] for item in timeline],
            "asset_id": [item["asset_id"] for item in assets],
            "narration_id": [item["narration_id"] for item in narration_tracks],
            "formula_id": [item["formula_id"] for item in formula_tracks],
            "action_id": [item["action_id"] for item in visual_actions],
            "anchor_id": [item["anchor_id"] for item in sync_anchors],
        }
        for label, values in groups.items():
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            if duplicates:
                errors.append(f"Duplicate {label}: {', '.join(duplicates)}.")

        scene_ids = set(groups["scene_id"])
        narration_ids = set(groups["narration_id"])
        formula_ids = set(groups["formula_id"])
        action_ids = set(groups["action_id"])
        asset_ids = set(groups["asset_id"])
        asset_ref_ids = {
            asset_id.removeprefix("asset_")
            for asset_id in asset_ids
            if asset_id.startswith("asset_")
        }
        known_targets = narration_ids | formula_ids | action_ids | asset_ids | asset_ref_ids

        for group_name, items in [
            ("timeline", timeline),
            ("narration", narration_tracks),
            ("formula", formula_tracks),
            ("visual_action", visual_actions),
            ("sync_anchor", sync_anchors),
        ]:
            missing_scene_ids = sorted(
                {
                    item["scene_id"]
                    for item in items
                    if item["scene_id"] not in scene_ids
                }
            )
            if missing_scene_ids:
                errors.append(f"{group_name} references missing scene_id: {', '.join(missing_scene_ids)}.")

        missing_timeline_targets = sorted(
            {
                item["target_id"]
                for item in timeline
                if item["target_id"] not in narration_ids | formula_ids | action_ids
            }
        )
        if missing_timeline_targets:
            errors.append(f"Timeline references missing target_id: {', '.join(missing_timeline_targets)}.")

        missing_action_targets = sorted(
            {
                item["target_id"]
                for item in visual_actions
                if item["target_id"] not in known_targets
            }
        )
        if missing_action_targets:
            errors.append(f"Visual actions reference missing target_id: {', '.join(missing_action_targets)}.")

        expected_start_ms = 0
        for scene in scenes:
            if scene["start_ms"] != expected_start_ms:
                errors.append(
                    f"Scene timeline is not continuous at {scene['scene_id']}: "
                    f"expected {expected_start_ms}, got {scene['start_ms']}."
                )
            expected_start_ms = scene["start_ms"] + scene["duration_ms"]
        if expected_start_ms != total_duration_ms:
            errors.append(
                f"Total duration mismatch: expected {expected_start_ms}, got {total_duration_ms}."
            )

        zero_duration_actions = sorted(
            item["action_id"]
            for item in visual_actions
            if item["duration_ms"] == 0
        )
        if zero_duration_actions:
            warnings.append(
                f"Visual actions have zero duration: {', '.join(zero_duration_actions)}."
            )

        return errors, warnings

    @staticmethod
    def _action_style(action_type: str, segment: ExplanationSegment) -> dict[str, Any]:
        if action_type == "highlight":
            return {"highlight": True, "tone": "focus"}
        if action_type == "final_answer_reveal":
            return {"highlight": True, "tone": "answer"}
        if action_type == "pause":
            return {"pacing": segment.pacing}
        return {}

    @staticmethod
    def _timeline_item(
        track_type: str,
        target_id: str,
        scene_id: str,
        start_ms: int,
        duration_ms: int,
        action: str,
    ) -> dict[str, Any]:
        return {
            "timeline_id": f"timeline_{track_type}_{target_id}",
            "scene_id": scene_id,
            "track_type": track_type,
            "target_id": target_id,
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "action": action,
        }

    @staticmethod
    def _safe_id(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
        return safe or "item"
