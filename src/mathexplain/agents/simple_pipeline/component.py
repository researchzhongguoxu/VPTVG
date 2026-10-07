"""Simple Pipeline baseline: direct MLLM explanation to weak EDS."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from mathexplain.config.language import is_english_language
from mathexplain.schemas.eds import ExecutableDirectorScript
from mathexplain.schemas.simple_video import SimpleVideoScript


class ImageJSONClient(Protocol):
    """Protocol implemented by Qwen image clients used by this baseline."""

    provider: str
    model: str
    last_usage: dict[str, Any]

    def generate_json_from_image(self, image_data_uri: str, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON string from an image prompt."""


@dataclass(frozen=True)
class SimplePipelineResult:
    """Direct generation result before normalization."""

    raw_response: str
    usage: dict[str, Any]
    provider: str
    model: str


class DirectExplanationGenerator:
    """Ask a multimodal LLM to directly produce a step-by-step video script."""

    SYSTEM_PROMPT = (
        "You are a careful math tutor and video-script writer. "
        "Read the math problem from the image and produce a concise, structured explanation. "
        "Return JSON only. Do not use Markdown. Do not mention internal system names."
    )

    USER_PROMPT = """Read the math problem in the image and solve it.

Return exactly this JSON object:
{
  "problem_text": "full problem text",
  "final_answer": "final answer only",
  "steps": [
    {
      "title": "short step title",
      "narration": "student-friendly narration",
      "formula": "one formula or computation to display"
    }
  ],
  "summary": "one sentence final recap"
}

Requirements:
- Use the same language as the problem unless instructed otherwise.
- Keep 3 to 6 steps when possible.
- Each step should explain one local reasoning move.
- The final_answer must be short and should not include extra prose.
- Return valid JSON only."""

    def __init__(self, client: ImageJSONClient) -> None:
        self.client = client

    def generate(self, image_data_uri: str, *, output_language: str = "auto") -> SimplePipelineResult:
        user_prompt = self.USER_PROMPT
        if output_language == "en-US":
            user_prompt += "\nUse English for all narration and labels."
        elif output_language == "zh-CN":
            user_prompt += "\n请使用中文生成全部讲解、标签和总结。"
        else:
            user_prompt += "\nUse the dominant natural language in the image."
        raw = self.client.generate_json_from_image(
            image_data_uri,
            self.SYSTEM_PROMPT,
            user_prompt,
        )
        return SimplePipelineResult(
            raw_response=raw,
            usage=dict(getattr(self.client, "last_usage", {}) or {}),
            provider=getattr(self.client, "provider", "unknown"),
            model=getattr(self.client, "model", "unknown"),
        )


class SimpleScriptNormalizer:
    """Normalize direct MLLM JSON into a bounded SimpleVideoScript."""

    MAX_STEPS = 8
    MAX_TEXT_CHARS = 420
    MAX_PROBLEM_CHARS = 900

    def normalize(
        self,
        raw_response: str,
        *,
        problem_id: str,
        source_image: str,
        output_language: str = "en-US",
    ) -> SimpleVideoScript:
        data = self._parse_json_object(raw_response)
        if data is None:
            raise ValueError("Simple Pipeline response was not valid JSON.")

        problem_text = self._clean_text(data.get("problem_text") or data.get("question") or "")
        final_answer = self._clean_text(data.get("final_answer") or data.get("answer") or "")
        if not problem_text:
            raise ValueError("Simple Pipeline response is missing problem_text.")
        if not final_answer:
            raise ValueError("Simple Pipeline response is missing final_answer.")
        if output_language == "auto":
            output_language = self._detect_language(problem_text)

        warnings: list[str] = []
        problem_text = self._truncate(problem_text, self.MAX_PROBLEM_CHARS)
        steps = data.get("steps") or []
        if not isinstance(steps, list):
            steps = []
            warnings.append("steps was not a list and was replaced with an empty list.")
        if len(steps) > self.MAX_STEPS:
            warnings.append(f"steps was truncated from {len(steps)} to {self.MAX_STEPS}.")
            steps = steps[: self.MAX_STEPS]
        summary = self._clean_text(data.get("summary") or "")

        segments: list[dict[str, Any]] = [
            {
                "segment_id": "seg_problem_overview",
                "segment_type": "problem_overview",
                "narration": self._overview_narration(problem_text, output_language),
                "display_text": problem_text,
                "formula": None,
                "estimated_duration_ms": self._duration_for_text(problem_text, 8000, 15000),
                "metadata": {"source": "simple_pipeline", "role": "problem"},
            }
        ]

        step_answer_evidence: list[dict[str, str]] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                warnings.append(f"step {index} was not an object and was skipped.")
                continue
            narration = self._clean_text(step.get("narration") or step.get("explanation") or step.get("title") or "")
            formula = self._clean_text(step.get("formula") or step.get("display_expression") or "")
            display = self._clean_text(step.get("display_text") or formula or step.get("title") or narration)
            if not narration:
                warnings.append(f"step {index} had empty narration and was skipped.")
                continue
            step_answer_evidence.append({"narration": narration, "formula": formula, "display": display})
            segments.append(
                {
                    "segment_id": f"seg_step_{index}",
                    "segment_type": "step",
                    "narration": self._truncate(narration, self.MAX_TEXT_CHARS),
                    "display_text": self._truncate(display, 180),
                    "formula": self._truncate(formula, 180) if formula else None,
                    "estimated_duration_ms": self._duration_for_text(narration, 5200, 11000),
                    "metadata": {
                        "source": "direct_mllm_step",
                        "step_index": index,
                        "title": self._clean_text(step.get("title") or ""),
                    },
                }
            )

        final_answer, reconciliation_warning = self._reconcile_final_answer(
            final_answer,
            step_answer_evidence,
            summary,
        )
        if reconciliation_warning:
            warnings.append(reconciliation_warning)

        answer_narration = self._answer_narration(final_answer, output_language)
        segments.append(
            {
                "segment_id": "seg_answer",
                "segment_type": "answer",
                "narration": answer_narration,
                "display_text": final_answer,
                "formula": final_answer,
                "estimated_duration_ms": self._duration_for_text(answer_narration, 4000, 8000),
                "metadata": {"source": "simple_pipeline", "role": "answer"},
            }
        )
        recap_narration = summary or self._recap_narration(final_answer, output_language)
        segments.append(
            {
                "segment_id": "seg_recap",
                "segment_type": "recap",
                "narration": self._truncate(recap_narration, self.MAX_TEXT_CHARS),
                "display_text": final_answer,
                "formula": final_answer,
                "estimated_duration_ms": self._duration_for_text(recap_narration, 3800, 8000),
                "metadata": {"source": "simple_pipeline", "role": "recap"},
            }
        )

        return SimpleVideoScript.model_validate(
            {
                "problem_id": problem_id,
                "source_image": source_image,
                "problem_text": problem_text,
                "final_answer": final_answer,
                "segments": segments,
                "quality_report": {
                    "valid": True,
                    "status": "passed",
                    "checked_segment_count": len(segments),
                    "errors": [],
                    "warnings": warnings,
                    "metadata": {
                        "normalizer": "SimpleScriptNormalizer",
                        "normalizer_version": "Simple Pipeline v1",
                    },
                },
                "metadata": {
                    "baseline": "simple_pipeline",
                    "does_not_use_cas": True,
                    "does_not_use_verifier": True,
                    "does_not_use_teaching_plan": True,
                },
            }
        )

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @classmethod
    def _reconcile_final_answer(
        cls,
        final_answer: str,
        step_evidence: list[dict[str, str]],
        summary: str,
    ) -> tuple[str, str | None]:
        """Repair direct-MLLM JSON when its top-level answer contradicts itself.

        The Simple/Plain Simple baselines are intentionally not verified by CAS.
        This guard only fixes internal JSON inconsistency: the final step result
        and either the LLM's own summary must agree, or the last step must
        clearly present the final computation.
        """

        step_candidate = cls._candidate_from_last_step(step_evidence)
        summary_candidates = cls._candidates_from_summary(summary)
        summary_candidate = summary_candidates[-1] if summary_candidates else None
        if not step_candidate:
            return final_answer, None

        step_number = cls._numeric_value(step_candidate)
        answer_number = cls._numeric_value(final_answer)
        if step_number is None:
            return final_answer, None
        evidence_source = "matching final step result and summary"
        matching_summary_candidate = cls._candidate_matching_number(summary_candidates, step_number)
        has_final_intent = cls._last_step_has_final_answer_intent(step_evidence)
        if not matching_summary_candidate:
            if not cls._last_step_has_final_answer_intent(step_evidence):
                return final_answer, None
            evidence_source = "explicit final-step result"
            summary_candidate = None
        else:
            summary_candidate = matching_summary_candidate

        if answer_number == step_number and cls._has_unit_or_entity(final_answer):
            return final_answer, None

        replacement = summary_candidate if cls._has_unit_or_entity(summary_candidate) else step_candidate
        if cls._clean_answer_text(replacement) == cls._clean_answer_text(final_answer):
            return final_answer, None
        reason = f"final_answer was reconciled with the {evidence_source} ({final_answer!r} -> {replacement!r})."
        return replacement, reason

    @staticmethod
    def _last_step_has_final_answer_intent(step_evidence: list[dict[str, str]]) -> bool:
        if not step_evidence:
            return False
        last = step_evidence[-1]
        text = " ".join(str(last.get(key, "")) for key in ("narration", "display", "formula"))
        return bool(
            re.search(
                r"\b(finally|final answer|answer|total|in all|altogether|overall|arrive at|this gives|find out how many)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _summary_has_final_answer_intent(summary: str) -> bool:
        return bool(
            re.search(
                r"\b(final answer|final result|answer is|arrive at|we get|we find|we determined|determined|which is|therefore|so|there (?:are|is))\b",
                str(summary or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _candidate_from_last_step(cls, step_evidence: list[dict[str, str]]) -> str | None:
        for step in reversed(step_evidence):
            for key in ("formula", "display", "narration"):
                candidate = cls._result_phrase_from_expression(step.get(key, ""))
                if candidate:
                    return candidate
        return None

    @classmethod
    def _candidate_from_summary(cls, summary: str) -> str | None:
        return cls._last_number_phrase(summary, require_near_end=True)

    @classmethod
    def _candidates_from_summary(cls, summary: str) -> list[str]:
        return [candidate["phrase"] for candidate in cls._number_phrase_candidates(summary)]

    @classmethod
    def _candidate_matching_number(cls, candidates: list[str], expected: Decimal) -> str | None:
        for candidate in reversed(candidates):
            number = cls._numeric_value(candidate)
            if number == expected:
                return candidate
        return None

    @classmethod
    def _result_phrase_from_expression(cls, value: str) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        tail = re.split(r"=>|->|→", text)[-1]
        if "=" in tail:
            tail = tail.rsplit("=", 1)[-1]
        return cls._last_number_phrase(tail)

    @classmethod
    def _number_phrase_candidates(cls, value: str) -> list[dict[str, str]]:
        text = cls._normalize_answer_candidate_text(value)
        if not text:
            return []
        unit_words = (
            r"percent|percentage|dollars?|cents?|hours?|minutes?|seconds?|miles?|kilometers?|km|feet|"
            r"yards?|inches?|millimeters?|mm|pounds?|lbs?|oranges?|watermelons?|melons?|"
            r"years?(?:\s+old)?|good\s+oranges?|people|persons?|supporters?|fries|bandages?|"
            r"books?|items?|pages?"
        )
        pattern = re.compile(
            r"(?P<prefix>\$?\s*)"
            r"(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)"
            rf"(?P<unit>\s*(?:%|(?:{unit_words})\b))?",
            flags=re.IGNORECASE,
        )
        candidates: list[dict[str, str]] = []
        for match in pattern.finditer(text):
            number = match.group("number").replace(",", "")
            prefix = (match.group("prefix") or "").strip()
            unit = (match.group("unit") or "").strip()
            phrase = f"{number}{unit}" if unit == "%" else f"{number} {unit}".strip()
            if prefix == "$" and not unit:
                phrase = f"${number}"
            candidates.append({"phrase": phrase, "start": str(match.start()), "end": str(match.end())})
        return candidates

    @staticmethod
    def _last_number_phrase(value: str, *, require_near_end: bool = False) -> str | None:
        text = SimpleScriptNormalizer._normalize_answer_candidate_text(value)
        if not text:
            return None
        candidates = SimpleScriptNormalizer._number_phrase_candidates(text)
        if not candidates:
            return None
        candidate = candidates[-1]
        end = int(candidate["end"])
        if require_near_end:
            raw_trailing = text[end:]
            trailing = raw_trailing.strip(" .,!?:;")
            if raw_trailing.startswith("-") or len(trailing) > 80:
                return None
        return candidate["phrase"]

    @staticmethod
    def _normalize_answer_candidate_text(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
        text = text.replace("¡Á", "×").replace("¡Â", "÷")
        return text

    @staticmethod
    def _numeric_value(value: str) -> Decimal | None:
        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", str(value or ""))
        if not match:
            return None
        number = match.group(0).replace(",", "")
        try:
            if "/" in number:
                numerator, denominator = number.split("/", 1)
                return Decimal(numerator) / Decimal(denominator)
            return Decimal(number)
        except (InvalidOperation, ZeroDivisionError):
            return None

    @staticmethod
    def _has_unit_or_entity(value: str) -> bool:
        if "$" in str(value or "") or "%" in str(value or ""):
            return True
        return bool(
            re.search(
                r"\b(percent|percentage|dollars?|cents?|hours?|minutes?|seconds?|miles?|kilometers?|km|feet|yards?|inches?|millimeters?|mm|pounds?|lbs?|oranges?|watermelons?|melons?|years?|old|people|persons?|supporters?|fries|bandages?|books?|items?|pages?)\b",
                str(value or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _clean_answer_text(value: str) -> str:
        text = str(value or "").strip().lower().replace(",", "")
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _duration_for_text(text: str, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, 1600 + len(text) * 85))

    @staticmethod
    def _overview_narration(problem_text: str, output_language: str) -> str:
        if is_english_language(output_language):
            return f"Let's read the problem first. {problem_text}"
        return f"我们先读题：{problem_text}"

    @staticmethod
    def _answer_narration(final_answer: str, output_language: str) -> str:
        if is_english_language(output_language):
            return f"The final answer is {final_answer}."
        return f"最终答案是 {final_answer}。"

    @staticmethod
    def _recap_narration(final_answer: str, output_language: str) -> str:
        if is_english_language(output_language):
            return f"To recap, the final result is {final_answer}."
        return f"最后回顾一下，结果是 {final_answer}。"

    @staticmethod
    def _detect_language(text: str) -> str:
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_count = len(re.findall(r"[A-Za-z]{2,}", text))
        if zh_count > latin_count * 2:
            return "zh-CN"
        return "en-US"


class SimpleEDSAdapter:
    """Convert SimpleVideoScript into a minimal EDS for the shared renderer."""

    FORMULA_OFFSET_MS = 300

    def to_eds(
        self,
        script: SimpleVideoScript,
        *,
        output_language: str = "en-US",
    ) -> ExecutableDirectorScript:
        scenes: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        narration_tracks: list[dict[str, Any]] = []
        formula_tracks: list[dict[str, Any]] = []
        visual_actions: list[dict[str, Any]] = []
        sync_anchors: list[dict[str, Any]] = []
        current_ms = 0

        for index, segment in enumerate(script.segments):
            safe_id = self._safe_id(segment.segment_id)
            scene_id = f"simple_scene_{index}_{safe_id}"
            narration_id = f"simple_nar_{safe_id}"
            anchor_id = f"simple_anchor_{safe_id}"
            duration = int(segment.estimated_duration_ms)
            pause_after = 400 if segment.segment_type not in {"recap"} else 0
            scene_type = "modeling" if segment.segment_type == "step" else segment.segment_type
            title = self._scene_title(segment.segment_type, output_language)

            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_type": scene_type,
                    "source_segment_ids": [segment.segment_id],
                    "title": title,
                    "start_ms": current_ms,
                    "duration_ms": duration + pause_after,
                    "layout": {"focus_area": self._focus_area(segment.segment_type)},
                    "metadata": {
                        "source": "simple_pipeline_adapter",
                        "simple_segment_type": segment.segment_type,
                        **segment.metadata,
                    },
                }
            )
            narration_tracks.append(
                {
                    "narration_id": narration_id,
                    "scene_id": scene_id,
                    "text": segment.narration,
                    "start_ms": current_ms,
                    "duration_ms": duration,
                    "pause_after_ms": pause_after,
                    "tts_hints": {"emphasis_terms": [script.final_answer]} if segment.segment_type in {"answer", "recap"} else {},
                    "sync_anchor_ids": [anchor_id],
                }
            )
            timeline.append(self._timeline_item("narration", narration_id, scene_id, current_ms, duration, "play"))

            formula_ids: list[str] = []
            expression = segment.formula or segment.display_text
            if expression:
                formula_id = f"simple_formula_{safe_id}_0"
                formula_ids.append(formula_id)
                formula_start = current_ms + min(self.FORMULA_OFFSET_MS, duration)
                formula_duration = max(0, duration - min(self.FORMULA_OFFSET_MS, duration))
                formula_tracks.append(
                    {
                        "formula_id": formula_id,
                        "scene_id": scene_id,
                        "expression": expression,
                        "display_mode": "final_answer" if segment.segment_type in {"answer", "recap"} else "block",
                        "source_step_ids": [],
                        "start_ms": formula_start,
                        "duration_ms": formula_duration,
                        "layout_slot": self._focus_area(segment.segment_type),
                        "style": {"emphasis": "final_answer"} if segment.segment_type in {"answer", "recap"} else {},
                        "sync_anchor_ids": [anchor_id],
                        "metadata": {
                            "source": "simple_video_script",
                            "display_text": segment.display_text,
                        },
                    }
                )
                timeline.append(self._timeline_item("formula", formula_id, scene_id, formula_start, formula_duration, "render"))

            action_type = "final_answer_reveal" if segment.segment_type in {"answer", "recap"} else "reveal"
            target_id = formula_ids[0] if formula_ids else narration_id
            action_id = f"simple_action_{safe_id}_0"
            visual_actions.append(
                {
                    "action_id": action_id,
                    "scene_id": scene_id,
                    "action_type": action_type,
                    "target_id": target_id,
                    "start_ms": current_ms + min(500, duration),
                    "duration_ms": max(800, duration - min(500, duration)),
                    "style": {"emphasis": "final_answer"} if action_type == "final_answer_reveal" else {},
                    "metadata": {"source": "simple_pipeline_adapter"},
                }
            )
            timeline.append(
                self._timeline_item(
                    "visual",
                    action_id,
                    scene_id,
                    current_ms + min(500, duration),
                    max(800, duration - min(500, duration)),
                    action_type,
                )
            )
            sync_anchors.append(
                {
                    "anchor_id": anchor_id,
                    "scene_id": scene_id,
                    "source_ref": segment.segment_id,
                    "time_ms": current_ms,
                    "linked_narration_id": narration_id,
                    "linked_formula_ids": formula_ids,
                    "linked_action_ids": [action_id],
                    "metadata": {"source": "simple_pipeline_adapter"},
                }
            )
            current_ms += duration + pause_after

        return ExecutableDirectorScript.model_validate(
            {
                "problem_id": script.problem_id,
                "source_explanation_id": f"simple_script_{script.problem_id}",
                "source_chain_id": "simple_pipeline_no_scs",
                "scenes": scenes,
                "timeline": timeline,
                "assets": [
                    {
                        "asset_id": "problem_text",
                        "asset_type": "text",
                        "uri": None,
                        "metadata": {"text": script.problem_text, "source_image": script.source_image},
                    }
                ],
                "narration_tracks": narration_tracks,
                "formula_tracks": formula_tracks,
                "visual_actions": visual_actions,
                "sync_anchors": sync_anchors,
                "render_hints": {
                    "canvas_width": 1920,
                    "canvas_height": 1080,
                    "fps": 15,
                    "theme": "simple_baseline",
                    "language": output_language,
                    "default_font_family": "sans-serif",
                    "formula_renderer": "programmatic_2d",
                    "metadata": {"source": "simple_pipeline_adapter"},
                },
                "quality_report": {
                    "valid": True,
                    "status": "passed",
                    "checked_scene_count": len(scenes),
                    "errors": [],
                    "warnings": [],
                    "metadata": {
                        "adapter": "SimpleEDSAdapter",
                        "adapter_version": "Simple Pipeline v1",
                        "weak_eds": True,
                    },
                },
                "metadata": {
                    "source": "simple_pipeline",
                    "simple_video_schema_version": script.schema_version,
                    "does_not_use_script_director": True,
                    "does_not_use_verifier": True,
                    "does_not_use_cas": True,
                },
            }
        )

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
            "timeline_id": f"simple_timeline_{track_type}_{target_id}",
            "scene_id": scene_id,
            "track_type": track_type,
            "target_id": target_id,
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "action": action,
            "metadata": {"source": "simple_pipeline_adapter"},
        }

    @staticmethod
    def _safe_id(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
        return cleaned or "segment"

    @staticmethod
    def _focus_area(segment_type: str) -> str:
        if segment_type == "problem_overview":
            return "problem_context"
        if segment_type in {"answer", "recap"}:
            return "final_answer"
        return "main_formula"

    @staticmethod
    def _scene_title(segment_type: str, output_language: str) -> str:
        english = is_english_language(output_language)
        if english:
            return {
                "problem_overview": "Problem",
                "step": "Step",
                "answer": "Answer",
                "recap": "Recap",
            }.get(segment_type, "Step")
        return {
            "problem_overview": "读题",
            "step": "步骤",
            "answer": "答案",
            "recap": "回顾",
        }.get(segment_type, "步骤")
