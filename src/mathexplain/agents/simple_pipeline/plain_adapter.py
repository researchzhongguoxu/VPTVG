"""Plain video adapter for the Simple Pipeline baseline."""

from __future__ import annotations

from typing import Any

from mathexplain.config.language import is_english_language
from mathexplain.rendering.plain_text import normalize_plain_math_text
from mathexplain.schemas.eds import ExecutableDirectorScript
from mathexplain.schemas.simple_video import SimpleVideoScript

PLAIN_SIMPLE_VISUAL_VERSION = "plain_simple_text_v4"


class PlainSimpleEDSAdapter:
    """Convert SimpleVideoScript into a plain slide-style EDS baseline.

    This adapter intentionally avoids MathExplainAgent's richer visual teaching
    affordances such as key-fact panels, semantic highlights, teaching hints,
    and formula cards. It exists as a plain MLLM-script-to-video baseline.
    """

    def to_eds(
        self,
        script: SimpleVideoScript,
        *,
        output_language: str = "en-US",
    ) -> ExecutableDirectorScript:
        scenes: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        narration_tracks: list[dict[str, Any]] = []
        current_ms = 0

        for index, segment in enumerate(script.segments):
            safe_id = self._safe_id(segment.segment_id)
            scene_id = f"plain_simple_scene_{index}_{safe_id}"
            narration_id = f"plain_simple_nar_{safe_id}"
            duration = int(segment.estimated_duration_ms)
            pause_after = 350 if segment.segment_type != "recap" else 0
            display_text = normalize_plain_math_text(segment.display_text or segment.narration)
            formula_text = normalize_plain_math_text(segment.formula)

            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_type": "problem_overview" if segment.segment_type == "problem_overview" else "modeling",
                    "source_segment_ids": [segment.segment_id],
                    "title": self._scene_title(segment.segment_type, output_language),
                    "start_ms": current_ms,
                    "duration_ms": duration + pause_after,
                    "layout": {"style": "plain_slide"},
                    "metadata": {
                        "source": "plain_simple_adapter",
                        "plain_display_text": display_text,
                        "plain_formula_text": formula_text,
                        "plain_visual_version": PLAIN_SIMPLE_VISUAL_VERSION,
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
                    "tts_hints": {},
                    "sync_anchor_ids": [],
                }
            )
            timeline.append(
                {
                    "timeline_id": f"plain_simple_timeline_narration_{narration_id}",
                    "scene_id": scene_id,
                    "track_type": "narration",
                    "target_id": narration_id,
                    "start_ms": current_ms,
                    "duration_ms": duration,
                    "action": "play",
                    "metadata": {"source": "plain_simple_adapter"},
                }
            )
            current_ms += duration + pause_after

        return ExecutableDirectorScript.model_validate(
            {
                "problem_id": script.problem_id,
                "source_explanation_id": f"plain_simple_script_{script.problem_id}",
                "source_chain_id": "plain_simple_no_scs",
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
                "formula_tracks": [],
                "visual_actions": [],
                "sync_anchors": [],
                "render_hints": {
                    "canvas_width": 1920,
                    "canvas_height": 1080,
                    "fps": 15,
                    "theme": "plain_simple_baseline",
                    "language": output_language,
                    "default_font_family": "sans-serif",
                    "formula_renderer": "plain_text",
                    "metadata": {
                        "source": "plain_simple_adapter",
                        "uses_key_facts_panel": False,
                        "uses_semantic_highlights": False,
                        "uses_formula_cards": False,
                        "plain_visual_version": PLAIN_SIMPLE_VISUAL_VERSION,
                    },
                },
                "quality_report": {
                    "valid": True,
                    "status": "passed",
                    "checked_scene_count": len(scenes),
                    "errors": [],
                    "warnings": [],
                    "metadata": {
                        "adapter": "PlainSimpleEDSAdapter",
                        "adapter_version": "Plain Simple Baseline v1",
                        "weak_eds": True,
                    },
                },
                "metadata": {
                    "source": "plain_simple_pipeline",
                    "simple_video_schema_version": script.schema_version,
                    "does_not_use_script_director": True,
                    "does_not_use_verifier": True,
                    "does_not_use_cas": True,
                    "plain_visual_baseline": True,
                    "plain_visual_version": PLAIN_SIMPLE_VISUAL_VERSION,
                },
            }
        )

    @staticmethod
    def _safe_id(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value).strip("_")
        return cleaned or "segment"

    @staticmethod
    def _scene_title(segment_type: str, output_language: str) -> str:
        english = is_english_language(output_language)
        if english:
            return {
                "problem_overview": "Problem",
                "step": "Step",
                "answer": "Answer",
                "recap": "Summary",
            }.get(segment_type, "Step")
        return {
            "problem_overview": "Problem",
            "step": "Step",
            "answer": "Answer",
            "recap": "Summary",
        }.get(segment_type, "Step")
