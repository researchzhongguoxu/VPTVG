"""Executable Director Script schema.

EDS is the Script Director output consumed by a future programmatic renderer.
It describes renderer-agnostic 2D teaching video actions, not a concrete
MoviePy, Manim, or browser rendering API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EDSStatus = Literal["passed", "failed", "unsupported"]
EDSSceneType = Literal[
    "problem_overview",
    "variable_definition",
    "modeling",
    "calculation",
    "answer",
    "recap",
    "unsupported",
]
EDSTrackType = Literal["narration", "formula", "visual", "pause"]
EDSFormulaDisplayMode = Literal["inline", "block", "final_answer"]
EDSVisualActionType = Literal[
    "show",
    "reveal",
    "highlight",
    "emphasize",
    "pause",
    "final_answer_reveal",
]


class EDSModel(BaseModel):
    """Base model with stable fields and no implicit extras."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class EDSScene(EDSModel):
    """One teaching scene in the executable director script."""

    scene_id: str
    scene_type: EDSSceneType
    source_segment_ids: list[str] = Field(default_factory=list)
    title: str | None = None
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    layout: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scene_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSTimelineItem(EDSModel):
    """A global timeline event pointing at a concrete EDS track item."""

    timeline_id: str
    scene_id: str
    track_type: EDSTrackType
    target_id: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timeline_id", "scene_id", "target_id", "action")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSNarrationItem(EDSModel):
    """Narration text and TTS hints for one scene."""

    narration_id: str
    scene_id: str
    text: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    tts_hints: dict[str, Any] = Field(default_factory=dict)
    sync_anchor_ids: list[str] = Field(default_factory=list)

    @field_validator("narration_id", "scene_id", "text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSFormulaItem(EDSModel):
    """Formula or math-text render instruction."""

    formula_id: str
    scene_id: str
    expression: str
    display_mode: EDSFormulaDisplayMode = "block"
    source_step_ids: list[str] = Field(default_factory=list)
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    layout_slot: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)
    sync_anchor_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("formula_id", "scene_id", "expression")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSVisualAction(EDSModel):
    """Executable visual action for a renderer v1 adapter."""

    action_id: str
    scene_id: str
    action_type: EDSVisualActionType
    target_id: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    style: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id", "scene_id", "target_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSSyncAnchor(EDSModel):
    """Synchronization point linking narration, formulas, and visual actions."""

    anchor_id: str
    scene_id: str
    source_ref: str
    time_ms: int = Field(ge=0)
    linked_narration_id: str | None = None
    linked_formula_ids: list[str] = Field(default_factory=list)
    linked_action_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("anchor_id", "scene_id", "source_ref")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSRenderHints(EDSModel):
    """Renderer-agnostic constraints for programmatic 2D video rendering."""

    canvas_width: int = Field(default=1920, gt=0)
    canvas_height: int = Field(default=1080, gt=0)
    fps: int = Field(default=15, gt=0)
    theme: str = "default"
    language: str = "zh-CN"
    default_font_family: str = "sans-serif"
    formula_renderer: str = "programmatic_2d"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("theme", "language", "default_font_family", "formula_renderer")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EDSQualityReport(EDSModel):
    """Director precheck report for the generated EDS artifact."""

    valid: bool
    status: EDSStatus
    checked_scene_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EDSAsset(EDSModel):
    """Renderer input asset referenced by EDS tracks or visual actions."""

    asset_id: str
    asset_type: str
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("asset_id", "asset_type")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ExecutableDirectorScript(EDSModel):
    """Executable Director Script, version 1.0."""

    schema_version: str = "EDS-1.0"
    problem_id: str
    source_explanation_id: str
    source_chain_id: str
    scenes: list[EDSScene] = Field(default_factory=list)
    timeline: list[EDSTimelineItem] = Field(default_factory=list)
    assets: list[EDSAsset] = Field(default_factory=list)
    narration_tracks: list[EDSNarrationItem] = Field(default_factory=list)
    formula_tracks: list[EDSFormulaItem] = Field(default_factory=list)
    visual_actions: list[EDSVisualAction] = Field(default_factory=list)
    sync_anchors: list[EDSSyncAnchor] = Field(default_factory=list)
    render_hints: EDSRenderHints = Field(default_factory=EDSRenderHints)
    quality_report: EDSQualityReport
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "source_explanation_id", "source_chain_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_executable_director_script_dict(data: dict[str, Any]) -> ExecutableDirectorScript:
    """Validate a dictionary and return an ExecutableDirectorScript object."""

    return ExecutableDirectorScript.model_validate(data)
