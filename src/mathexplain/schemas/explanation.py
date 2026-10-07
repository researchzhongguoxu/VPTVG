"""Pedagogical explanation script schema.

ExplanationScript is the Explainer output consumed by the future Script
Director. It describes teaching intent, not executable EDS timing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExplanationStatus = Literal["passed", "failed", "unsupported"]
AudienceLevel = Literal["concise", "standard", "detailed"]
SegmentType = Literal[
    "problem_overview",
    "variable_definition",
    "modeling",
    "calculation",
    "answer",
    "recap",
    "unsupported",
]
Pacing = Literal["slow", "normal", "fast"]


class ExplanationModel(BaseModel):
    """Base model with stable fields and no implicit extras."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class VisualAnchor(ExplanationModel):
    """A semantic visual reference for Script Director."""

    anchor_id: str
    kind: str
    label: str
    source_step_id: str | None = None
    expression: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("anchor_id", "kind", "label")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ExplanationSegment(ExplanationModel):
    """One video-friendly teaching segment."""

    segment_id: str
    segment_type: SegmentType
    linked_step_ids: list[str] = Field(default_factory=list)
    narration: str
    math_expressions: list[str] = Field(default_factory=list)
    visual_refs: list[str] = Field(default_factory=list)
    pedagogical_actions: list[str] = Field(default_factory=list)
    pacing: Pacing = "normal"
    estimated_duration_ms: int = Field(ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    tts_hints: dict[str, Any] = Field(default_factory=dict)
    layout_hints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("segment_id", "narration")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ExplanationConsistencyReport(ExplanationModel):
    """Rule-based factual consistency report for explanation generation."""

    valid: bool
    status: ExplanationStatus
    checked_segment_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplanationScript(ExplanationModel):
    """Explainer output, version 1.0."""

    schema_version: str = "EXPLANATION-1.0"
    problem_id: str
    source_chain_id: str
    status: ExplanationStatus
    audience_level: AudienceLevel = "standard"
    segments: list[ExplanationSegment] = Field(default_factory=list)
    visual_anchors: list[VisualAnchor] = Field(default_factory=list)
    consistency_report: ExplanationConsistencyReport
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "source_chain_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_explanation_script_dict(data: dict[str, Any]) -> ExplanationScript:
    """Validate a dictionary and return an ExplanationScript object."""

    return ExplanationScript.model_validate(data)
