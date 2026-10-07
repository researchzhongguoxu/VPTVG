"""TeachingPlan schema for Explainer-stage pedagogical planning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TeachingPlanStatus = Literal["passed", "failed", "unsupported"]
TeachingPlannerMode = Literal["rule_based", "deepseek", "auto"]
TeachingMoveType = Literal["modeling", "calculation", "answer", "recap", "fallback"]


class TeachingPlanModel(BaseModel):
    """Base model with stable fields and no implicit extras."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class TeachingPlanMove(TeachingPlanModel):
    """One planned teacher move before conversion into ExplanationSegment."""

    move_id: str
    move_type: TeachingMoveType = "modeling"
    source_step_ids: list[str] = Field(default_factory=list)
    source_sentence: str | None = None
    meaning: str | None = None
    raw_expression: str | None = None
    display_expression: str | None = None
    narration: str
    pedagogical_actions: list[str] = Field(default_factory=list)
    estimated_duration_ms: int = Field(ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("move_id", "narration")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class TeachingPlanQualityReport(TeachingPlanModel):
    """Validation and fallback summary for TeachingPlan generation."""

    valid: bool
    status: TeachingPlanStatus
    checked_move_count: int = Field(ge=0)
    expanded_step_ids: list[str] = Field(default_factory=list)
    fallback_step_ids: list[str] = Field(default_factory=list)
    strategy_names: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeachingPlan(TeachingPlanModel):
    """Explainer-stage plan that describes how to teach verified solution steps."""

    schema_version: str = "TEACHING_PLAN-1.0"
    problem_id: str
    source_chain_id: str
    planner_mode: TeachingPlannerMode = "rule_based"
    moves: list[TeachingPlanMove] = Field(default_factory=list)
    quality_report: TeachingPlanQualityReport
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "source_chain_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_teaching_plan_dict(data: dict[str, Any]) -> TeachingPlan:
    """Validate a dictionary and return a TeachingPlan object."""

    return TeachingPlan.model_validate(data)
