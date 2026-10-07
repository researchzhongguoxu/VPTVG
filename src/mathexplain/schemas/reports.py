"""Shared report schemas for pipeline components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ReportStatus = Literal["success", "warning", "failed", "skipped", "unknown"]


class ReportModel(BaseModel):
    """Base model for structured, inspectable component reports."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class PipelineStageReport(ReportModel):
    """Report for a single stage inside a component pipeline."""

    stage_name: str
    status: ReportStatus = "unknown"
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_name")
    @classmethod
    def stage_name_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ParsingReport(ReportModel):
    """Structured report emitted by SGVP / VisionParser."""

    schema_version: str = "ParsingReport-1.0"
    problem_id: str
    status: ReportStatus = "unknown"
    stages: list[PipelineStageReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence_summary: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_parsing_report_dict(data: dict[str, Any]) -> ParsingReport:
    """Validate a dictionary and return a ParsingReport object."""

    return ParsingReport.model_validate(data)
