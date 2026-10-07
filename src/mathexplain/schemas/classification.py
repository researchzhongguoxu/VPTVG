"""Problem classification and solver routing schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Difficulty = Literal["elementary", "intermediate", "advanced", "unknown"]


class ClassificationModel(BaseModel):
    """Base model for stable classification artifacts."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class SolverConfig(ClassificationModel):
    """Minimal solver configuration hint consumed by future Solver agents."""

    solver_name: str
    expected_emr_type: str
    task_type: str
    strategy: str

    @field_validator("solver_name", "expected_emr_type", "task_type", "strategy")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ClassificationResult(ClassificationModel):
    """Problem Classifier output, version 1.0."""

    schema_version: str = "CLASSIFICATION-1.0"
    problem_id: str
    primary_type: str = "unknown"
    task_type: str = "unknown"
    difficulty: Difficulty = "unknown"
    knowledge_units: list[str] = Field(default_factory=list)
    solver_route: str = "unknown"
    solver_config: SolverConfig
    requires_solver: bool = True
    requires_verifier: bool = True
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "primary_type", "task_type", "solver_route")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_classification_result_dict(data: dict[str, Any]) -> ClassificationResult:
    """Validate a dictionary and return a ClassificationResult object."""

    return ClassificationResult.model_validate(data)
