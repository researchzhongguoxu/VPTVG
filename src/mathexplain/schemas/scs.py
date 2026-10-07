"""Solution Chain Structure schema.

SCS stores candidate solution chains produced by a solver. It is a structured
artifact for verification, repair, and downstream explanation generation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


VerificationStatus = Literal["unverified", "verified", "failed", "repaired", "unknown"]


class SchemaModel(BaseModel):
    """Base model with stable field names and no implicit extra fields."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class SolutionStep(SchemaModel):
    """One verifiable step in a candidate solution chain."""

    step_id: str
    step_index: int = Field(ge=0)
    description: str
    math_expression: str | None = None
    rule_name: str | None = None
    justification: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = "unverified"
    verification_summary: str | None = None
    verification_score: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "description")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class SolutionChain(SchemaModel):
    """A complete candidate solution chain."""

    schema_version: str = "SCS-1.0"
    problem_id: str
    source_emr_id: str | None = None
    source_emr_version: str | None = None
    chain_id: str
    steps: list[SolutionStep] = Field(default_factory=list)
    final_answer: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "chain_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class SolutionCandidateSet(SchemaModel):
    """A set of K candidate solution chains for self-consistency and verification."""

    schema_version: str = "SCS-1.0"
    problem_id: str
    candidates: list[SolutionChain] = Field(default_factory=list)
    selected_chain_id: str | None = None
    ranking_report: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_solution_chain_dict(data: dict[str, Any]) -> SolutionChain:
    """Validate a dictionary and return a SolutionChain object."""

    return SolutionChain.model_validate(data)


def validate_solution_candidate_set_dict(data: dict[str, Any]) -> SolutionCandidateSet:
    """Validate a dictionary and return a SolutionCandidateSet object."""

    return SolutionCandidateSet.model_validate(data)
