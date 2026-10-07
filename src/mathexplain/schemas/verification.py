"""Verification report schema.

Verifier reports are saved as inspectable artifacts after SCS generation. The
first version records single-chain, CAS-backed checks and does not claim full
semantic proof of the original word problem.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


VerificationMode = Literal["tool_call_backed", "emr_backed", "unsupported"]
VerificationStatus = Literal["passed", "failed", "unsupported"]
StepVerificationStatus = Literal["passed", "failed", "skipped"]


class VerificationModel(BaseModel):
    """Base model with stable field names and no implicit extra fields."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class CheckedStep(VerificationModel):
    """One step-level verification result."""

    step_id: str
    tool_call_id: str | None = None
    operation: str | None = None
    status: StepVerificationStatus
    backend_replayed: bool = False
    message: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "message")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class VerificationReport(VerificationModel):
    """Verifier output, version 1.0."""

    schema_version: str = "VERIFICATION-1.0"
    problem_id: str
    chain_id: str
    valid: bool
    status: VerificationStatus
    verification_mode: VerificationMode
    scs_schema_valid: bool
    backend_steps_valid: bool
    final_answer_valid: bool
    checked_steps: list[CheckedStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id", "chain_id", "summary")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_verification_report_dict(data: dict[str, Any]) -> VerificationReport:
    """Validate a dictionary and return a VerificationReport object."""

    return VerificationReport.model_validate(data)
