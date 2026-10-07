"""Structured Problem Representation schema.

SPR describes the math problem as it appears in the source image. It should stay
faithful to the problem statement and avoid doing solver-level reasoning.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FormulaRole = Literal["condition", "goal", "option", "derived_candidate", "unknown"]
QuestionType = Literal[
    "solve",
    "prove",
    "compute",
    "simplify",
    "derive",
    "integrate",
    "limit",
    "select",
    "explain",
    "unknown",
]
TargetType = QuestionType
VisualObjectType = Literal[
    "point",
    "line",
    "segment",
    "ray",
    "angle",
    "triangle",
    "polygon",
    "circle",
    "arc",
    "curve",
    "axis",
    "table",
    "chart",
    "diagram",
    "other",
    "unknown",
]
LayoutBlockType = Literal["text", "formula", "figure", "table", "option", "other", "unknown"]
UncertaintySeverity = Literal["low", "medium", "high"]


class ProblemType(str, Enum):
    """High-level math problem type recognized from the image."""

    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    PROBABILITY = "probability"
    LINEAR_ALGEBRA = "linear_algebra"
    COMBINATORICS = "combinatorics"
    NUMBER_THEORY = "number_theory"
    WORD_PROBLEM = "word_problem"
    PROOF = "proof"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SchemaModel(BaseModel):
    """Base model with strict field names for stable intermediate artifacts."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class BoundingBox(SchemaModel):
    """A rectangular region in image coordinates."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SourceImage(SchemaModel):
    """Source image metadata. The path is recorded but not checked on disk."""

    image_path: str
    page_index: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @field_validator("image_path")
    @classmethod
    def image_path_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Condition(SchemaModel):
    """A given condition from the problem statement."""

    id: str
    text: str
    linked_formula_ids: list[str] = Field(default_factory=list)

    @field_validator("id", "text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Question(SchemaModel):
    """A question or prompt to be answered."""

    id: str
    text: str
    question_type: QuestionType = "unknown"
    target_ids: list[str] = Field(default_factory=list)

    @field_validator("id", "text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Formula(SchemaModel):
    """A formula recognized from the image."""

    id: str
    raw_text: str
    latex: str | None = None
    role: FormulaRole = "unknown"

    @field_validator("id", "raw_text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Variable(SchemaModel):
    """A mathematical variable mentioned in the problem."""

    symbol: str
    description: str | None = None
    domain: str | None = None

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class VisualObject(SchemaModel):
    """A visible mathematical object such as a point, segment, axis, or diagram."""

    id: str
    object_type: VisualObjectType = "unknown"
    description: str | None = None
    bbox: BoundingBox | None = None

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class LayoutBlock(SchemaModel):
    """A block in the image layout."""

    id: str
    block_type: LayoutBlockType = "unknown"
    text: str | None = None
    bbox: BoundingBox | None = None

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class LayoutInfo(SchemaModel):
    """Layout information for the parsed problem image."""

    blocks: list[LayoutBlock] = Field(default_factory=list)


class Target(SchemaModel):
    """A structured target requested by a question."""

    id: str
    text: str
    target_type: TargetType = "unknown"

    @field_validator("id", "text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class ConfidenceInfo(SchemaModel):
    """Confidence scores for the parsed problem representation."""

    overall: float = Field(ge=0.0, le=1.0)
    text: float | None = Field(default=None, ge=0.0, le=1.0)
    formula: float | None = Field(default=None, ge=0.0, le=1.0)
    layout: float | None = Field(default=None, ge=0.0, le=1.0)


class Uncertainty(SchemaModel):
    """A recorded uncertainty about a specific SPR field."""

    field_path: str
    message: str
    severity: UncertaintySeverity = "medium"

    @field_validator("field_path", "message")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class SPR(SchemaModel):
    """Structured Problem Representation, minimal version 1.0."""

    schema_version: str = "SPR-1.0"
    source_image: SourceImage
    problem_text: str
    problem_stem: str | None = None
    conditions: list[Condition] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    visual_objects: list[VisualObject] = Field(default_factory=list)
    layout: LayoutInfo | None = None
    problem_type: ProblemType = ProblemType.UNKNOWN
    knowledge_units: list[str] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)
    confidence: ConfidenceInfo
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_text")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_spr_dict(data: dict[str, Any]) -> SPR:
    """Validate a dictionary and return an SPR object."""

    return SPR.model_validate(data)
