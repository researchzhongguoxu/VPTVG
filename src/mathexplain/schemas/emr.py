"""Executable Mathematical Representation schema.

EMR is the solver-facing representation between SPR and the solving/verification
layers. It stores executable mathematical structure, not solution steps.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RepresentationType = Literal[
    "equation_system",
    "expression",
    "geometry_constraint_graph",
    "proof_task",
    "mixed",
    "unknown",
]
Relation = Literal["eq", "lt", "le", "gt", "ge", "neq"]
GoalType = Literal[
    "solve",
    "compute",
    "simplify",
    "derive",
    "integrate",
    "limit",
    "prove",
    "select",
    "explain",
    "unknown",
]
ConstraintType = Literal["domain", "inequality", "assumption", "condition", "other", "unknown"]
GeometryObjectType = Literal[
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


class ProblemType(str, Enum):
    """High-level problem type carried from SPR/classifier outputs."""

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


class Domain(str, Enum):
    """Variable domain understood by the solver-facing representation."""

    REAL = "real"
    INTEGER = "integer"
    RATIONAL = "rational"
    COMPLEX = "complex"
    NATURAL = "natural"
    POSITIVE_REAL = "positive_real"
    UNKNOWN = "unknown"


class SchemaModel(BaseModel):
    """Base model with stable field names and no implicit extra fields."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class MathVariable(SchemaModel):
    """A mathematical variable available to the solver."""

    id: str
    symbol: str
    domain: Domain = Domain.UNKNOWN
    description: str | None = None

    @field_validator("id", "symbol")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class MathExpression(SchemaModel):
    """A solver-facing mathematical expression."""

    id: str
    sympy: str
    raw_text: str | None = None
    latex: str | None = None
    source_formula_id: str | None = None

    @field_validator("id", "sympy")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Equation(SchemaModel):
    """A relation between two SymPy expressions."""

    id: str
    lhs_sympy: str
    rhs_sympy: str
    relation: Relation = "eq"
    source_condition_ids: list[str] = Field(default_factory=list)
    source_formula_ids: list[str] = Field(default_factory=list)

    @field_validator("id", "lhs_sympy", "rhs_sympy")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Constraint(SchemaModel):
    """A solver-facing constraint or assumption."""

    id: str
    expression_sympy: str
    constraint_type: ConstraintType = "unknown"
    description: str | None = None

    @field_validator("id", "expression_sympy")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class Goal(SchemaModel):
    """A target the solver should pursue."""

    id: str
    goal_type: GoalType = "unknown"
    target: str | None = None
    target_variable_ids: list[str] = Field(default_factory=list)
    target_expression_id: str | None = None
    description: str | None = None

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class GeometryObject(SchemaModel):
    """A geometric or visual-mathematical object reserved for future geometry solving."""

    id: str
    object_type: GeometryObjectType = "unknown"
    name: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class EMR(SchemaModel):
    """Executable Mathematical Representation, minimal version 1.0."""

    schema_version: str = "EMR-1.0"
    problem_id: str
    source_spr_version: str | None = None
    source_spr_id: str | None = None
    source_problem_type: ProblemType | str = ProblemType.UNKNOWN
    representation_type: RepresentationType = "unknown"
    variables: list[MathVariable] = Field(default_factory=list)
    expressions: list[MathExpression] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    geometry_objects: list[GeometryObject] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "problem_id")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_emr_dict(data: dict[str, Any]) -> EMR:
    """Validate a dictionary and return an EMR object."""

    return EMR.model_validate(data)
