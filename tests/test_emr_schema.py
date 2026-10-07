import json

import pytest
from pydantic import ValidationError

from mathexplain.schemas.emr import EMR, ProblemType, validate_emr_dict


def minimal_emr_data() -> dict:
    return {
        "problem_id": "problem_001",
    }


def algebra_emr_data() -> dict:
    return {
        "problem_id": "problem_001",
        "source_spr_version": "SPR-1.0",
        "source_spr_id": "spr_001",
        "source_problem_type": "algebra",
        "representation_type": "equation_system",
        "variables": [
            {
                "id": "var_x",
                "symbol": "x",
                "domain": "real",
                "description": "unknown value",
            }
        ],
        "expressions": [
            {
                "id": "expr_1",
                "sympy": "x + 2",
                "raw_text": "x + 2",
                "latex": "x + 2",
                "source_formula_id": "formula_1",
            }
        ],
        "equations": [
            {
                "id": "eq_1",
                "lhs_sympy": "x + 2",
                "rhs_sympy": "5",
                "relation": "eq",
                "source_condition_ids": ["cond_1"],
                "source_formula_ids": ["formula_1"],
            }
        ],
        "constraints": [
            {
                "id": "constraint_1",
                "expression_sympy": "x",
                "constraint_type": "domain",
                "description": "x is real",
            }
        ],
        "goals": [
            {
                "id": "goal_1",
                "goal_type": "solve",
                "target": "x",
                "target_variable_ids": ["var_x"],
                "description": "solve for x",
            }
        ],
        "geometry_objects": [
            {
                "id": "axis_1",
                "object_type": "axis",
                "name": "number line",
                "properties": {
                    "orientation": "horizontal",
                },
            }
        ],
        "metadata": {
            "builder_version": "emr-builder-v1",
            "source_spr_id": "spr_001",
            "created_at": "2026-05-18T13:30:00+08:00",
        },
    }


def test_validate_minimal_emr_dict() -> None:
    emr = validate_emr_dict(minimal_emr_data())

    assert isinstance(emr, EMR)
    assert emr.schema_version == "EMR-1.0"
    assert emr.problem_id == "problem_001"
    assert emr.representation_type == "unknown"
    assert emr.variables == []
    assert emr.metadata == {}


def test_validate_algebra_equation_system_emr_dict() -> None:
    emr = validate_emr_dict(algebra_emr_data())

    assert emr.problem_id == "problem_001"
    assert emr.source_spr_version == "SPR-1.0"
    assert emr.source_spr_id == "spr_001"
    assert emr.source_problem_type == ProblemType.ALGEBRA
    assert emr.representation_type == "equation_system"
    assert emr.variables[0].symbol == "x"
    assert emr.expressions[0].sympy == "x + 2"
    assert emr.equations[0].lhs_sympy == "x + 2"
    assert emr.equations[0].rhs_sympy == "5"
    assert emr.equations[0].source_condition_ids == ["cond_1"]
    assert emr.equations[0].source_formula_ids == ["formula_1"]
    assert emr.goals[0].target == "x"


def test_math_expression_requires_sympy_but_not_raw_text_or_latex() -> None:
    data = algebra_emr_data()
    data["expressions"][0].pop("raw_text")
    data["expressions"][0].pop("latex")

    emr = validate_emr_dict(data)

    assert emr.expressions[0].sympy == "x + 2"
    assert emr.expressions[0].raw_text is None
    assert emr.expressions[0].latex is None


def test_emr_serializes_to_dict_and_json() -> None:
    emr = validate_emr_dict(algebra_emr_data())

    dumped = emr.model_dump()
    dumped_json = emr.model_dump_json(indent=2)

    assert dumped["schema_version"] == "EMR-1.0"
    assert json.loads(dumped_json)["problem_id"] == "problem_001"


def test_missing_problem_id_fails_validation() -> None:
    data = minimal_emr_data()
    data.pop("problem_id")

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


@pytest.mark.parametrize(
    ("path", "empty_value"),
    [
        ("problem_id", " "),
        ("variables.0.id", ""),
        ("variables.0.symbol", " "),
        ("expressions.0.id", ""),
        ("expressions.0.sympy", " "),
        ("equations.0.id", ""),
        ("equations.0.lhs_sympy", " "),
        ("equations.0.rhs_sympy", ""),
        ("constraints.0.id", " "),
        ("constraints.0.expression_sympy", ""),
        ("goals.0.id", " "),
        ("geometry_objects.0.id", ""),
    ],
)
def test_empty_key_strings_fail_validation(path: str, empty_value: str) -> None:
    data = algebra_emr_data()
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = empty_value

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_representation_type_fails_validation() -> None:
    data = algebra_emr_data()
    data["representation_type"] = "polynomial_system"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_domain_fails_validation() -> None:
    data = algebra_emr_data()
    data["variables"][0]["domain"] = "boolean"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_relation_fails_validation() -> None:
    data = algebra_emr_data()
    data["equations"][0]["relation"] = "approximately_equal"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_goal_type_fails_validation() -> None:
    data = algebra_emr_data()
    data["goals"][0]["goal_type"] = "differentiate"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_constraint_type_fails_validation() -> None:
    data = algebra_emr_data()
    data["constraints"][0]["constraint_type"] = "range_check"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_invalid_geometry_object_type_fails_validation() -> None:
    data = algebra_emr_data()
    data["geometry_objects"][0]["object_type"] = "cube"

    with pytest.raises(ValidationError):
        validate_emr_dict(data)


def test_metadata_supports_extensions() -> None:
    emr = validate_emr_dict(algebra_emr_data())

    assert emr.metadata["builder_version"] == "emr-builder-v1"
    assert emr.metadata["source_spr_id"] == "spr_001"
    assert emr.metadata["created_at"] == "2026-05-18T13:30:00+08:00"


def test_geometry_objects_basic_structure() -> None:
    data = minimal_emr_data()
    data["representation_type"] = "geometry_constraint_graph"
    data["geometry_objects"] = [
        {
            "id": "circle_1",
            "object_type": "circle",
            "name": "Circle O",
            "properties": {
                "center": "O",
                "radius": "r",
            },
        }
    ]

    emr = validate_emr_dict(data)

    assert emr.geometry_objects[0].object_type == "circle"
    assert emr.geometry_objects[0].properties["center"] == "O"
