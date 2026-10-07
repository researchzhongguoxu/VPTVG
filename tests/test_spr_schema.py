import json

import pytest
from pydantic import ValidationError

from mathexplain.schemas.spr import ProblemType, SPR, validate_spr_dict


def minimal_spr_data() -> dict:
    return {
        "source_image": {
            "image_path": "data/inputs/example.png",
            "width": 1200,
            "height": 800,
        },
        "problem_text": "Solve x + 2 = 5.",
        "confidence": {
            "overall": 0.92,
            "text": 0.95,
            "formula": 0.9,
            "layout": 0.85,
        },
    }


def algebra_spr_data() -> dict:
    data = minimal_spr_data()
    data.update(
        {
            "problem_text": "Given x + 2 = 5, solve for x.",
            "problem_stem": "Given x + 2 = 5",
            "conditions": [
                {
                    "id": "cond_1",
                    "text": "x + 2 = 5",
                    "linked_formula_ids": ["formula_1"],
                }
            ],
            "questions": [
                {
                    "id": "question_1",
                    "text": "solve for x",
                    "question_type": "solve",
                    "target_ids": ["target_1"],
                }
            ],
            "formulas": [
                {
                    "id": "formula_1",
                    "raw_text": "x + 2 = 5",
                    "latex": "x + 2 = 5",
                    "role": "condition",
                }
            ],
            "variables": [
                {
                    "symbol": "x",
                    "description": "unknown value",
                    "domain": "real",
                }
            ],
            "visual_objects": [
                {
                    "id": "axis_1",
                    "object_type": "axis",
                    "description": "number line",
                    "bbox": {
                        "x": 10,
                        "y": 20,
                        "width": 300,
                        "height": 60,
                    },
                }
            ],
            "layout": {
                "blocks": [
                    {
                        "id": "block_1",
                        "block_type": "text",
                        "text": "Given x + 2 = 5, solve for x.",
                    }
                ]
            },
            "problem_type": "algebra",
            "knowledge_units": ["linear equations"],
            "targets": [
                {
                    "id": "target_1",
                    "text": "x",
                    "target_type": "solve",
                }
            ],
            "uncertainties": [
                {
                    "field_path": "visual_objects[0].description",
                    "message": "The number line may be decorative.",
                    "severity": "low",
                }
            ],
            "metadata": {
                "parser_model": "mock-parser",
                "prompt_version": "spr-v1",
                "created_at": "2026-05-18T13:00:00+08:00",
            },
        }
    )
    return data


def test_validate_minimal_spr_dict() -> None:
    spr = validate_spr_dict(minimal_spr_data())

    assert isinstance(spr, SPR)
    assert spr.schema_version == "SPR-1.0"
    assert spr.source_image.image_path == "data/inputs/example.png"
    assert spr.problem_text == "Solve x + 2 = 5."
    assert spr.conditions == []
    assert spr.metadata == {}
    assert spr.problem_type == ProblemType.UNKNOWN


def test_validate_algebra_spr_dict() -> None:
    spr = validate_spr_dict(algebra_spr_data())

    assert spr.problem_type == ProblemType.ALGEBRA
    assert spr.conditions[0].id == "cond_1"
    assert spr.questions[0].question_type == "solve"
    assert spr.formulas[0].role == "condition"
    assert spr.variables[0].symbol == "x"
    assert spr.visual_objects[0].object_type == "axis"
    assert spr.targets[0].target_type == "solve"


def test_spr_serializes_to_dict_and_json() -> None:
    spr = validate_spr_dict(algebra_spr_data())

    dumped = spr.model_dump()
    dumped_json = spr.model_dump_json(indent=2)

    assert dumped["schema_version"] == "SPR-1.0"
    assert json.loads(dumped_json)["problem_text"] == "Given x + 2 = 5, solve for x."


def test_missing_source_image_fails_validation() -> None:
    data = minimal_spr_data()
    data.pop("source_image")

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_confidence_above_one_fails_validation() -> None:
    data = minimal_spr_data()
    data["confidence"]["overall"] = 1.1

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_empty_condition_id_fails_validation() -> None:
    data = algebra_spr_data()
    data["conditions"][0]["id"] = "   "

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_empty_required_strings_fail_validation() -> None:
    for field_path in ("problem_text", "source_image.image_path"):
        data = minimal_spr_data()
        if field_path == "problem_text":
            data["problem_text"] = ""
        else:
            data["source_image"]["image_path"] = " "

        with pytest.raises(ValidationError):
            validate_spr_dict(data)


def test_invalid_formula_role_fails_validation() -> None:
    data = algebra_spr_data()
    data["formulas"][0]["role"] = "given"

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_invalid_question_type_fails_validation() -> None:
    data = algebra_spr_data()
    data["questions"][0]["question_type"] = "differentiate"

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_invalid_visual_object_type_fails_validation() -> None:
    data = algebra_spr_data()
    data["visual_objects"][0]["object_type"] = "cube"

    with pytest.raises(ValidationError):
        validate_spr_dict(data)


def test_problem_type_supports_new_categories() -> None:
    for problem_type in (
        "probability",
        "linear_algebra",
        "combinatorics",
        "proof",
        "mixed",
    ):
        data = minimal_spr_data()
        data["problem_type"] = problem_type

        spr = validate_spr_dict(data)

        assert spr.problem_type.value == problem_type


def test_target_type_supports_extended_actions() -> None:
    for target_type in ("derive", "integrate", "limit", "select"):
        data = minimal_spr_data()
        data["targets"] = [
            {
                "id": f"target_{target_type}",
                "text": f"{target_type} the expression",
                "target_type": target_type,
            }
        ]

        spr = validate_spr_dict(data)

        assert spr.targets[0].target_type == target_type
