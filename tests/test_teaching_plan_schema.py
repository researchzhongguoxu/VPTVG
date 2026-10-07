import pytest
from pydantic import ValidationError

from mathexplain.schemas.teaching_plan import (
    TeachingPlan,
    validate_teaching_plan_dict,
)


def minimal_teaching_plan_data() -> dict:
    return {
        "problem_id": "problem_1",
        "source_chain_id": "chain_main",
        "planner_mode": "rule_based",
        "moves": [
            {
                "move_id": "move_1",
                "move_type": "modeling",
                "source_step_ids": ["step_1"],
                "source_sentence": "题目说还剩 18 千克。",
                "meaning": "最终剩余量",
                "raw_expression": "x/4 + 4 = 18",
                "display_expression": "x ÷ 4 + 4 = 18",
                "narration": "题目说最后还剩 18 千克，所以建立方程。",
                "pedagogical_actions": ["RENDER_EXPRESSION"],
                "estimated_duration_ms": 5000,
                "pause_after_ms": 600,
            }
        ],
        "quality_report": {
            "valid": True,
            "status": "passed",
            "checked_move_count": 1,
            "expanded_step_ids": ["step_1"],
            "strategy_names": ["linear_word_problem"],
        },
    }


def test_minimal_teaching_plan_validates() -> None:
    plan = validate_teaching_plan_dict(minimal_teaching_plan_data())

    assert isinstance(plan, TeachingPlan)
    assert plan.schema_version == "TEACHING_PLAN-1.0"
    assert plan.moves[0].display_expression == "x ÷ 4 + 4 = 18"


def test_teaching_plan_rejects_empty_required_strings() -> None:
    data = minimal_teaching_plan_data()
    data["problem_id"] = " "

    with pytest.raises(ValidationError):
        validate_teaching_plan_dict(data)

    data = minimal_teaching_plan_data()
    data["moves"][0]["move_id"] = ""
    with pytest.raises(ValidationError):
        validate_teaching_plan_dict(data)

    data = minimal_teaching_plan_data()
    data["moves"][0]["narration"] = ""
    with pytest.raises(ValidationError):
        validate_teaching_plan_dict(data)


def test_teaching_plan_rejects_unknown_fields() -> None:
    data = minimal_teaching_plan_data()
    data["unexpected"] = True

    with pytest.raises(ValidationError):
        validate_teaching_plan_dict(data)


def test_teaching_plan_json_roundtrip() -> None:
    plan = validate_teaching_plan_dict(minimal_teaching_plan_data())

    assert validate_teaching_plan_dict(plan.model_dump(mode="json")) == plan


def test_teaching_plan_defaults_are_not_shared() -> None:
    first = validate_teaching_plan_dict(minimal_teaching_plan_data())
    second = validate_teaching_plan_dict(minimal_teaching_plan_data())

    first.moves[0].metadata["x"] = 1
    first.quality_report.warnings.append("warn")

    assert second.moves[0].metadata == {}
    assert second.quality_report.warnings == []
