from mathexplain.schemas.explanation import ExplanationScript, validate_explanation_script_dict


def test_explanation_script_schema_round_trips() -> None:
    script = ExplanationScript.model_validate(
        {
            "problem_id": "problem_1",
            "source_chain_id": "chain_main",
            "status": "passed",
            "audience_level": "standard",
            "visual_anchors": [
                {
                    "anchor_id": "var_x",
                    "kind": "variable",
                    "label": "x",
                    "expression": "x",
                }
            ],
            "segments": [
                {
                    "segment_id": "seg_1",
                    "segment_type": "variable_definition",
                    "linked_step_ids": ["step_1"],
                    "narration": "设 x 表示未知数。",
                    "math_expressions": ["x"],
                    "visual_refs": ["var_x"],
                    "pedagogical_actions": ["DEFINE_VARIABLE"],
                    "pacing": "normal",
                    "estimated_duration_ms": 3000,
                    "pause_after_ms": 200,
                }
            ],
            "consistency_report": {
                "valid": True,
                "status": "passed",
                "checked_segment_count": 1,
            },
        }
    )

    validated = validate_explanation_script_dict(script.model_dump(mode="json"))

    assert validated.problem_id == "problem_1"
    assert validated.segments[0].visual_refs == ["var_x"]
    assert "seg_1" in validated.model_dump_json()
