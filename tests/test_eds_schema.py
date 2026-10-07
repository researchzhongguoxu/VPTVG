import pytest
from pydantic import ValidationError

from mathexplain.schemas.eds import (
    ExecutableDirectorScript,
    validate_executable_director_script_dict,
)


def minimal_eds_data() -> dict:
    return {
        "problem_id": "problem_1",
        "source_explanation_id": "explanation_1",
        "source_chain_id": "chain_main",
        "scenes": [
            {
                "scene_id": "scene_1",
                "scene_type": "modeling",
                "source_segment_ids": ["seg_1"],
                "title": "建立方程",
                "start_ms": 0,
                "duration_ms": 3000,
            }
        ],
        "timeline": [
            {
                "timeline_id": "timeline_1",
                "scene_id": "scene_1",
                "track_type": "narration",
                "target_id": "nar_1",
                "start_ms": 0,
                "duration_ms": 3000,
                "action": "play",
            }
        ],
        "assets": [
            {
                "asset_id": "asset_problem_text",
                "asset_type": "problem_text",
            }
        ],
        "narration_tracks": [
            {
                "narration_id": "nar_1",
                "scene_id": "scene_1",
                "text": "根据题意建立方程。",
                "start_ms": 0,
                "duration_ms": 3000,
                "pause_after_ms": 200,
                "sync_anchor_ids": ["anchor_1"],
            }
        ],
        "formula_tracks": [
            {
                "formula_id": "formula_1",
                "scene_id": "scene_1",
                "expression": "x + 1 = 2",
                "display_mode": "block",
                "source_step_ids": ["step_1"],
                "start_ms": 600,
                "duration_ms": 1600,
                "layout_slot": "main_formula",
                "sync_anchor_ids": ["anchor_1"],
            }
        ],
        "visual_actions": [
            {
                "action_id": "action_1",
                "scene_id": "scene_1",
                "action_type": "highlight",
                "target_id": "formula_1",
                "start_ms": 800,
                "duration_ms": 900,
            }
        ],
        "sync_anchors": [
            {
                "anchor_id": "anchor_1",
                "scene_id": "scene_1",
                "source_ref": "seg_1",
                "time_ms": 600,
                "linked_narration_id": "nar_1",
                "linked_formula_ids": ["formula_1"],
                "linked_action_ids": ["action_1"],
            }
        ],
        "quality_report": {
            "valid": True,
            "status": "passed",
            "checked_scene_count": 1,
        },
    }


def test_eds_schema_accepts_minimal_valid_script() -> None:
    eds = validate_executable_director_script_dict(minimal_eds_data())

    assert eds.schema_version == "EDS-1.0"
    assert eds.problem_id == "problem_1"
    assert eds.render_hints.canvas_width == 1920
    assert eds.render_hints.canvas_height == 1080
    assert eds.render_hints.fps == 15
    assert eds.render_hints.language == "zh-CN"
    assert eds.visual_actions[0].action_type == "highlight"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("problem_id",), " "),
        (("scenes", 0, "scene_id"), ""),
        (("narration_tracks", 0, "text"), " "),
    ],
)
def test_eds_schema_rejects_empty_required_strings(path: tuple, value: str) -> None:
    data = minimal_eds_data()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        validate_executable_director_script_dict(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scenes", 0, "start_ms"), -1),
        (("timeline", 0, "duration_ms"), -1),
        (("narration_tracks", 0, "pause_after_ms"), -1),
        (("formula_tracks", 0, "start_ms"), -1),
        (("visual_actions", 0, "duration_ms"), -1),
        (("sync_anchors", 0, "time_ms"), -1),
    ],
)
def test_eds_schema_rejects_negative_timing_values(path: tuple, value: int) -> None:
    data = minimal_eds_data()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        validate_executable_director_script_dict(data)


def test_eds_schema_rejects_unknown_fields() -> None:
    data = minimal_eds_data()
    data["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        validate_executable_director_script_dict(data)


def test_eds_schema_default_collections_are_not_shared() -> None:
    first = ExecutableDirectorScript.model_validate(
        {
            "problem_id": "problem_1",
            "source_explanation_id": "explanation_1",
            "source_chain_id": "chain_main",
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_scene_count": 0,
            },
        }
    )
    second = ExecutableDirectorScript.model_validate(
        {
            "problem_id": "problem_2",
            "source_explanation_id": "explanation_2",
            "source_chain_id": "chain_main",
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_scene_count": 0,
            },
        }
    )

    first.scenes.append(
        {
            "scene_id": "scene_1",
            "scene_type": "answer",
            "start_ms": 0,
            "duration_ms": 1000,
        }
    )
    first.metadata["changed"] = True

    assert second.scenes == []
    assert second.metadata == {}


def test_eds_schema_json_round_trip_is_stable() -> None:
    eds = validate_executable_director_script_dict(minimal_eds_data())
    revalidated = validate_executable_director_script_dict(eds.model_dump(mode="json"))

    assert revalidated == eds
    assert "EDS-1.0" in revalidated.model_dump_json()
