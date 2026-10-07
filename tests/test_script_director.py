from mathexplain.agents.script_director import ScriptDirector
from mathexplain.schemas.eds import (
    ExecutableDirectorScript,
    validate_executable_director_script_dict,
)
from mathexplain.schemas.explanation import ExplanationScript
from mathexplain.schemas.scs import SolutionChain


class DummySPR:
    problem_text = "水果店运来一批苹果。"
    problem_type = "word_problem"


def math10_style_explanation() -> ExplanationScript:
    return ExplanationScript.model_validate(
        {
            "problem_id": "problem_math10",
            "source_chain_id": "chain_main",
            "status": "passed",
            "audience_level": "standard",
            "segments": [
                {
                    "segment_id": "seg_model_chain_main_step_4",
                    "segment_type": "modeling",
                    "linked_step_ids": ["chain_main_step_4"],
                    "narration": "根据题意建立方程。对应的式子是 x/4 + 4 = 18。",
                    "math_expressions": ["x/4 + 4 = 18"],
                    "visual_refs": ["expr_chain_main_step_4"],
                    "pedagogical_actions": [
                        "RENDER_EXPRESSION",
                        "HIGHLIGHT_TERMS",
                        "QUESTION_PAUSE",
                    ],
                    "pacing": "slow",
                    "estimated_duration_ms": 6200,
                    "pause_after_ms": 600,
                    "tts_hints": {
                        "pause_points": ["对应的式子是"],
                        "emphasis_terms": ["x/4 + 4 = 18"],
                    },
                    "layout_hints": {
                        "focus_area": "main_formula",
                        "keep_previous_context": True,
                    },
                    "metadata": {
                        "source_expression": "x/4 + 4 = 18",
                        "source_sentence": "这时还剩18千克苹果",
                        "meaning": "最终剩余量建立方程",
                    },
                },
                {
                    "segment_id": "seg_answer_1",
                    "segment_type": "answer",
                    "linked_step_ids": ["chain_main_step_10"],
                    "narration": "第一个问题，原来运来的苹果重量是 56 千克。",
                    "math_expressions": ["x = 56 千克"],
                    "visual_refs": ["answer_1"],
                    "pedagogical_actions": [
                        "FINAL_ANSWER_REVEAL",
                        "HIGHLIGHT_RESULT",
                    ],
                    "pacing": "normal",
                    "estimated_duration_ms": 4300,
                    "pause_after_ms": 500,
                    "layout_hints": {"focus_area": "final_answer"},
                },
            ],
            "visual_anchors": [
                {
                    "anchor_id": "expr_chain_main_step_4",
                    "kind": "expression",
                    "label": "x/4 + 4 = 18",
                    "source_step_id": "chain_main_step_4",
                    "expression": "x/4 + 4 = 18",
                },
                {
                    "anchor_id": "answer_1",
                    "kind": "answer",
                    "label": "x = 56 千克",
                    "expression": "x = 56 千克",
                },
            ],
            "consistency_report": {
                "valid": True,
                "status": "passed",
                "checked_segment_count": 2,
            },
        }
    )


def test_script_director_generates_valid_eds_from_passed_explanation() -> None:
    eds = ScriptDirector().direct(math10_style_explanation())

    assert isinstance(eds, ExecutableDirectorScript)
    assert eds.problem_id == "problem_math10"
    assert eds.source_explanation_id == "explanation_problem_math10_chain_main"
    assert eds.quality_report.status == "passed"
    assert len(eds.scenes) == 2
    assert len(eds.narration_tracks) == 2
    assert len(eds.formula_tracks) == 2
    assert len(eds.sync_anchors) == 2
    assert eds.scenes[1].start_ms == eds.scenes[0].duration_ms
    assert validate_executable_director_script_dict(eds.model_dump(mode="json")) == eds


def test_script_director_sets_english_render_language_and_titles() -> None:
    eds = ScriptDirector().direct(math10_style_explanation(), output_language="en-US")

    assert eds.render_hints.language == "en-US"
    assert eds.scenes[0].title == "Set Up"
    assert eds.scenes[1].title == "Final Answer"
    assert eds.metadata["output_language"] == "en-US"


def test_script_director_adds_back_substitution_verification_for_solve_for_answer() -> None:
    explanation = ExplanationScript.model_validate(
        {
            "problem_id": "apple_pie",
            "source_chain_id": "chain_main",
            "status": "passed",
            "audience_level": "standard",
            "segments": [
                {
                    "segment_id": "seg_answer",
                    "segment_type": "answer",
                    "linked_step_ids": ["chain_main_step_9"],
                    "narration": "Question 1: the apple pie cost is 7 dollars.",
                    "math_expressions": ["the apple pie cost = 7 dollars"],
                    "visual_refs": ["answer_1"],
                    "pedagogical_actions": ["FINAL_ANSWER_REVEAL"],
                    "pacing": "normal",
                    "estimated_duration_ms": 3000,
                    "pause_after_ms": 0,
                }
            ],
            "visual_anchors": [],
            "consistency_report": {"valid": True, "status": "passed", "checked_segment_count": 1},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "apple_pie",
            "chain_id": "chain_main",
            "final_answer": "x = 7",
            "confidence": 1.0,
            "steps": [
                {
                    "step_id": "chain_main_step_8",
                    "step_index": 8,
                    "description": "CAS solve",
                    "math_expression": "x = 7",
                    "rule_name": "cas_solve_for",
                    "diagnostics": {
                        "operation": "solve_for",
                        "tool_call_id": "solve_x",
                        "input": {
                            "equations": [
                                {
                                    "lhs_sympy": "2*2 + 8 + 1 + x",
                                    "rhs_sympy": "20",
                                }
                            ],
                            "target": "x",
                        },
                    },
                }
            ],
            "metadata": {
                "final_answer_items": [
                    {
                        "target": "x",
                        "value": "7",
                        "source_tool_call_id": "solve_x",
                        "backend_operation": "solve_for",
                        "backend_confirmed": True,
                    }
                ]
            },
        }
    )

    eds = ScriptDirector().direct(explanation, scs=scs, output_language="en-US")
    answer_formula = eds.formula_tracks[0]

    assert answer_formula.metadata["verification_expression"] == "2 × 2 + 8 + 1 + 7 = 20"
    assert answer_formula.metadata["verification_expression"] != "x = 7"


def test_script_director_substitutes_evaluate_symbols_in_answer_verification() -> None:
    explanation = ExplanationScript.model_validate(
        {
            "problem_id": "restaurant",
            "source_chain_id": "chain_main",
            "segments": [
                {
                    "segment_id": "seg_answer",
                    "segment_type": "answer",
                    "linked_step_ids": ["chain_main_step_2"],
                    "narration": "The total number of people is 320.",
                    "math_expressions": ["the total number of people = 320"],
                    "visual_refs": ["answer"],
                    "pedagogical_actions": ["FINAL_ANSWER_REVEAL"],
                    "pacing": "normal",
                    "estimated_duration_ms": 3000,
                }
            ],
            "status": "passed",
            "consistency_report": {"valid": True, "status": "passed", "checked_segment_count": 1},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "restaurant",
            "chain_id": "chain_main",
            "final_answer": "答案是 320",
            "confidence": 1.0,
            "steps": [
                {
                    "step_id": "chain_main_step_2",
                    "step_index": 2,
                    "description": "CAS evaluate",
                    "math_expression": "50 + 40 + lunch + evening",
                    "rule_name": "cas_evaluate",
                    "diagnostics": {
                        "operation": "evaluate",
                        "tool_call_id": "calc_total",
                        "input": {
                            "expression": "50 + 40 + lunch + evening",
                            "substitutions": {"lunch": "80", "evening": "150"},
                            "target": "total",
                        },
                    },
                }
            ],
            "metadata": {
                "final_answer_items": [
                    {
                        "target": "total",
                        "value": "320",
                        "source_tool_call_id": "calc_total",
                        "backend_operation": "evaluate",
                        "backend_confirmed": True,
                    }
                ]
            },
        }
    )

    eds = ScriptDirector().direct(explanation, scs=scs, output_language="en-US")
    verification = eds.formula_tracks[0].metadata["verification_expression"]

    assert verification == "50 + 40 + 80 + 150 = 320"
    assert "lunch" not in verification
    assert "evening" not in verification


def test_script_director_cleans_legacy_operator_encoding_in_verification() -> None:
    assert ScriptDirector._display_math("2 ¡Á 2 + 8 = 12") == "2 × 2 + 8 = 12"
    assert ScriptDirector._display_math("18 ¡Â 3 = 6") == "18 ÷ 3 = 6"


def test_script_director_maps_pedagogical_actions_to_visual_actions() -> None:
    eds = ScriptDirector().direct(math10_style_explanation())
    action_types = {action.action_type for action in eds.visual_actions}
    timeline_track_types = {item.track_type for item in eds.timeline}

    assert "final_answer_reveal" in action_types
    assert "reveal" in action_types
    assert "highlight" in action_types
    assert "pause" in action_types
    assert "pause" in timeline_track_types
    assert any(action.target_id.startswith("formula_") for action in eds.visual_actions)
    assert all(action.duration_ms > 0 for action in eds.visual_actions if action.action_type == "pause")


def test_script_director_preserves_traceability_metadata() -> None:
    eds = ScriptDirector().direct(math10_style_explanation())
    first_scene = eds.scenes[0]
    first_action = eds.visual_actions[0]
    first_formula = eds.formula_tracks[0]

    assert first_scene.source_segment_ids == ["seg_model_chain_main_step_4"]
    assert first_scene.metadata["linked_step_ids"] == ["chain_main_step_4"]
    assert first_scene.metadata["visual_refs"] == ["expr_chain_main_step_4"]
    assert first_action.metadata["source_pedagogical_action"] == "RENDER_EXPRESSION"
    assert first_formula.source_step_ids == ["chain_main_step_4"]
    assert first_formula.metadata["source_expression"] == "x/4 + 4 = 18"
    assert first_formula.metadata["source_sentence"] == "这时还剩18千克苹果"
    assert first_formula.metadata["meaning"] == "最终剩余量建立方程"


def test_script_director_marks_answer_card_and_formula_buildup_metadata() -> None:
    eds = ScriptDirector().direct(math10_style_explanation(), output_language="en-US")
    modeling_formula = next(item for item in eds.formula_tracks if item.scene_id == eds.scenes[0].scene_id)
    answer_formula = next(item for item in eds.formula_tracks if item.display_mode == "final_answer")
    answer_scene = next(scene for scene in eds.scenes if scene.scene_type == "answer")

    assert answer_formula.style["answer_card"] is True
    assert answer_formula.style["verified"] is True
    assert answer_formula.metadata["answer_card"] is True
    assert answer_formula.metadata["verified"] is True
    assert answer_scene.metadata["answer_card"] is True
    assert answer_scene.metadata["verified"] is True
    assert modeling_formula.metadata["buildup_mode"] == "sequential_terms"
    assert len(modeling_formula.metadata["buildup_parts"]) >= 3
    assert modeling_formula.metadata["buildup_interval_ms"] == 450


def test_script_director_uses_unique_asset_ids_with_spr_problem_text() -> None:
    explanation = math10_style_explanation()
    explanation_data = explanation.model_dump(mode="json")
    explanation_data["visual_anchors"].append(
        {
            "anchor_id": "problem_text",
            "kind": "problem_text",
            "label": "题干",
        }
    )
    explanation = ExplanationScript.model_validate(explanation_data)

    eds = ScriptDirector().direct(explanation, spr=DummySPR())
    asset_ids = [asset.asset_id for asset in eds.assets]

    assert len(asset_ids) == len(set(asset_ids))
    assert "asset_source_problem_text" in asset_ids
    assert "asset_problem_text" in asset_ids
    assert eds.quality_report.status == "passed"


def test_script_director_cleans_dataset_wrapper_from_problem_asset() -> None:
    class DatasetWrappedSPR:
        problem_text = (
            "GSM8K Math Word Problem\n"
            "sample_id: gsm8k_test_000509\n"
            "Sophia bought 3 cookies and then 1 more cookie. How many cookies did Sophia buy?"
        )
        problem_type = "word_problem"

    eds = ScriptDirector().direct(math10_style_explanation(), spr=DatasetWrappedSPR())
    problem_asset = next(asset for asset in eds.assets if asset.asset_id == "asset_source_problem_text")

    assert problem_asset.metadata["problem_text"] == (
        "Sophia bought 3 cookies and then 1 more cookie. How many cookies did Sophia buy?"
    )


def test_script_director_delays_highlight_after_reveal() -> None:
    eds = ScriptDirector().direct(math10_style_explanation())
    reveal = next(
        action
        for action in eds.visual_actions
        if action.metadata.get("source_pedagogical_action") == "RENDER_EXPRESSION"
    )
    highlight = next(
        action
        for action in eds.visual_actions
        if action.metadata.get("source_pedagogical_action") == "HIGHLIGHT_TERMS"
    )

    assert highlight.start_ms > reveal.start_ms


def test_script_director_structure_check_reports_broken_eds_references() -> None:
    errors, warnings = ScriptDirector._eds_structure_issues(
        scenes=[
            {"scene_id": "scene_1", "start_ms": 0, "duration_ms": 1000},
            {"scene_id": "scene_1", "start_ms": 1200, "duration_ms": 1000},
        ],
        timeline=[
            {
                "timeline_id": "timeline_missing",
                "scene_id": "missing_scene",
                "target_id": "missing_target",
            }
        ],
        assets=[
            {"asset_id": "asset_problem_text"},
            {"asset_id": "asset_problem_text"},
        ],
        narration_tracks=[{"narration_id": "nar_1", "scene_id": "scene_1"}],
        formula_tracks=[{"formula_id": "formula_1", "scene_id": "scene_1"}],
        visual_actions=[
            {
                "action_id": "action_1",
                "scene_id": "scene_1",
                "target_id": "missing_visual_target",
                "duration_ms": 0,
            }
        ],
        sync_anchors=[{"anchor_id": "anchor_1", "scene_id": "scene_1"}],
        total_duration_ms=2200,
    )

    error_text = "\n".join(errors)
    warning_text = "\n".join(warnings)

    assert "Duplicate scene_id" in error_text
    assert "Duplicate asset_id" in error_text
    assert "missing scene_id" in error_text
    assert "missing target_id" in error_text
    assert "not continuous" in error_text
    assert "zero duration" in warning_text


def test_script_director_returns_unsupported_eds_for_unsupported_explanation() -> None:
    explanation = ExplanationScript.model_validate(
        {
            "problem_id": "problem_bad",
            "source_chain_id": "chain_main",
            "status": "unsupported",
            "segments": [
                {
                    "segment_id": "seg_unsupported",
                    "segment_type": "unsupported",
                    "narration": "当前解题链没有通过验证，因此不会生成正式教学讲解。",
                    "math_expressions": [],
                    "visual_refs": [],
                    "pedagogical_actions": ["SHOW_UNSUPPORTED_NOTICE"],
                    "estimated_duration_ms": 3500,
                }
            ],
            "consistency_report": {
                "valid": False,
                "status": "unsupported",
                "checked_segment_count": 1,
                "errors": ["Explanation generation requires a passed VerificationReport."],
            },
        }
    )

    eds = ScriptDirector().direct(explanation)

    assert eds.quality_report.status == "unsupported"
    assert eds.quality_report.valid is False
    assert eds.scenes[0].scene_type == "unsupported"
    assert eds.visual_actions[0].action_type == "show"
    assert validate_executable_director_script_dict(eds.model_dump(mode="json"))
