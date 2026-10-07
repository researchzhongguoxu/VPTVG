import json

from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.vision_parser import EMRBuilder, SPRGenerator, VisionParser
from mathexplain.schemas.classification import (
    ClassificationResult,
    validate_classification_result_dict,
)
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.spr import SPR


def math3_spr_data() -> dict:
    return {
        "schema_version": "SPR-1.0",
        "source_image": {
            "image_path": "data/inputs/images/math3.jpg",
        },
        "problem_text": (
            "7. 已知正实数 a,b 满足 2a+b=1. 则 (5a+b)/(a^2+ab) 的最小值为 ( )\n"
            "A. 3\nB. 9\nC. 4\nD. 8"
        ),
        "problem_stem": "已知正实数 a,b 满足 2a+b=1. 则 (5a+b)/(a^2+ab) 的最小值为",
        "conditions": [
            {
                "id": "c1",
                "text": "a, b 为正实数",
                "linked_formula_ids": [],
            },
            {
                "id": "c2",
                "text": "2a+b=1",
                "linked_formula_ids": ["f1"],
            },
        ],
        "questions": [
            {
                "id": "q1",
                "text": "求 (5a+b)/(a^2+ab) 的最小值",
                "question_type": "select",
                "target_ids": ["t1"],
            }
        ],
        "formulas": [
            {
                "id": "f1",
                "raw_text": "2a+b=1",
                "latex": "2a+b=1",
                "role": "condition",
            },
            {
                "id": "f2",
                "raw_text": "(5a+b)/(a^2+ab)",
                "latex": "\\frac{5a+b}{a^2+ab}",
                "role": "goal",
            },
            {"id": "f3", "raw_text": "A. 3", "latex": "3", "role": "option"},
            {"id": "f4", "raw_text": "B. 9", "latex": "9", "role": "option"},
            {"id": "f5", "raw_text": "C. 4", "latex": "4", "role": "option"},
            {"id": "f6", "raw_text": "D. 8", "latex": "8", "role": "option"},
        ],
        "variables": [
            {
                "symbol": "a",
                "description": "positive real number",
                "domain": "(0, \\infty)",
            },
            {
                "symbol": "b",
                "description": "positive real number",
                "domain": "(0, \\infty)",
            },
        ],
        "problem_type": "algebra",
        "knowledge_units": ["inequality", "algebraic_expression", "minimum_value"],
        "targets": [
            {
                "id": "t1",
                "text": "最小值",
                "target_type": "select",
            }
        ],
        "confidence": {
            "overall": 0.95,
        },
        "metadata": {
            "problem_id": "problem_math3",
            "parser_mode": "qwen",
        },
    }


def geometry_spr_data() -> dict:
    return {
        "schema_version": "SPR-1.0",
        "source_image": {
            "image_path": "data/inputs/images/math1.png",
        },
        "problem_text": "如图，在 Rt△ABC 中，求证 BC 是 ⊙O 的切线。",
        "problem_stem": "如图，在 Rt△ABC 中，∠B=90°。",
        "conditions": [
            {
                "id": "c1",
                "text": "∠B=90°",
                "linked_formula_ids": ["f1"],
            }
        ],
        "questions": [
            {
                "id": "q1",
                "text": "求证 BC 是 ⊙O 的切线",
                "question_type": "prove",
                "target_ids": ["t1"],
            }
        ],
        "formulas": [
            {
                "id": "f1",
                "raw_text": "∠B=90°",
                "latex": "\\angle B=90^\\circ",
                "role": "condition",
            }
        ],
        "variables": [
            {"symbol": "A", "domain": "point"},
            {"symbol": "B", "domain": "point"},
            {"symbol": "C", "domain": "point"},
        ],
        "problem_type": "geometry",
        "knowledge_units": ["circle_geometry", "tangent_line_theorem"],
        "targets": [
            {
                "id": "t1",
                "text": "BC is tangent to circle O",
                "target_type": "prove",
            }
        ],
        "confidence": {
            "overall": 0.9,
        },
        "metadata": {
            "problem_id": "problem_geometry",
            "parser_mode": "qwen",
        },
    }


def test_problem_classifier_classifies_mock_linear_equation() -> None:
    output = VisionParser().run("any/path.png")

    result = ProblemClassifier().classify(output.spr, output.emr, output.parsing_report)

    assert isinstance(result, ClassificationResult)
    assert result.schema_version == "CLASSIFICATION-1.0"
    assert result.primary_type == "algebra"
    assert result.task_type == "equation_solving"
    assert result.solver_route == "algebra_solver"
    assert result.difficulty == "elementary"
    assert result.solver_config.solver_name == "AlgebraSolver"
    assert result.solver_config.expected_emr_type == "equation_system"
    assert result.requires_solver is True
    assert result.requires_verifier is True
    assert result.warnings == []


def test_problem_classifier_routes_math3_optimization() -> None:
    spr = SPR.model_validate(math3_spr_data())
    emr = EMRBuilder().build("problem_math3", spr)

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "algebra"
    assert result.task_type == "optimization"
    assert result.solver_route == "algebra_optimization_solver"
    assert result.difficulty == "intermediate"
    assert result.solver_config.solver_name == "AlgebraOptimizationSolver"
    assert result.solver_config.task_type == "optimization"
    assert result.solver_config.strategy == "sympy_optimization_planned"
    assert "optimization" in result.knowledge_units
    assert result.metadata["classifier_mode"] == "rule_based"


def test_problem_classifier_routes_geometry_proof_to_stub() -> None:
    spr = SPR.model_validate(geometry_spr_data())
    emr = EMR.model_validate(
        {
            "problem_id": "problem_geometry",
            "source_spr_version": "SPR-1.0",
            "source_spr_id": "problem_geometry",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "geometry"
    assert result.task_type == "proof"
    assert result.solver_route == "geometry_stub"


def test_problem_classifier_routes_coordinate_midpoint_without_diagram_to_algebra_solver() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/midpoint.png"},
            "problem_text": "The midpoint of a segment has coordinates (3, 5). One endpoint is (1, 2). Find the other endpoint.",
            "problem_stem": "Find the missing endpoint using the midpoint coordinate formula.",
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the other endpoint coordinate.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "problem_type": "geometry",
            "knowledge_units": ["coordinate_geometry", "midpoint"],
            "targets": [{"id": "t1", "text": "other endpoint coordinate", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_coordinate_midpoint",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
            "goals": [
                {
                    "id": "g1",
                    "goal_type": "compute",
                    "target": "other_endpoint",
                    "description": "compute midpoint coordinates",
                }
            ],
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "geometry"
    assert result.task_type == "equation_solving"
    assert result.solver_route == "algebra_solver"
    assert result.difficulty == "intermediate"
    assert result.requires_solver is True
    assert result.requires_verifier is True
    assert result.warnings == []


def test_problem_classifier_returns_unknown_for_insufficient_information() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "unknown.png"},
            "problem_text": "Unreadable problem.",
            "problem_type": "unknown",
            "confidence": {"overall": 0.1},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_unknown",
            "source_problem_type": "unknown",
            "representation_type": "unknown",
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "unknown"
    assert result.task_type == "unknown"
    assert result.solver_route == "unknown"
    assert result.difficulty == "unknown"
    assert result.requires_solver is False
    assert result.requires_verifier is False
    assert result.warnings


def test_problem_classifier_routes_linear_word_problem_to_algebra_tool_solver() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math9.png"},
            "problem_text": "A linear equation word problem.",
            "problem_type": "word_problem",
            "knowledge_units": ["linear equations", "arithmetic operations"],
            "variables": [{"symbol": "x", "domain": "integer"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math9",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "word_problem"
    assert result.task_type == "equation_solving"
    assert result.solver_route == "algebra_word_problem_solver"
    assert result.difficulty == "elementary"
    assert result.requires_solver is True
    assert result.solver_config.solver_name == "AlgebraSolver"
    assert result.solver_config.strategy == "planner_toolcall_algebra_word_problem"


def test_problem_classifier_routes_meeting_speed_ratio_word_problem() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": (
                "甲、乙两地相距900千米，一辆客车和一辆货车同时从两地相对开出，"
                "5小时相遇。已知客车和货车的速度比是5:4，客车平均每小时行多少千米？"
            ),
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_{bus}", "description": "客车的平均速度", "domain": "positive_real"},
                {"symbol": "v_{truck}", "description": "货车的平均速度", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.task_type == "equation_solving"
    assert result.solver_route == "algebra_word_problem_solver"
    assert result.requires_solver is True


def test_problem_classifier_routes_working_backwards_word_problem_without_variables() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math10.png"},
            "problem_text": (
                "水果店运来一批苹果。上午卖出总数的一半少4千克，"
                "下午卖出剩下的一半少2千克，这时还剩18千克苹果。"
                "原来运来多少千克苹果？一共可以装几箱？"
            ),
            "problem_type": "word_problem",
            "knowledge_units": ["arithmetic_operations", "working_backwards_strategy", "fractions_and_halves"],
            "variables": [],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math10",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "word_problem"
    assert result.task_type == "equation_solving"
    assert result.solver_route == "algebra_word_problem_solver"
    assert result.requires_solver is True


def test_problem_classifier_prefers_emr_when_spr_emr_types_conflict() -> None:
    data = math3_spr_data()
    data["problem_type"] = "geometry"
    spr = SPR.model_validate(data)
    emr = EMRBuilder().build("problem_conflict", SPR.model_validate(math3_spr_data()))

    result = ProblemClassifier().classify(spr, emr)

    assert result.primary_type == "algebra"
    assert result.task_type == "optimization"
    assert any("conflict" in warning for warning in result.warnings)


def test_classification_result_is_json_serializable_and_validatable() -> None:
    output = VisionParser().run("any/path.png")
    result = ProblemClassifier().classify(output.spr, output.emr)

    dumped = result.model_dump(mode="json")
    dumped_json = result.model_dump_json(indent=2)
    reloaded = validate_classification_result_dict(json.loads(dumped_json))

    assert dumped["schema_version"] == "CLASSIFICATION-1.0"
    assert reloaded.problem_id == result.problem_id
    assert reloaded.solver_config.solver_name == result.solver_config.solver_name
