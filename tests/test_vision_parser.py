import json
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mathexplain.agents.vision_parser import (
    EMRBuilder,
    EMRRepair,
    EMRValidator,
    ImagePreprocessor,
    SPRGenerator,
    SPRRepair,
    SPRValidator,
    VisionParser,
    VisionParserOutput,
)
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.reports import ParsingReport
from mathexplain.schemas.spr import SPR
from mathexplain.services import llm as llm_service
from mathexplain.services.llm import DASHSCOPE_API_KEY_ERROR, DashScopeQwenVisionClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PREPROCESS_PATH = PROJECT_ROOT / "scripts" / "run_preprocess.py"
RUN_PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "run_preprocess",
    RUN_PREPROCESS_PATH,
)
assert RUN_PREPROCESS_SPEC is not None
assert RUN_PREPROCESS_SPEC.loader is not None
run_preprocess = importlib.util.module_from_spec(RUN_PREPROCESS_SPEC)
RUN_PREPROCESS_SPEC.loader.exec_module(run_preprocess)


def create_test_problem_image(path: Path) -> None:
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 240, 110), outline="black", width=3)
    draw.text((100, 75), "2*x + 3 = 7", fill="black")
    image.save(path)


def valid_qwen_spr_json(image_path: Path) -> str:
    return json.dumps(
        {
            "schema_version": "SPR-1.0",
            "source_image": {
                "image_path": str(image_path),
            },
            "problem_text": "Solve 2*x + 3 = 7.",
            "problem_stem": "Solve 2*x + 3 = 7.",
            "conditions": [
                {
                    "id": "condition_1",
                    "text": "2*x + 3 = 7",
                    "linked_formula_ids": ["formula_1"],
                }
            ],
            "questions": [
                {
                    "id": "question_1",
                    "text": "Solve for x.",
                    "question_type": "solve",
                    "target_ids": ["target_1"],
                }
            ],
            "formulas": [
                {
                    "id": "formula_1",
                    "raw_text": "2*x + 3 = 7",
                    "latex": "2x + 3 = 7",
                    "role": "condition",
                }
            ],
            "variables": [
                {
                    "symbol": "x",
                    "description": "unknown variable",
                    "domain": "real",
                }
            ],
            "visual_objects": [],
            "layout": None,
            "problem_type": "algebra",
            "knowledge_units": ["linear equations"],
            "targets": [
                {
                    "id": "target_1",
                    "text": "x",
                    "target_type": "solve",
                }
            ],
            "confidence": {
                "overall": 0.9,
                "text": 0.9,
                "formula": 0.9,
                "layout": 0.8,
            },
            "uncertainties": [],
            "metadata": {
                "parser_model": "fake-qwen",
                "prompt_version": "fake",
            },
        }
    )


def loose_geometry_qwen_spr_json() -> str:
    return json.dumps(
        {
            "schema_version": "SPR-1.0",
            "source_image": "image_url_placeholder",
            "problem_text": "如图，Rt△ABC中，∠B=90°，AD平分∠BAC，求证BC是⊙O的切线。",
            "problem_stem": "如图，Rt△ABC中，∠B=90°，AD平分∠BAC。",
            "conditions": [
                "Rt△ABC",
                "∠B=90°",
                "AD 平分 ∠BAC",
                "AD 交 BC 于点 D",
                "点 E 在 AC 上",
                "AE 为 O 的直径",
                "⊙O 经过点 D",
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "求证：BC 是 ⊙O 的切线；求证：CD²=CE·CA",
                    "type": "proof",
                    "sub_questions": [
                        {"id": "1", "text": "求证：BC 是 ⊙O 的切线"},
                        {"id": "2", "text": "求证：CD²=CE·CA"},
                    ],
                }
            ],
            "formulas": ["∠B=90°", "CD²=CE·CA"],
            "variables": ["A", "B", "C", "D", "E", "F", "O"],
            "visual_objects": [
                {
                    "type": "triangle",
                    "label": "Rt△ABC",
                    "properties": {"angle_B": "90°"},
                },
                {
                    "type": "circle",
                    "label": "⊙O",
                    "properties": {"diameter": "AE", "passes_through": ["D"]},
                },
                {
                    "type": "line_segment",
                    "label": "AD",
                    "properties": {"role": "angle_bisector_of_BAC"},
                },
                {"type": "shaded_region", "description": "阴影区域"},
            ],
            "layout": "Text description at the top with diagram below.",
            "problem_type": "geometry_proof",
            "knowledge_units": ["圆的切线判定", "相似三角形"],
            "targets": ["Prove BC is tangent to circle O", "Prove CD^2 = CE * CA"],
            "confidence": 0.95,
            "uncertainties": [
                "The point F is visible in diagram but not explicit in text.",
                "The problem text ends with a partially visible statement.",
            ],
            "metadata": {"parser_model": "fake-qwen"},
        }
    )


def loose_algebra_choice_qwen_spr_json() -> str:
    return json.dumps(
        {
            "schema_version": "SPR-1.0",
            "source_image": {
                "image_path": "placeholder",
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
                    "question_type": "unknown",
                    "target_ids": [],
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
                    "role": "unknown",
                },
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
            "visual_objects": [],
            "layout": None,
            "problem_type": "algebra",
            "knowledge_units": ["inequality", "algebraic_expression", "minimum_value"],
            "targets": [
                {
                    "id": "t1",
                    "text": "最小值",
                    "target_type": "unknown",
                }
            ],
            "confidence": {
                "overall": 0.95,
                "text": 0.95,
                "formula": 0.95,
                "layout": 0.95,
            },
            "uncertainties": [],
            "metadata": {
                "parser_model": "fake-qwen",
                "processing_time": 0.5,
            },
        }
    )


def explicit_equation_word_problem_spr_json(image_path: Path) -> str:
    return json.dumps(
        {
            "schema_version": "SPR-1.0",
            "source_image": {
                "image_path": str(image_path),
            },
            "problem_text": (
                "A school bought some books. If each class gets 5 books there are 18 left; "
                "if each class gets 8 books there are 12 short. How many books were bought?"
            ),
            "problem_stem": "A word problem with an explicit equation 5*x+18=8*x-12.",
            "conditions": [
                {
                    "id": "condition_1",
                    "text": "5*x+18=8*x-12",
                    "linked_formula_ids": ["formula_1"],
                }
            ],
            "questions": [
                {
                    "id": "question_1",
                    "text": "How many books were bought?",
                    "question_type": "compute",
                    "target_ids": ["target_1"],
                }
            ],
            "formulas": [
                {
                    "id": "formula_1",
                    "raw_text": "5*x+18=8*x-12",
                    "latex": "5x+18=8x-12",
                    "role": "unknown",
                },
                {
                    "id": "formula_2",
                    "raw_text": "5*x+18",
                    "latex": "5x+18",
                    "role": "condition",
                }
            ],
            "variables": [
                {
                    "symbol": "x",
                    "description": "number of classes",
                    "domain": "integer",
                }
            ],
            "visual_objects": [],
            "problem_type": "word_problem",
            "knowledge_units": ["linear equations"],
            "targets": [
                {
                    "id": "target_1",
                    "text": "total books",
                    "target_type": "compute",
                }
            ],
            "confidence": {
                "overall": 0.9,
                "text": 0.9,
                "formula": 0.9,
            },
            "metadata": {
                "parser_model": "fake-qwen",
            },
        }
    )


class FakeQwenClient:
    model = "fake-qwen"

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_image_data_uri = ""

    def generate_json_from_image(
        self,
        image_data_uri: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.last_image_data_uri = image_data_uri
        assert system_prompt
        assert user_prompt
        return self.content


def test_vision_parser_package_exports_public_interfaces() -> None:
    assert VisionParser
    assert VisionParserOutput
    assert ImagePreprocessor
    assert SPRGenerator
    assert SPRValidator
    assert SPRRepair
    assert EMRBuilder
    assert EMRValidator
    assert EMRRepair


def test_spr_generator_qwen_mode_accepts_valid_fake_json(tmp_path: Path) -> None:
    image_path = tmp_path / "problem.png"
    create_test_problem_image(image_path)
    fake_client = FakeQwenClient(valid_qwen_spr_json(image_path))

    spr = SPRGenerator(mode="qwen", qwen_client=fake_client).generate("problem_001", str(image_path))

    assert isinstance(spr, SPR)
    assert spr.problem_text == "Solve 2*x + 3 = 7."
    assert spr.metadata["problem_id"] == "problem_001"
    assert spr.metadata["parser_mode"] == "qwen"
    assert spr.metadata["parser_model"] == "fake-qwen"
    assert spr.metadata["prompt_version"] == "qwen-spr-v1"
    assert fake_client.last_image_data_uri.startswith("data:image/png;base64,")


def test_spr_generator_qwen_mode_normalizes_loose_geometry_json(tmp_path: Path) -> None:
    image_path = tmp_path / "geometry.png"
    create_test_problem_image(image_path)

    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(loose_geometry_qwen_spr_json()),
    ).generate("problem_geometry", str(image_path))

    assert isinstance(spr, SPR)
    assert spr.problem_type.value == "geometry"
    assert spr.source_image.image_path == str(image_path)
    assert len(spr.conditions) == 7
    assert spr.conditions[0].id == "c1"
    assert spr.conditions[0].text == "Rt△ABC"
    assert spr.questions[0].question_type == "prove"
    assert len(spr.formulas) == 2
    assert spr.formulas[0].id == "f1"
    assert len(spr.variables) == 7
    assert spr.variables[0].symbol == "A"
    assert len(spr.visual_objects) == 4
    assert spr.visual_objects[0].id == "vo1"
    assert spr.visual_objects[2].object_type == "segment"
    assert spr.visual_objects[3].object_type == "other"
    assert spr.layout is None
    assert len(spr.targets) == 2
    assert spr.targets[0].target_type == "prove"
    assert spr.confidence.overall == 0.95
    assert len(spr.uncertainties) == 2
    assert spr.uncertainties[0].field_path == "unknown"
    assert spr.metadata["parser_mode"] == "qwen"
    assert spr.metadata["problem_id"] == "problem_geometry"


def test_spr_generator_qwen_mode_enhances_algebra_choice_json(tmp_path: Path) -> None:
    image_path = tmp_path / "choice.png"
    create_test_problem_image(image_path)

    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(loose_algebra_choice_qwen_spr_json()),
    ).generate("problem_choice", str(image_path))

    assert isinstance(spr, SPR)
    assert spr.problem_type.value == "algebra"
    assert spr.questions[0].question_type == "select"
    assert spr.questions[0].target_ids == ["t1"]
    assert spr.targets[0].target_type == "select"
    assert spr.formulas[0].role == "condition"
    assert spr.formulas[1].role == "goal"
    option_formulas = [formula for formula in spr.formulas if formula.role == "option"]
    assert [formula.raw_text for formula in option_formulas] == ["A. 3", "B. 9", "C. 4", "D. 8"]
    assert "processing_time" not in spr.metadata
    assert spr.metadata["problem_id"] == "problem_choice"


def test_spr_generator_qwen_mode_does_not_duplicate_existing_options(tmp_path: Path) -> None:
    image_path = tmp_path / "choice.png"
    create_test_problem_image(image_path)
    data = json.loads(loose_algebra_choice_qwen_spr_json())
    data["formulas"].extend(
        [
            {"id": "opt_a", "raw_text": "3", "latex": "3", "role": "option"},
            {"id": "opt_b", "raw_text": "9", "latex": "9", "role": "option"},
            {"id": "opt_c", "raw_text": "4", "latex": "4", "role": "option"},
            {"id": "opt_d", "raw_text": "8", "latex": "8", "role": "option"},
        ]
    )

    spr = SPRGenerator(mode="qwen", qwen_client=FakeQwenClient(json.dumps(data))).generate(
        "problem_choice",
        str(image_path),
    )

    option_formulas = [formula for formula in spr.formulas if formula.role == "option"]
    assert [formula.raw_text for formula in option_formulas] == ["3", "9", "4", "8"]


def test_emr_builder_builds_math3_style_algebra_emr(tmp_path: Path) -> None:
    image_path = tmp_path / "choice.png"
    create_test_problem_image(image_path)
    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(loose_algebra_choice_qwen_spr_json()),
    ).generate("problem_choice", str(image_path))

    emr = EMRBuilder().build("problem_choice", spr)

    assert emr.representation_type == "equation_system"
    assert [variable.symbol for variable in emr.variables] == ["a", "b"]
    assert [variable.domain.value for variable in emr.variables] == ["positive_real", "positive_real"]
    assert emr.equations[0].lhs_sympy == "2*a+b"
    assert emr.equations[0].rhs_sympy == "1"
    assert emr.equations[0].source_condition_ids == ["c2"]
    assert emr.equations[0].source_formula_ids == ["f1"]
    assert emr.expressions[0].sympy == "(5*a+b)/(a**2+a*b)"
    assert emr.expressions[0].source_formula_id == "f2"
    assert emr.goals[0].goal_type == "compute"
    assert emr.goals[0].target_expression_id == "expr_1"
    assert emr.metadata["builder_mode"] == "algebra_minimal"
    assert emr.metadata["choice_options"] == [
        {
            "label": "A",
            "value": "3",
            "source_formula_id": "f3",
            "latex": "3",
        },
        {
            "label": "B",
            "value": "9",
            "source_formula_id": "f4",
            "latex": "9",
        },
        {
            "label": "C",
            "value": "4",
            "source_formula_id": "f5",
            "latex": "4",
        },
        {
            "label": "D",
            "value": "8",
            "source_formula_id": "f6",
            "latex": "8",
        },
    ]


def test_emr_builder_builds_goal_expression_from_old_math3_artifact_shape() -> None:
    spr = SPR.model_validate(json.loads(loose_algebra_choice_qwen_spr_json()))

    emr = EMRBuilder().build("problem_choice_old_artifact", spr)

    assert emr.representation_type == "equation_system"
    assert emr.expressions[0].sympy == "(5*a+b)/(a**2+a*b)"
    assert emr.expressions[0].source_formula_id == "f2"
    assert emr.goals[0].goal_type == "compute"
    assert "choice_options" not in emr.metadata


def test_emr_builder_returns_unknown_for_non_algebra_spr(tmp_path: Path) -> None:
    image_path = tmp_path / "geometry.png"
    create_test_problem_image(image_path)
    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(loose_geometry_qwen_spr_json()),
    ).generate("problem_geometry", str(image_path))

    emr = EMRBuilder().build("problem_geometry", spr)

    assert emr.representation_type == "unknown"
    assert emr.equations == []
    assert emr.metadata["builder_mode"] == "fallback_unknown"
    assert emr.metadata["warnings"]


def test_emr_builder_builds_explicit_equation_inside_word_problem(tmp_path: Path) -> None:
    image_path = tmp_path / "word_problem.png"
    create_test_problem_image(image_path)
    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(explicit_equation_word_problem_spr_json(image_path)),
    ).generate("problem_word_explicit_equation", str(image_path))

    emr = EMRBuilder().build("problem_word_explicit_equation", spr)

    assert spr.problem_type.value == "word_problem"
    assert getattr(emr.source_problem_type, "value", emr.source_problem_type) == "algebra"
    assert emr.representation_type == "equation_system"
    assert emr.metadata["source_spr_problem_type"] == "word_problem"
    assert [variable.symbol for variable in emr.variables] == ["x"]
    assert emr.equations[0].lhs_sympy == "5*x+18"
    assert emr.equations[0].rhs_sympy == "8*x-12"
    assert emr.equations[0].source_condition_ids == ["condition_1"]
    assert emr.equations[0].source_formula_ids == ["formula_1"]
    assert emr.expressions[0].sympy == "5*x+18"
    assert emr.goals[0].target_expression_id == "expr_1"


def test_emr_builder_returns_unknown_for_unparseable_algebra_without_crashing() -> None:
    data = json.loads(loose_algebra_choice_qwen_spr_json())
    data["conditions"] = [
        {
            "id": "c1",
            "text": "a, b are related in an unclear way",
            "linked_formula_ids": ["f1"],
        }
    ]
    data["formulas"] = [
        {
            "id": "f1",
            "raw_text": "a and b are related",
            "latex": None,
            "role": "condition",
        }
    ]
    spr = SPR.model_validate(data)

    emr = EMRBuilder().build("problem_unparseable", spr)

    assert emr.representation_type == "unknown"
    assert emr.equations == []
    assert emr.metadata["builder_mode"] == "fallback_unknown"
    assert "No reliable algebra equation" in emr.metadata["warnings"][0]


def test_emr_validator_accepts_math3_algebra_emr(tmp_path: Path) -> None:
    image_path = tmp_path / "choice.png"
    create_test_problem_image(image_path)
    spr = SPRGenerator(
        mode="qwen",
        qwen_client=FakeQwenClient(loose_algebra_choice_qwen_spr_json()),
    ).generate("problem_choice", str(image_path))
    emr = EMRBuilder().build("problem_choice", spr)

    report = EMRValidator().validate(emr)

    assert report["valid"] is True
    assert report["schema_valid"] is True
    assert report["sympy_parse_valid"] is True
    assert report["sympy_parse_skipped"] is False
    assert report["errors"] == []
    assert set(report["checked_fields"]) == {
        "expressions[0].sympy",
        "equations[0].lhs_sympy",
        "equations[0].rhs_sympy",
    }


def test_emr_validator_accepts_mock_emr() -> None:
    emr = VisionParser().run("any/path.png").emr

    report = EMRValidator().validate(emr)

    assert report["valid"] is True
    assert report["sympy_parse_valid"] is True
    assert report["checked_fields"] == [
        "expressions[0].sympy",
        "equations[0].lhs_sympy",
        "equations[0].rhs_sympy",
    ]


def test_emr_validator_skips_non_algebra_unknown_emr() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_geometry",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is True
    assert report["sympy_parse_skipped"] is True
    assert report["checked_fields"] == []
    assert report["errors"] == []


def test_emr_validator_reports_invalid_sympy_expression_without_raising() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_invalid",
            "source_problem_type": "algebra",
            "representation_type": "expression",
            "variables": [{"id": "var_x", "symbol": "x"}],
            "expressions": [{"id": "expr_1", "sympy": "2**"}],
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is False
    assert report["sympy_parse_valid"] is False
    assert report["errors"][0]["field_path"] == "expressions[0].sympy"
    assert report["errors"][0]["value"] == "2**"
    assert "SymPy parse failed" in report["errors"][0]["message"]


def test_emr_validator_reports_tuple_parse_without_raising() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_tuple_parse",
            "source_problem_type": "algebra",
            "representation_type": "expression",
            "variables": [
                {"id": "var_x", "symbol": "x"},
                {"id": "var_y", "symbol": "y"},
            ],
            "expressions": [{"id": "expr_1", "sympy": "x, y"}],
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is False
    assert report["sympy_parse_valid"] is False
    assert report["errors"][0]["field_path"] == "expressions[0].sympy"
    assert "tuple/list" in report["errors"][0]["message"]


def test_emr_validator_reports_undeclared_symbols() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_undeclared",
            "source_problem_type": "algebra",
            "representation_type": "expression",
            "variables": [{"id": "var_x", "symbol": "x"}],
            "expressions": [{"id": "expr_1", "sympy": "x + y"}],
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is False
    assert report["errors"][0]["field_path"] == "expressions[0].sympy"
    assert report["errors"][0]["undeclared_symbols"] == ["y"]


def test_emr_validator_reports_invalid_constraint_field_path() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_bad_constraint",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_x", "symbol": "x"}],
            "constraints": [
                {
                    "id": "constraint_1",
                    "expression_sympy": "x +",
                    "constraint_type": "condition",
                }
            ],
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is False
    assert report["errors"][0]["field_path"] == "constraints[0].expression_sympy"
    assert report["errors"][0]["value"] == "x +"


def test_emr_validator_allows_empty_algebra_expression_fields() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_empty_algebra",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )

    report = EMRValidator().validate(emr)

    assert report["valid"] is True
    assert report["sympy_parse_valid"] is True
    assert report["checked_fields"] == []
    assert report["errors"] == []


def test_spr_generator_qwen_mode_enhances_proof_target_links(tmp_path: Path) -> None:
    image_path = tmp_path / "geometry.png"
    create_test_problem_image(image_path)
    data = json.loads(loose_geometry_qwen_spr_json())
    data["questions"][0]["question_type"] = "unknown"
    data["targets"] = [
        {
            "id": "t1",
            "text": "求证 BC 是圆 O 的切线",
            "target_type": "unknown",
        }
    ]

    spr = SPRGenerator(mode="qwen", qwen_client=FakeQwenClient(json.dumps(data))).generate(
        "problem_proof",
        str(image_path),
    )

    assert spr.questions[0].question_type == "prove"
    assert spr.questions[0].target_ids == ["t1"]
    assert spr.targets[0].target_type == "prove"


def test_spr_generator_qwen_mode_rejects_non_json_fake_response(tmp_path: Path) -> None:
    image_path = tmp_path / "problem.png"
    create_test_problem_image(image_path)

    generator = SPRGenerator(mode="qwen", qwen_client=FakeQwenClient("not json"))

    with pytest.raises(ValueError, match="non-JSON"):
        generator.generate("problem_001", str(image_path))


def test_spr_generator_qwen_mode_rejects_schema_invalid_fake_response(tmp_path: Path) -> None:
    image_path = tmp_path / "problem.png"
    create_test_problem_image(image_path)
    invalid_schema_json = json.dumps(
        {
            "schema_version": "SPR-1.0",
            "source_image": {
                "image_path": str(image_path),
            },
            "problem_text": "",
            "confidence": {
                "overall": 0.9,
            },
        }
    )

    generator = SPRGenerator(mode="qwen", qwen_client=FakeQwenClient(invalid_schema_json))

    with pytest.raises(ValueError, match="still invalid after normalization"):
        generator.generate("problem_001", str(image_path))


def test_dashscope_qwen_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.setattr("mathexplain.services.llm.load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY is required"):
        DashScopeQwenVisionClient()


def test_dashscope_qwen_client_disables_thinking_for_json(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeMessage:
        content = '{"ok": true}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(llm_service, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_service, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)

    content = DashScopeQwenVisionClient().generate_json_from_image(
        image_data_uri="data:image/png;base64,abc",
        system_prompt="system",
        user_prompt="user",
    )

    assert content == '{"ok": true}'
    assert captured_kwargs["response_format"] == {"type": "json_object"}
    assert captured_kwargs["extra_body"] == {"enable_thinking": False}


@pytest.mark.skipif(
    not (
        __import__("os").getenv("RUN_QWEN_INTEGRATION") == "1"
        and __import__("os").getenv("DASHSCOPE_API_KEY")
    ),
    reason="Set RUN_QWEN_INTEGRATION=1 and DASHSCOPE_API_KEY to run real Qwen API test.",
)
def test_spr_generator_qwen_real_api_optional() -> None:
    image_path = Path("data/inputs/images/math1.png")
    if not image_path.exists():
        pytest.skip("data/inputs/images/math1.png is not available.")

    spr = SPRGenerator(mode="qwen").generate("problem_real_qwen", str(image_path))

    assert isinstance(spr, SPR)
    assert spr.problem_text
    assert spr.metadata["parser_mode"] == "qwen"


def test_image_preprocessor_processes_existing_image(tmp_path: Path) -> None:
    image_path = tmp_path / "problem.png"
    processed_dir = tmp_path / "processed"
    create_test_problem_image(image_path)

    report = ImagePreprocessor().run(str(image_path), context={"processed_dir": processed_dir})

    assert report["status"] == "success"
    assert report["original_image_path"] == str(image_path)
    assert report["image_read"] is True
    assert report["preprocessing_applied"] is True
    assert report["original_size"] == {"width": 320, "height": 180}
    assert report["cropped_size"]["width"] <= 320
    assert report["cropped_size"]["height"] <= 180
    assert report["crop_bbox"]["width"] > 0
    assert report["crop_bbox"]["height"] > 0
    assert set(report["processed_image_paths"]) == {"crop", "enhanced", "binary"}
    for processed_path in report["processed_image_paths"].values():
        assert Path(processed_path).exists()


def test_image_preprocessor_skips_missing_image_without_raising(tmp_path: Path) -> None:
    missing_image_path = tmp_path / "missing.png"

    report = ImagePreprocessor().run(
        str(missing_image_path),
        context={"processed_dir": tmp_path / "processed"},
    )

    assert report["status"] == "skipped"
    assert report["image_read"] is False
    assert report["preprocessing_applied"] is False
    assert report["processed_image_paths"] == {}
    assert report["warnings"]
    assert report["errors"] == []


def test_run_preprocess_script_writes_processed_images(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    image_path = tmp_path / "problem.png"
    output_dir = tmp_path / "script_processed"
    create_test_problem_image(image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_preprocess.py",
            "--image",
            str(image_path),
            "--out",
            str(output_dir),
        ],
    )

    exit_code = run_preprocess.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out.split("\n\n")[0])

    assert exit_code == 0
    assert report["status"] == "success"
    assert Path(report["processed_image_paths"]["crop"]).exists()
    assert Path(report["processed_image_paths"]["enhanced"]).exists()
    assert Path(report["processed_image_paths"]["binary"]).exists()


def test_vision_parser_returns_valid_mock_outputs_for_nonexistent_image() -> None:
    output = VisionParser().run("data/inputs/does_not_exist.png")

    assert isinstance(output, VisionParserOutput)
    assert isinstance(output.spr, SPR)
    assert isinstance(output.emr, EMR)
    assert isinstance(output.parsing_report, ParsingReport)
    assert output.artifact_paths == {}
    assert output.spr.problem_text == "Solve 2*x + 3 = 7."
    assert output.spr.conditions[0].text == "2*x + 3 = 7"
    assert output.spr.questions[0].text == "Solve for x."
    assert output.spr.formulas[0].raw_text == "2*x + 3 = 7"
    assert output.spr.variables[0].symbol == "x"
    assert output.spr.targets[0].text == "x"
    assert output.spr.problem_type.value == "algebra"


def test_vision_parser_does_not_require_real_image_path() -> None:
    image_path = "Z:/a/path/that/should/not/be/read/problem.png"

    output = VisionParser().run(image_path)

    assert output.spr.source_image.image_path == image_path
    assert output.parsing_report.metadata["image_read"] is False


def test_vision_parser_mock_output_is_deterministic() -> None:
    parser = VisionParser()
    first = parser.run("same/image/path.png")
    second = parser.run("same/image/path.png")

    assert first.problem_id == second.problem_id
    assert first.spr.model_dump(mode="json") == second.spr.model_dump(mode="json")
    assert first.emr.model_dump(mode="json") == second.emr.model_dump(mode="json")
    assert first.parsing_report.model_dump(mode="json") == second.parsing_report.model_dump(
        mode="json"
    )


def test_vision_parser_generates_expected_mock_emr() -> None:
    output = VisionParser().run("any/path.png")

    assert output.emr.problem_id == output.problem_id
    assert output.emr.source_spr_version == "SPR-1.0"
    assert output.emr.source_spr_id == output.problem_id
    assert output.emr.representation_type == "equation_system"
    assert output.emr.variables[0].symbol == "x"
    assert output.emr.equations[0].lhs_sympy == "2*x + 3"
    assert output.emr.equations[0].rhs_sympy == "7"
    assert output.emr.goals[0].target == "x"
    assert output.emr.metadata["builder_mode"] == "deterministic_mock"


def test_vision_parser_outputs_are_serializable() -> None:
    output = VisionParser().run("any/path.png")

    spr_json = output.spr.model_dump_json(indent=2)
    emr_json = output.emr.model_dump_json(indent=2)
    report_json = output.parsing_report.model_dump_json(indent=2)

    assert json.loads(spr_json)["problem_text"] == "Solve 2*x + 3 = 7."
    assert json.loads(emr_json)["representation_type"] == "equation_system"
    assert json.loads(report_json)["status"] == "success"


def test_save_artifacts_false_does_not_return_artifact_paths(tmp_path: Path) -> None:
    output = VisionParser().run("any/path.png", save_artifacts=False, output_dir=tmp_path)

    assert output.artifact_paths == {}
    assert list(tmp_path.iterdir()) == []


def test_save_artifacts_true_writes_json_files(tmp_path: Path) -> None:
    output = VisionParser().run("any/path.png", save_artifacts=True, output_dir=tmp_path)

    assert set(output.artifact_paths) == {"spr", "emr", "parsing_report"}
    for artifact_path in output.artifact_paths.values():
        assert Path(artifact_path).exists()

    spr_data = json.loads(Path(output.artifact_paths["spr"]).read_text(encoding="utf-8"))
    emr_data = json.loads(Path(output.artifact_paths["emr"]).read_text(encoding="utf-8"))
    report_data = json.loads(
        Path(output.artifact_paths["parsing_report"]).read_text(encoding="utf-8")
    )

    assert spr_data["problem_text"] == "Solve 2*x + 3 = 7."
    assert emr_data["equations"][0]["lhs_sympy"] == "2*x + 3"
    assert report_data["problem_id"] == output.problem_id


def test_parsing_report_contains_expected_stages() -> None:
    output = VisionParser().run("any/path.png")

    stage_names = [stage.stage_name for stage in output.parsing_report.stages]

    assert stage_names == [
        "preprocess",
        "spr_generate",
        "spr_validate",
        "emr_build",
        "emr_validate",
        "confidence_estimate",
        "repair_stub",
    ]
    assert output.parsing_report.status == "success"
    assert output.parsing_report.confidence_summary["pipeline_overall"] == 1.0
