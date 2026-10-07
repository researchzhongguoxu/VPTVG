from copy import deepcopy

from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.solver import DeepSeekPlanner, Solver
from mathexplain.agents.verifier import Verifier
from mathexplain.agents.vision_parser import VisionParser
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport, validate_verification_report_dict

from test_solver import (
    FakeEquationSystemAggregatePlanningClient,
    FakeLinearFunctionListTargetPlanningClient,
    FakeMath10PlanningClient,
    FakeMath10MissingSubstitutionPlanningClient,
    FakeMath10RemainingAliasPlanningClient,
    FakeMath10TotalRemainingAliasPlanningClient,
    FakeMath10VerbosePlanningClient,
    FakeMath9PlanningClient,
    math10_spr,
    math10_unknown_emr,
    math9_spr,
    math9_unknown_emr,
    algebra_solver_classification_result,
)


class FakeMath11EmbeddedReferencePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_k","operation":"solve_for","variables":["k"],'
            '"equations":[{"lhs":"(5*k + 4*k)*5","rhs":"900"}],'
            '"expressions":[],"substitutions":{},"target":"k",'
            '"reason":"设一份速度为k，先求每份速度",'
            '"evidence_span":"两地相距900千米，5小时相遇，速度比是5:4"},'
            '{"id":"calc_vp","operation":"evaluate","variables":[],"equations":[],'
            '"expressions":[{"name":"v_p","expression":"5 * $solve_k.k"}],'
            '"substitutions":{},"target":"v_p",'
            '"reason":"客车速度是5份，所以计算5乘每份速度",'
            '"evidence_span":"客车平均每小时行多少千米"}'
            ']}'
        )


class FakeDerivativeExpressionPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"derivative_expr","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"derivative","expression":"diff(x^4 - 2*x^3 + 5*x, x)"}],'
            '"reason":"Apply the power rule to get the derivative.",'
            '"evidence_span":"y = x^4 - 2x^3 + 5x"}]}'
        )


def calculus_unknown_classification_result(problem_id: str = "problem_deriv") -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": problem_id,
            "primary_type": "calculus",
            "task_type": "derivative",
            "difficulty": "intermediate",
            "knowledge_units": ["derivative", "power_rule"],
            "solver_route": "unknown",
            "solver_config": {
                "solver_name": "PlannerToolCallSolver",
                "expected_emr_type": "unknown",
                "task_type": "derivative",
                "strategy": "planner_tool_calls",
            },
            "requires_solver": True,
            "requires_verifier": True,
            "confidence": 0.75,
        }
    )


def test_verifier_passes_math10_tool_call_backed_word_problem() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert isinstance(report, VerificationReport)
    assert report.valid is True
    assert report.status == "passed"
    assert report.verification_mode == "tool_call_backed"
    assert report.metadata["emr_required"] is False
    assert report.metadata["tool_call_trace_used"] is True
    assert report.metadata["explanatory_steps_checked"] > 0
    assert any(step.tool_call_id == "solve_T" for step in report.checked_steps)
    assert any(step.tool_call_id == "calc_boxes" for step in report.checked_steps)
    assert any(
        step.operation == "explanatory_structure"
        and step.metadata == {}
        and "not a full semantic proof" in step.message
        for step in report.checked_steps
    )
    assert validate_verification_report_dict(report.model_dump(mode="json"))


def test_verifier_passes_math10_verbose_named_expression_trace() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10VerbosePlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.verification_mode == "tool_call_backed"
    assert any(step.tool_call_id == "s6" for step in report.checked_steps)
    assert any(step.tool_call_id == "s7" for step in report.checked_steps)


def test_verifier_passes_math10_remaining_alias_trace() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "remaining",
            "description": "上午卖出后剩下的苹果重量",
            "domain": "positive_real",
        },
        {
            "symbol": "n",
            "description": "苹果的箱数",
            "domain": "integer",
        },
    ]
    spr = type(math10_spr()).model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10RemainingAliasPlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.verification_mode == "tool_call_backed"
    assert any(step.tool_call_id == "solve_R" for step in report.checked_steps)
    assert any(step.tool_call_id == "solve_x" for step in report.checked_steps)
    assert any(step.tool_call_id == "compute_boxes" for step in report.checked_steps)


def test_verifier_passes_math10_total_remaining_alias_trace() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "Remaining",
            "description": "最后剩下的苹果重量",
            "domain": "positive_real",
        },
        {
            "symbol": "Boxes",
            "description": "一共可以装的箱数",
            "domain": "integer",
        },
    ]
    spr = type(math10_spr()).model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10TotalRemainingAliasPlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.verification_mode == "tool_call_backed"
    assert any(step.tool_call_id == "s1" for step in report.checked_steps)
    assert any(step.tool_call_id == "s2" for step in report.checked_steps)
    assert any(step.tool_call_id == "s3" for step in report.checked_steps)


def test_verifier_passes_math9_tool_call_backed_word_problem() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath9PlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.verification_mode == "tool_call_backed"
    assert report.backend_steps_valid is True
    assert report.final_answer_valid is True


def test_verifier_replays_embedded_reference_expression() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "甲、乙两地相距900千米，客车和货车同时相对开出，5小时相遇。客车和货车速度比是5:4，求客车速度。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "S", "description": "Distance between places", "domain": "real_positive"},
                {"symbol": "t", "description": "Time until meeting", "domain": "real_positive"},
                {"symbol": "v_p", "description": "Speed of passenger car", "domain": "real_positive"},
                {"symbol": "v_t", "description": "Speed of truck", "domain": "real_positive"},
            ],
            "questions": [{"id": "q1", "text": "客车平均每小时行多少千米？", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "Speed of passenger car", "target_type": "unknown"}],
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
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath11EmbeddedReferencePlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.backend_steps_valid is True
    assert any(step.tool_call_id == "calc_vp" and step.status == "passed" for step in report.checked_steps)


def test_verifier_back_substitutes_equation_system_solution() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "eqsys_003.png"},
            "problem_text": "Solve the system: x + 8y = 15, 4x + y = 29.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "x + 8y = 15"},
                {"id": "c2", "text": "4x + y = 29"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the solution set (x, y).",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "Unknown variable x", "domain": "real"},
                {"symbol": "y", "description": "Unknown variable y", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "Solution set (x, y)", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_eqsys",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_eqsys")
    planner = DeepSeekPlanner(client=FakeEquationSystemAggregatePlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    back_substitution_steps = [
        step for step in report.checked_steps if step.operation == "equation_back_substitution"
    ]
    assert report.valid is True
    assert report.backend_steps_valid is True
    assert len(back_substitution_steps) == 2
    assert all(step.status == "passed" for step in back_substitution_steps)


def test_verifier_fails_when_equation_system_solution_does_not_back_substitute() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "eqsys_003.png"},
            "problem_text": "Solve the system: x + 8y = 15, 4x + y = 29.",
            "problem_type": "algebra",
            "variables": [
                {"symbol": "x", "description": "Unknown variable x", "domain": "real"},
                {"symbol": "y", "description": "Unknown variable y", "domain": "real"},
            ],
            "questions": [{"id": "q1", "text": "Find x and y.", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "Solution set (x, y)", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_eqsys",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_eqsys")
    planner = DeepSeekPlanner(client=FakeEquationSystemAggregatePlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    chain_data = deepcopy(chain.model_dump(mode="json"))
    for step in chain_data["steps"]:
        if step["rule_name"] == "cas_solve_equation_system":
            step["diagnostics"]["input"]["equations"][0]["rhs_sympy"] = "16"
            break
    corrupted = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is False
    assert report.backend_steps_valid is False
    assert any("back-substitution failed" in error for error in report.errors)
    assert any(
        step.operation == "equation_back_substitution" and step.status == "failed"
        for step in report.checked_steps
    )


def test_verifier_simplifies_radical_residuals_for_back_substitution() -> None:
    verifier = Verifier()

    assert verifier._is_zero_residual("-6 + (1 - sqrt(5))**2 + 2*sqrt(5)", ["x"]) is True
    assert (
        verifier._is_zero_residual(
            "-sqrt(6)*(-sqrt(2)/2 + sqrt(6)/2) + (-sqrt(2)/2 + sqrt(6)/2)**2 + 1",
            ["x"],
        )
        is True
    )


def test_verifier_passes_direct_emr_backed_algebra_chain() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)
    chain = Solver().solve(parser_output.emr, classification)

    report = Verifier().verify(
        chain,
        spr=parser_output.spr,
        emr=parser_output.emr,
        classification_result=classification,
    )

    assert report.valid is True
    assert report.verification_mode == "emr_backed"
    assert report.backend_steps_valid is True
    assert report.final_answer_valid is True
    source_step_id = chain.metadata["final_answer_items"][0]["source_step_id"]
    assert any(step.metadata.get("source_step_id") == source_step_id for step in report.checked_steps)


def test_verifier_fails_when_recorded_cas_result_is_changed() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    chain_data = deepcopy(chain.model_dump(mode="json"))
    for step in chain_data["steps"]:
        if step["rule_name"] == "cas_evaluate" and step["diagnostics"].get("tool_call_id") == "calc_boxes":
            step["diagnostics"]["output"]["value"] = "999"
            step["diagnostics"]["output"]["formatted_solution"] = "999"
    corrupted = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is False
    assert report.backend_steps_valid is False
    assert any("Replay output" in error for error in report.errors)
    assert any(step.status == "failed" for step in report.checked_steps)


def test_verifier_fails_when_final_answer_items_are_missing() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    chain_data = deepcopy(chain.model_dump(mode="json"))
    chain_data["metadata"]["final_answer_items"] = []
    corrupted = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is False
    assert report.final_answer_valid is False
    assert "metadata.final_answer_items is missing or empty." in report.errors


def test_verifier_rejects_symbolic_final_answer_item_value() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10MissingSubstitutionPlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    chain_data = deepcopy(chain.model_dump(mode="json"))
    chain_data["metadata"]["final_answer_items"][-1]["value"] = "x/8"
    corrupted = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is False
    assert report.final_answer_valid is False
    assert "final_answer_item.value contains unresolved symbols." in report.errors


def test_verifier_allows_symbolic_function_expression_answer_item() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_002.png"},
            "problem_text": "A line y = kx + b passes through (-1, 3) and crosses the x-axis at (2, 0). Find the explicit equation.",
            "problem_type": "algebra",
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "k", "description": "slope", "domain": "real"},
                {"symbol": "b", "description": "intercept", "domain": "real"},
            ],
            "questions": [
                {"id": "q1", "text": "Find the explicit equation of the line.", "target_ids": ["t1"]}
            ],
            "targets": [{"id": "t1", "text": "explicit equation y expression", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_002",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_002")
    planner = DeepSeekPlanner(client=FakeLinearFunctionListTargetPlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert report.final_answer_valid is True
    assert not any("unresolved symbols" in error for error in report.errors)


def test_verifier_allows_symbolic_derivative_expression_answer_item() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_003.png"},
            "problem_text": "Given y = x^4 - 2x^3 + 5x, find the derivative.",
            "problem_type": "calculus",
            "variables": [
                {"symbol": "x", "description": "Independent variable", "domain": "real"},
                {"symbol": "y", "description": "Dependent variable", "domain": "real"},
            ],
            "questions": [{"id": "q1", "text": "Find the derivative.", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "Derivative expression", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_003",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = calculus_unknown_classification_result(problem_id="problem_deriv_003")
    planner = DeepSeekPlanner(client=FakeDerivativeExpressionPlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "derivative",
            "value": "4*x**3 - 6*x**2 + 5",
            "source_tool_call_id": "derivative_expr",
            "backend_operation": "construct_expression",
            "backend_confirmed": True,
        }
    ]
    assert report.valid is True
    assert report.final_answer_valid is True


def test_verifier_warns_when_modeling_expression_is_not_parseable() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())
    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    chain_data = deepcopy(chain.model_dump(mode="json"))
    for step in chain_data["steps"]:
        if step["rule_name"] == "modeling_from_tool_call":
            step["math_expression"] = "T/**bad"
            break
    corrupted = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    assert report.valid is True
    assert any("could not be parsed" in warning for warning in report.warnings)


def test_verifier_marks_unsupported_chain_as_unsupported() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)
    chain_data = Solver().solve(parser_output.emr, classification).model_dump(mode="json")
    chain_data["steps"] = [
        {
            "step_id": "chain_main_step_0",
            "step_index": 0,
            "description": "Unsupported route.",
            "rule_name": "unsupported_route",
            "verification_status": "unverified",
            "diagnostics": {"solver_route": "geometry_stub"},
        }
    ]
    chain_data["final_answer"] = None
    chain_data["confidence"] = 0.0
    chain_data["metadata"] = {"solver_route": "geometry_stub", "unsupported_reason": "not supported"}
    unsupported = SolutionChain.model_validate(chain_data)

    report = Verifier().verify(unsupported, spr=parser_output.spr, emr=parser_output.emr)

    assert report.valid is False
    assert report.status == "unsupported"
    assert report.verification_mode == "unsupported"
