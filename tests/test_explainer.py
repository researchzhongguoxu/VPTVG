from copy import deepcopy

from mathexplain.agents.explainer import Explainer
from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.script_director import ScriptDirector
from mathexplain.agents.solver import DeepSeekPlanner, Solver
from mathexplain.agents.verifier import Verifier
from mathexplain.agents.vision_parser import VisionParser
from mathexplain.schemas.eds import validate_executable_director_script_dict
from mathexplain.schemas.explanation import ExplanationScript, validate_explanation_script_dict
from mathexplain.schemas.scs import SolutionChain, SolutionStep
from mathexplain.schemas.teaching_plan import TeachingPlan, TeachingPlanMove, TeachingPlanQualityReport
from mathexplain.schemas.verification import VerificationReport

from test_solver import (
    FakeMath7PlanningClient,
    FakeMath9PlanningClient,
    FakeMath10PlanningClient,
    FakeMath10NumBoxesPlanningClient,
    FakeMath10SoldBreakdownPlanningClient,
    math7_classification_result,
    math7_spr,
    math7_unknown_emr,
    math9_spr,
    math9_unknown_emr,
    math10_spr,
    math10_unknown_emr,
)


class FakeMath10RemainingPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_system","operation":"solve_equation_system",'
            '"variables":["x","x_rem"],'
            '"equations":['
            '{"lhs":"x_rem","rhs":"x/2 + 4"},'
            '{"lhs":"x_rem/2 + 2","rhs":"18"}'
            '],'
            '"target":"x","reason":"solve original weight and remaining amount",'
            '"evidence_span":"remaining 18 kg"},'
            '{"id":"calc_boxes","operation":"evaluate","variables":["x"],'
            '"expressions":[{"name":"boxes","expression":"x/8"}],'
            '"substitutions":{"x":"$solve_system.x"},"target":"boxes",'
            '"reason":"calculate boxes","evidence_span":"8 kg per box"}'
            ']}'
        )


def test_explainer_does_not_create_expr_visual_ref_for_text_only_teaching_move() -> None:
    move = TeachingPlanMove(
        move_id="move_1",
        move_type="modeling",
        source_step_ids=["step_1"],
        narration="先读出题目给出的已知条件。",
        pedagogical_actions=["READ_ALOUD", "HIGHLIGHT_NUMBER"],
        estimated_duration_ms=1000,
        pause_after_ms=0,
    )

    segments = Explainer()._segments_from_teaching_moves([move])

    assert segments[0]["math_expressions"] == []
    assert segments[0]["visual_refs"] == []


def test_explainer_preserves_split_teaching_moves_for_shared_cas_step() -> None:
    scs = SolutionChain(
        problem_id="p",
        chain_id="c",
        final_answer="答案是 7",
        confidence=1.0,
        steps=[
            SolutionStep(
                step_id="model",
                step_index=0,
                description="model the apple pie cost",
                math_expression="20 - (2*2 + 8 + 1)",
                rule_name="modeling_from_tool_call",
                diagnostics={"tool_call_id": "calc"},
            ),
            SolutionStep(
                step_id="cas",
                step_index=1,
                description="evaluate",
                rule_name="cas_evaluate",
                depends_on=["model"],
                diagnostics={
                    "tool_call_id": "calc",
                    "input": {
                        "expression": "20 - (2*2 + 8 + 1)",
                        "target": "cost",
                    },
                    "output": {"formatted_solution": "7", "value": "7"},
                },
            ),
        ],
    )
    moves = [
        TeachingPlanMove(
            move_id="move_total",
            move_type="modeling",
            source_step_ids=["model"],
            display_expression="20",
            narration="The total is 20.",
            pedagogical_actions=["RENDER_EXPRESSION"],
            estimated_duration_ms=1000,
        ),
        TeachingPlanMove(
            move_id="move_chips",
            move_type="modeling",
            source_step_ids=["model"],
            display_expression="2 * 2",
            narration="The chips cost 2 times 2.",
            pedagogical_actions=["RENDER_EXPRESSION"],
            estimated_duration_ms=1000,
        ),
        TeachingPlanMove(
            move_id="move_answer",
            move_type="answer",
            source_step_ids=["model"],
            display_expression="7",
            narration="So the answer is 7.",
            pedagogical_actions=["RENDER_EXPRESSION"],
            estimated_duration_ms=1000,
        ),
    ]

    segments = Explainer()._segments_from_teaching_moves(
        moves,
        presentation={"language": {"output_language": "en-US"}},
        scs=scs,
    )

    assert segments[0]["math_expressions"] == ["20"]
    assert segments[1]["math_expressions"] == ["2 × 2"]
    assert "This gives 7" not in segments[0]["narration"]
    assert "This gives 7" not in segments[1]["narration"]
    assert segments[2]["math_expressions"] == ["20 - (2 × 2 + 8 + 1) = 7"]


def test_explainer_answer_unit_prefers_question_seconds_over_problem_meters() -> None:
    spr = math7_spr()
    spr.problem_text = (
        "Each team runs a 4 by 400 meter relay. The first team takes 220 seconds "
        "and the second team takes 222 seconds. How many seconds will the faster team win by?"
    )
    spr.questions[0].text = "How many seconds will the faster team win by?"
    spr.targets[0].text = "difference in total time between the two teams"

    unit = Explainer._unit_for_answer(
        {
            "target": "difference in total time between the two teams",
            "display_name": "difference in total time between the two teams",
            "question_id": "q1",
            "value": "2",
        },
        spr,
    )

    assert unit == "seconds"


def test_explainer_problem_overview_prefers_full_problem_text() -> None:
    spr = math7_spr()
    spr.problem_text = "鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。"
    spr.problem_stem = "这是一个经典的鸡兔同笼问题。"
    segments = Explainer()._problem_segments(spr, SolutionChain(problem_id="p", chain_id="c", final_answer="", confidence=1.0))

    narration = segments[0]["narration"]

    assert "题目是：鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。" in narration
    assert "题目是：这是一个经典的鸡兔同笼问题。" not in narration


def test_explainer_problem_overview_ignores_dataset_wrapper_lines() -> None:
    spr = math7_spr()
    spr.problem_text = (
        "GSM8K Math Word Problem\n"
        "sample_id: gsm8k_test_000509\n"
        "Sophia bought 3 cookies and then 1 more cookie. How many cookies did Sophia buy"
    )

    segments = Explainer()._problem_segments(
        spr,
        SolutionChain(problem_id="p", chain_id="c", final_answer="", confidence=1.0),
        output_language="en-US",
    )

    narration = segments[0]["narration"]
    assert "GSM8K Math Word Problem" not in narration
    assert "sample_id" not in narration
    assert "The problem says: Sophia bought 3 cookies and then 1 more cookie. How many cookies did Sophia buy." in narration


def test_explainer_converts_all_provider_teaching_moves_including_answer() -> None:
    plan = TeachingPlan(
        problem_id="problem_provider",
        source_chain_id="chain_provider",
        planner_mode="deepseek",
        moves=[
            TeachingPlanMove(
                move_id="move_1",
                move_type="modeling",
                source_step_ids=["step_model"],
                narration="先说明题目里的数量关系。",
                pedagogical_actions=["HIGHLIGHT_CONDITION"],
                estimated_duration_ms=1000,
            ),
            TeachingPlanMove(
                move_id="move_2",
                move_type="answer",
                source_step_ids=["step_answer"],
                raw_expression="x = 66",
                display_expression="x = 66",
                narration="答：平日升国旗的护旗队员有66名。",
                pedagogical_actions=["FINAL_ANSWER_REVEAL"],
                estimated_duration_ms=1000,
            ),
        ],
        quality_report=TeachingPlanQualityReport(
            valid=True,
            status="passed",
            checked_move_count=2,
            expanded_step_ids=["step_model", "step_answer"],
            strategy_names=["deepseek_teaching_planner"],
        ),
        metadata={"provider_used": "deepseek"},
    )

    segments = Explainer()._step_segments(None, None, {}, plan)

    assert [segment["segment_type"] for segment in segments] == ["modeling", "answer"]
    assert segments[1]["linked_step_ids"] == ["step_answer"]
    assert segments[1]["math_expressions"] == ["x = 66"]


def test_explainer_links_provider_answer_move_when_final_answer_step_is_absent() -> None:
    chain = SolutionChain(
        problem_id="problem_provider",
        chain_id="chain_provider",
        final_answer="x = 52",
        confidence=1.0,
        steps=[
            SolutionStep(
                step_id="step_model",
                step_index=0,
                description="model",
                math_expression="x = 52",
                rule_name="modeling_from_tool_call",
                diagnostics={"tool_call_id": "solve_system"},
            ),
            SolutionStep(
                step_id="step_cas",
                step_index=1,
                description="solve",
                math_expression="x = 52",
                rule_name="cas_solve_equation_system",
                diagnostics={"tool_call_id": "solve_system"},
            ),
            SolutionStep(
                step_id="step_unsupported",
                step_index=2,
                description="fallback final answer",
                math_expression="x = 52",
                rule_name="unsupported_route",
            ),
        ],
    )
    plan = TeachingPlan(
        problem_id="problem_provider",
        source_chain_id="chain_provider",
        planner_mode="deepseek",
        moves=[
            TeachingPlanMove(
                move_id="move_answer",
                move_type="answer",
                source_step_ids=[],
                raw_expression="x = 52",
                display_expression="x = 52",
                narration="The final answer is 52.",
                pedagogical_actions=["FINAL_ANSWER_REVEAL"],
                estimated_duration_ms=1000,
            )
        ],
        quality_report=TeachingPlanQualityReport(
            valid=True,
            status="passed",
            checked_move_count=1,
            expanded_step_ids=[],
            strategy_names=["deepseek_teaching_planner"],
        ),
        metadata={"provider_used": "deepseek"},
    )

    segments = Explainer()._step_segments(
        None,
        chain,
        {
            "answers": [
                {
                    "target": "x",
                    "value": "52",
                    "student_label": "the requested quantity",
                    "source_tool_call_id": "solve_system",
                    "backend_confirmed": True,
                }
            ],
            "symbols": {},
            "output_language": "en-US",
        },
        plan,
        output_language="en-US",
    )

    assert segments[0]["segment_type"] == "answer"
    assert segments[0]["linked_step_ids"] == ["step_cas"]


def test_explainer_adds_variable_definition_when_provider_plan_omits_it() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)
    plan = TeachingPlan(
        problem_id=chain.problem_id,
        source_chain_id=chain.chain_id,
        planner_mode="deepseek",
        moves=[
            TeachingPlanMove(
                move_id="move_morning_sale",
                move_type="modeling",
                source_step_ids=["chain_main_step_2"],
                source_sentence="上午卖出总数的一半少4千克",
                meaning="上午卖出的苹果重量",
                raw_expression="T/2 - 4",
                display_expression="x ÷ 2 - 4",
                narration="题目说上午卖出总数的一半少4千克，所以先写出上午卖出的重量。",
                pedagogical_actions=["RENDER_EXPRESSION", "HIGHLIGHT_TERMS"],
                estimated_duration_ms=4000,
                pause_after_ms=200,
            )
        ],
        quality_report=TeachingPlanQualityReport(
            valid=True,
            status="passed",
            checked_move_count=1,
            expanded_step_ids=["chain_main_step_2"],
            strategy_names=["deepseek_teaching_planner"],
        ),
        metadata={"provider_used": "deepseek"},
    )

    script = Explainer().explain(spr, classification, chain, verification, teaching_plan=plan)

    assert script.status == "passed"
    variable_segments = [segment for segment in script.segments if segment.segment_type == "variable_definition"]
    assert variable_segments
    assert any("x" in segment.narration and "表示" in segment.narration for segment in variable_segments)


def test_explainer_removes_variable_definition_when_symbol_is_not_reused() -> None:
    segments = [
        {
            "segment_id": "seg_define_total",
            "segment_type": "variable_definition",
            "narration": "Let x represent the total number of stamps.",
            "math_expressions": ["x"],
            "metadata": {"display_symbol": "x"},
        },
        {
            "segment_id": "seg_model",
            "segment_type": "modeling",
            "narration": "Now use S for snowflake stamps and T for truck stamps.",
            "math_expressions": ["T = S + 3"],
            "metadata": {"display_expression": "T = S + 3"},
        },
    ]

    filtered, report = Explainer()._enforce_symbol_consistency(segments)

    assert [segment["segment_id"] for segment in filtered] == ["seg_model"]
    assert report["removed_variable_definition_count"] == 1
    assert report["removed_symbols"][0]["symbols"] == ["x"]


def test_explainer_keeps_variable_definition_when_symbol_is_reused() -> None:
    segments = [
        {
            "segment_id": "seg_define_x",
            "segment_type": "variable_definition",
            "narration": "Let x represent the starting amount.",
            "math_expressions": ["x"],
            "metadata": {"display_symbol": "x"},
        },
        {
            "segment_id": "seg_model",
            "segment_type": "modeling",
            "narration": "The ending amount is x plus 59.",
            "math_expressions": ["x + 59 = 78"],
            "metadata": {"display_expression": "x + 59 = 78"},
        },
    ]

    filtered, report = Explainer()._enforce_symbol_consistency(segments)

    assert [segment["segment_id"] for segment in filtered] == ["seg_define_x", "seg_model"]
    assert report["removed_variable_definition_count"] == 0
    assert report["kept_symbols"] == ["x"]


def test_explainer_generates_math10_video_ready_script() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)

    assert isinstance(script, ExplanationScript)
    assert script.status == "passed"
    assert script.consistency_report.valid is True
    assert {segment.segment_type for segment in script.segments} >= {
        "problem_overview",
        "variable_definition",
        "modeling",
        "calculation",
        "answer",
    }
    assert "recap" not in {segment.segment_type for segment in script.segments}
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )
    answer_segments = [segment for segment in script.segments if segment.segment_type == "answer"]
    problem_segment = next(segment for segment in script.segments if segment.segment_type == "problem_overview")

    assert "q1" not in narration_text
    assert "q2" not in narration_text
    assert "CAS" not in narration_text
    assert "Total apple weight" not in narration_text
    assert "Number of boxes" not in narration_text
    assert "N_boxes" not in narration_text
    assert "千克" in narration_text
    assert "箱" in narration_text
    assert len(answer_segments) == 2
    assert any("56 千克" in segment.narration for segment in answer_segments)
    assert any("7 箱" in segment.narration for segment in answer_segments)
    assert any("可以装的箱数是 7 箱" in segment.narration for segment in answer_segments)
    assert not any("原来运来的苹果重量是 7 箱" in segment.narration for segment in answer_segments)
    assert problem_segment.metadata["reads_problem_text"] is True
    assert problem_segment.estimated_duration_ms >= 7600
    assert (spr.problem_stem or spr.raw_text)[:10] in problem_segment.narration
    assert "T" not in expression_text
    assert "N_boxes" not in expression_text
    assert any(
        segment.metadata.get("internal_symbol") == "T"
        and segment.metadata.get("display_symbol") == "x"
        for segment in script.segments
        if segment.segment_type == "variable_definition"
    )
    assert any("FINAL_ANSWER_REVEAL" in segment.pedagogical_actions for segment in script.segments)
    assert any(anchor.anchor_id == "final_answer" for anchor in script.visual_anchors)
    assert all("x" not in segment.layout_hints for segment in script.segments)
    assert validate_explanation_script_dict(script.model_dump(mode="json"))


def test_explainer_expands_math10_remaining_quantity_modeling_step() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )
    expansion = script.metadata["teaching_expansion"]

    assert expansion["enabled"] is True
    assert expansion["strategy"] == "multi_strategy"
    assert expansion["strategies"] == ["linear_word_problem", "box_division_word_problem"]
    assert expansion["strategy_count"] == 2
    assert expansion["expanded_step_ids"]
    assert "先看上午" in narration_text
    assert "少 4 千克" in narration_text
    assert "上午卖出后" in narration_text
    assert "再看下午" in narration_text
    assert "少 2 千克" in narration_text
    assert "下午卖出后剩下的数量" in narration_text
    assert "题目说最后还剩 18 千克" in narration_text
    assert "每箱装 8 千克" in narration_text
    assert "用总重量除以 8" in narration_text
    assert "得到 7 箱" in narration_text
    assert "x ÷ 2 - 4" in expression_text
    assert "x - (x ÷ 2 - 4) = x ÷ 2 + 4" in expression_text
    assert "(x ÷ 2 + 4) ÷ 2 - 2" in expression_text
    assert "x ÷ 4 + 4" in expression_text
    assert "x ÷ 4 + 4 = 18" in expression_text
    assert "/" not in expression_text
    assert "*" not in expression_text
    assert "Total apple weight" not in narration_text
    assert "N_boxes" not in narration_text
    expanded_segments = [
        segment for segment in script.segments
        if segment.metadata.get("source") == "teaching_expansion_rule"
    ]
    assert expanded_segments
    assert all(segment.metadata.get("independent_verification") is False for segment in expanded_segments)
    assert all("verified_by" not in segment.metadata for segment in expanded_segments)
    calc_text = "\n".join(
        segment.narration for segment in script.segments if segment.segment_type == "calculation"
    )
    assert "得到 7 箱" in calc_text
    assert validate_explanation_script_dict(script.model_dump(mode="json"))

    eds = ScriptDirector().direct(script, spr=spr, scs=chain, verification_report=verification)
    assert validate_executable_director_script_dict(eds.model_dump(mode="json"))
    assert eds.quality_report.valid is True
    asset_ids = {asset.asset_id for asset in eds.assets}
    assert "asset_var_N_boxes" not in asset_ids
    assert "asset_expr_chain_main_step_3" not in asset_ids
    final_asset = next(asset for asset in eds.assets if asset.asset_id == "asset_final_answer")
    assert final_asset.metadata["expression"] == (
        "原来运来的苹果重量是 56 千克；可以装的箱数是 7 箱"
    )
    assert "N_boxes" not in final_asset.metadata["expression"]


def test_explainer_compacts_raw_remaining_equation_for_teaching() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(
        planner=DeepSeekPlanner(client=FakeMath10NumBoxesPlanningClient()),
        use_planner=True,
    ).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    teaching_expressions = [
        expression
        for segment in script.segments
        if segment.metadata.get("source") == "teaching_expansion_rule"
        for expression in segment.math_expressions
    ]

    assert "x ÷ 4 + 4" in teaching_expressions
    assert "(x ÷ 2 + 4) - ((x ÷ 2 + 4) ÷ 2 - 2) = x ÷ 4 + 4" not in teaching_expressions
    assert "x ÷ 4 + 4 = 18" in teaching_expressions
    assert not any(
        expression.startswith("x - (x ÷ 2 - 4)") and expression.endswith("= 18")
        for expression in teaching_expressions
    )
    eds = ScriptDirector().direct(script, spr=spr, scs=chain, verification_report=verification)
    asset_expressions = {
        asset.metadata.get("expression")
        for asset in eds.assets
        if asset.asset_type == "expression"
    }
    assert "x ÷ 4 + 4 = 18" in asset_expressions
    assert "x - (x/2 - 4) - ((x - (x/2 - 4))/2 - 2) = 18" not in asset_expressions


def test_explainer_expands_math9_equivalent_total_modeling_step() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath9PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    chain_data = chain.model_dump(mode="json")
    chain_data["metadata"]["variable_bindings"]["x"]["description"] = (
        "Number of pencils per class when distributed to 6 classes"
    )
    chain = SolutionChain.model_validate(chain_data)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    expansion = script.metadata["teaching_expansion"]
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )

    assert script.status == "passed"
    assert expansion["enabled"] is True
    assert "equivalent_total_distribution" in expansion["strategies"]
    assert "6 个班一共分走 6 乘以 x 支，还剩 12 支" in narration_text
    assert "每班 x 减 3 支" in narration_text
    assert "同一批铅笔的总数" in narration_text
    assert "6 × x + 12" in expression_text
    assert "8 × (x - 3)" in expression_text
    assert "6*x+12" not in expression_text
    assert validate_explanation_script_dict(script.model_dump(mode="json"))


def test_explainer_expands_math7_additive_total_modeling_step() -> None:
    spr = math7_spr()
    emr = math7_unknown_emr()
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath7PlanningClient()), use_planner=True).solve(
        emr,
        math7_classification_result(),
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=math7_classification_result())

    script = Explainer().explain(spr, math7_classification_result(), chain, verification)
    expansion = script.metadata["teaching_expansion"]
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )

    assert script.status == "passed"
    assert "additive_total_word_problem" in expansion["strategies"]
    assert "第一天看了 28 页" in narration_text
    assert "第二天是 28 加 x 页" in narration_text
    assert "第三天看的页数是第二天的 2 倍" in narration_text
    assert "28 + (28 + x) + 2 × (28 + x) = 120" in expression_text
    assert "2*(28+x)" not in expression_text


def test_explainer_generates_direct_emr_backed_script() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)
    chain = Solver().solve(parser_output.emr, classification, spr=parser_output.spr)
    verification = Verifier().verify(
        chain,
        spr=parser_output.spr,
        emr=parser_output.emr,
        classification_result=classification,
    )

    script = Explainer().explain(parser_output.spr, classification, chain, verification)

    assert script.status == "passed"
    assert any(segment.segment_type == "calculation" for segment in script.segments)
    assert any("x = 2" in segment.narration for segment in script.segments)
    assert script.consistency_report.valid is True


def test_explainer_returns_unsupported_script_when_verification_fails() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    chain_data = deepcopy(chain.model_dump(mode="json"))
    chain_data["metadata"]["final_answer_items"] = []
    corrupted = SolutionChain.model_validate(chain_data)
    verification = Verifier().verify(corrupted, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, corrupted, verification)

    assert script.status == "unsupported"
    assert script.consistency_report.valid is False
    assert script.segments[0].segment_type == "unsupported"
    assert "没有通过验证" in script.segments[0].narration


def test_explanation_consistency_reports_missing_step_reference() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)
    explainer = Explainer()
    script = explainer.explain(spr, classification, chain, verification)
    segment_data = [segment.model_dump(mode="json") for segment in script.segments]
    segment_data[1]["linked_step_ids"] = ["missing_step"]

    report = explainer._consistency_report(
        segment_data,
        [anchor.model_dump(mode="json") for anchor in script.visual_anchors],
        spr,
        chain,
    )

    assert report["valid"] is False
    assert "missing_step" in report["errors"][0]


def test_explanation_consistency_reports_student_facing_language_warnings() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)
    explainer = Explainer()
    script = explainer.explain(spr, classification, chain, verification)
    segment_data = [segment.model_dump(mode="json") for segment in script.segments]
    segment_data[0]["narration"] = "回答 q_1：Total apple weight is 56 kg. CAS checked it."

    report = explainer._consistency_report(
        segment_data,
        [anchor.model_dump(mode="json") for anchor in script.visual_anchors],
        spr,
        chain,
        script.metadata["presentation"],
    )

    warning_text = "\n".join(report["warnings"])
    assert "raw question id" in warning_text
    assert "unlocalized unit" in warning_text
    assert "English words" in warning_text
    assert "CAS" in warning_text


def test_explainer_labels_planner_local_symbol_when_spr_has_no_variables() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = []
    spr = type(math10_spr()).model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    narration_text = "\n".join(segment.narration for segment in script.segments)

    assert script.status == "passed"
    assert "设 x 表示原来运来的苹果重量" in narration_text
    assert "Number of boxes" not in narration_text
    assert any(
        segment.segment_type == "answer" and "可以装的箱数" in segment.narration
        for segment in script.segments
    )


def test_explainer_prefers_total_answer_over_intermediate_sold_amounts() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    chain_data = chain.model_dump(mode="json")
    chain_data["metadata"]["answer_bindings"] = [
        {
            "target": "S_AM",
            "display_name": "上午卖出的苹果重量",
            "value": "24",
            "unit": None,
            "source_tool_call_id": "solve_T",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
            "question_id": "q1",
        },
        {
            "target": "S_PM",
            "display_name": "下午卖出的苹果重量",
            "value": "14",
            "unit": None,
            "source_tool_call_id": "solve_T",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
            "question_id": "q1",
        },
        *chain_data["metadata"]["answer_bindings"],
    ]
    chain = SolutionChain.model_validate(chain_data)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    answer_text = "\n".join(
        segment.narration for segment in script.segments if segment.segment_type == "answer"
    )

    assert "原来运来的苹果重量是 56" in answer_text
    assert "上午卖出的苹果重量是 24" not in answer_text
    assert "下午卖出的苹果重量是 14" not in answer_text


def test_explainer_labels_math9_six_class_variable_precisely() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath9PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    chain_data = chain.model_dump(mode="json")
    chain_data["metadata"]["variable_bindings"]["x"]["description"] = (
        "Number of pencils per class when distributed to 6 classes"
    )
    chain = SolutionChain.model_validate(chain_data)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    variable_text = "\n".join(
        segment.narration for segment in script.segments if segment.segment_type == "variable_definition"
    )

    assert "平均分给 6 个班时每班分到的铅笔数量" in variable_text


def test_explainer_preserves_specific_sold_amount_variable_labels() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["problem_text"] = (
        "水果店运来一批苹果。上午卖出总数的一半少4千克，"
        "下午卖出剩下的一半少2千克，这时还剩18千克苹果。"
    )
    spr_data["questions"] = [
        {
            "id": "q1",
            "text": "水果店原来运来多少千克苹果？",
            "question_type": "compute",
            "target_ids": ["t1"],
        },
        {
            "id": "q2",
            "text": "如果每箱装8千克苹果，这些苹果一共可以装几箱？",
            "question_type": "compute",
            "target_ids": ["t2"],
        },
    ]
    spr_data["variables"] = [
        {"symbol": "x", "description": "原来运来的苹果总重量", "domain": "real_positive"},
        {"symbol": "Sold_AM", "description": "上午卖出的苹果重量", "domain": "real_positive"},
        {"symbol": "Sold_PM", "description": "下午卖出的苹果重量", "domain": "real_positive"},
    ]
    spr_data["targets"] = [
        {"id": "t1", "text": "原来运来的苹果总重量", "target_type": "compute"},
        {"id": "t2", "text": "可以装的箱数", "target_type": "compute"},
    ]
    spr = type(math10_spr()).model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10SoldBreakdownPlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    narration_text = "\n".join(segment.narration for segment in script.segments)
    variable_labels = {
        segment.metadata.get("internal_symbol"): segment.metadata.get("student_label")
        for segment in script.segments
        if segment.segment_type == "variable_definition"
    }

    assert script.status == "passed"
    assert script.consistency_report.valid is True
    assert variable_labels["Sold_AM"] == "上午卖出的苹果重量"
    assert variable_labels["Sold_PM"] == "下午卖出的苹果重量"
    assert variable_labels["x"] == "原来运来的苹果重量"
    assert "设 x 表示原来运来的苹果重量" in narration_text
    assert "设 y 表示上午卖出的苹果重量" in narration_text
    assert "设 z 表示下午卖出的苹果重量" in narration_text
    assert "Multiple displayed variables share student_label" not in "\n".join(
        script.consistency_report.warnings
    )


def test_explainer_uses_variable_bindings_for_chicken_rabbit_labels() -> None:
    spr = math7_spr()
    spr_data = spr.model_dump(mode="json")
    spr_data.update(
        {
            "problem_text": "鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。",
            "problem_stem": "鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。",
            "questions": [
                {
                    "id": "q1",
                    "text": "问鸡和兔各有多少只",
                    "question_type": "compute",
                    "target_ids": ["t1", "t2"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "number of chickens", "domain": "integer"},
                {"symbol": "y", "description": "number of rabbits", "domain": "integer"},
            ],
            "targets": [
                {"id": "t1", "text": "number of chickens", "target_type": "unknown"},
                {"id": "t2", "text": "number of rabbits", "target_type": "unknown"},
            ],
        }
    )
    spr = type(spr).model_validate(spr_data)
    chain = SolutionChain.model_validate(
        {
            "problem_id": "problem_chicken_rabbit",
            "chain_id": "chain_main",
            "steps": [],
            "final_answer": "x = 23, y = 12",
            "confidence": 0.85,
            "metadata": {
                "variable_bindings": {
                    "x": {
                        "symbol": "x",
                        "description": "number of chickens",
                        "variable_role": "primary_unknown",
                        "used_in_computation": True,
                    },
                    "y": {
                        "symbol": "y",
                        "description": "number of rabbits",
                        "variable_role": "final_target",
                        "used_in_computation": True,
                    },
                },
                "answer_bindings": [
                    {
                        "target": "x",
                        "display_name": "number of chickens",
                        "value": "23",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    },
                    {
                        "target": "y",
                        "display_name": "number of rabbits",
                        "value": "12",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    },
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, chain)

    assert presentation["symbols"]["x"]["student_label"] == "鸡的数量"
    assert presentation["symbols"]["y"]["student_label"] == "兔的数量"
    assert [item["student_label"] for item in presentation["answers"]] == ["鸡的数量", "兔的数量"]
    assert Explainer()._friendly_final_answer_text(chain, presentation) == "鸡的数量是 23；兔的数量是 12"


def test_explainer_reports_duplicate_displayed_variable_labels() -> None:
    warnings = Explainer._duplicate_student_label_warnings(
        {
            "symbols": {
                "a": {"display_symbol": "x", "student_label": "同一个量"},
                "b": {"display_symbol": "y", "student_label": "同一个量"},
                "c": {"display_symbol": None, "student_label": "同一个量"},
            }
        }
    )

    assert warnings == ["Multiple displayed variables share student_label=同一个量: a, b."]


def test_explainer_selects_math9_total_pencils_and_cleans_numbered_label() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath9PlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    chain_data = chain.model_dump(mode="json")
    chain_data["metadata"]["answer_bindings"] = [
        {
            "target": "T",
            "display_name": "Total number of pencils",
            "value": "120",
            "unit": "piece",
            "source_tool_call_id": "calculate_T",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
            "question_id": None,
        },
        {
            "target": "x",
            "display_name": "Number of pencils per class in the first scenario (dividing among 6 classes)",
            "value": "18",
            "unit": "piece",
            "source_tool_call_id": "solve_x",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
            "question_id": "q1",
        },
    ]
    chain = SolutionChain.model_validate(chain_data)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )
    answer_text = "\n".join(
        segment.narration for segment in script.segments if segment.segment_type == "answer"
    )

    assert script.status == "passed"
    assert "买来的铅笔总数是 120 个" in answer_text
    assert "18 个" not in answer_text
    assert "设 y 表示买来的铅笔总数" not in narration_text
    assert "每班分到的铅笔数量" in narration_text
    assert "y = 6*y + 12" not in expression_text
    assert "y = 120, y = 18" not in expression_text
    assert "(6)" not in narration_text
    assert "dividing" not in narration_text
    assert script.metadata["presentation"]["selected_answer_targets"] == ["T"]


def test_explainer_cleans_parenthesized_english_number_fragments() -> None:
    cleaned = Explainer._clean_student_label_text(
        "Number of pencils per class in the first scenario (dividing among 6 classes) 学校一共买来多少支铅笔？"
    )

    assert "(6)" not in cleaned
    assert Explainer._student_label_from_text("学校一共买来多少支铅笔？", fallback="要求的数量") == "买来的铅笔总数"


def test_explainer_labels_boxes_before_total_apples_and_ribbon_units() -> None:
    assert Explainer._student_label_from_text("如果每箱装8千克苹果，这些苹果一共可以装几箱？", fallback="要求的数量") == "可以装的箱数"
    assert Explainer._student_label_from_text("这根彩带原来长多少米？", fallback="要求的数量") == "彩带原来的长度"
    assert Explainer._unit_for_answer(
        {"target": "x", "display_name": "Original length of the ribbon in meters", "question_id": None},
        math10_spr(),
    ) == "m"


def test_explainer_binds_answer_to_scs_final_answer_over_intermediate_items() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["problem_text"] = "Mike had some bandages. He used 59 and had 19 left. How many bandages did he start with?"
    spr_data["problem_stem"] = spr_data["problem_text"]
    spr_data["questions"] = [
        {
            "id": "q_final",
            "text": "How many bandages did he start with?",
            "question_type": "compute",
            "target_ids": ["x"],
        }
    ]
    spr_data["targets"] = [{"id": "x", "text": "starting bandages", "target_type": "compute"}]
    spr = type(math10_spr()).model_validate(spr_data)
    chain = SolutionChain(
        problem_id="p",
        chain_id="c",
        final_answer="x = 19.0000000000000",
        confidence=1.0,
        metadata={
            "answer_bindings": [
                {
                    "target": "net_change",
                    "value": "59",
                    "display_name": "net change",
                    "unit": "bandages",
                    "question_id": "q_final",
                }
            ]
        },
    )

    answers = Explainer()._presentation_context(spr, chain, output_language="en-US")["answers"]

    assert answers[0]["target"] == "x"
    assert answers[0]["value"] == "19"
    assert answers[0]["unit"] == "bandages"


def test_explainer_collapses_repeated_single_question_final_answer_targets() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["problem_text"] = (
        "A bird flew south, then north, then south again. "
        "What is the distance, in miles, between the bird's northern and southern homes?"
    )
    spr_data["problem_stem"] = spr_data["problem_text"]
    spr_data["questions"] = [
        {
            "id": "q1",
            "text": "What is the distance, in miles, between the bird's northern and southern homes?",
            "question_type": "compute",
            "target_ids": ["t1"],
        }
    ]
    spr_data["targets"] = [{"id": "t1", "text": "distance between homes", "target_type": "compute"}]
    spr = type(math10_spr()).model_validate(spr_data)
    chain = SolutionChain(
        problem_id="p",
        chain_id="c",
        final_answer=(
            "d_north_south_1 = 300；d_north_south_1 = 36；"
            "d_north_south_1 = 110；d_north_south_1 = 374"
        ),
        confidence=1.0,
        metadata={
            "answer_bindings": [
                {
                    "target": "d_north_south_1",
                    "value": "300",
                    "display_name": "Distance flown south in the first leg",
                    "backend_confirmed": True,
                },
                {
                    "target": "d_north_south_1",
                    "value": "36",
                    "display_name": "Distance flown south in the first leg",
                    "backend_confirmed": True,
                },
                {
                    "target": "d_north_south_1",
                    "value": "110",
                    "display_name": "Distance flown south in the first leg",
                    "backend_confirmed": True,
                },
                {
                    "target": "d_north_south_1",
                    "value": "374",
                    "display_name": "Distance flown south in the first leg",
                    "backend_confirmed": True,
                },
            ]
        },
    )

    verification = VerificationReport(
        problem_id="p",
        chain_id="c",
        valid=True,
        status="passed",
        verification_mode="tool_call_backed",
        scs_schema_valid=True,
        backend_steps_valid=True,
        final_answer_valid=True,
        summary="passed",
    )

    script = Explainer().explain(spr, math7_classification_result(), chain, verification, output_language="en-US")
    answer_segments = [segment for segment in script.segments if segment.segment_type == "answer"]

    assert len(answer_segments) == 1
    assert "374" in answer_segments[0].narration
    assert "miles" in answer_segments[0].narration
    assert "first leg" not in answer_segments[0].narration
    assert "300" not in answer_segments[0].narration
    assert "36" not in answer_segments[0].narration
    assert "110" not in answer_segments[0].narration


def test_explainer_english_final_answer_narration_is_natural_for_single_question() -> None:
    narration = Explainer._english_final_answer_narration(
        label="the total money saved",
        value="2",
        unit_text=" dollars",
        question_label=None,
        option_label=None,
    )

    assert narration == "The final answer is 2 dollars. This is the total money saved."
    assert "Question 1" not in narration


def test_explainer_english_final_answer_narration_adds_article_to_spoken_label() -> None:
    narration = Explainer._english_final_answer_narration(
        label="total time taken to return home",
        value="15",
        unit_text=" minutes",
        question_label=None,
        option_label=None,
    )

    assert narration == "The final answer is 15 minutes. This is the total time taken to return home."


def test_explainer_english_answer_label_prefers_savings_over_generic_total() -> None:
    spr_data = math7_spr().model_dump(mode="json")
    spr_data["questions"] = [
        {
            "id": "q1",
            "text": "Calculate the total savings from buying in bunches rather than individually.",
            "question_type": "compute",
            "target_ids": ["Savings"],
        }
    ]
    spr = type(math7_spr()).model_validate(spr_data)

    label = Explainer._english_student_label_for_answer(
        {"target": "Savings", "question_id": "q1", "source": "scs.final_answer"},
        spr,
    )

    assert label == "the total money saved"


def test_explainer_prefers_problem_unit_over_candidate_unit() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["problem_text"] = "A box weighs 260 pounds. What is the total weight in pounds?"
    spr_data["problem_stem"] = spr_data["problem_text"]
    spr_data["questions"] = [
        {
            "id": "q_weight",
            "text": "What is the total weight in pounds?",
            "question_type": "compute",
            "target_ids": ["total_weight"],
        }
    ]
    spr_data["targets"] = [{"id": "total_weight", "text": "total weight in pounds", "target_type": "compute"}]
    spr = type(math10_spr()).model_validate(spr_data)

    assert (
        Explainer._unit_for_answer(
            {"target": "total_weight", "display_name": "total weight", "unit": "kg", "question_id": "q_weight"},
            spr,
        )
        == "pounds"
    )


def test_explainer_recap_does_not_repeat_final_answer_number() -> None:
    spr = math10_spr()
    chain = SolutionChain(
        problem_id="p",
        chain_id="c",
        final_answer="answer = 260",
        confidence=1.0,
        metadata={"answer_bindings": [{"target": "answer", "value": "260", "question_id": "q1"}]},
    )

    presentation = Explainer()._presentation_context(spr, chain, output_language="en-US")
    recap = Explainer()._english_recap_segment(chain, presentation)

    assert "260" not in recap["narration"]
    assert recap["math_expressions"] == []
    assert "FINAL_ANSWER_REVEAL" not in recap["pedagogical_actions"]


def test_explainer_warns_for_untraceable_formula_number() -> None:
    spr = math10_spr()
    chain = SolutionChain(problem_id="p", chain_id="c", final_answer="answer = 18", confidence=1.0)
    segments = [
        {
            "segment_id": "seg_bad_formula",
            "segment_type": "calculation",
            "linked_step_ids": [],
            "narration": "Compute the local result.",
            "math_expressions": ["x = 999"],
            "visual_refs": [],
            "metadata": {"source_sentence": "known value", "meaning": "local result"},
        }
    ]

    report = Explainer()._consistency_report(segments, [], spr, chain)

    assert any("formula number 999" in warning for warning in report["warnings"])


def test_explainer_display_expression_replaces_symbols_atomically() -> None:
    expression = Explainer._display_expression(
        "T = 6*x + 12 ; T/8 = x - 3 ; T = 120, x = 18",
        {
            "symbols": {
                "T": {"display_symbol": "x"},
                "x": {"display_symbol": "y"},
            }
        },
    )

    assert expression == "x = 6 × y + 12 ; x ÷ 8 = y - 3 ; x = 120, y = 18"


def test_explainer_uses_unique_display_symbols_for_remaining_amount() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量（千克）",
            "domain": "positive_real",
        },
        {
            "symbol": "x_rem",
            "description": "上午卖出后剩下的苹果重量（千克）",
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
    chain = Solver(planner=DeepSeekPlanner(client=FakeMath10RemainingPlanningClient()), use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    script = Explainer().explain(spr, classification, chain, verification)
    narration_text = "\n".join(segment.narration for segment in script.segments)
    expression_text = "\n".join(
        expression
        for segment in script.segments
        for expression in segment.math_expressions
    )
    presentation_symbols = script.metadata["presentation"]["symbols"]

    assert script.status == "passed"
    assert presentation_symbols["x"]["display_symbol"] == "x"
    assert presentation_symbols["x_rem"]["display_symbol"] == "y"
    assert "设 x 表示原来运来的苹果重量" in narration_text
    assert "设 y 表示上午卖出后剩下的苹果重量" in narration_text
    assert "y = x ÷ 2 + 4" in expression_text
    assert "y ÷ 2 + 2 = 18" in expression_text
    assert "x = x ÷ 2 + 4" not in expression_text
    assert "后端确认" not in narration_text
