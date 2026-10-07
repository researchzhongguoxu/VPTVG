import json
import time
from typing import Any

from mathexplain.agents.explainer.teaching_planner import (
    DeepSeekTeachingPlannerProvider,
    RuleBasedTeachingPlanner,
    TeachingPlanQualityGate,
)
from mathexplain.agents.explainer.teaching_expansion import StudentMathFormatter
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.teaching_plan import TeachingPlan
from mathexplain.schemas.verification import VerificationReport


class FakeTeachingPlannerClient:
    provider = "deepseek"
    model = "fake-deepseek-v4-pro"

    def __init__(self, payload: dict[str, Any] | str) -> None:
        self.payload = payload
        self.calls = 0

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "TeachingPlan" in system_prompt
        assert "verified_solution" in user_prompt
        self.calls += 1
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False)


class SlowTeachingPlannerClient(FakeTeachingPlannerClient):
    timeout_seconds = 0.01

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        time.sleep(0.05)
        return super().generate_text(system_prompt, user_prompt)


class FakeTeachingPlanProvider:
    def __init__(self, plan: TeachingPlan) -> None:
        self._plan = plan
        self.calls = 0

    def plan(self, **kwargs: Any) -> TeachingPlan:  # type: ignore[no-redef]
        self.calls += 1
        return self._plan


class FailingTeachingPlanProvider:
    def __init__(self, raw_response: str = '{"moves": []}') -> None:
        self.last_raw_response = raw_response
        self.calls = 0

    def plan(self, **kwargs: Any) -> TeachingPlan:  # type: ignore[no-redef]
        self.calls += 1
        raise ValueError("provider failed")


def passed_verification(problem_id: str = "problem_1") -> VerificationReport:
    return VerificationReport.model_validate(
        {
            "problem_id": problem_id,
            "chain_id": "chain_main",
            "valid": True,
            "status": "passed",
            "verification_mode": "tool_call_backed",
            "scs_schema_valid": True,
            "backend_steps_valid": True,
            "final_answer_valid": True,
            "summary": "ok",
        }
    )


def spr_with_text(problem_id: str, text: str, variables: list[dict]) -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "test.png"},
            "problem_text": text,
            "problem_stem": text,
            "questions": [],
            "variables": variables,
            "problem_type": "word_problem",
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": problem_id},
        }
    )


def scs_with_modeling_step(problem_id: str, expression: str, diagnostics: dict | None = None) -> SolutionChain:
    return SolutionChain.model_validate(
        {
            "problem_id": problem_id,
            "chain_id": "chain_main",
            "confidence": 0.9,
            "steps": [
                {
                    "step_id": "step_model",
                    "step_index": 0,
                    "description": "建立关系",
                    "math_expression": expression,
                    "rule_name": "modeling_from_tool_call",
                    "diagnostics": {
                        "math_role": "modeling",
                        "explainer_ready": True,
                        "source": "planner_tool_call",
                        **(diagnostics or {}),
                    },
                }
            ],
        }
    )


def test_rule_based_teaching_planner_expands_ribbon_remainder_multiple_without_operation() -> None:
    spr = spr_with_text(
        "problem_ribbon",
        "一根彩带长 x 米，剪去 18 米后，剩下的长度正好是剪去部分的 3 倍。这根彩带原来长多少米？",
        [{"symbol": "x", "description": "彩带原来的长度", "domain": "positive_real"}],
    )
    scs = scs_with_modeling_step("problem_ribbon", "x - 18 = 3*18")
    presentation = {"symbols": {"x": {"display_symbol": "x"}}}

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_ribbon"), presentation)

    assert isinstance(plan, TeachingPlan)
    assert plan.quality_report.strategy_names == ["ribbon_remainder_multiple"]
    assert plan.quality_report.fallback_step_ids == []
    text = "\n".join(move.narration for move in plan.moves)
    expressions = [move.display_expression for move in plan.moves]
    assert "剪去了 18 米" in text
    assert "剩下的长度" in text
    assert "3 倍" in text
    assert "x - 18 = 3 × 18" in expressions


def test_rule_based_teaching_planner_expands_ribbon_two_step_arithmetic_without_leaking_names() -> None:
    spr = spr_with_text(
        "problem_ribbon",
        "一根彩带剪去 18 米后，剩下的长度正好是剪去部分的 3 倍。这根彩带原来长多少米？",
        [
            {"symbol": "L_{remaining}", "description": "剩下的长度", "domain": "positive_real"},
            {"symbol": "L_{total}", "description": "彩带原来的长度", "domain": "positive_real"},
        ],
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "problem_ribbon",
            "chain_id": "chain_main",
            "confidence": 0.9,
            "steps": [
                {
                    "step_id": "step_remaining",
                    "step_index": 0,
                    "description": "计算剩下的长度",
                    "math_expression": "3*18",
                    "rule_name": "modeling_from_tool_call",
                    "diagnostics": {
                        "math_role": "modeling",
                        "explainer_ready": True,
                        "source": "planner_tool_call",
                        "normalized_request": {
                            "id": "calc_remaining",
                            "operation": "evaluate",
                            "target": "remaining_length",
                        },
                        "evidence_span": "剩下的长度正好是剪去部分的 3 倍",
                    },
                },
                {
                    "step_id": "step_total",
                    "step_index": 1,
                    "description": "计算原来的长度",
                    "math_expression": "18 + remaining_length",
                    "rule_name": "modeling_from_tool_call",
                    "diagnostics": {
                        "math_role": "modeling",
                        "explainer_ready": True,
                        "source": "planner_tool_call",
                        "normalized_request": {
                            "id": "calc_total",
                            "operation": "evaluate",
                            "substitutions": {"remaining_length": "$calc_remaining.remaining_length"},
                            "target": "total_length",
                        },
                        "evidence_span": "一根彩带剪去 18 米后，剩下的长度正好是剪去部分的 3 倍",
                    },
                },
            ],
            "metadata": {
                "execution_context": {
                    "calc_remaining": {"remaining_length": "54"},
                    "calc_total": {"total_length": "72"},
                }
            },
        }
    )
    presentation = {"symbols": {}}

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_ribbon"), presentation)

    assert plan.quality_report.strategy_names == ["ribbon_arithmetic"]
    assert plan.quality_report.fallback_step_ids == []
    expressions = [move.display_expression for move in plan.moves]
    text = "\n".join(move.narration for move in plan.moves)
    assert expressions == ["18 × 3", "18 + 54 = 72"]
    assert "remaining_length" not in text
    assert "total_length" not in text


def test_rule_based_teaching_planner_expands_math9_system_shape() -> None:
    spr = spr_with_text(
        "problem_math9",
        "学校买来一些铅笔，如果平均分给6个班，还剩12支；如果平均分给8个班，则每班比原来少3支。",
        [
            {"symbol": "T", "description": "买来的铅笔总数", "domain": "integer"},
            {"symbol": "x", "description": "第一种分法每班铅笔数", "domain": "integer"},
        ],
    )
    scs = scs_with_modeling_step("problem_math9", "T = 6*x + 12 ; T/8 = x - 3")
    presentation = {
        "symbols": {
            "T": {"display_symbol": "x"},
            "x": {"display_symbol": "y"},
        }
    }

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_math9"), presentation)

    assert plan.quality_report.strategy_names == ["equivalent_total_distribution"]
    assert plan.quality_report.fallback_step_ids == []
    expressions = [move.display_expression for move in plan.moves]
    narration = "\n".join(move.narration for move in plan.moves)
    assert "6 × y + 12" in expressions
    assert "8 × (y - 3)" in expressions
    assert "6 × y + 12 = 8 × (y - 3)" in expressions
    assert "同一批铅笔的总数" in narration


def test_rule_based_teaching_planner_expands_math9_three_variable_system_shape() -> None:
    spr = spr_with_text(
        "problem_math9",
        "学校买来一些铅笔。如果平均分给 6 个班，每班分到同样多，还剩 12 支；如果平均分给 8 个班，那么每班比平均分给 6 个班时少 3 支。学校一共买来多少支铅笔？",
        [
            {"symbol": "T", "description": "买来的铅笔总数", "domain": "integer"},
            {"symbol": "x", "description": "平均分给 6 个班时每班铅笔数", "domain": "integer"},
            {"symbol": "y", "description": "平均分给 8 个班时每班铅笔数", "domain": "integer"},
        ],
    )
    scs = scs_with_modeling_step("problem_math9", "T = 6*x + 12 ; y = x - 3 ; T = 8*y")
    presentation = {
        "symbols": {
            "T": {"display_symbol": "x"},
            "x": {"display_symbol": "y"},
            "y": {"display_symbol": "z"},
        }
    }

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_math9"), presentation)

    assert plan.quality_report.strategy_names == ["equivalent_total_distribution"]
    assert plan.quality_report.fallback_step_ids == []
    expressions = [move.display_expression for move in plan.moves]
    assert "6 × y + 12" in expressions
    assert "8 × (y - 3)" in expressions
    assert "6 × y + 12 = 8 × (y - 3)" in expressions


def test_rule_based_teaching_planner_expands_meeting_speed_ratio() -> None:
    spr = spr_with_text(
        "problem_math11",
        "甲、乙两地相距900千米，一辆客车和一辆货车同时从两地相对开出，5小时相遇。已知客车和货车的速度比是5:4，客车平均每小时行多少千米？",
        [
            {"symbol": "v_p", "description": "Average speed of the passenger car", "domain": "positive_real"},
            {"symbol": "v_f", "description": "Average speed of the freight truck", "domain": "positive_real"},
        ],
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "problem_math11",
            "chain_id": "chain_main",
            "confidence": 0.9,
            "steps": [
                {
                    "step_id": "step_speed_sum",
                    "step_index": 0,
                    "description": "计算速度和",
                    "math_expression": "900/5",
                    "rule_name": "modeling_from_tool_call",
                    "diagnostics": {
                        "math_role": "modeling",
                        "explainer_ready": True,
                        "source": "planner_tool_call",
                        "normalized_request": {"id": "step1", "operation": "evaluate"},
                        "evidence_span": "甲、乙两地相距900千米，一辆客车和一辆货车同时从两地相对开出，5小时相遇",
                    },
                },
                {
                    "step_id": "step_ratio",
                    "step_index": 1,
                    "description": "根据速度比求客车速度",
                    "math_expression": "v_p + v_f = $step1.sum_speed ; v_p / v_f = 5/4",
                    "rule_name": "modeling_from_tool_call",
                    "diagnostics": {
                        "math_role": "modeling",
                        "explainer_ready": True,
                        "source": "planner_tool_call",
                        "normalized_request": {"id": "step2", "operation": "solve_equation_system"},
                        "evidence_span": "已知客车和货车的速度比是5:4",
                    },
                },
            ],
            "metadata": {"execution_context": {"step1": {"sum_speed": "180"}}},
        }
    )
    presentation = {"symbols": {"v_p": {"display_symbol": "x"}, "v_f": {"display_symbol": "y"}}}

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_math11"), presentation)

    assert plan.quality_report.strategy_names == ["meeting_speed_ratio"]
    assert plan.quality_report.fallback_step_ids == []
    expressions = [move.display_expression for move in plan.moves]
    assert "900 ÷ 5 = 180" in expressions
    assert "5 + 4 = 9" in expressions
    assert "180 × 5 ÷ 9 = 100" in expressions
    assert "$step1" not in "\n".join(expressions)


def test_rule_based_teaching_planner_generates_fallback_move() -> None:
    spr = spr_with_text(
        "problem_unknown",
        "一个普通应用题。",
        [{"symbol": "x", "description": "未知数", "domain": "real"}],
    )
    scs = scs_with_modeling_step("problem_unknown", "x + 1 = 2")
    presentation = {"symbols": {"x": {"display_symbol": "x"}}}

    plan = RuleBasedTeachingPlanner().plan(spr, scs, passed_verification("problem_unknown"), presentation)

    assert plan.quality_report.strategy_names == []
    assert plan.quality_report.fallback_step_ids == ["step_model"]
    assert plan.moves[0].move_type == "fallback"
    assert plan.moves[0].display_expression == "x + 1 = 2"


def valid_deepseek_plan(problem_id: str = "problem_unknown") -> TeachingPlan:
    return TeachingPlan.model_validate(
        {
            "problem_id": problem_id,
            "source_chain_id": "chain_main",
            "planner_mode": "deepseek",
            "moves": [
                {
                    "move_id": "deepseek_move_1",
                    "move_type": "modeling",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "题目给出一个数量关系。",
                    "meaning": "解释局部数量关系",
                    "raw_expression": "x + 1 = 2",
                    "display_expression": "x + 1 = 2",
                    "narration": "先看题目给出的数量关系，再把它写成一个简单方程。",
                    "pedagogical_actions": ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS"],
                    "estimated_duration_ms": 6200,
                    "pause_after_ms": 600,
                }
            ],
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_move_count": 1,
                "expanded_step_ids": ["step_model"],
                "fallback_step_ids": [],
                "strategy_names": ["deepseek_teaching_planner"],
            },
            "metadata": {"provider_used": "deepseek"},
        }
    )


def test_auto_teaching_planner_uses_deepseek_provider_for_fallback_rule_plan() -> None:
    spr = spr_with_text(
        "problem_unknown",
        "一个普通应用题。",
        [{"symbol": "x", "description": "未知数", "domain": "real"}],
    )
    scs = scs_with_modeling_step("problem_unknown", "x + 1 = 2")
    provider = FakeTeachingPlanProvider(valid_deepseek_plan("problem_unknown"))

    plan = RuleBasedTeachingPlanner(provider_mode="auto", provider=provider).plan(
        spr,
        scs,
        passed_verification("problem_unknown"),
        {"symbols": {"x": {"display_symbol": "x"}}},
    )

    assert provider.calls == 1
    assert plan.planner_mode == "deepseek"
    assert plan.metadata["provider_used"] == "deepseek"
    assert plan.metadata["provider_attempted"] is True


def test_auto_teaching_planner_keeps_good_rule_plan_without_deepseek_call() -> None:
    spr = spr_with_text(
        "problem_ribbon",
        "一根彩带剪去 18 米后，剩下的长度正好是剪去部分的 3 倍。这根彩带原来长多少米？",
        [{"symbol": "x", "description": "彩带原来的长度", "domain": "positive_real"}],
    )
    scs = scs_with_modeling_step("problem_ribbon", "x - 18 = 3*18")
    provider = FakeTeachingPlanProvider(valid_deepseek_plan("problem_ribbon"))

    plan = RuleBasedTeachingPlanner(provider_mode="auto", provider=provider).plan(
        spr,
        scs,
        passed_verification("problem_ribbon"),
        {"symbols": {"x": {"display_symbol": "x"}}},
    )

    assert provider.calls == 0
    assert plan.planner_mode == "rule_based"
    assert plan.metadata["provider_used"] == "rule"
    assert plan.metadata["quality_gate_status"] == "passed"


def test_deepseek_teaching_planner_provider_returns_valid_math13_plan() -> None:
    spr = spr_with_text(
        "problem_math13",
        "小亮读一本书，已经读了全书的1/4，如果再读15页，则读过的页数与未读的页数之比是2:3，这本书有多少页？",
        [{"symbol": "x", "description": "Total number of pages in the book", "domain": "positive_integer"}],
    )
    scs = scs_with_modeling_step(
        "problem_math13",
        "(x/4 + 15)/(3*x/4 - 15) = 2/3",
        {
            "normalized_request": {
                "id": "eq1",
                "operation": "solve_equation_system",
                "target": "x",
            }
        },
    )
    scs_data = scs.model_dump(mode="json")
    scs_data["metadata"] = {
        "final_answer_items": [
            {
                "target": "x",
                "value": "100",
                "source_tool_call_id": "eq1",
                "backend_operation": "solve_equation_system",
                "backend_confirmed": True,
            }
        ]
    }
    scs = SolutionChain.model_validate(scs_data)
    client = FakeTeachingPlannerClient(
        {
            "moves": [
                {
                    "move_id": "read_pages_1",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "已经读了全书的1/4",
                    "meaning": "已经读的页数",
                    "raw_expression": "x/4",
                    "display_expression": "x ÷ 4",
                    "narration": "设全书共有 x 页，已经读了全书的四分之一，所以已经读的是 x ÷ 4 页。",
                },
                {
                    "move_id": "read_pages_2",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "如果再读15页",
                    "meaning": "再读后读过的页数",
                    "raw_expression": "x/4 + 15",
                    "display_expression": "x ÷ 4 + 15",
                    "narration": "如果再读 15 页，读过的页数就变成 x ÷ 4 + 15。",
                },
                {
                    "move_id": "unread_pages",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "读过的页数与未读的页数之比是2:3",
                    "meaning": "未读的页数",
                    "raw_expression": "x - (x/4 + 15)",
                    "display_expression": "x - (x ÷ 4 + 15)",
                    "narration": "未读页数就是总页数减去已经读过的页数。",
                },
            ]
        }
    )

    plan = DeepSeekTeachingPlannerProvider(client=client).plan(
        spr=spr,
        scs=scs,
        verification_report=passed_verification("problem_math13"),
        presentation={"symbols": {"x": {"display_symbol": "x"}}},
        rule_plan=None,
    )

    assert client.calls == 1
    assert plan.planner_mode == "deepseek"
    assert plan.metadata["provider_used"] == "deepseek"
    assert [move.display_expression for move in plan.moves] == [
        "x ÷ 4",
        "x ÷ 4 + 15",
        "x - (x ÷ 4 + 15)",
    ]


def test_deepseek_provider_normalizes_common_llm_move_types() -> None:
    spr = spr_with_text(
        "problem_math13",
        "A book ratio problem.",
        [{"symbol": "x", "description": "Total pages", "domain": "positive_integer"}],
    )
    scs = scs_with_modeling_step("problem_math13", "x/4 + 15 = x*(2/5)")
    scs_data = scs.model_dump(mode="json")
    scs_data["metadata"] = {
        "final_answer_items": [
            {
                "target": "x",
                "value": "100",
                "source_tool_call_id": "solve_x",
                "backend_operation": "solve_equation_system",
                "backend_confirmed": True,
            }
        ]
    }
    scs = SolutionChain.model_validate(scs_data)
    client = FakeTeachingPlannerClient(
        {
            "moves": [
                {
                    "move_id": "m1",
                    "move_type": "set_variable",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "The whole book has x pages.",
                    "meaning": "Define the unknown.",
                    "raw_expression": "x",
                    "narration": "Let x represent the total number of pages.",
                },
                {
                    "move_id": "m2",
                    "move_type": "solve_equation",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "The read pages equal two fifths of the book.",
                    "meaning": "Solve the verified equation.",
                    "raw_expression": "x/4 + 15 = x*(2/5)",
                    "narration": "Now solve the equation.",
                },
                {
                    "move_id": "m3",
                    "move_type": "conclude_answer",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "The question asks for total pages.",
                    "meaning": "State the final answer.",
                    "raw_expression": "x = 100",
                    "narration": "So the book has 100 pages.",
                },
            ]
        }
    )

    plan = DeepSeekTeachingPlannerProvider(client=client).plan(
        spr=spr,
        scs=scs,
        verification_report=passed_verification("problem_math13"),
        presentation={"symbols": {"x": {"display_symbol": "x"}}},
        rule_plan=None,
    )

    assert [move.move_type for move in plan.moves] == ["modeling", "calculation", "answer"]
    assert plan.moves[1].display_expression == "x ÷ 4 + 15 = x × (2 ÷ 5)"


def test_deepseek_provider_normalizes_list_display_expression() -> None:
    spr = spr_with_text(
        "problem_math14",
        "平日护旗队员人数比每月第一天少5/16。",
        [{"symbol": "x", "description": "平日人数", "domain": "positive_integer"}],
    )
    scs = scs_with_modeling_step("problem_math14", "96 * (1 - 5/16)")
    client = FakeTeachingPlannerClient(
        {
            "moves": [
                {
                    "move_id": "m1",
                    "move_type": "calculation",
                    "source_step_ids": "step_model",
                    "source_sentence": "平日护旗队员人数比每月第一天少5/16。",
                    "meaning": "先算出平日人数占第一天人数的几分之几。",
                    "raw_expression": ["1 - 5/16", "96 * 11/16"],
                    "display_expression": ["1 - 5 ÷ 16 = 11 ÷ 16", "96 × 11 ÷ 16 = 66"],
                    "narration": "先把少去的部分扣掉，再计算平日人数。",
                    "pedagogical_actions": "RENDER_EXPRESSION",
                    "estimated_duration_ms": "5200",
                }
            ]
        }
    )

    plan = DeepSeekTeachingPlannerProvider(client=client).plan(
        spr=spr,
        scs=scs,
        verification_report=passed_verification("problem_math14"),
        presentation={"symbols": {"x": {"display_symbol": "x"}}},
        rule_plan=None,
    )

    assert plan.moves[0].source_step_ids == ["step_model"]
    assert plan.moves[0].raw_expression == "1 - 5/16；96 * 11/16"
    assert plan.moves[0].display_expression == "1 - 5 ÷ 16 = 11 ÷ 16\n96 × 11 ÷ 16 = 66"
    assert plan.moves[0].pedagogical_actions == ["RENDER_EXPRESSION"]
    assert any("display_expression was a list" in warning for warning in plan.metadata["normalization_warnings"])


def test_deepseek_provider_accepts_later_long_equation_with_explanation() -> None:
    spr = spr_with_text(
        "problem_math13",
        "一本书已经读了1/4，再读15页后，已读和未读的比是2:3。",
        [{"symbol": "x", "description": "总页数", "domain": "positive_integer"}],
    )
    scs = scs_with_modeling_step("problem_math13", "(x/4 + 15)/(3*x/4 - 15) = 2/3")
    scs_data = scs.model_dump(mode="json")
    scs_data["metadata"] = {"final_answer_items": [{"target": "x", "value": "100"}]}
    scs = SolutionChain.model_validate(scs_data)
    client = FakeTeachingPlannerClient(
        {
            "moves": [
                {
                    "move_id": "m1",
                    "move_type": "modeling",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "已经读了全书的1/4。",
                    "meaning": "表示已经读的页数。",
                    "display_expression": "x ÷ 4",
                    "narration": "先表示已经读的页数。",
                },
                {
                    "move_id": "m2",
                    "move_type": "calculation",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "已读和未读的比是2:3。",
                    "meaning": "把已读和未读的比例写成方程。",
                    "display_expression": "(x ÷ 4 + 15) ÷ (3 × x ÷ 4 - 15) = 2 ÷ 3",
                    "narration": "再把比例关系写成方程。",
                },
                {
                    "move_id": "m3",
                    "move_type": "answer",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "问题求这本书共有多少页。",
                    "meaning": "回答最终问题。",
                    "display_expression": "x = 100",
                    "narration": "所以这本书共有100页。",
                },
            ]
        }
    )

    plan = DeepSeekTeachingPlannerProvider(client=client).plan(
        spr=spr,
        scs=scs,
        verification_report=passed_verification("problem_math13"),
        presentation={"symbols": {"x": {"display_symbol": "x"}}},
        rule_plan=None,
    )

    assert plan.moves[1].display_expression.startswith("(x ÷ 4 + 15)")


def test_deepseek_provider_failure_records_raw_response_preview() -> None:
    spr = spr_with_text(
        "problem_unknown",
        "一个普通应用题。",
        [{"symbol": "x", "description": "未知数", "domain": "real"}],
    )
    scs = scs_with_modeling_step("problem_unknown", "x + 1 = 2")
    provider = FailingTeachingPlanProvider(raw_response='{"moves":[{"bad":true}]}')

    plan = RuleBasedTeachingPlanner(provider_mode="deepseek", provider=provider).plan(
        spr,
        scs,
        passed_verification("problem_unknown"),
        {"symbols": {"x": {"display_symbol": "x"}}},
    )

    assert provider.calls == 1
    assert plan.metadata["provider_used"] == "rule"
    assert plan.metadata["provider_attempted"] is True
    assert plan.metadata["provider_fallback_reason"] == "provider failed"
    assert plan.metadata["provider_raw_response_preview"] == '{"moves":[{"bad":true}]}'


def test_student_math_formatter_does_not_replace_fraction_constants_as_symbols() -> None:
    expression = StudentMathFormatter.display(
        "x*(1/4)+15 = x*(2/5)",
        {
            "symbols": {
                "x": {"display_symbol": "x"},
                "1/4": {"display_symbol": "y"},
            }
        },
    )

    assert expression == "x × (1 ÷ 4) + 15 = x × (2 ÷ 5)"


def test_deepseek_teaching_planner_provider_rejects_internal_references() -> None:
    spr = spr_with_text(
        "problem_math11",
        "客车和货车相对开出，求客车速度。",
        [{"symbol": "v_p", "description": "客车速度", "domain": "positive_real"}],
    )
    scs = scs_with_modeling_step("problem_math11", "v_p + v_f = $step1.sum_speed")
    client = FakeTeachingPlannerClient(
        {
            "moves": [
                {
                    "move_id": "bad_move",
                    "source_step_ids": ["step_model"],
                    "source_sentence": "两车相遇",
                    "meaning": "速度和",
                    "display_expression": "x + y = $step1.sum_speed",
                    "narration": "用 $step1.sum_speed 表示速度和。",
                }
            ]
        }
    )

    try:
        DeepSeekTeachingPlannerProvider(client=client).plan(
            spr=spr,
            scs=scs,
            verification_report=passed_verification("problem_math11"),
            presentation={"symbols": {}},
            rule_plan=None,
        )
    except ValueError as exc:
        assert "internal references" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected provider plan rejection.")


def test_deepseek_teaching_planner_provider_enforces_wall_clock_timeout() -> None:
    spr = spr_with_text(
        "problem_timeout",
        "A simple problem.",
        [{"symbol": "x", "description": "unknown", "domain": "real"}],
    )
    scs = scs_with_modeling_step("problem_timeout", "x + 1 = 2")
    client = SlowTeachingPlannerClient({"moves": []})

    started = time.perf_counter()
    try:
        DeepSeekTeachingPlannerProvider(client=client).plan(
            spr=spr,
            scs=scs,
            verification_report=passed_verification("problem_timeout"),
            presentation={"symbols": {"x": {"display_symbol": "x"}}},
            rule_plan=None,
        )
    except TimeoutError as exc:
        elapsed = time.perf_counter() - started
        assert "timed out after 0.01 seconds" in str(exc)
        assert elapsed < 0.04
    else:  # pragma: no cover
        raise AssertionError("Expected provider timeout.")


def test_quality_gate_flags_generic_long_fallback_plan() -> None:
    spr = spr_with_text(
        "problem_math13",
        "一本书的已读和未读比例题。",
        [{"symbol": "x", "description": "页数", "domain": "positive_integer"}],
    )
    scs = scs_with_modeling_step("problem_math13", "(x/4 + 15)/(3*x/4 - 15) = 2/3")

    plan = RuleBasedTeachingPlanner().plan(
        spr,
        scs,
        passed_verification("problem_math13"),
        {"symbols": {"x": {"display_symbol": "x"}}},
    )
    quality = TeachingPlanQualityGate.assess(plan)

    assert plan.quality_report.strategy_names == ["reading_ratio_progress"]
    assert plan.quality_report.fallback_step_ids == []
    assert quality["valid"] is False
    assert "move_directly_displays_long_equation" in quality["reasons"]
