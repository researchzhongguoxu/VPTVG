import json

from mathexplain.agents.solver import VerifierGuidedReplanner
from mathexplain.agents.verifier import Verifier
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR
from mathexplain.schemas.verification import VerificationReport


class FakeReplanningClient:
    provider = "fake"
    model = "fake-replanner"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_usage = {"total_tokens": 7, "prompt_tokens": 3, "completion_tokens": 4}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "verifier errors" in user_prompt.lower()
        assert "tool_calls" in user_prompt
        return json.dumps(self.payload)


def _spr() -> SPR:
    question = (
        "Maggie works for 180 minutes, takes a 30 minute break, and earns "
        "$1 for every 2 minutes worked. How much does she earn?"
    )
    return SPR.model_validate(
        {
            "source_image": {"image_path": "earnings.png"},
            "problem_text": question,
            "conditions": [],
            "questions": [{"id": "q1", "text": question, "question_type": "compute", "target_ids": ["t1"]}],
            "variables": [],
            "problem_type": "word_problem",
            "knowledge_units": ["arithmetic"],
            "targets": [{"id": "t1", "text": "earnings", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": "earnings"},
        }
    )


def _emr() -> EMR:
    return EMR.model_validate(
        {
            "problem_id": "earnings",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def _classification() -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": "earnings",
            "primary_type": "arithmetic",
            "task_type": "word_problem",
            "difficulty": "elementary",
            "knowledge_units": ["arithmetic"],
            "solver_route": "unknown",
            "solver_config": {
                "solver_name": "PlannerToolCallSolver",
                "expected_emr_type": "unknown",
                "task_type": "word_problem",
                "strategy": "planner_tool_calls",
            },
            "requires_solver": True,
            "requires_verifier": True,
            "confidence": 0.9,
        }
    )


def _partial_chain() -> SolutionChain:
    return SolutionChain.model_validate(
        {
            "problem_id": "earnings",
            "chain_id": "chain_main",
            "steps": [
                {
                    "step_id": "planner_total",
                    "step_index": 0,
                    "description": "Plan total time.",
                    "math_expression": "180",
                    "rule_name": "planner_evaluate",
                    "diagnostics": {
                        "tool_call_id": "calc_total_time",
                        "normalized_request": {
                            "id": "calc_total_time",
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "total_time", "expression": "180"}],
                            "target": "total_time",
                        },
                    },
                },
                {
                    "step_id": "cas_total",
                    "step_index": 1,
                    "description": "CAS total time.",
                    "math_expression": "180",
                    "rule_name": "cas_evaluate",
                    "diagnostics": {
                        "tool_call_id": "calc_total_time",
                        "operation": "evaluate",
                        "input": {
                            "id": "calc_total_time",
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "total_time", "expression": "180"}],
                            "target": "total_time",
                        },
                        "output": {"value": "180", "formatted_solution": "180"},
                        "success": True,
                        "backend_confirmed": True,
                    },
                },
                {
                    "step_id": "final",
                    "step_index": 2,
                    "description": "Final answer.",
                    "math_expression": "答案是 180",
                    "rule_name": "final_answer",
                    "diagnostics": {"output": {"formatted_solution": "答案是 180"}},
                },
            ],
            "final_answer": "答案是 180",
            "confidence": 0.55,
            "metadata": {
                "partial_failure": True,
                "backend_confirmed": True,
                "completed_tool_call_count": 1,
                "tool_call_count": 1,
                "final_answer_items": [
                    {
                        "target": "total_time",
                        "value": "180",
                        "source_tool_call_id": "calc_total_time",
                        "source_step_id": "cas_total",
                        "backend_operation": "evaluate",
                        "backend_confirmed": True,
                    }
                ],
            },
        }
    )


def _failed_verification(chain: SolutionChain) -> VerificationReport:
    return VerificationReport.model_validate(
        {
            "problem_id": chain.problem_id,
            "chain_id": chain.chain_id,
            "valid": False,
            "status": "failed",
            "verification_mode": "tool_call_backed",
            "scs_schema_valid": True,
            "backend_steps_valid": False,
            "final_answer_valid": True,
            "summary": "failed before replanning",
            "errors": ["partial chain"],
        }
    )


def test_replanner_appends_missing_final_cas_computation() -> None:
    chain = _partial_chain()
    client = FakeReplanningClient(
        {
            "target_contract": {
                "requested_quantity": "earnings after subtracting the break",
                "target_type": "money_amount",
                "unit": "dollars",
                "why_current_is_insufficient": "180 is total minutes, not earnings.",
            },
            "replan_needed": True,
            "tool_calls": [
                {
                    "id": "calc_earnings",
                    "operation": "evaluate",
                    "variables": [],
                    "expressions": [
                        {
                            "name": "earnings",
                            "expression": "($calc_total_time.result - 30) / 2",
                        }
                    ],
                    "target": "earnings",
                    "reason": "Subtract break time and convert minutes to dollars.",
                    "evidence_span": "takes a 30 minute break, earns $1 for every 2 minutes",
                }
            ],
            "reason": "The current answer is an intermediate total time.",
        }
    )

    result = VerifierGuidedReplanner(client=client).replan(
        _spr(),
        _emr(),
        _classification(),
        chain,
        _failed_verification(chain),
        mode="always",
    )

    assert result.report["repair_performed"] is True
    assert result.chain.final_answer == "earnings = 75"
    assert result.chain.metadata["final_answer_items"][0]["target"] == "earnings"
    verification = Verifier().verify(result.chain, spr=_spr(), emr=_emr(), classification_result=_classification())
    assert verification.valid is True


def test_replanner_rejects_direct_final_answer_payload() -> None:
    chain = _partial_chain()
    client = FakeReplanningClient(
        {
            "target_contract": {"requested_quantity": "earnings", "target_type": "money_amount"},
            "replan_needed": True,
            "final_answer": "75",
            "tool_calls": [],
            "reason": "bad direct answer",
        }
    )

    result = VerifierGuidedReplanner(client=client).replan(
        _spr(),
        _emr(),
        _classification(),
        chain,
        _failed_verification(chain),
        mode="always",
    )

    assert result.report["repair_performed"] is False
    assert result.chain.final_answer == "答案是 180"
    assert "direct final answer" in result.report["rounds"][0]["reason"]
