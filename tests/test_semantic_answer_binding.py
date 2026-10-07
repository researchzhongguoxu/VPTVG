import json

from mathexplain.agents.verifier import SemanticAnswerBindingVerifier, Verifier
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR


class FakeSemanticClient:
    provider = "fake"
    model = "fake-semantic-verifier"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_usage = {"total_tokens": 1}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "CAS-confirmed candidates" in user_prompt
        return json.dumps(self.payload)


class FencedSemanticClient(FakeSemanticClient):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return "```json\n" + json.dumps(self.payload) + "\n```"


def _spr(problem_id: str = "semantic_profit") -> SPR:
    question = "A store earns 1115 dollars in revenue and pays 576 dollars in costs. What is the profit?"
    return SPR.model_validate(
        {
            "source_image": {"image_path": f"{problem_id}.png"},
            "problem_text": question,
            "conditions": [],
            "questions": [{"id": "q1", "text": question, "question_type": "compute", "target_ids": ["t1"]}],
            "variables": [],
            "problem_type": "word_problem",
            "knowledge_units": ["arithmetic"],
            "targets": [{"id": "t1", "text": "profit", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": problem_id},
        }
    )


def _emr(problem_id: str = "semantic_profit") -> EMR:
    return EMR.model_validate(
        {
            "problem_id": problem_id,
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def _classification(problem_id: str = "semantic_profit") -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": problem_id,
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


def _chain(final_answer: str = "total_rev = 1115") -> SolutionChain:
    return SolutionChain.model_validate(
        {
            "problem_id": "semantic_profit",
            "chain_id": "chain_main",
            "steps": [
                {
                    "step_id": "planner_total",
                    "step_index": 0,
                    "description": "Plan total revenue.",
                    "math_expression": None,
                    "rule_name": "planner_tool_call",
                    "diagnostics": {
                        "tool_call_id": "calc_total_rev",
                        "normalized_request": {
                            "id": "calc_total_rev",
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "total_rev", "expression": "1115"}],
                            "target": "total_rev",
                        },
                    },
                },
                {
                    "step_id": "cas_total",
                    "step_index": 1,
                    "description": "CAS total revenue.",
                    "math_expression": "1115",
                    "rule_name": "cas_evaluate",
                    "diagnostics": {
                        "tool_call_id": "calc_total_rev",
                        "operation": "evaluate",
                        "input": {
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "total_rev", "expression": "1115"}],
                            "target": "total_rev",
                        },
                        "output": {"value": "1115"},
                        "success": True,
                        "backend_confirmed": True,
                    },
                },
                {
                    "step_id": "planner_profit",
                    "step_index": 2,
                    "description": "Plan profit.",
                    "math_expression": None,
                    "rule_name": "planner_tool_call",
                    "diagnostics": {
                        "tool_call_id": "calc_profit",
                        "normalized_request": {
                            "id": "calc_profit",
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "profit", "expression": "1115 - 576"}],
                            "target": "profit",
                        },
                    },
                },
                {
                    "step_id": "cas_profit",
                    "step_index": 3,
                    "description": "CAS profit.",
                    "math_expression": "539",
                    "rule_name": "cas_evaluate",
                    "diagnostics": {
                        "tool_call_id": "calc_profit",
                        "operation": "evaluate",
                        "input": {
                            "operation": "evaluate",
                            "variables": [],
                            "expressions": [{"name": "profit", "expression": "1115 - 576"}],
                            "target": "profit",
                        },
                        "output": {"value": "539"},
                        "success": True,
                        "backend_confirmed": True,
                    },
                },
                {
                    "step_id": "final",
                    "step_index": 4,
                    "description": "Final answer.",
                    "math_expression": final_answer,
                    "rule_name": "final_answer",
                    "diagnostics": {"input": {"final_answer_items": []}, "output": {"formatted_solution": final_answer}},
                },
            ],
            "final_answer": final_answer,
            "confidence": 0.8,
            "metadata": {
                "solver_route": "algebra_word_problem_solver",
                "final_answer_items": [
                    {
                        "target": "total_rev",
                        "value": "1115",
                        "source_tool_call_id": "calc_total_rev",
                        "backend_operation": "evaluate",
                        "backend_confirmed": True,
                    }
                ],
            },
        }
    )


def _verification(chain: SolutionChain):
    return Verifier().verify(chain, spr=_spr(), emr=_emr(), classification_result=_classification())


def test_semantic_binding_accepts_current_without_mutation() -> None:
    chain = _chain("profit = 539")
    chain.metadata["final_answer_items"] = [
        {
            "target": "profit",
            "value": "539",
            "source_tool_call_id": "calc_profit",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]
    client = FakeSemanticClient(
        {
            "question_intent": "profit",
            "current_answer_matches_intent": True,
            "binding_error_type": "none",
            "repair_action": "accept_current",
            "preferred_candidate": {"source_tool_call_id": "calc_profit", "target": "profit", "value": "539"},
            "confidence": 0.96,
            "reason": "Current answer is the profit.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert repaired.final_answer == "profit = 539"
    assert report["accepted"] is True
    assert report["repair_performed"] is False


def test_semantic_binding_rebinds_existing_cas_candidate() -> None:
    chain = _chain()
    client = FakeSemanticClient(
        {
            "question_intent": "profit",
            "current_answer_matches_intent": False,
            "binding_error_type": "selected_intermediate_total",
            "repair_action": "rebind_existing_cas_candidate",
            "preferred_candidate": {"source_tool_call_id": "calc_profit", "target": "profit", "value": "539"},
            "confidence": 0.92,
            "reason": "The question asks for profit, not revenue.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert report["repair_performed"] is True
    assert report["repair_type"] == "semantic_rebind_existing_cas_candidate"
    assert repaired.final_answer == "profit = 539"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "calc_profit"


def test_semantic_binding_rejects_nonexistent_candidate() -> None:
    chain = _chain()
    client = FakeSemanticClient(
        {
            "repair_action": "rebind_existing_cas_candidate",
            "preferred_candidate": {"source_tool_call_id": "made_up", "target": "profit", "value": "539"},
            "confidence": 0.99,
            "reason": "Bad candidate.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert repaired.final_answer == "total_rev = 1115"
    assert report["repair_performed"] is False
    assert "not found" in report["reason"]


def test_semantic_binding_rejects_direct_answer_without_cas_candidate() -> None:
    chain = _chain()
    client = FakeSemanticClient(
        {
            "repair_action": "rebind_existing_cas_candidate",
            "preferred_candidate": {"target": "profit", "value": "540"},
            "confidence": 0.99,
            "reason": "Invented answer.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert repaired.final_answer == "total_rev = 1115"
    assert report["repair_performed"] is False


def test_semantic_binding_parses_markdown_fenced_json() -> None:
    chain = _chain()
    client = FencedSemanticClient(
        {
            "repair_action": "rebind_existing_cas_candidate",
            "preferred_candidate": {"source_tool_call_id": "calc_profit", "target": "profit", "value": "539"},
            "confidence": 0.9,
            "reason": "Fence should parse.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert repaired.final_answer == "profit = 539"
    assert report["repair_performed"] is True


def test_semantic_binding_missing_cas_request_must_execute_successfully() -> None:
    chain = _chain()
    chain.steps = [step for step in chain.steps if step.step_id not in {"planner_profit", "cas_profit"}]
    client = FakeSemanticClient(
        {
            "question_intent": "profit",
            "current_answer_matches_intent": False,
            "binding_error_type": "missing_candidate",
            "repair_action": "request_missing_cas_computation",
            "cas_request": {
                "id": "semantic_profit",
                "operation": "evaluate",
                "variables": [],
                "expressions": [{"name": "profit", "expression": "1115 - 576"}],
                "substitutions": {},
                "target": "profit",
                "reason": "Compute profit from revenue and costs.",
                "evidence_span": "revenue and costs",
            },
            "confidence": 0.94,
            "reason": "The needed profit candidate is missing.",
        }
    )

    repaired, report = SemanticAnswerBindingVerifier(client=client).verify_and_repair(
        _spr(), chain, _verification(chain), emr=_emr(), mode="always"
    )

    assert report["repair_performed"] is True
    assert report["repair_type"] == "semantic_missing_cas_request"
    assert repaired.final_answer == "profit = 539"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "semantic_profit"


def test_semantic_binding_risk_triggers_on_generic_symbol_final_answer() -> None:
    chain = _chain("x = 199")
    chain.metadata["final_answer_items"] = [
        {
            "target": "x",
            "value": "199",
            "source_tool_call_id": "solve_x",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        }
    ]
    report = _verification(chain)

    spr = _spr()
    spr.questions[0].text = "How many toys does Jonathan currently have?"
    spr.targets[0].text = "Calculate Jonathan's current number of toys"

    assert SemanticAnswerBindingVerifier().should_run(chain, report, None, spr) is True
