from mathexplain.agents.solver import DeepSeekPlanner, Solver
from mathexplain.agents.verifier import Verifier
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.spr import SPR


class FakeLinearFunctionLocalCoefficientPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_ab","operation":"solve_equation_system",'
            '"variables":["a","b"],'
            '"equations":[{"lhs":"2*a + b","rhs":"90"},{"lhs":"5*a + b","rhs":"180"}],'
            '"target":"a_and_b","reason":"Solve local coefficients for y = ax + b.",'
            '"evidence_span":"(2,90), (5,180)"},'
            '{"id":"final_line","operation":"substitute",'
            '"variables":["x","a","b"],'
            '"expressions":[{"name":"y_expr","expression":"a*x + b"}],'
            '"substitutions":{"a":"$solve_ab.a","b":"$solve_ab.b"},'
            '"target":"y_expr","reason":"Build explicit function expression.",'
            '"evidence_span":"Find the function expression."}'
            ']}'
        )


def algebra_solver_classification_result(problem_id: str) -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": problem_id,
            "primary_type": "algebra",
            "task_type": "equation_solving",
            "difficulty": "intermediate",
            "knowledge_units": ["linear_functions"],
            "solver_route": "algebra_solver",
            "solver_config": {
                "solver_name": "AlgebraSolver",
                "expected_emr_type": "unknown",
                "task_type": "equation_solving",
                "strategy": "tool_call",
            },
            "requires_solver": True,
            "requires_verifier": True,
            "confidence": 0.85,
        }
    )


def test_solver_allows_local_linear_function_coefficients_with_declared_xy() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_016.png"},
            "problem_text": (
                "A factory produces 90 parts after 2 hours and 180 parts after 5 hours. "
                "Let time be x and parts be y. Find the function expression."
            ),
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "When x = 2, y = 90."},
                {"id": "c2", "text": "When x = 5, y = 180."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the function expression y in terms of x.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "time", "domain": "real"},
                {"symbol": "y", "description": "parts produced", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "function expression", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_016",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_016")
    planner = DeepSeekPlanner(client=FakeLinearFunctionLocalCoefficientPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "30*x + 30",
            "source_tool_call_id": "final_line",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert verification.valid is True
