import json

import pytest
from pydantic import ValidationError

from mathexplain.schemas.scs import (
    SolutionCandidateSet,
    SolutionChain,
    validate_solution_candidate_set_dict,
    validate_solution_chain_dict,
)


def solution_chain_data(chain_id: str = "chain_1", final_answer: str = "x = 3") -> dict:
    return {
        "problem_id": "problem_001",
        "source_emr_id": "emr_001",
        "source_emr_version": "EMR-1.0",
        "chain_id": chain_id,
        "steps": [
            {
                "step_id": f"{chain_id}_step_0",
                "step_index": 0,
                "description": "Start from the original equation.",
                "math_expression": "x + 2 = 5",
                "rule_name": "given_equation",
                "justification": "This equation comes from the EMR.",
                "verification_status": "verified",
                "verification_summary": "The step matches the source equation.",
                "verification_score": 1.0,
                "diagnostics": {
                    "source_equation_id": "eq_1",
                },
            },
            {
                "step_id": f"{chain_id}_step_1",
                "step_index": 1,
                "description": "Subtract 2 from both sides.",
                "math_expression": "x = 3",
                "rule_name": "subtract_same_value",
                "justification": "Subtracting the same value preserves equality.",
                "depends_on": [f"{chain_id}_step_0"],
                "verification_status": "unverified",
            },
        ],
        "final_answer": final_answer,
        "confidence": 0.93,
        "metadata": {
            "solver_backend": "mock-cas",
            "generation_strategy": "algebraic_manipulation",
        },
    }


def candidate_set_data() -> dict:
    return {
        "problem_id": "problem_001",
        "candidates": [
            solution_chain_data("chain_1", "x = 3"),
            solution_chain_data("chain_2", "3"),
        ],
        "selected_chain_id": "chain_1",
        "ranking_report": {
            "self_consistency_vote": {
                "x = 3": 2,
            },
            "selected_reason": "highest verification score",
        },
        "metadata": {
            "candidate_count": 2,
            "generation_policy": "multi_strategy",
        },
    }


def test_validate_solution_chain_dict() -> None:
    chain = validate_solution_chain_dict(solution_chain_data())

    assert isinstance(chain, SolutionChain)
    assert chain.schema_version == "SCS-1.0"
    assert chain.problem_id == "problem_001"
    assert chain.chain_id == "chain_1"
    assert chain.steps[0].verification_status == "verified"
    assert chain.steps[0].verification_summary == "The step matches the source equation."
    assert chain.steps[0].verification_score == 1.0
    assert chain.steps[0].diagnostics["source_equation_id"] == "eq_1"


def test_validate_solution_candidate_set_dict() -> None:
    candidate_set = validate_solution_candidate_set_dict(candidate_set_data())

    assert isinstance(candidate_set, SolutionCandidateSet)
    assert candidate_set.schema_version == "SCS-1.0"
    assert candidate_set.problem_id == "problem_001"
    assert len(candidate_set.candidates) == 2
    assert candidate_set.selected_chain_id == "chain_1"
    assert candidate_set.ranking_report["selected_reason"] == "highest verification score"


def test_missing_problem_id_fails_for_solution_chain() -> None:
    data = solution_chain_data()
    data.pop("problem_id")

    with pytest.raises(ValidationError):
        validate_solution_chain_dict(data)


def test_missing_problem_id_fails_for_candidate_set() -> None:
    data = candidate_set_data()
    data.pop("problem_id")

    with pytest.raises(ValidationError):
        validate_solution_candidate_set_dict(data)


@pytest.mark.parametrize(
    ("path", "empty_value"),
    [
        ("schema_version", ""),
        ("problem_id", " "),
        ("chain_id", ""),
        ("steps.0.step_id", " "),
        ("steps.0.description", ""),
    ],
)
def test_empty_key_strings_fail_for_solution_chain(path: str, empty_value: str) -> None:
    data = solution_chain_data()
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = empty_value

    with pytest.raises(ValidationError):
        validate_solution_chain_dict(data)


def test_empty_schema_version_fails_for_candidate_set() -> None:
    data = candidate_set_data()
    data["schema_version"] = " "

    with pytest.raises(ValidationError):
        validate_solution_candidate_set_dict(data)


def test_invalid_verification_status_fails_validation() -> None:
    data = solution_chain_data()
    data["steps"][0]["verification_status"] = "partially_verified"

    with pytest.raises(ValidationError):
        validate_solution_chain_dict(data)


def test_invalid_verification_score_fails_validation() -> None:
    data = solution_chain_data()
    data["steps"][0]["verification_score"] = 1.01

    with pytest.raises(ValidationError):
        validate_solution_chain_dict(data)


def test_invalid_confidence_fails_validation() -> None:
    data = solution_chain_data()
    data["confidence"] = -0.1

    with pytest.raises(ValidationError):
        validate_solution_chain_dict(data)


def test_steps_can_be_ordered_by_step_index() -> None:
    chain = validate_solution_chain_dict(solution_chain_data())

    ordered_step_ids = [step.step_id for step in sorted(chain.steps, key=lambda step: step.step_index)]

    assert ordered_step_ids == ["chain_1_step_0", "chain_1_step_1"]


def test_solution_chain_serializes_to_dict_and_json() -> None:
    chain = validate_solution_chain_dict(solution_chain_data())

    dumped = chain.model_dump()
    dumped_json = chain.model_dump_json(indent=2)

    assert dumped["schema_version"] == "SCS-1.0"
    assert json.loads(dumped_json)["chain_id"] == "chain_1"


def test_candidate_set_serializes_to_dict_and_json() -> None:
    candidate_set = validate_solution_candidate_set_dict(candidate_set_data())

    dumped = candidate_set.model_dump()
    dumped_json = candidate_set.model_dump_json(indent=2)

    assert dumped["schema_version"] == "SCS-1.0"
    assert json.loads(dumped_json)["selected_chain_id"] == "chain_1"


def test_ranking_report_and_metadata_support_extensions() -> None:
    candidate_set = validate_solution_candidate_set_dict(candidate_set_data())

    assert candidate_set.ranking_report["self_consistency_vote"] == {"x = 3": 2}
    assert candidate_set.metadata["candidate_count"] == 2
    assert candidate_set.candidates[0].metadata["solver_backend"] == "mock-cas"
