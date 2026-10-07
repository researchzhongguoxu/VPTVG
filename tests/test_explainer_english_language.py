from mathexplain.agents.explainer.component import Explainer
from mathexplain.agents.explainer.teaching_expansion import StudentMathFormatter
from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR


def test_english_variable_labels_do_not_use_chinese_fallback() -> None:
    label = Explainer._english_student_label_for_symbol(
        "G",
        {"description": "Number of green flowers"},
        SPR.model_validate(
            {
                "source_image": {"image_path": "english.png"},
                "problem_text": "How many flowers are there?",
                "problem_type": "word_problem",
                "confidence": {"overall": 0.9},
            }
        ),
    )

    assert label == "the number of green flowers"
    assert "未知量" not in label


def test_english_presentation_formats_semantic_variables_and_decimal_answers() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "english.png"},
            "problem_text": (
                "Ten of them are yellow, and there are 80% more of those in purple. "
                "There are only 25% as many green flowers as there are yellow and purple flowers. "
                "How many flowers does Mark have in his garden?"
            ),
            "problem_type": "word_problem",
            "questions": [{"id": "q1", "text": "How many flowers does Mark have in his garden?"}],
            "confidence": {"overall": 0.9},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "english_flowers",
            "chain_id": "chain_main",
            "final_answer": "N_{total} = 35.0000000000000",
            "steps": [],
            "confidence": 0.9,
            "metadata": {
                "variable_bindings": {
                    "N_{yellow}": {"description": "Number of yellow flowers", "variable_role": "final_target"},
                    "N_{purple}": {"description": "Number of purple flowers", "variable_role": "final_target"},
                    "N_{green}": {"description": "Number of green flowers", "variable_role": "final_target"},
                    "N_{total}": {
                        "description": "Total number of flowers",
                        "variable_role": "final_target",
                        "is_final_answer_target": True,
                    },
                },
                "final_answer_items": [
                    {
                        "target": "N_{total}",
                        "value": "35.0000000000000",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    }
                ],
                "answer_bindings": [
                    {
                        "target": "N_{total}",
                        "display_name": "Total number of flowers",
                        "value": "35.0000000000000",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    }
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, scs, output_language="en-US")
    displayed = StudentMathFormatter.display("0.25 * (N_yellow + N_purple)", presentation)

    assert presentation["answers"][0]["value"] == "35"
    assert displayed is not None
    assert "N_yellow" not in displayed
    assert "N_purple" not in displayed
    assert "yellow" in displayed
    assert "purple" in displayed
    assert StudentMathFormatter.display("the total number of flowers = 35.0000000000000", presentation) == (
        "the total number of flowers = 35"
    )


def test_english_generic_solver_symbol_uses_answer_display_name_and_money_unit() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "english.png"},
            "problem_text": (
                "Gus spent $20.00 at the grocery store. He bought 2 bags of chips for $2.00 each, "
                "a bucket of fried chicken for $8.00 and a bottle of soda for $1.00. "
                "How much did the apple pie cost?"
            ),
            "problem_type": "word_problem",
            "questions": [
                {
                    "id": "q1",
                    "text": "How much did the apple pie cost?",
                    "target_ids": ["t1"],
                }
            ],
            "targets": [{"id": "t1", "text": "Calculate the cost of the apple pie", "target_type": "unknown"}],
            "confidence": {"overall": 0.9},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "apple_pie",
            "chain_id": "chain_main",
            "final_answer": "x = 7",
            "steps": [],
            "confidence": 0.9,
            "metadata": {
                "answer_bindings": [
                    {
                        "target": "x",
                        "display_name": "The unknown cost of the apple pie",
                        "value": "7",
                        "unit": "dollars",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    }
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, scs, output_language="en-US")

    assert presentation["answers"][0]["student_label"] == "the cost of the apple pie"
    assert presentation["answers"][0]["unit"] == "dollars"
    assert "x" not in presentation["answers"][0]["student_label"]


def test_english_final_answer_label_uses_question_when_binding_display_name_is_wrong() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "english.png"},
            "problem_text": (
                "The rainstorm washed Phineas Frog 200 yards away from home. "
                "He hops on land at 20 yards per minute and swims at 10 yards per minute. "
                "How long will it take Phineas, in minutes, to return home?"
            ),
            "problem_type": "word_problem",
            "questions": [
                {
                    "id": "q1",
                    "text": "How long will it take Phineas, in minutes, to return home?",
                    "target_ids": ["t1"],
                }
            ],
            "targets": [
                {
                    "id": "t1",
                    "text": "Calculate the total time taken to travel on land and in water.",
                    "target_type": "unknown",
                }
            ],
            "confidence": {"overall": 0.9},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "frog_time",
            "chain_id": "chain_main",
            "final_answer": "答案是 15",
            "steps": [],
            "confidence": 0.9,
            "metadata": {
                "answer_bindings": [
                    {
                        "target": "T",
                        "display_name": "Speed hopping on land",
                        "value": "15",
                        "backend_confirmed": True,
                    }
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, scs, output_language="en-US")

    assert presentation["answers"][0]["student_label"] == "the total time to return home"
    assert presentation["answers"][0]["unit"] == "minutes"


def test_english_final_answer_money_difference_ignores_hours_in_problem_context() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "english.png"},
            "problem_text": (
                "Nick is choosing between two jobs. Job A pays $15 an hour for 2000 hours a year. "
                "Job B pays $42,000 a year. How much more money will Nick make at the job "
                "with a higher net pay rate, compared to the other job?"
            ),
            "problem_type": "word_problem",
            "questions": [
                {
                    "id": "q1",
                    "text": (
                        "How much more money will Nick make at the job with a higher net pay rate, "
                        "compared to the other job?"
                    ),
                    "target_ids": ["target_1"],
                }
            ],
            "targets": [
                {
                    "id": "target_1",
                    "text": "Difference in net pay between Job A and Job B",
                    "target_type": "unknown",
                }
            ],
            "confidence": {"overall": 0.9},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "job_difference",
            "chain_id": "chain_main",
            "final_answer": "答案是 8400",
            "steps": [],
            "confidence": 0.9,
            "metadata": {
                "answer_bindings": [
                    {
                        "target": "diff",
                        "display_name": "Difference in net pay between Job A and Job B",
                        "value": "8400",
                        "unit": "hours",
                        "backend_confirmed": True,
                        "question_id": "q1",
                    }
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, scs, output_language="en-US")

    assert presentation["answers"][0]["student_label"] == "the difference in money"
    assert presentation["answers"][0]["unit"] == "dollars"


def test_english_final_answer_spend_question_overrides_cost_per_item_label() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "english.png"},
            "problem_text": (
                "Alicia sends clothes to the dry cleaners weekly. "
                "How much does she spend on dry-cleaning in 5 weeks?"
            ),
            "problem_type": "word_problem",
            "questions": [
                {
                    "id": "q1",
                    "text": "How much does she spend on dry-cleaning in 5 weeks?",
                    "target_ids": ["t1"],
                }
            ],
            "targets": [
                {
                    "id": "t1",
                    "text": "Total amount spent on dry cleaning over 5 weeks",
                    "target_type": "unknown",
                }
            ],
            "confidence": {"overall": 0.9},
        }
    )
    scs = SolutionChain.model_validate(
        {
            "problem_id": "dry_cleaning",
            "chain_id": "chain_main",
            "final_answer": "答案是 235",
            "steps": [],
            "confidence": 0.9,
            "metadata": {
                "answer_bindings": [
                    {
                        "target": "c_b",
                        "display_name": "Cost per blouse",
                        "value": "235",
                        "unit": "dollars",
                        "backend_confirmed": True,
                    }
                ],
            },
        }
    )

    presentation = Explainer()._presentation_context(spr, scs, output_language="en-US")

    assert presentation["answers"][0]["student_label"] == "the total amount she spends on dry-cleaning in 5 weeks"
    assert presentation["answers"][0]["unit"] == "dollars"


def test_english_provider_narration_strips_result_tail_not_in_display_expression() -> None:
    text = (
        "Now write an equation for the total amount spent. "
        "The sum equals the $20 he spent. This gives 7."
    )

    cleaned = Explainer._strip_unverified_result_tail(text, "2 × 2 + 8 + 1 + x = 20")
    kept = Explainer._strip_unverified_result_tail(
        "Now compute the difference. This gives 7.",
        "20 - 13 = 7",
    )

    assert cleaned == "Now write an equation for the total amount spent. The sum equals the $20 he spent."
    assert kept == "Now compute the difference. This gives 7."


def test_english_cas_result_not_appended_to_symbolic_setup_equation() -> None:
    assert Explainer._should_append_result_for_expression("2 × 2 + 8 + 1 + x = 20", "7") is False
    assert Explainer._should_append_result_for_expression("20 - 13 = 7", "7") is True
    assert Explainer._should_append_result_for_expression("20 - 13", "7") is True
