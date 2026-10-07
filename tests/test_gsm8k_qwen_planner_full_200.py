from __future__ import annotations

import os

from scripts.run_gsm8k_qwen_planner_full_200 import load_experiment_env, normalize_answer


def test_normalize_answer_evaluates_plain_arithmetic_expression() -> None:
    assert normalize_answer("252 * 7 / 12 * 2") == "294"
    assert normalize_answer("4 * 45 / 30") == "6"


def test_normalize_answer_keeps_existing_numeric_style() -> None:
    assert normalize_answer("绛旀鏄?6.00000000000000") == "6"
    assert normalize_answer("speed = 10 miles") == "1E+1"


def test_normalize_answer_accepts_latex_fraction_gold() -> None:
    assert normalize_answer(r"\frac{13}{15}") == normalize_answer("13/15")


def test_load_experiment_env_prefers_first_real_dashscope_key(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=real-test-key-123456",
                "DASHSCOPE_API_KEY=your_dashscope_api_key_here",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "your_dashscope_api_key_here")

    load_experiment_env(env_path)

    assert os.environ["DASHSCOPE_API_KEY"] == "real-test-key-123456"

