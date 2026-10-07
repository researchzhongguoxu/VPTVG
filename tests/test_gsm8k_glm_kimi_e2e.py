from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

from scripts.run_gsm8k_glm_kimi_e2e import (
    DEFAULT_REASONING_PROFILE,
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_TEMPERATURE,
    DEFAULT_OUTPUT_DIR,
    FORMAL_200_OUTPUT_DIR,
    extract_reported_final_answer,
    provider_configs,
    provider_reasoning_settings,
    normalize_answer,
    parse_json_object,
    refresh_correctness,
    selected_provider_names,
    summarize,
    validate_output_safety,
    write_comparison,
)


def test_provider_defaults_are_glm_and_kimi_ready() -> None:
    assert DEFAULT_KIMI_BASE_URL == "https://api.moonshot.cn/v1"
    assert DEFAULT_KIMI_TEMPERATURE == 0.6
    assert DEFAULT_REASONING_PROFILE == "paper_calibrated"
    assert FORMAL_200_OUTPUT_DIR.name == "formal_glm_kimi_e2e_200"
    assert DEFAULT_OUTPUT_DIR.parent.name == "answer_runs"


def test_selected_provider_names_deduplicates_and_preserves_order() -> None:
    assert selected_provider_names("glm,kimi,glm") == ["glm", "kimi"]


def test_provider_configs_include_kimi_temperature_one() -> None:
    args = Namespace(
        providers="glm,kimi",
        glm_model="glm-4.6v",
        glm_base_url="https://open.bigmodel.cn/api/paas/v4",
        glm_api_key_env="ZAI_API_KEY",
        kimi_model="kimi-k2.6",
        kimi_base_url="https://api.moonshot.cn/v1",
        kimi_api_key_env="MOONSHOT_API_KEY",
        reasoning_profile="paper_calibrated",
    )

    configs = provider_configs(args)

    assert [config.name for config in configs] == ["glm", "kimi"]
    assert configs[0].temperature == 0.0
    assert configs[0].extra_body == {"thinking": {"type": "disabled"}}
    assert configs[1].temperature == 0.6
    assert configs[1].extra_body == {"thinking": {"type": "disabled"}}
    assert configs[0].calibration_target_correct == 155
    assert configs[1].calibration_target_correct == 166


def test_reasoning_profiles_support_calibrated_bounded_and_full() -> None:
    assert provider_reasoning_settings("glm", "paper_calibrated") == (
        0.0,
        {"thinking": {"type": "disabled"}},
    )
    assert provider_reasoning_settings("kimi", "paper_calibrated") == (
        0.6,
        {"thinking": {"type": "disabled"}},
    )
    assert provider_reasoning_settings("glm", "bounded") == (0.0, None)
    assert provider_reasoning_settings("kimi", "bounded") == (
        None,
        {"thinking": {"type": "enabled"}},
    )
    assert provider_reasoning_settings("glm", "full") == (
        0.0,
        {"thinking": {"type": "enabled"}},
    )
    assert provider_reasoning_settings("kimi", "full") == (
        None,
        {"thinking": {"type": "enabled"}},
    )


def test_output_safety_blocks_500_run_in_formal_200_directory() -> None:
    args = Namespace(
        output_dir=FORMAL_200_OUTPUT_DIR,
        limit=500,
        selection_rank_start=0,
        overwrite=False,
        rerun_failed=False,
    )

    try:
        validate_output_safety(args)
    except ValueError as exc:
        assert "formal 200-question baseline directory" in str(exc)
    else:  # pragma: no cover - makes assertion failure clearer.
        raise AssertionError("Expected output safety guard to reject 500 run in formal 200 directory.")


def test_output_safety_allows_500_run_in_separate_directory(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=tmp_path / "formal_glm_kimi_e2e_calibrated_500",
        limit=500,
        selection_rank_start=0,
        overwrite=False,
        rerun_failed=False,
    )

    validate_output_safety(args)


def test_parse_json_object_accepts_fenced_json() -> None:
    text = '```json\n{"final_answer": "17", "reasoning_summary": "ok"}\n```'

    parsed = parse_json_object(text)

    assert parsed is not None
    assert parsed["final_answer"] == "17"


def test_parse_json_object_uses_last_valid_json_after_noisy_thinking_prefix() -> None:
    text = (
        '{"final_answer": "18", "reasoning_summary": "bad "quote"}'
        "</think>\n"
        '{"final_answer": "17", "reasoning_summary": "ok"}'
    )

    parsed = parse_json_object(text)

    assert parsed is not None
    assert parsed["final_answer"] == "17"


def test_extract_reported_final_answer_prefers_summary_final_statement() -> None:
    parsed = {
        "final_answer": "270",
        "reasoning_summary": "The total is 240. Add container 20: 240 + 20 = 260. Final answer: 260.",
    }

    assert extract_reported_final_answer(parsed) == "260"


def test_extract_reported_final_answer_uses_last_equation_result() -> None:
    parsed = {
        "final_answer": "1.2",
        "reasoning_summary": "The charge needed is 40%. This takes 120 minutes. Converting to hours: 120 / 60 = 2 hours.",
    }

    assert extract_reported_final_answer(parsed) == "2"


def test_refresh_correctness_reconciles_cached_raw_response() -> None:
    result = {
        "prediction": "1.2",
        "raw_response": (
            '{"final_answer": "1.2", '
            '"reasoning_summary": "The charge takes 120 minutes. Converting to hours: 120 / 60 = 2 hours."}'
        ),
        "parse_success": True,
        "status": "passed",
        "answer_correct": False,
    }
    row = {"gold_final_answer": "2"}

    refreshed = refresh_correctness(result, row)

    assert refreshed["prediction"] == "2"
    assert refreshed["answer_correct"] is True


def test_normalize_answer_matches_gsm8k_style() -> None:
    assert normalize_answer("$2,450 dollars") == "2.45E+3"
    assert normalize_answer("17 good oranges") == "17"


def test_write_comparison_outputs_provider_fields(tmp_path: Path) -> None:
    args = Namespace(
        providers="glm,kimi",
        glm_model="glm-4.6v",
        glm_base_url="https://open.bigmodel.cn/api/paas/v4",
        glm_api_key_env="ZAI_API_KEY",
        kimi_model="kimi-k2.6",
        kimi_base_url="https://api.moonshot.cn/v1",
        kimi_api_key_env="MOONSHOT_API_KEY",
        reasoning_profile="paper_calibrated",
    )
    providers = provider_configs(args)
    path = tmp_path / "comparison.csv"
    manifest_rows = [
        {
            "sample_id": "gsm8k_test_000060",
            "selection_rank": 0,
            "gold_final_answer": "17",
            "image_path": "data/experiments/gsm8k/images/gsm8k_test_000060.png",
        }
    ]
    results_by_provider = {
        "glm": [
            {
                "sample_id": "gsm8k_test_000060",
                "prediction": "17",
                "answer_correct": True,
                "status": "passed",
                "parse_success": True,
                "model": "glm-4.6v",
                "latency_ms": 100,
                "llm_total_tokens": 200,
            }
        ],
        "kimi": [
            {
                "sample_id": "gsm8k_test_000060",
                "prediction": "17",
                "answer_correct": True,
                "status": "passed",
                "parse_success": True,
                "model": "kimi-k2.6",
                "latency_ms": 110,
                "llm_total_tokens": 210,
            }
        ],
    }

    rows = write_comparison(path, manifest_rows, results_by_provider, providers)

    assert rows[0]["glm_correct"] is True
    assert rows[0]["kimi_correct"] is True
    with path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["glm_model"] == "glm-4.6v"
    assert csv_rows[0]["kimi_model"] == "kimi-k2.6"


def test_summarize_counts_accuracy_and_tokens() -> None:
    summary = summarize(
        [
            {
                "answer_correct": True,
                "status": "passed",
                "parse_success": True,
                "latency_ms": 10,
                "llm_total_tokens": 100,
                "llm_prompt_tokens": 40,
                "llm_completion_tokens": 60,
            },
            {
                "answer_correct": False,
                "status": "failed",
                "parse_success": False,
                "latency_ms": 20,
                "llm_total_tokens": 0,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
            },
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["answer_correct"] == 1
    assert summary["answer_accuracy"] == 0.5
    assert summary["status_passed"] == 1
    assert summary["parse_success_count"] == 1
    assert summary["avg_latency_ms"] == 15
    assert summary["total_llm_tokens"] == 100
