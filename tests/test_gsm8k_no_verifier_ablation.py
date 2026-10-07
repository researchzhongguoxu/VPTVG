from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_gsm8k_no_verifier_ablation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_PLANNER_PROVIDER,
    load_manifest,
    normalize_answer,
    refresh_correctness,
    summarize,
    write_comparison,
)


def test_load_manifest_filters_answer_eval_and_rank_start(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "a", "selection_rank": 0, "include_in_answer_eval": True},
        {"sample_id": "b", "selection_rank": 1, "include_in_answer_eval": False},
        {"sample_id": "c", "selection_rank": 2, "include_in_answer_eval": True},
        {"sample_id": "d", "selection_rank": 3, "include_in_answer_eval": True},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = load_manifest(manifest, limit=2, selection_rank_start=1)

    assert [row["sample_id"] for row in selected] == ["c", "d"]


def test_formal_defaults_use_qwen_planner_and_separate_output_dir() -> None:
    assert DEFAULT_PLANNER_PROVIDER == "qwen"
    assert DEFAULT_PLANNER_MODEL == "qwen3.6-plus"
    assert DEFAULT_OUTPUT_DIR.name == "formal_no_verifier_qwen_planner_200"


def test_normalize_answer_matches_existing_experiment_style() -> None:
    assert normalize_answer("答案是 6.00000000000000") == "6"
    assert normalize_answer("speed = 10 miles") == "1E+1"
    assert normalize_answer("$2,450 dollars") == "2.45E+3"


def test_refresh_correctness_updates_reused_result() -> None:
    row = {"gold_final_answer": "12"}
    result = {"prediction": "answer = 12.0", "gold_final_answer": "10", "answer_correct": False}

    refreshed = refresh_correctness(result, row)

    assert refreshed["gold_final_answer"] == "12"
    assert refreshed["normalized_prediction"] == "12"
    assert refreshed["normalized_gold"] == "12"
    assert refreshed["answer_correct"] is True


def test_write_comparison_outputs_no_verifier_fields(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    manifest_rows = [
        {
            "sample_id": "gsm8k_test_000060",
            "selection_rank": 0,
            "gold_final_answer": "17",
            "image_path": "data/experiments/gsm8k/images/gsm8k_test_000060.png",
        }
    ]
    results = [
        {
            "sample_id": "gsm8k_test_000060",
            "prediction": "17",
            "answer_correct": True,
            "status": "passed",
            "solver_success": True,
            "solver_partial_failure": False,
            "cas_backed": True,
            "planner_provider": "qwen",
            "planner_model": "qwen3.6-plus",
            "llm_total_tokens": 100,
            "llm_call_count": 2,
            "latency_ms": 1000,
        }
    ]

    rows = write_comparison(path, manifest_rows, results)

    assert rows[0]["no_verifier_prediction"] == "17"
    assert rows[0]["no_verifier_correct"] is True
    assert rows[0]["no_verifier_solver_success"] is True
    assert rows[0]["no_verifier_planner_provider"] == "qwen"
    assert rows[0]["no_verifier_planner_model"] == "qwen3.6-plus"
    with path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["sample_id"] == "gsm8k_test_000060"
    assert csv_rows[0]["no_verifier_correct"] == "True"


def test_summarize_counts_accuracy_cost_and_solver_flags() -> None:
    summary = summarize(
        [
            {
                "status": "passed",
                "answer_correct": True,
                "solver_success": True,
                "solver_partial_failure": False,
                "cas_backed": True,
                "llm_total_tokens": 10,
                "llm_call_count": 2,
                "latency_ms": 5,
            },
            {
                "status": "failed",
                "answer_correct": False,
                "solver_success": False,
                "solver_partial_failure": True,
                "cas_backed": False,
                "llm_total_tokens": 20,
                "llm_call_count": 1,
                "latency_ms": 15,
            },
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["answer_correct"] == 1
    assert summary["answer_accuracy"] == 0.5
    assert summary["status_passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["solver_success_count"] == 1
    assert summary["solver_partial_failure_count"] == 1
    assert summary["cas_backed_count"] == 1
    assert summary["avg_latency_ms"] == 10
    assert summary["total_llm_tokens"] == 30
    assert summary["avg_llm_tokens"] == 15
    assert summary["total_llm_calls"] == 3
    assert summary["avg_llm_calls"] == 1.5
