from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_gsm8k_video_comparison_experiment import (
    load_manifest,
    normalize_answer,
    reusable_existing_result,
    result_is_reusable,
    summarize,
    write_comparison,
)


def test_load_manifest_uses_video_eval_and_rank_start(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "a", "selection_rank": 0, "include_in_video_eval": True},
        {"sample_id": "b", "selection_rank": 1, "include_in_video_eval": False},
        {"sample_id": "c", "selection_rank": 2, "include_in_video_eval": True},
        {"sample_id": "d", "selection_rank": 3, "include_in_video_eval": True},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = load_manifest(manifest, limit=2, selection_rank_start=1)

    assert [row["sample_id"] for row in selected] == ["c", "d"]


def test_load_manifest_expands_beyond_video_eval_when_needed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "a", "selection_rank": 0, "include_in_video_eval": True},
        {"sample_id": "b", "selection_rank": 1, "include_in_video_eval": True},
        {"sample_id": "c", "selection_rank": 2, "include_in_video_eval": False},
        {"sample_id": "d", "selection_rank": 3, "include_in_video_eval": False},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = load_manifest(manifest, limit=4, selection_rank_start=0)

    assert [row["sample_id"] for row in selected] == ["a", "b", "c", "d"]


def test_normalize_answer_extracts_final_number() -> None:
    assert normalize_answer("答案是 6.00000000000000") == "6"
    assert normalize_answer("speed = 10 miles") == "1E+1"
    assert normalize_answer("$2,450 dollars") == "2.45E+3"


def test_write_comparison_outputs_video_fields(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    manifest_rows = [
        {
            "sample_id": "gsm8k_test_000060",
            "selection_rank": 0,
            "gold_final_answer": "17",
            "image_path": "data/experiments/gsm8k/images/gsm8k_test_000060.png",
        }
    ]
    full_results = [
        {
            "sample_id": "gsm8k_test_000060",
            "prediction": "17",
            "answer_correct": True,
            "status": "passed",
            "video_generated": True,
            "video_path": "full/video.mp4",
            "llm_total_tokens": 100,
            "llm_call_count": 2,
            "duration_ms": 1000,
        }
    ]
    plain_results = [
        {
            "sample_id": "gsm8k_test_000060",
            "prediction": "17",
            "answer_correct": True,
            "status": "passed",
            "video_generated": True,
            "video_path": "plain/video.mp4",
            "llm_total_tokens": 75,
            "llm_call_count": 1,
            "duration_ms": 650,
        }
    ]

    rows = write_comparison(path, manifest_rows, full_results, plain_results)

    assert rows[0]["full_system_video_path"] == "full/video.mp4"
    assert rows[0]["plain_simple_video_path"] == "plain/video.mp4"
    assert "full_system_reused" in rows[0]
    assert "plain_simple_reused" in rows[0]
    with path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["sample_id"] == "gsm8k_test_000060"
    assert csv_rows[0]["plain_simple_correct"] == "True"


def test_summarize_counts_video_success_and_cost() -> None:
    summary = summarize(
        [
            {"status": "passed", "video_generated": True, "answer_correct": True, "llm_total_tokens": 10, "duration_ms": 5},
            {"status": "failed", "video_generated": False, "answer_correct": False, "llm_total_tokens": 20, "duration_ms": 15},
        ],
        "simple_pipeline",
    )

    assert summary["simple_pipeline_pass_rate"] == 0.5
    assert summary["simple_pipeline_video_generated_rate"] == 0.5
    assert summary["simple_pipeline_answer_accuracy"] == 0.5
    assert summary["simple_pipeline_reused_existing_results"] == 0
    assert summary["simple_pipeline_fresh_runs"] == 2
    assert summary["simple_pipeline_llm_total_tokens"] == 30
    assert summary["simple_pipeline_avg_duration_ms"] == 10


def test_plain_simple_reuse_requires_current_visual_version() -> None:
    old_result = {
        "status": "passed",
        "prediction": "17",
        "video_generated": True,
        "visual_version": "old_plain_visual",
    }
    current_result = {**old_result, "visual_version": "plain_simple_text_v4"}

    assert not result_is_reusable(
        old_result,
        require_video=True,
        required_visual_version="plain_simple_text_v4",
    )
    assert result_is_reusable(
        current_result,
        require_video=True,
        required_visual_version="plain_simple_text_v4",
    )


def test_plain_simple_reuse_can_accept_older_visual_version_for_cumulative_runs() -> None:
    old_result = {
        "status": "passed",
        "prediction": "17",
        "video_generated": True,
        "visual_version": "plain_simple_text_v3",
    }

    assert result_is_reusable(
        old_result,
        require_video=True,
        required_visual_version=None,
    )


def test_reuse_failed_results_keeps_failed_video_artifacts() -> None:
    failed_result = {
        "status": "failed",
        "prediction": "答案是 52",
        "video_generated": False,
        "subprocess_returncode": 1,
    }

    assert not result_is_reusable(failed_result, require_video=True)
    assert result_is_reusable(failed_result, require_video=True, allow_failed=True)


def test_resume_can_reconstruct_reusable_result_from_summary(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    video = sample_dir / "video.mp4"
    video.write_bytes(b"fake video")
    summary = {
        "status": "passed",
        "final_answer": "17",
        "visual_version": "plain_simple_text_v4",
        "artifact_paths": {"video": str(video), "summary": str(sample_dir / "summary.json")},
    }
    (sample_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    row = {
        "sample_id": "gsm8k_test_000060",
        "selection_rank": 0,
        "image_path": "image.png",
        "gold_final_answer": "17",
    }

    def builder(row, sample_dir, summary, returncode, duration_ms, completed):
        return {
            "sample_id": row["sample_id"],
            "selection_rank": row["selection_rank"],
            "pipeline": "plain_simple_pipeline",
            "visual_version": summary.get("visual_version"),
            "prediction": summary.get("final_answer"),
            "normalized_prediction": normalize_answer(summary.get("final_answer")),
            "normalized_gold": normalize_answer(row.get("gold_final_answer")),
            "answer_correct": True,
            "status": summary.get("status"),
            "video_generated": True,
            "video_path": str(video),
            "duration_ms": duration_ms,
        }

    result = reusable_existing_result(
        row=row,
        sample_dir=sample_dir,
        result_path=sample_dir / "result.json",
        require_video=True,
        required_visual_version="plain_simple_text_v4",
        builder=builder,
    )

    assert result is not None
    assert result["reused_existing_result"] is True
    assert result["reuse_source"] == "summary_json"
    assert result["answer_correct"] is True
    assert (sample_dir / "result.json").exists()
