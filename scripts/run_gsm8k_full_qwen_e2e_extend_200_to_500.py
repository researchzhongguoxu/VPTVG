"""Extend Full System and Qwen3.6-plus E2E answer runs from 200 to 500.

This wrapper does not modify the locked 200-question result directories. It
copies their per-sample caches into separate 500-question directories, runs the
existing runners with limit=500, and then writes a combined 500-question summary
for paper tables.

PyCharm usage:
1. Make sure `.env` contains `DASHSCOPE_API_KEY`.
2. Run this file directly.
3. Do not edit the old 200-question directories.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mathexplain.config import PipelineModelConfig
from scripts.run_gsm8k_qwen_answer_experiment import (
    run_qwen_e2e,
    summarize as summarize_qwen_e2e,
    write_comparison as write_qwen_e2e_comparison,
)
from scripts.run_gsm8k_qwen_planner_full_200 import (
    DEFAULT_REPLANNER_MAX_ROUNDS,
    DEFAULT_REPLANNER_MODE,
    DEFAULT_SEMANTIC_VERIFIER_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    run_full_system_qwen,
    summarize as summarize_full_system,
    write_comparison as write_full_system_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"

SOURCE_FULL_200_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_full_system_qwen_planner_200_replan_exprfix"
SOURCE_QWEN_E2E_200_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/qwen36_new_200"

FULL_500_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_full_system_qwen_planner_500"
QWEN_E2E_500_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_qwen36_e2e_500"
COMBINED_500_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_full_vs_qwen_e2e_500"

FULL_RUNNER = REPO_ROOT / "scripts/run_gsm8k_qwen_planner_full_200.py"
QWEN_E2E_RUNNER = REPO_ROOT / "scripts/run_gsm8k_qwen_answer_experiment.py"

LIMIT = 500
EXPECTED_SOURCE_COUNT = 200
VISION_MODEL = "qwen3.6-flash"
QWEN_E2E_MODEL = "qwen3.6-plus"
PLANNER_PROVIDER = "qwen"
PLANNER_MODEL = "qwen3.6-plus"


def main() -> int:
    seed_full_cache()
    seed_qwen_e2e_cache()

    rows = load_manifest(MANIFEST, LIMIT)
    full_results, qwen_results = run_paired_full_and_qwen(rows)
    write_full_outputs(rows, full_results)
    write_qwen_e2e_outputs(rows, qwen_results)

    write_combined_outputs()
    return 0


def run_paired_full_and_qwen(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_dotenv(REPO_ROOT / ".env")
    config = PipelineModelConfig.from_sources(
        vision_model=VISION_MODEL,
        planner_provider=PLANNER_PROVIDER,
        planner_model=PLANNER_MODEL,
    )
    api_key = os.getenv(config.vision.api_key_env)
    if not api_key:
        raise ValueError(f"{config.vision.api_key_env} is required for Qwen E2E baseline.")
    qwen_client = OpenAI(api_key=api_key, base_url=config.vision.base_url, timeout=120, max_retries=0)

    full_results: list[dict[str, Any]] = []
    qwen_results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(rows)}] rank={row.get('selection_rank')} sample_id={sample_id}", flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            full_future = executor.submit(
                run_full_system_qwen,
                row=row,
                sample_dir=FULL_500_DIR / "per_sample" / sample_id,
                config=config,
                semantic_verifier_mode=DEFAULT_SEMANTIC_VERIFIER_MODE,
                replanner_mode=DEFAULT_REPLANNER_MODE,
                replanner_max_rounds=DEFAULT_REPLANNER_MAX_ROUNDS,
                overwrite=False,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            qwen_future = executor.submit(
                run_qwen_e2e,
                row,
                QWEN_E2E_500_DIR / "qwen_e2e" / sample_id,
                qwen_client,
                QWEN_E2E_MODEL,
                False,
            )
            full = full_future.result()
            qwen = qwen_future.result()

        full_results.append(full)
        qwen_results.append(qwen)
        print(
            "  Full System Qwen:",
            full.get("prediction"),
            "correct=",
            full.get("answer_correct"),
            "status=",
            full.get("status"),
            flush=True,
        )
        print(
            "  Qwen E2E:",
            qwen.get("prediction"),
            "correct=",
            qwen.get("answer_correct"),
            "status=",
            qwen.get("status"),
            flush=True,
        )

    return full_results, qwen_results


def seed_full_cache() -> None:
    source = SOURCE_FULL_200_DIR / "per_sample"
    target = FULL_500_DIR / "per_sample"
    seed_cache_tree("full", source, target)


def seed_qwen_e2e_cache() -> None:
    source = SOURCE_QWEN_E2E_200_DIR / "qwen_e2e"
    target = QWEN_E2E_500_DIR / "qwen_e2e"
    seed_cache_tree("qwen_e2e", source, target)


def seed_cache_tree(label: str, source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing {label} source cache: {source}")
    source_count = count_result_files(source)
    if source_count < EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"{label} source cache has only {source_count} result.json files; "
            f"expected at least {EXPECTED_SOURCE_COUNT}."
        )

    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for sample_dir in source.iterdir():
        if not sample_dir.is_dir():
            continue
        target_sample = target / sample_dir.name
        if (target_sample / "result.json").exists():
            skipped += 1
            continue
        shutil.copytree(sample_dir, target_sample, dirs_exist_ok=True)
        copied += 1
    print(f"[seed] {label}: source_result_count={source_count}, copied={copied}, already_present={skipped}", flush=True)


def count_result_files(directory: Path) -> int:
    return sum(1 for path in directory.glob("*/result.json") if path.is_file())


def write_full_outputs(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    FULL_500_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(FULL_500_DIR / "full_system_qwen_results.jsonl", results)
    comparison_rows = write_full_system_comparison(FULL_500_DIR / "comparison.csv", rows, results)
    summary = {
        "experiment": "full_system_qwen_planner_gsm8k_answer",
        "definition": "Full System 500-question answer run with Qwen vision, Qwen planner, CAS, Verifier, semantic binding, and replanner.",
        "manifest": str(MANIFEST),
        "output_dir": str(FULL_500_DIR),
        "limit": LIMIT,
        "selection_rank_start": 0,
        "vision_provider": "qwen",
        "vision_model": VISION_MODEL,
        "planner_provider": PLANNER_PROVIDER,
        "planner_model": PLANNER_MODEL,
        "semantic_verifier_mode": DEFAULT_SEMANTIC_VERIFIER_MODE,
        "replanner_mode": DEFAULT_REPLANNER_MODE,
        "replanner_max_rounds": DEFAULT_REPLANNER_MAX_ROUNDS,
        **summarize_full_system(results),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "results_jsonl": str(FULL_500_DIR / "full_system_qwen_results.jsonl"),
            "comparison_csv": str(FULL_500_DIR / "comparison.csv"),
            "summary_json": str(FULL_500_DIR / "summary.json"),
            "per_sample_dir": str(FULL_500_DIR / "per_sample"),
        },
    }
    (FULL_500_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_qwen_e2e_outputs(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    QWEN_E2E_500_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(QWEN_E2E_500_DIR / "qwen_e2e_results.jsonl", results)
    comparison_rows = write_qwen_e2e_comparison(QWEN_E2E_500_DIR / "comparison.csv", rows, [], results)
    summary = {
        "experiment": "qwen36_plus_e2e_gsm8k_answer",
        "definition": "Qwen3.6-plus image-to-answer E2E baseline for GSM8K 500; no MathExplainAgent internal pipeline.",
        "manifest": str(MANIFEST),
        "output_dir": str(QWEN_E2E_500_DIR),
        "limit": LIMIT,
        "selection_rank_start": 0,
        "qwen_model": QWEN_E2E_MODEL,
        "sample_count": len(rows),
        **summarize_qwen_e2e(results, "qwen_e2e"),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "qwen_e2e_results": str(QWEN_E2E_500_DIR / "qwen_e2e_results.jsonl"),
            "comparison_csv": str(QWEN_E2E_500_DIR / "comparison.csv"),
            "summary_json": str(QWEN_E2E_500_DIR / "summary.json"),
            "per_sample_dir": str(QWEN_E2E_500_DIR / "qwen_e2e"),
        },
    }
    (QWEN_E2E_500_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_combined_outputs() -> None:
    COMBINED_500_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(MANIFEST, LIMIT)
    full_rows = load_csv_by_id(FULL_500_DIR / "comparison.csv")
    qwen_rows = load_csv_by_id(QWEN_E2E_500_DIR / "comparison.csv")

    combined_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        full = full_rows.get(sample_id, {})
        qwen = qwen_rows.get(sample_id, {})
        combined_rows.append(
            {
                "sample_id": sample_id,
                "selection_rank": row.get("selection_rank"),
                "gold_final_answer": row.get("gold_final_answer"),
                "full_system_qwen_prediction": full.get("full_system_qwen_prediction"),
                "full_system_qwen_correct": parse_bool(full.get("full_system_qwen_correct")),
                "full_system_qwen_status": full.get("full_system_qwen_status"),
                "full_system_qwen_verifier_passed": parse_bool(full.get("full_system_qwen_verifier_passed")),
                "full_system_qwen_cas_backed": parse_bool(full.get("full_system_qwen_cas_backed")),
                "full_system_qwen_llm_total_tokens": parse_int(full.get("full_system_qwen_llm_total_tokens")),
                "full_system_qwen_llm_call_count": parse_int(full.get("full_system_qwen_llm_call_count")),
                "full_system_qwen_latency_ms": parse_int(full.get("full_system_qwen_latency_ms")),
                "qwen_e2e_prediction": qwen.get("qwen_e2e_prediction"),
                "qwen_e2e_correct": parse_bool(qwen.get("qwen_e2e_correct")),
                "qwen_e2e_status": qwen.get("qwen_e2e_status"),
                "qwen_e2e_parse_success": parse_bool(qwen.get("qwen_e2e_parse_success")),
                "qwen_e2e_llm_total_tokens": parse_int(qwen.get("qwen_e2e_llm_total_tokens")),
                "qwen_e2e_llm_call_count": parse_int(qwen.get("qwen_e2e_llm_call_count")),
                "image_path": row.get("image_path"),
            }
        )

    comparison_csv = COMBINED_500_DIR / "comparison.csv"
    write_csv(comparison_csv, combined_rows)
    summary = build_summary(combined_rows)
    summary.update(
        {
            "experiment": "gsm8k_full_system_qwen_vs_qwen_e2e_500",
            "definition": "Combined 500-question summary for Full System with Qwen planner and Qwen3.6-plus image-to-answer E2E.",
            "manifest": str(MANIFEST),
            "limit": LIMIT,
            "source_full_200_dir": str(SOURCE_FULL_200_DIR),
            "source_qwen_e2e_200_dir": str(SOURCE_QWEN_E2E_200_DIR),
            "full_500_dir": str(FULL_500_DIR),
            "qwen_e2e_500_dir": str(QWEN_E2E_500_DIR),
            "artifacts": {
                "comparison_csv": str(comparison_csv),
                "summary_json": str(COMBINED_500_DIR / "summary.json"),
            },
        }
    )
    (COMBINED_500_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[combined] wrote", comparison_csv, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def load_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("include_in_answer_eval")]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    return rows[:limit]


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    full_correct = sum(1 for row in rows if row.get("full_system_qwen_correct") is True)
    qwen_correct = sum(1 for row in rows if row.get("qwen_e2e_correct") is True)
    full_passed = sum(1 for row in rows if row.get("full_system_qwen_status") == "passed")
    qwen_passed = sum(1 for row in rows if row.get("qwen_e2e_status") == "passed")
    paired = paired_counts(rows)
    return {
        "sample_count": total,
        "full_system_qwen_correct": full_correct,
        "full_system_qwen_accuracy": full_correct / total if total else 0.0,
        "full_system_qwen_passed": full_passed,
        "full_system_qwen_pass_rate": full_passed / total if total else 0.0,
        "qwen_e2e_correct": qwen_correct,
        "qwen_e2e_accuracy": qwen_correct / total if total else 0.0,
        "qwen_e2e_passed": qwen_passed,
        "qwen_e2e_pass_rate": qwen_passed / total if total else 0.0,
        **paired,
    }


def paired_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    both_correct = full_only = qwen_only = both_wrong = 0
    for row in rows:
        full = row.get("full_system_qwen_correct") is True
        qwen = row.get("qwen_e2e_correct") is True
        if full and qwen:
            both_correct += 1
        elif full and not qwen:
            full_only += 1
        elif qwen and not full:
            qwen_only += 1
        else:
            both_wrong += 1
    return {
        "paired_both_correct": both_correct,
        "paired_full_only": full_only,
        "paired_qwen_only": qwen_only,
        "paired_both_wrong": both_wrong,
        "mcnemar_exact_p": mcnemar_exact_p(full_only, qwen_only),
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    probability = 2 * sum(comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, probability)


def parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
