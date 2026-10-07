"""Run the formal GSM8K Full System answer experiment with a Qwen planner.

This runner is intentionally answer-mode only:
- image -> Vision Parser -> Classifier -> Solver/CAS -> Verifier
- no TeachingPlan, Explanation, EDS, TTS, or video rendering
- independent output directory and resumable per-sample result.json files

PyCharm usage:
1. Make sure `.env` contains `DASHSCOPE_API_KEY`.
2. Edit the DEFAULT_* constants below if needed.
3. Click Run. Existing per-sample result.json files are skipped unless
   `--overwrite` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import sympy as sp
from dotenv import load_dotenv

from mathexplain.config import PipelineModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# PyCharm-friendly defaults. Edit these values directly, then click Run.
DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_full_system_qwen_planner_200_replan_exprfix"
DEFAULT_LIMIT = 200
DEFAULT_SELECTION_RANK_START = 0
DEFAULT_VISION_MODEL = "qwen3.6-flash"
DEFAULT_PLANNER_PROVIDER = "qwen"
DEFAULT_PLANNER_MODEL = "qwen3.6-plus"
DEFAULT_SEMANTIC_VERIFIER_MODE = "risk"
DEFAULT_REPLANNER_MODE = "risk"
DEFAULT_REPLANNER_MAX_ROUNDS = 1
DEFAULT_TIMEOUT_SECONDS = 300


def load_experiment_env(env_path: Path) -> None:
    """Load .env while avoiding later placeholder keys overriding real keys."""

    load_dotenv(env_path)
    first_real_dashscope_key = first_non_placeholder_env_value(env_path, "DASHSCOPE_API_KEY")
    current_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if first_real_dashscope_key and is_placeholder_env_value(current_key):
        os.environ["DASHSCOPE_API_KEY"] = first_real_dashscope_key


def first_non_placeholder_env_value(env_path: Path, key_name: str) -> str | None:
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != key_name:
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned and not is_placeholder_env_value(cleaned):
            return cleaned
    return None


def is_placeholder_env_value(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return not text or "your_" in text or text.endswith("_here") or "placeholder" in text


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    per_sample_dir: Path
    comparison_csv: Path
    results_jsonl: Path
    summary_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun GSM8K Full System 200 with Qwen planner, stopping after Verifier."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--selection-rank-start", type=int, default=DEFAULT_SELECTION_RANK_START)
    parser.add_argument("--vision-provider", default=None)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--vision-base-url", default=None)
    parser.add_argument("--vision-api-key-env", default=None)
    parser.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-base-url", default=None)
    parser.add_argument("--planner-api-key-env", default=None)
    parser.add_argument(
        "--semantic-verifier-mode",
        choices=["off", "risk", "always"],
        default=DEFAULT_SEMANTIC_VERIFIER_MODE,
        help="Semantic final-answer binding verifier mode forwarded to run_pipeline.py.",
    )
    parser.add_argument(
        "--replanner-mode",
        choices=["off", "risk", "always"],
        default=DEFAULT_REPLANNER_MODE,
        help="Verifier-guided replanner mode forwarded to run_pipeline.py.",
    )
    parser.add_argument(
        "--replanner-max-rounds",
        type=int,
        default=DEFAULT_REPLANNER_MAX_ROUNDS,
        help="Maximum verifier-guided replanning rounds forwarded to run_pipeline.py.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout for one sample pipeline run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun samples even when per_sample/<sample_id>/result.json already exists.",
    )
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def make_paths(output_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        output_dir=output_dir,
        per_sample_dir=output_dir / "per_sample",
        comparison_csv=output_dir / "comparison.csv",
        results_jsonl=output_dir / "full_system_qwen_results.jsonl",
        summary_json=output_dir / "summary.json",
    )


def load_manifest(path: Path, limit: int, selection_rank_start: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("include_in_answer_eval")]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    return rows[:limit]


def run_full_system_qwen(
    *,
    row: dict[str, Any],
    sample_dir: Path,
    config: PipelineModelConfig,
    semantic_verifier_mode: str,
    replanner_mode: str,
    replanner_max_rounds: int,
    overwrite: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        refreshed = refresh_correctness(existing, row)
        result_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
        return refreshed

    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = resolve_repo_path(row["image_path"])
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_pipeline.py"),
        "--image",
        str(image_path),
        "--output-dir",
        str(sample_dir),
        "--vision-mode",
        "qwen",
        "--vision-provider",
        config.vision.provider,
        "--vision-model",
        config.vision.model,
        "--vision-base-url",
        config.vision.base_url,
        "--vision-api-key-env",
        config.vision.api_key_env,
        "--planner-provider",
        config.planner.provider,
        "--planner-model",
        config.planner.model,
        "--planner-base-url",
        config.planner.base_url,
        "--planner-api-key-env",
        config.planner.api_key_env,
        "--semantic-verifier-mode",
        semantic_verifier_mode,
        "--replanner-mode",
        replanner_mode,
        "--replanner-max-rounds",
        str(replanner_max_rounds),
        "--stop-after",
        "verifier",
        "--quiet-progress",
    ]

    started = time.perf_counter()
    stdout_tail = ""
    stderr_tail = ""
    returncode = 1
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout_tail = (completed.stdout or "")[-4000:]
        stderr_tail = (completed.stderr or "")[-4000:]
    except subprocess.TimeoutExpired as exc:
        stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        stderr_tail = (stderr_tail + f"\nTimed out after {timeout_seconds} seconds.").strip()

    duration_ms = int((time.perf_counter() - started) * 1000)
    summary_path = find_pipeline_summary(sample_dir)
    summary: dict[str, Any] = {}
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = build_result(
        row=row,
        sample_dir=sample_dir,
        summary=summary,
        summary_path=summary_path,
        duration_ms=duration_ms,
        returncode=returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_result(
    *,
    row: dict[str, Any],
    sample_dir: Path,
    summary: dict[str, Any],
    summary_path: Path | None,
    duration_ms: int,
    returncode: int,
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    final_answer = nested_get(summary, ["solver", "final_answer"])
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    llm_usage = summary.get("llm_usage") if isinstance(summary.get("llm_usage"), dict) else {}
    solver_partial_failure = bool(nested_get(summary, ["solver", "metadata", "partial_failure"]))
    status = summary.get("status")
    if status is None:
        status = "failed" if returncode else "unknown"
    failure_reason = summary.get("failure_reason")
    if failure_reason is None and returncode:
        failure_reason = stderr_tail or stdout_tail or f"run_pipeline.py exited with code {returncode}."
    semantic = summary.get("semantic_verifier") if isinstance(summary.get("semantic_verifier"), dict) else {}
    replanner = summary.get("replanner") if isinstance(summary.get("replanner"), dict) else {}
    return {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "image_path": row["image_path"],
        "question_text": get_question_text(row),
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "status": status,
        "failed_stage": summary.get("failed_stage"),
        "failure_reason": failure_reason,
        "solver_success": status == "passed" and final_answer not in {None, ""} and not solver_partial_failure,
        "solver_partial_failure": solver_partial_failure,
        "verifier_passed": bool(nested_get(summary, ["verification", "valid"]))
        and nested_get(summary, ["verification", "status"]) == "passed",
        "cas_backed": bool(nested_get(summary, ["cas", "backend_confirmed"])),
        "semantic_verifier_mode": semantic.get("mode"),
        "semantic_verifier_triggered": bool(semantic.get("triggered")),
        "semantic_verifier_accepted": bool(semantic.get("accepted")),
        "semantic_repair_performed": bool(semantic.get("repair_performed")),
        "semantic_decision": semantic.get("decision"),
        "replanner_mode": replanner.get("mode"),
        "replanner_triggered": bool(replanner.get("triggered")),
        "replanner_accepted": bool(replanner.get("accepted")),
        "replanner_repair_performed": bool(replanner.get("repair_performed")),
        "replanner_round_count": int(replanner.get("round_count") or 0),
        "llm_usage": llm_usage,
        "llm_total_tokens": int(llm_usage.get("total_tokens") or 0),
        "llm_prompt_tokens": int(llm_usage.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(llm_usage.get("completion_tokens") or 0),
        "llm_call_count": int(llm_usage.get("llm_call_count") or 0),
        "latency_ms": duration_ms,
        "returncode": returncode,
        "artifact_dir": str(summary_path.parent if summary_path is not None else sample_dir),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def find_pipeline_summary(sample_dir: Path) -> Path | None:
    candidates = list(sample_dir.glob("summary.json")) + list(sample_dir.glob("*/summary.json"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = _normalize_latex_fraction_text(text)
    if "=" in text:
        text = text.rsplit("=", 1)[-1].strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\b(dollars?|hours?|minutes?|pages?|oranges?|feet|miles|liters?|ml)\b", "", text)
    text = re.sub(r"[^0-9a-zA-Z./+\-*() ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    arithmetic_value = evaluate_arithmetic_answer(text)
    if arithmetic_value is not None:
        return arithmetic_value

    number_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
    if number_match:
        number = number_match.group(0)
        try:
            if "/" in number:
                numerator, denominator = number.split("/", 1)
                decimal_value = Decimal(numerator) / Decimal(denominator)
            else:
                decimal_value = Decimal(number)
            return str(decimal_value.normalize())
        except (InvalidOperation, ZeroDivisionError):
            return number
    return text


def _normalize_latex_fraction_text(text: str) -> str:
    text = re.sub(
        r"\\frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
        r"\1/\2",
        text,
    )
    return re.sub(
        r"frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
        r"\1/\2",
        text,
    )


def evaluate_arithmetic_answer(text: str) -> str | None:
    expression = str(text or "").strip()
    if not expression or not re.search(r"[+\-*/]", expression):
        return None
    if not re.fullmatch(r"[-+*/().0-9\s]+", expression):
        return None
    try:
        value = sp.sympify(expression)
        if not getattr(value, "is_number", False):
            return None
        decimal_value = Decimal(str(sp.N(value, 30)))
    except Exception:
        return None
    return str(decimal_value.normalize())


def refresh_correctness(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(result)
    normalized_prediction = normalize_answer(refreshed.get("prediction"))
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    refreshed["gold_final_answer"] = row.get("gold_final_answer")
    refreshed["normalized_prediction"] = normalized_prediction
    refreshed["normalized_gold"] = normalized_gold
    refreshed["answer_correct"] = normalized_prediction == normalized_gold and normalized_gold != ""
    return refreshed


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def get_question_text(row: dict[str, Any]) -> str:
    return str(row.get("source_question") or row.get("question_text") or "").strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_comparison(
    path: Path,
    manifest_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in results}
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        sample_id = item["sample_id"]
        result = by_id.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "selection_rank": item["selection_rank"],
                "gold_final_answer": item.get("gold_final_answer"),
                "full_system_qwen_prediction": result.get("prediction"),
                "full_system_qwen_correct": result.get("answer_correct"),
                "full_system_qwen_status": result.get("status"),
                "full_system_qwen_failed_stage": result.get("failed_stage"),
                "full_system_qwen_solver_success": result.get("solver_success"),
                "full_system_qwen_solver_partial_failure": result.get("solver_partial_failure"),
                "full_system_qwen_verifier_passed": result.get("verifier_passed"),
                "full_system_qwen_cas_backed": result.get("cas_backed"),
                "semantic_verifier_triggered": result.get("semantic_verifier_triggered"),
                "semantic_verifier_accepted": result.get("semantic_verifier_accepted"),
                "semantic_repair_performed": result.get("semantic_repair_performed"),
                "replanner_triggered": result.get("replanner_triggered"),
                "replanner_accepted": result.get("replanner_accepted"),
                "replanner_repair_performed": result.get("replanner_repair_performed"),
                "replanner_round_count": result.get("replanner_round_count"),
                "full_system_qwen_llm_total_tokens": result.get("llm_total_tokens"),
                "full_system_qwen_llm_call_count": result.get("llm_call_count"),
                "full_system_qwen_latency_ms": result.get("latency_ms"),
                "image_path": item.get("image_path"),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row.get("answer_correct"))
    passed = sum(1 for row in results if row.get("status") == "passed")
    latency = [int(row.get("latency_ms") or 0) for row in results]
    total_tokens = [int(row.get("llm_total_tokens") or 0) for row in results]
    prompt_tokens = [int(row.get("llm_prompt_tokens") or 0) for row in results]
    completion_tokens = [int(row.get("llm_completion_tokens") or 0) for row in results]
    call_counts = [int(row.get("llm_call_count") or 0) for row in results]
    return {
        "sample_count": total,
        "answer_correct": correct,
        "answer_accuracy": correct / total if total else 0.0,
        "status_passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "verifier_passed_count": sum(1 for row in results if row.get("verifier_passed")),
        "cas_backed_count": sum(1 for row in results if row.get("cas_backed")),
        "solver_success_count": sum(1 for row in results if row.get("solver_success")),
        "solver_partial_failure_count": sum(1 for row in results if row.get("solver_partial_failure")),
        "semantic_verifier_triggered_count": sum(1 for row in results if row.get("semantic_verifier_triggered")),
        "semantic_verifier_accepted_count": sum(1 for row in results if row.get("semantic_verifier_accepted")),
        "semantic_repair_performed_count": sum(1 for row in results if row.get("semantic_repair_performed")),
        "replanner_triggered_count": sum(1 for row in results if row.get("replanner_triggered")),
        "replanner_accepted_count": sum(1 for row in results if row.get("replanner_accepted")),
        "replanner_repair_performed_count": sum(1 for row in results if row.get("replanner_repair_performed")),
        "avg_latency_ms": int(sum(latency) / total) if total else 0,
        "total_llm_tokens": sum(total_tokens),
        "avg_llm_tokens": int(sum(total_tokens) / total) if total else 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "avg_prompt_tokens": int(sum(prompt_tokens) / total) if total else 0,
        "total_completion_tokens": sum(completion_tokens),
        "avg_completion_tokens": int(sum(completion_tokens) / total) if total else 0,
        "total_llm_calls": sum(call_counts),
        "avg_llm_calls": sum(call_counts) / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    load_experiment_env(REPO_ROOT / ".env")
    config = PipelineModelConfig.from_sources(
        vision_provider=args.vision_provider,
        vision_model=args.vision_model,
        vision_base_url=args.vision_base_url,
        vision_api_key_env=args.vision_api_key_env,
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        planner_base_url=args.planner_base_url,
        planner_api_key_env=args.planner_api_key_env,
    )
    paths = make_paths(args.output_dir)
    paths.per_sample_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, args.limit, args.selection_rank_start)
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        if not args.quiet_progress:
            print(f"[{index}/{len(rows)}] {sample_id}")
        result = run_full_system_qwen(
            row=row,
            sample_dir=paths.per_sample_dir / sample_id,
            config=config,
            semantic_verifier_mode=args.semantic_verifier_mode,
            replanner_mode=args.replanner_mode,
            replanner_max_rounds=args.replanner_max_rounds,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
        )
        results.append(result)
        if not args.quiet_progress:
            print(
                "  Full System Qwen:",
                result.get("prediction"),
                "correct=",
                result.get("answer_correct"),
                "status=",
                result.get("status"),
                "verifier=",
                result.get("verifier_passed"),
            )

    write_jsonl(paths.results_jsonl, results)
    comparison_rows = write_comparison(paths.comparison_csv, rows, results)
    summary = {
        "experiment": "full_system_qwen_planner_gsm8k_answer",
        "definition": (
            "Full System rerun from GSM8K images up to Verifier. Vision uses Qwen vision; "
            "math planner and semantic answer-binding verifier use the configured Qwen text model."
        ),
        "manifest": str(args.manifest),
        "output_dir": str(paths.output_dir),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "vision_provider": config.vision.provider,
        "vision_model": config.vision.model,
        "planner_provider": config.planner.provider,
        "planner_model": config.planner.model,
        "semantic_verifier_mode": args.semantic_verifier_mode,
        "replanner_mode": args.replanner_mode,
        "replanner_max_rounds": args.replanner_max_rounds,
        **summarize(results),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "results_jsonl": str(paths.results_jsonl),
            "comparison_csv": str(paths.comparison_csv),
            "summary_json": str(paths.summary_json),
            "per_sample_dir": str(paths.per_sample_dir),
        },
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
