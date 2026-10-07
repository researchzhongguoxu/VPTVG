"""Run GSM8K No Verifier reasoning ablation.

This script intentionally stops at Solver and does not run Verifier,
FinalAnswerRepair, TeachingPlan, EDS, TTS, or Renderer. It evaluates the raw
Solver final_answer against GSM8K gold answers for the strict No Verifier
ablation.

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
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.solver import DeepSeekPlanner, Solver
from mathexplain.agents.vision_parser import SPRGenerator, VisionParser
from mathexplain.config import PipelineModelConfig
from mathexplain.schemas.classification import validate_classification_result_dict
from mathexplain.schemas.scs import validate_solution_chain_dict
from mathexplain.services.deepseek import DeepSeekChatClient
from mathexplain.services.llm import DashScopeQwenTextClient, DashScopeQwenVisionClient
from mathexplain.services.usage import summarize_usage, usage_call


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/ablation_runs/formal_no_verifier_qwen_planner_200"
DEFAULT_LIMIT = 200
DEFAULT_SELECTION_RANK_START = 0
DEFAULT_VISION_MODEL = "qwen3.6-flash"
DEFAULT_PLANNER_PROVIDER = "qwen"
DEFAULT_PLANNER_MODEL = "qwen3.6-plus"
DEFAULT_USE_PLANNER = True


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
    results_jsonl: Path
    comparison_csv: Path
    summary_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict No Verifier GSM8K ablation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--selection-rank-start", type=int, default=DEFAULT_SELECTION_RANK_START)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--vision-provider", default=None)
    parser.add_argument("--vision-base-url", default=None)
    parser.add_argument("--vision-api-key-env", default=None)
    parser.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-base-url", default=None)
    parser.add_argument("--planner-api-key-env", default=None)
    parser.add_argument(
        "--no-planner",
        action="store_false",
        dest="use_planner",
        default=DEFAULT_USE_PLANNER,
        help="Disable the LLM solver planner and run deterministic solver routes only.",
    )
    parser.add_argument(
        "--no-deepseek",
        action="store_false",
        dest="use_planner",
        help="Backward-compatible alias for --no-planner.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun existing per-sample results.")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def make_paths(output_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        output_dir=output_dir,
        per_sample_dir=output_dir / "per_sample",
        results_jsonl=output_dir / "no_verifier_results.jsonl",
        comparison_csv=output_dir / "comparison.csv",
        summary_json=output_dir / "summary.json",
    )


def load_manifest(path: Path, limit: int, selection_rank_start: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("include_in_answer_eval")]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    return rows[:limit]


def create_text_client(provider: str, base_url: str, model: str, api_key_env: str) -> Any:
    provider_value = provider.lower()
    if provider_value == "qwen":
        return DashScopeQwenTextClient(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
        )
    if provider_value == "deepseek":
        return DeepSeekChatClient(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
        )
    raise ValueError(f"Unsupported planner provider for No Verifier ablation: {provider!r}")


def run_no_verifier_sample(
    row: dict[str, Any],
    sample_dir: Path,
    config: PipelineModelConfig,
    use_planner: bool,
    overwrite: bool,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        refreshed = refresh_correctness(existing, row)
        result_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
        return refreshed

    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = resolve_repo_path(row["image_path"])
    artifact_paths: dict[str, str] = {}
    llm_usage_calls: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        qwen_client = DashScopeQwenVisionClient(
            base_url=config.vision.base_url,
            model=config.vision.model,
            api_key_env=config.vision.api_key_env,
        )
        parser = VisionParser(spr_generator=SPRGenerator(mode="qwen", qwen_client=qwen_client))
        parser_output = parser.run(str(image_path), save_artifacts=True, output_dir=sample_dir)
        artifact_paths.update(parser_output.artifact_paths)
        if getattr(qwen_client, "last_usage", None):
            llm_usage_calls.append(
                usage_call(
                    stage="vision_parser",
                    provider=getattr(qwen_client, "provider", config.vision.provider),
                    model=getattr(qwen_client, "model", config.vision.model),
                    usage=getattr(qwen_client, "last_usage", {}),
                )
            )

        classification = ProblemClassifier().classify(
            parser_output.spr,
            parser_output.emr,
            parser_output.parsing_report,
        )
        validate_classification_result_dict(classification.model_dump(mode="json"))
        classification_path = sample_dir / "classification.json"
        classification_path.write_text(classification.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths["classification"] = str(classification_path)

        planner = None
        if use_planner:
            planner = DeepSeekPlanner(
                client=create_text_client(
                    config.planner.provider,
                    config.planner.base_url,
                    config.planner.model,
                    config.planner.api_key_env,
                )
            )
        solver_planner_client = planner.client if planner is not None else None
        solution_chain = Solver(planner=planner, use_planner=use_planner).solve(
            parser_output.emr,
            classification,
            spr=parser_output.spr,
        )
        if solver_planner_client is not None and getattr(solver_planner_client, "last_usage", None):
            llm_usage_calls.append(
                usage_call(
                    stage="solver_planner",
                    provider=getattr(solver_planner_client, "provider", config.planner.provider),
                    model=getattr(solver_planner_client, "model", config.planner.model),
                    usage=getattr(solver_planner_client, "last_usage", {}),
                )
            )
        validate_solution_chain_dict(solution_chain.model_dump(mode="json"))
        scs_path = sample_dir / "scs.json"
        scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths["scs"] = str(scs_path)

        duration_ms = int((time.perf_counter() - started) * 1000)
        result = build_result(
            row=row,
            sample_dir=sample_dir,
            final_answer=solution_chain.final_answer,
            status="passed",
            failed_stage=None,
            failure_reason=None,
            solver_partial_failure=bool(solution_chain.metadata.get("partial_failure")),
            cas_backed=bool(solution_chain.metadata.get("backend_confirmed")),
            llm_usage=summarize_usage(llm_usage_calls),
            latency_ms=duration_ms,
            artifact_paths=artifact_paths,
            planner_provider=config.planner.provider if use_planner else "none",
            planner_model=config.planner.model if use_planner else "none",
        )
    except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = build_result(
            row=row,
            sample_dir=sample_dir,
            final_answer=None,
            status="failed",
            failed_stage="no_verifier_solver",
            failure_reason=str(exc),
            solver_partial_failure=True,
            cas_backed=False,
            llm_usage=summarize_usage(llm_usage_calls),
            latency_ms=duration_ms,
            artifact_paths=artifact_paths,
            planner_provider=config.planner.provider if use_planner else "none",
            planner_model=config.planner.model if use_planner else "none",
        )

    timing_report = {
        "total_duration_ms": result["latency_ms"],
        "stages": [],
    }
    timing_path = sample_dir / "timing_report.json"
    timing_path.write_text(json.dumps(timing_report, ensure_ascii=False, indent=2), encoding="utf-8")
    result["artifact_paths"]["timing_report"] = str(timing_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_result(
    *,
    row: dict[str, Any],
    sample_dir: Path,
    final_answer: Any,
    status: str,
    failed_stage: str | None,
    failure_reason: str | None,
    solver_partial_failure: bool,
    cas_backed: bool,
    llm_usage: dict[str, Any],
    latency_ms: int,
    artifact_paths: dict[str, str],
    planner_provider: str,
    planner_model: str,
) -> dict[str, Any]:
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    solver_success = status == "passed" and final_answer not in {None, ""} and not solver_partial_failure
    return {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "status": status,
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "solver_success": solver_success,
        "solver_partial_failure": solver_partial_failure,
        "cas_backed": cas_backed,
        "planner_provider": planner_provider,
        "planner_model": planner_model,
        "llm_usage": llm_usage,
        "llm_total_tokens": int(llm_usage.get("total_tokens") or 0),
        "llm_prompt_tokens": int(llm_usage.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(llm_usage.get("completion_tokens") or 0),
        "llm_call_count": int(llm_usage.get("llm_call_count") or 0),
        "latency_ms": latency_ms,
        "artifact_dir": str(sample_dir),
        "artifact_paths": dict(artifact_paths),
    }


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if "=" in text:
        text = text.rsplit("=", 1)[-1].strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\b(dollars?|hours?|minutes?|pages?|oranges?|feet|miles|liters?|ml)\b", "", text)
    text = re.sub(r"[^0-9a-zA-Z./+\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

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


def refresh_correctness(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(result)
    normalized_prediction = normalize_answer(refreshed.get("prediction"))
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    refreshed["gold_final_answer"] = row.get("gold_final_answer")
    refreshed["normalized_prediction"] = normalized_prediction
    refreshed["normalized_gold"] = normalized_gold
    refreshed["answer_correct"] = normalized_prediction == normalized_gold and normalized_gold != ""
    return refreshed


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_comparison(path: Path, manifest_rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in results}
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        result = by_id.get(item["sample_id"], {})
        rows.append(
            {
                "sample_id": item["sample_id"],
                "selection_rank": item["selection_rank"],
                "gold_final_answer": item.get("gold_final_answer"),
                "no_verifier_prediction": result.get("prediction"),
                "no_verifier_correct": result.get("answer_correct"),
                "no_verifier_status": result.get("status"),
                "no_verifier_solver_success": result.get("solver_success"),
                "no_verifier_solver_partial_failure": result.get("solver_partial_failure"),
                "no_verifier_cas_backed": result.get("cas_backed"),
                "no_verifier_planner_provider": result.get("planner_provider"),
                "no_verifier_planner_model": result.get("planner_model"),
                "no_verifier_llm_total_tokens": result.get("llm_total_tokens"),
                "no_verifier_llm_call_count": result.get("llm_call_count"),
                "no_verifier_latency_ms": result.get("latency_ms"),
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
    solver_success = sum(1 for row in results if row.get("solver_success"))
    solver_partial_failure = sum(1 for row in results if row.get("solver_partial_failure"))
    cas_backed = sum(1 for row in results if row.get("cas_backed"))
    latency = [int(row.get("latency_ms") or 0) for row in results]
    total_tokens = [int(row.get("llm_total_tokens") or 0) for row in results]
    call_counts = [int(row.get("llm_call_count") or 0) for row in results]
    return {
        "sample_count": total,
        "answer_correct": correct,
        "answer_accuracy": correct / total if total else 0.0,
        "status_passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "solver_success_count": solver_success,
        "solver_partial_failure_count": solver_partial_failure,
        "cas_backed_count": cas_backed,
        "avg_latency_ms": int(sum(latency) / total) if total else 0,
        "total_llm_tokens": sum(total_tokens),
        "avg_llm_tokens": int(sum(total_tokens) / total) if total else 0,
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
        result = run_no_verifier_sample(
            row=row,
            sample_dir=paths.per_sample_dir / sample_id,
            config=config,
            use_planner=args.use_planner,
            overwrite=args.overwrite,
        )
        results.append(result)
        if not args.quiet_progress:
            print(
                "  No Verifier:",
                result.get("prediction"),
                "correct=",
                result.get("answer_correct"),
                "status=",
                result.get("status"),
            )

    write_jsonl(paths.results_jsonl, results)
    comparison_rows = write_comparison(paths.comparison_csv, rows, results)
    summary = {
        "experiment": "no_verifier_ablation",
        "definition": "VisionParser -> ProblemClassifier -> Solver; no Verifier, no FinalAnswerRepair, no video stages.",
        "manifest": str(args.manifest),
        "output_dir": str(paths.output_dir),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "model_config": config.metadata(),
        "vision_model": args.vision_model,
        "planner_provider": config.planner.provider if args.use_planner else "none",
        "planner_model": config.planner.model if args.use_planner else "none",
        "use_planner": args.use_planner,
        **summarize(results),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "results_jsonl": str(paths.results_jsonl),
            "comparison_csv": str(paths.comparison_csv),
            "per_sample_dir": str(paths.per_sample_dir),
        },
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
