"""Run a batch video comparison across Full System and Plain Simple baseline.

This script is intentionally PyCharm-friendly: edit the DEFAULT_* values below,
then click Run. It calls the existing single-image runners and writes a compact
experiment table for later human video evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAIN_SIMPLE_VISUAL_VERSION = "plain_simple_text_v4"

# PyCharm-friendly defaults. Edit these values directly, then click Run.
DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/video_runs/full_vs_simple_pilot_200"
DEFAULT_LIMIT = 200
DEFAULT_SELECTION_RANK_START = 0
DEFAULT_RUN_FULL_SYSTEM = True
DEFAULT_RUN_PLAIN_SIMPLE_PIPELINE = True
DEFAULT_FULL_SYSTEM_USE_DEEPSEEK = True
DEFAULT_FULL_SYSTEM_VISION_MODEL =  "qwen3.6-flash"
DEFAULT_PLAIN_SIMPLE_PIPELINE_MODEL = "qwen3.6-flash"
DEFAULT_OUTPUT_LANGUAGE = "auto"  # auto, en-US, zh-CN
DEFAULT_SYNTHESIZE_TTS = True
DEFAULT_RENDER_VIDEO = True
DEFAULT_OVERWRITE = False
DEFAULT_REUSE_EXISTING_PLAIN_SIMPLE_ANY_VERSION = True
DEFAULT_REUSE_FAILED_RESULTS = True


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    full_system_dir: Path
    plain_simple_pipeline_dir: Path
    comparison_csv: Path
    full_results_jsonl: Path
    plain_simple_results_jsonl: Path
    summary_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run Full System and Plain Simple Pipeline videos on GSM8K rendered images."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many video samples to run.")
    parser.add_argument(
        "--selection-rank-start",
        type=int,
        default=DEFAULT_SELECTION_RANK_START,
        help="Start from this manifest selection_rank.",
    )
    parser.add_argument("--run-full-system", action="store_true", default=DEFAULT_RUN_FULL_SYSTEM)
    parser.add_argument("--no-full-system", action="store_false", dest="run_full_system")
    parser.add_argument("--run-plain-simple-pipeline", action="store_true", default=DEFAULT_RUN_PLAIN_SIMPLE_PIPELINE)
    parser.add_argument("--no-plain-simple-pipeline", action="store_false", dest="run_plain_simple_pipeline")
    parser.add_argument("--full-system-vision-model", default=DEFAULT_FULL_SYSTEM_VISION_MODEL)
    parser.add_argument("--plain-simple-pipeline-model", default=DEFAULT_PLAIN_SIMPLE_PIPELINE_MODEL)
    parser.add_argument("--full-system-use-deepseek", action="store_true", default=DEFAULT_FULL_SYSTEM_USE_DEEPSEEK)
    parser.add_argument("--no-full-system-deepseek", action="store_false", dest="full_system_use_deepseek")
    parser.add_argument("--output-language", choices=["auto", "en-US", "zh-CN"], default=DEFAULT_OUTPUT_LANGUAGE)
    parser.add_argument(
        "--tts",
        action="store_true",
        dest="synthesize_tts",
        default=DEFAULT_SYNTHESIZE_TTS,
        help="Enable narration audio synthesis. Disabled by default to avoid TTS API cost.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_false",
        dest="synthesize_tts",
        help="Skip narration audio synthesis and render silent subtitle videos.",
    )
    parser.add_argument("--render-video", action="store_true", dest="render_video", default=DEFAULT_RENDER_VIDEO)
    parser.add_argument("--no-render-video", action="store_false", dest="render_video")
    parser.add_argument("--tts-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--full-system-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--plain-simple-pipeline-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    parser.add_argument(
        "--reuse-existing-plain-simple-any-version",
        action="store_true",
        default=DEFAULT_REUSE_EXISTING_PLAIN_SIMPLE_ANY_VERSION,
        help=(
            "Reuse completed Plain Simple results even if their visual_version is older. "
            "This is useful for cumulative batch runs where earlier videos should not be regenerated."
        ),
    )
    parser.add_argument(
        "--require-current-plain-simple-version",
        action="store_false",
        dest="reuse_existing_plain_simple_any_version",
        help="Only reuse Plain Simple results whose visual_version matches the current script.",
    )
    parser.add_argument(
        "--reuse-failed-results",
        action="store_true",
        default=DEFAULT_REUSE_FAILED_RESULTS,
        help="Reuse existing per-sample results even when status failed or video generation failed.",
    )
    parser.add_argument(
        "--rerun-failed-results",
        action="store_false",
        dest="reuse_failed_results",
        help="Rerun samples whose existing result failed or did not generate video.",
    )
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def make_paths(output_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        output_dir=output_dir,
        full_system_dir=output_dir / "full_system",
        plain_simple_pipeline_dir=output_dir / "plain_simple_pipeline",
        comparison_csv=output_dir / "comparison.csv",
        full_results_jsonl=output_dir / "full_system_results.jsonl",
        plain_simple_results_jsonl=output_dir / "plain_simple_pipeline_results.jsonl",
        summary_json=output_dir / "summary.json",
    )


def load_manifest(path: Path, limit: int, selection_rank_start: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    video_rows = [row for row in rows if row.get("include_in_video_eval")]
    if len(video_rows) >= limit:
        return video_rows[:limit]
    return rows[:limit]


def run_full_system(row: dict[str, Any], args: argparse.Namespace, paths: ExperimentPaths) -> dict[str, Any]:
    sample_id = row["sample_id"]
    sample_dir = paths.full_system_dir / sample_id
    result_path = sample_dir / "result.json"
    if not args.overwrite:
        existing = reusable_existing_result(
            row=row,
            sample_dir=sample_dir,
            result_path=result_path,
            require_video=args.render_video,
            allow_failed=args.reuse_failed_results,
            builder=build_full_system_result,
        )
        if existing is not None:
            return existing

    image_path = resolve_repo_path(row["image_path"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_pipeline.py"),
        "--image",
        str(image_path),
        "--output-dir",
        str(sample_dir),
        "--vision-mode",
        "qwen",
        "--vision-model",
        args.full_system_vision_model,
        "--output-language",
        args.output_language,
        "--stop-after",
        "full",
        "--quiet-progress",
        "--tts-timeout-seconds",
        str(args.tts_timeout_seconds),
    ]
    command.append("--render-video" if args.render_video else "--no-render-video")
    command.append("--tts" if args.synthesize_tts else "--no-tts")
    if not args.full_system_use_deepseek:
        command.append("--no-deepseek")

    completed, duration_ms = run_command(command, timeout_seconds=args.full_system_timeout_seconds)
    summary = read_summary(sample_dir)
    result = build_full_system_result(row, sample_dir, summary, completed.returncode, duration_ms, completed)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_plain_simple_pipeline(row: dict[str, Any], args: argparse.Namespace, paths: ExperimentPaths) -> dict[str, Any]:
    sample_id = row["sample_id"]
    sample_dir = paths.plain_simple_pipeline_dir / sample_id
    result_path = sample_dir / "result.json"
    if not args.overwrite:
        existing = reusable_existing_result(
            row=row,
            sample_dir=sample_dir,
            result_path=result_path,
            require_video=args.render_video,
            allow_failed=args.reuse_failed_results,
            required_visual_version=None
            if args.reuse_existing_plain_simple_any_version
            else PLAIN_SIMPLE_VISUAL_VERSION,
            builder=build_plain_simple_pipeline_result,
        )
        if existing is not None:
            return existing

    image_path = resolve_repo_path(row["image_path"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_simple_pipeline.py"),
        "--image",
        str(image_path),
        "--output-dir",
        str(sample_dir),
        "--model",
        args.plain_simple_pipeline_model,
        "--output-language",
        args.output_language,
        "--visual-style",
        "plain",
        "--quiet-progress",
        "--tts-timeout-seconds",
        str(args.tts_timeout_seconds),
    ]
    command.append("--render-video" if args.render_video else "--no-render-video")
    command.append("--tts" if args.synthesize_tts else "--no-tts")

    completed, duration_ms = run_command(command, timeout_seconds=args.plain_simple_pipeline_timeout_seconds)
    summary = read_summary(sample_dir)
    result = build_plain_simple_pipeline_result(
        row,
        sample_dir,
        summary,
        completed.returncode,
        duration_ms,
        completed,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_command(command: list[str], timeout_seconds: float) -> tuple[subprocess.CompletedProcess[str], int]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=process_text(exc.stdout),
            stderr=process_text(exc.stderr) + f"\nTimeout after {timeout_seconds:.0f} seconds.",
        )
    duration_ms = int(round((time.perf_counter() - started) * 1000))
    return completed, duration_ms


def read_summary(sample_dir: Path) -> dict[str, Any]:
    direct = sample_dir / "summary.json"
    if direct.exists():
        return json.loads(direct.read_text(encoding="utf-8"))
    summaries = sorted(sample_dir.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summaries:
        return {}
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def build_full_system_result(
    row: dict[str, Any],
    sample_dir: Path,
    summary: dict[str, Any],
    returncode: int,
    duration_ms: int,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    final_answer = nested_get(summary, ["solver", "final_answer"])
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    llm_usage = summary.get("llm_usage") if isinstance(summary.get("llm_usage"), dict) else {}
    artifact_paths = summary.get("artifact_paths") if isinstance(summary.get("artifact_paths"), dict) else {}
    video_path = artifact_paths.get("video") or nested_get(summary, ["renderer", "video_path"])
    return {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "pipeline": "full_system",
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "status": summary.get("status", "failed" if returncode else "unknown"),
        "failed_stage": summary.get("failed_stage"),
        "failure_reason": summary.get("failure_reason"),
        "verifier_passed": nested_get(summary, ["verification", "valid"]),
        "eds_valid": nested_get(summary, ["eds", "valid"]),
        "tts_status": nested_get(summary, ["tts", "status"]),
        "renderer_status": nested_get(summary, ["renderer", "status"]),
        "video_generated": bool(video_path and Path(video_path).exists()),
        "video_path": video_path,
        "artifact_dir": summary.get("artifact_dir") or str(sample_dir),
        "summary_path": artifact_paths.get("summary"),
        "llm_usage": llm_usage,
        "llm_total_tokens": llm_usage.get("total_tokens"),
        "llm_call_count": llm_usage.get("llm_call_count"),
        "duration_ms": nested_get(summary, ["timing", "total_duration_ms"]) or duration_ms,
        "subprocess_returncode": returncode,
        "stdout_tail": process_text(completed.stdout)[-4000:],
        "stderr_tail": process_text(completed.stderr)[-4000:],
    }


def build_plain_simple_pipeline_result(
    row: dict[str, Any],
    sample_dir: Path,
    summary: dict[str, Any],
    returncode: int,
    duration_ms: int,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    final_answer = summary.get("final_answer")
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    llm_usage = summary.get("llm_usage") if isinstance(summary.get("llm_usage"), dict) else {}
    artifact_paths = summary.get("artifact_paths") if isinstance(summary.get("artifact_paths"), dict) else {}
    video_path = artifact_paths.get("video") or nested_get(summary, ["render", "video_path"])
    return {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "pipeline": "plain_simple_pipeline",
        "visual_version": summary.get("visual_version"),
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "status": summary.get("status", "failed" if returncode else "unknown"),
        "failed_stage": summary.get("failed_stage"),
        "failure_reason": summary.get("failure_reason"),
        "pipeline_success": summary.get("pipeline_success"),
        "answer_text_present": summary.get("answer_text_present"),
        "scene_count": summary.get("scene_count"),
        "tts_status": nested_get(summary, ["tts", "status"]),
        "renderer_status": nested_get(summary, ["render", "status"]),
        "video_generated": bool(video_path and Path(video_path).exists()),
        "video_path": video_path,
        "artifact_dir": summary.get("artifact_dir") or str(sample_dir),
        "summary_path": artifact_paths.get("summary"),
        "llm_usage": llm_usage,
        "llm_total_tokens": llm_usage.get("total_tokens"),
        "llm_call_count": llm_usage.get("llm_call_count"),
        "duration_ms": nested_get(summary, ["timing", "total_duration_ms"]) or duration_ms,
        "subprocess_returncode": returncode,
        "stdout_tail": process_text(completed.stdout)[-4000:],
        "stderr_tail": process_text(completed.stderr)[-4000:],
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
    refreshed["normalized_prediction"] = normalized_prediction
    refreshed["normalized_gold"] = normalized_gold
    refreshed["answer_correct"] = normalized_prediction == normalized_gold and normalized_gold != ""
    return refreshed


def reusable_existing_result(
    *,
    row: dict[str, Any],
    sample_dir: Path,
    result_path: Path,
    require_video: bool,
    builder: Any,
    allow_failed: bool = False,
    required_visual_version: str | None = None,
) -> dict[str, Any] | None:
    """Return a reusable per-sample result for resume runs.

    The batch runner writes one result.json per sample. If a run is interrupted
    before the aggregate jsonl/csv files are rewritten, the per-sample result is
    still enough to resume without calling the model again. As a fallback, this
    also reconstructs the result from a finished summary.json artifact.
    """

    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result_is_reusable(
            result,
            require_video=require_video,
            allow_failed=allow_failed,
            required_visual_version=required_visual_version,
        ):
            refreshed = refresh_correctness(result, row)
            refreshed["reused_existing_result"] = True
            refreshed["reuse_source"] = "result_json"
            return refreshed

    summary = read_summary(sample_dir)
    if summary:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        reconstructed = builder(row, sample_dir, summary, 0, 0, completed)
        if result_is_reusable(
            reconstructed,
            require_video=require_video,
            allow_failed=allow_failed,
            required_visual_version=required_visual_version,
        ):
            reconstructed["reused_existing_result"] = True
            reconstructed["reuse_source"] = "summary_json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(reconstructed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return refresh_correctness(reconstructed, row)

    return None


def result_is_reusable(
    result: dict[str, Any],
    *,
    require_video: bool,
    allow_failed: bool = False,
    required_visual_version: str | None = None,
) -> bool:
    if allow_failed:
        return result.get("status") is not None or result.get("subprocess_returncode") is not None
    if result.get("status") != "passed":
        return False
    if result.get("prediction") in {None, ""}:
        return False
    if require_video and result.get("video_generated") is not True:
        return False
    if required_visual_version and result.get("visual_version") != required_visual_version:
        return False
    return True


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


def process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_comparison(
    path: Path,
    manifest_rows: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
    plain_simple_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_by_id = {row["sample_id"]: row for row in full_results}
    plain_by_id = {row["sample_id"]: row for row in plain_simple_results}
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        sample_id = item["sample_id"]
        full = full_by_id.get(sample_id, {})
        plain = plain_by_id.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "selection_rank": item["selection_rank"],
                "gold_final_answer": item.get("gold_final_answer"),
                "full_system_prediction": full.get("prediction"),
                "full_system_correct": full.get("answer_correct"),
                "full_system_status": full.get("status"),
                "full_system_video_generated": full.get("video_generated"),
                "full_system_video_path": full.get("video_path"),
                "full_system_llm_total_tokens": full.get("llm_total_tokens"),
                "full_system_llm_call_count": full.get("llm_call_count"),
                "full_system_duration_ms": full.get("duration_ms"),
                "full_system_reused": full.get("reused_existing_result"),
                "full_system_reuse_source": full.get("reuse_source"),
                "plain_simple_prediction": plain.get("prediction"),
                "plain_simple_correct": plain.get("answer_correct"),
                "plain_simple_status": plain.get("status"),
                "plain_simple_video_generated": plain.get("video_generated"),
                "plain_simple_video_path": plain.get("video_path"),
                "plain_simple_llm_total_tokens": plain.get("llm_total_tokens"),
                "plain_simple_llm_call_count": plain.get("llm_call_count"),
                "plain_simple_duration_ms": plain.get("duration_ms"),
                "plain_simple_reused": plain.get("reused_existing_result"),
                "plain_simple_reuse_source": plain.get("reuse_source"),
                "image_path": item["image_path"],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(results: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row.get("status") == "passed")
    videos = sum(1 for row in results if row.get("video_generated") is True)
    correct = sum(1 for row in results if row.get("answer_correct") is True)
    reused = sum(1 for row in results if row.get("reused_existing_result") is True)
    token_values = [row.get("llm_total_tokens") for row in results if isinstance(row.get("llm_total_tokens"), int)]
    duration_values = [row.get("duration_ms") for row in results if isinstance(row.get("duration_ms"), int)]
    return {
        f"{prefix}_count": total,
        f"{prefix}_passed": passed,
        f"{prefix}_pass_rate": passed / total if total else None,
        f"{prefix}_video_generated": videos,
        f"{prefix}_video_generated_rate": videos / total if total else None,
        f"{prefix}_answer_correct": correct,
        f"{prefix}_answer_accuracy": correct / total if total else None,
        f"{prefix}_reused_existing_results": reused,
        f"{prefix}_fresh_runs": total - reused,
        f"{prefix}_llm_total_tokens": sum(token_values),
        f"{prefix}_avg_total_tokens": (sum(token_values) / len(token_values)) if token_values else None,
        f"{prefix}_avg_duration_ms": (sum(duration_values) / len(duration_values)) if duration_values else None,
    }


def main() -> int:
    args = parse_args()
    paths = make_paths(args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.full_system_dir.mkdir(parents=True, exist_ok=True)
    paths.plain_simple_pipeline_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, args.limit, args.selection_rank_start)
    if not rows:
        raise SystemExit(f"No video-eval rows found in manifest: {args.manifest}")

    full_results: list[dict[str, Any]] = [] if args.run_full_system else read_jsonl(paths.full_results_jsonl)
    plain_simple_results: list[dict[str, Any]] = (
        [] if args.run_plain_simple_pipeline else read_jsonl(paths.plain_simple_results_jsonl)
    )

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(rows)}] {sample_id}")
        if args.run_full_system:
            full = run_full_system(row, args, paths)
            full_results.append(full)
            print(
                "  Full System: "
                f"answer={full.get('prediction')} "
                f"correct={full.get('answer_correct')} "
                f"video={full.get('video_generated')} "
                f"status={full.get('status')}"
            )
        if args.run_plain_simple_pipeline:
            plain = run_plain_simple_pipeline(row, args, paths)
            plain_simple_results.append(plain)
            print(
                "  Plain Simple: "
                f"answer={plain.get('prediction')} "
                f"correct={plain.get('answer_correct')} "
                f"video={plain.get('video_generated')} "
                f"status={plain.get('status')}"
            )

    write_jsonl(paths.full_results_jsonl, full_results)
    write_jsonl(paths.plain_simple_results_jsonl, plain_simple_results)
    comparison_rows = write_comparison(paths.comparison_csv, rows, full_results, plain_simple_results)
    summary = {
        "manifest": str(args.manifest),
        "output_dir": str(paths.output_dir),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "sample_count": len(rows),
        "full_system_vision_model": args.full_system_vision_model,
        "plain_simple_pipeline_model": args.plain_simple_pipeline_model,
        "full_system_use_deepseek": args.full_system_use_deepseek,
        "output_language": args.output_language,
        "synthesize_tts": args.synthesize_tts,
        "render_video": args.render_video,
        "reuse_existing_plain_simple_any_version": args.reuse_existing_plain_simple_any_version,
        "reuse_failed_results": args.reuse_failed_results,
        **summarize(full_results, "full_system"),
        **summarize(plain_simple_results, "plain_simple"),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "full_system_results": str(paths.full_results_jsonl),
            "plain_simple_results": str(paths.plain_simple_results_jsonl),
            "comparison_csv": str(paths.comparison_csv),
        },
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
