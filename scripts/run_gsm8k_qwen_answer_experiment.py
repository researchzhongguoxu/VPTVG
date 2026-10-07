"""Run GSM8K answer-accuracy experiment: Full System vs Qwen end-to-end.

PyCharm usage:
1. Make sure `.env` contains `DASHSCOPE_API_KEY` and `DEEPSEEK_API_KEY`.
2. Open this file and click Run.
3. By default it runs the first 5 manifest samples as a real API smoke test.

For the full pilot, change DEFAULT_LIMIT to 100.
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

from dotenv import load_dotenv
from openai import OpenAI

from mathexplain.agents.vision_parser.spr_generator import SPRGenerator
from mathexplain.config import PipelineModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# PyCharm-friendly defaults. Edit these values directly, then click Run.
DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/qwen36_new_200"
DEFAULT_LIMIT = 200
DEFAULT_SELECTION_RANK_START = 0
DEFAULT_RUN_FULL_SYSTEM = True
DEFAULT_RUN_QWEN_E2E = True
DEFAULT_FULL_SYSTEM_USE_DEEPSEEK = True
DEFAULT_VISION_MODEL = "qwen3.6-flash"
DEFAULT_QWEN_MODEL = "qwen3.6-plus"


QWEN_E2E_SYSTEM_PROMPT = (
    "You are a careful grade-school math solver. Read the problem from the image, "
    "solve it independently, and return only valid JSON."
)

QWEN_E2E_USER_PROMPT = """Solve the math word problem shown in the image.

Return only this JSON object:
{
  "final_answer": "the final answer only",
  "reasoning_summary": "brief reasoning, no more than 5 sentences"
}

Do not include Markdown fences or extra text."""


@dataclass
class ExperimentPaths:
    output_dir: Path
    full_system_dir: Path
    qwen_e2e_dir: Path
    comparison_csv: Path
    full_results_jsonl: Path
    qwen_results_jsonl: Path
    summary_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K answer experiment comparing Full System and Qwen E2E."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--selection-rank-start", type=int, default=DEFAULT_SELECTION_RANK_START)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--full-system-use-deepseek",
        action="store_true",
        default=DEFAULT_FULL_SYSTEM_USE_DEEPSEEK,
        help="Allow the Full System Solver to call DeepSeek planner. On by default.",
    )
    parser.add_argument(
        "--skip-full-system",
        action="store_false",
        dest="run_full_system",
        default=DEFAULT_RUN_FULL_SYSTEM,
    )
    parser.add_argument(
        "--skip-qwen-e2e",
        action="store_false",
        dest="run_qwen_e2e",
        default=DEFAULT_RUN_QWEN_E2E,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun samples even when result files already exist.",
    )
    return parser.parse_args()


def make_paths(output_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        output_dir=output_dir,
        full_system_dir=output_dir / "full_system",
        qwen_e2e_dir=output_dir / "qwen_e2e",
        comparison_csv=output_dir / "comparison.csv",
        full_results_jsonl=output_dir / "full_system_results.jsonl",
        qwen_results_jsonl=output_dir / "qwen_e2e_results.jsonl",
        summary_json=output_dir / "summary.json",
    )


def load_manifest(path: Path, limit: int, selection_rank_start: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("include_in_answer_eval")]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    return rows[:limit]


def run_full_system(
    row: dict[str, Any],
    sample_dir: Path,
    config: PipelineModelConfig,
    use_deepseek: bool,
    overwrite: bool,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing = refresh_correctness(existing, row)
        if result_is_reusable(existing, require_correct=True) and result_has_llm_usage(existing):
            result_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            return existing
        summary_path = find_pipeline_summary(sample_dir)
        if summary_path is not None:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result = build_full_system_result(
                row=row,
                sample_dir=sample_dir,
                summary=summary,
                summary_path=summary_path,
                duration_ms=int(nested_get(summary, ["timing", "total_duration_ms"]) or 0),
                returncode=int(existing.get("returncode") or 0),
                stdout_tail=existing.get("stdout_tail", ""),
                stderr_tail=existing.get("stderr_tail", ""),
            )
            if result_is_reusable(result, require_correct=True) and result_has_llm_usage(result):
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result

    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = REPO_ROOT / row["image_path"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_pipeline.py"),
        "--image",
        str(image_path),
        "--output-dir",
        str(sample_dir),
        "--vision-mode",
        "qwen",
        "--vision-model",
        config.vision.model,
        "--vision-base-url",
        config.vision.base_url,
        "--vision-api-key-env",
        config.vision.api_key_env,
        "--stop-after",
        "verifier",
        "--quiet-progress",
    ]
    if not use_deepseek:
        cmd.append("--no-deepseek")

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=300,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)

    summary_path = find_pipeline_summary(sample_dir)
    summary: dict[str, Any] = {}
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = build_full_system_result(
        row=row,
        sample_dir=sample_dir,
        summary=summary,
        summary_path=summary_path,
        duration_ms=duration_ms,
        returncode=completed.returncode,
        stdout_tail=(completed.stdout or "")[-4000:],
        stderr_tail=(completed.stderr or "")[-4000:],
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_full_system_result(
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
    return {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "status": summary.get("status", "failed" if returncode else "unknown"),
        "failed_stage": summary.get("failed_stage"),
        "failure_reason": summary.get("failure_reason"),
        "solver_passed": not bool(nested_get(summary, ["solver", "metadata", "partial_failure"])),
        "verifier_passed": bool(nested_get(summary, ["verification", "valid"]))
        and nested_get(summary, ["verification", "status"]) == "passed",
        "cas_backed": bool(nested_get(summary, ["cas", "backend_confirmed"])),
        "llm_usage": llm_usage,
        "llm_total_tokens": int(llm_usage.get("total_tokens") or 0),
        "llm_prompt_tokens": int(llm_usage.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(llm_usage.get("completion_tokens") or 0),
        "llm_call_count": int(llm_usage.get("llm_call_count") or 0),
        "latency_ms": duration_ms,
        "returncode": returncode,
        "artifact_dir": str(summary_path.parent if summary_path is not None else sample_dir),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def find_pipeline_summary(sample_dir: Path) -> Path | None:
    candidates = list(sample_dir.glob("summary.json")) + list(sample_dir.glob("*/summary.json"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_qwen_e2e(
    row: dict[str, Any],
    sample_dir: Path,
    client: OpenAI,
    model: str,
    overwrite: bool,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing = refresh_correctness(existing, row)
        if result_is_reusable(existing, require_correct=False):
            result_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            return existing

    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = REPO_ROOT / row["image_path"]
    image_data_uri = SPRGenerator._image_to_data_uri(str(image_path))
    started = time.perf_counter()
    raw_content = ""
    error = None
    usage: dict[str, Any] = {}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QWEN_E2E_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QWEN_E2E_USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw_content = response.choices[0].message.content or ""
        if getattr(response, "usage", None) is not None:
            usage = response.usage.model_dump()
    except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
        error = str(exc)

    duration_ms = int((time.perf_counter() - started) * 1000)
    parsed = parse_json_object(raw_content)
    final_answer = parsed.get("final_answer") if isinstance(parsed, dict) else None
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    result = {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "parse_success": isinstance(parsed, dict) and final_answer is not None,
        "status": "passed" if error is None else "failed",
        "failure_reason": error,
        "latency_ms": duration_ms,
        "model": model,
        "usage": usage,
        "llm_total_tokens": int(usage.get("total_tokens") or 0),
        "llm_prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(usage.get("completion_tokens") or 0),
        "llm_call_count": 1 if usage else 0,
        "raw_response": raw_content,
        "artifact_dir": str(sample_dir),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


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


def result_is_reusable(result: dict[str, Any], *, require_correct: bool = False) -> bool:
    if result.get("status") != "passed":
        return False
    if result.get("prediction") in {None, ""}:
        return False
    if "parse_success" in result and result.get("parse_success") is not True:
        return False
    if require_correct and result.get("answer_correct") is not True:
        return False
    return True


def result_has_llm_usage(result: dict[str, Any]) -> bool:
    usage = result.get("llm_usage")
    return isinstance(usage, dict) and "total_tokens" in usage and "llm_call_count" in usage


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_comparison(
    path: Path,
    manifest_rows: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
    qwen_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_by_id = {row["sample_id"]: row for row in full_results}
    qwen_by_id = {row["sample_id"]: row for row in qwen_results}
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        sample_id = item["sample_id"]
        full = full_by_id.get(sample_id, {})
        qwen = qwen_by_id.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "selection_rank": item["selection_rank"],
                "gold_final_answer": item.get("gold_final_answer"),
                "full_system_prediction": full.get("prediction"),
                "full_system_correct": full.get("answer_correct"),
                "full_system_status": full.get("status"),
                "full_system_failed_stage": full.get("failed_stage"),
                "full_system_verifier_passed": full.get("verifier_passed"),
                "full_system_cas_backed": full.get("cas_backed"),
                "full_system_llm_total_tokens": full.get("llm_total_tokens"),
                "full_system_llm_call_count": full.get("llm_call_count"),
                "qwen_e2e_prediction": qwen.get("prediction"),
                "qwen_e2e_correct": qwen.get("answer_correct"),
                "qwen_e2e_status": qwen.get("status"),
                "qwen_e2e_parse_success": qwen.get("parse_success"),
                "qwen_e2e_llm_total_tokens": qwen.get("llm_total_tokens"),
                "qwen_e2e_llm_call_count": qwen.get("llm_call_count"),
                "image_path": item["image_path"],
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(results: list[dict[str, Any]], key_prefix: str) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row.get("answer_correct"))
    passed = sum(1 for row in results if row.get("status") == "passed")
    latency = [int(row.get("latency_ms") or 0) for row in results]
    total_tokens = [int(row.get("llm_total_tokens") or 0) for row in results]
    prompt_tokens = [int(row.get("llm_prompt_tokens") or 0) for row in results]
    completion_tokens = [int(row.get("llm_completion_tokens") or 0) for row in results]
    call_counts = [int(row.get("llm_call_count") or 0) for row in results]
    return {
        f"{key_prefix}_total": total,
        f"{key_prefix}_correct": correct,
        f"{key_prefix}_accuracy": correct / total if total else 0.0,
        f"{key_prefix}_passed": passed,
        f"{key_prefix}_pass_rate": passed / total if total else 0.0,
        f"{key_prefix}_avg_latency_ms": int(sum(latency) / total) if total else 0,
        f"{key_prefix}_total_llm_tokens": sum(total_tokens),
        f"{key_prefix}_avg_llm_tokens": int(sum(total_tokens) / total) if total else 0,
        f"{key_prefix}_total_prompt_tokens": sum(prompt_tokens),
        f"{key_prefix}_avg_prompt_tokens": int(sum(prompt_tokens) / total) if total else 0,
        f"{key_prefix}_total_completion_tokens": sum(completion_tokens),
        f"{key_prefix}_avg_completion_tokens": int(sum(completion_tokens) / total) if total else 0,
        f"{key_prefix}_total_llm_calls": sum(call_counts),
        f"{key_prefix}_avg_llm_calls": sum(call_counts) / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    config = PipelineModelConfig.from_sources(
        vision_model=args.vision_model,
    )
    qwen_model = args.qwen_model or config.vision.model
    api_key = os.getenv(config.vision.api_key_env)
    if args.run_qwen_e2e and not api_key:
        raise ValueError(f"{config.vision.api_key_env} is required for Qwen E2E baseline.")

    paths = make_paths(args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.full_system_dir.mkdir(parents=True, exist_ok=True)
    paths.qwen_e2e_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, args.limit, args.selection_rank_start)
    qwen_client = (
        OpenAI(api_key=api_key, base_url=config.vision.base_url, timeout=120, max_retries=0)
        if api_key
        else None
    )
    full_results: list[dict[str, Any]] = []
    qwen_results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(rows)}] {sample_id}")
        if args.run_full_system:
            full = run_full_system(
                row,
                paths.full_system_dir / sample_id,
                config,
                use_deepseek=args.full_system_use_deepseek,
                overwrite=args.overwrite,
            )
            full_results.append(full)
            print(
                "  Full System:",
                full.get("prediction"),
                "correct=",
                full.get("answer_correct"),
                "status=",
                full.get("status"),
            )
        if args.run_qwen_e2e and qwen_client is not None:
            qwen = run_qwen_e2e(
                row,
                paths.qwen_e2e_dir / sample_id,
                qwen_client,
                qwen_model,
                overwrite=args.overwrite,
            )
            qwen_results.append(qwen)
            print(
                "  Qwen E2E:",
                qwen.get("prediction"),
                "correct=",
                qwen.get("answer_correct"),
                "status=",
                qwen.get("status"),
            )

    write_jsonl(paths.full_results_jsonl, full_results)
    write_jsonl(paths.qwen_results_jsonl, qwen_results)
    comparison_rows = write_comparison(paths.comparison_csv, rows, full_results, qwen_results)
    summary = {
        "manifest": str(args.manifest),
        "output_dir": str(paths.output_dir),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "qwen_model": qwen_model,
        "full_system_use_deepseek": args.full_system_use_deepseek,
        "sample_count": len(rows),
        **summarize(full_results, "full_system"),
        **summarize(qwen_results, "qwen_e2e"),
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "full_system_results": str(paths.full_results_jsonl),
            "qwen_e2e_results": str(paths.qwen_results_jsonl),
            "comparison_csv": str(paths.comparison_csv),
        },
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
