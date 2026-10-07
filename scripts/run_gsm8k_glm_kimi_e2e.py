"""Run GSM8K image-to-answer E2E baselines for GLM and Kimi.

This runner is for external multimodal end-to-end baselines only:
- input is the rendered GSM8K problem image;
- no Vision Parser, Solver, CAS, Verifier, TeachingPlan, EDS, TTS, or Renderer;
- each model returns JSON with final_answer and reasoning_summary;
- answers are compared with manifest gold_final_answer.

PyCharm usage:
1. Make sure `.env` contains `ZAI_API_KEY` and `MOONSHOT_API_KEY`.
2. Edit the DEFAULT_* constants below if needed.
3. Click Run. Existing per-sample result.json files are reused unless
   `--overwrite` or `--rerun-failed` is passed.
"""

from __future__ import annotations

import argparse
import base64
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
from openai import OpenAI

from mathexplain.services.usage import usage_to_dict


REPO_ROOT = Path(__file__).resolve().parents[1]

# PyCharm-friendly defaults. Edit these values directly, then click Run.
DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
FORMAL_200_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_glm_kimi_e2e_200"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/test1"
DEFAULT_LIMIT = 20
DEFAULT_SELECTION_RANK_START = 0
DEFAULT_PROVIDERS = "glm,kimi"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_TOKENS = 2048
DEFAULT_REASONING_PROFILE = "paper_calibrated"
CALIBRATION_TARGETS = {
    "glm": {"correct": 155, "accuracy": 0.775},
    "kimi": {"correct": 166, "accuracy": 0.83},
}

# GLM / Z.AI defaults.
# The paper_calibrated profile intentionally preserves the settings that
# produced the current 200-question baseline: GLM 155/200 and Kimi 166/200.
# Do not change the default profile when rerunning the paper baseline.
DEFAULT_GLM_MODEL = "glm-4.6v"
DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_API_KEY_ENV = "ZAI_API_KEY"
DEFAULT_GLM_TEMPERATURE = 0.0

# Kimi / Moonshot defaults.
# On this local network, https://api.moonshot.ai/v1 fails TLS, while the China
# endpoint is reachable with the configured key.
DEFAULT_KIMI_MODEL = "kimi-k2.6"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_API_KEY_ENV = "MOONSHOT_API_KEY"
DEFAULT_KIMI_TEMPERATURE = 0.6


E2E_SYSTEM_PROMPT = (
    "You are a careful math word-problem solver. Read the problem from the image, "
    "solve it independently, and return only valid JSON. Keep internal reasoning brief "
    "so that the final JSON is always emitted."
)

E2E_USER_PROMPT = """Solve the math word problem shown in the image.

Return exactly this JSON object, with no markdown:
{
  "final_answer": "the final answer only",
  "reasoning_summary": "brief reasoning, no more than 5 sentences"
}

Do not include units unless they are necessary to distinguish the answer.
Do not include Markdown fences or extra text."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float | None
    extra_body: dict[str, Any] | None = None
    reasoning_profile: str = DEFAULT_REASONING_PROFILE
    calibration_target_correct: int | None = None
    calibration_target_accuracy: float | None = None


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    comparison_csv: Path
    summary_json: Path
    results_jsonl_by_provider: dict[str, Path]
    per_provider_dir: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLM/Kimi GSM8K image E2E answer baselines.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--selection-rank-start", type=int, default=DEFAULT_SELECTION_RANK_START)
    parser.add_argument(
        "--providers",
        default=DEFAULT_PROVIDERS,
        help="Comma-separated providers to run. Supported: glm,kimi.",
    )
    parser.add_argument("--glm-model", default=DEFAULT_GLM_MODEL)
    parser.add_argument("--glm-base-url", default=DEFAULT_GLM_BASE_URL)
    parser.add_argument("--glm-api-key-env", default=DEFAULT_GLM_API_KEY_ENV)
    parser.add_argument("--kimi-model", default=DEFAULT_KIMI_MODEL)
    parser.add_argument("--kimi-base-url", default=DEFAULT_KIMI_BASE_URL)
    parser.add_argument("--kimi-api-key-env", default=DEFAULT_KIMI_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-profile",
        default=DEFAULT_REASONING_PROFILE,
        choices=["paper_calibrated", "disabled", "bounded", "full"],
        help=(
            "paper_calibrated preserves the current 200-question baseline settings; "
            "disabled forces no thinking; bounded/full are diagnostic profiles only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun existing per-sample results.")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun existing failed or unparsable samples while still reusing passed samples.",
    )
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def validate_output_safety(args: argparse.Namespace) -> None:
    """Protect the formal 200-question baseline directory from accidental mutation."""

    default_output = FORMAL_200_OUTPUT_DIR.resolve()
    requested_output = Path(args.output_dir).resolve()
    if requested_output != default_output:
        return

    baseline_shape = args.limit == DEFAULT_LIMIT and args.selection_rank_start == DEFAULT_SELECTION_RANK_START
    if not baseline_shape:
        raise ValueError(
            "Refusing to write a non-200 run into the formal 200-question baseline directory. "
            "Use a separate output directory, for example "
            "data/experiments/gsm8k/answer_runs/formal_glm_kimi_e2e_calibrated_500."
        )
    if args.overwrite or args.rerun_failed:
        raise ValueError(
            "Refusing to overwrite or rerun failed samples in the formal 200-question baseline directory. "
            "Use a separate output directory for exploratory reruns."
        )


def load_experiment_env(env_path: Path) -> None:
    """Load .env without allowing placeholder values to shadow real keys."""

    load_dotenv(env_path, override=False)
    defaults = env_values_first_non_placeholder(env_path)
    for key, value in defaults.items():
        current = os.environ.get(key)
        if is_placeholder_env_value(current):
            os.environ[key] = value


def env_values_first_non_placeholder(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        if not is_placeholder_env_value(cleaned):
            values.setdefault(key.strip(), cleaned)
    return values


def is_placeholder_env_value(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return not text or "your_" in text or text.endswith("_here") or "placeholder" in text


def selected_provider_names(value: str) -> list[str]:
    names: list[str] = []
    for raw in value.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in {"glm", "kimi"}:
            raise ValueError(f"Unsupported provider {name!r}; supported providers are glm,kimi.")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one provider must be selected.")
    return names


def provider_configs(args: argparse.Namespace) -> list[ProviderConfig]:
    selected = selected_provider_names(args.providers)
    profile = args.reasoning_profile
    glm_temperature, glm_extra_body = provider_reasoning_settings("glm", profile)
    kimi_temperature, kimi_extra_body = provider_reasoning_settings("kimi", profile)
    configs = {
        "glm": ProviderConfig(
            name="glm",
            model=args.glm_model,
            base_url=args.glm_base_url.rstrip("/"),
            api_key_env=args.glm_api_key_env,
            temperature=glm_temperature,
            extra_body=glm_extra_body,
            reasoning_profile=profile,
            calibration_target_correct=CALIBRATION_TARGETS["glm"]["correct"],
            calibration_target_accuracy=CALIBRATION_TARGETS["glm"]["accuracy"],
        ),
        "kimi": ProviderConfig(
            name="kimi",
            model=args.kimi_model,
            base_url=args.kimi_base_url.rstrip("/"),
            api_key_env=args.kimi_api_key_env,
            temperature=kimi_temperature,
            extra_body=kimi_extra_body,
            reasoning_profile=profile,
            calibration_target_correct=CALIBRATION_TARGETS["kimi"]["correct"],
            calibration_target_accuracy=CALIBRATION_TARGETS["kimi"]["accuracy"],
        ),
    }
    return [configs[name] for name in selected]


def provider_reasoning_settings(provider_name: str, profile: str) -> tuple[float | None, dict[str, Any] | None]:
    """Return temperature and extra_body for a calibrated reasoning profile."""

    if profile in {"paper_calibrated", "disabled"}:
        if provider_name == "glm":
            return DEFAULT_GLM_TEMPERATURE, {"thinking": {"type": "disabled"}}
        if provider_name == "kimi":
            return DEFAULT_KIMI_TEMPERATURE, {"thinking": {"type": "disabled"}}

    if profile == "bounded":
        if provider_name == "glm":
            # GLM-4.6V defaults to hybrid thinking when the thinking field is omitted.
            return DEFAULT_GLM_TEMPERATURE, None
        if provider_name == "kimi":
            # Kimi K2.6 thinking mode fixes sampling internally; omit temperature.
            return None, {"thinking": {"type": "enabled"}}

    if profile == "full":
        if provider_name == "glm":
            return DEFAULT_GLM_TEMPERATURE, {"thinking": {"type": "enabled"}}
        if provider_name == "kimi":
            return None, {"thinking": {"type": "enabled"}}

    raise ValueError(f"Unsupported provider/profile combination: {provider_name!r}/{profile!r}")


def make_paths(output_dir: Path, providers: list[ProviderConfig]) -> ExperimentPaths:
    results_jsonl_by_provider = {
        provider.name: output_dir / f"{provider.name}_e2e_results.jsonl" for provider in providers
    }
    per_provider_dir = {provider.name: output_dir / provider.name for provider in providers}
    return ExperimentPaths(
        output_dir=output_dir,
        comparison_csv=output_dir / "comparison.csv",
        summary_json=output_dir / "summary.json",
        results_jsonl_by_provider=results_jsonl_by_provider,
        per_provider_dir=per_provider_dir,
    )


def load_manifest(path: Path, limit: int, selection_rank_start: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("include_in_answer_eval")]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    return rows[:limit]


def run_provider_sample(
    *,
    row: dict[str, Any],
    provider: ProviderConfig,
    sample_dir: Path,
    timeout_seconds: int,
    max_tokens: int,
    overwrite: bool,
    rerun_failed: bool,
) -> dict[str, Any]:
    result_path = sample_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        refreshed = refresh_correctness(existing, row)
        if not rerun_failed or result_is_reusable(refreshed):
            result_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
            return refreshed

    sample_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise ValueError(f"{provider.api_key_env} is required for {provider.name} E2E baseline.")

    image_path = resolve_repo_path(row["image_path"])
    image_data_uri = image_to_data_uri(image_path)
    started = time.perf_counter()
    raw_content = ""
    error: str | None = None
    usage: dict[str, Any] = {}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        request_kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": E2E_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": E2E_USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                },
            ],
            "max_tokens": max_tokens,
        }
        if provider.temperature is not None:
            request_kwargs["temperature"] = provider.temperature
        if provider.extra_body is not None:
            request_kwargs["extra_body"] = provider.extra_body
        response = client.chat.completions.create(**request_kwargs)
        raw_content = response.choices[0].message.content or ""
        usage = usage_to_dict(getattr(response, "usage", None))
    except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
        error = f"{type(exc).__name__}: {exc}"

    duration_ms = int((time.perf_counter() - started) * 1000)
    parsed = parse_json_object(raw_content)
    final_answer = extract_reported_final_answer(parsed)
    parse_success = isinstance(parsed, dict) and final_answer not in {None, ""}
    if error is None and not raw_content.strip():
        error = (
            "empty_model_response"
            if int(usage.get("completion_tokens") or 0) < max_tokens
            else "empty_model_response_after_max_tokens"
        )
    elif error is None and not parse_success:
        error = "model_response_did_not_contain_valid_final_answer_json"
    normalized_prediction = normalize_answer(final_answer)
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    finish_reason = None
    try:
        finish_reason = response.choices[0].finish_reason  # type: ignore[name-defined]
    except Exception:
        finish_reason = None
    result = {
        "sample_id": row["sample_id"],
        "selection_rank": row["selection_rank"],
        "image_path": row["image_path"],
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": final_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "answer_correct": normalized_prediction == normalized_gold and normalized_gold != "",
        "parse_success": parse_success,
        "status": "passed" if error is None else "failed",
        "failure_reason": error,
        "provider": provider.name,
        "model": provider.model,
        "base_url": provider.base_url,
        "temperature": provider.temperature,
        "extra_body": provider.extra_body,
        "reasoning_profile": provider.reasoning_profile,
        "calibration_target_correct": provider.calibration_target_correct,
        "calibration_target_accuracy": provider.calibration_target_accuracy,
        "latency_ms": duration_ms,
        "usage": usage,
        "finish_reason": finish_reason,
        "max_tokens": max_tokens,
        "llm_total_tokens": int(usage.get("total_tokens") or 0),
        "llm_prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(usage.get("completion_tokens") or 0),
        "llm_call_count": 1 if error is None else 0,
        "parsed_response": parsed,
        "raw_response": raw_content,
        "artifact_dir": str(sample_dir),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def extract_reported_final_answer(parsed: dict[str, Any] | None) -> str | None:
    """Extract the answer the model appears to report, without using gold labels.

    Some providers emit an inconsistent JSON final_answer while the immediately
    following reasoning_summary states the final result correctly. For baseline
    robustness, prefer an explicit final-answer statement or the last equation
    result in the summary when it disagrees with the JSON field.
    """

    if not isinstance(parsed, dict):
        return None
    json_answer = parsed.get("final_answer")
    summary = parsed.get("reasoning_summary")
    if not isinstance(summary, str) or not summary.strip():
        return str(json_answer).strip() if json_answer not in {None, ""} else None

    summary_answer = extract_answer_from_reasoning_summary(summary)
    if summary_answer not in {None, ""}:
        return summary_answer
    return str(json_answer).strip() if json_answer not in {None, ""} else None


def extract_answer_from_reasoning_summary(summary: str) -> str | None:
    text = normalize_reasoning_text(summary)
    explicit_patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([$]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
        r"(?:therefore|thus)\b[^.。]*?(?:final\s+answer|answer)\s*(?:is|:|=)\s*([$]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
        r"\bso\s+[A-Za-z][A-Za-z0-9_]*\s*=\s*([$]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
    ]
    for pattern in explicit_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return clean_extracted_answer(matches[-1].group(1))

    sentence_candidates = re.split(r"(?<=[.。!?])\s+", text.strip())
    for sentence in reversed(sentence_candidates[-3:]):
        equation_matches = list(
            re.finditer(
                r"=\s*([$]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(?:[a-zA-Z%$]+)?",
                sentence,
            )
        )
        if equation_matches:
            return clean_extracted_answer(equation_matches[-1].group(1))
    return None


def normalize_reasoning_text(value: str) -> str:
    text = value.replace("×", "*").replace("÷", "/")
    text = text.replace("¡Á", "*").replace("¡Â", "/").replace("¡ú", "=")
    text = text.replace("鈥攚", " ").replace("鈥攕", " ")
    return text


def clean_extracted_answer(value: str) -> str:
    return value.strip().replace(" ", "").rstrip(".,;:)")


def image_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    think_end = stripped.rfind("</think>")
    if think_end >= 0:
        after_think = stripped[think_end + len("</think>") :].strip()
        if after_think:
            candidates.append(after_think)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    parsed_objects = parse_embedded_json_objects(stripped)
    for value in reversed(parsed_objects):
        if value.get("final_answer") not in {None, ""}:
            return value
    return None


def parse_embedded_json_objects(text: str) -> list[dict[str, Any]]:
    """Return valid JSON objects embedded in noisy model text."""

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


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
    raw_response = refreshed.get("raw_response")
    if isinstance(raw_response, str) and raw_response.strip():
        parsed = parse_json_object(raw_response)
        final_answer = extract_reported_final_answer(parsed)
        if final_answer not in {None, ""}:
            refreshed["prediction"] = final_answer
            refreshed["parsed_response"] = parsed
            refreshed["parse_success"] = True
            if refreshed.get("failure_reason") == "model_response_did_not_contain_valid_final_answer_json":
                refreshed["failure_reason"] = None
                refreshed["status"] = "passed"
            if refreshed.get("status") == "passed" and int(refreshed.get("llm_call_count") or 0) == 0:
                refreshed["llm_call_count"] = 1
    normalized_prediction = normalize_answer(refreshed.get("prediction"))
    normalized_gold = normalize_answer(row.get("gold_final_answer"))
    refreshed["gold_final_answer"] = row.get("gold_final_answer")
    refreshed["normalized_prediction"] = normalized_prediction
    refreshed["normalized_gold"] = normalized_gold
    refreshed["answer_correct"] = normalized_prediction == normalized_gold and normalized_gold != ""
    return refreshed


def result_is_reusable(result: dict[str, Any]) -> bool:
    if result.get("status") != "passed":
        return False
    if result.get("parse_success") is not True:
        return False
    if result.get("prediction") in {None, ""}:
        return False
    return True


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_comparison(
    path: Path,
    manifest_rows: list[dict[str, Any]],
    results_by_provider: dict[str, list[dict[str, Any]]],
    providers: list[ProviderConfig],
) -> list[dict[str, Any]]:
    by_provider = {
        provider.name: {row["sample_id"]: row for row in results_by_provider.get(provider.name, [])}
        for provider in providers
    }
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        sample_id = item["sample_id"]
        row = {
            "sample_id": sample_id,
            "selection_rank": item["selection_rank"],
            "gold_final_answer": item.get("gold_final_answer"),
            "image_path": item.get("image_path"),
        }
        for provider in providers:
            result = by_provider.get(provider.name, {}).get(sample_id, {})
            prefix = provider.name
            row.update(
                {
                    f"{prefix}_prediction": result.get("prediction"),
                    f"{prefix}_correct": result.get("answer_correct"),
                    f"{prefix}_status": result.get("status"),
                    f"{prefix}_parse_success": result.get("parse_success"),
                    f"{prefix}_model": result.get("model"),
                    f"{prefix}_latency_ms": result.get("latency_ms"),
                    f"{prefix}_llm_total_tokens": result.get("llm_total_tokens"),
                }
            )
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row.get("answer_correct"))
    passed = sum(1 for row in results if row.get("status") == "passed")
    parse_success = sum(1 for row in results if row.get("parse_success"))
    latency = [int(row.get("latency_ms") or 0) for row in results]
    total_tokens = [int(row.get("llm_total_tokens") or 0) for row in results]
    prompt_tokens = [int(row.get("llm_prompt_tokens") or 0) for row in results]
    completion_tokens = [int(row.get("llm_completion_tokens") or 0) for row in results]
    return {
        "sample_count": total,
        "answer_correct": correct,
        "answer_accuracy": correct / total if total else 0.0,
        "status_passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "parse_success_count": parse_success,
        "parse_success_rate": parse_success / total if total else 0.0,
        "avg_latency_ms": int(sum(latency) / total) if total else 0,
        "total_llm_tokens": sum(total_tokens),
        "avg_llm_tokens": int(sum(total_tokens) / total) if total else 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "avg_prompt_tokens": int(sum(prompt_tokens) / total) if total else 0,
        "total_completion_tokens": sum(completion_tokens),
        "avg_completion_tokens": int(sum(completion_tokens) / total) if total else 0,
    }


def main() -> int:
    args = parse_args()
    validate_output_safety(args)
    load_experiment_env(REPO_ROOT / ".env")
    providers = provider_configs(args)
    paths = make_paths(args.output_dir, providers)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    for provider in providers:
        paths.per_provider_dir[provider.name].mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, args.limit, args.selection_rank_start)
    results_by_provider: dict[str, list[dict[str, Any]]] = {provider.name: [] for provider in providers}

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        if not args.quiet_progress:
            print(f"[{index}/{len(rows)}] rank={row.get('selection_rank')} sample_id={sample_id}", flush=True)
        for provider in providers:
            result = run_provider_sample(
                row=row,
                provider=provider,
                sample_dir=paths.per_provider_dir[provider.name] / sample_id,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                overwrite=args.overwrite,
                rerun_failed=args.rerun_failed,
            )
            results_by_provider[provider.name].append(result)
            if not args.quiet_progress:
                print(
                    f"  {provider.name}:",
                    result.get("prediction"),
                    "correct=",
                    result.get("answer_correct"),
                    "status=",
                    result.get("status"),
                    flush=True,
                )

    for provider in providers:
        write_jsonl(paths.results_jsonl_by_provider[provider.name], results_by_provider[provider.name])
    comparison_rows = write_comparison(paths.comparison_csv, rows, results_by_provider, providers)
    summary = {
        "experiment": "gsm8k_external_mllm_e2e_answer",
        "definition": (
            "External multimodal end-to-end image-to-answer baselines; no MathExplainAgent internal pipeline. "
            "The default paper_calibrated profile preserves the current 200-question baseline settings."
        ),
        "manifest": str(args.manifest),
        "output_dir": str(paths.output_dir),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "reasoning_profile": args.reasoning_profile,
        "calibration_targets": CALIBRATION_TARGETS,
        "providers": [
            {
                "name": provider.name,
                "model": provider.model,
                "base_url": provider.base_url,
                "api_key_env": provider.api_key_env,
                "temperature": provider.temperature,
                "extra_body": provider.extra_body,
                "reasoning_profile": provider.reasoning_profile,
                "calibration_target_correct": provider.calibration_target_correct,
                "calibration_target_accuracy": provider.calibration_target_accuracy,
            }
            for provider in providers
        ],
        "provider_summaries": {
            provider.name: summarize(results_by_provider[provider.name]) for provider in providers
        },
        "comparison_count": len(comparison_rows),
        "artifacts": {
            "comparison_csv": str(paths.comparison_csv),
            "summary_json": str(paths.summary_json),
            "results_jsonl_by_provider": {
                provider.name: str(paths.results_jsonl_by_provider[provider.name]) for provider in providers
            },
            "per_provider_dir": {
                provider.name: str(paths.per_provider_dir[provider.name]) for provider in providers
            },
        },
    }
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
