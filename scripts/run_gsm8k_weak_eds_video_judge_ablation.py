"""Run MLLM A/B judging for GSM8K Full System vs Weak EDS videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_no_teaching_plan_video_ablation import (
    DEFAULT_FULL_ROOT,
    DEFAULT_MANIFEST,
    METRICS,
    eds_transcript,
    extract_frames,
    find_full_artifacts,
    judge_configs,
    load_manifest,
    parse_json_object,
    read_json,
    resolve_repo_path,
    score_for_system,
    winner_to_system,
    write_csv,
    write_json,
    write_results_jsonl,
)
from scripts.run_gsm8k_weak_eds_video_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_WEAK_EDS_ROOT,
    DEFAULT_SAMPLE_LIST,
)


PROMPT_VERSION = "mllm-video-judge-weak-eds-v1"
DEFAULT_LIMIT = 50
DEFAULT_MAX_FRAMES = 12
DEFAULT_AB_SEED = 20260714
DEFAULT_JUDGE_TIMEOUT_SECONDS = 300.0
DEFAULT_JUDGE_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 12.0
PROGRESS_PREFIX = "[weak-eds-judge]"

JUDGE_PROMPT = """You are evaluating two math explanation videos for the same problem.

You will receive the problem image, the gold final answer, optional reference notes, and two anonymized videos represented by key frames plus narration transcripts.

Judge Video A and Video B blindly. Do not infer system names. Focus on teaching quality, not visual beauty.

Evaluate exactly these four dimensions:
1. formula_source_clarity: whether formulas and quantitative relationships are clearly tied to problem facts or previous verified steps.
2. step_completeness: whether the explanation covers the necessary solution steps without skipping key intermediate reasoning.
3. narration_clarity: whether narration is clear, natural, and mathematically grounded.
4. self_learning_suitability: whether a student could learn the solution from the video without seeing the gold answer first.

For each dimension, assign integer scores from 1 to 5 for A and B. Choose winner as "A", "B", or "tie". Each reason must cite concrete evidence from the frames, formulas, or transcript.

Return JSON only with this schema:
{
  "formula_source_clarity": {"score_A": 1, "score_B": 1, "winner": "A|B|tie", "reason": "..."},
  "step_completeness": {"score_A": 1, "score_B": 1, "winner": "A|B|tie", "reason": "..."},
  "narration_clarity": {"score_A": 1, "score_B": 1, "winner": "A|B|tie", "reason": "..."},
  "self_learning_suitability": {"score_A": 1, "score_B": 1, "winner": "A|B|tie", "reason": "..."},
  "overall_preference": "A|B|tie",
  "overall_reason": "..."
}
"""


@dataclass(frozen=True)
class WeakJudgePaths:
    output_dir: Path
    weak_eds_dir: Path
    judge_inputs_dir: Path
    judge_results_dir: Path
    frames_dir: Path
    prompt_md: Path
    pair_manifest_jsonl: Path
    comparison_csv: Path
    metric_summary_csv: Path
    majority_vote_csv: Path
    summary_json: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Full System vs Weak EDS MLLM A/B judging.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-system-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--weak-eds-root", type=Path, default=DEFAULT_WEAK_EDS_ROOT)
    parser.add_argument("--sample-list", type=Path, default=DEFAULT_SAMPLE_LIST)
    parser.add_argument("--limit", "--num-samples", "--count", dest="limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--selection-rank-start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--ab-seed", type=int, default=DEFAULT_AB_SEED)
    parser.add_argument("--judge-timeout-seconds", type=float, default=DEFAULT_JUDGE_TIMEOUT_SECONDS)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-retries", type=int, default=DEFAULT_JUDGE_RETRIES)
    parser.add_argument("--retry-sleep-seconds", type=float, default=DEFAULT_RETRY_SLEEP_SECONDS)
    parser.add_argument("--no-json-mode", action="store_false", dest="json_mode", default=True)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare frames and judge inputs, but skip MLLM calls.")
    parser.add_argument("--rerun-judge", action="store_true", help="Regenerate existing judge results.")
    parser.add_argument("--judge-glm", action="store_true", dest="judge_glm", default=False)
    parser.add_argument("--no-judge-glm", action="store_false", dest="judge_glm")
    parser.add_argument("--judge-qwen", action="store_true", dest="judge_qwen", default=False)
    parser.add_argument("--no-judge-qwen", action="store_false", dest="judge_qwen")
    parser.add_argument("--judge-doubao", action="store_true", dest="judge_doubao", default=False)
    parser.add_argument("--no-judge-doubao", action="store_false", dest="judge_doubao")
    parser.add_argument("--judge-kimi", action="store_true", dest="judge_kimi", default=True)
    parser.add_argument("--no-judge-kimi", action="store_false", dest="judge_kimi")
    return parser.parse_args(argv)


def weak_eds_judge_configs(args: argparse.Namespace) -> list[Any]:
    configs = judge_configs(args)
    normalized: list[Any] = []
    for config in configs:
        if config.name == "kimi":
            normalized.append(replace(config, temperature=0.6))
        else:
            normalized.append(config)
    return normalized


def weak_judge_paths(root: Path) -> WeakJudgePaths:
    root = root.resolve()
    return WeakJudgePaths(
        output_dir=root,
        weak_eds_dir=root / "weak_eds",
        judge_inputs_dir=root / "judge_inputs",
        judge_results_dir=root / "judge_results",
        frames_dir=root / "frames",
        prompt_md=root / "prompt.md",
        pair_manifest_jsonl=root / "pair_manifest.jsonl",
        comparison_csv=root / "judge_comparison.csv",
        metric_summary_csv=root / "metric_summary_by_judge.csv",
        majority_vote_csv=root / "majority_vote_summary.csv",
        summary_json=root / "judge_summary.json",
    )


def find_weak_result(sample_id: str, paths: WeakJudgePaths) -> dict[str, Any]:
    result_path = paths.weak_eds_dir / sample_id / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Weak EDS result.json missing for {sample_id}: {result_path}")
    result = read_json(result_path)
    video_path = Path(str(result.get("weak_eds_video_path") or result.get("video_path") or ""))
    eds_path = Path(str(result.get("weak_eds_path") or result.get("artifact_paths", {}).get("weak_eds") or ""))
    if not video_path.exists():
        raise FileNotFoundError(f"Weak EDS video missing for {sample_id}: {video_path}")
    if not eds_path.exists():
        raise FileNotFoundError(f"Weak EDS JSON missing for {sample_id}: {eds_path}")
    result["resolved_video_path"] = str(video_path)
    result["resolved_eds_path"] = str(eds_path)
    return result


def ab_mapping(sample_id: str, seed: int) -> dict[str, str]:
    rng = random.Random(f"{seed}:{sample_id}")
    systems = ["full_system", "weak_eds"]
    rng.shuffle(systems)
    return {"A": systems[0], "B": systems[1]}


def compact_video_text(label: str, payload: dict[str, Any]) -> str:
    transcript = "\n".join(f"- {item.get('text')}" for item in payload.get("narration", []) if item.get("text"))
    formulas = "\n".join(f"- {item.get('expression')}" for item in payload.get("formulas", []) if item.get("expression"))
    return (
        f"\nVideo {label} transcript:\n{transcript}\n"
        f"\nVideo {label} formula list:\n{formulas or '(none shown as structured formula tracks)'}\n"
        f"\nVideo {label} final answer shown: {payload.get('final_answer')}\n"
    )


def image_content(path: Path) -> dict[str, Any]:
    import base64

    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def judge_messages(judge_input: dict[str, Any]) -> list[dict[str, Any]]:
    text = (
        f"{JUDGE_PROMPT}\n\n"
        f"Sample id: {judge_input['sample_id']}\n"
        f"Gold final answer: {judge_input.get('gold_final_answer')}\n"
        f"{compact_video_text('A', judge_input['videos']['A'])}\n"
        f"{compact_video_text('B', judge_input['videos']['B'])}\n"
        "Images follow in this order: problem image, Video A frames, Video B frames."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.append(image_content(Path(judge_input["problem_image"])))
    for frame in judge_input["videos"]["A"].get("frames", []):
        content.append(image_content(Path(frame["path"])))
    for frame in judge_input["videos"]["B"].get("frames", []):
        content.append(image_content(Path(frame["path"])))
    return [{"role": "user", "content": content}]


def _judge_payload(config: Any, judge_input: dict[str, Any], *, max_tokens: int, json_mode: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": judge_messages(judge_input),
        "temperature": config.temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if config.name in {"glm", "kimi"}:
        payload["thinking"] = {"type": "disabled"}
    return payload


def _retryable_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in [
            "timeout",
            "timed out",
            "http 429",
            "rate_limit",
            "rate limit",
            "overloaded",
            "temporarily",
            "json",
            "valid json",
        ]
    )


def call_judge_once(
    config: Any,
    judge_input: dict[str, Any],
    *,
    timeout_seconds: float,
    max_tokens: int,
    json_mode: bool,
) -> dict[str, Any]:
    payload = _judge_payload(config, judge_input, max_tokens=max_tokens, json_mode=json_mode)
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_json_object(content)
            return {
                "status": "passed" if parsed is not None else "failed",
                "judge": config.name,
                "model": config.model,
                "elapsed_s": round(time.time() - started, 2),
                "raw_content": content,
                "parsed": parsed,
                "usage": data.get("usage") or {},
                "error": None if parsed is not None else "Judge response did not contain valid JSON.",
                "json_mode": json_mode,
                "thinking_disabled": config.name in {"glm", "kimi"},
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": "failed",
            "judge": config.name,
            "model": config.model,
            "elapsed_s": round(time.time() - started, 2),
            "raw_content": "",
            "parsed": None,
            "usage": {},
            "error": f"HTTP {exc.code}: {body[:1000]}",
            "json_mode": json_mode,
            "thinking_disabled": config.name in {"glm", "kimi"},
        }
    except Exception as exc:  # pragma: no cover - network dependent
        return {
            "status": "failed",
            "judge": config.name,
            "model": config.model,
            "elapsed_s": round(time.time() - started, 2),
            "raw_content": "",
            "parsed": None,
            "usage": {},
            "error": repr(exc),
            "json_mode": json_mode,
            "thinking_disabled": config.name in {"glm", "kimi"},
        }


def call_judge(
    config: Any,
    judge_input: dict[str, Any],
    *,
    timeout_seconds: float,
    max_tokens: int,
    json_mode: bool,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    use_json_mode = json_mode
    total_attempts = max(1, retries + 1)
    for attempt in range(1, total_attempts + 1):
        result = call_judge_once(
            config,
            judge_input,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            json_mode=use_json_mode,
        )
        result["attempt"] = attempt
        attempts.append(_attempt_snapshot(result))
        if result.get("status") == "passed":
            result["attempts"] = attempts
            return result
        error = str(result.get("error") or "")
        if use_json_mode and "response_format" in error and ("400" in error or "unsupported" in error.lower()):
            use_json_mode = False
        if attempt >= total_attempts or not _retryable_error(error):
            result["attempts"] = attempts
            return result
        sleep_seconds = retry_sleep_seconds * attempt
        if "429" in error or "rate" in error.lower() or "overloaded" in error.lower():
            sleep_seconds = max(sleep_seconds, retry_sleep_seconds * attempt * 3)
        print(
            f"{PROGRESS_PREFIX} retry judge={config.name} sample_id={judge_input['sample_id']} "
            f"attempt={attempt}/{total_attempts} sleep={sleep_seconds:.1f}s error={error[:180]}",
            flush=True,
        )
        time.sleep(sleep_seconds)
    final = attempts[-1]
    final["attempts"] = attempts
    return final


def _attempt_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": result.get("attempt"),
        "status": result.get("status"),
        "judge": result.get("judge"),
        "model": result.get("model"),
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
        "json_mode": result.get("json_mode"),
        "thinking_disabled": result.get("thinking_disabled"),
        "usage": result.get("usage") or {},
    }


def prepare_judge_input(
    row: dict[str, Any],
    *,
    full_root: Path,
    paths: WeakJudgePaths,
    max_frames: int,
    ab_seed: int,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    full = find_full_artifacts(row, full_root)
    weak = find_weak_result(sample_id, paths)

    full_frames = extract_frames(
        Path(full["video_path"]),
        Path(full["eds_path"]),
        paths.frames_dir / sample_id / "full_system",
        max_frames=max_frames,
    )
    weak_frames = extract_frames(
        Path(weak["resolved_video_path"]),
        Path(weak["resolved_eds_path"]),
        paths.frames_dir / sample_id / "weak_eds",
        max_frames=max_frames,
    )
    mapping = ab_mapping(sample_id, ab_seed)
    system_payloads = {
        "full_system": {
            "video_path": str(full["video_path"]),
            "frames": full_frames,
            **eds_transcript(Path(full["eds_path"])),
        },
        "weak_eds": {
            "video_path": weak["resolved_video_path"],
            "frames": weak_frames,
            **eds_transcript(Path(weak["resolved_eds_path"])),
        },
    }
    judge_input = {
        "sample_id": sample_id,
        "selection_rank": int(row.get("selection_rank", -1)),
        "prompt_version": PROMPT_VERSION,
        "problem_image": str(resolve_repo_path(row["image_path"])),
        "problem_text": row.get("problem_text") or row.get("question_text"),
        "gold_final_answer": row.get("gold_final_answer"),
        "ab_mapping": mapping,
        "videos": {
            "A": system_payloads[mapping["A"]],
            "B": system_payloads[mapping["B"]],
        },
    }
    write_json(paths.judge_inputs_dir / sample_id / "judge_input.json", judge_input)
    return judge_input


def run_judges_for_sample(
    sample_id: str,
    judge_input: dict[str, Any],
    configs: list[Any],
    *,
    paths: WeakJudgePaths,
    rerun_judge: bool,
    timeout_seconds: float,
    max_tokens: int,
    json_mode: bool,
    retries: int,
    retry_sleep_seconds: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for config in configs:
        path = paths.judge_results_dir / config.name / f"{sample_id}.json"
        if path.exists() and not rerun_judge:
            result = read_json(path)
            if result.get("status") == "passed" and isinstance(result.get("parsed"), dict):
                result["reused_existing_result"] = True
                results.append(result)
                continue
            print(
                f"{PROGRESS_PREFIX} retrying existing failed judge result "
                f"sample_id={sample_id} judge={config.name} error={str(result.get('error'))[:180]}",
                flush=True,
            )
        result = call_judge(
            config,
            judge_input,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            json_mode=json_mode,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        result.update(
            {
                "sample_id": sample_id,
                "prompt_version": PROMPT_VERSION,
                "ab_mapping": judge_input["ab_mapping"],
                "reused_existing_result": False,
            }
        )
        write_json(path, result)
        results.append(result)
    return results


def write_pair_manifest(path: Path, judge_inputs: list[dict[str, Any]]) -> None:
    rows = [
        {
            "sample_id": item["sample_id"],
            "selection_rank": item["selection_rank"],
            "A_system": item["ab_mapping"]["A"],
            "B_system": item["ab_mapping"]["B"],
            "A_frame_count": len(item["videos"]["A"].get("frames", [])),
            "B_frame_count": len(item["videos"]["B"].get("frames", [])),
        }
        for item in judge_inputs
    ]
    write_results_jsonl(path, rows)


def write_judge_comparison(path: Path, rows: list[dict[str, Any]], judge_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for result in judge_results:
        by_sample.setdefault(result["sample_id"], []).append(result)
    comparison: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        output = {
            "sample_id": sample_id,
            "selection_rank": int(row.get("selection_rank", -1)),
            "gold_final_answer": row.get("gold_final_answer"),
        }
        for judge in by_sample.get(sample_id, []):
            parsed = judge.get("parsed") if isinstance(judge.get("parsed"), dict) else {}
            mapping = judge.get("ab_mapping") if isinstance(judge.get("ab_mapping"), dict) else {}
            output[f"{judge['judge']}_status"] = judge.get("status")
            output[f"{judge['judge']}_overall_winner"] = winner_to_system(parsed.get("overall_preference"), mapping)
            output[f"{judge['judge']}_total_tokens"] = (judge.get("usage") or {}).get("total_tokens")
            for metric in METRICS:
                output[f"{judge['judge']}_{metric}_full"] = score_for_system(parsed, metric, "full_system", mapping)
                output[f"{judge['judge']}_{metric}_weak_eds"] = score_for_system(parsed, metric, "weak_eds", mapping)
        comparison.append(output)
    fieldnames = sorted({key for row in comparison for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def write_metric_summary(path: Path, judge_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judge_name in sorted({result["judge"] for result in judge_results}):
        relevant = [result for result in judge_results if result["judge"] == judge_name and isinstance(result.get("parsed"), dict)]
        for metric in METRICS:
            full_scores: list[float] = []
            weak_scores: list[float] = []
            wins = {"full_system": 0, "weak_eds": 0, "tie": 0}
            for result in relevant:
                parsed = result["parsed"]
                mapping = result.get("ab_mapping") or {}
                metric_data = parsed.get(metric) if isinstance(parsed.get(metric), dict) else {}
                wins[winner_to_system(metric_data.get("winner"), mapping)] += 1
                full = score_for_system(parsed, metric, "full_system", mapping)
                weak = score_for_system(parsed, metric, "weak_eds", mapping)
                if full is not None:
                    full_scores.append(full)
                if weak is not None:
                    weak_scores.append(weak)
            rows.append(
                {
                    "judge": judge_name,
                    "metric": metric,
                    "n": len(relevant),
                    "full_system_mean": round(sum(full_scores) / len(full_scores), 4) if full_scores else None,
                    "weak_eds_mean": round(sum(weak_scores) / len(weak_scores), 4) if weak_scores else None,
                    "mean_diff_full_minus_weak_eds": round(
                        (sum(full_scores) / len(full_scores)) - (sum(weak_scores) / len(weak_scores)),
                        4,
                    )
                    if full_scores and weak_scores
                    else None,
                    "full_system_wins": wins["full_system"],
                    "weak_eds_wins": wins["weak_eds"],
                    "ties": wins["tie"],
                }
            )
    write_csv(path, rows)
    return rows


def write_majority_vote_summary(path: Path, judge_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for result in judge_results:
        by_sample.setdefault(result["sample_id"], []).append(result)
    rows: list[dict[str, Any]] = []
    for sample_id, results in sorted(by_sample.items()):
        counts = {"full_system": 0, "weak_eds": 0, "tie": 0}
        for result in results:
            parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
            counts[winner_to_system(parsed.get("overall_preference"), result.get("ab_mapping") or {})] += 1
        majority = max(counts, key=lambda key: counts[key])
        if list(counts.values()).count(counts[majority]) > 1:
            majority = "tie"
        rows.append({"sample_id": sample_id, **counts, "majority_overall_winner": majority})
    write_csv(path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weak_root = resolve_repo_path(args.weak_eds_root)
    full_root = resolve_repo_path(args.full_system_root)
    manifest = resolve_repo_path(args.manifest)
    sample_list = resolve_repo_path(args.sample_list) if args.sample_list else None
    if sample_list is not None and not sample_list.exists():
        print(f"{PROGRESS_PREFIX} sample_list not found, falling back to manifest selection: {sample_list}", flush=True)
        sample_list = None
    paths = weak_judge_paths(weak_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.prompt_md.write_text(JUDGE_PROMPT, encoding="utf-8")
    rows = load_manifest(manifest, limit=args.limit, selection_rank_start=args.selection_rank_start, sample_list=sample_list)
    configs = [] if args.prepare_only else weak_eds_judge_configs(args)
    print(f"{PROGRESS_PREFIX} selected_samples={len(rows)} max_frames={args.max_frames} judges={[c.name for c in configs]}", flush=True)

    judge_inputs: list[dict[str, Any]] = []
    judge_results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        print(f"{PROGRESS_PREFIX} [{index}/{len(rows)}] prepare sample_id={sample_id}", flush=True)
        judge_input = prepare_judge_input(
            row,
            full_root=full_root,
            paths=paths,
            max_frames=args.max_frames,
            ab_seed=args.ab_seed,
        )
        judge_inputs.append(judge_input)
        if not args.prepare_only:
            results = run_judges_for_sample(
                sample_id,
                judge_input,
                configs,
                paths=paths,
                rerun_judge=args.rerun_judge,
                timeout_seconds=args.judge_timeout_seconds,
                max_tokens=args.judge_max_tokens,
                json_mode=args.json_mode,
                retries=args.judge_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            judge_results.extend(results)
            status_text = ", ".join(f"{r['judge']}={r.get('status')}" for r in results)
            print(f"{PROGRESS_PREFIX} [{index}/{len(rows)}] judged sample_id={sample_id} {status_text}", flush=True)

    write_pair_manifest(paths.pair_manifest_jsonl, judge_inputs)
    comparison = write_judge_comparison(paths.comparison_csv, rows, judge_results)
    metric_summary = write_metric_summary(paths.metric_summary_csv, judge_results)
    majority_rows = write_majority_vote_summary(paths.majority_vote_csv, judge_results)
    summary = {
        "experiment": "weak_eds_video_judge_ablation",
        "definition": "A/B MLLM judging for Full System vs Weak EDS videos.",
        "sample_count": len(rows),
        "max_frames": args.max_frames,
        "ab_seed": args.ab_seed,
        "json_mode": args.json_mode,
        "judge_timeout_seconds": args.judge_timeout_seconds,
        "judge_retries": args.judge_retries,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "judge_names": [config.name for config in configs],
        "prepare_only": args.prepare_only,
        "judge_result_count": len(judge_results),
        "judge_passed": sum(1 for result in judge_results if result.get("status") == "passed"),
        "judge_reused_existing_results": sum(1 for result in judge_results if result.get("reused_existing_result")),
        "comparison_count": len(comparison),
        "metric_summary_count": len(metric_summary),
        "majority_vote_count": len(majority_rows),
        "artifacts": {
            "pair_manifest": str(paths.pair_manifest_jsonl),
            "comparison_csv": str(paths.comparison_csv),
            "metric_summary_by_judge": str(paths.metric_summary_csv),
            "majority_vote_summary": str(paths.majority_vote_csv),
            "summary_json": str(paths.summary_json),
            "prompt_md": str(paths.prompt_md),
        },
    }
    write_json(paths.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
