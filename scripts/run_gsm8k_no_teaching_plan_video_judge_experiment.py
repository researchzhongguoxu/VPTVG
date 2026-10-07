"""Run MLLM A/B judge experiment for existing No TeachingPlan videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_no_teaching_plan_video_ablation import (
    DEFAULT_FULL_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    JUDGE_PROMPT,
    experiment_paths,
    judge_configs,
    load_manifest,
    prepare_judge_input,
    resolve_repo_path,
    run_judges_for_sample,
    summarize,
    write_comparison,
    write_json,
    write_majority_vote_summary,
    write_metric_summary,
    write_pair_manifest,
    read_json,
)


AVAILABLE_JUDGES = ("glm", "qwen", "doubao", "kimi")
DEFAULT_NUM_SAMPLES = 50
DEFAULT_JUDGE_PROVIDERS = "qwen,glm,kimi,doubao"
DEFAULT_JUDGE_COUNT = 3
DEFAULT_JUDGE_MAX_TOKENS = 4096


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Full vs No TeachingPlan GSM8K videos.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-system-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-samples",
        "--count",
        "--limit",
        dest="limit",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of question pairs to evaluate. Default: {DEFAULT_NUM_SAMPLES}.",
    )
    parser.add_argument("--selection-rank-start", type=int, default=0)
    parser.add_argument("--sample-list", type=Path, default=None)
    parser.add_argument(
        "--judge-providers",
        default=DEFAULT_JUDGE_PROVIDERS,
        help="Comma-separated judge priority order. --judge-count chooses from this order.",
    )
    parser.add_argument("--judge-count", type=int, default=DEFAULT_JUDGE_COUNT, help="Number of enabled judge models.")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ab-seed", type=int, default=20260711)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare judge inputs but do not call models.")
    parser.add_argument("--rerun-judge", action="store_true", help="Regenerate existing judge results.")
    parser.add_argument("--judge-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    return parser.parse_args(argv)


def selected_judges(provider_order: str, judge_count: int) -> list[str]:
    providers: list[str] = []
    for raw in provider_order.split(","):
        provider = raw.strip().lower()
        if not provider:
            continue
        if provider not in AVAILABLE_JUDGES:
            raise ValueError(f"Unknown judge provider: {provider}")
        if provider not in providers:
            providers.append(provider)
    if judge_count < 0:
        raise ValueError("--judge-count must be non-negative.")
    return providers[:judge_count]


def namespace_for_judges(judges: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        judge_glm="glm" in judges,
        judge_qwen="qwen" in judges,
        judge_doubao="doubao" in judges,
        judge_kimi="kimi" in judges,
    )


def print_judge_preflight(configs: list) -> None:
    if not configs:
        return
    print("[judge] enabled providers:", ",".join(config.name for config in configs), flush=True)
    for config in configs:
        print(
            f"[judge] provider={config.name} model={config.model} "
            f"base_url={config.base_url} temperature={config.temperature} "
            f"api_key_env={config.api_key_env} api_key=set",
            flush=True,
        )


def load_no_teaching_plan_result(paths, sample_id: str) -> dict:
    result_path = paths.no_teaching_plan_dir / sample_id / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"No TeachingPlan result is missing for {sample_id}. "
            "Run scripts/generate_gsm8k_no_teaching_plan_videos.py first."
        )
    return read_json(result_path)


def main() -> int:
    args = parse_args()
    paths = experiment_paths(args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.prompt_md.write_text(JUDGE_PROMPT, encoding="utf-8")

    rows = load_manifest(
        resolve_repo_path(args.manifest),
        limit=args.limit,
        selection_rank_start=args.selection_rank_start,
        sample_list=args.sample_list,
    )
    judges = selected_judges(args.judge_providers, args.judge_count)
    configs = [] if args.prepare_only else judge_configs(namespace_for_judges(judges))
    print(f"[judge] selected judge_names={judges}", flush=True)
    print_judge_preflight(configs)
    no_tp_results: list[dict] = []
    judge_inputs: list[dict] = []
    judge_results: list[dict] = []
    missing: list[dict] = []

    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            no_tp_result = load_no_teaching_plan_result(paths, sample_id)
        except FileNotFoundError as exc:
            print(f"[judge] {exc}")
            missing.append({"sample_id": sample_id, "error": str(exc)})
            continue
        no_tp_results.append(no_tp_result)
        judge_input = prepare_judge_input(
            row,
            full_root=resolve_repo_path(args.full_system_root),
            no_tp_result=no_tp_result,
            paths=paths,
            max_frames=args.max_frames,
            ab_seed=args.ab_seed,
        )
        judge_inputs.append(judge_input)
        if not args.prepare_only:
            judge_results.extend(
                run_judges_for_sample(
                    sample_id,
                    judge_input,
                    configs,
                    paths=paths,
                    rerun_judge=args.rerun_judge,
                    timeout_seconds=args.judge_timeout_seconds,
                    max_tokens=args.judge_max_tokens,
                )
            )

    write_pair_manifest(paths.pair_manifest_jsonl, judge_inputs)
    comparison_rows = write_comparison(paths.comparison_csv, rows, no_tp_results, judge_results)
    metric_summary = write_metric_summary(paths.metric_summary_csv, judge_results)
    majority_rows = write_majority_vote_summary(paths.majority_vote_csv, judge_results)
    summary = {
        **summarize(
            rows=rows,
            no_tp_results=no_tp_results,
            judge_results=judge_results,
            configs=configs,
            prepare_only=args.prepare_only,
        ),
        "stage": "judge_experiment",
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "judge_providers_requested": args.judge_providers,
        "judge_count": args.judge_count,
        "judge_names_selected": judges,
        "max_frames": args.max_frames,
        "ab_seed": args.ab_seed,
        "missing_video_results": missing,
        "artifacts": {
            "pair_manifest": str(paths.pair_manifest_jsonl),
            "comparison_csv": str(paths.comparison_csv),
            "metric_summary_by_judge": str(paths.metric_summary_csv),
            "majority_vote_summary": str(paths.majority_vote_csv),
            "summary_json": str(paths.summary_json),
            "prompt_md": str(paths.prompt_md),
        },
        "comparison_count": len(comparison_rows),
        "metric_summary_count": len(metric_summary),
        "majority_vote_count": len(majority_rows),
    }
    write_json(paths.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
