"""Generate GSM8K No TeachingPlan videos without running MLLM judges."""

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
    experiment_paths,
    load_manifest,
    resolve_repo_path,
    run_no_teaching_plan_video,
    summarize,
    write_json,
    write_results_jsonl,
)

DEFAULT_NUM_SAMPLES = 50
DEFAULT_SAVE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PROGRESS_PREFIX = "[no-tp-video]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate No TeachingPlan GSM8K videos.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-system-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SAVE_OUTPUT_DIR)
    parser.add_argument(
        "--num-samples",
        "--count",
        "--limit",
        dest="limit",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of questions/videos to generate. Default: {DEFAULT_NUM_SAMPLES}.",
    )
    parser.add_argument("--selection-rank-start", type=int, default=0)
    parser.add_argument("--sample-list", type=Path, default=None)
    parser.add_argument("--output-language", default="auto")
    parser.add_argument("--rerun-video", action="store_true", help="Regenerate existing No TeachingPlan videos.")
    parser.add_argument("--no-tts", action="store_false", dest="synthesize_tts", default=True)
    parser.add_argument("--no-render-video", action="store_false", dest="render_video", default=True)
    parser.add_argument("--tts-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    full_system_root = resolve_repo_path(args.full_system_root)
    manifest_path = resolve_repo_path(args.manifest)
    paths = experiment_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(
        manifest_path,
        limit=args.limit,
        selection_rank_start=args.selection_rank_start,
        sample_list=args.sample_list,
    )
    print(f"{PROGRESS_PREFIX} selected_samples={len(rows)} requested_limit={args.limit}", flush=True)
    print(f"{PROGRESS_PREFIX} manifest={manifest_path}", flush=True)
    print(f"{PROGRESS_PREFIX} full_system_root={full_system_root}", flush=True)
    print(f"{PROGRESS_PREFIX} output_dir={paths.output_dir}", flush=True)
    print(f"{PROGRESS_PREFIX} no_teaching_plan_dir={paths.no_teaching_plan_dir}", flush=True)
    results = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        sample_dir = paths.no_teaching_plan_dir / sample_id
        print(
            f"{PROGRESS_PREFIX} [{index}/{total}] start sample_id={sample_id} "
            f"selection_rank={row.get('selection_rank')} artifact_dir={sample_dir}",
            flush=True,
        )
        try:
            result = run_no_teaching_plan_video(
                row,
                full_root=full_system_root,
                paths=paths,
                output_language=args.output_language,
                rerun_video=args.rerun_video,
                synthesize_tts=args.synthesize_tts,
                render_video=args.render_video,
                tts_timeout_seconds=args.tts_timeout_seconds,
            )
        except Exception as exc:
            print(f"{PROGRESS_PREFIX} [{index}/{total}] failed sample_id={sample_id} error={exc}", flush=True)
            raise
        results.append(result)
        action = "reused" if result.get("reused_existing_result") else "generated"
        print(
            f"{PROGRESS_PREFIX} [{index}/{total}] {action} sample_id={sample_id} "
            f"status={result.get('status')} scenes={result.get('eds_scene_count')} "
            f"video={result.get('video_path')} summary={result.get('summary_path')}",
            flush=True,
        )
    write_results_jsonl(paths.results_jsonl, results)
    summary = {
        **summarize(rows=rows, no_tp_results=results, judge_results=[], configs=[], prepare_only=True),
        "stage": "video_generation",
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "artifacts": {
            "no_teaching_plan_results": str(paths.results_jsonl),
            "summary_json": str(paths.summary_json),
            "no_teaching_plan_dir": str(paths.no_teaching_plan_dir),
        },
    }
    write_json(paths.summary_json, summary)
    print(f"{PROGRESS_PREFIX} wrote no_teaching_plan_results={paths.results_jsonl}", flush=True)
    print(f"{PROGRESS_PREFIX} wrote summary_json={paths.summary_json}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
