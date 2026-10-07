"""Generate GSM8K Weak EDS video ablation artifacts.

The experiment reuses existing Full System artifacts, keeps the same narration
text and synthesized audio, but downgrades the visual director script to a plain
slide-style EDS. It does not call TTS, LLMs, or modify Full System outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathexplain.agents.renderer import Renderer
from mathexplain.schemas import validate_executable_director_script_dict

from scripts.run_gsm8k_no_teaching_plan_video_ablation import (
    DEFAULT_FULL_ROOT,
    DEFAULT_MANIFEST,
    find_full_artifacts,
    load_manifest,
    read_json,
    resolve_repo_path,
    write_json,
    write_results_jsonl,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/video_ablation_runs/weak_eds"
DEFAULT_SAMPLE_LIST = REPO_ROOT / "data/experiments/gsm8k/video_ablation_runs/no_teaching_plan/pair_manifest.jsonl"
DEFAULT_LIMIT = 50
WEAK_EDS_VERSION = "weak_eds_video_v1"
PROGRESS_PREFIX = "[weak-eds-video]"


@dataclass(frozen=True)
class WeakEDSPaths:
    output_dir: Path
    weak_eds_dir: Path
    results_jsonl: Path
    comparison_csv: Path
    summary_json: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Weak EDS GSM8K video ablation artifacts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-system-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-samples",
        "--count",
        "--limit",
        dest="limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of questions/videos to generate. Default: {DEFAULT_LIMIT}.",
    )
    parser.add_argument("--selection-rank-start", type=int, default=0)
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=DEFAULT_SAMPLE_LIST,
        help="Sample id list. Defaults to the No TeachingPlan 50-sample pair manifest.",
    )
    parser.add_argument("--rerun-video", action="store_true", help="Regenerate existing Weak EDS videos.")
    parser.add_argument("--no-render-video", action="store_false", dest="render_video", default=True)
    return parser.parse_args(argv)


def weak_eds_paths(output_dir: Path) -> WeakEDSPaths:
    output_dir = output_dir.resolve()
    return WeakEDSPaths(
        output_dir=output_dir,
        weak_eds_dir=output_dir / "weak_eds",
        results_jsonl=output_dir / "weak_eds_results.jsonl",
        comparison_csv=output_dir / "comparison.csv",
        summary_json=output_dir / "summary.json",
    )


def _path_from_summary(full: dict[str, Any], *keys: str) -> Path | None:
    artifact_paths = full["summary"].get("artifact_paths")
    if isinstance(artifact_paths, dict):
        for key in keys:
            value = artifact_paths.get(key)
            if value and Path(str(value)).exists():
                return Path(str(value))
    for key in keys:
        section = full["summary"].get("tts") if key == "tts_audio_track" else None
        if isinstance(section, dict):
            value = section.get("audio_track_path")
            if value and Path(str(value)).exists():
                return Path(str(value))
    return None


def _scene_narration_texts(eds_data: dict[str, Any]) -> dict[str, str]:
    by_scene: dict[str, list[str]] = {}
    for item in eds_data.get("narration_tracks") or []:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id") or "")
        text = str(item.get("text") or "").strip()
        if scene_id and text:
            by_scene.setdefault(scene_id, []).append(text)
    return {scene_id: "\n\n".join(texts) for scene_id, texts in by_scene.items()}


def build_weak_eds(
    full_eds_data: dict[str, Any],
    *,
    row: dict[str, Any],
    source_eds_path: Path,
    source_audio_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a plain visual EDS while preserving Full narration timings."""

    scene_texts = _scene_narration_texts(full_eds_data)
    scenes: list[dict[str, Any]] = []
    for index, scene in enumerate(full_eds_data.get("scenes") or []):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{index}")
        display_text = scene_texts.get(scene_id) or str(scene.get("title") or "")
        metadata = dict(scene.get("metadata") or {})
        metadata.update(
            {
                "source": "weak_eds_ablation",
                "source_full_scene_id": scene_id,
                "plain_display_text": display_text,
                "plain_formula_text": "",
                "weak_eds_version": WEAK_EDS_VERSION,
                "structured_formula_panel_removed": True,
                "key_formula_display_removed": True,
                "visual_highlighting_removed": True,
                "retiming_used_only_for_audio_alignment": True,
            }
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_type": scene.get("scene_type") or "modeling",
                "source_segment_ids": list(scene.get("source_segment_ids") or []),
                "title": scene.get("title"),
                "start_ms": int(scene.get("start_ms") or 0),
                "duration_ms": int(scene.get("duration_ms") or 0),
                "layout": {"style": "plain_slide", "source_layout_removed": True},
                "metadata": metadata,
            }
        )

    narration_tracks: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for item in full_eds_data.get("narration_tracks") or []:
        if not isinstance(item, dict):
            continue
        narration_id = str(item.get("narration_id") or f"narration_{len(narration_tracks)}")
        scene_id = str(item.get("scene_id") or (scenes[0]["scene_id"] if scenes else "scene_0"))
        narration = {
            "narration_id": narration_id,
            "scene_id": scene_id,
            "text": str(item.get("text") or ""),
            "start_ms": int(item.get("start_ms") or 0),
            "duration_ms": int(item.get("duration_ms") or 0),
            "pause_after_ms": int(item.get("pause_after_ms") or 0),
            "tts_hints": dict(item.get("tts_hints") or {}),
            "sync_anchor_ids": [],
        }
        narration_tracks.append(narration)
        timeline.append(
            {
                "timeline_id": f"weak_eds_timeline_{narration_id}",
                "scene_id": scene_id,
                "track_type": "narration",
                "target_id": narration_id,
                "start_ms": narration["start_ms"],
                "duration_ms": narration["duration_ms"],
                "action": "play",
                "metadata": {"source": "weak_eds_ablation"},
            }
        )

    render_hints = dict(full_eds_data.get("render_hints") or {})
    render_metadata = dict(render_hints.get("metadata") or {})
    render_metadata.update(
        {
            "source": "weak_eds_ablation",
            "weak_eds": True,
            "uses_structured_formula_panels": False,
            "uses_key_formula_display": False,
            "uses_visual_highlights": False,
            "uses_formula_cards": False,
            "reuses_full_narration_audio": True,
        }
    )
    render_hints.update(
        {
            "theme": "plain_simple_baseline",
            "formula_renderer": "plain_text",
            "metadata": render_metadata,
        }
    )

    question_text = (
        row.get("question_text")
        or row.get("problem_text")
        or next(
            (
                ((asset.get("metadata") or {}).get("problem_text") or (asset.get("metadata") or {}).get("label"))
                for asset in full_eds_data.get("assets") or []
                if isinstance(asset, dict) and asset.get("asset_type") == "problem_text"
            ),
            "",
        )
    )
    assets = [
        {
            "asset_id": "weak_eds_problem_text",
            "asset_type": "problem_text",
            "uri": None,
            "metadata": {
                "problem_text": question_text,
                "source_image": row.get("image_path"),
                "source": "weak_eds_ablation",
            },
        }
    ]

    total_duration_ms = max(
        [scene["start_ms"] + scene["duration_ms"] for scene in scenes]
        + [
            item["start_ms"] + item["duration_ms"] + item["pause_after_ms"]
            for item in narration_tracks
        ],
        default=0,
    )
    source_formula_count = len(full_eds_data.get("formula_tracks") or [])
    source_action_count = len(full_eds_data.get("visual_actions") or [])
    source_anchor_count = len(full_eds_data.get("sync_anchors") or [])
    warnings: list[str] = []
    if not scenes:
        warnings.append("Source EDS had no scenes.")
    if not narration_tracks:
        warnings.append("Source EDS had no narration tracks.")

    weak_eds = {
        "schema_version": full_eds_data.get("schema_version") or "EDS-1.0",
        "problem_id": full_eds_data.get("problem_id") or str(row.get("sample_id")),
        "source_explanation_id": f"weak_eds_from_{full_eds_data.get('source_explanation_id') or row.get('sample_id')}",
        "source_chain_id": full_eds_data.get("source_chain_id") or "unknown",
        "scenes": scenes,
        "timeline": timeline,
        "assets": assets,
        "narration_tracks": narration_tracks,
        "formula_tracks": [],
        "visual_actions": [],
        "sync_anchors": [],
        "render_hints": render_hints,
        "quality_report": {
            "valid": bool(scenes and narration_tracks),
            "status": "passed" if scenes and narration_tracks else "failed",
            "checked_scene_count": len(scenes),
            "errors": [] if scenes and narration_tracks else ["Weak EDS requires scenes and narration tracks."],
            "warnings": warnings,
            "metadata": {
                "source": "weak_eds_ablation",
                "source_formula_count_removed": source_formula_count,
                "source_visual_action_count_removed": source_action_count,
                "source_sync_anchor_count_removed": source_anchor_count,
            },
        },
        "metadata": {
            "source": "weak_eds_ablation",
            "ablation": "weak_eds",
            "ablation_version": WEAK_EDS_VERSION,
            "source_full_eds_path": str(source_eds_path),
            "source_full_audio_path": str(source_audio_path),
            "reuses_full_narration_text": True,
            "reuses_full_tts_audio": True,
            "removes_structured_formula_panels": True,
            "removes_key_formula_display": True,
            "removes_step_visual_highlighting": True,
            "formula_track_count_removed": source_formula_count,
            "visual_action_count_removed": source_action_count,
            "sync_anchor_count_removed": source_anchor_count,
            "total_duration_ms": total_duration_ms,
        },
    }
    report = {
        "status": "passed" if scenes and narration_tracks else "failed",
        "weak_eds_version": WEAK_EDS_VERSION,
        "source_eds_path": str(source_eds_path),
        "source_audio_path": str(source_audio_path),
        "scene_count": len(scenes),
        "narration_count": len(narration_tracks),
        "formula_count_removed": source_formula_count,
        "visual_action_count_removed": source_action_count,
        "sync_anchor_count_removed": source_anchor_count,
        "total_duration_ms": total_duration_ms,
        "audio_reused": True,
    }
    return weak_eds, report


def run_weak_eds_video(
    row: dict[str, Any],
    *,
    full_root: Path,
    paths: WeakEDSPaths,
    rerun_video: bool,
    render_video: bool,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = paths.weak_eds_dir / sample_id
    result_path = sample_dir / "result.json"
    if result_path.exists() and not rerun_video:
        result = read_json(result_path)
        video_path = result.get("weak_eds_video_path") or result.get("video_path")
        if not render_video or (video_path and Path(str(video_path)).exists()):
            result["reused_existing_result"] = True
            result["reuse_source"] = "result_json"
            return result

    started = time.time()
    sample_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    status = "passed"
    failed_stage = None
    failure_reason = None
    render_result = None

    try:
        full = find_full_artifacts(row, full_root)
        source_eds_path = Path(full["eds_path"])
        source_audio_path = _path_from_summary(full, "tts_audio_track")
        if source_audio_path is None:
            raise FileNotFoundError(f"Full System TTS audio not found for {sample_id}.")
        full_eds_data = read_json(source_eds_path)
        weak_eds_data, weak_report = build_weak_eds(
            full_eds_data,
            row=row,
            source_eds_path=source_eds_path,
            source_audio_path=source_audio_path,
        )
        weak_eds = validate_executable_director_script_dict(weak_eds_data)
        weak_eds_path = sample_dir / "weak_eds.json"
        weak_report_path = sample_dir / "weak_eds_report.json"
        weak_eds_path.write_text(weak_eds.model_dump_json(indent=2), encoding="utf-8")
        write_json(weak_report_path, weak_report)
        artifact_paths.update(
            {
                "weak_eds": str(weak_eds_path),
                "weak_eds_report": str(weak_report_path),
                "source_full_eds": str(source_eds_path),
                "source_full_audio": str(source_audio_path),
            }
        )
        if not weak_eds.quality_report.valid:
            status, failed_stage, failure_reason = "failed", "weak_eds", "Weak EDS quality gate did not pass."
        elif render_video:
            render_result = Renderer().render(
                weak_eds,
                output_dir=sample_dir,
                output_name="video.mp4",
                audio_path=source_audio_path,
            )
            if render_result.video_path:
                artifact_paths["video"] = render_result.video_path
            if render_result.render_report_path:
                artifact_paths["render_report"] = render_result.render_report_path
            if render_result.status != "passed":
                status, failed_stage, failure_reason = "failed", "renderer", "Renderer did not pass."
    except Exception as exc:
        full = locals().get("full", {})
        source_audio_path = locals().get("source_audio_path")
        source_eds_path = locals().get("source_eds_path")
        status, failed_stage, failure_reason = "failed", failed_stage or "weak_eds", str(exc)

    result = {
        "sample_id": sample_id,
        "selection_rank": int(row.get("selection_rank", -1)),
        "pipeline": "weak_eds",
        "ablation_version": WEAK_EDS_VERSION,
        "image_path": row.get("image_path"),
        "gold_final_answer": row.get("gold_final_answer"),
        "status": status,
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "full_system_summary_path": str(full.get("summary_path")) if isinstance(full, dict) and full.get("summary_path") else None,
        "full_system_video_path": str(full.get("video_path")) if isinstance(full, dict) and full.get("video_path") else None,
        "source_full_eds_path": str(source_eds_path) if source_eds_path else None,
        "source_full_audio_path": str(source_audio_path) if source_audio_path else None,
        "audio_reused": bool(source_audio_path and Path(str(source_audio_path)).exists()),
        "weak_eds_path": artifact_paths.get("weak_eds"),
        "weak_eds_video_path": artifact_paths.get("video"),
        "video_path": artifact_paths.get("video"),
        "video_generated": bool(artifact_paths.get("video") and Path(artifact_paths["video"]).exists()),
        "artifact_dir": str(sample_dir),
        "summary_path": str(sample_dir / "summary.json"),
        "render_video": render_video,
        "renderer_status": render_result.status if render_result else "not_run",
        "renderer_duration_ms": render_result.duration_ms if render_result else None,
        "duration_ms": int((time.time() - started) * 1000),
        "reused_existing_result": False,
        "reuse_source": None,
        "artifact_paths": {**artifact_paths, "summary": str(sample_dir / "summary.json")},
    }
    write_json(sample_dir / "summary.json", result)
    write_json(result_path, result)
    return result


def write_comparison(path: Path, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_by_sample = {str(row["sample_id"]): row for row in rows}
    comparison: list[dict[str, Any]] = []
    for result in results:
        row = row_by_sample.get(str(result["sample_id"]), {})
        comparison.append(
            {
                "sample_id": result.get("sample_id"),
                "selection_rank": result.get("selection_rank"),
                "gold_final_answer": row.get("gold_final_answer"),
                "full_video_path": result.get("full_system_video_path"),
                "weak_eds_video_path": result.get("weak_eds_video_path"),
                "status": result.get("status"),
                "failed_stage": result.get("failed_stage"),
                "failure_reason": result.get("failure_reason"),
                "audio_reused": result.get("audio_reused"),
                "weak_eds_path": result.get("weak_eds_path"),
                "artifact_dir": result.get("artifact_dir"),
            }
        )
    fieldnames = [
        "sample_id",
        "selection_rank",
        "gold_final_answer",
        "full_video_path",
        "weak_eds_video_path",
        "status",
        "failed_stage",
        "failure_reason",
        "audio_reused",
        "weak_eds_path",
        "artifact_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def summarize(
    *,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    limit: int,
    selection_rank_start: int,
    sample_list: Path | None,
    render_video: bool,
    paths: WeakEDSPaths,
) -> dict[str, Any]:
    return {
        "experiment": "weak_eds_video_ablation",
        "definition": (
            "Full System video with the same narration text and reused TTS audio, "
            "but with structured formula panels, key-formula display, step-level "
            "visual highlighting, and fine EDS visual structuring removed."
        ),
        "sample_count": len(rows),
        "result_count": len(results),
        "passed": sum(1 for result in results if result.get("status") == "passed"),
        "failed": sum(1 for result in results if result.get("status") != "passed"),
        "video_generated": sum(1 for result in results if result.get("video_generated")),
        "audio_reused": sum(1 for result in results if result.get("audio_reused")),
        "reused_existing_results": sum(1 for result in results if result.get("reused_existing_result")),
        "limit": limit,
        "selection_rank_start": selection_rank_start,
        "sample_list": str(sample_list) if sample_list else None,
        "render_video": render_video,
        "ablation_version": WEAK_EDS_VERSION,
        "artifacts": {
            "weak_eds_dir": str(paths.weak_eds_dir),
            "weak_eds_results": str(paths.results_jsonl),
            "comparison_csv": str(paths.comparison_csv),
            "summary_json": str(paths.summary_json),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_repo_path(args.output_dir)
    full_system_root = resolve_repo_path(args.full_system_root)
    manifest_path = resolve_repo_path(args.manifest)
    sample_list = resolve_repo_path(args.sample_list) if args.sample_list else None
    if sample_list is not None and not sample_list.exists():
        print(f"{PROGRESS_PREFIX} sample_list not found, falling back to manifest selection: {sample_list}", flush=True)
        sample_list = None

    paths = weak_eds_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(
        manifest_path,
        limit=args.limit,
        selection_rank_start=args.selection_rank_start,
        sample_list=sample_list,
    )
    print(f"{PROGRESS_PREFIX} selected_samples={len(rows)} requested_limit={args.limit}", flush=True)
    print(f"{PROGRESS_PREFIX} manifest={manifest_path}", flush=True)
    print(f"{PROGRESS_PREFIX} sample_list={sample_list}", flush=True)
    print(f"{PROGRESS_PREFIX} full_system_root={full_system_root}", flush=True)
    print(f"{PROGRESS_PREFIX} output_dir={paths.output_dir}", flush=True)
    print(f"{PROGRESS_PREFIX} weak_eds_dir={paths.weak_eds_dir}", flush=True)

    results: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        sample_dir = paths.weak_eds_dir / sample_id
        print(
            f"{PROGRESS_PREFIX} [{index}/{total}] start sample_id={sample_id} "
            f"selection_rank={row.get('selection_rank')} artifact_dir={sample_dir}",
            flush=True,
        )
        result = run_weak_eds_video(
            row,
            full_root=full_system_root,
            paths=paths,
            rerun_video=args.rerun_video,
            render_video=args.render_video,
        )
        results.append(result)
        action = "reused" if result.get("reused_existing_result") else "generated"
        print(
            f"{PROGRESS_PREFIX} [{index}/{total}] {action} sample_id={sample_id} "
            f"status={result.get('status')} audio_reused={result.get('audio_reused')} "
            f"video={result.get('weak_eds_video_path')} summary={result.get('summary_path')}",
            flush=True,
        )

    write_results_jsonl(paths.results_jsonl, results)
    comparison = write_comparison(paths.comparison_csv, rows, results)
    summary = summarize(
        rows=rows,
        results=results,
        limit=args.limit,
        selection_rank_start=args.selection_rank_start,
        sample_list=sample_list,
        render_video=args.render_video,
        paths=paths,
    )
    summary["comparison_count"] = len(comparison)
    write_json(paths.summary_json, summary)
    print(f"{PROGRESS_PREFIX} wrote weak_eds_results={paths.results_jsonl}", flush=True)
    print(f"{PROGRESS_PREFIX} wrote comparison_csv={paths.comparison_csv}", flush=True)
    print(f"{PROGRESS_PREFIX} wrote summary_json={paths.summary_json}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
