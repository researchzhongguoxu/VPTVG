"""Run GSM8K No TeachingPlan video ablation and MLLM A/B judging.

The experiment reuses existing Full System artifacts, removes only the
TeachingPlan stage, then keeps ScriptDirector -> TTS -> Renderer unchanged.
Existing per-sample video and judge results are reused by default.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from dotenv import dotenv_values, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathexplain.agents.explainer import Explainer
from mathexplain.agents.renderer import Renderer
from mathexplain.agents.script_director import ScriptDirector
from mathexplain.agents.tts import TTSProvider
from mathexplain.schemas import (
    validate_classification_result_dict,
    validate_executable_director_script_dict,
    validate_explanation_script_dict,
    validate_tts_result_dict,
    validate_verification_report_dict,
)
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.explanation import ExplanationScript
from mathexplain.schemas.scs import SolutionChain, validate_solution_chain_dict
from mathexplain.schemas.spr import SPR, validate_spr_dict
from mathexplain.schemas.verification import VerificationReport


DEFAULT_MANIFEST = REPO_ROOT / "data/experiments/gsm8k/manifest.jsonl"
DEFAULT_FULL_ROOT = REPO_ROOT / "data/experiments/gsm8k/video_runs/full_vs_simple_pilot_200/full_system"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/gsm8k/video_ablation_runs/no_teaching_plan"
PROMPT_VERSION = "mllm-video-judge-no-teaching-plan-v1"
NO_TEACHING_PLAN_VERSION = "no_teaching_plan_video_v1"
JUDGE_FIELD_DEFAULTS = {
    "JUDGE_GLM_MODEL": "glm-4.6v",
    "JUDGE_GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
    "JUDGE_GLM_API_KEY_ENV": "ZAI_API_KEY",
    "JUDGE_QWEN_MODEL": "qwen3.6-plus",
    "JUDGE_QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "JUDGE_QWEN_API_KEY_ENV": "DASHSCOPE_API_KEY",
    "JUDGE_KIMI_MODEL": "kimi-k2.6",
    "JUDGE_KIMI_BASE_URL": "https://api.moonshot.cn/v1",
    "JUDGE_KIMI_API_KEY_ENV": "KIMI_API_KEY",
    "JUDGE_DOUBAO_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    "JUDGE_DOUBAO_API_KEY_ENV": "ARK_API_KEY",
}

METRICS = [
    "formula_source_clarity",
    "step_completeness",
    "narration_clarity",
    "self_learning_suitability",
]

JUDGE_PROMPT = """You are evaluating two math explanation videos for the same problem.

You will receive the problem image, the gold final answer, optional reference notes, and two anonymized videos represented by key frames plus narration transcripts.

Judge Video A and Video B blindly. Do not infer system names. Focus on teaching quality, not visual beauty.

Evaluate exactly these four dimensions:
1. formula_source_clarity: whether formulas are clearly tied to problem facts or previous verified steps.
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
class ExperimentPaths:
    output_dir: Path
    no_teaching_plan_dir: Path
    judge_inputs_dir: Path
    judge_results_dir: Path
    frames_dir: Path
    prompt_md: Path
    results_jsonl: Path
    pair_manifest_jsonl: Path
    comparison_csv: Path
    metric_summary_csv: Path
    majority_vote_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    api_key: str
    temperature: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run No TeachingPlan GSM8K video ablation with MLLM judges.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-system-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection-rank-start", type=int, default=0)
    parser.add_argument("--sample-list", type=Path, default=None)
    parser.add_argument("--output-language", default="auto")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ab-seed", type=int, default=20260711)
    parser.add_argument("--tts-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true", help="Generate videos and judge inputs, but skip MLLM calls.")
    parser.add_argument("--rerun-video", action="store_true", help="Regenerate No TeachingPlan video artifacts.")
    parser.add_argument("--rerun-judge", action="store_true", help="Regenerate existing judge results.")
    parser.add_argument("--no-tts", action="store_false", dest="synthesize_tts", default=True)
    parser.add_argument("--no-render-video", action="store_false", dest="render_video", default=True)
    parser.add_argument("--judge-glm", action="store_true", dest="judge_glm", default=True)
    parser.add_argument("--no-judge-glm", action="store_false", dest="judge_glm")
    parser.add_argument("--judge-qwen", action="store_true", dest="judge_qwen", default=True)
    parser.add_argument("--no-judge-qwen", action="store_false", dest="judge_qwen")
    parser.add_argument("--judge-doubao", action="store_true", dest="judge_doubao", default=False)
    parser.add_argument("--no-judge-doubao", action="store_false", dest="judge_doubao")
    parser.add_argument("--judge-kimi", action="store_true", dest="judge_kimi", default=False)
    parser.add_argument("--no-judge-kimi", action="store_false", dest="judge_kimi")
    return parser.parse_args()


def experiment_paths(output_dir: Path) -> ExperimentPaths:
    output_dir = output_dir.resolve()
    return ExperimentPaths(
        output_dir=output_dir,
        no_teaching_plan_dir=output_dir / "no_teaching_plan",
        judge_inputs_dir=output_dir / "judge_inputs",
        judge_results_dir=output_dir / "judge_results",
        frames_dir=output_dir / "frames",
        prompt_md=output_dir / "prompt.md",
        results_jsonl=output_dir / "no_teaching_plan_results.jsonl",
        pair_manifest_jsonl=output_dir / "pair_manifest.jsonl",
        comparison_csv=output_dir / "comparison.csv",
        metric_summary_csv=output_dir / "metric_summary_by_judge.csv",
        majority_vote_csv=output_dir / "majority_vote_summary.csv",
        summary_json=output_dir / "summary.json",
    )


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("{"):
            ids.append(str(json.loads(raw)["sample_id"]))
        else:
            ids.append(raw.split(",", 1)[0].strip())
    return ids


def load_manifest(
    manifest: Path,
    *,
    limit: int,
    selection_rank_start: int,
    sample_list: Path | None = None,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda item: int(item.get("selection_rank", 0)))
    sample_ids = load_sample_ids(sample_list)
    if sample_ids is not None:
        by_id = {str(row["sample_id"]): row for row in rows}
        return [by_id[sample_id] for sample_id in sample_ids if sample_id in by_id][:limit]
    rows = [row for row in rows if int(row.get("selection_rank", 0)) >= selection_rank_start]
    video_rows = [row for row in rows if row.get("include_in_video_eval")]
    if len(video_rows) >= limit:
        return video_rows[:limit]
    return rows[:limit]


def find_pipeline_summary(sample_dir: Path) -> Path | None:
    candidates = list(sample_dir.glob("summary.json")) + list(sample_dir.glob("*/summary.json"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_full_artifacts(row: dict[str, Any], full_root: Path) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = full_root / sample_id
    result_path = sample_dir / "result.json"
    result = read_json(result_path) if result_path.exists() else {}
    summary_path = Path(str(result.get("summary_path") or "")) if result.get("summary_path") else None
    if summary_path is None or not summary_path.exists():
        summary_path = find_pipeline_summary(sample_dir)
    if summary_path is None:
        raise FileNotFoundError(f"Full System summary.json not found for {sample_id}.")
    summary = read_json(summary_path)
    artifact_paths = summary.get("artifact_paths") if isinstance(summary.get("artifact_paths"), dict) else {}
    video_path = Path(str(result.get("video_path") or artifact_paths.get("video") or ""))
    if not video_path.exists():
        raise FileNotFoundError(f"Full System video.mp4 not found for {sample_id}.")
    eds_tts = artifact_paths.get("eds_tts") or str(summary_path.parent / "eds_tts.json")
    eds = artifact_paths.get("eds") or str(summary_path.parent / "eds.json")
    return {
        "sample_dir": sample_dir,
        "result_path": result_path,
        "result": result,
        "summary_path": summary_path,
        "summary": summary,
        "artifact_dir": summary_path.parent,
        "video_path": video_path,
        "eds_path": Path(eds_tts if Path(eds_tts).exists() else eds),
    }


def build_no_teaching_plan_explanation(
    *,
    spr: SPR,
    classification: ClassificationResult,
    scs: SolutionChain,
    verification: VerificationReport,
    output_language: str,
    language_metadata: dict[str, Any] | None,
) -> ExplanationScript:
    explainer = Explainer()
    if verification.valid is not True:
        return explainer._unsupported_script(spr, scs, verification, "standard")  # noqa: SLF001

    presentation = explainer._presentation_context(spr, scs, output_language=output_language)  # noqa: SLF001
    segments = explainer._problem_segments(spr, scs, output_language=output_language)  # noqa: SLF001
    if explainer._is_english(output_language):  # noqa: SLF001
        segments.extend(explainer._english_variable_segments(scs, presentation))  # noqa: SLF001
    else:
        segments.extend(explainer._variable_segments(scs, presentation))  # noqa: SLF001
    segments.extend(_direct_scs_step_segments(explainer, scs, presentation, output_language))
    if explainer._is_english(output_language):  # noqa: SLF001
        segments.extend(explainer._english_answer_segments(scs, presentation))  # noqa: SLF001
    else:
        segments.extend(explainer._answer_segments(scs, presentation))  # noqa: SLF001
    segments, symbol_consistency = explainer._enforce_symbol_consistency(segments)  # noqa: SLF001
    anchors = explainer._visual_anchors(scs, presentation, segments)  # noqa: SLF001
    report = explainer._consistency_report(segments, anchors, spr, scs, presentation)  # noqa: SLF001
    return ExplanationScript.model_validate(
        {
            "problem_id": scs.problem_id,
            "source_chain_id": scs.chain_id,
            "status": "passed" if report["valid"] else "failed",
            "audience_level": "standard",
            "segments": segments,
            "visual_anchors": anchors,
            "consistency_report": report,
            "metadata": {
                "explainer": "Explainer",
                "explainer_version": "No TeachingPlan direct SCS ablation",
                "ablation": "no_teaching_plan",
                "teaching_plan_removed": True,
                "source_verification_status": verification.status,
                "source_verification_mode": verification.verification_mode,
                "solver_route": classification.solver_route,
                "task_type": classification.task_type,
                "detected_source_language": (language_metadata or {}).get("detected_source_language"),
                "output_language": output_language,
                "language_policy": (language_metadata or {}).get("language_policy"),
                "language": language_metadata or {"output_language": output_language},
                "does_not_generate_eds": True,
                "presentation": {
                    "symbols": presentation["symbols"],
                    "selected_answer_targets": [item.get("target") for item in presentation["answers"]],
                },
                "symbol_consistency": symbol_consistency,
            },
        }
    )


def _direct_scs_step_segments(
    explainer: Explainer,
    scs: SolutionChain,
    presentation: dict[str, Any],
    output_language: str,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for step in scs.steps:
        role = step.diagnostics.get("math_role")
        if role == "modeling" and step.diagnostics.get("explainer_ready") is True:
            expression = explainer._display_expression(step.math_expression, presentation)  # noqa: SLF001
            narration = explainer._modeling_narration(step, expression, output_language=output_language)  # noqa: SLF001
            segments.append(
                explainer._segment(  # noqa: SLF001
                    f"seg_direct_model_{step.step_id}",
                    "modeling",
                    [step.step_id],
                    narration,
                    [expression] if expression else [],
                    [f"expr_{step.step_id}"] if expression else [],
                    ["RENDER_EXPRESSION", "HIGHLIGHT_TERMS"],
                    "normal",
                    explainer._duration_for_text(narration, 4600),  # noqa: SLF001
                    250,
                    {"focus_area": "main_formula", "keep_previous_context": True},
                    {
                        "source": "direct_scs_no_teaching_plan",
                        "original_source": step.diagnostics.get("source"),
                        "evidence_span": step.diagnostics.get("evidence_span"),
                    },
                    tts_hints={"emphasis_terms": [expression] if expression else []},
                )
            )
        elif role == "cas_result" and step.diagnostics.get("explainer_ready") is True:
            expression = explainer._cas_step_display_expression(step, presentation) or explainer._display_expression(  # noqa: SLF001
                step.math_expression,
                presentation,
            )
            operation = step.diagnostics.get("operation") or "calculate"
            narration = explainer._cas_narration(operation, expression, step, presentation, output_language=output_language)  # noqa: SLF001
            segments.append(
                explainer._segment(  # noqa: SLF001
                    f"seg_direct_calc_{step.step_id}",
                    "calculation",
                    [step.step_id],
                    narration,
                    [expression] if expression else [],
                    [f"expr_{step.step_id}"] if expression else [],
                    ["SHOW_RESULT", "HIGHLIGHT_RESULT"],
                    "normal",
                    explainer._duration_for_text(narration, 3600),  # noqa: SLF001
                    200,
                    {"focus_area": "main_formula", "keep_previous_context": True},
                    {"source": "direct_scs_no_teaching_plan", "operation": operation},
                    tts_hints={"emphasis_terms": [expression] if expression else []},
                )
            )
    return segments


def run_no_teaching_plan_video(
    row: dict[str, Any],
    *,
    full_root: Path,
    paths: ExperimentPaths,
    output_language: str,
    rerun_video: bool,
    synthesize_tts: bool,
    render_video: bool,
    tts_timeout_seconds: float,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = paths.no_teaching_plan_dir / sample_id
    result_path = sample_dir / "result.json"
    if result_path.exists() and not rerun_video:
        result = read_json(result_path)
        video_path = result.get("video_path")
        if not render_video or (video_path and Path(video_path).exists()):
            result["reused_existing_result"] = True
            result["reuse_source"] = "result_json"
            return result

    started = time.time()
    full = find_full_artifacts(row, full_root)
    full_artifact_dir = Path(full["artifact_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    spr = validate_spr_dict(read_json(full_artifact_dir / "spr.json"))
    classification = validate_classification_result_dict(read_json(full_artifact_dir / "classification.json"))
    scs = validate_solution_chain_dict(read_json(full_artifact_dir / "scs.json"))
    verification = validate_verification_report_dict(read_json(full_artifact_dir / "verification.json"))
    language_metadata = full["summary"].get("language") if isinstance(full["summary"].get("language"), dict) else {}
    resolved_language = (
        str(full["summary"].get("output_language") or language_metadata.get("output_language") or "zh-CN")
        if output_language == "auto"
        else output_language
    )

    explanation = build_no_teaching_plan_explanation(
        spr=spr,
        classification=classification,
        scs=scs,
        verification=verification,
        output_language=resolved_language,
        language_metadata=language_metadata,
    )
    validate_explanation_script_dict(explanation.model_dump(mode="json"))
    explanation_path = sample_dir / "explanation.json"
    explanation_path.write_text(explanation.model_dump_json(indent=2), encoding="utf-8")

    eds = ScriptDirector().direct(
        explanation,
        spr=spr,
        scs=scs,
        verification_report=verification,
        output_language=resolved_language,
    )
    validate_executable_director_script_dict(eds.model_dump(mode="json"))
    eds_path = sample_dir / "eds.json"
    eds_path.write_text(eds.model_dump_json(indent=2), encoding="utf-8")

    tts_result = None
    eds_for_render = eds
    artifact_paths: dict[str, str] = {
        "explanation": str(explanation_path),
        "eds": str(eds_path),
    }
    if synthesize_tts:
        tts_result = TTSProvider(timeout_seconds=tts_timeout_seconds).synthesize(eds, output_dir=sample_dir)
        validate_tts_result_dict(tts_result.model_dump(mode="json"))
        if tts_result.tts_report_path:
            artifact_paths["tts_report"] = tts_result.tts_report_path
        if tts_result.audio_track_path:
            artifact_paths["tts_audio_track"] = tts_result.audio_track_path
        if tts_result.retimed_eds_path:
            artifact_paths["eds_tts"] = tts_result.retimed_eds_path
            eds_for_render = validate_executable_director_script_dict(read_json(Path(tts_result.retimed_eds_path)))

    render_result = None
    if render_video:
        render_result = Renderer().render(
            eds_for_render,
            output_dir=sample_dir,
            output_name="video.mp4",
            audio_path=tts_result.audio_track_path if tts_result and tts_result.status == "passed" else None,
        )
        if render_result.video_path:
            artifact_paths["video"] = render_result.video_path
        if render_result.render_report_path:
            artifact_paths["render_report"] = render_result.render_report_path

    status = "passed"
    failed_stage = None
    failure_reason = None
    if explanation.status != "passed" or not explanation.consistency_report.valid:
        status, failed_stage, failure_reason = "failed", "explainer", "No TeachingPlan explanation did not pass."
    elif not eds.quality_report.valid:
        status, failed_stage, failure_reason = "failed", "script_director", "EDS quality gate did not pass."
    elif synthesize_tts and (tts_result is None or tts_result.status != "passed"):
        status, failed_stage, failure_reason = "failed", "tts", "TTS did not pass."
    elif render_video and (render_result is None or render_result.status != "passed"):
        status, failed_stage, failure_reason = "failed", "renderer", "Renderer did not pass."

    result = {
        "sample_id": sample_id,
        "selection_rank": int(row.get("selection_rank", -1)),
        "pipeline": "no_teaching_plan",
        "ablation_version": NO_TEACHING_PLAN_VERSION,
        "image_path": row.get("image_path"),
        "gold_final_answer": row.get("gold_final_answer"),
        "prediction": scs.final_answer,
        "status": status,
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "full_system_summary_path": str(full["summary_path"]),
        "full_system_video_path": str(full["video_path"]),
        "video_generated": bool(artifact_paths.get("video") and Path(artifact_paths["video"]).exists()),
        "video_path": artifact_paths.get("video"),
        "artifact_dir": str(sample_dir),
        "summary_path": str(sample_dir / "summary.json"),
        "explanation_status": explanation.status,
        "explanation_segment_count": len(explanation.segments),
        "eds_valid": bool(eds.quality_report.valid),
        "eds_scene_count": len(eds.scenes),
        "tts_status": tts_result.status if tts_result else "not_run",
        "renderer_status": render_result.status if render_result else "not_run",
        "duration_ms": int((time.time() - started) * 1000),
        "reused_existing_result": False,
        "reuse_source": None,
        "artifact_paths": {**artifact_paths, "summary": str(sample_dir / "summary.json")},
    }
    write_json(sample_dir / "summary.json", result)
    write_json(result_path, result)
    return result


def frame_times_from_eds(eds: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    scenes = eds.get("scenes") if isinstance(eds.get("scenes"), list) else []
    formulas_by_scene: dict[str, list[dict[str, Any]]] = {}
    for formula in eds.get("formula_tracks") or []:
        if isinstance(formula, dict):
            formulas_by_scene.setdefault(str(formula.get("scene_id")), []).append(formula)
    picks: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{index}")
        start_ms = int(scene.get("start_ms") or 0)
        duration_ms = int(scene.get("duration_ms") or 0)
        formulas = formulas_by_scene.get(scene_id) or []
        if formulas:
            formula = max(formulas, key=lambda item: int(item.get("start_ms") or 0) + int(item.get("duration_ms") or 0))
            time_ms = int((formula.get("start_ms") or start_ms) + 0.75 * int(formula.get("duration_ms") or duration_ms))
            reason = "formula_complete"
        else:
            time_ms = start_ms + max(1000, int(duration_ms * 0.55))
            reason = "scene_mid"
        if duration_ms:
            time_ms = min(time_ms, start_ms + max(200, duration_ms - 500))
        picks.append({"scene_index": index, "scene_id": scene_id, "time_ms": max(0, time_ms), "reason": reason})
    if len(picks) <= max_frames:
        return picks
    keep = {0, len(picks) - 1}
    slots = max_frames - len(keep)
    middle = list(range(1, len(picks) - 1))
    for offset in range(slots):
        keep.add(middle[round(offset * (len(middle) - 1) / max(1, slots - 1))])
    return [picks[index] for index in sorted(keep)]


def extract_frames(video_path: Path, eds_path: Path, output_dir: Path, max_frames: int) -> list[dict[str, Any]]:
    eds = read_json(eds_path)
    picks = frame_times_from_eds(eds, max_frames=max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = int((frame_count / fps) * 1000) if fps else 0
    frames: list[dict[str, Any]] = []
    for index, pick in enumerate(picks, start=1):
        time_ms = min(int(pick["time_ms"]), max(0, duration_ms - 100))
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = output_dir / f"frame_{index:02d}_scene_{pick['scene_index']:02d}_{time_ms}ms.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frames.append({**pick, "time_ms": time_ms, "path": str(frame_path)})
    cap.release()
    return frames


def eds_transcript(eds_path: Path) -> dict[str, Any]:
    eds = read_json(eds_path)
    narration = [
        {
            "scene_id": item.get("scene_id"),
            "text": item.get("text"),
            "start_ms": item.get("start_ms"),
            "duration_ms": item.get("duration_ms"),
        }
        for item in eds.get("narration_tracks", [])
        if isinstance(item, dict)
    ]
    formulas = [
        {
            "scene_id": item.get("scene_id"),
            "expression": item.get("expression"),
            "start_ms": item.get("start_ms"),
            "metadata": item.get("metadata") or {},
        }
        for item in eds.get("formula_tracks", [])
        if isinstance(item, dict)
    ]
    final_answer = next((item["expression"] for item in reversed(formulas) if item.get("expression")), None)
    return {"narration": narration, "formulas": formulas, "final_answer": final_answer}


def ab_mapping(sample_id: str, seed: int) -> dict[str, str]:
    rng = random.Random(f"{seed}:{sample_id}")
    systems = ["full_system", "no_teaching_plan"]
    rng.shuffle(systems)
    return {"A": systems[0], "B": systems[1]}


def prepare_judge_input(
    row: dict[str, Any],
    *,
    full_root: Path,
    no_tp_result: dict[str, Any],
    paths: ExperimentPaths,
    max_frames: int,
    ab_seed: int,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    full = find_full_artifacts(row, full_root)
    no_tp_video = Path(str(no_tp_result.get("video_path") or ""))
    if not no_tp_video.exists():
        raise FileNotFoundError(f"No TeachingPlan video missing for {sample_id}.")
    no_tp_eds = Path(str(no_tp_result.get("artifact_paths", {}).get("eds_tts") or no_tp_result.get("artifact_paths", {}).get("eds")))
    if not no_tp_eds.exists():
        raise FileNotFoundError(f"No TeachingPlan EDS missing for {sample_id}.")

    full_frames = extract_frames(
        Path(full["video_path"]),
        Path(full["eds_path"]),
        paths.frames_dir / sample_id / "full_system",
        max_frames=max_frames,
    )
    no_tp_frames = extract_frames(
        no_tp_video,
        no_tp_eds,
        paths.frames_dir / sample_id / "no_teaching_plan",
        max_frames=max_frames,
    )
    mapping = ab_mapping(sample_id, ab_seed)
    system_payloads = {
        "full_system": {
            "video_path": str(full["video_path"]),
            "frames": full_frames,
            **eds_transcript(Path(full["eds_path"])),
        },
        "no_teaching_plan": {
            "video_path": str(no_tp_video),
            "frames": no_tp_frames,
            **eds_transcript(no_tp_eds),
        },
    }
    judge_input = {
        "sample_id": sample_id,
        "selection_rank": int(row.get("selection_rank", -1)),
        "prompt_version": PROMPT_VERSION,
        "problem_image": str(resolve_repo_path(row["image_path"])),
        "problem_text": row.get("problem_text"),
        "gold_final_answer": row.get("gold_final_answer"),
        "ab_mapping": mapping,
        "videos": {
            "A": system_payloads[mapping["A"]],
            "B": system_payloads[mapping["B"]],
        },
    }
    path = paths.judge_inputs_dir / sample_id / "judge_input.json"
    write_json(path, judge_input)
    return judge_input


def env_values_first_non_placeholder(env_file: Path = REPO_ROOT / ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value or value.startswith("your_"):
            continue
        values.setdefault(key, value)
    return values


def env_get(name: str, defaults: dict[str, str]) -> str | None:
    value = os.getenv(name)
    if value and not value.startswith("your_"):
        return value
    return defaults.get(name) or JUDGE_FIELD_DEFAULTS.get(name)


def judge_configs(args: argparse.Namespace) -> list[JudgeConfig]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    defaults = env_values_first_non_placeholder()
    provider_specs = {
        "glm": ("JUDGE_GLM_MODEL", "JUDGE_GLM_BASE_URL", "JUDGE_GLM_API_KEY_ENV"),
        "qwen": ("JUDGE_QWEN_MODEL", "JUDGE_QWEN_BASE_URL", "JUDGE_QWEN_API_KEY_ENV"),
        "doubao": ("JUDGE_DOUBAO_MODEL", "JUDGE_DOUBAO_BASE_URL", "JUDGE_DOUBAO_API_KEY_ENV"),
        "kimi": ("JUDGE_KIMI_MODEL", "JUDGE_KIMI_BASE_URL", "JUDGE_KIMI_API_KEY_ENV"),
    }
    enabled = {
        "glm": args.judge_glm,
        "qwen": args.judge_qwen,
        "doubao": args.judge_doubao,
        "kimi": args.judge_kimi,
    }
    configs: list[JudgeConfig] = []
    for name, is_enabled in enabled.items():
        if not is_enabled:
            continue
        model_key, base_key, key_env_key = provider_specs[name]
        model = env_get(model_key, defaults) or ""
        base_url = (env_get(base_key, defaults) or "").rstrip("/")
        api_key_env = env_get(key_env_key, defaults) or ""
        api_key = env_get(api_key_env, defaults) if api_key_env else None
        if not model or not base_url or not api_key_env or not api_key:
            raise ValueError(f"Judge {name} is enabled but model/base_url/api key config is incomplete.")
        temperature = 1.0 if name == "kimi" else 0.0
        configs.append(
            JudgeConfig(
                name=name,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                api_key=api_key,
                temperature=temperature,
            )
        )
    return configs


def image_content(path: Path) -> dict[str, Any]:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def compact_video_text(label: str, payload: dict[str, Any]) -> str:
    transcript = "\n".join(
        f"- {item.get('text')}" for item in payload.get("narration", []) if item.get("text")
    )
    formulas = "\n".join(
        f"- {item.get('expression')}" for item in payload.get("formulas", []) if item.get("expression")
    )
    return (
        f"\nVideo {label} transcript:\n{transcript}\n"
        f"\nVideo {label} formula list:\n{formulas}\n"
        f"\nVideo {label} final answer shown: {payload.get('final_answer')}\n"
    )


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


def call_openai_compatible_judge(
    config: JudgeConfig,
    judge_input: dict[str, Any],
    *,
    timeout_seconds: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": judge_messages(judge_input),
        "temperature": config.temperature,
        "max_tokens": max_tokens,
    }
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
        }


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def run_judges_for_sample(
    sample_id: str,
    judge_input: dict[str, Any],
    configs: list[JudgeConfig],
    *,
    paths: ExperimentPaths,
    rerun_judge: bool,
    timeout_seconds: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for config in configs:
        path = paths.judge_results_dir / config.name / f"{sample_id}.json"
        if path.exists() and not rerun_judge:
            result = read_json(path)
            result["reused_existing_result"] = True
            results.append(result)
            continue
        result = call_openai_compatible_judge(
            config,
            judge_input,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
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


def winner_to_system(winner: Any, mapping: dict[str, str]) -> str:
    value = str(winner or "").strip()
    if value in {"A", "B"}:
        return mapping[value]
    return "tie"


def score_for_system(parsed: dict[str, Any], metric: str, system: str, mapping: dict[str, str]) -> float | None:
    metric_data = parsed.get(metric)
    if not isinstance(metric_data, dict):
        return None
    label = next((side for side, mapped in mapping.items() if mapped == system), None)
    if label is None:
        return None
    value = metric_data.get(f"score_{label}")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_results_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


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


def write_comparison(path: Path, rows: list[dict[str, Any]], no_tp_results: list[dict[str, Any]], judge_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample = {row["sample_id"]: row for row in rows}
    no_tp_by_sample = {result["sample_id"]: result for result in no_tp_results}
    judge_by_sample: dict[str, list[dict[str, Any]]] = {}
    for result in judge_results:
        judge_by_sample.setdefault(result["sample_id"], []).append(result)
    comparison: list[dict[str, Any]] = []
    for sample_id, row in by_sample.items():
        no_tp = no_tp_by_sample.get(sample_id, {})
        output = {
            "sample_id": sample_id,
            "selection_rank": row.get("selection_rank"),
            "gold_final_answer": row.get("gold_final_answer"),
            "no_teaching_plan_status": no_tp.get("status"),
            "no_teaching_plan_video_generated": no_tp.get("video_generated"),
            "no_teaching_plan_video_path": no_tp.get("video_path"),
            "full_system_video_path": no_tp.get("full_system_video_path"),
        }
        for judge in judge_by_sample.get(sample_id, []):
            parsed = judge.get("parsed") if isinstance(judge.get("parsed"), dict) else {}
            mapping = judge.get("ab_mapping") if isinstance(judge.get("ab_mapping"), dict) else {}
            output[f"{judge['judge']}_status"] = judge.get("status")
            output[f"{judge['judge']}_overall_winner"] = winner_to_system(parsed.get("overall_preference"), mapping)
            output[f"{judge['judge']}_total_tokens"] = (judge.get("usage") or {}).get("total_tokens")
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
            no_tp_scores: list[float] = []
            wins = {"full_system": 0, "no_teaching_plan": 0, "tie": 0}
            for result in relevant:
                parsed = result["parsed"]
                mapping = result.get("ab_mapping") or {}
                metric_data = parsed.get(metric) if isinstance(parsed.get(metric), dict) else {}
                wins[winner_to_system(metric_data.get("winner"), mapping)] += 1
                full = score_for_system(parsed, metric, "full_system", mapping)
                no_tp = score_for_system(parsed, metric, "no_teaching_plan", mapping)
                if full is not None:
                    full_scores.append(full)
                if no_tp is not None:
                    no_tp_scores.append(no_tp)
            rows.append(
                {
                    "judge": judge_name,
                    "metric": metric,
                    "n": len(relevant),
                    "full_system_mean": round(sum(full_scores) / len(full_scores), 4) if full_scores else None,
                    "no_teaching_plan_mean": round(sum(no_tp_scores) / len(no_tp_scores), 4) if no_tp_scores else None,
                    "mean_diff_full_minus_no_tp": round(
                        (sum(full_scores) / len(full_scores)) - (sum(no_tp_scores) / len(no_tp_scores)),
                        4,
                    )
                    if full_scores and no_tp_scores
                    else None,
                    "full_system_wins": wins["full_system"],
                    "no_teaching_plan_wins": wins["no_teaching_plan"],
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
        counts = {"full_system": 0, "no_teaching_plan": 0, "tie": 0}
        for result in results:
            parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
            counts[winner_to_system(parsed.get("overall_preference"), result.get("ab_mapping") or {})] += 1
        majority = max(counts, key=lambda key: counts[key])
        if list(counts.values()).count(counts[majority]) > 1:
            majority = "tie"
        rows.append({"sample_id": sample_id, **counts, "majority_overall_winner": majority})
    write_csv(path, rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def summarize(
    *,
    rows: list[dict[str, Any]],
    no_tp_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
    configs: list[JudgeConfig],
    prepare_only: bool,
) -> dict[str, Any]:
    video_total = len(no_tp_results)
    judge_total = len(judge_results)
    return {
        "experiment": "no_teaching_plan_video_ablation",
        "definition": "Full System video vs Full System with TeachingPlan removed; Solver/Verifier artifacts are reused.",
        "sample_count": len(rows),
        "no_teaching_plan_result_count": video_total,
        "no_teaching_plan_passed": sum(1 for result in no_tp_results if result.get("status") == "passed"),
        "no_teaching_plan_video_generated": sum(1 for result in no_tp_results if result.get("video_generated")),
        "no_teaching_plan_reused_existing_results": sum(1 for result in no_tp_results if result.get("reused_existing_result")),
        "judge_names": [config.name for config in configs],
        "prepare_only": prepare_only,
        "judge_result_count": judge_total,
        "judge_passed": sum(1 for result in judge_results if result.get("status") == "passed"),
        "judge_reused_existing_results": sum(1 for result in judge_results if result.get("reused_existing_result")),
    }


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
    configs = [] if args.prepare_only else judge_configs(args)
    no_tp_results: list[dict[str, Any]] = []
    judge_inputs: list[dict[str, Any]] = []
    judge_results: list[dict[str, Any]] = []
    for row in rows:
        no_tp = run_no_teaching_plan_video(
            row,
            full_root=resolve_repo_path(args.full_system_root),
            paths=paths,
            output_language=args.output_language,
            rerun_video=args.rerun_video,
            synthesize_tts=args.synthesize_tts,
            render_video=args.render_video,
            tts_timeout_seconds=args.tts_timeout_seconds,
        )
        no_tp_results.append(no_tp)
        try:
            judge_input = prepare_judge_input(
                row,
                full_root=resolve_repo_path(args.full_system_root),
                no_tp_result=no_tp,
                paths=paths,
                max_frames=args.max_frames,
                ab_seed=args.ab_seed,
            )
            judge_inputs.append(judge_input)
        except Exception as exc:
            print(f"[judge-input] {row['sample_id']} skipped: {exc}")
            continue
        if not args.prepare_only:
            judge_results.extend(
                run_judges_for_sample(
                    str(row["sample_id"]),
                    judge_input,
                    configs,
                    paths=paths,
                    rerun_judge=args.rerun_judge,
                    timeout_seconds=args.judge_timeout_seconds,
                    max_tokens=args.judge_max_tokens,
                )
            )

    write_results_jsonl(paths.results_jsonl, no_tp_results)
    write_pair_manifest(paths.pair_manifest_jsonl, judge_inputs)
    comparison_rows = write_comparison(paths.comparison_csv, rows, no_tp_results, judge_results)
    metric_summary = write_metric_summary(paths.metric_summary_csv, judge_results)
    majority_rows = write_majority_vote_summary(paths.majority_vote_csv, judge_results)
    summary = {
        **summarize(rows=rows, no_tp_results=no_tp_results, judge_results=judge_results, configs=configs, prepare_only=args.prepare_only),
        "limit": args.limit,
        "selection_rank_start": args.selection_rank_start,
        "max_frames": args.max_frames,
        "ab_seed": args.ab_seed,
        "artifacts": {
            "no_teaching_plan_results": str(paths.results_jsonl),
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
