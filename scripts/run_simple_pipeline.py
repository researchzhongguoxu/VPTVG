"""Run Simple Pipeline baseline on one problem image.

Simple Pipeline is a strong non-verifiable video baseline:
image -> direct MLLM script -> weak EDS adapter -> shared TTS/Renderer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathexplain.agents.renderer import Renderer
from mathexplain.agents.simple_pipeline import (
    DirectExplanationGenerator,
    PlainSimpleEDSAdapter,
    SimpleEDSAdapter,
    SimpleScriptNormalizer,
)
from mathexplain.agents.simple_pipeline.plain_adapter import PLAIN_SIMPLE_VISUAL_VERSION
from mathexplain.agents.tts import TTSProvider
from mathexplain.agents.vision_parser.spr_generator import SPRGenerator
from mathexplain.schemas.eds import validate_executable_director_script_dict
from mathexplain.schemas.simple_video import validate_simple_video_script_dict
from mathexplain.schemas.tts import TTSResult, validate_tts_result_dict
from mathexplain.services.llm import DEFAULT_DASHSCOPE_BASE_URL, DashScopeQwenVisionClient
from mathexplain.services.usage import summarize_usage, usage_call


REPO_ROOT = Path(__file__).resolve().parents[1]

# PyCharm-friendly defaults. Edit these values directly, then click Run.
DEFAULT_IMAGE_PATH = "data/experiments/gsm8k/images/gsm8k_test_000060.png"
DEFAULT_OUTPUT_DIR = "data/intermediate/simple_pipeline"
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_RENDER_VIDEO = True
DEFAULT_SYNTHESIZE_TTS = False
DEFAULT_VISUAL_STYLE = "shared"  # shared or plain


@dataclass
class MockImageJSONClient:
    """Small mock client used for local dry runs and tests."""

    response: str
    provider: str = "mock"
    model: str = "mock-direct-explainer"
    last_usage: dict[str, Any] | None = None

    def generate_json_from_image(self, image_data_uri: str, system_prompt: str, user_prompt: str) -> str:
        del image_data_uri, system_prompt, user_prompt
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return self.response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Simple Pipeline video baseline on one image.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Input problem image.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model for direct baseline generation.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible DashScope base URL.")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable containing API key.")
    parser.add_argument(
        "--output-language",
        choices=["auto", "en-US", "zh-CN"],
        default="auto",
        help="Language for narration and rendering.",
    )
    parser.add_argument("--mock-response", default=None, help="Path to a JSON/text response to use instead of real API.")
    parser.add_argument(
        "--visual-style",
        choices=["shared", "plain"],
        default=DEFAULT_VISUAL_STYLE,
        help="shared uses the MathExplainAgent visual backend; plain uses a simple slide baseline.",
    )
    parser.add_argument("--render-video", action="store_true", dest="render_video", default=DEFAULT_RENDER_VIDEO)
    parser.add_argument("--no-render-video", action="store_false", dest="render_video")
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
    parser.add_argument("--tts-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_root = Path(args.output_dir)
    run_dir = output_root / image_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    timings: list[dict[str, Any]] = []
    started_total = time.perf_counter()

    def stage(name: str, label: str):
        return _StageTimer(name, label, timings, enabled=not args.quiet_progress)

    raw_response_path = run_dir / "raw_llm_response.json"
    simple_script_path = run_dir / "simple_script.json"
    simple_eds_path = run_dir / ("plain_simple_eds.json" if args.visual_style == "plain" else "simple_eds.json")
    summary_path = run_dir / "summary.json"

    raw_response = ""
    usage_call_data: dict[str, Any] | None = None
    simple_script = None
    eds = None
    tts_result = None
    render_result = None
    effective_eds = None

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        with stage("direct_generation", "Direct MLLM explanation"):
            image_data_uri = SPRGenerator._image_to_data_uri(str(image_path))
            if args.mock_response:
                mock_text = Path(args.mock_response).read_text(encoding="utf-8")
                client = MockImageJSONClient(mock_text)
            else:
                client = DashScopeQwenVisionClient(
                    model=args.model,
                    base_url=args.base_url or DEFAULT_DASHSCOPE_BASE_URL,
                    api_key_env=args.api_key_env,
                    timeout_seconds=120,
                )
            generation = DirectExplanationGenerator(client).generate(
                image_data_uri,
                output_language=args.output_language,
            )
            raw_response = generation.raw_response
            usage_call_data = usage_call(
                stage="simple_direct_generation",
                provider=generation.provider,
                model=generation.model,
                usage=generation.usage,
            )
            raw_response_path.write_text(
                json.dumps(
                    {
                        "provider": generation.provider,
                        "model": generation.model,
                        "raw_response": generation.raw_response,
                        "usage": generation.usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifact_paths["raw_llm_response"] = str(raw_response_path)

        with stage("normalizer", "Simple script normalization"):
            simple_script = SimpleScriptNormalizer().normalize(
                raw_response,
                problem_id=image_path.stem,
                source_image=str(image_path),
                output_language=args.output_language,
            )
            validate_simple_video_script_dict(simple_script.model_dump(mode="json"))
            simple_script_path.write_text(simple_script.model_dump_json(indent=2), encoding="utf-8")
            artifact_paths["simple_script"] = str(simple_script_path)
            output_language = _resolved_language(args.output_language, simple_script.problem_text)

        with stage("eds_adapter", "Weak EDS conversion"):
            adapter = PlainSimpleEDSAdapter() if args.visual_style == "plain" else SimpleEDSAdapter()
            eds = adapter.to_eds(simple_script, output_language=output_language)
            validate_executable_director_script_dict(eds.model_dump(mode="json"))
            simple_eds_path.write_text(eds.model_dump_json(indent=2), encoding="utf-8")
            artifact_paths["plain_simple_eds" if args.visual_style == "plain" else "simple_eds"] = str(simple_eds_path)
            effective_eds = eds

        if args.synthesize_tts:
            with stage("tts", "TTS synthesis"):
                try:
                    tts_result = TTSProvider(timeout_seconds=args.tts_timeout_seconds).synthesize(eds, output_dir=run_dir)
                    validate_tts_result_dict(tts_result.model_dump(mode="json"))
                    if tts_result.tts_report_path:
                        artifact_paths["tts_report"] = tts_result.tts_report_path
                    if tts_result.audio_track_path:
                        artifact_paths["tts_audio_track"] = tts_result.audio_track_path
                    if tts_result.retimed_eds_path:
                        artifact_paths["eds_tts"] = tts_result.retimed_eds_path
                        effective_eds = validate_executable_director_script_dict(
                            json.loads(Path(tts_result.retimed_eds_path).read_text(encoding="utf-8"))
                        )
                except Exception as exc:
                    tts_report_path = run_dir / "tts_report.json"
                    tts_result = TTSResult(
                        status="failed",
                        provider="volcengine",
                        resource_id="seed-tts-2.0",
                        voice_type="unknown_voice",
                        clip_count=0,
                        duration_ms=0,
                        tts_report_path=str(tts_report_path),
                        errors=[str(exc)],
                        metadata={"pipeline_fallback": "silent_video"},
                    )
                    tts_report_path.write_text(
                        json.dumps(tts_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    artifact_paths["tts_report"] = str(tts_report_path)

        if args.render_video:
            with stage("renderer", "Video rendering"):
                render_result = Renderer().render(
                    effective_eds,
                    output_dir=run_dir,
                    output_name="video.mp4",
                    audio_path=tts_result.audio_track_path if tts_result and tts_result.status == "passed" else None,
                )
                if render_result.video_path:
                    artifact_paths["video"] = render_result.video_path
                if render_result.render_report_path:
                    artifact_paths["render_report"] = render_result.render_report_path

        summary = _summary(
            status="passed",
            image_path=image_path,
            run_dir=run_dir,
            model=args.model,
            visual_style=args.visual_style,
            output_language=output_language,
            simple_script=simple_script,
            eds=eds,
            tts_result=tts_result,
            render_result=render_result,
            usage_call_data=usage_call_data,
            timings=timings,
            started_total=started_total,
            artifact_paths=artifact_paths,
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nArtifacts saved to: {run_dir.resolve()}")
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_stage": timings[-1]["stage"] if timings else None,
            "failure_reason": str(exc),
            "image_path": str(image_path),
            "artifact_dir": str(run_dir),
            "artifact_paths": {**artifact_paths, "summary": str(summary_path)},
            "timing": _timing_report(timings, started_total),
        }
        summary_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        print(f"\nArtifacts saved to: {run_dir.resolve()}")
        return 1


class _StageTimer:
    def __init__(self, stage: str, label: str, timings: list[dict[str, Any]], *, enabled: bool) -> None:
        self.stage = stage
        self.label = label
        self.timings = timings
        self.enabled = enabled
        self.started = 0.0
        self.started_at = ""

    def __enter__(self):
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        if self.enabled:
            print(f"[Simple Pipeline] {self.label} started...")
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.perf_counter() - self.started) * 1000)
        status = "failed" if exc_type else "passed"
        self.timings.append(
            {
                "stage": self.stage,
                "label": self.label,
                "status": status,
                "started_at": self.started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms,
            }
        )
        if self.enabled:
            print(f"[Simple Pipeline] {self.label} {status} in {duration_ms / 1000:.2f}s")
        return False


def _resolved_language(requested: str, text: str) -> str:
    if requested != "auto":
        return requested
    zh_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return "zh-CN" if zh_count > 8 else "en-US"


def _summary(
    *,
    status: str,
    image_path: Path,
    run_dir: Path,
    model: str,
    visual_style: str,
    output_language: str,
    simple_script: Any,
    eds: Any,
    tts_result: Any,
    render_result: Any,
    usage_call_data: dict[str, Any] | None,
    timings: list[dict[str, Any]],
    started_total: float,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    usage = usage_call_data or {}
    llm_usage = summarize_usage([usage]) if usage else summarize_usage([])
    return {
        "status": status,
        "pipeline": "plain_simple_pipeline" if visual_style == "plain" else "simple_pipeline",
        "visual_style": visual_style,
        "visual_version": PLAIN_SIMPLE_VISUAL_VERSION if visual_style == "plain" else "shared_visual_v1",
        "image_path": str(image_path),
        "problem_id": image_path.stem,
        "artifact_dir": str(run_dir),
        "model": model,
        "output_language": output_language,
        "final_answer": getattr(simple_script, "final_answer", None),
        "pipeline_success": status == "passed",
        "video_generated": bool(render_result and render_result.status == "passed" and render_result.video_path),
        "answer_text_present": bool(getattr(simple_script, "final_answer", None)),
        "scene_count": len(getattr(eds, "scenes", []) or []) if eds is not None else 0,
        "duration_ms": getattr(render_result, "duration_ms", 0) if render_result is not None else 0,
        "llm_usage": llm_usage,
        "tts": tts_result.model_dump(mode="json") if tts_result is not None else None,
        "render": render_result.model_dump(mode="json") if render_result is not None else None,
        "artifact_paths": artifact_paths,
        "timing": _timing_report(timings, started_total),
        "baseline_controls": {
            "uses_cas": False,
            "uses_verifier": False,
            "uses_solver_repair": False,
            "uses_teaching_plan": False,
            "uses_shared_tts_renderer": True,
            "uses_key_facts_panel": visual_style != "plain",
            "uses_semantic_highlights": visual_style != "plain",
            "uses_formula_cards": visual_style != "plain",
            "plain_visual_version": PLAIN_SIMPLE_VISUAL_VERSION if visual_style == "plain" else None,
        },
    }


def _timing_report(timings: list[dict[str, Any]], started_total: float) -> dict[str, Any]:
    return {
        "total_duration_ms": int((time.perf_counter() - started_total) * 1000),
        "stages": timings,
    }


if __name__ == "__main__":
    raise SystemExit(main())
