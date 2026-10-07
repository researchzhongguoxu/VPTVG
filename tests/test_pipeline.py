from pathlib import Path
from types import SimpleNamespace

from mathexplain.config import PipelineModelConfig
from scripts import run_pipeline
from scripts.run_pipeline import PipelineTimer, build_summary


class Dumpable:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, mode: str = "json") -> dict:
        return self.data


def test_pipeline_summary_includes_eds_artifact_and_counts() -> None:
    parser_output = SimpleNamespace(
        problem_id="problem_1",
        artifact_paths={"spr": "run/problem_1/spr.json"},
        spr=SimpleNamespace(
            problem_type="word_problem",
            problem_text="题干",
            variables=[Dumpable(symbol="x")],
            confidence=Dumpable(overall=0.9),
        ),
        emr=SimpleNamespace(
            source_problem_type="word_problem",
            representation_type="unknown",
            variables=[],
            equations=[],
            expressions=[],
            metadata={},
        ),
    )
    classification = Dumpable(task_type="equation_solving")
    solution_chain = SimpleNamespace(
        chain_id="chain_main",
        final_answer="x = 2",
        steps=[
            SimpleNamespace(rule_name="planner_tool_call", diagnostics={"planner_model": "fake"}),
            SimpleNamespace(rule_name="cas_solve", diagnostics={"success": True}),
        ],
        metadata={"backend_confirmed": True},
    )
    verification = SimpleNamespace(
        valid=True,
        status="passed",
        verification_mode="tool_call_backed",
        summary="ok",
        errors=[],
        warnings=[],
        checked_steps=[1],
    )
    explanation = SimpleNamespace(
        status="passed",
        audience_level="standard",
        segments=[1, 2],
        visual_anchors=[1],
        consistency_report=SimpleNamespace(valid=True, errors=[], warnings=[]),
    )
    teaching_plan = SimpleNamespace(
        planner_mode="rule_based",
        moves=[1, 2, 3],
        quality_report=SimpleNamespace(
            status="passed",
            valid=True,
            expanded_step_ids=["step_1", "step_2"],
            fallback_step_ids=["step_3"],
            strategy_names=["demo_strategy"],
        ),
    )
    eds = SimpleNamespace(
        scenes=[1, 2],
        timeline=[1, 2, 3],
        narration_tracks=[1, 2],
        formula_tracks=[1],
        visual_actions=[1, 2],
        sync_anchors=[1, 2],
        quality_report=SimpleNamespace(status="passed", valid=True),
        metadata={"total_duration_ms": 5000},
    )
    render_result = SimpleNamespace(
        status="passed",
        video_path="run/problem_1/video.mp4",
        duration_ms=5000,
        frame_count=150,
        fps=30,
        errors=[],
        warnings=["font fallback"],
    )
    tts_result = SimpleNamespace(
        status="passed",
        provider="fake",
        resource_id="fake-resource",
        voice_type="fake-voice",
        clip_count=2,
        duration_ms=5200,
        audio_track_path="run/problem_1/narration_track.wav",
        retimed_eds_path="run/problem_1/eds_tts.json",
        errors=[],
        warnings=[],
    )
    artifact_paths = {
        "spr": "run/problem_1/spr.json",
        "eds": "run/problem_1/eds.json",
        "eds_tts": "run/problem_1/eds_tts.json",
        "tts_report": "run/problem_1/tts_report.json",
        "tts_audio_track": "run/problem_1/narration_track.wav",
        "tts_audio_clips_dir": "run/problem_1/audio_clips",
        "video": "run/problem_1/video.mp4",
        "render_report": "run/problem_1/render_report.json",
        "timing_report": "run/problem_1/timing_report.json",
        "summary": "run/problem_1/summary.json",
    }
    timing = {
        "total_duration_ms": 1234,
        "slowest_stage": "vision_parser",
        "slowest_label": "视觉解析",
        "stages": [
            {
                "stage": "vision_parser",
                "label": "视觉解析",
                "status": "passed",
                "started_at": "2026-05-28T00:00:00+00:00",
                "ended_at": "2026-05-28T00:00:01+00:00",
                "duration_ms": 1000,
            }
        ],
    }

    summary = build_summary(
        image_path=Path("problem.png"),
        parser_output=parser_output,
        classification=classification,
        solution_chain=solution_chain,
        verification=verification,
        explanation=explanation,
        teaching_plan=teaching_plan,
        eds=eds,
        model_config=PipelineModelConfig.from_sources(env={}, load_project_env=False),
        artifact_paths=artifact_paths,
        tts_result=tts_result,
        render_result=render_result,
        timing=timing,
    )

    assert summary["artifact_paths"]["eds"] == "run/problem_1/eds.json"
    assert summary["eds"] == {
        "status": "passed",
        "valid": True,
        "scene_count": 2,
        "timeline_count": 3,
        "narration_count": 2,
        "formula_count": 1,
        "visual_action_count": 2,
        "sync_anchor_count": 2,
        "total_duration_ms": 5000,
    }
    assert summary["artifact_paths"]["video"] == "run/problem_1/video.mp4"
    assert summary["artifact_paths"]["render_report"] == "run/problem_1/render_report.json"
    assert summary["artifact_paths"]["tts_report"] == "run/problem_1/tts_report.json"
    assert summary["artifact_paths"]["tts_audio_track"] == "run/problem_1/narration_track.wav"
    assert summary["artifact_paths"]["eds_tts"] == "run/problem_1/eds_tts.json"
    assert summary["artifact_paths"]["timing_report"] == "run/problem_1/timing_report.json"
    assert summary["timing"] == timing
    assert summary["teaching_plan"] == {
        "status": "passed",
        "valid": True,
        "planner_mode": "rule_based",
        "move_count": 3,
        "expanded_step_count": 2,
        "fallback_step_count": 1,
        "strategy_names": ["demo_strategy"],
        "provider_used": None,
        "provider_attempted": None,
        "provider_fallback_reason": None,
        "quality_gate_status": None,
    }
    assert summary["renderer"] == {
        "status": "passed",
        "video_path": "run/problem_1/video.mp4",
        "duration_ms": 5000,
        "frame_count": 150,
        "fps": 30,
        "error_count": 0,
        "warning_count": 1,
    }
    assert summary["tts"] == {
        "status": "passed",
        "provider": "fake",
        "resource_id": "fake-resource",
        "voice_type": "fake-voice",
        "language": None,
        "voice_selection": None,
        "clip_count": 2,
        "duration_ms": 5200,
        "audio_track_path": "run/problem_1/narration_track.wav",
        "retimed_eds_path": "run/problem_1/eds_tts.json",
        "error_count": 0,
        "warning_count": 0,
    }
    assert summary["llm_usage"]["llm_call_count"] == 0
    assert summary["llm_usage"]["total_tokens"] == 0


def test_pipeline_summary_omits_renderer_when_not_requested() -> None:
    parser_output = SimpleNamespace(
        problem_id="problem_1",
        artifact_paths={"spr": "run/problem_1/spr.json"},
        spr=SimpleNamespace(
            problem_type="word_problem",
            problem_text="题干",
            variables=[],
            confidence=Dumpable(overall=0.9),
        ),
        emr=SimpleNamespace(
            source_problem_type="word_problem",
            representation_type="unknown",
            variables=[],
            equations=[],
            expressions=[],
            metadata={},
        ),
    )
    eds = SimpleNamespace(
        scenes=[],
        timeline=[],
        narration_tracks=[],
        formula_tracks=[],
        visual_actions=[],
        sync_anchors=[],
        quality_report=SimpleNamespace(status="passed", valid=True),
        metadata={},
    )

    summary = build_summary(
        image_path=Path("problem.png"),
        parser_output=parser_output,
        classification=Dumpable(task_type="equation_solving"),
        solution_chain=SimpleNamespace(chain_id="chain", final_answer=None, steps=[], metadata={}),
        verification=SimpleNamespace(
            valid=True,
            status="passed",
            verification_mode="tool_call_backed",
            summary="ok",
            errors=[],
            warnings=[],
            checked_steps=[],
        ),
        explanation=SimpleNamespace(
            status="passed",
            audience_level="standard",
            segments=[],
            visual_anchors=[],
            consistency_report=SimpleNamespace(valid=True, errors=[], warnings=[]),
        ),
        teaching_plan=None,
        eds=eds,
        model_config=PipelineModelConfig.from_sources(env={}, load_project_env=False),
        artifact_paths={"spr": "run/problem_1/spr.json", "eds": "run/problem_1/eds.json"},
    )

    assert "renderer" not in summary
    assert summary["timing"] == {
        "total_duration_ms": None,
        "slowest_stage": None,
        "slowest_label": None,
        "stages": [],
    }


def test_run_pipeline_defaults_to_video_rendering(monkeypatch) -> None:
    monkeypatch.setattr(
        run_pipeline,
        "DEFAULT_IMAGE_PATH",
        "data/inputs/images/math10.png",
    )
    monkeypatch.setattr(
        run_pipeline,
        "DEFAULT_OUTPUT_DIR",
        "data/intermediate/runs10_Director_enthanced",
    )
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    args = run_pipeline.parse_args()

    assert args.render_video is True


def test_run_pipeline_can_disable_default_video_rendering(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--no-render-video"])

    args = run_pipeline.parse_args()

    assert args.render_video is False


def test_run_pipeline_defaults_to_tts(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    args = run_pipeline.parse_args()

    assert args.synthesize_tts is True
    assert args.tts_timeout_seconds == 60.0


def test_run_pipeline_can_disable_default_tts(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--no-tts"])

    args = run_pipeline.parse_args()

    assert args.synthesize_tts is False


def test_run_pipeline_progress_and_timing_flags(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--quiet-progress", "--no-timing-report"])

    args = run_pipeline.parse_args()

    assert args.quiet_progress is True
    assert args.write_timing_report is False


def test_run_pipeline_stop_after_defaults_to_full(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    args = run_pipeline.parse_args()

    assert args.stop_after == "full"


def test_run_pipeline_can_stop_after_verifier(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stop-after", "verifier"])

    args = run_pipeline.parse_args()

    assert args.stop_after == "verifier"


def test_run_pipeline_teaching_planner_timeout_defaults_to_two_minutes(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    args = run_pipeline.parse_args()

    assert args.teaching_planner_timeout_seconds == 120.0


def test_run_pipeline_teaching_planner_defaults_to_deepseek_quality_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    args = run_pipeline.parse_args()

    assert args.teaching_planner_provider == "deepseek"


def test_run_pipeline_teaching_planner_timeout_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["run_pipeline.py", "--teaching-planner-timeout-seconds", "45"],
    )

    args = run_pipeline.parse_args()

    assert args.teaching_planner_timeout_seconds == 45.0


def test_run_pipeline_creates_qwen_math_text_client(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_TEST_KEY", "fake-qwen-key")
    config = PipelineModelConfig.from_sources(
        planner_provider="qwen",
        planner_model="qwen3.6-plus",
        planner_api_key_env="QWEN_TEST_KEY",
        env={},
        load_project_env=False,
    )

    client = run_pipeline.create_math_text_client(config)

    assert client.provider == "qwen"
    assert client.model == "qwen3.6-plus"
    assert client.api_key_env == "QWEN_TEST_KEY"


def test_pipeline_timer_collects_chinese_labeled_stage() -> None:
    timer = PipelineTimer(enabled=False)

    with timer.stage("vision_parser", "视觉解析", 1, 1):
        pass

    report = timer.report()
    assert report["slowest_stage"] == "vision_parser"
    assert report["slowest_label"] == "视觉解析"
    assert report["stages"][0]["status"] == "passed"
    assert report["stages"][0]["duration_ms"] >= 0
