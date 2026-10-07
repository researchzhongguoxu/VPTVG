import json
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from mathexplain.agents.renderer import Renderer
from mathexplain.rendering.eds_player import EDSFrameRenderer
from mathexplain.rendering.visual_diagrams import (
    DiagramSpec,
    _draw_text_fit_no_ellipsis,
    _localize_spec,
    _quantity_facts,
    infer_diagram_spec,
)
from mathexplain.schemas.eds import ExecutableDirectorScript


pytest.importorskip("imageio_ffmpeg")


def renderer_eds_data() -> dict:
    return {
        "problem_id": "problem_render",
        "source_explanation_id": "explanation_problem_render_chain_main",
        "source_chain_id": "chain_main",
        "scenes": [
            {
                "scene_id": "scene_1",
                "scene_type": "modeling",
                "source_segment_ids": ["seg_1"],
                "title": "建立关系",
                "start_ms": 0,
                "duration_ms": 1200,
            },
            {
                "scene_id": "scene_2",
                "scene_type": "answer",
                "source_segment_ids": ["seg_2"],
                "title": "给出答案",
                "start_ms": 1200,
                "duration_ms": 1000,
            },
        ],
        "timeline": [
            {
                "timeline_id": "timeline_nar_1",
                "scene_id": "scene_1",
                "track_type": "narration",
                "target_id": "nar_1",
                "start_ms": 0,
                "duration_ms": 1200,
                "action": "play",
            },
            {
                "timeline_id": "timeline_formula_1",
                "scene_id": "scene_1",
                "track_type": "formula",
                "target_id": "formula_1",
                "start_ms": 200,
                "duration_ms": 900,
                "action": "render",
            },
            {
                "timeline_id": "timeline_answer",
                "scene_id": "scene_2",
                "track_type": "formula",
                "target_id": "formula_answer",
                "start_ms": 1400,
                "duration_ms": 700,
                "action": "render",
            },
        ],
        "assets": [
            {
                "asset_id": "asset_source_problem_text",
                "asset_type": "problem_text",
                "metadata": {"problem_text": "水果店运来一批苹果。求原来重量。"},
            },
            {
                "asset_id": "asset_final_answer",
                "asset_type": "answer",
                "metadata": {"expression": "原来运来的苹果重量是 56 千克"},
            },
        ],
        "narration_tracks": [
            {
                "narration_id": "nar_1",
                "scene_id": "scene_1",
                "text": "题目说最后还剩 18 千克，所以建立方程。",
                "start_ms": 0,
                "duration_ms": 1200,
                "pause_after_ms": 0,
            },
            {
                "narration_id": "nar_2",
                "scene_id": "scene_2",
                "text": "最后得到原来运来的苹果重量是 56 千克。",
                "start_ms": 1200,
                "duration_ms": 1000,
                "pause_after_ms": 0,
            },
        ],
        "formula_tracks": [
            {
                "formula_id": "formula_1",
                "scene_id": "scene_1",
                "expression": "x/4 + 4 = 18",
                "display_mode": "block",
                "source_step_ids": ["step_1"],
                "start_ms": 200,
                "duration_ms": 900,
                "layout_slot": "main_formula",
                "metadata": {
                    "meaning": "最终剩余量建立方程",
                    "source_sentence": "这时还剩18千克苹果",
                },
            },
            {
                "formula_id": "formula_answer",
                "scene_id": "scene_2",
                "expression": "原来运来的苹果重量是 56 千克",
                "display_mode": "final_answer",
                "source_step_ids": ["step_2"],
                "start_ms": 1400,
                "duration_ms": 700,
                "layout_slot": "final_answer",
            },
        ],
        "visual_actions": [
            {
                "action_id": "action_highlight",
                "scene_id": "scene_1",
                "action_type": "highlight",
                "target_id": "formula_1",
                "start_ms": 400,
                "duration_ms": 500,
            },
            {
                "action_id": "action_answer",
                "scene_id": "scene_2",
                "action_type": "final_answer_reveal",
                "target_id": "formula_answer",
                "start_ms": 1500,
                "duration_ms": 500,
            },
        ],
        "sync_anchors": [],
        "render_hints": {
            "canvas_width": 640,
            "canvas_height": 360,
            "fps": 10,
            "theme": "default",
            "language": "zh-CN",
            "default_font_family": "sans-serif",
            "formula_renderer": "programmatic_2d",
        },
        "quality_report": {
            "valid": True,
            "status": "passed",
            "checked_scene_count": 2,
        },
        "metadata": {"total_duration_ms": 2200},
    }


def test_english_diagram_panel_localizes_item_prefixes_and_numbers() -> None:
    spec = DiagramSpec(
        diagram_type="generic",
        title="关键数量",
        items=["当前式：N_total = 35.0000000000000", "依据：How many flowers?"],
        highlight_index=0,
    )

    localized = _localize_spec(spec, "en-US")

    assert localized.title == "Key Facts"
    assert localized.items[0] == "Current expression: N_total = 35"
    assert localized.items[1] == "From problem: How many flowers?"


def test_quantity_facts_ignore_clock_minutes() -> None:
    facts = _quantity_facts("Twice the number of people who entered at 10:00 came in; 40 people entered at 10:00.")

    assert "00" not in facts
    assert any("40" in fact for fact in facts)


def test_renderer_generates_mp4_and_report(tmp_path: Path) -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())

    result = Renderer().render(eds, output_dir=tmp_path)

    assert result.status == "passed"
    assert result.video_path is not None
    video_path = Path(result.video_path)
    assert video_path.exists()
    assert video_path.stat().st_size > 1000
    assert result.render_report_path is not None
    report = json.loads(Path(result.render_report_path).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert result.frame_count == 22

    capture = cv2.VideoCapture(str(video_path))
    ok, first = capture.read()
    assert ok is True
    assert float(np.std(first)) > 1.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, result.frame_count - 1))
    ok, last = capture.read()
    capture.release()
    assert ok is True
    assert float(np.std(last)) > 1.0


def test_renderer_failed_eds_writes_report_without_video(tmp_path: Path) -> None:
    data = renderer_eds_data()
    data["quality_report"]["valid"] = False
    data["quality_report"]["status"] = "failed"
    eds = ExecutableDirectorScript.model_validate(data)

    result = Renderer().render(eds, output_dir=tmp_path)

    assert result.status == "failed"
    assert result.video_path is None
    assert result.errors
    assert Path(result.render_report_path or "").exists()
    assert not (tmp_path / "video.mp4").exists()


def test_renderer_can_mux_audio_track(tmp_path: Path) -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())
    audio_path = tmp_path / "track.wav"
    _write_test_wav(audio_path, duration_ms=2200)

    result = Renderer().render(eds, output_dir=tmp_path, audio_path=audio_path)

    assert result.status == "passed"
    assert result.video_path is not None
    assert Path(result.video_path).exists()
    assert result.metadata["has_audio"] is True
    assert result.metadata["audio_path"] == str(audio_path)


def test_renderer_reports_empty_scene_error(tmp_path: Path) -> None:
    data = renderer_eds_data()
    data["scenes"] = []
    eds = ExecutableDirectorScript.model_validate(data)

    result = Renderer().render(eds, output_dir=tmp_path)

    assert result.status == "failed"
    assert any("no scenes" in error for error in result.errors)


def test_renderer_font_fallback_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    import mathexplain.rendering.eds_player as eds_player

    def fake_load_font(size: int, *, bold: bool = False):
        return ImageFont.load_default(), None

    monkeypatch.setattr(eds_player, "load_font", fake_load_font)
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())

    player = EDSFrameRenderer(eds)
    frame = player.render_frame(600)

    assert frame.shape == (360, 640, 3)
    assert player.warnings == ["No system TrueType font found; using Pillow default font."]


def _write_test_wav(path: Path, *, duration_ms: int, sample_rate: int = 24000) -> None:
    frame_count = round(duration_ms * sample_rate / 1000)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        for index in range(frame_count):
            sample = int(800 * math.sin(2 * math.pi * 440 * index / sample_rate))
            writer.writeframes(struct.pack("<h", sample))


def test_renderer_scene_header_uses_chinese_title_only() -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())
    player = EDSFrameRenderer(eds)
    scene = eds.scenes[0]

    assert scene.title
    assert player._scene_type_label("modeling") != "modeling"
    assert (scene.title or player._scene_type_label(scene.scene_type)) == scene.title


def test_renderer_formula_context_text_uses_metadata() -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())
    player = EDSFrameRenderer(eds)

    assert player._formula_context_text(eds.formula_tracks[0]) == "这一步：最终剩余量建立方程｜原题干依据：这时还剩18千克苹果"


def test_renderer_formula_buildup_changes_display_text_over_time() -> None:
    data = renderer_eds_data()
    data["formula_tracks"][0]["metadata"]["buildup_mode"] = "sequential_terms"
    data["formula_tracks"][0]["metadata"]["buildup_parts"] = ["x/4", "x/4 + 4", "x/4 + 4 = 18"]
    data["formula_tracks"][0]["metadata"]["buildup_interval_ms"] = 450
    eds = ExecutableDirectorScript.model_validate(data)
    formula = eds.formula_tracks[0]

    early = EDSFrameRenderer._formula_display_text(formula, formula.start_ms)
    later = EDSFrameRenderer._formula_display_text(formula, formula.start_ms + 900)

    assert early == "x/4"
    assert later == "x/4 + 4 = 18"


def test_renderer_rejects_non_cumulative_formula_buildup_parts() -> None:
    data = renderer_eds_data()
    data["formula_tracks"][0]["expression"] = "90 ÷ (30 ÷ 10)"
    data["formula_tracks"][0]["metadata"]["buildup_mode"] = "sequential_terms"
    data["formula_tracks"][0]["metadata"]["buildup_parts"] = ["90", "÷(30", "÷10)"]
    data["formula_tracks"][0]["metadata"]["buildup_interval_ms"] = 450
    eds = ExecutableDirectorScript.model_validate(data)
    formula = eds.formula_tracks[0]

    assert EDSFrameRenderer._formula_display_text(formula, formula.start_ms + 900) == "90 ÷ (30 ÷ 10)"


def test_renderer_draws_final_answer_card_without_blank_frame() -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())
    player = EDSFrameRenderer(eds)

    frame = player.render_frame(1500)

    assert frame.shape == (360, 640, 3)
    assert float(np.std(frame)) > 1.0


def test_renderer_draws_recap_takeaway_without_formula_tracks() -> None:
    data = renderer_eds_data()
    data["scenes"].append(
        {
            "scene_id": "scene_recap",
            "scene_type": "recap",
            "source_segment_ids": ["seg_recap"],
            "title": "Recap",
            "start_ms": 2200,
            "duration_ms": 1200,
        }
    )
    data["narration_tracks"].append(
        {
            "narration_id": "nar_recap",
            "scene_id": "scene_recap",
            "text": "To wrap up, each equation came from the verified solution chain.",
            "start_ms": 2200,
            "duration_ms": 1200,
            "pause_after_ms": 0,
        }
    )
    data["timeline"].append(
        {
            "timeline_id": "timeline_recap",
            "scene_id": "scene_recap",
            "track_type": "narration",
            "target_id": "nar_recap",
            "start_ms": 2200,
            "duration_ms": 1200,
            "action": "play",
        }
    )
    data["quality_report"]["checked_scene_count"] = 3
    data["metadata"]["total_duration_ms"] = 3400
    eds = ExecutableDirectorScript.model_validate(data)
    player = EDSFrameRenderer(eds)

    frame = player.render_frame(2400)

    assert frame.shape == (360, 640, 3)
    assert float(np.std(frame)) > 1.0


def test_renderer_subtitle_layout_fits_long_chinese_text() -> None:
    data = renderer_eds_data()
    data["narration_tracks"][0]["text"] = (
        "这是一段比较长的讲解字幕，用来模拟老师在解释方程来源时需要多说几句话。"
        "字幕区域应该自动增高或者缩小字号，不能让最后一行被画到字幕框外面。"
    )
    eds = ExecutableDirectorScript.model_validate(data)
    player = EDSFrameRenderer(eds)
    image = Image.new("RGB", (player.width, player.height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font, lines, line_gap = player._subtitle_layout(draw, data["narration_tracks"][0]["text"], int(player.width * 0.84))

    assert 1 <= len(lines) <= 5
    text_height = len(lines) * max(getattr(font, "size", 16), 1) + max(0, len(lines) - 1) * line_gap
    assert text_height < int(player.height * 0.24)


def test_renderer_infers_lightweight_relation_panel_from_formula() -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())

    spec = infer_diagram_spec(eds.scenes[0], [eds.formula_tracks[0]])

    assert spec.title == "关键数量"
    assert spec.diagram_type == "generic"
    assert spec.items
    assert any(item.startswith("目标：") for item in spec.items)
    assert any(item.startswith("已知量：") for item in spec.items)
    assert any(item.startswith("使用式：") for item in spec.items)
    assert not any(item.startswith("含义：") for item in spec.items)


def test_renderer_infers_chicken_rabbit_head_relation_panel() -> None:
    scene = SimpleNamespace(scene_type="modeling", title="建立关系", metadata={})
    formula = SimpleNamespace(
        expression="x + y = 35",
        display_mode="block",
        metadata={
            "display_expression": "x + y = 35",
            "meaning": "每只鸡和每只兔都有一个头，头的总数等于鸡和兔的总数。",
            "source_sentence": "共有头35个",
        },
    )

    spec = infer_diagram_spec(
        scene,
        [formula],
        context={"all_formulas": [formula], "all_scenes": [scene]},
    )

    assert spec.title == "数量关系"
    assert spec.diagram_type == "quantity_table"
    assert spec.items == ["鸡：x 只", "兔：y 只", "头数：x + y = 35"]
    assert spec.highlight_index == 2
    assert not any(item.startswith("含义：") for item in spec.items)


def test_renderer_infers_chicken_rabbit_feet_relation_panel() -> None:
    scene = SimpleNamespace(scene_type="modeling", title="建立关系", metadata={})
    formula = SimpleNamespace(
        expression="2 × x + 4 × y = 94",
        display_mode="block",
        metadata={
            "display_expression": "2 × x + 4 × y = 94",
            "meaning": "脚的总数等于鸡脚数加上兔脚数。",
            "source_sentence": "共有脚94只",
        },
    )

    spec = infer_diagram_spec(
        scene,
        [formula],
        context={"all_formulas": [formula], "all_scenes": [scene]},
    )

    assert spec.title == "数量关系"
    assert spec.diagram_type == "quantity_table"
    assert spec.items == ["鸡脚：2x", "兔脚：4y", "脚数：2x + 4y = 94"]
    assert spec.highlight_index == 2


def test_renderer_infers_chicken_rabbit_answer_verification_panel() -> None:
    scene = SimpleNamespace(scene_type="answer", title="回答问题", metadata={})
    head = SimpleNamespace(expression="x + y = 35", display_mode="block", metadata={})
    feet = SimpleNamespace(expression="2*x + 4*y = 94", display_mode="block", metadata={})
    answer = SimpleNamespace(expression="x = 23, y = 12", display_mode="final_answer", metadata={})

    spec = infer_diagram_spec(
        scene,
        [answer],
        context={"all_formulas": [head, feet, answer], "all_scenes": [scene]},
    )

    assert spec.title == "回代检查"
    assert spec.diagram_type == "verification_check"
    assert spec.items == ["头数：23 + 12 = 35 通过", "脚数：2×23 + 4×12 = 94 通过"]


def test_renderer_infers_quadratic_function_summary_panel() -> None:
    scene = SimpleNamespace(scene_type="modeling", title="函数分析", metadata={})
    formula = SimpleNamespace(
        expression="y = x^2 - 4x + 3",
        display_mode="block",
        metadata={"meaning": "求二次函数顶点坐标", "source_sentence": "Find the vertex coordinates."},
    )

    spec = infer_diagram_spec(
        scene,
        [formula],
        context={"all_formulas": [formula], "all_scenes": [scene]},
    )

    assert spec.title == "函数信息"
    assert spec.diagram_type == "function_summary"
    assert "表达式：y = x^2 - 4x + 3" in spec.items
    assert "a=1, b=-4, c=3" in spec.items
    assert "对称轴：x=2" in spec.items


def test_renderer_key_facts_do_not_split_clock_times_into_zero_fact() -> None:
    scene = SimpleNamespace(scene_type="modeling", title="Set Up", metadata={})
    formula = SimpleNamespace(
        expression="2 × 40 = 80",
        display_mode="block",
        metadata={
            "display_expression": "2 × 40 = 80",
            "meaning": "Find the number of people who came for lunch.",
            "source_sentence": "Twice the number of people who entered the restaurant at 10:00 came in for lunch.",
        },
    )

    spec = infer_diagram_spec(
        scene,
        [formula],
        context={"language": "en-US", "all_formulas": [formula], "all_scenes": [scene]},
    )

    assert "Given: 00" not in spec.items
    assert not any(item.strip() == "Given: 10" for item in spec.items)


def test_renderer_key_facts_use_domain_specific_need_labels() -> None:
    scene = SimpleNamespace(scene_type="modeling", title="Set Up", metadata={})
    banana_formula = SimpleNamespace(
        expression="32 - 30 = 2",
        display_mode="block",
        metadata={
            "display_expression": "32 - 30 = 2",
            "meaning": "Find how much money Jenny saved by buying bananas in bunches.",
            "source_sentence": "How much money did she save by buying the bananas in bunches instead of individually?",
        },
    )
    dry_cleaning_formula = SimpleNamespace(
        expression="47 × 5 = 235",
        display_mode="block",
        metadata={
            "display_expression": "47 × 5 = 235",
            "meaning": "Find how much Alicia spends on dry-cleaning in 5 weeks.",
            "source_sentence": "How much does she spend on dry-cleaning in 5 weeks?",
        },
    )

    banana_spec = infer_diagram_spec(
        scene,
        [banana_formula],
        context={"language": "en-US", "all_formulas": [banana_formula], "all_scenes": [scene]},
    )
    dry_cleaning_spec = infer_diagram_spec(
        scene,
        [dry_cleaning_formula],
        context={"language": "en-US", "all_formulas": [dry_cleaning_formula], "all_scenes": [scene]},
    )

    assert "Need: banana savings" in banana_spec.items
    assert "Need: 5-week dry-cleaning cost" in dry_cleaning_spec.items
    assert not any("replacement cost" in item for item in [*banana_spec.items, *dry_cleaning_spec.items])


def test_renderer_hides_relation_panel_without_structural_content() -> None:
    scene = SimpleNamespace(scene_type="problem_overview", title="读题", metadata={})
    formula = SimpleNamespace(expression="", display_mode="block", metadata={})

    spec = infer_diagram_spec(scene, [formula], context={"all_formulas": [formula], "all_scenes": [scene]})

    assert spec.items == []
    assert spec.diagram_type == ""


def test_renderer_draw_relation_panel_skips_empty_spec() -> None:
    data = renderer_eds_data()
    data["scenes"][0]["scene_type"] = "problem_overview"
    data["formula_tracks"][0]["expression"] = "说明文字"
    data["formula_tracks"][0]["metadata"] = {}
    eds = ExecutableDirectorScript.model_validate(data)
    player = EDSFrameRenderer(eds)
    image = Image.new("RGB", (player.width, player.height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    player._draw_relation_panel(draw, eds.scenes[0], [eds.formula_tracks[0]])

    assert np.asarray(image).std() == 0


def test_renderer_draws_chicken_rabbit_relation_panel_without_blank_frame() -> None:
    data = renderer_eds_data()
    data["assets"][0]["metadata"]["problem_text"] = "鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。"
    data["formula_tracks"][0]["expression"] = "2 × x + 4 × y = 94"
    data["formula_tracks"][0]["metadata"] = {
        "display_expression": "2 × x + 4 × y = 94",
        "meaning": "脚的总数等于鸡脚数加上兔脚数。",
        "source_sentence": "共有脚94只",
    }
    eds = ExecutableDirectorScript.model_validate(data)
    player = EDSFrameRenderer(eds)
    image = Image.new("RGB", (player.width, player.height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    player._draw_relation_panel(draw, eds.scenes[0], [eds.formula_tracks[0]])

    assert np.asarray(image).std() > 0


def test_renderer_fit_panel_text_does_not_draw_literal_ellipsis() -> None:
    class RecordingDraw:
        def __init__(self, inner):
            self.inner = inner
            self.text_values: list[str] = []

        def textbbox(self, *args, **kwargs):
            return self.inner.textbbox(*args, **kwargs)

        def text(self, xy, text, *args, **kwargs):
            self.text_values.append(str(text))
            return self.inner.text(xy, text, *args, **kwargs)

    image = Image.new("RGB", (480, 240), (255, 255, 255))
    recorder = RecordingDraw(ImageDraw.Draw(image))
    font = ImageFont.load_default()

    _draw_text_fit_no_ellipsis(
        recorder,
        "脚数：2x + 4y = 94，并且这是一个故意很长的辅助说明文本",
        (20, 20, 160, 52),
        base_font=font,
        fill=(0, 0, 0),
    )

    assert recorder.text_values
    assert not any("..." in item for item in recorder.text_values)
