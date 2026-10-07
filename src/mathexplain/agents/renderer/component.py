"""Renderer v1: deterministic EDS-to-MP4 rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mathexplain.rendering.eds_player import EDSFrameRenderer
from mathexplain.rendering.video_renderer import FFmpegVideoWriter, mux_audio
from mathexplain.schemas.eds import ExecutableDirectorScript


RenderStatus = Literal["passed", "failed"]


class RenderResult(BaseModel):
    """Renderer output report returned to callers and saved as JSON."""

    model_config = ConfigDict(extra="forbid")

    status: RenderStatus
    video_path: str | None = None
    render_report_path: str | None = None
    frame_count: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    fps: int = Field(gt=0)
    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Renderer:
    """Render EDS-1.0 into a deterministic whiteboard-style MP4."""

    def render(
        self,
        eds: ExecutableDirectorScript,
        *,
        output_dir: Path | str,
        output_name: str = "video.mp4",
        audio_path: Path | str | None = None,
    ) -> RenderResult:
        """Render an executable director script and save a render report."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_path = output_path / output_name
        audio_input_path = Path(audio_path) if audio_path else None
        silent_video_path = (
            output_path / f"{video_path.stem}.silent{video_path.suffix}"
            if audio_input_path is not None
            else video_path
        )
        report_path = output_path / "render_report.json"

        errors = self._precheck_errors(eds)
        warnings: list[str] = []
        duration_ms = self._duration_ms(eds)
        fps = eds.render_hints.fps
        width = eds.render_hints.canvas_width
        height = eds.render_hints.canvas_height

        if errors:
            self._remove_stale_outputs(video_path, silent_video_path)
            result = RenderResult(
                status="failed",
                video_path=None,
                render_report_path=str(report_path),
                frame_count=0,
                duration_ms=duration_ms,
                fps=fps,
                canvas_width=width,
                canvas_height=height,
                errors=errors,
                warnings=warnings,
                metadata={
                    "renderer": "Renderer",
                    "renderer_version": "Renderer v1 pillow-imageio-ffmpeg",
                    "source_quality_status": eds.quality_report.status,
                    "does_not_generate_audio": True,
                },
            )
            self._write_report(report_path, result)
            return result

        frame_renderer = EDSFrameRenderer(eds)
        warnings.extend(frame_renderer.warnings)
        frame_count = max(1, round(duration_ms * fps / 1000))
        writer = FFmpegVideoWriter(silent_video_path, width=width, height=height, fps=fps)

        try:
            writer.open()
            for frame_index in range(frame_count):
                time_ms = min(duration_ms, round(frame_index * 1000 / fps))
                writer.write_frame(frame_renderer.render_frame(time_ms))
            writer.close()
            if audio_input_path is not None:
                mux_audio(silent_video_path, audio_input_path, video_path)
                if silent_video_path != video_path and silent_video_path.exists():
                    silent_video_path.unlink()
        except Exception as exc:  # pragma: no cover - exercised when local ffmpeg is unavailable
            writer.close(suppress_errors=True)
            if video_path.exists():
                video_path.unlink()
            if silent_video_path.exists() and silent_video_path != video_path:
                silent_video_path.unlink()
            errors.append(f"Video encoding failed: {exc}")

        status: RenderStatus = "failed" if errors else "passed"
        result = RenderResult(
            status=status,
            video_path=str(video_path) if status == "passed" else None,
            render_report_path=str(report_path),
            frame_count=frame_count if status == "passed" else 0,
            duration_ms=duration_ms,
            fps=fps,
            canvas_width=width,
            canvas_height=height,
            errors=errors,
            warnings=warnings,
            metadata={
                "renderer": "Renderer",
                "renderer_version": "Renderer v1 pillow-imageio-ffmpeg",
                "source_quality_status": eds.quality_report.status,
                "scene_count": len(eds.scenes),
                "formula_count": len(eds.formula_tracks),
                "visual_action_count": len(eds.visual_actions),
                "does_not_generate_audio": True,
                "has_audio": bool(audio_input_path and status == "passed"),
                "audio_path": str(audio_input_path) if audio_input_path else None,
            },
        )
        self._write_report(report_path, result)
        return result

    @staticmethod
    def _remove_stale_outputs(*paths: Path) -> None:
        for path in set(paths):
            if path.exists():
                path.unlink()

    @staticmethod
    def _precheck_errors(eds: ExecutableDirectorScript) -> list[str]:
        errors: list[str] = []
        if eds.quality_report.valid is not True:
            errors.append("EDS quality_report is not valid; refusing to render formal video.")
        if not eds.scenes:
            errors.append("EDS has no scenes to render.")
        if not eds.narration_tracks:
            errors.append("EDS has no narration tracks for subtitle rendering.")
        return errors

    @staticmethod
    def _duration_ms(eds: ExecutableDirectorScript) -> int:
        candidates = [scene.start_ms + scene.duration_ms for scene in eds.scenes]
        candidates.extend(item.start_ms + item.duration_ms for item in eds.timeline)
        candidates.extend(track.start_ms + track.duration_ms + track.pause_after_ms for track in eds.narration_tracks)
        return max(candidates, default=0)

    @staticmethod
    def _write_report(path: Path, result: RenderResult) -> None:
        path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
