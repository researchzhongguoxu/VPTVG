"""TTSProvider v1: synthesize EDS narration into an audio track."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Protocol

from mathexplain.schemas.eds import ExecutableDirectorScript
from mathexplain.schemas.tts import TTSClip, TTSResult
from mathexplain.services.tts import (
    DEFAULT_TTS_AUDIO_FORMAT,
    DEFAULT_TTS_RESOURCE_ID,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_SPEED_RATIO,
    VolcengineTTSClient,
)


ANSWER_SCENE_TAIL_MS = 2500


class SpeechSynthesisClient(Protocol):
    """Small client protocol used by TTSProvider and tests."""

    provider: str
    resource_id: str
    voice_type: str
    audio_format: str
    sample_rate: int
    speed_ratio: float

    def synthesize(self, text: str, *, request_id: str | None = None) -> bytes:
        """Return synthesized audio bytes for one text segment."""


class TTSProvider:
    """Generate narration audio from EDS narration tracks."""

    def __init__(
        self,
        *,
        client: SpeechSynthesisClient | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.client = client
        self.timeout_seconds = timeout_seconds

    def synthesize(
        self,
        eds: ExecutableDirectorScript,
        *,
        output_dir: Path | str,
    ) -> TTSResult:
        """Synthesize EDS narration tracks and save audio/report artifacts."""

        if self.client is None:
            self.client = VolcengineTTSClient(
                timeout_seconds=self.timeout_seconds,
                language=eds.render_hints.language,
            )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        clips_dir = output_path / "audio_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "tts_report.json"
        retimed_eds_path = output_path / "eds_tts.json"
        audio_track_path = output_path / "narration_track.wav"

        warnings: list[str] = []
        voice_selection = getattr(self.client, "voice_selection", None)
        if isinstance(voice_selection, dict) and voice_selection.get("fallback"):
            language = str(eds.render_hints.language or "")
            if language.lower().startswith("en"):
                warnings.append(
                    "English TTS is using the fallback voice from "
                    f"{voice_selection.get('source')}; set TTS_VOICE_TYPE_EN to an English voice type "
                    "to keep the accent consistent."
                )
            else:
                warnings.append(
                    f"TTS voice fell back to {voice_selection.get('source')} for language {eds.render_hints.language}."
                )
        errors = self._precheck_errors(eds)
        if errors:
            self._remove_stale_outputs(audio_track_path, retimed_eds_path)
            result = self._result(
                status="failed",
                clips=[],
                duration_ms=0,
                report_path=report_path,
                audio_track_path=None,
                retimed_eds_path=None,
                errors=errors,
                warnings=warnings,
            )
            self._write_report(report_path, result)
            return result

        clips: list[TTSClip] = []
        for index, narration in enumerate(sorted(eds.narration_tracks, key=lambda item: item.start_ms), start=1):
            clip_id = f"clip_{index}_{narration.narration_id}"
            clip_path = clips_dir / f"{clip_id}.{self.client.audio_format}"
            try:
                audio_bytes = self.client.synthesize(narration.text, request_id=clip_id)
                clip_path.write_bytes(audio_bytes)
                duration_ms = self._measure_audio_duration_ms(clip_path)
            except Exception as exc:
                errors.append(f"{narration.narration_id}: TTS synthesis failed: {exc}")
                continue
            clips.append(
                TTSClip(
                    clip_id=clip_id,
                    narration_id=narration.narration_id,
                    scene_id=narration.scene_id,
                    text=narration.text,
                    audio_path=str(clip_path),
                    start_ms=narration.start_ms,
                    duration_ms=duration_ms,
                    original_start_ms=narration.start_ms,
                    original_duration_ms=narration.duration_ms,
                    pause_after_ms=narration.pause_after_ms,
                    metadata={
                        "original_duration_ms": narration.duration_ms,
                        "tts_hints": dict(narration.tts_hints),
                    },
                )
            )

        if errors:
            result = self._result(
                status="failed",
                clips=clips,
                duration_ms=0,
                report_path=report_path,
                audio_track_path=None,
                retimed_eds_path=None,
                errors=errors,
                warnings=warnings,
            )
            self._write_report(report_path, result)
            return result

        retimed_eds, retimed_clips, total_duration_ms = self._retime_eds(eds, clips)
        retimed_eds_path.write_text(
            retimed_eds.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._write_audio_track(audio_track_path, retimed_clips, total_duration_ms)
        result = self._result(
            status="passed",
            clips=retimed_clips,
            duration_ms=total_duration_ms,
            report_path=report_path,
            audio_track_path=audio_track_path,
            retimed_eds_path=retimed_eds_path,
            errors=[],
            warnings=warnings,
        )
        self._write_report(report_path, result)
        return result

    def _result(
        self,
        *,
        status: str,
        clips: list[TTSClip],
        duration_ms: int,
        report_path: Path,
        audio_track_path: Path | None,
        retimed_eds_path: Path | None,
        errors: list[str],
        warnings: list[str],
    ) -> TTSResult:
        return TTSResult(
            status=status,  # type: ignore[arg-type]
            provider=getattr(self.client, "provider", "volcengine"),
            resource_id=getattr(self.client, "resource_id", DEFAULT_TTS_RESOURCE_ID),
            voice_type=getattr(self.client, "voice_type", "unknown_voice"),
            audio_format=getattr(self.client, "audio_format", DEFAULT_TTS_AUDIO_FORMAT),
            sample_rate=getattr(self.client, "sample_rate", DEFAULT_TTS_SAMPLE_RATE),
            clip_count=len(clips),
            duration_ms=duration_ms,
            audio_track_path=str(audio_track_path) if audio_track_path else None,
            retimed_eds_path=str(retimed_eds_path) if retimed_eds_path else None,
            tts_report_path=str(report_path),
            clips=clips,
            errors=errors,
            warnings=warnings,
            metadata={
                "provider_public_name": "TTSProvider",
                "clip_directory": str(report_path.parent / "audio_clips"),
                "does_not_rewrite_narration": True,
                "sync_granularity": "narration_segment",
                "speed_ratio": getattr(self.client, "speed_ratio", DEFAULT_TTS_SPEED_RATIO),
                "language": getattr(self.client, "language", None),
                "voice_selection": getattr(self.client, "voice_selection", None),
            },
        )

    @staticmethod
    def _precheck_errors(eds: ExecutableDirectorScript) -> list[str]:
        errors: list[str] = []
        if eds.quality_report.valid is not True:
            errors.append("EDS quality_report is not valid; refusing to synthesize formal narration.")
        if not eds.narration_tracks:
            errors.append("EDS has no narration tracks for TTS.")
        return errors

    @staticmethod
    def _remove_stale_outputs(*paths: Path) -> None:
        for path in set(paths):
            if path.exists():
                path.unlink()

    @staticmethod
    def _retime_eds(
        eds: ExecutableDirectorScript,
        clips: list[TTSClip],
    ) -> tuple[ExecutableDirectorScript, list[TTSClip], int]:
        data = eds.model_dump(mode="json")
        original_scene_starts = {scene.scene_id: scene.start_ms for scene in eds.scenes}
        scene_new_start: dict[str, int] = {}
        scene_new_duration: dict[str, int] = {}
        clip_by_narration = {clip.narration_id: clip for clip in clips}

        cursor = 0
        for scene in sorted(eds.scenes, key=lambda item: item.start_ms):
            original_start = scene.start_ms
            duration = scene.duration_ms
            answer_narration_end: int | None = None
            for collection in (eds.timeline, eds.formula_tracks, eds.visual_actions):
                for item in collection:
                    if item.scene_id == scene.scene_id:
                        relative_start = max(0, item.start_ms - original_start)
                        duration = max(duration, relative_start + item.duration_ms)
            for narration in eds.narration_tracks:
                if narration.scene_id == scene.scene_id:
                    clip = clip_by_narration[narration.narration_id]
                    relative_start = max(0, narration.start_ms - original_start)
                    if scene.scene_type == "answer":
                        answer_narration_end = max(
                            answer_narration_end or 0,
                            relative_start + clip.duration_ms + ANSWER_SCENE_TAIL_MS,
                        )
                    duration = max(duration, relative_start + clip.duration_ms + narration.pause_after_ms)
            if scene.scene_type == "answer" and answer_narration_end is not None:
                duration = min(duration, max(answer_narration_end, 2500))
            scene_new_start[scene.scene_id] = cursor
            scene_new_duration[scene.scene_id] = duration
            cursor += duration

        def shifted_start(scene_id: str, original_start_ms: int) -> int:
            return scene_new_start[scene_id] + max(0, original_start_ms - original_scene_starts[scene_id])

        for scene_data in data["scenes"]:
            scene_id = scene_data["scene_id"]
            scene_data["start_ms"] = scene_new_start[scene_id]
            scene_data["duration_ms"] = scene_new_duration[scene_id]
            scene_data.setdefault("metadata", {})["tts_retimed"] = True

        retimed_clips: list[TTSClip] = []
        for narration_data in data["narration_tracks"]:
            narration_id = narration_data["narration_id"]
            scene_id = narration_data["scene_id"]
            clip = clip_by_narration[narration_id]
            new_start = shifted_start(scene_id, narration_data["start_ms"])
            narration_data["start_ms"] = new_start
            narration_data["duration_ms"] = clip.duration_ms
            narration_data.setdefault("tts_hints", {})["actual_audio_duration_ms"] = clip.duration_ms
            retimed_clips.append(
                clip.model_copy(
                    update={
                        "start_ms": new_start,
                        "duration_ms": clip.duration_ms,
                    }
                )
            )

        for key in ("timeline", "formula_tracks", "visual_actions", "sync_anchors"):
            for item_data in data[key]:
                scene_id = item_data["scene_id"]
                time_key = "time_ms" if key == "sync_anchors" else "start_ms"
                item_data[time_key] = shifted_start(scene_id, item_data[time_key])
                if key != "sync_anchors" and item_data.get("duration_ms") is not None:
                    scene_end = scene_new_start[scene_id] + scene_new_duration[scene_id]
                    remaining = max(0, scene_end - item_data[time_key])
                    item_data["duration_ms"] = min(int(item_data["duration_ms"]), remaining)
                if key == "timeline" and item_data["track_type"] == "narration":
                    clip = clip_by_narration.get(item_data["target_id"])
                    if clip is not None:
                        item_data["duration_ms"] = clip.duration_ms

        data.setdefault("metadata", {})["tts_retimed"] = True
        data["metadata"]["total_duration_ms"] = cursor
        return ExecutableDirectorScript.model_validate(data), retimed_clips, cursor

    def _write_audio_track(self, output_path: Path, clips: list[TTSClip], total_duration_ms: int) -> None:
        sample_rate = getattr(self.client, "sample_rate", DEFAULT_TTS_SAMPLE_RATE)
        sample_width = 2
        channels = 1
        cursor_ms = 0
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            for clip in sorted(clips, key=lambda item: item.start_ms):
                if clip.start_ms > cursor_ms:
                    self._write_silence(writer, clip.start_ms - cursor_ms, sample_rate, sample_width, channels)
                    cursor_ms = clip.start_ms
                audio_bytes, clip_duration_ms = self._as_pcm_wav_frames(
                    Path(clip.audio_path),
                    sample_rate=sample_rate,
                    sample_width=sample_width,
                    channels=channels,
                )
                writer.writeframes(audio_bytes)
                cursor_ms += clip_duration_ms
            if total_duration_ms > cursor_ms:
                self._write_silence(writer, total_duration_ms - cursor_ms, sample_rate, sample_width, channels)

    @staticmethod
    def _write_silence(writer: wave.Wave_write, duration_ms: int, sample_rate: int, sample_width: int, channels: int) -> None:
        frame_count = max(0, round(duration_ms * sample_rate / 1000))
        writer.writeframes(b"\x00" * frame_count * sample_width * channels)

    def _as_pcm_wav_frames(
        self,
        path: Path,
        *,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> tuple[bytes, int]:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as reader:
                    if (
                        reader.getframerate() == sample_rate
                        and reader.getsampwidth() == sample_width
                        and reader.getnchannels() == channels
                    ):
                        frames = reader.readframes(reader.getnframes())
                        duration_ms = round(reader.getnframes() * 1000 / reader.getframerate())
                        return frames, duration_ms
            except wave.Error:
                pass

        ffmpeg = self._ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "clip.wav"
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(path),
                    "-ac",
                    str(channels),
                    "-ar",
                    str(sample_rate),
                    "-sample_fmt",
                    "s16",
                    str(wav_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with wave.open(str(wav_path), "rb") as reader:
                frames = reader.readframes(reader.getnframes())
                duration_ms = round(reader.getnframes() * 1000 / reader.getframerate())
                return frames, duration_ms

    def _measure_audio_duration_ms(self, path: Path) -> int:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as reader:
                    return round(reader.getnframes() * 1000 / reader.getframerate())
            except wave.Error:
                pass
        _, duration_ms = self._as_pcm_wav_frames(
            path,
            sample_rate=getattr(self.client, "sample_rate", DEFAULT_TTS_SAMPLE_RATE),
            sample_width=2,
            channels=1,
        )
        return duration_ms

    @staticmethod
    def _ffmpeg_exe() -> str:
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                return ffmpeg
            raise RuntimeError("FFmpeg is required to normalize non-WAV TTS audio.")

    @staticmethod
    def _write_report(path: Path, result: TTSResult) -> None:
        path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
