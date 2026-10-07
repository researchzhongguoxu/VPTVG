"""Video encoding utilities for Renderer v1."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


class FFmpegVideoWriter:
    """Small wrapper around imageio-ffmpeg's streaming writer."""

    def __init__(self, output_path: Path, *, width: int, height: int, fps: int) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._writer: Any | None = None

    def open(self) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "imageio-ffmpeg is required for Renderer v1 video encoding. "
                "Install project dependencies before rendering MP4."
            ) from exc

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio_ffmpeg.write_frames(
            str(self.output_path),
            size=(self.width, self.height),
            fps=self.fps,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        self._writer.send(None)

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("Video writer has not been opened.")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} does not match expected "
                f"({self.height}, {self.width}, 3)."
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        self._writer.send(frame.tobytes())

    def close(self, *, suppress_errors: bool = False) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:
            if not suppress_errors:
                raise
        finally:
            self._writer = None


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Mux an existing video file and audio track into one MP4."""

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to mux TTS audio into video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
