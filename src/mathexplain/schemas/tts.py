"""TTSProvider v1 output schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TTSStatus = Literal["passed", "failed"]


class TTSModel(BaseModel):
    """Base model with strict fields and shared string validation."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _ensure_non_empty(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class TTSClip(TTSModel):
    """One synthesized narration clip aligned to an EDS narration item."""

    clip_id: str
    narration_id: str
    scene_id: str
    text: str
    audio_path: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    original_start_ms: int = Field(ge=0)
    original_duration_ms: int = Field(ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("clip_id", "narration_id", "scene_id", "text", "audio_path")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


class TTSResult(TTSModel):
    """TTSProvider report returned to callers and saved as JSON."""

    schema_version: str = "TTS-1.0"
    status: TTSStatus
    provider: str = "volcengine"
    resource_id: str = "seed-tts-2.0"
    voice_type: str
    audio_format: str = "wav"
    sample_rate: int = Field(default=24000, gt=0)
    clip_count: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    audio_track_path: str | None = None
    retimed_eds_path: str | None = None
    tts_report_path: str | None = None
    clips: list[TTSClip] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "status", "provider", "resource_id", "voice_type", "audio_format")
    @classmethod
    def required_text_fields_must_be_non_empty(cls, value: str) -> str:
        return cls._ensure_non_empty(value)


def validate_tts_result_dict(data: dict[str, Any]) -> TTSResult:
    """Validate a dictionary and return a TTSResult object."""

    return TTSResult.model_validate(data)
