"""Lightweight model configuration for provider/model selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mathexplain.services.deepseek import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from mathexplain.services.llm import DEFAULT_DASHSCOPE_BASE_URL, DEFAULT_DASHSCOPE_MODEL, DEFAULT_DASHSCOPE_TEXT_MODEL


class UnsupportedModelProviderError(ValueError):
    """Raised when a configured provider is not supported by the current build."""


@dataclass(frozen=True)
class ModelProviderConfig:
    """Configuration for one model provider endpoint."""

    provider: str
    model: str
    base_url: str
    api_key_env: str

    def metadata(self) -> dict[str, str]:
        """Return provider metadata safe to persist in JSON artifacts."""

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
        }


@dataclass(frozen=True)
class PipelineModelConfig:
    """Model configuration used by a pipeline run."""

    vision: ModelProviderConfig
    planner: ModelProviderConfig

    @classmethod
    def from_sources(
        cls,
        *,
        vision_provider: str | None = None,
        vision_model: str | None = None,
        vision_base_url: str | None = None,
        vision_api_key_env: str | None = None,
        planner_provider: str | None = None,
        planner_model: str | None = None,
        planner_base_url: str | None = None,
        planner_api_key_env: str | None = None,
        env: dict[str, str] | None = None,
        load_project_env: bool = True,
    ) -> "PipelineModelConfig":
        """Build config from CLI overrides, environment variables, and defaults.

        CLI arguments should be passed as keyword values. They take precedence
        over environment variables; environment variables take precedence over
        the currently verified default model names.
        """

        if load_project_env:
            load_dotenv(cls._project_root() / ".env")
        source = env if env is not None else os.environ
        planner_provider_value = cls._coalesce(planner_provider, source.get("PLANNER_PROVIDER"), "deepseek").lower()
        if planner_provider_value == "qwen":
            planner_model_default = cls._coalesce(
                source.get("PLANNER_MODEL"),
                source.get("DASHSCOPE_TEXT_MODEL"),
                source.get("DASHSCOPE_MODEL"),
                DEFAULT_DASHSCOPE_TEXT_MODEL,
            )
            planner_base_url_default = cls._coalesce(
                source.get("PLANNER_BASE_URL"),
                source.get("DASHSCOPE_BASE_URL"),
                DEFAULT_DASHSCOPE_BASE_URL,
            )
            planner_api_key_env_default = cls._coalesce(
                source.get("PLANNER_API_KEY_ENV"),
                source.get("VISION_API_KEY_ENV"),
                "DASHSCOPE_API_KEY",
            )
        else:
            planner_model_default = cls._coalesce(source.get("PLANNER_MODEL"), source.get("DEEPSEEK_MODEL"), DEFAULT_DEEPSEEK_MODEL)
            planner_base_url_default = cls._coalesce(source.get("PLANNER_BASE_URL"), source.get("DEEPSEEK_BASE_URL"), DEFAULT_DEEPSEEK_BASE_URL)
            planner_api_key_env_default = cls._coalesce(source.get("PLANNER_API_KEY_ENV"), "DEEPSEEK_API_KEY")

        vision = ModelProviderConfig(
            provider=cls._coalesce(vision_provider, source.get("VISION_PROVIDER"), "qwen").lower(),
            model=cls._coalesce(vision_model, source.get("VISION_MODEL"), source.get("DASHSCOPE_MODEL"), DEFAULT_DASHSCOPE_MODEL),
            base_url=cls._coalesce(vision_base_url, source.get("VISION_BASE_URL"), source.get("DASHSCOPE_BASE_URL"), DEFAULT_DASHSCOPE_BASE_URL),
            api_key_env=cls._coalesce(vision_api_key_env, source.get("VISION_API_KEY_ENV"), "DASHSCOPE_API_KEY"),
        )
        planner = ModelProviderConfig(
            provider=planner_provider_value,
            model=cls._coalesce(planner_model, planner_model_default),
            base_url=cls._coalesce(planner_base_url, planner_base_url_default),
            api_key_env=cls._coalesce(planner_api_key_env, planner_api_key_env_default),
        )
        config = cls(vision=vision, planner=planner)
        config.validate_supported()
        return config

    def validate_supported(self) -> None:
        """Validate provider names supported by this implementation."""

        if self.vision.provider != "qwen":
            raise UnsupportedModelProviderError(
                f"Unsupported vision_provider={self.vision.provider}. Currently supported: qwen."
            )
        if self.planner.provider not in {"deepseek", "qwen"}:
            raise UnsupportedModelProviderError(
                f"Unsupported planner_provider={self.planner.provider}. Currently supported: deepseek, qwen."
            )

    def metadata(self) -> dict[str, Any]:
        """Return safe metadata for run summaries and artifacts."""

        return {
            "vision_provider": self.vision.provider,
            "vision_model": self.vision.model,
            "vision_base_url": self.vision.base_url,
            "vision_api_key_env": self.vision.api_key_env,
            "planner_provider": self.planner.provider,
            "planner_model": self.planner.model,
            "planner_base_url": self.planner.base_url,
            "planner_api_key_env": self.planner.api_key_env,
        }

    @staticmethod
    def _coalesce(*values: str | None) -> str:
        for value in values:
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]
