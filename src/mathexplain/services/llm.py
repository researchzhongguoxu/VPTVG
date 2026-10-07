"""Large language model service adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mathexplain.services.usage import usage_to_dict


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3.6-flash"
DEFAULT_DASHSCOPE_TEXT_MODEL = "qwen3.6-plus"
DASHSCOPE_API_KEY_ERROR = (
    "DASHSCOPE_API_KEY is required for SPRGenerator mode='qwen'. "
    "Please set it in .env or environment variables."
)


class DashScopeQwenVisionClient:
    """OpenAI-compatible client for DashScope Qwen vision models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_seconds: float = 60,
        max_retries: int = 0,
    ) -> None:
        self._load_project_dotenv()
        self.provider = "qwen"
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            if api_key_env == "DASHSCOPE_API_KEY":
                raise ValueError(DASHSCOPE_API_KEY_ERROR)
            raise ValueError(
                f"{api_key_env} is required for SPRGenerator mode='qwen'. "
                "Please set it in .env or environment variables."
            )

        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL
        self.model = model or os.getenv("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.last_usage: dict[str, Any] = {}
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def generate_json_from_image(
        self,
        image_data_uri: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return the model's JSON response content for one image prompt."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri,
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_object",
            },
            extra_body={
                "enable_thinking": False,
            },
        )
        self.last_usage = usage_to_dict(getattr(response, "usage", None))
        content: Any = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Qwen response content is not a JSON string.")
        return content

    @staticmethod
    def _load_project_dotenv() -> None:
        project_root = Path(__file__).resolve().parents[3]
        load_dotenv(project_root / ".env")


class DashScopeQwenTextClient:
    """OpenAI-compatible client for DashScope Qwen text planning calls."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_seconds: float = 120,
        max_retries: int = 0,
    ) -> None:
        self._load_project_dotenv()
        self.provider = "qwen"
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            if api_key_env == "DASHSCOPE_API_KEY":
                raise ValueError(DASHSCOPE_API_KEY_ERROR)
            raise ValueError(
                f"{api_key_env} is required for Qwen text planning. "
                "Please set it in .env or environment variables."
            )

        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL
        self.model = model or os.getenv("DASHSCOPE_TEXT_MODEL") or os.getenv("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_TEXT_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.last_usage: dict[str, Any] = {}
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a text response for Solver/Semantic planning."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            extra_body={
                "enable_thinking": False,
            },
        )
        self.last_usage = usage_to_dict(getattr(response, "usage", None))
        content: Any = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Qwen text response content is not a string.")
        return content

    @staticmethod
    def _load_project_dotenv() -> None:
        project_root = Path(__file__).resolve().parents[3]
        load_dotenv(project_root / ".env")
