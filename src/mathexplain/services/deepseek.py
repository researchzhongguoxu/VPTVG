"""DeepSeek service adapter for non-vision language planning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mathexplain.services.usage import usage_to_dict


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_API_KEY_ERROR = (
    "DEEPSEEK_API_KEY is required for DeepSeekPlanner. "
    "Please set it in .env or environment variables."
)


class DeepSeekChatClient:
    """OpenAI-compatible DeepSeek chat client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout_seconds: float = 120,
        max_retries: int = 0,
    ) -> None:
        self._load_project_dotenv()
        self.provider = "deepseek"
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            if api_key_env == "DEEPSEEK_API_KEY":
                raise ValueError(DEEPSEEK_API_KEY_ERROR)
            raise ValueError(
                f"{api_key_env} is required for DeepSeekPlanner. "
                "Please set it in .env or environment variables."
            )
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
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
        """Return a planning text response."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        self.last_usage = usage_to_dict(getattr(response, "usage", None))
        content: Any = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content is not a string.")
        return content

    @staticmethod
    def _load_project_dotenv() -> None:
        project_root = Path(__file__).resolve().parents[3]
        load_dotenv(project_root / ".env")
