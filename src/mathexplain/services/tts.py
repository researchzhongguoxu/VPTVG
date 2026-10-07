"""Volcengine/Doubao TTS HTTP client used behind TTSProvider v1."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_TTS_BASE_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DEFAULT_TTS_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_TTS_AUDIO_FORMAT = "wav"
DEFAULT_TTS_SAMPLE_RATE = 24000
DEFAULT_TTS_SPEED_RATIO = 0.9
TTS_API_KEY_ERROR = (
    "TTS credentials are required for TTSProvider. "
    "Please set TTS_API_KEY for the new console, or TTS_APP_ID and "
    "TTS_ACCESS_TOKEN for the service credentials page."
)


class VolcengineTTSClient:
    """Minimal V3 HTTP TTS client for Doubao/Volcengine speech synthesis."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        resource_id: str | None = None,
        voice_type: str | None = None,
        language: str | None = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        speed_ratio: float | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._load_project_dotenv()
        self.provider = "volcengine"
        self.language = language or "unknown"
        self.base_url = base_url or os.getenv("TTS_BASE_URL") or DEFAULT_TTS_BASE_URL
        self.resource_id = resource_id or os.getenv("TTS_RESOURCE_ID") or DEFAULT_TTS_RESOURCE_ID
        self.voice_selection = self._select_voice_type(voice_type, language)
        self.voice_type = self.voice_selection["voice_type"]
        self.audio_format = audio_format or os.getenv("TTS_AUDIO_FORMAT") or DEFAULT_TTS_AUDIO_FORMAT
        self.sample_rate = int(sample_rate or os.getenv("TTS_SAMPLE_RATE") or DEFAULT_TTS_SAMPLE_RATE)
        self.speed_ratio = self._speed_ratio(speed_ratio if speed_ratio is not None else os.getenv("TTS_SPEED_RATIO"))
        self.timeout_seconds = timeout_seconds
        self.app_id = os.getenv("TTS_APP_ID") or os.getenv("DOUBAO_TTS_APP_ID")
        self.access_token = os.getenv("TTS_ACCESS_TOKEN") or os.getenv("DOUBAO_TTS_ACCESS_TOKEN")
        self.api_key = (
            api_key
            or os.getenv("TTS_API_KEY")
            or os.getenv("DOUBAO_TTS_API_KEY")
            or os.getenv("VOLCENGINE_TTS_API_KEY")
        )
        self.auth_mode = os.getenv("TTS_AUTH_MODE", "auto").strip().lower()
        if self.auth_mode not in {"auto", "api_key", "app_access"}:
            raise ValueError("TTS_AUTH_MODE must be one of: auto, api_key, app_access.")
        if self.auth_mode == "auto":
            self.auth_mode = "app_access" if self.app_id and self.access_token else "api_key"
        if self.auth_mode == "app_access" and not (self.app_id and self.access_token):
            raise ValueError("TTS_APP_ID and TTS_ACCESS_TOKEN are required when TTS_AUTH_MODE=app_access.")
        if self.auth_mode == "api_key" and not self.api_key:
            raise ValueError(TTS_API_KEY_ERROR)
        if not self.voice_type:
            raise ValueError("TTS_VOICE_TYPE is required for TTSProvider.")

    @staticmethod
    def _select_voice_type(voice_type: str | None, language: str | None) -> dict[str, str | bool]:
        if voice_type:
            return {"voice_type": voice_type, "source": "explicit", "fallback": False}
        language_text = str(language or "").lower()
        if language_text.startswith("en"):
            language_voice = os.getenv("TTS_VOICE_TYPE_EN")
            if language_voice:
                return {"voice_type": language_voice, "source": "TTS_VOICE_TYPE_EN", "fallback": False}
        elif language_text.startswith("zh"):
            language_voice = os.getenv("TTS_VOICE_TYPE_ZH")
            if language_voice:
                return {"voice_type": language_voice, "source": "TTS_VOICE_TYPE_ZH", "fallback": False}
        fallback = os.getenv("TTS_VOICE_TYPE") or ""
        return {"voice_type": fallback, "source": "TTS_VOICE_TYPE", "fallback": bool(fallback)}

    def synthesize(self, text: str, *, request_id: str | None = None) -> bytes:
        """Synthesize one text segment and return raw audio bytes."""

        payload = {
            "user": {"uid": "mathexplain"},
            "req_params": {
                "text": text,
                "speaker": self.voice_type,
                "audio_params": {
                    "format": self.audio_format,
                    "sample_rate": self.sample_rate,
                    "speed_ratio": self.speed_ratio,
                },
                "additions": json.dumps(
                    {
                        "disable_markdown_filter": True,
                        "enable_latex_tn": True,
                    },
                    ensure_ascii=False,
                ),
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": request_id or str(uuid.uuid4()),
        }
        if self.auth_mode == "app_access":
            headers.update(
                {
                    "X-Api-App-Key": self.app_id or "",
                    "X-Api-App-Id": self.app_id or "",
                    "X-Api-Access-Key": self.access_token or "",
                }
            )
        else:
            headers["X-Api-Key"] = self.api_key or ""

        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TTS request failed: {exc}") from exc

        return self._extract_audio_bytes(response_body, content_type)

    @staticmethod
    def _speed_ratio(value: float | str | None) -> float:
        if value is None or value == "":
            return DEFAULT_TTS_SPEED_RATIO
        try:
            speed_ratio = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("TTS_SPEED_RATIO must be a number between 0.1 and 2.0.") from exc
        if not 0.1 <= speed_ratio <= 2.0:
            raise ValueError("TTS_SPEED_RATIO must be between 0.1 and 2.0.")
        return round(speed_ratio, 2)

    @staticmethod
    def _extract_audio_bytes(response_body: bytes, content_type: str) -> bytes:
        if "audio" in content_type.lower():
            return response_body
        stripped = response_body.strip()
        if not stripped:
            raise RuntimeError("TTS response is empty.")
        if stripped[:1] in {b"{", b"["}:
            text = stripped.decode("utf-8")
            payloads = VolcengineTTSClient._decode_json_stream(text)
            chunks: list[bytes] = []
            final_payload: dict[str, Any] | None = None
            for payload in payloads:
                if not isinstance(payload, dict):
                    continue
                code = payload.get("code", 0)
                if code not in {0, "0", 20000000, "20000000", None}:
                    raise RuntimeError(f"TTS API returned code={code}: {payload.get('message')}")
                if code in {20000000, "20000000"}:
                    final_payload = payload
                data = payload.get("data") or payload.get("audio") or payload.get("result")
                if isinstance(data, str):
                    chunks.append(base64.b64decode(data))
                elif isinstance(data, dict):
                    for key in ("audio", "data", "binary"):
                        value = data.get(key)
                        if isinstance(value, str):
                            chunks.append(base64.b64decode(value))
            if chunks:
                return b"".join(chunks)
            if final_payload is not None:
                raise RuntimeError("TTS completed successfully but returned no audio chunks.")
            raise RuntimeError("TTS JSON response did not contain base64 audio data.")
        return response_body

    @staticmethod
    def _decode_json_stream(text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        index = 0
        payloads: list[Any] = []
        while index < len(text):
            while index < len(text) and text[index] in " \r\n\t":
                index += 1
            if index >= len(text):
                break
            if text.startswith("data:", index):
                index += len("data:")
                while index < len(text) and text[index] == " ":
                    index += 1
            payload, index = decoder.raw_decode(text, index)
            payloads.append(payload)
        return payloads

    @staticmethod
    def _load_project_dotenv() -> None:
        project_root = Path(__file__).resolve().parents[3]
        load_dotenv(project_root / ".env")
