"""SPR generation for SGVP."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image
from pydantic import ValidationError

from mathexplain.agents.vision_parser.prompts import (
    DETERMINISTIC_MOCK_PROMPT_VERSION,
    QWEN_SPR_PROMPT_VERSION,
    QWEN_SPR_SYSTEM_PROMPT,
    QWEN_SPR_USER_PROMPT,
)
from mathexplain.agents.vision_parser.spr_normalizer import QwenSPRNormalizer
from mathexplain.schemas.spr import SPR, validate_spr_dict
from mathexplain.services.llm import DashScopeQwenVisionClient


class QwenVisionJSONClient(Protocol):
    """Protocol for fake and real Qwen vision clients used by SPRGenerator."""

    model: str

    def generate_json_from_image(
        self,
        image_data_uri: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return JSON content from an image prompt."""


class SPRGenerator:
    """Generate SPR using deterministic mock data or Qwen vision parsing."""

    MOCK_PROBLEM_TEXT = "Solve 2*x + 3 = 7."
    MOCK_CONDITION_TEXT = "2*x + 3 = 7"
    MOCK_QUESTION_TEXT = "Solve for x."
    MOCK_VARIABLE = "x"

    def __init__(
        self,
        mode: Literal["deterministic_mock", "qwen"] = "deterministic_mock",
        qwen_client: QwenVisionJSONClient | None = None,
        normalizer: QwenSPRNormalizer | None = None,
    ) -> None:
        self.mode = mode
        self.qwen_client = qwen_client
        self.normalizer = normalizer or QwenSPRNormalizer()

    def generate(self, problem_id: str, image_path: str) -> SPR:
        """Generate SPR in the configured mode."""

        if self.mode == "deterministic_mock":
            return self._generate_mock(problem_id, image_path)
        if self.mode == "qwen":
            return self._generate_with_qwen(problem_id, image_path)
        raise ValueError(f"Unsupported SPRGenerator mode: {self.mode}")

    def _generate_mock(self, problem_id: str, image_path: str) -> SPR:
        """Generate the fixed algebra SPR used by the default VisionParser skeleton."""

        data = {
            "source_image": {
                "image_path": image_path,
            },
            "problem_text": self.MOCK_PROBLEM_TEXT,
            "problem_stem": self.MOCK_PROBLEM_TEXT,
            "conditions": [
                {
                    "id": "condition_1",
                    "text": self.MOCK_CONDITION_TEXT,
                    "linked_formula_ids": ["formula_1"],
                }
            ],
            "questions": [
                {
                    "id": "question_1",
                    "text": self.MOCK_QUESTION_TEXT,
                    "question_type": "solve",
                    "target_ids": ["target_1"],
                }
            ],
            "formulas": [
                {
                    "id": "formula_1",
                    "raw_text": self.MOCK_CONDITION_TEXT,
                    "latex": "2x + 3 = 7",
                    "role": "condition",
                }
            ],
            "variables": [
                {
                    "symbol": self.MOCK_VARIABLE,
                    "description": "unknown variable",
                    "domain": "real",
                }
            ],
            "problem_type": "algebra",
            "knowledge_units": ["linear equations"],
            "targets": [
                {
                    "id": "target_1",
                    "text": self.MOCK_VARIABLE,
                    "target_type": "solve",
                }
            ],
            "confidence": {
                "overall": 1.0,
                "text": 1.0,
                "formula": 1.0,
                "layout": 1.0,
            },
            "metadata": {
                "problem_id": problem_id,
                "parser_mode": "deterministic_mock",
                "prompt_version": DETERMINISTIC_MOCK_PROMPT_VERSION,
            },
        }
        return validate_spr_dict(data)

    def _generate_with_qwen(self, problem_id: str, image_path: str) -> SPR:
        """Generate SPR by calling a Qwen vision client and validating the JSON result."""

        client = self.qwen_client or DashScopeQwenVisionClient()
        image_data_uri = self._image_to_data_uri(image_path)
        raw_content = client.generate_json_from_image(
            image_data_uri=image_data_uri,
            system_prompt=QWEN_SPR_SYSTEM_PROMPT,
            user_prompt=QWEN_SPR_USER_PROMPT,
        )

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen returned non-JSON content for SPR generation.") from exc

        if not isinstance(data, dict):
            raise ValueError("Qwen SPR response must be a JSON object.")

        normalized_data = self.normalizer.normalize(
            data,
            problem_id=problem_id,
            image_path=image_path,
            parser_model=getattr(client, "model", "unknown"),
            parser_provider=getattr(client, "provider", "qwen"),
            prompt_version=QWEN_SPR_PROMPT_VERSION,
        )

        try:
            return validate_spr_dict(normalized_data)
        except ValidationError as exc:
            raise ValueError("Qwen SPR response is still invalid after normalization.") from exc

    @staticmethod
    def _image_to_data_uri(image_path: str) -> str:
        path = Path(image_path)
        with Image.open(path) as image:
            image_format = image.format or path.suffix.lstrip(".") or "PNG"
        mime_type = SPRGenerator._mime_type_for_image_format(image_format)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _mime_type_for_image_format(image_format: str) -> str:
        normalized = image_format.lower().lstrip(".")
        if normalized in {"jpg", "jpeg"}:
            return "image/jpeg"
        if normalized == "png":
            return "image/png"
        if normalized == "webp":
            return "image/webp"
        if normalized == "bmp":
            return "image/bmp"
        return f"image/{normalized or 'png'}"
