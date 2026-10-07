"""Image preprocessing for SGVP.

This module performs lightweight deterministic image preprocessing only. It does
not call OCR, Qwen, LLMs, GPU code, or downstream reasoning components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_PROCESSED_DIR = Path("data/intermediate/image_preprocessor/processed")


class ImagePreprocessor:
    """Basic OpenCV/Pillow image preprocessor for SGVP."""

    def run(self, image_path: str, context: dict | None = None) -> dict:
        """Preprocess an image if it exists, otherwise return a non-fatal report."""

        context = context or {}
        source_path = Path(image_path)
        report = self._empty_report(image_path)

        if not source_path.exists():
            report["status"] = "skipped"
            report["warnings"].append("Image path does not exist; preprocessing skipped.")
            return report

        if not source_path.is_file():
            report["status"] = "failed"
            report["errors"].append("Image path exists but is not a file.")
            return report

        processed_dir = Path(context.get("processed_dir", DEFAULT_PROCESSED_DIR))

        try:
            image = Image.open(source_path).convert("RGB")
            rgb = np.array(image)
            original_height, original_width = rgb.shape[:2]

            crop_bbox = self._detect_content_bbox(rgb)
            x, y, width, height = (
                crop_bbox["x"],
                crop_bbox["y"],
                crop_bbox["width"],
                crop_bbox["height"],
            )
            cropped_rgb = rgb[y : y + height, x : x + width]

            if width == original_width and height == original_height:
                report["warnings"].append(
                    "No confident non-white content bbox detected; kept full image."
                )

            gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )

            processed_dir.mkdir(parents=True, exist_ok=True)
            crop_path = processed_dir / "crop.png"
            enhanced_path = processed_dir / "enhanced.png"
            binary_path = processed_dir / "binary.png"

            Image.fromarray(cropped_rgb).save(crop_path)
            Image.fromarray(enhanced).save(enhanced_path)
            Image.fromarray(binary).save(binary_path)

            report.update(
                {
                    "status": "success",
                    "image_read": True,
                    "preprocessing_applied": True,
                    "processed_image_paths": {
                        "crop": str(crop_path),
                        "enhanced": str(enhanced_path),
                        "binary": str(binary_path),
                    },
                    "crop_bbox": crop_bbox,
                    "original_size": {
                        "width": original_width,
                        "height": original_height,
                    },
                    "cropped_size": {
                        "width": width,
                        "height": height,
                    },
                }
            )
            return report
        except Exception as exc:  # pragma: no cover - defensive guard for image IO edge cases.
            report["status"] = "failed"
            report["errors"].append(f"Image preprocessing failed: {exc}")
            return report

    @staticmethod
    def _empty_report(image_path: str) -> dict[str, Any]:
        return {
            "original_image_path": image_path,
            "processed_image_paths": {},
            "crop_bbox": None,
            "original_size": None,
            "cropped_size": None,
            "status": "unknown",
            "warnings": [],
            "errors": [],
            "image_read": False,
            "preprocessing_applied": False,
            "mode": "opencv_pillow_numpy",
        }

    @staticmethod
    def _detect_content_bbox(rgb: np.ndarray) -> dict[str, int]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < 245
        coords = cv2.findNonZero(mask.astype(np.uint8))
        image_height, image_width = gray.shape[:2]

        if coords is None:
            return {
                "x": 0,
                "y": 0,
                "width": image_width,
                "height": image_height,
            }

        x, y, width, height = cv2.boundingRect(coords)
        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }
