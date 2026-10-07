"""Normalize loose Qwen SPR JSON into strict SPR-1.0 dictionaries."""

from __future__ import annotations

import re
from typing import Any


ALLOWED_PROBLEM_TYPES = {
    "algebra",
    "calculus",
    "geometry",
    "statistics",
    "probability",
    "linear_algebra",
    "combinatorics",
    "number_theory",
    "word_problem",
    "proof",
    "mixed",
    "unknown",
}
ALLOWED_QUESTION_TYPES = {
    "solve",
    "prove",
    "compute",
    "simplify",
    "derive",
    "integrate",
    "limit",
    "select",
    "explain",
    "unknown",
}
ALLOWED_FORMULA_ROLES = {"condition", "goal", "option", "derived_candidate", "unknown"}
ALLOWED_VISUAL_OBJECT_TYPES = {
    "point",
    "line",
    "segment",
    "ray",
    "angle",
    "triangle",
    "polygon",
    "circle",
    "arc",
    "curve",
    "axis",
    "table",
    "chart",
    "diagram",
    "other",
    "unknown",
}
OPTION_PATTERN = re.compile(
    r"(?:^|[\s\n])(?P<label>[A-Da-d])\s*[.．、:：]\s*(?P<value>[^\n\r]+?)(?=(?:\s+[A-Da-d]\s*[.．、:：])|$)"
)
RUNTIME_METADATA_KEYS = {
    "processing_time",
    "elapsed_time",
    "elapsed_seconds",
    "duration",
    "duration_seconds",
    "latency",
    "latency_seconds",
    "token_count",
    "tokens",
    "usage",
}


class QwenSPRNormalizer:
    """Convert semantically useful but loose Qwen JSON into strict SPR-1.0 shape."""

    def normalize(
        self,
        raw_data: dict,
        problem_id: str,
        image_path: str,
        parser_model: str = "unknown",
        parser_provider: str = "qwen",
        prompt_version: str = "qwen-spr-v1",
    ) -> dict:
        """Normalize raw Qwen JSON into a dictionary accepted by SPR validation."""

        normalized = {
            "schema_version": "SPR-1.0",
            "source_image": self._normalize_source_image(raw_data.get("source_image"), image_path),
            "problem_text": self._string_or_empty(raw_data.get("problem_text")),
            "problem_stem": self._optional_string(raw_data.get("problem_stem")),
            "conditions": self._normalize_conditions(raw_data.get("conditions")),
            "questions": self._normalize_questions(raw_data.get("questions")),
            "formulas": self._normalize_formulas(raw_data.get("formulas")),
            "variables": self._normalize_variables(raw_data.get("variables")),
            "visual_objects": self._normalize_visual_objects(raw_data.get("visual_objects")),
            "layout": self._normalize_layout(raw_data.get("layout")),
            "problem_type": self._normalize_problem_type(raw_data.get("problem_type")),
            "knowledge_units": self._normalize_string_list(raw_data.get("knowledge_units")),
            "targets": self._normalize_targets(raw_data.get("targets"), raw_data.get("questions")),
            "confidence": self._normalize_confidence(raw_data.get("confidence")),
            "uncertainties": self._normalize_uncertainties(raw_data.get("uncertainties")),
            "metadata": self._normalize_metadata(
                raw_data.get("metadata"),
                problem_id,
                parser_model,
                parser_provider,
                prompt_version,
            ),
        }
        return self._postprocess(normalized)

    @classmethod
    def _postprocess(cls, data: dict) -> dict:
        problem_text = cls._collect_problem_text(data)
        intent = cls._infer_problem_intent(problem_text)
        condition_formula_ids = cls._condition_formula_ids(data.get("conditions"))

        if intent != "unknown":
            cls._fill_unknown_question_types(data.get("questions"), intent)
            cls._fill_unknown_target_types(data.get("targets"), intent)

        cls._mark_condition_formulas(data.get("formulas"), condition_formula_ids)
        cls._link_questions_to_targets(data.get("questions"), data.get("targets"))
        cls._append_option_formulas(data, str(data.get("problem_text") or ""))
        cls._mark_goal_formulas(
            data.get("formulas"),
            data.get("questions"),
            data.get("targets"),
            condition_formula_ids,
        )
        return data

    @staticmethod
    def _normalize_source_image(value: Any, image_path: str) -> dict:
        if isinstance(value, dict):
            return {
                "image_path": image_path or value.get("image_path") or "unknown",
                "page_index": value.get("page_index"),
                "width": value.get("width"),
                "height": value.get("height"),
            }
        return {
            "image_path": image_path,
            "page_index": None,
            "width": None,
            "height": None,
        }

    @staticmethod
    def _normalize_conditions(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                text = item
                linked_formula_ids = []
            elif isinstance(item, dict):
                text = item.get("text") or item.get("description") or item.get("label") or ""
                linked_formula_ids = item.get("linked_formula_ids") or []
            else:
                continue
            if text:
                normalized.append(
                    {
                        "id": QwenSPRNormalizer._id_from_item(item, f"c{index}"),
                        "text": str(text),
                        "linked_formula_ids": linked_formula_ids
                        if isinstance(linked_formula_ids, list)
                        else [],
                    }
                )
        return normalized

    @staticmethod
    def _normalize_questions(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                text = item
                raw_type = "unknown"
                target_ids = []
            elif isinstance(item, dict):
                text = item.get("text") or item.get("description") or ""
                raw_type = item.get("question_type") or item.get("type") or "unknown"
                target_ids = item.get("target_ids") or []
            else:
                continue
            if text:
                normalized.append(
                    {
                        "id": QwenSPRNormalizer._id_from_item(item, f"q{index}"),
                        "text": str(text),
                        "question_type": QwenSPRNormalizer._normalize_action_type(raw_type),
                        "target_ids": target_ids if isinstance(target_ids, list) else [],
                    }
                )
        return normalized

    @staticmethod
    def _normalize_formulas(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                raw_text = item
                latex = None
                role = "unknown"
            elif isinstance(item, dict):
                raw_text = item.get("raw_text") or item.get("text") or item.get("formula") or ""
                latex = item.get("latex")
                role = item.get("role") or "unknown"
            else:
                continue
            if raw_text:
                normalized.append(
                    {
                        "id": QwenSPRNormalizer._id_from_item(item, f"f{index}"),
                        "raw_text": str(raw_text),
                        "latex": latex if isinstance(latex, str) else None,
                        "role": role if role in ALLOWED_FORMULA_ROLES else "unknown",
                    }
                )
        return normalized

    @staticmethod
    def _normalize_variables(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for item in items:
            if isinstance(item, str):
                symbol = item
                description = None
                domain = None
            elif isinstance(item, dict):
                symbol = item.get("symbol") or item.get("name") or item.get("label") or ""
                description = item.get("description")
                domain = item.get("domain")
            else:
                continue
            if symbol:
                normalized.append(
                    {
                        "symbol": str(symbol),
                        "description": description if isinstance(description, str) else None,
                        "domain": domain if isinstance(domain, str) else None,
                    }
                )
        return normalized

    @staticmethod
    def _normalize_visual_objects(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                normalized.append(
                    {
                        "id": f"vo{index}",
                        "object_type": "unknown",
                        "description": item,
                        "bbox": None,
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            raw_type = item.get("object_type") or item.get("type") or "unknown"
            label = item.get("label") or item.get("name")
            description = item.get("description")
            properties = item.get("properties")
            description_parts = [part for part in (label, description) if isinstance(part, str)]
            if isinstance(properties, dict) and properties:
                description_parts.append(str(properties))
            normalized.append(
                {
                    "id": item.get("id") or f"vo{index}",
                    "object_type": QwenSPRNormalizer._normalize_visual_object_type(raw_type),
                    "description": "; ".join(description_parts) if description_parts else None,
                    "bbox": item.get("bbox") if isinstance(item.get("bbox"), dict) else None,
                }
            )
        return normalized

    @staticmethod
    def _normalize_layout(value: Any) -> dict | None:
        if isinstance(value, dict):
            blocks = value.get("blocks")
            if isinstance(blocks, list):
                return {"blocks": blocks}
        return None

    @staticmethod
    def _normalize_problem_type(value: Any) -> str:
        if value == "geometry_proof":
            return "geometry"
        return value if isinstance(value, str) and value in ALLOWED_PROBLEM_TYPES else "unknown"

    @staticmethod
    def _normalize_targets(value: Any, questions: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        default_type = "prove" if QwenSPRNormalizer._contains_proof_intent(questions) else "unknown"
        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                text = item
                target_type = default_type
            elif isinstance(item, dict):
                text = item.get("text") or item.get("description") or item.get("target") or ""
                target_type = item.get("target_type") or item.get("type") or default_type
            else:
                continue
            if text:
                normalized.append(
                    {
                        "id": QwenSPRNormalizer._id_from_item(item, f"t{index}"),
                        "text": str(text),
                        "target_type": QwenSPRNormalizer._normalize_action_type(target_type),
                    }
                )
        return normalized

    @staticmethod
    def _normalize_confidence(value: Any) -> dict:
        if isinstance(value, int | float):
            return {"overall": QwenSPRNormalizer._clamp_score(float(value))}
        if isinstance(value, dict):
            overall = QwenSPRNormalizer._clamp_score(float(value.get("overall", 0.0) or 0.0))
            return {
                "overall": overall,
                "text": QwenSPRNormalizer._optional_score(value.get("text")),
                "formula": QwenSPRNormalizer._optional_score(value.get("formula")),
                "layout": QwenSPRNormalizer._optional_score(value.get("layout")),
            }
        return {"overall": 0.0}

    @staticmethod
    def _normalize_uncertainties(value: Any) -> list[dict]:
        items = value if isinstance(value, list) else []
        normalized = []
        for item in items:
            if isinstance(item, str):
                normalized.append(
                    {
                        "field_path": "unknown",
                        "message": item,
                        "severity": "medium",
                    }
                )
            elif isinstance(item, dict):
                message = item.get("message") or item.get("text") or item.get("description")
                if message:
                    severity = item.get("severity") if item.get("severity") in {"low", "medium", "high"} else "medium"
                    normalized.append(
                        {
                            "field_path": item.get("field_path") or "unknown",
                            "message": str(message),
                            "severity": severity,
                        }
                    )
        return normalized

    @staticmethod
    def _normalize_metadata(
        value: Any,
        problem_id: str,
        parser_model: str,
        parser_provider: str,
        prompt_version: str,
    ) -> dict:
        metadata = {}
        if isinstance(value, dict):
            metadata = {
                str(key): item
                for key, item in value.items()
                if not QwenSPRNormalizer._is_runtime_metadata_key(str(key))
            }
        metadata["problem_id"] = problem_id
        metadata["parser_mode"] = "qwen"
        metadata["parser_provider"] = parser_provider
        metadata["parser_model"] = parser_model
        metadata["prompt_version"] = prompt_version
        return metadata

    @staticmethod
    def _mark_condition_formulas(formulas: Any, condition_formula_ids: set[str]) -> None:
        if not isinstance(formulas, list):
            return
        for formula in formulas:
            if (
                isinstance(formula, dict)
                and formula.get("id") in condition_formula_ids
                and formula.get("role") == "unknown"
            ):
                formula["role"] = "condition"

    @staticmethod
    def _collect_problem_text(data: dict) -> str:
        parts = [data.get("problem_text"), data.get("problem_stem")]
        for key in ("conditions", "questions", "targets", "formulas"):
            items = data.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    parts.extend([item.get("text"), item.get("raw_text"), item.get("latex")])
        return "\n".join(str(part) for part in parts if isinstance(part, str))

    @staticmethod
    def _infer_problem_intent(text: str) -> str:
        if QwenSPRNormalizer._contains_proof_intent(text):
            return "prove"
        if QwenSPRNormalizer._contains_choice_intent(text):
            return "select"
        if QwenSPRNormalizer._contains_compute_intent(text):
            return "compute"
        return "unknown"

    @staticmethod
    def _fill_unknown_question_types(questions: Any, intent: str) -> None:
        if not isinstance(questions, list):
            return
        for question in questions:
            if isinstance(question, dict) and question.get("question_type") == "unknown":
                question["question_type"] = intent

    @staticmethod
    def _fill_unknown_target_types(targets: Any, intent: str) -> None:
        if not isinstance(targets, list):
            return
        for target in targets:
            if isinstance(target, dict) and target.get("target_type") == "unknown":
                target["target_type"] = intent

    @staticmethod
    def _link_questions_to_targets(questions: Any, targets: Any) -> None:
        if not isinstance(questions, list) or not isinstance(targets, list):
            return
        for index, question in enumerate(questions):
            if not isinstance(question, dict) or question.get("target_ids"):
                continue
            if index < len(targets) and isinstance(targets[index], dict) and targets[index].get("id"):
                question["target_ids"] = [targets[index]["id"]]
            elif len(targets) == 1 and isinstance(targets[0], dict) and targets[0].get("id"):
                question["target_ids"] = [targets[0]["id"]]

    @classmethod
    def _append_option_formulas(cls, data: dict, text: str) -> None:
        options = cls._extract_choice_options(text)
        if not options:
            return
        formulas = data.setdefault("formulas", [])
        if not isinstance(formulas, list):
            data["formulas"] = []
            formulas = data["formulas"]
        existing_option_labels = {
            str(item.get("raw_text", "")).strip().upper()[:1]
            for item in formulas
            if isinstance(item, dict) and item.get("role") == "option"
        }
        existing_option_values = {
            cls._compact_math_text(str(item.get("latex") or item.get("raw_text") or ""))
            for item in formulas
            if isinstance(item, dict) and item.get("role") == "option"
        }
        next_index = cls._next_formula_index(formulas)
        for label, value in options:
            label = label.upper()
            if label in existing_option_labels or cls._compact_math_text(value) in existing_option_values:
                continue
            formulas.append(
                {
                    "id": f"f{next_index}",
                    "raw_text": f"{label}. {value}",
                    "latex": value,
                    "role": "option",
                }
            )
            next_index += 1

    @classmethod
    def _mark_goal_formulas(
        cls,
        formulas: Any,
        questions: Any,
        targets: Any,
        condition_formula_ids: set[str],
    ) -> None:
        if not isinstance(formulas, list):
            return
        goal_text = cls._goal_context_text(questions, targets)
        if not goal_text:
            return
        for formula in formulas:
            if not isinstance(formula, dict):
                continue
            if formula.get("role") != "unknown" or formula.get("id") in condition_formula_ids:
                continue
            raw_text = formula.get("raw_text")
            if isinstance(raw_text, str) and cls._formula_matches_goal(raw_text, goal_text):
                formula["role"] = "goal"

    @staticmethod
    def _goal_context_text(questions: Any, targets: Any) -> str:
        parts = []
        for items in (questions, targets):
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(parts)

    @staticmethod
    def _formula_matches_goal(raw_text: str, goal_text: str) -> bool:
        raw_normalized = QwenSPRNormalizer._compact_math_text(raw_text)
        goal_normalized = QwenSPRNormalizer._compact_math_text(goal_text)
        return bool(raw_normalized and raw_normalized in goal_normalized)

    @staticmethod
    def _condition_formula_ids(conditions: Any) -> set[str]:
        formula_ids = set()
        if not isinstance(conditions, list):
            return formula_ids
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            linked_ids = condition.get("linked_formula_ids")
            if isinstance(linked_ids, list):
                formula_ids.update(str(item) for item in linked_ids if item)
        return formula_ids

    @staticmethod
    def _extract_choice_options(text: str) -> list[tuple[str, str]]:
        options = []
        for match in OPTION_PATTERN.finditer(text):
            value = match.group("value").strip()
            if value:
                options.append((match.group("label").upper(), value))
        return options if len(options) >= 2 else []

    @staticmethod
    def _contains_choice_intent(value: Any) -> bool:
        text = str(value)
        return len(QwenSPRNormalizer._extract_choice_options(text)) >= 2 or "( )" in text or "（ ）" in text

    @staticmethod
    def _contains_compute_intent(value: Any) -> bool:
        text = str(value).lower()
        keywords = (
            "最小值",
            "最大值",
            "求值",
            "计算",
            "minimum",
            "maximum",
            "compute",
            "value",
        )
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _next_formula_index(formulas: list[Any]) -> int:
        max_index = 0
        for formula in formulas:
            if not isinstance(formula, dict):
                continue
            match = re.fullmatch(r"f(\d+)", str(formula.get("id", "")))
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    @staticmethod
    def _compact_math_text(value: str) -> str:
        return re.sub(r"\s+", "", value).lower().replace("·", "*").replace("^", "")

    @staticmethod
    def _is_runtime_metadata_key(key: str) -> bool:
        normalized = key.lower()
        return normalized in RUNTIME_METADATA_KEYS or "token" in normalized or "usage" in normalized

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _normalize_action_type(value: Any) -> str:
        mapping = {
            "proof": "prove",
            "prove": "prove",
            "solve": "solve",
            "compute": "compute",
            "simplify": "simplify",
            "derive": "derive",
            "integrate": "integrate",
            "limit": "limit",
            "select": "select",
            "explain": "explain",
        }
        if not isinstance(value, str):
            return "unknown"
        normalized = mapping.get(value.lower(), value.lower())
        return normalized if normalized in ALLOWED_QUESTION_TYPES else "unknown"

    @staticmethod
    def _normalize_visual_object_type(value: Any) -> str:
        if not isinstance(value, str):
            return "unknown"
        normalized = value.lower()
        mapping = {
            "line_segment": "segment",
            "linesegment": "segment",
            "shaded_region": "other",
        }
        normalized = mapping.get(normalized, normalized)
        return normalized if normalized in ALLOWED_VISUAL_OBJECT_TYPES else "unknown"

    @staticmethod
    def _id_from_item(item: Any, fallback: str) -> str:
        if isinstance(item, dict) and item.get("id"):
            return str(item["id"])
        return fallback

    @staticmethod
    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _optional_score(value: Any) -> float | None:
        if isinstance(value, int | float):
            return QwenSPRNormalizer._clamp_score(float(value))
        return None

    @staticmethod
    def _contains_proof_intent(value: Any) -> bool:
        return "proof" in str(value).lower() or "prove" in str(value).lower() or "求证" in str(value)
