"""Main SGVP / Vision Parser component flow."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mathexplain.agents.vision_parser.emr_builder import EMRBuilder
from mathexplain.agents.vision_parser.emr_repair import EMRRepair
from mathexplain.agents.vision_parser.emr_validator import EMRValidator
from mathexplain.agents.vision_parser.image_preprocessor import ImagePreprocessor
from mathexplain.agents.vision_parser.spr_generator import SPRGenerator
from mathexplain.agents.vision_parser.spr_repair import SPRRepair
from mathexplain.agents.vision_parser.spr_validator import SPRValidator
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.reports import ParsingReport, PipelineStageReport
from mathexplain.schemas.spr import SPR


class VisionParserOutput(BaseModel):
    """Output object returned by VisionParser.run."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    spr: SPR
    emr: EMR
    parsing_report: ParsingReport
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    @field_validator("problem_id")
    @classmethod
    def problem_id_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class VisionParser:
    """Schema-Guided Vision Parser component skeleton."""

    def __init__(
        self,
        image_preprocessor: ImagePreprocessor | None = None,
        spr_generator: SPRGenerator | None = None,
        spr_validator: SPRValidator | None = None,
        spr_repair: SPRRepair | None = None,
        emr_builder: EMRBuilder | None = None,
        emr_validator: EMRValidator | None = None,
        emr_repair: EMRRepair | None = None,
    ) -> None:
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.spr_generator = spr_generator or SPRGenerator()
        self.spr_validator = spr_validator or SPRValidator()
        self.spr_repair = spr_repair or SPRRepair()
        self.emr_builder = emr_builder or EMRBuilder()
        self.emr_validator = emr_validator or EMRValidator()
        self.emr_repair = emr_repair or EMRRepair()

    def run(
        self,
        image_path: str,
        save_artifacts: bool = False,
        output_dir: str | Path = "data/intermediate",
    ) -> VisionParserOutput:
        """Parse an image path into deterministic mock SPR, EMR, and report objects."""

        problem_id = self._problem_id_from_image_path(image_path)
        stages: list[PipelineStageReport] = []

        preprocess_report = self.image_preprocessor.run(image_path)
        parser_mode = getattr(self.spr_generator, "mode", "deterministic_mock")
        stages.append(
            self._stage(
                "preprocess",
                "success",
                "Preprocessed image for VisionParser.",
                preprocess_report,
            )
        )

        spr = self.spr_generator.generate(problem_id, image_path)
        parser_provider = spr.metadata.get("parser_provider") or (
            "qwen" if parser_mode == "qwen" else "deterministic_mock"
        )
        parser_model = spr.metadata.get("parser_model")
        stages.append(
            self._stage(
                "spr_generate",
                "success",
                "Generated SPR with Qwen vision model."
                if parser_mode == "qwen"
                else "Generated deterministic mock SPR.",
                {
                    "generator_mode": parser_mode,
                    "vision_provider": parser_provider,
                    "vision_model": parser_model,
                },
            )
        )

        spr_validation_report = self.spr_validator.validate(spr)
        stages.append(
            self._stage(
                "spr_validate",
                "success",
                "SPR schema validation passed.",
                spr_validation_report,
            )
        )

        spr, spr_repair_report = self.spr_repair.repair(spr, spr_validation_report)

        emr = self.emr_builder.build(problem_id, spr)
        stages.append(
            self._stage(
                "emr_build",
                "success",
                "Built EMR from SPR.",
                {
                    "builder_mode": emr.metadata.get("builder_mode", "unknown"),
                    "source_problem_type": emr.source_problem_type,
                    "representation_type": emr.representation_type,
                },
            )
        )

        emr_validation_report = self.emr_validator.validate(emr)
        emr_validation_valid = bool(emr_validation_report.get("valid", False))
        stages.append(
            self._stage(
                "emr_validate",
                "success" if emr_validation_valid else "warning",
                "EMR validation passed."
                if emr_validation_valid
                else "EMR validation reported issues.",
                emr_validation_report,
            )
        )

        emr, emr_repair_report = self.emr_repair.repair(emr, emr_validation_report)

        confidence_summary = {
            "spr_overall": spr.confidence.overall,
            "emr_schema": 1.0,
            "pipeline_overall": 1.0,
        }
        stages.append(
            self._stage(
                "confidence_estimate",
                "success",
                "Estimated parser confidence summary.",
                confidence_summary,
            )
        )

        stages.append(
            self._stage(
                "repair_stub",
                "skipped",
                "Repair agent is not implemented in this skeleton.",
                {
                    "repair_executed": False,
                    "spr_repair": spr_repair_report,
                    "emr_repair": emr_repair_report,
                },
            )
        )

        parsing_report = ParsingReport(
            problem_id=problem_id,
            status="success",
            stages=stages,
            confidence_summary=confidence_summary,
            metadata={
                "component": "SGVP",
                "parser_mode": parser_mode,
                "vision_provider": parser_provider,
                "vision_model": parser_model,
                "image_path": image_path,
                "image_read": Path(image_path).exists(),
            },
        )

        artifact_paths: dict[str, str] = {}
        if save_artifacts:
            artifact_paths = self._save_artifacts(output_dir, problem_id, spr, emr, parsing_report)

        return VisionParserOutput(
            problem_id=problem_id,
            spr=spr,
            emr=emr,
            parsing_report=parsing_report,
            artifact_paths=artifact_paths,
        )

    @staticmethod
    def _problem_id_from_image_path(image_path: str) -> str:
        digest = sha256(image_path.encode("utf-8")).hexdigest()[:12]
        return f"problem_{digest}"

    @staticmethod
    def _stage(
        stage_name: str,
        status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineStageReport:
        return PipelineStageReport(
            stage_name=stage_name,
            status=status,
            message=message,
            metadata=metadata or {},
        )

    @staticmethod
    def _save_artifacts(
        output_dir: str | Path,
        problem_id: str,
        spr: SPR,
        emr: EMR,
        parsing_report: ParsingReport,
    ) -> dict[str, str]:
        run_dir = Path(output_dir) / problem_id
        run_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "spr": run_dir / "spr.json",
            "emr": run_dir / "emr.json",
            "parsing_report": run_dir / "parsing_report.json",
        }
        artifacts["spr"].write_text(spr.model_dump_json(indent=2), encoding="utf-8")
        artifacts["emr"].write_text(emr.model_dump_json(indent=2), encoding="utf-8")
        artifacts["parsing_report"].write_text(
            parsing_report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return {name: str(path) for name, path in artifacts.items()}
