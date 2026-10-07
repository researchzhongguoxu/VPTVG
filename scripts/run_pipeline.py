"""Run implemented MathExplainAgent components on one problem image.

This script currently runs:
VisionParser -> ProblemClassifier -> Solver -> Verifier -> Explainer -> ScriptDirector

It saves inspectable intermediate JSON artifacts:
spr.json, emr.json, parsing_report.json, classification.json, scs.json,
verification.json, explanation.json, eds.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathexplain.agents.explainer import DeepSeekTeachingPlannerProvider, Explainer, RuleBasedTeachingPlanner
from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.renderer import Renderer
from mathexplain.agents.script_director import ScriptDirector
from mathexplain.agents.solver import DeepSeekPlanner, FinalAnswerRepair, Solver, VerifierGuidedReplanner
from mathexplain.agents.tts import TTSProvider
from mathexplain.agents.verifier import SemanticAnswerBindingVerifier, Verifier
from mathexplain.agents.vision_parser import SPRGenerator, VisionParser
from mathexplain.config import PipelineModelConfig, resolve_output_language
from mathexplain.schemas.classification import validate_classification_result_dict
from mathexplain.schemas.eds import validate_executable_director_script_dict
from mathexplain.schemas.explanation import validate_explanation_script_dict
from mathexplain.schemas.scs import validate_solution_chain_dict
from mathexplain.schemas.teaching_plan import validate_teaching_plan_dict
from mathexplain.schemas.tts import TTSResult, validate_tts_result_dict
from mathexplain.schemas.verification import validate_verification_report_dict
from mathexplain.services.deepseek import DeepSeekChatClient
from mathexplain.services.llm import DashScopeQwenTextClient, DashScopeQwenVisionClient
from mathexplain.services.usage import summarize_usage, usage_call


# PyCharm-friendly configuration.
# Edit these values directly, then click Run on this script.
DEFAULT_IMAGE_PATH = "data/experiments/gsm8k/images/gsm8k_test_000025.png"
DEFAULT_OUTPUT_DIR = "data/intermediate/example_run"
DEFAULT_VISION_MODE = "qwen"  # "qwen" or "determini stic_mock"
DEFAULT_VISION_MODEL = "qwen3.6-flash"
DEFAULT_USE_DEEPSEEK = True
DEFAULT_RENDER_VIDEO = True
DEFAULT_SYNTHESIZE_TTS = False


@dataclass
class StageTiming:
    """One pipeline stage timing record."""

    stage: str
    label: str
    status: str
    started_at: str
    ended_at: str
    duration_ms: int


@dataclass
class PipelineTimer:
    """Small progress printer and timing collector for run_pipeline."""

    enabled: bool = True
    stages: list[StageTiming] = field(default_factory=list)
    _pipeline_started_at: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, stage: str, label: str, index: int, total: int):
        started_at_perf = time.perf_counter()
        started_at = self._now_iso()
        if self.enabled:
            print(f"[{index}/{total}] {label}开始...")
        status = "passed"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            ended_at_perf = time.perf_counter()
            ended_at = self._now_iso()
            duration_ms = int(round((ended_at_perf - started_at_perf) * 1000))
            self.stages.append(
                StageTiming(
                    stage=stage,
                    label=label,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                )
            )
            if self.enabled:
                duration_s = duration_ms / 1000
                if status == "passed":
                    print(f"[{index}/{total}] {label}完成，用时 {duration_s:.2f} 秒")
                else:
                    print(f"[{index}/{total}] {label}失败，用时 {duration_s:.2f} 秒")

    def report(self) -> dict[str, Any]:
        total_duration_ms = int(round((time.perf_counter() - self._pipeline_started_at) * 1000))
        slowest = max(self.stages, key=lambda item: item.duration_ms, default=None)
        return {
            "total_duration_ms": total_duration_ms,
            "slowest_stage": slowest.stage if slowest else None,
            "slowest_label": slowest.label if slowest else None,
            "stages": [stage.__dict__ for stage in self.stages],
        }

    def print_done(self, image_path: Path) -> None:
        if not self.enabled:
            return
        report = self.report()
        total_s = report["total_duration_ms"] / 1000
        slowest_label = report.get("slowest_label") or "无"
        print(f"[流水线] 完成：{image_path.name}，总用时 {total_s:.2f} 秒，最耗时阶段：{slowest_label}")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VisionParser, ProblemClassifier, Solver, and Verifier on one math problem image."
        )
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help=(
            "Path to the input problem image. If omitted, uses DEFAULT_IMAGE_PATH "
            "defined at the top of this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts will be saved.",
    )
    parser.add_argument(
        "--vision-mode",
        choices=["qwen", "deterministic_mock"],
        default=DEFAULT_VISION_MODE,
        help="Use qwen for real image parsing or deterministic_mock for local smoke tests.",
    )
    parser.add_argument("--vision-provider", default=None, help="Vision provider. Currently supported: qwen.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Qwen vision model name to use.")
    parser.add_argument("--vision-base-url", default=None, help="OpenAI-compatible Qwen/DashScope base URL.")
    parser.add_argument(
        "--vision-api-key-env",
        default=None,
        help="Environment variable name containing the Qwen/DashScope API key.",
    )
    parser.add_argument("--planner-provider", default=None, help="Planner provider. Currently supported: deepseek, qwen.")
    parser.add_argument("--planner-model", default=None, help="Planner model name to use.")
    parser.add_argument("--planner-base-url", default=None, help="OpenAI-compatible planner base URL.")
    parser.add_argument(
        "--planner-api-key-env",
        default=None,
        help="Environment variable name containing the planner API key.",
    )
    parser.add_argument(
        "--no-deepseek",
        action="store_false",
        dest="use_deepseek",
        default=DEFAULT_USE_DEEPSEEK,
        help="Disable DeepSeek planner and run only deterministic Solver paths.",
    )
    parser.add_argument(
        "--teaching-planner-provider",
        choices=["auto", "rule", "deepseek"],
        default="deepseek",
        help=(
            "TeachingPlanner provider mode. deepseek prioritizes teacher-like planning and falls back to rules; "
            "auto uses rules first and DeepSeek only when rule quality is weak."
        ),
    )
    parser.add_argument(
        "--teaching-planner-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout in seconds for one DeepSeek TeachingPlanner request. Defaults to 120.",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        dest="render_video",
        default=DEFAULT_RENDER_VIDEO,
        help="Render eds.json into video.mp4 and render_report.json. Enabled by default.",
    )
    parser.add_argument(
        "--no-render-video",
        action="store_false",
        dest="render_video",
        help="Skip MP4 rendering even if DEFAULT settings are changed.",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        dest="synthesize_tts",
        default=DEFAULT_SYNTHESIZE_TTS,
        help="Synthesize narration audio from EDS. Disabled by default to avoid TTS API cost.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_false",
        dest="synthesize_tts",
        help="Skip TTS synthesis and render a silent subtitle video.",
    )
    parser.add_argument(
        "--tts-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout in seconds for one TTS request. Defaults to 60.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Do not print per-stage Chinese progress messages.",
    )
    parser.add_argument(
        "--no-timing-report",
        action="store_false",
        dest="write_timing_report",
        default=True,
        help="Do not write timing_report.json. summary.json still includes timing.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["verifier", "eds", "full"],
        default="full",
        help=(
            "Stop after a major pipeline stage. verifier runs the diagnostic math chain only; "
            "eds skips TTS and video rendering; full keeps the existing end-to-end behavior."
        ),
    )
    parser.add_argument(
        "--semantic-verifier-mode",
        choices=["off", "risk", "always"],
        default="risk",
        help=(
            "Semantic final-answer binding verifier mode. risk only calls the LLM on high-risk bindings; "
            "always calls it for diagnostics; off preserves deterministic verifier behavior."
        ),
    )
    parser.add_argument(
        "--replanner-mode",
        choices=["off", "risk", "always"],
        default="risk",
        help=(
            "Verifier-guided replanner mode. risk only calls the LLM on failed or partial math chains; "
            "always calls it for diagnostics; off disables the repair loop."
        ),
    )
    parser.add_argument(
        "--replanner-max-rounds",
        type=int,
        default=1,
        help="Maximum verifier-guided replanning rounds. Defaults to 1.",
    )
    parser.add_argument(
        "--output-language",
        choices=["auto", "en-US", "zh-CN"],
        default="auto",
        help=(
            "Language for Explainer, EDS, TTS, and Renderer. auto detects from SPR text; "
            "formula-only inputs default to en-US."
        ),
    )
    return parser.parse_args()


def create_math_text_client(
    model_config: PipelineModelConfig,
    *,
    timeout_seconds: float = 120,
) -> DeepSeekChatClient | DashScopeQwenTextClient:
    """Create the text LLM client used only by math-chain planning/verification."""

    if model_config.planner.provider == "deepseek":
        return DeepSeekChatClient(
            base_url=model_config.planner.base_url,
            model=model_config.planner.model,
            api_key_env=model_config.planner.api_key_env,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )
    if model_config.planner.provider == "qwen":
        return DashScopeQwenTextClient(
            base_url=model_config.planner.base_url,
            model=model_config.planner.model,
            api_key_env=model_config.planner.api_key_env,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )
    raise ValueError(f"Unsupported planner provider: {model_config.planner.provider}")


def main() -> int:
    args = parse_args()
    timer = PipelineTimer(enabled=not args.quiet_progress)
    effective_tts = args.synthesize_tts and args.stop_after == "full"
    effective_render = args.render_video and args.stop_after == "full"
    semantic_stage_enabled = args.semantic_verifier_mode != "off"
    replanner_stage_enabled = args.replanner_mode != "off"
    if args.stop_after == "verifier":
        total_stages = 7 + (1 if semantic_stage_enabled else 0) + (1 if replanner_stage_enabled else 0)
    elif args.stop_after == "eds":
        total_stages = 10 + (1 if semantic_stage_enabled else 0) + (1 if replanner_stage_enabled else 0)
    else:
        total_stages = (
            11
            + (1 if semantic_stage_enabled else 0)
            + (1 if replanner_stage_enabled else 0)
            + (1 if effective_tts else 0)
            + (1 if effective_render else 0)
        )
    stage_index = 1
    image_path = Path(args.image)
    output_root = Path(args.output_dir)
    artifact_paths: dict[str, str] = {}
    llm_usage_calls: list[dict[str, Any]] = []
    run_dir = output_root
    summary_path = output_root / "summary.json"
    timing_report_path = output_root / "timing_report.json"
    semantic_binding_report: dict[str, Any] | None = None
    replan_report: dict[str, Any] | None = None
    if timer.enabled:
        print(f"[流水线] 开始运行：{image_path.name}")

    try:
        with timer.stage("model_config", "模型配置", stage_index, total_stages):
            model_config = PipelineModelConfig.from_sources(
                vision_provider=args.vision_provider,
                vision_model=args.vision_model,
                vision_base_url=args.vision_base_url,
                vision_api_key_env=args.vision_api_key_env,
                planner_provider=args.planner_provider,
                planner_model=args.planner_model,
                planner_base_url=args.planner_base_url,
                planner_api_key_env=args.planner_api_key_env,
            )
            if not image_path.exists() and args.vision_mode != "deterministic_mock":
                raise FileNotFoundError(f"Input image not found: {image_path}")
            output_root.mkdir(parents=True, exist_ok=True)
        stage_index += 1

        with timer.stage("vision_parser", "视觉解析", stage_index, total_stages):
            qwen_client = None
            if args.vision_mode == "qwen":
                qwen_client = DashScopeQwenVisionClient(
                    base_url=model_config.vision.base_url,
                    model=model_config.vision.model,
                    api_key_env=model_config.vision.api_key_env,
                )
            parser = VisionParser(spr_generator=SPRGenerator(mode=args.vision_mode, qwen_client=qwen_client))
            parser_output = parser.run(
                str(image_path),
                save_artifacts=True,
                output_dir=output_root,
            )
            if qwen_client is not None and getattr(qwen_client, "last_usage", None):
                llm_usage_calls.append(
                    usage_call(
                        stage="vision_parser",
                        provider=getattr(qwen_client, "provider", model_config.vision.provider),
                        model=getattr(qwen_client, "model", model_config.vision.model),
                        usage=getattr(qwen_client, "last_usage", {}),
                    )
                )
            run_dir = Path(parser_output.artifact_paths["spr"]).parent
            summary_path = run_dir / "summary.json"
            timing_report_path = run_dir / "timing_report.json"
            artifact_paths = {
                **parser_output.artifact_paths,
                "summary": str(summary_path),
            }
            language_decision = resolve_output_language(parser_output.spr, args.output_language)
            language_metadata = language_decision.model_dump()
        stage_index += 1

        classification_path = run_dir / "classification.json"
        scs_path = run_dir / "scs.json"
        repaired_scs_path = run_dir / "scs_repaired.json"
        repair_report_path = run_dir / "repair_report.json"
        semantic_binding_report_path = run_dir / "semantic_binding_report.json"
        replan_report_path = run_dir / "replan_report.json"
        verification_path = run_dir / "verification.json"
        teaching_plan_path = run_dir / "teaching_plan.json"
        explanation_path = run_dir / "explanation.json"
        eds_path = run_dir / "eds.json"

        with timer.stage("problem_classifier", "题型分类", stage_index, total_stages):
            classification = ProblemClassifier().classify(
                parser_output.spr,
                parser_output.emr,
                parser_output.parsing_report,
            )
            validate_classification_result_dict(classification.model_dump(mode="json"))
            classification_path.write_text(
                classification.model_dump_json(indent=2),
                encoding="utf-8",
            )
            artifact_paths["classification"] = str(classification_path)
        stage_index += 1

        with timer.stage("solver", "求解器", stage_index, total_stages):
            planner = None
            if args.use_deepseek:
                planner = DeepSeekPlanner(
                    client=create_math_text_client(model_config)
                )
            solver_planner_client = planner.client if planner is not None else None
            solver = Solver(
                planner=planner,
                use_planner=args.use_deepseek,
            )
            solution_chain = solver.solve(
                parser_output.emr,
                classification,
                spr=parser_output.spr,
            )
            if solver_planner_client is not None and getattr(solver_planner_client, "last_usage", None):
                llm_usage_calls.append(
                    usage_call(
                        stage="solver_planner",
                        provider=getattr(solver_planner_client, "provider", model_config.planner.provider),
                        model=getattr(solver_planner_client, "model", model_config.planner.model),
                        usage=getattr(solver_planner_client, "last_usage", {}),
                    )
                )
            validate_solution_chain_dict(solution_chain.model_dump(mode="json"))
            scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
            artifact_paths["scs"] = str(scs_path)
        stage_index += 1

        repair_report: dict[str, Any] | None = None
        with timer.stage("solver_repair", "求解修复", stage_index, total_stages):
            repaired_chain, repair_report = FinalAnswerRepair().repair(
                solution_chain,
                spr=parser_output.spr,
            )
            if repair_report.get("repair_performed"):
                repair_verification = Verifier().verify(
                    repaired_chain,
                    spr=parser_output.spr,
                    emr=parser_output.emr,
                    classification_result=classification,
                )
                repair_report["repair_validation"] = {
                    "valid": repair_verification.valid,
                    "status": repair_verification.status,
                    "summary": repair_verification.summary,
                    "error_count": len(repair_verification.errors),
                    "warning_count": len(repair_verification.warnings),
                }
                if repair_verification.valid and repair_verification.status == "passed":
                    solution_chain = repaired_chain
                    scs_path.write_text(
                        solution_chain.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    repaired_scs_path.write_text(
                        solution_chain.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    artifact_paths["scs_repaired"] = str(repaired_scs_path)
                else:
                    repair_report["repair_performed"] = False
                    repair_report["repair_rolled_back"] = True
                    repair_report["reason"] = "Repair candidate did not pass Verifier; original SCS was kept."
            repair_report_path.write_text(
                json.dumps(repair_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            artifact_paths["repair_report"] = str(repair_report_path)
        stage_index += 1

        with timer.stage("verifier", "结果验证", stage_index, total_stages):
            verification = Verifier().verify(
                solution_chain,
                spr=parser_output.spr,
                emr=parser_output.emr,
                classification_result=classification,
            )
            validate_verification_report_dict(verification.model_dump(mode="json"))
            verification_path.write_text(
                verification.model_dump_json(indent=2),
                encoding="utf-8",
            )
            artifact_paths["verification"] = str(verification_path)
        stage_index += 1

        semantic_binding_report: dict[str, Any] | None = None
        replan_report: dict[str, Any] | None = None
        if semantic_stage_enabled:
            with timer.stage("semantic_verifier", "语义答案验证", stage_index, total_stages):
                semantic_probe = SemanticAnswerBindingVerifier()
                semantic_should_call = (
                    args.semantic_verifier_mode == "always"
                    or semantic_probe.should_run(solution_chain, verification, repair_report, parser_output.spr)
                )
                semantic_client = None
                if semantic_should_call:
                    try:
                        semantic_client = create_math_text_client(model_config)
                    except Exception as exc:
                        semantic_binding_report = {
                            "schema_version": "SEMANTIC_ANSWER_BINDING-1.0",
                            "problem_id": solution_chain.problem_id,
                            "chain_id": solution_chain.chain_id,
                            "mode": args.semantic_verifier_mode,
                            "triggered": True,
                            "accepted": False,
                            "repair_performed": False,
                            "decision": "skip",
                            "reason": f"Semantic verifier client unavailable; fallback to deterministic verifier. {exc}",
                        }
                if semantic_client is not None:
                    semantic_verifier = SemanticAnswerBindingVerifier(client=semantic_client)
                    semantic_chain, semantic_binding_report = semantic_verifier.verify_and_repair(
                        parser_output.spr,
                        solution_chain,
                        verification,
                        emr=parser_output.emr,
                        repair_report=repair_report,
                        mode=args.semantic_verifier_mode,
                    )
                    if getattr(semantic_client, "last_usage", None):
                        llm_usage_calls.append(
                            usage_call(
                                stage="semantic_verifier",
                                provider=getattr(semantic_client, "provider", model_config.planner.provider),
                                model=getattr(semantic_client, "model", model_config.planner.model),
                                usage=getattr(semantic_client, "last_usage", {}),
                            )
                        )
                    if semantic_binding_report.get("repair_performed"):
                        semantic_verification = Verifier().verify(
                            semantic_chain,
                            spr=parser_output.spr,
                            emr=parser_output.emr,
                            classification_result=classification,
                        )
                        semantic_binding_report["repair_validation"] = {
                            "valid": semantic_verification.valid,
                            "status": semantic_verification.status,
                            "summary": semantic_verification.summary,
                            "error_count": len(semantic_verification.errors),
                            "warning_count": len(semantic_verification.warnings),
                        }
                        if semantic_verification.valid and semantic_verification.status == "passed":
                            solution_chain = semantic_chain
                            verification = semantic_verification
                            validate_solution_chain_dict(solution_chain.model_dump(mode="json"))
                            validate_verification_report_dict(verification.model_dump(mode="json"))
                            scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
                            repaired_scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
                            verification_path.write_text(verification.model_dump_json(indent=2), encoding="utf-8")
                            artifact_paths["scs_repaired"] = str(repaired_scs_path)
                        else:
                            semantic_binding_report["repair_performed"] = False
                            semantic_binding_report["repair_rolled_back"] = True
                            semantic_binding_report["reason"] = (
                                "Semantic repair candidate did not pass Verifier; original SCS was kept."
                            )
                if semantic_binding_report is None:
                    _, semantic_binding_report = semantic_probe.verify_and_repair(
                        parser_output.spr,
                        solution_chain,
                        verification,
                        emr=parser_output.emr,
                        repair_report=repair_report,
                        mode=args.semantic_verifier_mode,
                    )
                semantic_binding_report_path.write_text(
                    json.dumps(semantic_binding_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                artifact_paths["semantic_binding_report"] = str(semantic_binding_report_path)
            stage_index += 1

        if replanner_stage_enabled:
            with timer.stage("verifier_guided_replanner", "verifier-guided replanner", stage_index, total_stages):
                replanner_probe = VerifierGuidedReplanner(max_rounds=args.replanner_max_rounds)
                replan_should_call = (
                    args.replanner_mode == "always"
                    or replanner_probe.should_run(
                        solution_chain,
                        verification,
                        spr=parser_output.spr,
                        repair_report=repair_report,
                        semantic_report=semantic_binding_report,
                    )
                )
                replan_client = None
                if replan_should_call:
                    try:
                        replan_client = create_math_text_client(model_config)
                    except Exception as exc:
                        replan_report = {
                            "schema_version": VerifierGuidedReplanner.REPORT_VERSION,
                            "problem_id": solution_chain.problem_id,
                            "chain_id": solution_chain.chain_id,
                            "mode": args.replanner_mode,
                            "triggered": True,
                            "accepted": False,
                            "repair_performed": False,
                            "reason": f"Replanner client unavailable; original SCS was kept. {exc}",
                            "rounds": [],
                        }
                if replan_client is not None:
                    replanner = VerifierGuidedReplanner(
                        client=replan_client,
                        max_rounds=args.replanner_max_rounds,
                    )
                    replan_result = replanner.replan(
                        parser_output.spr,
                        parser_output.emr,
                        classification,
                        solution_chain,
                        verification,
                        repair_report=repair_report,
                        semantic_report=semantic_binding_report,
                        mode=args.replanner_mode,
                    )
                    replan_report = replan_result.report
                    for round_data in replan_report.get("rounds") or []:
                        if not isinstance(round_data, dict):
                            continue
                        usage = round_data.get("usage")
                        if usage:
                            llm_usage_calls.append(
                                usage_call(
                                    stage="verifier_guided_replanner",
                                    provider=round_data.get("provider") or model_config.planner.provider,
                                    model=round_data.get("model") or model_config.planner.model,
                                    usage=usage,
                                )
                            )
                    if replan_report.get("repair_performed"):
                        replan_verification = Verifier().verify(
                            replan_result.chain,
                            spr=parser_output.spr,
                            emr=parser_output.emr,
                            classification_result=classification,
                        )
                        replan_report["repair_validation"] = {
                            "valid": replan_verification.valid,
                            "status": replan_verification.status,
                            "summary": replan_verification.summary,
                            "error_count": len(replan_verification.errors),
                            "warning_count": len(replan_verification.warnings),
                        }
                        if replan_verification.valid and replan_verification.status == "passed":
                            accepted_chain = replan_result.chain
                            accepted_verification = replan_verification
                            post_replan_semantic_report = None
                            try:
                                post_replan_client = create_math_text_client(model_config)
                                post_replan_semantic = SemanticAnswerBindingVerifier(client=post_replan_client)
                                semantic_checked_chain, post_replan_semantic_report = post_replan_semantic.verify_and_repair(
                                    parser_output.spr,
                                    accepted_chain,
                                    accepted_verification,
                                    emr=parser_output.emr,
                                    repair_report=repair_report,
                                    mode="always",
                                )
                                if getattr(post_replan_client, "last_usage", None):
                                    llm_usage_calls.append(
                                        usage_call(
                                            stage="post_replan_semantic_verifier",
                                            provider=getattr(post_replan_client, "provider", model_config.planner.provider),
                                            model=getattr(post_replan_client, "model", model_config.planner.model),
                                            usage=getattr(post_replan_client, "last_usage", {}),
                                        )
                                    )
                                if post_replan_semantic_report.get("repair_performed"):
                                    semantic_checked_verification = Verifier().verify(
                                        semantic_checked_chain,
                                        spr=parser_output.spr,
                                        emr=parser_output.emr,
                                        classification_result=classification,
                                    )
                                    post_replan_semantic_report["repair_validation"] = {
                                        "valid": semantic_checked_verification.valid,
                                        "status": semantic_checked_verification.status,
                                        "summary": semantic_checked_verification.summary,
                                        "error_count": len(semantic_checked_verification.errors),
                                        "warning_count": len(semantic_checked_verification.warnings),
                                    }
                                    if semantic_checked_verification.valid and semantic_checked_verification.status == "passed":
                                        accepted_chain = semantic_checked_chain
                                        accepted_verification = semantic_checked_verification
                            except Exception as exc:
                                post_replan_semantic_report = {
                                    "decision": "skip",
                                    "accepted": False,
                                    "repair_performed": False,
                                    "reason": f"Post-replan semantic verifier failed open: {exc}",
                                }
                            replan_report["post_replan_semantic_validation"] = post_replan_semantic_report
                            semantic_decision = (
                                (post_replan_semantic_report or {}).get("llm_decision")
                                if isinstance(post_replan_semantic_report, dict)
                                else None
                            ) or {}
                            semantic_rejects_replan = (
                                semantic_decision.get("current_answer_matches_intent") is False
                                and (post_replan_semantic_report or {}).get("decision") != "accept_current"
                            )
                            if semantic_rejects_replan:
                                replan_report["repair_performed"] = False
                                replan_report["repair_rolled_back"] = True
                                replan_report["reason"] = (
                                    "Post-replan semantic verifier judged the repaired answer as not matching the question intent."
                                )
                            else:
                                solution_chain = accepted_chain
                                verification = accepted_verification
                                validate_solution_chain_dict(solution_chain.model_dump(mode="json"))
                                validate_verification_report_dict(verification.model_dump(mode="json"))
                                scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
                                repaired_scs_path.write_text(solution_chain.model_dump_json(indent=2), encoding="utf-8")
                                verification_path.write_text(verification.model_dump_json(indent=2), encoding="utf-8")
                                artifact_paths["scs_repaired"] = str(repaired_scs_path)
                        else:
                            replan_report["repair_performed"] = False
                            replan_report["repair_rolled_back"] = True
                            replan_report["reason"] = (
                                "Verifier-guided replan candidate did not pass Verifier; original SCS was kept."
                            )
                if replan_report is None:
                    replan_report = {
                        "schema_version": VerifierGuidedReplanner.REPORT_VERSION,
                        "problem_id": solution_chain.problem_id,
                        "chain_id": solution_chain.chain_id,
                        "mode": args.replanner_mode,
                        "triggered": False,
                        "accepted": False,
                        "repair_performed": False,
                        "reason": "No replanning trigger fired.",
                        "rounds": [],
                    }
                replan_report_path.write_text(
                    json.dumps(replan_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                artifact_paths["replan_report"] = str(replan_report_path)
            stage_index += 1

        render_result = None
        tts_result = None
        teaching_plan = None
        explanation = None
        eds = None

        if args.stop_after != "verifier":
            with timer.stage("teaching_planner", "教学规划生成", stage_index, total_stages):
                teaching_provider_mode = args.teaching_planner_provider
                teaching_provider = None
                if not args.use_deepseek and teaching_provider_mode != "rule":
                    teaching_provider_mode = "rule"
                if model_config.planner.provider != "deepseek" and teaching_provider_mode != "rule":
                    teaching_provider_mode = "rule"
                if teaching_provider_mode in {"auto", "deepseek"}:
                    teaching_provider = DeepSeekTeachingPlannerProvider(
                        client_factory=lambda: DeepSeekChatClient(
                            base_url=model_config.planner.base_url,
                            model=model_config.planner.model,
                            api_key_env=model_config.planner.api_key_env,
                            timeout_seconds=args.teaching_planner_timeout_seconds,
                            max_retries=0,
                        )
                    )
                explainer = Explainer(
                    teaching_planner=RuleBasedTeachingPlanner(
                        provider_mode=teaching_provider_mode,
                        provider=teaching_provider,
                    )
                )
                teaching_plan = explainer.plan_teaching(
                    parser_output.spr,
                    classification,
                    solution_chain,
                    verification,
                    output_language=language_decision.output_language,
                )
                if teaching_provider is not None and getattr(teaching_provider, "last_usage", None):
                    llm_usage_calls.append(
                        usage_call(
                            stage="teaching_planner",
                            provider=model_config.planner.provider,
                            model=model_config.planner.model,
                            usage=getattr(teaching_provider, "last_usage", {}),
                        )
                    )
                validate_teaching_plan_dict(teaching_plan.model_dump(mode="json"))
                teaching_plan_path.write_text(
                    teaching_plan.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                artifact_paths["teaching_plan"] = str(teaching_plan_path)
            stage_index += 1

            with timer.stage("explainer", "教学讲解生成", stage_index, total_stages):
                explanation = explainer.explain(
                    parser_output.spr,
                    classification,
                    solution_chain,
                    verification,
                    teaching_plan=teaching_plan,
                    output_language=language_decision.output_language,
                    language_metadata=language_metadata,
                )
                validate_explanation_script_dict(explanation.model_dump(mode="json"))
                explanation_path.write_text(
                    explanation.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                artifact_paths["explanation"] = str(explanation_path)
            stage_index += 1

            with timer.stage("script_director", "导演脚本生成", stage_index, total_stages):
                eds = ScriptDirector().direct(
                    explanation,
                    spr=parser_output.spr,
                    scs=solution_chain,
                    verification_report=verification,
                    output_language=language_decision.output_language,
                )
                validate_executable_director_script_dict(eds.model_dump(mode="json"))
                eds_path.write_text(eds.model_dump_json(indent=2), encoding="utf-8")
                artifact_paths["eds"] = str(eds_path)
            stage_index += 1

        eds_for_render = eds
        if effective_tts and eds is not None:
            with timer.stage("tts", "语音合成", stage_index, total_stages):
                try:
                    tts_result = TTSProvider(timeout_seconds=args.tts_timeout_seconds).synthesize(
                        eds,
                        output_dir=run_dir,
                    )
                    validate_tts_result_dict(tts_result.model_dump(mode="json"))
                    if tts_result.tts_report_path:
                        artifact_paths["tts_report"] = tts_result.tts_report_path
                    if tts_result.audio_track_path:
                        artifact_paths["tts_audio_track"] = tts_result.audio_track_path
                    if tts_result.retimed_eds_path:
                        artifact_paths["eds_tts"] = tts_result.retimed_eds_path
                        eds_for_render = validate_executable_director_script_dict(
                            json.loads(Path(tts_result.retimed_eds_path).read_text(encoding="utf-8"))
                        )
                    artifact_paths["tts_audio_clips_dir"] = str(run_dir / "audio_clips")
                except Exception as exc:
                    print(f"[语音合成] 失败，后续将降级为静音视频：{exc}")
                    tts_report_path = run_dir / "tts_report.json"
                    tts_result = TTSResult(
                        status="failed",
                        provider="volcengine",
                        resource_id="seed-tts-2.0",
                        voice_type="unknown_voice",
                        clip_count=0,
                        duration_ms=0,
                        tts_report_path=str(tts_report_path),
                        errors=[str(exc)],
                        metadata={"pipeline_fallback": "silent_video"},
                    )
                    tts_report_path.write_text(
                        json.dumps(tts_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    artifact_paths["tts_report"] = str(tts_report_path)
            stage_index += 1
        if effective_render and eds_for_render is not None:
            with timer.stage("renderer", "视频渲染", stage_index, total_stages):
                render_result = Renderer().render(
                    eds_for_render,
                    output_dir=run_dir,
                    output_name="video.mp4",
                    audio_path=tts_result.audio_track_path if tts_result and tts_result.status == "passed" else None,
                )
                if render_result.video_path:
                    artifact_paths["video"] = render_result.video_path
                if render_result.render_report_path:
                    artifact_paths["render_report"] = render_result.render_report_path
            stage_index += 1

        with timer.stage("summary", "生成运行摘要", stage_index, total_stages):
            timing_report = timer.report()
            if args.write_timing_report:
                artifact_paths["timing_report"] = str(timing_report_path)
            summary = build_summary(
                image_path=image_path,
                parser_output=parser_output,
                classification=classification,
                solution_chain=solution_chain,
                verification=verification,
                explanation=explanation,
                teaching_plan=teaching_plan,
                eds=eds,
                tts_result=tts_result,
                render_result=render_result,
                model_config=model_config,
                artifact_paths=artifact_paths,
                timing=timing_report,
                language_metadata=language_metadata,
                llm_usage=summarize_usage(llm_usage_calls),
                semantic_binding_report=semantic_binding_report,
                replan_report=replan_report,
            )
            summary.update(
                derive_summary_status(
                    solution_chain=solution_chain,
                    verification=verification,
                    teaching_plan=teaching_plan,
                    explanation=explanation,
                    eds=eds,
                    tts_result=tts_result,
                    render_result=render_result,
                    stop_after=args.stop_after,
                    effective_tts=effective_tts,
                    effective_render=effective_render,
                )
            )
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if args.write_timing_report:
                timing_report_path.write_text(
                    json.dumps(timing_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    except Exception as exc:
        summary = build_failure_summary(
            image_path=image_path,
            output_dir=run_dir,
            artifact_paths=artifact_paths,
            timer=timer,
            failed_stage=_failed_stage_from_timer(timer),
            failure_reason=str(exc),
            language_metadata=locals().get("language_metadata"),
            llm_usage=summarize_usage(locals().get("llm_usage_calls", [])),
        )
        if all(
            name in locals()
            for name in ("parser_output", "classification", "solution_chain", "verification", "model_config")
        ):
            summary = build_summary(
                image_path=image_path,
                parser_output=parser_output,
                classification=classification,
                solution_chain=solution_chain,
                verification=verification,
                explanation=locals().get("explanation"),
                teaching_plan=locals().get("teaching_plan"),
                eds=locals().get("eds"),
                tts_result=locals().get("tts_result"),
                render_result=locals().get("render_result"),
                model_config=model_config,
                artifact_paths=artifact_paths,
                timing=timer.report(),
                language_metadata=locals().get("language_metadata"),
                llm_usage=summarize_usage(locals().get("llm_usage_calls", [])),
                semantic_binding_report=locals().get("semantic_binding_report"),
                replan_report=locals().get("replan_report"),
                status="failed",
                failed_stage=_failed_stage_from_timer(timer),
                failure_reason=str(exc),
            )
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if args.write_timing_report:
            artifact_paths["timing_report"] = str(timing_report_path)
            timing_report_path.write_text(
                json.dumps(timer.report(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nArtifacts saved to: {run_dir.resolve()}")
        return 1

    final_timing_report = timer.report()
    summary["timing"] = final_timing_report
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.write_timing_report:
        timing_report_path.write_text(
            json.dumps(final_timing_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved to: {run_dir.resolve()}")
    timer.print_done(image_path)
    return 0


def _failed_stage_from_timer(timer: PipelineTimer) -> str | None:
    for stage in reversed(timer.stages):
        if stage.status == "failed":
            return stage.stage
    return None


def build_failure_summary(
    image_path: Path,
    output_dir: Path,
    artifact_paths: dict[str, str],
    timer: PipelineTimer,
    failed_stage: str | None,
    failure_reason: str,
    language_metadata: dict[str, Any] | None = None,
    llm_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "image_path": str(image_path),
        "problem_id": image_path.stem,
        "detected_source_language": (language_metadata or {}).get("detected_source_language"),
        "output_language": (language_metadata or {}).get("output_language"),
        "language_policy": (language_metadata or {}).get("language_policy"),
        "language": language_metadata or {},
        "artifact_paths": {
            **artifact_paths,
            "summary": str(output_dir / "summary.json"),
        },
        "llm_usage": llm_usage or summarize_usage([]),
        "timing": timer.report(),
    }


def derive_summary_status(
    *,
    solution_chain: Any,
    verification: Any,
    teaching_plan: Any | None,
    explanation: Any | None,
    eds: Any | None,
    tts_result: Any | None,
    render_result: Any | None,
    stop_after: str,
    effective_tts: bool,
    effective_render: bool,
) -> dict[str, str | None]:
    checks: list[tuple[str, bool, str]] = [
        (
            "solver",
            not bool(solution_chain.metadata.get("partial_failure")),
            str(solution_chain.metadata.get("unsupported_reason") or "Solver did not complete all required tool calls."),
        ),
        (
            "verifier",
            bool(verification.valid) and verification.status == "passed",
            verification.summary or "Verifier did not pass.",
        ),
    ]
    if stop_after != "verifier":
        checks.extend(
            [
                (
                    "teaching_planner",
                    teaching_plan is not None and bool(teaching_plan.quality_report.valid),
                    "Teaching planner quality gate did not pass.",
                ),
                (
                    "explainer",
                    explanation is not None
                    and explanation.status == "passed"
                    and bool(explanation.consistency_report.valid),
                    "Explanation consistency gate did not pass.",
                ),
                (
                    "script_director",
                    eds is not None and bool(eds.quality_report.valid),
                    "EDS quality gate did not pass.",
                ),
            ]
        )
    if effective_tts:
        checks.append(
            (
                "tts",
                tts_result is not None and tts_result.status == "passed",
                "TTS synthesis did not pass.",
            )
        )
    if effective_render:
        checks.append(
            (
                "renderer",
                render_result is not None and render_result.status == "passed",
                "Renderer did not produce a passed video result.",
            )
        )
    for stage, passed, reason in checks:
        if not passed:
            return {"status": "failed", "failed_stage": stage, "failure_reason": reason}
    return {"status": "passed", "failed_stage": None, "failure_reason": None}


def build_summary(
    image_path: Path,
    parser_output: Any,
    classification: Any,
    solution_chain: Any,
    verification: Any,
    explanation: Any | None,
    teaching_plan: Any | None,
    eds: Any | None,
    model_config: PipelineModelConfig,
    artifact_paths: dict[str, str],
    tts_result: Any | None = None,
    render_result: Any | None = None,
    timing: dict[str, Any] | None = None,
    language_metadata: dict[str, Any] | None = None,
    llm_usage: dict[str, Any] | None = None,
    semantic_binding_report: dict[str, Any] | None = None,
    replan_report: dict[str, Any] | None = None,
    status: str = "passed",
    failed_stage: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    cas_steps = [
        step
        for step in solution_chain.steps
        if step.rule_name and step.rule_name.startswith("cas_")
    ]
    planner_steps = [
        step
        for step in solution_chain.steps
        if step.rule_name and step.rule_name.startswith("planner_")
    ]
    summary = {
        "status": status,
        "failed_stage": failed_stage,
        "failure_reason": failure_reason,
        "image_path": str(image_path),
        "problem_id": parser_output.problem_id,
        "detected_source_language": (language_metadata or {}).get("detected_source_language"),
        "output_language": (language_metadata or {}).get("output_language"),
        "language_policy": (language_metadata or {}).get("language_policy"),
        "language": language_metadata or {},
        "artifact_paths": artifact_paths,
        "model_config": model_config.metadata(),
        "llm_usage": llm_usage or summarize_usage([]),
        "spr": {
            "problem_type": str(parser_output.spr.problem_type),
            "problem_text": parser_output.spr.problem_text,
            "variables": [
                variable.model_dump(mode="json")
                for variable in parser_output.spr.variables
            ],
            "confidence": parser_output.spr.confidence.model_dump(mode="json"),
        },
        "emr": {
            "source_problem_type": str(parser_output.emr.source_problem_type),
            "representation_type": parser_output.emr.representation_type,
            "variables": [
                variable.model_dump(mode="json")
                for variable in parser_output.emr.variables
            ],
            "equation_count": len(parser_output.emr.equations),
            "expression_count": len(parser_output.emr.expressions),
            "metadata": parser_output.emr.metadata,
        },
        "classification": classification.model_dump(mode="json"),
        "solver": {
            "chain_id": solution_chain.chain_id,
            "final_answer": solution_chain.final_answer,
            "step_count": len(solution_chain.steps),
            "step_rule_names": [step.rule_name for step in solution_chain.steps],
            "metadata": solution_chain.metadata,
        },
        "repair": solution_chain.metadata.get("repair") or {
            "repair_performed": False,
        },
        "deepseek": {
            "enabled": any(
                "planner_model" in step.diagnostics for step in solution_chain.steps
            ),
            "planner_provider": model_config.planner.provider,
            "planner_model": model_config.planner.model,
            "planner_step_count": len(planner_steps),
            "planner_steps": [step.diagnostics for step in planner_steps],
        },
        "cas": {
            "backend_confirmed": bool(
                solution_chain.metadata.get("backend_confirmed")
                or any(step.diagnostics.get("success") for step in cas_steps)
            ),
            "cas_step_count": len(cas_steps),
            "cas_steps": [step.diagnostics for step in cas_steps],
        },
        "verification": {
            "valid": verification.valid,
            "status": verification.status,
            "verification_mode": verification.verification_mode,
            "summary": verification.summary,
            "error_count": len(verification.errors),
            "warning_count": len(verification.warnings),
            "checked_step_count": len(verification.checked_steps),
        },
        "semantic_verifier": (
            {
                "mode": semantic_binding_report.get("mode"),
                "triggered": semantic_binding_report.get("triggered"),
                "accepted": semantic_binding_report.get("accepted"),
                "repair_performed": semantic_binding_report.get("repair_performed"),
                "repair_type": semantic_binding_report.get("repair_type"),
                "decision": semantic_binding_report.get("decision"),
                "confidence": semantic_binding_report.get("confidence"),
                "reason": semantic_binding_report.get("reason"),
            }
            if semantic_binding_report is not None
            else {
                "mode": "not_run",
                "triggered": False,
                "accepted": False,
                "repair_performed": False,
                "repair_type": None,
                "decision": "not_run",
                "confidence": 0.0,
                "reason": "Semantic verifier was not run.",
            }
        ),
        "replanner": (
            {
                "mode": replan_report.get("mode"),
                "triggered": replan_report.get("triggered"),
                "accepted": replan_report.get("accepted"),
                "repair_performed": replan_report.get("repair_performed"),
                "round_count": len(replan_report.get("rounds") or []),
                "reason": replan_report.get("reason"),
            }
            if replan_report is not None
            else {
                "mode": "not_run",
                "triggered": False,
                "accepted": False,
                "repair_performed": False,
                "round_count": 0,
                "reason": "Verifier-guided replanner was not run.",
            }
        ),
        "explanation": (
            {
                "status": explanation.status,
                "audience_level": explanation.audience_level,
                "segment_count": len(explanation.segments),
                "visual_anchor_count": len(explanation.visual_anchors),
                "consistency_valid": explanation.consistency_report.valid,
                "error_count": len(explanation.consistency_report.errors),
                "warning_count": len(explanation.consistency_report.warnings),
            }
            if explanation is not None
            else {
                "status": "not_run",
                "audience_level": None,
                "segment_count": 0,
                "visual_anchor_count": 0,
                "consistency_valid": False,
                "error_count": 0,
                "warning_count": 0,
            }
        ),
        "teaching_plan": (
            {
                "status": teaching_plan.quality_report.status,
                "valid": teaching_plan.quality_report.valid,
                "planner_mode": teaching_plan.planner_mode,
                "move_count": len(teaching_plan.moves),
                "expanded_step_count": len(teaching_plan.quality_report.expanded_step_ids),
                "fallback_step_count": len(teaching_plan.quality_report.fallback_step_ids),
                "strategy_names": teaching_plan.quality_report.strategy_names,
                "provider_used": getattr(teaching_plan, "metadata", {}).get("provider_used"),
                "provider_attempted": getattr(teaching_plan, "metadata", {}).get("provider_attempted"),
                "provider_fallback_reason": getattr(teaching_plan, "metadata", {}).get("provider_fallback_reason"),
                "quality_gate_status": getattr(teaching_plan, "metadata", {}).get("quality_gate_status"),
            }
            if teaching_plan is not None
            else {
                "status": "unsupported",
                "valid": False,
                "planner_mode": None,
                "move_count": 0,
                "expanded_step_count": 0,
                "fallback_step_count": 0,
                "strategy_names": [],
                "provider_used": None,
                "provider_attempted": False,
                "provider_fallback_reason": None,
                "quality_gate_status": None,
            }
        ),
        "eds": (
            {
                "status": eds.quality_report.status,
                "valid": eds.quality_report.valid,
                "scene_count": len(eds.scenes),
                "timeline_count": len(eds.timeline),
                "narration_count": len(eds.narration_tracks),
                "formula_count": len(eds.formula_tracks),
                "visual_action_count": len(eds.visual_actions),
                "sync_anchor_count": len(eds.sync_anchors),
                "total_duration_ms": eds.metadata.get("total_duration_ms"),
            }
            if eds is not None
            else {
                "status": "not_run",
                "valid": False,
                "scene_count": 0,
                "timeline_count": 0,
                "narration_count": 0,
                "formula_count": 0,
                "visual_action_count": 0,
                "sync_anchor_count": 0,
                "total_duration_ms": None,
            }
        ),
        "timing": timing or {
            "total_duration_ms": None,
            "slowest_stage": None,
            "slowest_label": None,
            "stages": [],
        },
    }
    if tts_result is not None:
        summary["tts"] = {
            "status": tts_result.status,
            "provider": tts_result.provider,
            "resource_id": tts_result.resource_id,
            "voice_type": tts_result.voice_type,
            "language": getattr(tts_result, "metadata", {}).get("language"),
            "voice_selection": getattr(tts_result, "metadata", {}).get("voice_selection"),
            "clip_count": tts_result.clip_count,
            "duration_ms": tts_result.duration_ms,
            "audio_track_path": tts_result.audio_track_path,
            "retimed_eds_path": tts_result.retimed_eds_path,
            "error_count": len(tts_result.errors),
            "warning_count": len(tts_result.warnings),
        }
    if render_result is not None:
        summary["renderer"] = {
            "status": render_result.status,
            "video_path": render_result.video_path,
            "duration_ms": render_result.duration_ms,
            "frame_count": render_result.frame_count,
            "fps": render_result.fps,
            "error_count": len(render_result.errors),
            "warning_count": len(render_result.warnings),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
