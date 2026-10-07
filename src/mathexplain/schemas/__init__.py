"""Shared schema package for intermediate representations."""

from mathexplain.schemas.classification import (
    ClassificationResult,
    SolverConfig,
    validate_classification_result_dict,
)
from mathexplain.schemas.explanation import (
    ExplanationConsistencyReport,
    ExplanationScript,
    ExplanationSegment,
    VisualAnchor,
    validate_explanation_script_dict,
)
from mathexplain.schemas.eds import (
    EDSAsset,
    EDSFormulaItem,
    EDSNarrationItem,
    EDSQualityReport,
    EDSRenderHints,
    EDSScene,
    EDSSyncAnchor,
    EDSTimelineItem,
    EDSVisualAction,
    ExecutableDirectorScript,
    validate_executable_director_script_dict,
)
from mathexplain.schemas.verification import (
    CheckedStep,
    VerificationReport,
    validate_verification_report_dict,
)
from mathexplain.schemas.teaching_plan import (
    TeachingPlan,
    TeachingPlanMove,
    TeachingPlanQualityReport,
    validate_teaching_plan_dict,
)
from mathexplain.schemas.simple_video import (
    SimpleVideoQualityReport,
    SimpleVideoScript,
    SimpleVideoSegment,
    validate_simple_video_script_dict,
)
from mathexplain.schemas.tts import TTSClip, TTSResult, validate_tts_result_dict

__all__ = [
    "ClassificationResult",
    "SolverConfig",
    "ExplanationConsistencyReport",
    "ExplanationScript",
    "ExplanationSegment",
    "VisualAnchor",
    "EDSAsset",
    "EDSFormulaItem",
    "EDSNarrationItem",
    "EDSQualityReport",
    "EDSRenderHints",
    "EDSScene",
    "EDSSyncAnchor",
    "EDSTimelineItem",
    "EDSVisualAction",
    "ExecutableDirectorScript",
    "CheckedStep",
    "VerificationReport",
    "TeachingPlan",
    "TeachingPlanMove",
    "TeachingPlanQualityReport",
    "SimpleVideoQualityReport",
    "SimpleVideoScript",
    "SimpleVideoSegment",
    "TTSClip",
    "TTSResult",
    "validate_classification_result_dict",
    "validate_executable_director_script_dict",
    "validate_explanation_script_dict",
    "validate_teaching_plan_dict",
    "validate_simple_video_script_dict",
    "validate_tts_result_dict",
    "validate_verification_report_dict",
]
