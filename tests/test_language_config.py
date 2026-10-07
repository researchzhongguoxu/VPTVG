from mathexplain.config.language import resolve_output_language
from mathexplain.schemas.spr import ConfidenceInfo, SourceImage, SPR


def make_spr(text: str) -> SPR:
    return SPR(
        source_image=SourceImage(image_path="problem.png"),
        problem_text=text,
        confidence=ConfidenceInfo(overall=0.9),
    )


def test_resolve_output_language_detects_english() -> None:
    decision = resolve_output_language(make_spr("Solve for x if 2x + 3 = 9."), "auto")

    assert decision.detected_source_language == "en-US"
    assert decision.output_language == "en-US"


def test_resolve_output_language_detects_chinese() -> None:
    decision = resolve_output_language(make_spr("解方程：2x+3=9。"), "auto")

    assert decision.detected_source_language == "zh-CN"
    assert decision.output_language == "zh-CN"


def test_resolve_output_language_formula_only_defaults_to_english() -> None:
    decision = resolve_output_language(make_spr("2x + 3 = 9"), "auto")

    assert decision.detected_source_language == "formula-only"
    assert decision.output_language == "en-US"


def test_resolve_output_language_explicit_overrides_detection() -> None:
    decision = resolve_output_language(make_spr("Solve for x if 2x + 3 = 9."), "zh-CN")

    assert decision.detected_source_language == "en-US"
    assert decision.output_language == "zh-CN"
    assert decision.language_policy == "explicit"
