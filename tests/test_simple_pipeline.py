import json
import subprocess
import sys
from pathlib import Path

import pytest

from mathexplain.agents.simple_pipeline import PlainSimpleEDSAdapter, SimpleEDSAdapter, SimpleScriptNormalizer
from mathexplain.rendering.plain_text import normalize_plain_math_text
from mathexplain.schemas.eds import validate_executable_director_script_dict
from mathexplain.schemas.simple_video import SimpleVideoScript, validate_simple_video_script_dict


def simple_response() -> str:
    return json.dumps(
        {
            "problem_text": "A basket has 25 oranges. 8 are not good. How many are good?",
            "final_answer": "17",
            "steps": [
                {
                    "title": "Find not-good oranges",
                    "narration": "There are 8 oranges that are not good.",
                    "formula": "8",
                },
                {
                    "title": "Subtract",
                    "narration": "Subtract the not-good oranges from the total.",
                    "formula": "25 - 8 = 17",
                },
            ],
            "summary": "The final answer is 17 good oranges.",
        }
    )


def test_simple_video_script_schema_accepts_minimal_valid_script() -> None:
    script = SimpleVideoScript.model_validate(
        {
            "problem_id": "simple_001",
            "source_image": "simple_001.png",
            "problem_text": "How many oranges are good?",
            "final_answer": "17",
            "segments": [
                {
                    "segment_id": "seg_answer",
                    "segment_type": "answer",
                    "narration": "The final answer is 17.",
                    "display_text": "17",
                    "formula": "17",
                    "estimated_duration_ms": 4000,
                }
            ],
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_segment_count": 1,
            },
        }
    )

    assert validate_simple_video_script_dict(script.model_dump(mode="json")) == script


def test_simple_video_script_rejects_empty_final_answer() -> None:
    with pytest.raises(ValueError):
        SimpleVideoScript.model_validate(
            {
                "problem_id": "simple_001",
                "source_image": "simple_001.png",
                "problem_text": "How many oranges are good?",
                "final_answer": "",
                "segments": [
                    {
                        "segment_id": "seg_answer",
                        "segment_type": "answer",
                        "narration": "The final answer is 17.",
                        "display_text": "17",
                        "estimated_duration_ms": 4000,
                    }
                ],
                "quality_report": {
                    "valid": True,
                    "status": "passed",
                    "checked_segment_count": 1,
                },
            }
        )


def test_simple_script_normalizer_truncates_long_steps() -> None:
    data = json.loads(simple_response())
    data["steps"] = [
        {"title": f"Step {index}", "narration": "Compute carefully.", "formula": f"{index}+1"}
        for index in range(12)
    ]

    script = SimpleScriptNormalizer().normalize(
        json.dumps(data),
        problem_id="long_steps",
        source_image="long_steps.png",
        output_language="en-US",
    )

    step_segments = [segment for segment in script.segments if segment.segment_type == "step"]
    assert len(step_segments) == 8
    assert script.quality_report.warnings


def test_simple_script_normalizer_reconciles_inconsistent_final_answer() -> None:
    response = json.dumps(
        {
            "problem_text": "A basket has 25 oranges. 8 are not good. How many are good?",
            "final_answer": "16",
            "steps": [
                {
                    "title": "Subtract",
                    "narration": "Subtract the not-good oranges from the total.",
                    "formula": "25 - 8 = 17",
                }
            ],
            "summary": "There are 17 good oranges.",
        }
    )

    script = SimpleScriptNormalizer().normalize(
        response,
        problem_id="reconciled_answer",
        source_image="reconciled_answer.png",
        output_language="en-US",
    )

    assert script.final_answer == "17 good oranges"
    answer_segment = next(segment for segment in script.segments if segment.segment_type == "answer")
    assert answer_segment.display_text == "17 good oranges"
    assert answer_segment.formula == "17 good oranges"
    assert any("reconciled" in warning for warning in script.quality_report.warnings)


def test_simple_script_normalizer_keeps_final_answer_when_evidence_disagrees() -> None:
    response = json.dumps(
        {
            "problem_text": "A basket has oranges.",
            "final_answer": "16",
            "steps": [{"narration": "Subtract.", "formula": "25 - 8 = 17"}],
            "summary": "There are 18 good oranges.",
        }
    )

    script = SimpleScriptNormalizer().normalize(
        response,
        problem_id="unreconciled_answer",
        source_image="unreconciled_answer.png",
        output_language="en-US",
    )

    assert script.final_answer == "16"


def test_simple_script_normalizer_uses_explicit_final_step_when_summary_has_no_number() -> None:
    response = json.dumps(
        {
            "problem_text": "Find the total weight.",
            "final_answer": "230 pounds",
            "steps": [
                {
                    "narration": "Finally, add the weights and the container to find the total.",
                    "formula": "Total = 50 + 50 + 50 + 60 + 30 + 20 = 260 lbs",
                }
            ],
            "summary": "By adding each part, we arrive at the total.",
        }
    )

    script = SimpleScriptNormalizer().normalize(
        response,
        problem_id="explicit_final_step",
        source_image="explicit_final_step.png",
        output_language="en-US",
    )

    assert script.final_answer == "260 lbs"
    assert any("explicit final-step result" in warning for warning in script.quality_report.warnings)


def test_simple_script_normalizer_ignores_background_number_in_summary() -> None:
    response = json.dumps(
        {
            "problem_text": "Find the total weight.",
            "final_answer": "230 pounds",
            "steps": [
                {
                    "narration": "Finally, add all weights to find the total.",
                    "formula": "Total = 50 + 50 + 50 + 60 + 30 + 20 = 260 lbs",
                }
            ],
            "summary": "By determining the animal weights relative to the 50-pound frog, we arrive at the total.",
        }
    )

    script = SimpleScriptNormalizer().normalize(
        response,
        problem_id="background_summary_number",
        source_image="background_summary_number.png",
        output_language="en-US",
    )

    assert script.final_answer == "260 lbs"


@pytest.mark.parametrize(
    ("final_answer", "formula", "summary", "expected"),
    [
        (
            "37",
            r"20 + 16 = 36 \text{ hours}",
            "Josh spent a total of 36 hours working out over the 8 weeks.",
            "36 hours",
        ),
        (
            "$3000",
            "65 * 20 = 1300",
            "Elise earned $1300 in the second year after calculating that she sold 65 books during that time.",
            "$1300",
        ),
        (
            "20 yards",
            "3,000 - 2,920 = 80 yards",
            "Blake wins by running 3,000 yards, which is 80 yards farther than Kelly's 2,920 yards.",
            "80 yards",
        ),
        (
            "20%",
            "(3600 km / 6000 km) * 100 = 60%",
            "Michael covered 3600 km, which is 60% of the total 6000 km distance.",
            "60%",
        ),
        (
            "10",
            "G = 27 - 7 = 20",
            "Griffin needed 20 additional fries from Ginger to reach his final total of 27.",
            "20",
        ),
        (
            "14",
            "40 - 24 = 16",
            "After calculating the supporters for horses #2 and #7, 16 people remain who think horse #12 will win.",
            "16 people",
        ),
        (
            "3",
            "6s + 9 = 3s + 27 -> 3s = 18 -> s = 6",
            "By setting up an equation, we determined that Sam is currently 6 years old.",
            "6 years old",
        ),
        (
            "24",
            "X + 59 = 78 => X = 19",
            "By tracking the bandages against the final remaining count, we determined the initial stock was 19 bandages.",
            "19 bandages",
        ),
        (
            "30",
            "84 - 63 = 21",
            "After harvesting twice, there are 21 melons left that were not ready to be harvested.",
            "21 melons",
        ),
    ],
)
def test_simple_script_normalizer_reconciles_summary_supported_final_answer(
    final_answer: str,
    formula: str,
    summary: str,
    expected: str,
) -> None:
    response = json.dumps(
        {
            "problem_text": "Solve the word problem.",
            "final_answer": final_answer,
            "steps": [
                {
                    "narration": "Finally, compute the requested quantity.",
                    "formula": formula,
                }
            ],
            "summary": summary,
        }
    )

    script = SimpleScriptNormalizer().normalize(
        response,
        problem_id="summary_supported_answer",
        source_image="summary_supported_answer.png",
        output_language="en-US",
    )

    assert script.final_answer == expected
    answer_segment = next(segment for segment in script.segments if segment.segment_type == "answer")
    assert answer_segment.display_text == expected
    assert any("reconciled" in warning for warning in script.quality_report.warnings)


def test_simple_eds_adapter_produces_valid_eds() -> None:
    script = SimpleScriptNormalizer().normalize(
        simple_response(),
        problem_id="simple_eds",
        source_image="simple_eds.png",
        output_language="en-US",
    )

    eds = SimpleEDSAdapter().to_eds(script, output_language="en-US")

    assert validate_executable_director_script_dict(eds.model_dump(mode="json")) == eds
    assert eds.narration_tracks
    assert eds.scenes
    assert eds.quality_report.valid is True
    assert eds.metadata["does_not_use_verifier"] is True


def test_plain_simple_eds_adapter_avoids_rich_visual_tracks() -> None:
    script = SimpleScriptNormalizer().normalize(
        simple_response(),
        problem_id="plain_simple_eds",
        source_image="plain_simple_eds.png",
        output_language="en-US",
    )

    eds = PlainSimpleEDSAdapter().to_eds(script, output_language="en-US")

    assert validate_executable_director_script_dict(eds.model_dump(mode="json")) == eds
    assert eds.render_hints.theme == "plain_simple_baseline"
    assert eds.formula_tracks == []
    assert eds.visual_actions == []
    assert eds.metadata["plain_visual_baseline"] is True


def test_plain_text_normalizer_converts_common_latex_fragments() -> None:
    assert normalize_plain_math_text(r"25 \times 0.20 = 5 \text{ unripe oranges}") == "25 × 0.20 = 5 unripe oranges"
    assert normalize_plain_math_text(r"\text{Bad} = 1, \quad \text{Sour} = 2") == "Bad = 1, Sour = 2"
    assert normalize_plain_math_text("25 ¡Á 0.20 = 5") == "25 × 0.20 = 5"


def test_plain_simple_eds_adapter_writes_readable_plain_text() -> None:
    script = SimpleVideoScript.model_validate(
        {
            "problem_id": "plain_latex",
            "source_image": "plain_latex.png",
            "problem_text": "A crate has oranges.",
            "final_answer": "17",
            "segments": [
                {
                    "segment_id": "seg_step",
                    "segment_type": "step",
                    "narration": "Find the unripe oranges.",
                    "display_text": r"25 \times 0.20 = 5 \text{ unripe oranges}",
                    "formula": r"\text{Bad} = 1, \quad \text{Sour} = 2",
                    "estimated_duration_ms": 4000,
                }
            ],
            "quality_report": {
                "valid": True,
                "status": "passed",
                "checked_segment_count": 1,
            },
        }
    )

    eds = PlainSimpleEDSAdapter().to_eds(script, output_language="en-US")
    scene_metadata = eds.scenes[0].metadata

    assert scene_metadata["plain_display_text"] == "25 × 0.20 = 5 unripe oranges"
    assert scene_metadata["plain_formula_text"] == "Bad = 1, Sour = 2"
    assert scene_metadata["plain_visual_version"] == "plain_simple_text_v4"


def test_run_simple_pipeline_mock_response_writes_json_artifacts(tmp_path: Path) -> None:
    mock_path = tmp_path / "mock_response.json"
    mock_path.write_text(simple_response(), encoding="utf-8")
    image_path = Path("data/experiments/gsm8k/images/gsm8k_test_000060.png").resolve()
    output_dir = tmp_path / "simple_runs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_simple_pipeline.py",
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--mock-response",
            str(mock_path),
            "--no-tts",
            "--no-render-video",
            "--quiet-progress",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_dir = output_dir / image_path.stem
    assert (run_dir / "raw_llm_response.json").exists()
    assert (run_dir / "simple_script.json").exists()
    assert (run_dir / "simple_eds.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pipeline"] == "simple_pipeline"
    assert summary["baseline_controls"]["uses_cas"] is False


def test_run_simple_pipeline_plain_mock_response_writes_plain_artifacts(tmp_path: Path) -> None:
    mock_path = tmp_path / "mock_response.json"
    mock_path.write_text(simple_response(), encoding="utf-8")
    image_path = Path("data/experiments/gsm8k/images/gsm8k_test_000060.png").resolve()
    output_dir = tmp_path / "plain_simple_runs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_simple_pipeline.py",
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--mock-response",
            str(mock_path),
            "--visual-style",
            "plain",
            "--no-tts",
            "--no-render-video",
            "--quiet-progress",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_dir = output_dir / image_path.stem
    assert (run_dir / "plain_simple_eds.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pipeline"] == "plain_simple_pipeline"
    assert summary["baseline_controls"]["uses_key_facts_panel"] is False
