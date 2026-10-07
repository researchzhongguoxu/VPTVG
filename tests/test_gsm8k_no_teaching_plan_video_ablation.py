from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_gsm8k_no_teaching_plan_video_ablation import (
    JUDGE_PROMPT,
    ab_mapping,
    env_values_first_non_placeholder,
    frame_times_from_eds,
    judge_configs,
    load_manifest,
    parse_json_object,
    winner_to_system,
)
from scripts.run_gsm8k_no_teaching_plan_video_judge_experiment import (
    DEFAULT_JUDGE_COUNT,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PROVIDERS,
    namespace_for_judges,
    parse_args as parse_judge_experiment_args,
    selected_judges,
)
from scripts.generate_gsm8k_no_teaching_plan_videos import (
    DEFAULT_NUM_SAMPLES as DEFAULT_GENERATE_NUM_SAMPLES,
    DEFAULT_SAVE_OUTPUT_DIR,
    parse_args as parse_generate_video_args,
)


def test_load_manifest_uses_limit_rank_and_video_eval(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "a", "selection_rank": 0, "include_in_video_eval": True},
        {"sample_id": "b", "selection_rank": 1, "include_in_video_eval": False},
        {"sample_id": "c", "selection_rank": 2, "include_in_video_eval": True},
        {"sample_id": "d", "selection_rank": 3, "include_in_video_eval": True},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = load_manifest(manifest, limit=2, selection_rank_start=1)

    assert [row["sample_id"] for row in selected] == ["c", "d"]


def test_load_manifest_sample_list_preserves_requested_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "a", "selection_rank": 0, "include_in_video_eval": True},
        {"sample_id": "b", "selection_rank": 1, "include_in_video_eval": True},
        {"sample_id": "c", "selection_rank": 2, "include_in_video_eval": True},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sample_list = tmp_path / "samples.txt"
    sample_list.write_text("c\na\n", encoding="utf-8")

    selected = load_manifest(manifest, limit=10, selection_rank_start=0, sample_list=sample_list)

    assert [row["sample_id"] for row in selected] == ["c", "a"]


def test_ab_mapping_is_reproducible_and_blind() -> None:
    first = ab_mapping("gsm8k_test_000060", 20260711)
    second = ab_mapping("gsm8k_test_000060", 20260711)

    assert first == second
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {"full_system", "no_teaching_plan"}


def test_frame_times_keep_final_scene_and_respect_max_frames() -> None:
    eds = {
        "scenes": [
            {"scene_id": f"scene_{index}", "start_ms": index * 1000, "duration_ms": 1000}
            for index in range(12)
        ],
        "formula_tracks": [
            {"scene_id": f"scene_{index}", "start_ms": index * 1000 + 100, "duration_ms": 600}
            for index in range(12)
        ],
    }

    picks = frame_times_from_eds(eds, max_frames=8)

    assert len(picks) == 8
    assert picks[0]["scene_index"] == 0
    assert picks[-1]["scene_index"] == 11
    assert all(pick["reason"] == "formula_complete" for pick in picks)


def test_parse_json_object_accepts_fenced_json() -> None:
    parsed = parse_json_object('```json\n{"overall_preference":"A"}\n```')

    assert parsed == {"overall_preference": "A"}


def test_winner_to_system_uses_ab_mapping() -> None:
    mapping = {"A": "no_teaching_plan", "B": "full_system"}

    assert winner_to_system("A", mapping) == "no_teaching_plan"
    assert winner_to_system("B", mapping) == "full_system"
    assert winner_to_system("tie", mapping) == "tie"


def test_prompt_mentions_required_schema_fields() -> None:
    for field in [
        "formula_source_clarity",
        "step_completeness",
        "narration_clarity",
        "self_learning_suitability",
        "overall_preference",
    ]:
        assert field in JUDGE_PROMPT


def test_env_reader_prefers_first_non_placeholder_duplicate(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=real-key\nDASHSCOPE_API_KEY=your_dashscope_key_here\n",
        encoding="utf-8",
    )

    values = env_values_first_non_placeholder(env_file)

    assert values["DASHSCOPE_API_KEY"] == "real-key"


def test_judge_configs_default_enable_glm_and_qwen(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_GLM_MODEL", "glm-4.6v")
    monkeypatch.setenv("JUDGE_GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setenv("JUDGE_GLM_API_KEY_ENV", "ZAI_API_KEY")
    monkeypatch.setenv("ZAI_API_KEY", "glm-key")
    monkeypatch.setenv("JUDGE_QWEN_MODEL", "qwen3.6-vl-plus")
    monkeypatch.setenv("JUDGE_QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("JUDGE_QWEN_API_KEY_ENV", "DASHSCOPE_API_KEY")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")

    args = argparse.Namespace(judge_glm=True, judge_qwen=True, judge_doubao=False, judge_kimi=False)
    configs = judge_configs(args)

    assert [config.name for config in configs] == ["glm", "qwen"]


def test_judge_configs_support_kimi(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_KIMI_MODEL", "kimi-k2.6")
    monkeypatch.setenv("JUDGE_KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setenv("JUDGE_KIMI_API_KEY_ENV", "KIMI_API_KEY")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key")

    args = argparse.Namespace(judge_glm=False, judge_qwen=False, judge_doubao=False, judge_kimi=True)
    configs = judge_configs(args)

    assert [config.name for config in configs] == ["kimi"]
    assert configs[0].model == "kimi-k2.6"
    assert configs[0].temperature == 1.0


def test_selected_judges_respects_count_and_provider_order() -> None:
    assert selected_judges("glm,qwen,doubao,kimi", 2) == ["glm", "qwen"]
    assert selected_judges("qwen,glm,kimi", 1) == ["qwen"]
    assert selected_judges("glm,glm,qwen", 3) == ["glm", "qwen"]
    assert selected_judges(DEFAULT_JUDGE_PROVIDERS, DEFAULT_JUDGE_COUNT) == ["qwen", "glm", "kimi"]


def test_namespace_for_judges_maps_to_existing_config_flags() -> None:
    args = namespace_for_judges(["glm", "qwen"])

    assert args.judge_glm is True
    assert args.judge_qwen is True
    assert args.judge_doubao is False
    assert args.judge_kimi is False


def test_generate_video_args_support_explicit_num_samples_alias() -> None:
    args = parse_generate_video_args(["--num-samples", "50"])

    assert args.limit == 50


def test_generate_video_args_default_to_configured_sample_count() -> None:
    args = parse_generate_video_args([])

    assert args.limit == DEFAULT_GENERATE_NUM_SAMPLES
    assert args.output_dir == DEFAULT_SAVE_OUTPUT_DIR


def test_judge_experiment_args_support_explicit_num_samples_alias() -> None:
    args = parse_judge_experiment_args(["--num-samples", "50", "--judge-count", "2"])

    assert args.limit == 50
    assert args.judge_count == 2


def test_judge_experiment_args_default_to_formal_sample_count() -> None:
    args = parse_judge_experiment_args([])

    assert args.limit == 50
    assert args.judge_count == 3
    assert args.judge_providers == DEFAULT_JUDGE_PROVIDERS
    assert args.judge_max_tokens == DEFAULT_JUDGE_MAX_TOKENS
