import pytest

from mathexplain.config import PipelineModelConfig, UnsupportedModelProviderError
from mathexplain.services.deepseek import DeepSeekChatClient
from mathexplain.services.llm import DashScopeQwenTextClient, DashScopeQwenVisionClient


def test_pipeline_model_config_uses_defaults_without_env() -> None:
    config = PipelineModelConfig.from_sources(env={}, load_project_env=False)

    assert config.vision.provider == "qwen"
    assert config.vision.model == "qwen3.6-flash"
    assert config.vision.api_key_env == "DASHSCOPE_API_KEY"
    assert config.planner.provider == "deepseek"
    assert config.planner.model == "deepseek-v4-pro"
    assert config.planner.api_key_env == "DEEPSEEK_API_KEY"


def test_pipeline_model_config_cli_overrides_env() -> None:
    config = PipelineModelConfig.from_sources(
        vision_model="qwen-vl-plus",
        planner_model="deepseek-chat",
        env={
            "VISION_MODEL": "qwen-env-model",
            "PLANNER_MODEL": "deepseek-env-model",
            "VISION_API_KEY_ENV": "QWEN_KEY",
            "PLANNER_API_KEY_ENV": "DEEPSEEK_KEY",
        },
        load_project_env=False,
    )

    assert config.vision.model == "qwen-vl-plus"
    assert config.planner.model == "deepseek-chat"
    assert config.vision.api_key_env == "QWEN_KEY"
    assert config.planner.api_key_env == "DEEPSEEK_KEY"


def test_pipeline_model_config_rejects_unsupported_provider() -> None:
    with pytest.raises(UnsupportedModelProviderError, match="Currently supported: qwen"):
        PipelineModelConfig.from_sources(vision_provider="openai", env={}, load_project_env=False)

    with pytest.raises(UnsupportedModelProviderError, match="Currently supported: deepseek"):
        PipelineModelConfig.from_sources(planner_provider="openai", env={}, load_project_env=False)


def test_pipeline_model_config_supports_qwen_planner_defaults() -> None:
    config = PipelineModelConfig.from_sources(
        planner_provider="qwen",
        env={},
        load_project_env=False,
    )

    assert config.planner.provider == "qwen"
    assert config.planner.model == "qwen3.6-plus"
    assert config.planner.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.planner.api_key_env == "DASHSCOPE_API_KEY"


def test_pipeline_model_config_qwen_planner_prefers_planner_over_dashscope_env() -> None:
    config = PipelineModelConfig.from_sources(
        planner_provider="qwen",
        env={
            "PLANNER_MODEL": "qwen-planner-exp",
            "DASHSCOPE_MODEL": "qwen3.6-flash",
            "PLANNER_API_KEY_ENV": "QWEN_PLANNER_KEY",
        },
        load_project_env=False,
    )

    assert config.planner.provider == "qwen"
    assert config.planner.model == "qwen-planner-exp"
    assert config.planner.api_key_env == "QWEN_PLANNER_KEY"


def test_clients_read_custom_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_TEST_KEY", "fake-qwen-key")
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "fake-deepseek-key")

    qwen_client = DashScopeQwenVisionClient(
        api_key_env="QWEN_TEST_KEY",
        model="qwen-vl-plus",
        base_url="https://example.invalid/qwen",
    )
    deepseek_client = DeepSeekChatClient(
        api_key_env="DEEPSEEK_TEST_KEY",
        model="deepseek-chat",
        base_url="https://example.invalid/deepseek",
    )
    qwen_text_client = DashScopeQwenTextClient(
        api_key_env="QWEN_TEST_KEY",
        model="qwen3.6-plus",
        base_url="https://example.invalid/qwen",
    )

    assert qwen_client.provider == "qwen"
    assert qwen_client.model == "qwen-vl-plus"
    assert qwen_client.api_key_env == "QWEN_TEST_KEY"
    assert qwen_text_client.provider == "qwen"
    assert qwen_text_client.model == "qwen3.6-plus"
    assert qwen_text_client.api_key_env == "QWEN_TEST_KEY"
    assert qwen_text_client.timeout_seconds == 120
    assert qwen_text_client.max_retries == 0
    assert deepseek_client.provider == "deepseek"
    assert deepseek_client.model == "deepseek-chat"
    assert deepseek_client.api_key_env == "DEEPSEEK_TEST_KEY"
    assert deepseek_client.timeout_seconds == 120
    assert deepseek_client.max_retries == 0
