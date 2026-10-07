import json
import math
import struct
import wave
from io import BytesIO
from pathlib import Path

from mathexplain.agents.tts import TTSProvider
from mathexplain.schemas.eds import ExecutableDirectorScript
from mathexplain.schemas.tts import TTSClip, TTSResult, validate_tts_result_dict
from mathexplain.services.tts import VolcengineTTSClient
from test_renderer import renderer_eds_data


class FakeTTSClient:
    provider = "fake"
    resource_id = "fake-resource"
    voice_type = "fake-voice"
    audio_format = "wav"
    sample_rate = 24000
    speed_ratio = 1.0

    def __init__(self, duration_ms: int = 1600) -> None:
        self.duration_ms = duration_ms
        self.requests: list[str] = []

    def synthesize(self, text: str, *, request_id: str | None = None) -> bytes:
        self.requests.append(text)
        return make_wav(self.duration_ms, self.sample_rate)


class FakeFallbackEnglishTTSClient(FakeTTSClient):
    voice_selection = {
        "voice_type": "fallback_voice",
        "source": "TTS_VOICE_TYPE",
        "fallback": True,
    }


def make_wav(duration_ms: int, sample_rate: int = 24000) -> bytes:
    frame_count = round(duration_ms * sample_rate / 1000)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        for index in range(frame_count):
            sample = int(1200 * math.sin(2 * math.pi * 440 * index / sample_rate))
            writer.writeframes(struct.pack("<h", sample))
    return buffer.getvalue()


def test_tts_result_schema_roundtrip() -> None:
    result = TTSResult(
        status="passed",
        provider="fake",
        resource_id="fake-resource",
        voice_type="fake-voice",
        clip_count=0,
        duration_ms=0,
    )

    loaded = validate_tts_result_dict(result.model_dump(mode="json"))

    assert loaded.status == "passed"
    assert loaded.clips == []


def test_tts_provider_generates_clips_report_and_retimed_eds(tmp_path: Path) -> None:
    eds = ExecutableDirectorScript.model_validate(renderer_eds_data())
    client = FakeTTSClient(duration_ms=1600)

    result = TTSProvider(client=client).synthesize(eds, output_dir=tmp_path)

    assert result.status == "passed"
    assert result.clip_count == 2
    assert len(client.requests) == 2
    assert result.tts_report_path is not None
    assert Path(result.tts_report_path).exists()
    assert result.audio_track_path is not None
    assert Path(result.audio_track_path).exists()
    assert result.retimed_eds_path is not None
    assert result.metadata["speed_ratio"] == 1.0
    retimed = json.loads(Path(result.retimed_eds_path).read_text(encoding="utf-8"))
    assert retimed["metadata"]["tts_retimed"] is True
    assert retimed["scenes"][0]["duration_ms"] >= 1600
    assert retimed["narration_tracks"][0]["duration_ms"] == 1600
    assert retimed["scenes"][1]["start_ms"] == retimed["scenes"][0]["duration_ms"]


def test_tts_provider_warns_when_english_uses_fallback_voice(tmp_path: Path) -> None:
    data = renderer_eds_data()
    data["render_hints"]["language"] = "en-US"
    eds = ExecutableDirectorScript.model_validate(data)

    result = TTSProvider(client=FakeFallbackEnglishTTSClient()).synthesize(eds, output_dir=tmp_path)

    assert result.status == "passed"
    assert any("TTS_VOICE_TYPE_EN" in warning for warning in result.warnings)
    assert result.metadata["voice_selection"]["fallback"] is True


def test_tts_retime_caps_answer_scene_tail() -> None:
    data = renderer_eds_data()
    data["scenes"][1]["duration_ms"] = 12500
    data["timeline"][2]["duration_ms"] = 11600
    data["formula_tracks"][1]["duration_ms"] = 11600
    eds = ExecutableDirectorScript.model_validate(data)
    clips = [
        TTSClip(
            clip_id="clip_1",
            narration_id="nar_1",
            scene_id="scene_1",
            text="setup",
            audio_path="setup.wav",
            start_ms=0,
            duration_ms=1600,
            original_start_ms=0,
            original_duration_ms=1200,
        ),
        TTSClip(
            clip_id="clip_2",
            narration_id="nar_2",
            scene_id="scene_2",
            text="answer",
            audio_path="answer.wav",
            start_ms=1200,
            duration_ms=6000,
            original_start_ms=1200,
            original_duration_ms=1000,
        ),
    ]

    retimed, _, total_duration_ms = TTSProvider._retime_eds(eds, clips)
    answer_scene = next(scene for scene in retimed.scenes if scene.scene_type == "answer")
    answer_formula = next(formula for formula in retimed.formula_tracks if formula.scene_id == answer_scene.scene_id)

    assert answer_scene.duration_ms == 8500
    assert answer_formula.duration_ms <= answer_scene.duration_ms
    assert total_duration_ms == retimed.metadata["total_duration_ms"]


def test_tts_provider_failed_eds_writes_failed_report(tmp_path: Path) -> None:
    data = renderer_eds_data()
    data["quality_report"]["valid"] = False
    data["quality_report"]["status"] = "failed"
    eds = ExecutableDirectorScript.model_validate(data)

    result = TTSProvider(client=FakeTTSClient()).synthesize(eds, output_dir=tmp_path)

    assert result.status == "failed"
    assert result.clip_count == 0
    assert result.errors
    assert result.tts_report_path is not None
    assert Path(result.tts_report_path).exists()


def test_volcengine_tts_client_prefers_service_credentials_in_auto_mode(monkeypatch) -> None:
    monkeypatch.setenv("TTS_APP_ID", "appid-demo")
    monkeypatch.setenv("TTS_ACCESS_TOKEN", "token-demo")
    monkeypatch.setenv("TTS_API_KEY", "wrong-api-key")
    monkeypatch.setenv("TTS_VOICE_TYPE", "zh_male_shaonianzixin_moon_bigtts")

    client = VolcengineTTSClient()

    assert client.auth_mode == "app_access"
    assert client.app_id == "appid-demo"
    assert client.access_token == "token-demo"


def test_volcengine_tts_client_can_use_new_api_key_mode(monkeypatch) -> None:
    monkeypatch.delenv("TTS_APP_ID", raising=False)
    monkeypatch.delenv("TTS_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("TTS_API_KEY", "api-key-demo")
    monkeypatch.setenv("TTS_AUTH_MODE", "api_key")
    monkeypatch.setenv("TTS_VOICE_TYPE", "zh_male_shaonianzixin_moon_bigtts")

    client = VolcengineTTSClient()

    assert client.auth_mode == "api_key"
    assert client.api_key == "api-key-demo"


def test_volcengine_tts_client_reads_speed_ratio(monkeypatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", "api-key-demo")
    monkeypatch.setenv("TTS_AUTH_MODE", "api_key")
    monkeypatch.setenv("TTS_VOICE_TYPE", "zh_male_shaonianzixin_moon_bigtts")
    monkeypatch.setenv("TTS_SPEED_RATIO", "0.85")

    client = VolcengineTTSClient()

    assert client.speed_ratio == 0.85


def test_volcengine_tts_client_selects_english_voice(monkeypatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", "api-key-demo")
    monkeypatch.setenv("TTS_AUTH_MODE", "api_key")
    monkeypatch.setenv("TTS_VOICE_TYPE", "fallback_voice")
    monkeypatch.setenv("TTS_VOICE_TYPE_EN", "english_voice")

    client = VolcengineTTSClient(language="en-US")

    assert client.voice_type == "english_voice"
    assert client.voice_selection["source"] == "TTS_VOICE_TYPE_EN"


def test_volcengine_tts_client_falls_back_to_default_voice(monkeypatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", "api-key-demo")
    monkeypatch.setenv("TTS_AUTH_MODE", "api_key")
    monkeypatch.setenv("TTS_VOICE_TYPE", "fallback_voice")
    monkeypatch.delenv("TTS_VOICE_TYPE_EN", raising=False)

    client = VolcengineTTSClient(language="en-US")

    assert client.voice_type == "fallback_voice"
    assert client.voice_selection["fallback"] is True
