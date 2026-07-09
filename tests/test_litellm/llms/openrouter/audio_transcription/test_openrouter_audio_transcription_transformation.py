import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.llms.openrouter.audio_transcription.transformation import (
    OpenRouterAudioTranscriptionConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _clear_openrouter_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openrouter_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)


def test_provider_config_manager_registers_openrouter_audio_transcription_config() -> None:
    assert isinstance(
        ProviderConfigManager.get_provider_audio_transcription_config(
            model="openai/whisper-large-v3",
            provider=LlmProviders.OPENROUTER,
        ),
        OpenRouterAudioTranscriptionConfig,
    )


def test_get_supported_openai_params_returns_openrouter_transcription_params() -> None:
    assert get_supported_openai_params(
        model="openai/whisper-large-v3",
        custom_llm_provider="openrouter",
        request_type="transcription",
    ) == [
        "language",
        "prompt",
        "response_format",
        "temperature",
        "timestamp_granularities",
    ]


class TestOpenRouterAudioTranscriptionConfig:
    def setup_method(self) -> None:
        self.config = OpenRouterAudioTranscriptionConfig()

    def test_validate_environment_sets_multipart_safe_headers(self) -> None:
        headers = self.config.validate_environment(
            headers={"Content-Type": "application/json", "X-Custom": "value"},
            model="openai/whisper-large-v3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["HTTP-Referer"] == "https://litellm.ai"
        assert headers["X-Title"] == "liteLLM"
        assert headers["X-Custom"] == "value"
        assert "Content-Type" not in headers
        assert "content-type" not in headers

    def test_validate_environment_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_openrouter_keys(monkeypatch)

        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            self.config.validate_environment(
                headers={},
                model="openai/whisper-large-v3",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_get_complete_url(self) -> None:
        assert (
            self.config.get_complete_url(
                api_base="https://openrouter.ai/api/v1/",
                api_key="test-key",
                model="openai/whisper-large-v3",
                optional_params={},
                litellm_params={},
            )
            == "https://openrouter.ai/api/v1/audio/transcriptions"
        )

    def test_transform_audio_transcription_request_preserves_openai_fields(self) -> None:
        request_data = self.config.transform_audio_transcription_request(
            model="openai/whisper-large-v3",
            audio_file=("sample.wav", b"RIFF....", "audio/wav"),
            optional_params={
                "language": "en",
                "prompt": "Product narration",
                "response_format": "verbose_json",
                "temperature": 0,
                "timestamp_granularities": ["word"],
            },
            litellm_params={},
        )

        assert request_data.data["model"] == "openai/whisper-large-v3"
        assert request_data.data["language"] == "en"
        assert request_data.data["prompt"] == "Product narration"
        assert request_data.data["response_format"] == "verbose_json"
        assert request_data.data["temperature"] == 0
        assert request_data.data["timestamp_granularities[]"] == ["word"]
        assert request_data.files["file"][0] == "sample.wav"
        assert request_data.files["file"][1] == b"RIFF...."
        assert request_data.files["file"][2] == "audio/wav"

    def test_transform_audio_transcription_response_preserves_words(self) -> None:
        raw_response = httpx.Response(
            200,
            json={
                "text": "hello world",
                "language": "en",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ],
                "usage": {"duration": 1.0},
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        response = self.config.transform_audio_transcription_response(raw_response)

        assert response.text == "hello world"
        assert response["language"] == "en"
        assert response["words"][0] == {"word": "hello", "start": 0.0, "end": 0.4}
        assert response["usage"] == {"duration": 1.0}
        assert response._hidden_params["words"][1]["word"] == "world"

    def test_transform_audio_transcription_response_raises_openrouter_error(self) -> None:
        raw_response = httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        with pytest.raises(OpenRouterException, match="bad key"):
            self.config.transform_audio_transcription_response(raw_response)
