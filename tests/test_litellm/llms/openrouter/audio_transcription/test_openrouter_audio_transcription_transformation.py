import json
import wave
from collections.abc import Mapping
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.exceptions import RateLimitError, UnsupportedParamsError
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openrouter.audio_transcription.transformation import (
    OpenRouterAudioTranscriptionConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.types.utils import (
    LlmProviders,
    TranscriptionUsageDurationObject,
    TranscriptionUsageTokensObject,
)
from litellm.utils import ProviderConfigManager


@pytest.fixture
def openrouter_stt_response() -> Mapping[str, object]:
    return {
        "text": "Hello, this is a test of speech-to-text transcription.",
        "task": "transcribe",
        "language": "en",
        "duration": 9.2,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 9.2,
                "text": "Hello, this is a test of speech-to-text transcription.",
            }
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "transcription", "start": 8.4, "end": 9.2},
        ],
        "usage": {
            "cost": 0.000508,
            "input_tokens": 83,
            "output_tokens": 30,
            "seconds": 9.2,
            "total_tokens": 113,
        },
    }


def _clear_openrouter_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openrouter_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)


def _short_wav() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def _capturing_transport(
    captured_requests: list[httpx.Request],
    response_json: Mapping[str, object],
    status_code: int = 200,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            status_code,
            json=response_json,
            headers={"X-Generation-Id": "gen-test-123"},
        )

    return httpx.MockTransport(handle)


def _parse_multipart_request(request: httpx.Request) -> tuple[dict[str, str], tuple[str, bytes, str]]:
    content_type = request.headers["content-type"]
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
    )
    fields: dict[str, str] = {}
    uploaded_file: tuple[str, bytes, str] | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        filename = part.get_filename()
        assert isinstance(name, str)
        assert isinstance(payload, bytes)
        if filename is None:
            fields[name] = payload.decode()
        else:
            uploaded_file = (filename, payload, part.get_content_type())
    assert uploaded_file is not None
    return fields, uploaded_file


def _assert_normalized_usage(response: object) -> None:
    assert isinstance(response, litellm.TranscriptionResponse)
    assert isinstance(response.usage, TranscriptionUsageTokensObject)
    assert response.usage.type == "tokens"
    assert response.usage.input_tokens == 83
    assert response.usage.output_tokens == 30
    assert response.usage.total_tokens == 113
    assert response.usage.input_token_details.audio_tokens == 83
    assert response.usage.input_token_details.text_tokens == 0
    assert response._hidden_params["audio_transcription_duration"] == 9.2
    assert response._hidden_params["usage"]["seconds"] == 9.2
    assert get_response_cost_from_hidden_params(response._hidden_params) == 0.000508


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


def test_provider_endpoint_metadata_marks_openrouter_transcriptions_supported() -> None:
    repository_root = Path(__file__).resolve().parents[5]
    metadata = json.loads((repository_root / "provider_endpoints_support.json").read_text())
    assert metadata["providers"]["openrouter"]["endpoints"]["audio_transcriptions"] is True


class TestOpenRouterAudioTranscriptionConfig:
    def setup_method(self) -> None:
        self.config = OpenRouterAudioTranscriptionConfig()

    def test_validate_environment_sets_multipart_safe_headers(self) -> None:
        headers = self.config.validate_environment(
            headers={"cOnTeNt-TyPe": "application/json", "X-Custom": "value"},
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
        assert all(key.lower() != "content-type" for key in headers)

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

    def test_transform_audio_transcription_request_preserves_provider_fields_only(self) -> None:
        request_data = self.config.transform_audio_transcription_request(
            model="openai/whisper-large-v3",
            audio_file=("sample.wav", b"RIFF....", "audio/wav"),
            optional_params={
                "language": "en",
                "prompt": "Product narration",
                "response_format": "verbose_json",
                "temperature": 0,
                "timestamp_granularities": ["word"],
                "provider": "groq",
                "extra_headers": {"X-Custom": "value"},
                "api_key": "must-not-leak",
                "litellm_call_id": "must-not-leak",
            },
            litellm_params={},
        )

        assert request_data.data["model"] == "openai/whisper-large-v3"
        assert request_data.data["language"] == "en"
        assert request_data.data["prompt"] == "Product narration"
        assert request_data.data["response_format"] == "verbose_json"
        assert request_data.data["temperature"] == 0
        assert request_data.data["timestamp_granularities[]"] == ["word"]
        assert request_data.data["provider"] == "groq"
        assert "extra_headers" not in request_data.data
        assert "api_key" not in request_data.data
        assert "litellm_call_id" not in request_data.data
        assert request_data.files["file"] == ("sample.wav", b"RIFF....", "audio/wav")

    @pytest.mark.parametrize("response_format", ["json", "verbose_json"])
    def test_map_openai_params_accepts_documented_response_formats(self, response_format: str) -> None:
        assert self.config.map_openai_params(
            non_default_params={"response_format": response_format, "prompt": "Accepted but ignored"},
            optional_params={},
            model="openai/whisper-large-v3",
            drop_params=False,
        ) == {"response_format": response_format, "prompt": "Accepted but ignored"}

    @pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
    def test_map_openai_params_rejects_unsupported_response_formats(self, response_format: str) -> None:
        with pytest.raises(UnsupportedParamsError, match=response_format):
            self.config.map_openai_params(
                non_default_params={"response_format": response_format},
                optional_params={},
                model="openai/whisper-large-v3",
                drop_params=False,
            )

    @pytest.mark.parametrize("drop_params", [True, False])
    def test_map_openai_params_drops_unsupported_response_format(
        self,
        drop_params: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(litellm, "drop_params", not drop_params)
        assert self.config.map_openai_params(
            non_default_params={"response_format": "text", "prompt": "Accepted but ignored"},
            optional_params={},
            model="openai/whisper-large-v3",
            drop_params=drop_params,
        ) == {"prompt": "Accepted but ignored"}

    def test_transform_audio_transcription_response_normalizes_usage_and_preserves_fields(
        self,
        openrouter_stt_response: Mapping[str, object],
    ) -> None:
        raw_response = httpx.Response(
            200,
            json=openrouter_stt_response,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        response = self.config.transform_audio_transcription_response(raw_response)

        assert response.text == "Hello, this is a test of speech-to-text transcription."
        assert response["task"] == "transcribe"
        assert response["language"] == "en"
        assert response["duration"] == 9.2
        assert response["segments"][0]["end"] == 9.2
        assert response["words"][1]["word"] == "transcription"
        assert response._hidden_params["text"] == response.text
        _assert_normalized_usage(response)

    def test_transform_audio_transcription_response_repairs_zero_duration_word_timing(
        self,
        openrouter_stt_response: Mapping[str, object],
    ) -> None:
        provider_response = {
            **openrouter_stt_response,
            "words": [
                {"word": "paso", "start": 2.0, "end": 2.94},
                {"word": "paso", "start": 2.94, "end": 2.94},
                {"word": "siguiente", "start": 3.6, "end": 4.1},
                {"word": "final", "start": 4.1, "end": 4.1},
            ],
        }
        raw_response = httpx.Response(
            200,
            json=provider_response,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        response = self.config.transform_audio_transcription_response(raw_response)

        assert response["words"] == (
            {"word": "paso", "start": 2.0, "end": 2.94},
            {"word": "paso", "start": 2.94, "end": 3.6},
            {"word": "siguiente", "start": 3.6, "end": 4.1},
        )
        assert response._hidden_params["words"] == provider_response["words"]

    @pytest.mark.parametrize(
        ("provider_usage", "expected_usage_type"),
        [
            ({"seconds": 18, "cost": 0.0048}, TranscriptionUsageDurationObject),
            (
                {"total_tokens": 239, "input_tokens": 175, "output_tokens": 64, "cost": 0.00053875},
                TranscriptionUsageTokensObject,
            ),
        ],
    )
    def test_transform_audio_transcription_response_supports_live_usage_variants(
        self,
        provider_usage: Mapping[str, object],
        expected_usage_type: type[TranscriptionUsageDurationObject] | type[TranscriptionUsageTokensObject],
    ) -> None:
        raw_response = httpx.Response(
            200,
            json={"text": "Four score and seven years ago", "usage": provider_usage},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        response = self.config.transform_audio_transcription_response(raw_response)

        assert isinstance(response.usage, expected_usage_type)
        assert get_response_cost_from_hidden_params(response._hidden_params) == provider_usage["cost"]
        if isinstance(response.usage, TranscriptionUsageDurationObject):
            assert response.usage.seconds == provider_usage["seconds"]
            assert response._hidden_params["audio_transcription_duration"] == provider_usage["seconds"]
        else:
            assert response.usage.input_tokens == provider_usage["input_tokens"]
            assert response.usage.output_tokens == provider_usage["output_tokens"]
            assert response.usage.total_tokens == provider_usage["total_tokens"]
            assert "audio_transcription_duration" not in response._hidden_params

    def test_transform_audio_transcription_response_raises_openrouter_error(self) -> None:
        raw_response = httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        with pytest.raises(OpenRouterException, match="bad key"):
            self.config.transform_audio_transcription_response(raw_response)


def test_public_transcription_sends_multipart_headers_and_normalizes_response(
    openrouter_stt_response: Mapping[str, object],
) -> None:
    audio_bytes = _short_wav()
    captured_requests: list[httpx.Request] = []
    transport = _capturing_transport(captured_requests, openrouter_stt_response)
    with httpx.Client(transport=transport) as http_client:
        response = litellm.transcription(
            model="openrouter/openai/whisper-large-v3",
            file=("sample.wav", audio_bytes, "audio/wav"),
            language="en",
            prompt="OpenRouter, LiteLLM",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            api_key="test-key",
            extra_headers={"X-Custom-STT": "sync", "cOnTeNt-TyPe": "application/json"},
            client=HTTPHandler(client=http_client),
            caching=False,
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-custom-stt"] == "sync"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    fields, uploaded_file = _parse_multipart_request(request)
    assert fields == {
        "model": "openai/whisper-large-v3",
        "language": "en",
        "prompt": "OpenRouter, LiteLLM",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    assert "extra_headers" not in fields
    assert uploaded_file == ("sample.wav", audio_bytes, "audio/wav")
    assert response["language"] == "en"
    assert response["words"][0]["word"] == "Hello"
    _assert_normalized_usage(response)


@pytest.mark.asyncio
async def test_public_atranscription_sends_multipart_headers_and_normalizes_response(
    openrouter_stt_response: Mapping[str, object],
) -> None:
    audio_bytes = _short_wav()
    captured_requests: list[httpx.Request] = []
    async_handler = AsyncHTTPHandler()
    await async_handler.client.aclose()
    async_handler.client = httpx.AsyncClient(transport=_capturing_transport(captured_requests, openrouter_stt_response))
    try:
        response = await litellm.atranscription(
            model="openrouter/openai/whisper-large-v3",
            file=("sample.wav", audio_bytes, "audio/wav"),
            prompt="Accepted but ignored",
            response_format="verbose_json",
            api_key="test-key",
            extra_headers={"X-Custom-STT": "async", "CONTENT-TYPE": "text/plain"},
            client=async_handler,
            caching=False,
        )
    finally:
        await async_handler.close()

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.headers["x-custom-stt"] == "async"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    fields, uploaded_file = _parse_multipart_request(request)
    assert fields["prompt"] == "Accepted but ignored"
    assert "extra_headers" not in fields
    assert uploaded_file == ("sample.wav", audio_bytes, "audio/wav")
    _assert_normalized_usage(response)


@pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
def test_public_transcription_rejects_unsupported_response_format(response_format: str) -> None:
    captured_requests: list[httpx.Request] = []
    transport = _capturing_transport(captured_requests, {"text": "must not be returned"})
    with httpx.Client(transport=transport) as http_client:
        with pytest.raises(UnsupportedParamsError, match=response_format):
            litellm.transcription(
                model="openrouter/openai/whisper-large-v3",
                file=("sample.wav", b"RIFF....", "audio/wav"),
                response_format=response_format,
                api_key="test-key",
                client=HTTPHandler(client=http_client),
                caching=False,
            )
    assert captured_requests == []


@pytest.mark.parametrize("use_global_drop_params", [False, True])
def test_public_transcription_drops_unsupported_response_format(
    use_global_drop_params: bool,
    monkeypatch: pytest.MonkeyPatch,
    openrouter_stt_response: Mapping[str, object],
) -> None:
    monkeypatch.setattr(litellm, "drop_params", use_global_drop_params)
    captured_requests: list[httpx.Request] = []
    transport = _capturing_transport(captured_requests, openrouter_stt_response)
    with httpx.Client(transport=transport) as http_client:
        litellm.transcription(
            model="openrouter/openai/whisper-large-v3",
            file=("sample.wav", b"RIFF....", "audio/wav"),
            response_format="text",
            drop_params=not use_global_drop_params,
            api_key="test-key",
            client=HTTPHandler(client=http_client),
            caching=False,
        )
    fields, _ = _parse_multipart_request(captured_requests[0])
    assert "response_format" not in fields


def test_public_transcription_maps_openrouter_authentication_error() -> None:
    captured_requests: list[httpx.Request] = []
    transport = _capturing_transport(captured_requests, {"error": {"message": "bad key"}}, status_code=401)
    with httpx.Client(transport=transport) as http_client:
        with pytest.raises(OpenRouterException, match="bad key") as exc_info:
            litellm.transcription(
                model="openrouter/openai/whisper-large-v3",
                file=("sample.wav", b"RIFF....", "audio/wav"),
                api_key="test-key",
                client=HTTPHandler(client=http_client),
                caching=False,
            )
    assert exc_info.value.status_code == 401
    assert len(captured_requests) == 1


@pytest.mark.asyncio
async def test_public_atranscription_maps_openrouter_rate_limit_error() -> None:
    captured_requests: list[httpx.Request] = []
    async_handler = AsyncHTTPHandler()
    await async_handler.client.aclose()
    async_handler.client = httpx.AsyncClient(
        transport=_capturing_transport(
            captured_requests,
            {"error": {"message": "rate limit exceeded"}},
            status_code=429,
        )
    )
    try:
        with pytest.raises(RateLimitError, match="rate limit exceeded"):
            await litellm.atranscription(
                model="openrouter/openai/whisper-large-v3",
                file=("sample.wav", b"RIFF....", "audio/wav"),
                api_key="test-key",
                client=async_handler,
                caching=False,
            )
    finally:
        await async_handler.close()
    assert len(captured_requests) == 1
