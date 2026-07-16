import json
import wave
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openrouter.audio_transcription.transformation import (
    OpenRouterAudioTranscriptionConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.types.utils import (
    LlmProviders,
    TranscriptionUsageTokensObject,
)
from litellm.utils import ProviderConfigManager

MODEL = "openai/gpt-audio-mini"


def _short_wav() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def _transport(
    captured_requests: list[httpx.Request],
    response_json: dict,
    status_code: int = 200,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(status_code, json=response_json)

    return httpx.MockTransport(handle)


def _multipart_fields(request: httpx.Request) -> tuple[dict[str, str], tuple[str, bytes, str]]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + request.headers["content-type"].encode()
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + request.content
    )
    fields: dict[str, str] = {}
    uploaded_file: tuple[str, bytes, str] | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        assert isinstance(name, str)
        assert isinstance(payload, bytes)
        if part.get_filename() is None:
            fields[name] = payload.decode()
        else:
            uploaded_file = (part.get_filename() or "", payload, part.get_content_type())
    assert uploaded_file is not None
    return fields, uploaded_file


def test_openrouter_audio_transcription_is_registered() -> None:
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model=MODEL,
        provider=LlmProviders.OPENROUTER,
    )
    assert isinstance(config, OpenRouterAudioTranscriptionConfig)

    repository_root = Path(__file__).resolve().parents[5]
    metadata = json.loads((repository_root / "provider_endpoints_support.json").read_text())
    assert metadata["providers"]["openrouter"]["endpoints"]["audio_transcriptions"] is True


def test_public_transcription_sends_multipart_and_maps_usage() -> None:
    audio_bytes = _short_wav()
    captured_requests: list[httpx.Request] = []
    provider_response = {
        "text": "A concise transcript.",
        "language": "en",
        "words": [{"word": "concise", "start": 0.2, "end": 0.5}],
        "usage": {
            "cost": 0.000508,
            "input_tokens": 83,
            "output_tokens": 30,
            "seconds": 9.2,
            "total_tokens": 113,
        },
    }
    with httpx.Client(transport=_transport(captured_requests, provider_response)) as http_client:
        response = litellm.transcription(
            model=f"openrouter/{MODEL}",
            file=("sample.wav", audio_bytes, "audio/wav"),
            language="en",
            response_format="verbose_json",
            temperature=0,
            timestamp_granularities=["word"],
            api_key="test-key",
            extra_headers={"X-Custom-STT": "sync", "Content-Type": "application/json"},
            client=HTTPHandler(client=http_client),
            caching=False,
        )

    request = captured_requests[0]
    assert request.url == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-custom-stt"] == "sync"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    fields, uploaded_file = _multipart_fields(request)
    assert fields == {
        "model": MODEL,
        "language": "en",
        "response_format": "verbose_json",
        "temperature": "0",
        "timestamp_granularities[]": "word",
    }
    assert uploaded_file == ("sample.wav", audio_bytes, "audio/wav")
    assert response.text == "A concise transcript."
    assert response["words"][0]["word"] == "concise"
    assert isinstance(response.usage, TranscriptionUsageTokensObject)
    assert response.usage.input_tokens == 83
    assert response.usage.output_tokens == 30
    assert response.usage.total_tokens == 113
    assert response._hidden_params["audio_transcription_duration"] == 9.2
    assert get_response_cost_from_hidden_params(response._hidden_params) == 0.000508


@pytest.mark.asyncio
async def test_public_atranscription_accepts_text_only_response() -> None:
    captured_requests: list[httpx.Request] = []
    async_handler = AsyncHTTPHandler()
    await async_handler.client.aclose()
    async_handler.client = httpx.AsyncClient(transport=_transport(captured_requests, {"text": "Usage is optional."}))
    try:
        response = await litellm.atranscription(
            model=f"openrouter/{MODEL}",
            file=("sample.wav", _short_wav(), "audio/wav"),
            api_key="test-key",
            extra_headers={"X-Custom-STT": "async"},
            client=async_handler,
            caching=False,
        )
    finally:
        await async_handler.close()

    assert captured_requests[0].headers["x-custom-stt"] == "async"
    assert response.text == "Usage is optional."
    assert response.usage is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompt": "unsupported"}, "not supported by openrouter"),
        ({"response_format": "text"}, "response_format"),
    ],
)
def test_public_transcription_rejects_undocumented_params(kwargs: dict, match: str) -> None:
    with pytest.raises(UnsupportedParamsError, match=match):
        litellm.transcription(
            model=f"openrouter/{MODEL}",
            file=("sample.wav", b"RIFF....", "audio/wav"),
            api_key="test-key",
            caching=False,
            **kwargs,
        )


def test_openrouter_transcription_response_allows_partial_usage() -> None:
    config = OpenRouterAudioTranscriptionConfig()
    raw_response = httpx.Response(
        200,
        json={"text": "Duration usage.", "usage": {"seconds": 4.5}},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
    )

    response = config.transform_audio_transcription_response(raw_response)

    assert response.text == "Duration usage."
    assert response.usage is not None
    assert response.usage.type == "duration"
    assert response.usage.seconds == 4.5
    assert response._hidden_params["audio_transcription_duration"] == 4.5
    assert get_response_cost_from_hidden_params(response._hidden_params) is None


def test_openrouter_transcription_response_maps_provider_error() -> None:
    config = OpenRouterAudioTranscriptionConfig()
    raw_response = httpx.Response(
        401,
        json={"error": {"message": "bad key"}},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
    )

    with pytest.raises(OpenRouterException, match="bad key"):
        config.transform_audio_transcription_response(raw_response)
