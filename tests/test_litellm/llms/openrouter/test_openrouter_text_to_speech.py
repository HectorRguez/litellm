import json
from pathlib import Path

import httpx
import pytest
import respx
from openai import AsyncOpenAI, RateLimitError

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent


@respx.mock
def test_openrouter_speech_uses_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.post("https://openrouter.test/api/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=b"sync-audio",
            headers={"Content-Type": "audio/mpeg"},
        )
    )
    monkeypatch.setenv("OR_SITE_URL", "https://tests.litellm.ai")
    monkeypatch.setenv("OR_APP_NAME", "LiteLLM Tests")
    monkeypatch.setattr(
        litellm,
        "headers",
        {"X-Global": "global", "X-Precedence": "global"},
    )

    response = litellm.speech(
        model="openrouter/mistralai/voxtral-mini-tts-2603",
        input="hello",
        voice="alloy",
        api_key="test-key",
        api_base="https://openrouter.test/api/v1/",
        response_format="mp3",
        speed=1,
        instructions="Do not forward this",
        headers={"X-Public": "public", "X-Precedence": "public"},
        extra_headers={"X-Extra": "extra", "X-Precedence": "extra"},
        extra_body={"provider": {"order": ["preferred-provider"]}},
    )

    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.response.content == b"sync-audio"
    assert route.call_count == 1
    request = route.calls[0].request
    assert json.loads(request.content) == {
        "model": "mistralai/voxtral-mini-tts-2603",
        "input": "hello",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
        "provider": {"order": ["preferred-provider"]},
    }
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["http-referer"] == "https://tests.litellm.ai"
    assert request.headers["x-title"] == "LiteLLM Tests"
    assert request.headers["x-global"] == "global"
    assert request.headers["x-public"] == "public"
    assert request.headers["x-extra"] == "extra"
    assert request.headers["x-precedence"] == "extra"


@pytest.mark.asyncio
async def test_openrouter_aspeech_returns_binary_audio() -> None:
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            content=b"async-audio",
            headers={"Content-Type": "audio/pcm"},
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ),
    )

    try:
        response = await litellm.aspeech(
            model="openrouter/google/gemini-3.1-flash-tts-preview",
            input="hello async",
            voice="nova",
            client=client,
        )
    finally:
        await client.close()

    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.response.content == b"async-audio"
    assert len(captured_requests) == 1
    assert json.loads(captured_requests[0].content) == {
        "model": "google/gemini-3.1-flash-tts-preview",
        "input": "hello async",
        "voice": "nova",
    }


def test_openrouter_speech_requires_voice() -> None:
    with pytest.raises(litellm.BadRequestError, match="'voice' is required"):
        litellm.speech(
            model="openrouter/mistralai/voxtral-mini-tts-2603",
            input="hello",
            api_key="test-key",
        )


@respx.mock
def test_openrouter_speech_maps_rate_limit_errors() -> None:
    respx.post("https://openrouter.test/api/v1/audio/speech").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "capacity exhausted"}},
        )
    )

    with pytest.raises(RateLimitError, match="capacity exhausted"):
        litellm.speech(
            model="openrouter/mistralai/voxtral-mini-tts-2603",
            input="hello",
            voice="alloy",
            api_key="test-key",
            api_base="https://openrouter.test/api/v1",
            max_retries=0,
        )


def test_openrouter_endpoint_metadata_marks_speech_supported() -> None:
    repository_root = Path(__file__).parents[4]
    endpoint_metadata = json.loads((repository_root / "provider_endpoints_support.json").read_text())

    assert endpoint_metadata["providers"]["openrouter"]["endpoints"]["audio_speech"] is True
