import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.text_to_speech import transformation
from litellm.llms.openrouter.text_to_speech.transformation import (
    OpenRouterTextToSpeechConfig,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_GENERATION_ID = "gen-tts-test-123"
_PROVIDER_COST = 0.002144


def _openrouter_tts_response(request: httpx.Request, audio: bytes) -> httpx.Response:
    if request.url.path.endswith("/generation"):
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": _GENERATION_ID,
                    "model": "google/gemini-3.1-flash-tts-preview",
                    "total_cost": _PROVIDER_COST,
                }
            },
        )
    return httpx.Response(
        200,
        content=audio,
        headers={"X-Generation-Id": _GENERATION_ID},
    )


@pytest.fixture(autouse=True)
def _deterministic_openrouter_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_SITE_URL", "https://tests.litellm.ai")
    monkeypatch.setenv("OR_APP_NAME", "LiteLLM Tests")


def _clear_openrouter_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openrouter_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)


def test_provider_config_manager_registers_openrouter_text_to_speech_config() -> None:
    assert isinstance(
        ProviderConfigManager.get_provider_text_to_speech_config(
            model="hexgrad/kokoro-82m",
            provider=LlmProviders.OPENROUTER,
        ),
        OpenRouterTextToSpeechConfig,
    )


def test_provider_endpoint_metadata_marks_openrouter_speech_supported() -> None:
    repository_root = Path(__file__).parents[5]
    endpoint_metadata = json.loads((repository_root / "provider_endpoints_support.json").read_text())

    assert endpoint_metadata["providers"]["openrouter"]["endpoints"]["audio_speech"] is True


def test_litellm_speech_uses_shared_handler_with_openai_compatible_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return _openrouter_tts_response(request, b"sync-audio")

    transport_client = httpx.Client(transport=httpx.MockTransport(respond))
    client = HTTPHandler(client=transport_client)
    global_headers = {"X-Global": "global", "X-Precedence": "global"}
    public_headers = {"X-Public": "public", "X-Precedence": "public"}
    extra_headers = {"X-Extra": "extra", "X-Precedence": "extra"}
    original_global_headers = global_headers.copy()
    original_public_headers = public_headers.copy()
    original_extra_headers = extra_headers.copy()
    monkeypatch.setattr(litellm, "headers", global_headers)

    try:
        response = litellm.speech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello",
            voice="af_alloy",
            api_key="test-key",
            api_base="https://openrouter.test/api/v1/",
            headers=public_headers,
            extra_headers=extra_headers,
            extra_body={"provider": {"order": ["preferred-provider"]}},
            instructions="Do not forward this",
            client=client,
        )
    finally:
        client.close()

    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.response.content == b"sync-audio"
    assert response._hidden_params["response_cost"] == _PROVIDER_COST
    assert len(captured_requests) == 2
    request = captured_requests[0]
    assert str(request.url) == "https://openrouter.test/api/v1/audio/speech"
    assert json.loads(request.content) == {
        "model": "hexgrad/kokoro-82m",
        "input": "hello",
        "voice": "af_alloy",
        "response_format": "mp3",
        "provider": {"order": ["preferred-provider"]},
    }
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["http-referer"] == "https://tests.litellm.ai"
    assert request.headers["x-title"] == "LiteLLM Tests"
    assert request.headers["x-global"] == "global"
    assert request.headers["x-public"] == "public"
    assert request.headers["x-extra"] == "extra"
    assert request.headers["x-precedence"] == "extra"
    assert str(captured_requests[1].url) == (f"https://openrouter.test/api/v1/generation?id={_GENERATION_ID}")
    assert global_headers == original_global_headers
    assert public_headers == original_public_headers
    assert extra_headers == original_extra_headers


def test_litellm_speech_polls_until_generation_cost_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[httpx.Request] = []
    generation_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal generation_requests
        captured_requests.append(request)
        if request.url.path.endswith("/generation"):
            generation_requests += 1
            if generation_requests < 3:
                return httpx.Response(
                    404,
                    json={"error": {"message": f"Generation {_GENERATION_ID} not found"}},
                )
        return _openrouter_tts_response(request, b"sync-audio")

    monkeypatch.setattr(transformation, "_GENERATION_STATS_POLL_INTERVAL_SECONDS", 0, raising=False)
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    try:
        response = litellm.speech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello",
            voice="af_alloy",
            api_key="test-key",
            api_base="https://openrouter.test/api/v1",
            client=client,
        )
    finally:
        client.close()

    assert response._hidden_params["response_cost"] == _PROVIDER_COST
    assert generation_requests == 3
    assert len(captured_requests) == 4


def test_litellm_speech_stops_polling_when_generation_cost_remains_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path.endswith("/generation"):
            return httpx.Response(
                404,
                json={"error": {"message": f"Generation {_GENERATION_ID} not found"}},
            )
        return _openrouter_tts_response(request, b"sync-audio")

    monkeypatch.setattr(transformation, "_GENERATION_STATS_MAX_POLLS", 3, raising=False)
    monkeypatch.setattr(transformation, "_GENERATION_STATS_POLL_INTERVAL_SECONDS", 0, raising=False)
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    try:
        with pytest.raises(OpenRouterException) as exc_info:
            litellm.speech(
                model="openrouter/hexgrad/kokoro-82m",
                input="hello",
                voice="af_alloy",
                api_key="test-key",
                api_base="https://openrouter.test/api/v1",
                client=client,
            )
    finally:
        client.close()

    assert exc_info.value.status_code == 404
    assert len(captured_requests) == 4


def test_litellm_speech_merges_all_header_sources_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return _openrouter_tts_response(request, b"audio")

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    global_headers = {"X-Global": "global", "X-Precedence": "global"}
    public_headers = {"X-Public": "public", "X-Precedence": "public"}
    extra_headers = {"X-Extra": "extra", "X-Precedence": "extra"}
    original_headers = (global_headers.copy(), public_headers.copy(), extra_headers.copy())
    monkeypatch.setattr(litellm, "headers", global_headers)

    try:
        litellm.speech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello",
            voice="af_alloy",
            api_key="test-key",
            headers=public_headers,
            extra_headers=extra_headers,
            client=client,
        )
    finally:
        client.close()

    request_headers = captured_requests[0].headers
    assert request_headers["x-global"] == "global"
    assert request_headers["x-public"] == "public"
    assert request_headers["x-extra"] == "extra"
    assert request_headers["x-precedence"] == "extra"
    assert (global_headers, public_headers, extra_headers) == original_headers


@pytest.mark.asyncio
async def test_litellm_aspeech_uses_injected_async_transport() -> None:
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return _openrouter_tts_response(request, b"async-audio")

    client = AsyncHTTPHandler()
    await client.close()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    try:
        response = await litellm.aspeech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello async",
            voice="af_heart",
            response_format="wav",
            speed=2,
            api_key="test-key",
            api_base="https://openrouter.test/api/v1",
            client=client,
        )
    finally:
        await client.close()

    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.response.content == b"async-audio"
    assert response._hidden_params["response_cost"] == _PROVIDER_COST
    assert len(captured_requests) == 2
    request = captured_requests[0]
    assert str(request.url) == "https://openrouter.test/api/v1/audio/speech"
    assert json.loads(request.content) == {
        "model": "hexgrad/kokoro-82m",
        "input": "hello async",
        "voice": "af_heart",
        "response_format": "wav",
        "speed": 2,
    }
    assert str(captured_requests[1].url) == (f"https://openrouter.test/api/v1/generation?id={_GENERATION_ID}")


@pytest.mark.asyncio
async def test_litellm_aspeech_polls_until_generation_cost_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[httpx.Request] = []
    generation_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal generation_requests
        captured_requests.append(request)
        if request.url.path.endswith("/generation"):
            generation_requests += 1
            if generation_requests < 3:
                return httpx.Response(
                    404,
                    json={"error": {"message": f"Generation {_GENERATION_ID} not found"}},
                )
        return _openrouter_tts_response(request, b"async-audio")

    monkeypatch.setattr(transformation, "_GENERATION_STATS_POLL_INTERVAL_SECONDS", 0, raising=False)
    client = AsyncHTTPHandler()
    await client.close()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    try:
        response = await litellm.aspeech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello async",
            voice="af_heart",
            api_key="test-key",
            api_base="https://openrouter.test/api/v1",
            client=client,
        )
    finally:
        await client.close()

    assert response._hidden_params["response_cost"] == _PROVIDER_COST
    assert generation_requests == 3
    assert len(captured_requests) == 4


def test_litellm_speech_maps_openrouter_non_success_response() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "capacity exhausted"}})

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    try:
        with pytest.raises(OpenRouterException) as exc_info:
            litellm.speech(
                model="openrouter/hexgrad/kokoro-82m",
                input="hello",
                voice="af_alloy",
                api_key="test-key",
                api_base="https://openrouter.test/api/v1",
                client=client,
            )
    finally:
        client.close()

    assert exc_info.value.status_code == 429
    assert "capacity exhausted" in exc_info.value.message


def test_litellm_speech_requires_openrouter_voice() -> None:
    with pytest.raises(litellm.BadRequestError, match="'voice' is required"):
        litellm.speech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello",
            api_key="test-key",
        )


class TestOpenRouterTextToSpeechConfig:
    def setup_method(self) -> None:
        self.config = OpenRouterTextToSpeechConfig()
        self.logging_obj = Mock()

    def test_supported_openai_params_match_openrouter_schema(self) -> None:
        assert self.config.get_supported_openai_params("hexgrad/kokoro-82m") == [
            "voice",
            "response_format",
            "speed",
        ]

    def test_validate_environment_sets_openrouter_headers(self) -> None:
        caller_headers = {"X-Custom": "value"}

        headers = self.config.validate_environment(
            headers=caller_headers,
            model="hexgrad/kokoro-82m",
            api_key="test-key",
        )

        assert headers == {
            "Authorization": "Bearer test-key",
            "HTTP-Referer": "https://tests.litellm.ai",
            "X-Title": "LiteLLM Tests",
            "Content-Type": "application/json",
            "X-Custom": "value",
        }
        assert caller_headers == {"X-Custom": "value"}

    def test_validate_environment_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_openrouter_keys(monkeypatch)

        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            self.config.validate_environment(
                headers={},
                model="hexgrad/kokoro-82m",
                api_key=None,
            )

    def test_get_complete_url(self) -> None:
        assert (
            self.config.get_complete_url(
                model="hexgrad/kokoro-82m",
                api_base="https://openrouter.ai/api/v1/",
                litellm_params={},
            )
            == "https://openrouter.ai/api/v1/audio/speech"
        )

    def test_map_openai_params_filters_instructions_and_preserves_extra_body(self) -> None:
        voice, params = self.config.map_openai_params(
            model="hexgrad/kokoro-82m",
            optional_params={"speed": 1, "instructions": "Do not forward this"},
            voice={"voice_id": "af_alloy"},
            kwargs={"extra_body": {"provider": {"order": ["preferred-provider"]}}},
        )

        assert voice == "af_alloy"
        assert params == {
            "speed": 1,
            "response_format": "mp3",
            "provider": {"order": ["preferred-provider"]},
        }

    def test_transform_text_to_speech_request_preserves_supported_fields(self) -> None:
        request_data = self.config.transform_text_to_speech_request(
            model="hexgrad/kokoro-82m",
            input="Narrate this scene",
            voice="af_alloy",
            optional_params={"response_format": "mp3", "speed": 1},
            litellm_params={},
            headers={},
        )

        assert request_data["dict_body"] == {
            "model": "hexgrad/kokoro-82m",
            "input": "Narrate this scene",
            "voice": "af_alloy",
            "response_format": "mp3",
            "speed": 1,
        }

    def test_transform_text_to_speech_response_returns_binary_content(self) -> None:
        raw_response = httpx.Response(
            200,
            content=b"audio-bytes",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/speech"),
        )

        result = self.config.transform_text_to_speech_response(
            model="hexgrad/kokoro-82m",
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.response.content == b"audio-bytes"
