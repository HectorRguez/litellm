from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.openrouter.common_utils import (
    OpenRouterException,
    get_openrouter_endpoint,
    get_openrouter_error_message,
    get_openrouter_headers,
    raise_openrouter_error,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


_EMPTY_JSON_OBJECT: Mapping[str, JsonValue] = MappingProxyType(
    {}  # mutable-ok: MappingProxyType requires a dictionary source for an immutable empty mapping
)
_EXTRA_BODY_ADAPTER = TypeAdapter(Mapping[str, JsonValue])
_GENERATION_STATS_MAX_POLLS = 15
_GENERATION_STATS_POLL_INTERVAL_SECONDS = 2.0
_RESPONSE_COST_HEADER = "llm_provider-x-litellm-response-cost"
_SPEECH_ENDPOINT_SUFFIX = "/audio/speech"


class _OpenRouterSpeechParams(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    response_format: str = "mp3"
    speed: int | float | None = None


class _OpenRouterGenerationStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    model: str
    total_cost: float = Field(ge=0)


class _OpenRouterGenerationStatsEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: _OpenRouterGenerationStats


def _generation_stats_request(
    raw_response: httpx.Response,
) -> tuple[str, str, dict[str, str]]:
    generation_id = raw_response.headers.get("x-generation-id")
    authorization = raw_response.request.headers.get("authorization")
    request_url = raw_response.request.url
    if generation_id is None or not generation_id.strip():
        raise OpenRouterException(
            message="OpenRouter TTS response omitted X-Generation-Id",
            status_code=502,
            headers=raw_response.headers,
        )
    if authorization is None or not authorization.strip():
        raise OpenRouterException(
            message="OpenRouter TTS request omitted authorization",
            status_code=502,
            headers=raw_response.headers,
        )
    if not request_url.path.endswith(_SPEECH_ENDPOINT_SUFFIX):
        raise OpenRouterException(
            message=f"Unexpected OpenRouter TTS endpoint: {request_url}",
            status_code=502,
            headers=raw_response.headers,
        )
    generation_path = request_url.path[: -len(_SPEECH_ENDPOINT_SUFFIX)] + "/generation"
    generation_url = str(request_url.copy_with(path=generation_path))
    return generation_url, generation_id, {"Authorization": authorization}


def _parse_generation_stats(
    response: httpx.Response,
    generation_id: str,
) -> _OpenRouterGenerationStats:
    raise_openrouter_error(response)
    try:
        stats = _OpenRouterGenerationStatsEnvelope.model_validate_json(response.content).data
    except ValidationError:
        raise OpenRouterException(
            message=response.text,
            status_code=response.status_code,
            headers=response.headers,
        ) from None
    if stats.id != generation_id:
        raise OpenRouterException(
            message=f"OpenRouter generation stats id {stats.id!r} did not match {generation_id!r}",
            status_code=502,
            headers=response.headers,
        )
    return stats


def _poll_generation_stats(
    client: HTTPHandler,
    generation_url: str,
    generation_id: str,
    headers: dict[str, str],
) -> httpx.Response:
    response: httpx.Response
    for poll_index in range(_GENERATION_STATS_MAX_POLLS):
        response = client.get(
            url=generation_url,
            headers=headers,
            params={"id": generation_id},
        )
        if response.status_code != 404:
            return response
        if poll_index + 1 < _GENERATION_STATS_MAX_POLLS:
            time.sleep(_GENERATION_STATS_POLL_INTERVAL_SECONDS)
    return response


async def _async_poll_generation_stats(
    client: AsyncHTTPHandler,
    generation_url: str,
    generation_id: str,
    headers: dict[str, str],
) -> httpx.Response:
    response: httpx.Response
    for poll_index in range(_GENERATION_STATS_MAX_POLLS):
        response = await client.get(
            url=generation_url,
            headers=headers,
            params={"id": generation_id},
        )
        if response.status_code != 404:
            return response
        if poll_index + 1 < _GENERATION_STATS_MAX_POLLS:
            await asyncio.sleep(_GENERATION_STATS_POLL_INTERVAL_SECONDS)
    return response


def _record_generation_cost(
    result: HttpxBinaryResponseContent,
    stats: _OpenRouterGenerationStats,
    logging_obj: LiteLLMLoggingObj,
) -> HttpxBinaryResponseContent:
    tracking_id = f"openrouter-tts-cost:{stats.id}"
    hidden_params = {
        "additional_headers": {
            _RESPONSE_COST_HEADER: stats.total_cost,
            "x-generation-id": stats.id,
        },
        "response_cost": stats.total_cost,
        "provider_cost_authoritative": True,
        "provider_cost_tracking_id": tracking_id,
        "provider_cost_tracking_model": stats.model,
    }
    result._hidden_params = hidden_params
    logging_obj.model_call_details.update(hidden_params)
    return result


class OpenRouterTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: inherited provider contract returns a list
        return [  # mutable-ok: inherited provider contract returns a list
            "voice",
            "response_format",
            "speed",
        ]

    def _resolve_voice(self, voice: str | Mapping[str, JsonValue] | None) -> str | None:
        if voice is None or isinstance(voice, str):
            return voice

        voice_candidates = tuple(voice.get(key) for key in ("voice_id", "id", "name"))
        candidate = next(
            (value for value in voice_candidates if isinstance(value, str) and value.strip()),
            None,
        )
        if candidate is None:
            raise ValueError("OpenRouter TTS voice must be a string or a mapping with voice_id, id, or name.")
        return candidate

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, JsonValue],
        voice: str | Mapping[str, JsonValue] | None = None,
        drop_params: bool = False,
        kwargs: Mapping[str, JsonValue] | None = None,
    ) -> tuple[  # mutable-ok: inherited transformation contract returns mutable params
        str | None, dict[str, JsonValue]
    ]:
        speech_params = _OpenRouterSpeechParams.model_validate(optional_params)
        raw_extra_body = kwargs.get("extra_body") if kwargs is not None else None
        extra_body = (
            _EXTRA_BODY_ADAPTER.validate_python(raw_extra_body) if raw_extra_body is not None else _EMPTY_JSON_OBJECT
        )
        speed_items: tuple[tuple[str, JsonValue], ...] = (
            (("speed", speech_params.speed),) if speech_params.speed is not None else ()
        )
        standard_param_items: tuple[tuple[str, JsonValue], ...] = (
            ("response_format", speech_params.response_format),
            *speed_items,
        )
        extra_body_items = tuple((key, value) for key, value in extra_body.items() if value is not None)
        mapped_params: dict[str, JsonValue] = dict(  # mutable-ok: inherited contract returns mutable params
            (*standard_param_items, *extra_body_items)
        )
        return self._resolve_voice(voice), mapped_params

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: inherited provider validation contract returns mutable headers
        return get_openrouter_headers(api_key=api_key, headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, JsonValue],
    ) -> str:
        return get_openrouter_endpoint(api_base, "audio/speech")

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        voice_items: tuple[tuple[str, JsonValue], ...] = (("voice", voice),) if voice is not None else ()
        request_items: tuple[tuple[str, JsonValue], ...] = (
            ("model", model),
            ("input", input),
            *voice_items,
            *optional_params.items(),
        )
        request_body = dict(request_items)  # mutable-ok: shared TTS handler requires a JSON dictionary body
        return TextToSpeechRequestData(dict_body=request_body)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        raise_openrouter_error(raw_response)
        return HttpxBinaryResponseContent(raw_response)

    def resolve_text_to_speech_provider_cost(
        self,
        result: HttpxBinaryResponseContent,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        client: HTTPHandler,
    ) -> HttpxBinaryResponseContent:
        generation_url, generation_id, headers = _generation_stats_request(raw_response)
        stats_response = _poll_generation_stats(
            client=client,
            generation_url=generation_url,
            generation_id=generation_id,
            headers=headers,
        )
        return _record_generation_cost(
            result,
            _parse_generation_stats(stats_response, generation_id),
            logging_obj,
        )

    async def async_resolve_text_to_speech_provider_cost(
        self,
        result: HttpxBinaryResponseContent,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        client: AsyncHTTPHandler,
    ) -> HttpxBinaryResponseContent:
        generation_url, generation_id, headers = _generation_stats_request(raw_response)
        stats_response = await _async_poll_generation_stats(
            client=client,
            generation_url=generation_url,
            generation_id=generation_id,
            headers=headers,
        )
        return _record_generation_cost(
            result,
            _parse_generation_stats(stats_response, generation_id),
            logging_obj,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return OpenRouterException(
            message=get_openrouter_error_message(error_message, error_message),
            status_code=status_code,
            headers=httpx.Headers(headers),
        )
