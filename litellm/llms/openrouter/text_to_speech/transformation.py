from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

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


_EMPTY_JSON_OBJECT: Mapping[str, JsonValue] = MappingProxyType(
    {}  # mutable-ok: MappingProxyType requires a dictionary source for an immutable empty mapping
)
_EXTRA_BODY_ADAPTER = TypeAdapter(Mapping[str, JsonValue])


class _OpenRouterSpeechParams(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    response_format: str = "mp3"
    speed: int | float | None = None


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
