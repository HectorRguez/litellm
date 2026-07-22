from collections.abc import Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openrouter.common_utils import (
    OpenRouterException,
    get_openrouter_endpoint,
    get_openrouter_headers,
    parse_openrouter_error_message,
    raise_openrouter_error,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageDurationObject,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)

_SUPPORTED_OPENAI_PARAMS: tuple[OpenAIAudioTranscriptionOptionalParams, ...] = (
    "language",
    "prompt",
    "response_format",
    "temperature",
    "timestamp_granularities",
)
_SUPPORTED_RESPONSE_FORMATS = frozenset(("json", "verbose_json"))
_EXCLUDED_FORM_PARAMS = frozenset(
    (
        "OPENAI_TRANSCRIPTION_PARAMS",
        "additional_drop_params",
        "api_base",
        "api_key",
        "caching",
        "client",
        "custom_llm_provider",
        "drop_params",
        "extra_headers",
        "headers",
        "litellm_call_id",
        "litellm_logging_obj",
        "max_retries",
        "metadata",
        "model",
        "model_info",
        "proxy_server_request",
        "shared_session",
        "timeout",
        "user",
    )
)
_RESPONSE_COST_HEADER = "llm_provider-x-litellm-response-cost"
_JSON_OBJECT_ADAPTER: TypeAdapter[Mapping[str, JsonValue]] = TypeAdapter(Mapping[str, JsonValue])


class _OpenRouterTranscriptionUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float


class _OpenRouterTranscriptionWord(BaseModel):
    model_config = ConfigDict(frozen=True)

    word: str
    start: float
    end: float


class _OpenRouterTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    text: str
    usage: _OpenRouterTranscriptionUsage
    words: tuple[_OpenRouterTranscriptionWord, ...] | None = None


def _normalize_transcription_word(
    words: tuple[_OpenRouterTranscriptionWord, ...], index: int
) -> dict[str, JsonValue] | None:
    word = words[index]
    next_start = words[index + 1].start if index + 1 < len(words) else None
    normalized_end = word.end if word.end > word.start else next_start
    if normalized_end is None or normalized_end <= word.start:
        return None
    return {"word": word.word, "start": word.start, "end": normalized_end}


class OpenRouterAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: inherited contract returns list
        return [*_SUPPORTED_OPENAI_PARAMS]  # mutable-ok: inherited contract returns list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: inherited contract returns mutable optional params
        mapped_params = {  # mutable-ok: inherited contract returns mutable optional params
            **optional_params,
            **{  # mutable-ok: inherited contract returns mutable optional params
                key: value
                for key, value in non_default_params.items()
                if value is not None and key in _SUPPORTED_OPENAI_PARAMS
            },
        }
        response_format = mapped_params.get("response_format")
        if response_format is None or response_format in _SUPPORTED_RESPONSE_FORMATS:
            return mapped_params
        if drop_params or litellm.drop_params:
            return {  # mutable-ok: inherited contract returns mutable optional params
                key: value for key, value in mapped_params.items() if key != "response_format"
            }
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                f"OpenRouter does not support response_format={response_format!r}. "
                f"Supported values: {', '.join(sorted(_SUPPORTED_RESPONSE_FORMATS))}. "
                "To drop unsupported OpenAI params from the call, set `litellm.drop_params = True`"
            ),
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_openrouter_endpoint(api_base, "audio/transcriptions")

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: inherited contract returns mutable headers
        return {  # mutable-ok: inherited contract returns mutable headers
            key: value
            for key, value in get_openrouter_headers(api_key=api_key, headers=headers, content_type=None).items()
            if key.lower() != "content-type"
        }

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        openai_form_fields = {  # mutable-ok: multipart handler requires mutable form data
            "timestamp_granularities[]" if key == "timestamp_granularities" else key: value
            for key in _SUPPORTED_OPENAI_PARAMS
            if (value := optional_params.get(key)) is not None
        }
        provider_form_fields = {  # mutable-ok: multipart handler requires mutable form data
            key: value
            for key, value in optional_params.items()
            if key not in _SUPPORTED_OPENAI_PARAMS
            and key not in _EXCLUDED_FORM_PARAMS
            and not key.startswith("litellm_")
        }
        form_fields = {  # mutable-ok: multipart handler requires mutable form data
            "model": model,
            **openai_form_fields,
            **provider_form_fields,
        }
        files = {  # mutable-ok: multipart handler requires mutable file data
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }
        return AudioTranscriptionRequestData(data=form_fields, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        raise_openrouter_error(raw_response)
        try:
            provider_fields = _JSON_OBJECT_ADAPTER.validate_json(raw_response.content)
            provider_response = _OpenRouterTranscriptionResponse.model_validate(provider_fields)
        except ValidationError:
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None

        match provider_response.usage:
            case _OpenRouterTranscriptionUsage(
                input_tokens=int() as input_tokens,
                output_tokens=int() as output_tokens,
                total_tokens=int() as total_tokens,
            ):
                normalized_usage = TranscriptionUsageTokensObject(
                    type="tokens",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details=TranscriptionUsageInputTokenDetailsObject(
                        audio_tokens=input_tokens,
                        text_tokens=0,
                    ),
                )
            case _OpenRouterTranscriptionUsage(seconds=float() as seconds):
                normalized_usage = TranscriptionUsageDurationObject(type="duration", seconds=seconds)
            case _:
                raise OpenRouterException(
                    message=raw_response.text,
                    status_code=raw_response.status_code,
                    headers=raw_response.headers,
                )
        response = TranscriptionResponse(text=provider_response.text)
        response.usage = normalized_usage
        for key, value in provider_fields.items():
            if key not in ("text", "usage", "words"):
                response[key] = value
        if provider_response.words is not None:
            response["words"] = tuple(
                normalized_word
                for index in range(len(provider_response.words))
                if (normalized_word := _normalize_transcription_word(provider_response.words, index)) is not None
            )
        duration_hidden_params = (
            {}
            if provider_response.usage.seconds is None
            else {"audio_transcription_duration": provider_response.usage.seconds}
        )
        hidden_params = {  # mutable-ok: TranscriptionResponse requires mutable hidden params
            **provider_fields,
            **duration_hidden_params,
            "additional_headers": {  # mutable-ok: cost tracking requires mutable response headers
                _RESPONSE_COST_HEADER: provider_response.usage.cost
            },
        }
        response._hidden_params = (  # pyright: ignore[reportPrivateUsage]  # LiteLLM metadata contract uses _hidden_params
            hidden_params
        )
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: inherited error contract passes mutable headers
    ) -> BaseLLMException:
        return OpenRouterException(
            message=parse_openrouter_error_message(response_content=error_message, default=error_message),
            status_code=status_code,
            headers=headers,
        )
