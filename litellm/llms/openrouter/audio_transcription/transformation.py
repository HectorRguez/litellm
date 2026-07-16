import json

import httpx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret, get_secret_str
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

from ..common_utils import OpenRouterException


class OpenRouterAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return [
            "language",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for key, value in non_default_params.items():
            if key in self.get_supported_openai_params(model):
                optional_params[key] = value

        response_format = optional_params.get("response_format")
        if response_format in (None, "json", "verbose_json"):
            return optional_params
        if drop_params or litellm.drop_params:
            optional_params.pop("response_format", None)
            return optional_params
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                f"OpenRouter does not support response_format={response_format!r}. "
                "Supported values are 'json' and 'verbose_json'."
            ),
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = (
            api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        if base_url.endswith("/audio/transcriptions"):
            return base_url
        return f"{base_url}/audio/transcriptions"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openrouter_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        merged_headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": get_secret("OR_SITE_URL") or "https://litellm.ai",
            "X-Title": get_secret("OR_APP_NAME") or "liteLLM",
            **(litellm.headers or {}),
            **headers,
        }
        return {key: value for key, value in merged_headers.items() if key.lower() != "content-type"}

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        form_fields = {"model": model}
        for key in self.get_supported_openai_params(model):
            value = optional_params.get(key)
            if value is None:
                continue
            request_key = "timestamp_granularities[]" if key == "timestamp_granularities" else key
            form_fields[request_key] = value

        return AudioTranscriptionRequestData(
            data=form_fields,
            files={
                "file": (
                    processed_audio.filename,
                    processed_audio.file_content,
                    processed_audio.content_type,
                )
            },
        )

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        if not 200 <= raw_response.status_code < 300:
            raise OpenRouterException(
                message=self._get_error_message(raw_response.text),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_json = raw_response.json()
        except json.JSONDecodeError:
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None

        if not isinstance(response_json, dict) or not isinstance(response_json.get("text"), str):
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        response = TranscriptionResponse(text=response_json["text"])
        for key, value in response_json.items():
            if key not in ("text", "usage"):
                response[key] = value

        hidden_params = dict(response_json)
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            seconds = usage.get("seconds")
            cost = usage.get("cost")

            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                )
            ):
                response.usage = TranscriptionUsageTokensObject(
                    type="tokens",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details=TranscriptionUsageInputTokenDetailsObject(
                        audio_tokens=input_tokens,
                        text_tokens=0,
                    ),
                )
            elif isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                response.usage = TranscriptionUsageDurationObject(
                    type="duration",
                    seconds=float(seconds),
                )

            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                hidden_params["audio_transcription_duration"] = float(seconds)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                hidden_params["additional_headers"] = {"llm_provider-x-litellm-response-cost": float(cost)}

        response._hidden_params = hidden_params
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,
    ) -> BaseLLMException:
        return OpenRouterException(
            message=self._get_error_message(error_message),
            status_code=status_code,
            headers=headers,
        )

    @staticmethod
    def _get_error_message(error_message: str) -> str:
        try:
            response_json = json.loads(error_message)
        except json.JSONDecodeError:
            return error_message
        if not isinstance(response_json, dict):
            return error_message
        error = response_json.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        message = response_json.get("message")
        return message if isinstance(message, str) else error_message
