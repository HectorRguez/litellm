from collections.abc import Awaitable, Callable
from math import gcd
from urllib.parse import unquote, urlparse

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.litellm_core_utils.url_utils import (
    async_safe_get,
    encode_url_path_segment,
    encode_url_path_segments,
    safe_get,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)


class _FalQueueSubmitResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str


class _FalQueueStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    request_id: str
    response_url: str | None = None
    error: str | None = None
    error_type: str | None = None


class _FalVideoFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str


class _FalVideoResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video: _FalVideoFile


_SyncMediaFetcher = Callable[[str], httpx.Response]
_AsyncMediaFetcher = Callable[[str], Awaitable[httpx.Response]]


def _sync_media_fetcher(url: str) -> httpx.Response:
    return safe_get(litellm.module_level_client, url, timeout=600.0)


async def _async_media_fetcher(url: str) -> httpx.Response:
    return await async_safe_get(
        litellm.module_level_aclient,
        url,
        timeout=600.0,
    )


class FalAIVideoConfig(BaseVideoConfig):
    DEFAULT_BASE_URL = "https://queue.fal.run"
    _STANDARD_PARAMS = frozenset(
        (
            "extra_body",
            "extra_headers",
            "image",
            "input_reference",
            "model",
            "parameters",
            "prompt",
            "seconds",
            "size",
            "user",
        )
    )

    def __init__(
        self,
        sync_media_fetcher: _SyncMediaFetcher = _sync_media_fetcher,
        async_media_fetcher: _AsyncMediaFetcher = _async_media_fetcher,
    ) -> None:
        super().__init__()
        self._sync_media_fetcher = sync_media_fetcher
        self._async_media_fetcher = async_media_fetcher

    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "model",
            "prompt",
            "image",
            "input_reference",
            "parameters",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        parameters = video_create_optional_params.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError("fal.ai parameters must be a dictionary")

        mapped_params: dict[str, object] = dict(parameters or {})
        for key, value in video_create_optional_params.items():
            if key not in self._STANDARD_PARAMS and value is not None:
                mapped_params[key] = value

        input_reference = video_create_optional_params.get("input_reference")
        if input_reference is None:
            input_reference = video_create_optional_params.get("image")
        if input_reference is not None:
            if not isinstance(input_reference, str):
                raise ValueError("fal.ai input_reference must be a URL or data URI string")
            mapped_params["image_url"] = input_reference

        seconds = video_create_optional_params.get("seconds")
        if seconds is not None:
            mapped_params["duration"] = self._duration(seconds)

        size = video_create_optional_params.get("size")
        if size is not None:
            mapped_params["aspect_ratio"] = self._aspect_ratio(size)

        return mapped_params

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:
        params_api_key = litellm_params.api_key if litellm_params is not None else None
        resolved_api_key = api_key or params_api_key or get_secret_str("FAL_AI_API_KEY") or get_secret_str("FAL_KEY")
        if resolved_api_key is None:
            raise ValueError("FAL_AI_API_KEY or FAL_KEY is not set")
        return {
            **headers,
            "Authorization": f"Key {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        return (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:
        request_data = {
            **video_create_optional_request_params,
            "prompt": prompt,
        }
        encoded_model = encode_url_path_segments(model, field_name="model")
        return request_data, (), f"{api_base}/{encoded_model}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,
    ) -> VideoObject:
        self._raise_for_error(raw_response)
        response = _FalQueueSubmitResponse.model_validate_json(raw_response.content)
        video_id = (
            encode_video_id_with_provider(
                response.request_id,
                custom_llm_provider,
                model,
            )
            if custom_llm_provider
            else response.request_id
        )
        duration = request_data.get("duration") if request_data is not None else None
        aspect_ratio = request_data.get("aspect_ratio") if request_data is not None else None
        duration_seconds = self._duration_seconds(duration)
        return VideoObject(
            id=video_id,
            object="video",
            status="queued",
            model=model,
            seconds=str(duration) if duration is not None else None,
            size=str(aspect_ratio) if aspect_ratio is not None else None,
            usage=({"duration_seconds": duration_seconds} if duration_seconds is not None else {}),
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, object]]:
        model, request_id = self._request_parts(video_id)
        return (
            f"{api_base}/{model}/requests/{request_id}/status",
            {},
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_error(raw_response)
        response = _FalQueueStatusResponse.model_validate_json(raw_response.content)
        model = self._model_from_response_url(response.response_url)
        video_id = (
            encode_video_id_with_provider(
                response.request_id,
                custom_llm_provider,
                model,
            )
            if custom_llm_provider
            else response.request_id
        )
        error_message = response.error or response.error_type
        return VideoObject(
            id=video_id,
            object="video",
            status=self._status(response.status, error_message),
            model=model,
            error=({"message": error_message} if error_message is not None else None),
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        model, request_id = self._request_parts(video_id)
        return f"{api_base}/{model}/requests/{request_id}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> bytes:
        self._raise_for_error(raw_response)
        result = _FalVideoResult.model_validate_json(raw_response.content)
        media_response = self._sync_media_fetcher(result.video.url)
        media_response.raise_for_status()
        return media_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> bytes:
        self._raise_for_error(raw_response)
        result = _FalVideoResult.model_validate_json(raw_response.content)
        media_response = await self._async_media_fetcher(result.video.url)
        media_response.raise_for_status()
        return media_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("video list is not supported for fal.ai")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("video list is not supported for fal.ai")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("video deletion is not supported for fal.ai")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> VideoObject:
        raise NotImplementedError("video deletion is not supported for fal.ai")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _raise_for_error(self, raw_response: httpx.Response) -> None:
        if raw_response.is_error:
            self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

    def _request_parts(self, video_id: str) -> tuple[str, str]:
        decoded = decode_video_id_with_provider(video_id)
        model = decoded.get("model_id")
        if model is None:
            raise ValueError("fal.ai video ID is missing its model")
        request_id = decoded.get("video_id")
        if request_id is None:
            raise ValueError("fal.ai video ID is missing its request ID")
        return (
            encode_url_path_segments(model, field_name="model"),
            encode_url_path_segment(request_id, field_name="video_id"),
        )

    def _model_from_response_url(self, response_url: str | None) -> str:
        if response_url is None:
            raise ValueError("fal.ai queue response did not include a response URL")
        path_parts = [unquote(part) for part in urlparse(response_url).path.split("/") if part]
        try:
            requests_index = path_parts.index("requests")
        except ValueError as exc:
            raise ValueError("fal.ai queue response URL is malformed") from exc
        model_parts = path_parts[:requests_index]
        if len(model_parts) < 2:
            raise ValueError("fal.ai queue response URL is missing its model")
        return "/".join(model_parts)

    def _aspect_ratio(self, size: object) -> str:
        if not isinstance(size, str):
            raise TypeError("fal.ai video size must be a string")
        separator = ":" if ":" in size else "x"
        try:
            width_text, height_text = size.lower().split(
                separator,
                maxsplit=1,
            )
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise ValueError("fal.ai video size must use WIDTHxHEIGHT or WIDTH:HEIGHT format") from exc
        if width <= 0 or height <= 0:
            raise ValueError("fal.ai video size dimensions must be positive")
        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def _duration(self, seconds: object) -> int:
        if isinstance(seconds, bool):
            raise TypeError("fal.ai video seconds must be a whole number")
        normalized = seconds.removesuffix("s") if isinstance(seconds, str) else seconds
        try:
            duration = float(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError("fal.ai video seconds must be a whole number") from exc
        if duration <= 0 or not duration.is_integer():
            raise ValueError("fal.ai video seconds must be a positive whole number")
        return int(duration)

    def _duration_seconds(self, duration: object) -> float | None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float, str)):
            return None
        normalized = duration.removesuffix("s") if isinstance(duration, str) else duration
        try:
            return float(normalized)
        except ValueError:
            return None

    def _status(
        self,
        status: str,
        error_message: str | None,
    ) -> str:
        if error_message is not None:
            return "failed"
        return {
            "IN_QUEUE": "queued",
            "IN_PROGRESS": "in_progress",
            "COMPLETED": "completed",
        }.get(status.upper(), "queued")
