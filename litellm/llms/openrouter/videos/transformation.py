import base64
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Tuple, Union, runtime_checkable
from urllib.parse import quote

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_VIDEO_PARAMS = frozenset(
    {
        "aspect_ratio",
        "callback_url",
        "duration",
        "frame_images",
        "generate_audio",
        "input_references",
        "provider",
        "resolution",
        "seed",
        "size",
    }
)
_HEADERS_ADAPTER = TypeAdapter(dict[str, str])
_OBJECT_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_OBJECT_TUPLE_ADAPTER = TypeAdapter(tuple[object, ...])
_PATH_ADAPTER = TypeAdapter(Path)


@runtime_checkable
class _BinaryFile(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...

    def tell(self) -> int: ...


class _OpenRouterVideoUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float | None = None
    is_byok: bool | None = None


class _OpenRouterVideoError(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    code: Union[str, int] | None = None
    message: str | None = None


class _OpenRouterVideoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    status: Literal["pending", "in_progress", "completed", "failed", "cancelled", "expired"]
    error: Union[str, _OpenRouterVideoError] | None = None
    generation_id: str | None = None
    model: str | None = None
    polling_url: str | None = None
    unsigned_urls: Tuple[str, ...] = ()
    usage: _OpenRouterVideoUsage | None = None


class _OpenRouterErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error: Union[str, _OpenRouterVideoError] | None = None
    message: str | None = None


class OpenRouterVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "model",
            "prompt",
            "input_reference",
            "image",
            "seconds",
            "size",
            "extra_headers",
            "extra_body",
            *sorted(_OPENROUTER_VIDEO_PARAMS),
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        optional_params = _OBJECT_MAPPING_ADAPTER.validate_python(video_create_optional_params)
        explicit_duration = optional_params.get("duration")
        duration = explicit_duration if explicit_duration is not None else optional_params.get("seconds")
        explicit_frames = optional_params.get("frame_images")
        input_reference = optional_params.get("input_reference") or optional_params.get("image")
        frame_images = (
            explicit_frames
            if explicit_frames is not None
            else self._frame_images_from_reference(input_reference)
            if input_reference is not None
            else None
        )
        passthrough_params = {
            key: value
            for key, value in optional_params.items()
            if key in _OPENROUTER_VIDEO_PARAMS and key not in {"duration", "frame_images"} and value is not None
        }
        mapped_duration = self._coerce_duration(duration) if duration is not None else None
        return {
            **passthrough_params,
            **({"duration": mapped_duration} if mapped_duration is not None else {}),
            **({"frame_images": frame_images} if frame_images is not None else {}),
        }

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:
        params_api_key = litellm_params.api_key if litellm_params is not None else None
        resolved_api_key = (
            api_key
            or params_api_key
            or litellm.api_key
            or litellm.openrouter_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        caller_headers = _HEADERS_ADAPTER.validate_python(headers)
        return {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": get_secret_str("OR_SITE_URL") or "https://litellm.ai",
            "X-Title": get_secret_str("OR_APP_NAME") or "liteLLM",
            **caller_headers,
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        return (api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or _OPENROUTER_API_BASE).rstrip(
            "/"
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> Tuple[dict[str, object], RequestFiles, str]:
        optional_params = _OBJECT_MAPPING_ADAPTER.validate_python(video_create_optional_request_params)
        request_data = {
            "model": model,
            "prompt": prompt,
            **{
                key: value
                for key, value in optional_params.items()
                if key in _OPENROUTER_VIDEO_PARAMS and value is not None
            },
        }
        return request_data, (), self._videos_url(api_base)

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,
    ) -> VideoObject:
        return self._to_video_object(
            response=self._parse_video_response(raw_response),
            custom_llm_provider=custom_llm_provider,
            model=model,
            request_data=request_data,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        variant: str | None = None,
    ) -> Tuple[str, dict[str, object]]:
        encoded_video_id = self._encoded_original_video_id(video_id)
        index_query = "" if variant is None else f"?index={quote(variant, safe='')}"
        return f"{self._videos_url(api_base)}/{encoded_video_id}/content{index_query}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_error(raw_response)
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        extra_body: dict[str, object] | None = None,
    ) -> Tuple[str, dict[str, object]]:
        raise NotImplementedError("video remix is not supported for OpenRouter")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for OpenRouter")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> Tuple[str, dict[str, object]]:
        raise NotImplementedError("video list is not supported for OpenRouter")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("video list is not supported for OpenRouter")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> Tuple[str, dict[str, object]]:
        raise NotImplementedError("video delete is not supported for OpenRouter")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("video delete is not supported for OpenRouter")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> Tuple[str, dict[str, object]]:
        return f"{self._videos_url(api_base)}/{self._encoded_original_video_id(video_id)}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response = self._parse_video_response(raw_response)
        video = self._to_video_object(
            response=response,
            custom_llm_provider=custom_llm_provider,
            model=response.model,
            request_data=None,
        )
        self._record_provider_cost(
            response=response,
            video=video,
            logging_obj=logging_obj,
        )
        return video

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict[str, str], httpx.Headers],
    ) -> BaseLLMException:
        return OpenRouterException(message=error_message, status_code=status_code, headers=headers)

    def _parse_video_response(self, raw_response: httpx.Response) -> _OpenRouterVideoResponse:
        self._raise_for_error(raw_response)
        try:
            return _OpenRouterVideoResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise OpenRouterException(
                message=f"OpenRouter returned an invalid video response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from exc

    def _raise_for_error(self, raw_response: httpx.Response) -> None:
        if 200 <= raw_response.status_code < 300:
            return
        try:
            error_response = _OpenRouterErrorEnvelope.model_validate_json(raw_response.content)
            error_message = self._error_message(error_response.error) or error_response.message or raw_response.text
        except ValidationError:
            error_message = raw_response.text
        raise OpenRouterException(
            message=error_message,
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _to_video_object(
        self,
        response: _OpenRouterVideoResponse,
        custom_llm_provider: str | None,
        model: str | None,
        request_data: Mapping[str, object] | None,
    ) -> VideoObject:
        video_id = (
            encode_video_id_with_provider(response.id, custom_llm_provider, model)
            if custom_llm_provider
            else response.id
        )
        duration = request_data.get("duration") if request_data is not None else None
        size = request_data.get("size") or request_data.get("resolution") if request_data is not None else None
        usage = self._usage(response.usage, duration=duration, size=size)
        error = self._video_error(response.error)
        return VideoObject(
            id=video_id,
            object="video",
            status=self._status(response.status),
            error=error,
            seconds=str(duration) if duration is not None else None,
            size=str(size) if size is not None else None,
            model=model,
            usage=usage or None,
        )

    def _usage(
        self,
        response_usage: _OpenRouterVideoUsage | None,
        duration: object | None,
        size: object | None,
    ) -> dict[str, object]:
        return {
            **({"cost": response_usage.cost} if response_usage is not None and response_usage.cost is not None else {}),
            **(
                {"is_byok": response_usage.is_byok}
                if response_usage is not None and response_usage.is_byok is not None
                else {}
            ),
            **({"duration_seconds": self._numeric_value(duration)} if duration is not None else {}),
            **({"video_resolution": str(size)} if size is not None else {}),
        }

    def _record_provider_cost(
        self,
        response: _OpenRouterVideoResponse,
        video: VideoObject,
        logging_obj: LiteLLMLoggingObj,
    ) -> None:
        if response.usage is None or response.usage.cost is None:
            return
        tracking_id = f"openrouter-video-cost:{response.generation_id or response.id}"
        tracking_model = response.model or logging_obj.model_call_details.get("model") or "openrouter"
        logging_obj.model_call_details["response_cost"] = response.usage.cost
        logging_obj.model_call_details["provider_cost_tracking_id"] = tracking_id
        logging_obj.model_call_details["provider_cost_tracking_model"] = tracking_model
        video._hidden_params = {
            "response_cost": response.usage.cost,
            "provider_cost_tracking_id": tracking_id,
            "provider_cost_tracking_model": tracking_model,
        }

    def _video_error(self, error: Union[str, _OpenRouterVideoError] | None) -> dict[str, object] | None:
        if error is None:
            return None
        if isinstance(error, str):
            return {"message": error}
        return _OBJECT_MAPPING_ADAPTER.validate_python(error.model_dump(exclude_none=True))

    def _error_message(self, error: Union[str, _OpenRouterVideoError] | None) -> str | None:
        if error is None:
            return None
        if isinstance(error, str):
            return error
        return error.message

    def _status(
        self,
        status: Literal["pending", "in_progress", "completed", "failed", "cancelled", "expired"],
    ) -> Literal["queued", "in_progress", "completed", "failed"]:
        if status == "pending":
            return "queued"
        if status == "in_progress":
            return "in_progress"
        if status == "completed":
            return "completed"
        return "failed"

    def _videos_url(self, api_base: str) -> str:
        normalized_base = api_base.rstrip("/")
        return normalized_base if normalized_base.endswith("/videos") else f"{normalized_base}/videos"

    def _encoded_original_video_id(self, video_id: str) -> str:
        return encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")

    def _frame_images_from_reference(self, reference: object) -> list[dict[str, object]]:
        return [
            {
                "type": "image_url",
                "image_url": {"url": self._image_reference_url(reference)},
                "frame_type": "first_frame",
            }
        ]

    def _image_reference_url(self, reference: object) -> str:
        if isinstance(reference, str):
            return reference
        if isinstance(reference, tuple):
            typed_reference = _OBJECT_TUPLE_ADAPTER.validate_python(reference)
            if len(typed_reference) < 2:
                raise ValueError("OpenRouter input_reference tuples must include file content")
            file_reference = typed_reference[1]
            explicit_content_type = (
                typed_reference[2] if len(typed_reference) >= 3 and isinstance(typed_reference[2], str) else None
            )
            return self._data_url(file_reference, explicit_content_type)
        return self._data_url(reference, None)

    def _data_url(self, reference: object, explicit_content_type: str | None) -> str:
        image_bytes = self._read_image_bytes(reference)
        content_type = explicit_content_type or ImageEditRequestUtils.get_image_content_type(image_bytes)
        return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    def _read_image_bytes(self, reference: object) -> bytes:
        if isinstance(reference, bytes):
            return reference
        if isinstance(reference, Path):
            return reference.read_bytes()
        try:
            path = _PATH_ADAPTER.validate_python(reference)
        except ValidationError:
            path = None
        if path is not None:
            return path.read_bytes()
        if isinstance(reference, _BinaryFile):
            current_position = reference.tell()
            reference.seek(0)
            image_bytes = reference.read()
            reference.seek(current_position)
            return image_bytes
        raise ValueError("Unsupported input_reference type for OpenRouter video generation")

    def _coerce_duration(self, value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("OpenRouter video duration must be a positive whole number")
        numeric_duration = self._numeric_value(value)
        if numeric_duration < 1 or not numeric_duration.is_integer():
            raise ValueError("OpenRouter video duration must be a positive whole number")
        return int(numeric_duration)

    def _numeric_value(self, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("OpenRouter video duration must be a positive whole number")
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("OpenRouter video duration must be a positive whole number") from exc
