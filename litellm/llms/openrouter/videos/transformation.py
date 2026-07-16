import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import OpenRouterException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

OPENROUTER_VIDEO_PARAMS = {
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
OPENROUTER_VIDEO_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
    "expired",
}


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
            *sorted(OPENROUTER_VIDEO_PARAMS),
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = dict(video_create_optional_params)
        duration = optional_params.get("duration")
        if duration is None:
            duration = optional_params.get("seconds")

        frame_images = optional_params.get("frame_images")
        input_reference = optional_params.get("input_reference")
        if input_reference is None:
            input_reference = optional_params.get("image")
        if frame_images is None and input_reference is not None:
            frame_images = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_reference_url(input_reference)},
                    "frame_type": "first_frame",
                }
            ]

        mapped_params = {
            key: value
            for key, value in optional_params.items()
            if key in OPENROUTER_VIDEO_PARAMS and key not in {"duration", "frame_images"} and value is not None
        }
        if duration is not None:
            mapped_params["duration"] = self._coerce_duration(duration)
        if frame_images is not None:
            mapped_params["frame_images"] = frame_images
        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
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
        return {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": get_secret_str("OR_SITE_URL") or "https://litellm.ai",
            "X-Title": get_secret_str("OR_APP_NAME") or "liteLLM",
            **(litellm.headers or {}),
            **headers,
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        return base_url if base_url.endswith("/videos") else f"{base_url}/videos"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        request_data = {
            "model": model,
            "prompt": prompt,
            **{
                key: value
                for key, value in video_create_optional_request_params.items()
                if key in OPENROUTER_VIDEO_PARAMS and value is not None
            },
        }
        return request_data, (), api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
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
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        encoded_video_id = self._encoded_video_id(video_id)
        query = "" if variant is None else f"?index={quote(variant, safe='')}"
        return f"{api_base}/{encoded_video_id}/content{query}", {}

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
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
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
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
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
        headers: dict,
    ) -> tuple[str, dict]:
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
        headers: dict,
    ) -> tuple[str, dict]:
        return f"{api_base}/{self._encoded_video_id(video_id)}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response = self._parse_video_response(raw_response)
        model = response.get("model")
        return self._to_video_object(
            response=response,
            custom_llm_provider=custom_llm_provider,
            model=model if isinstance(model, str) else None,
            request_data=None,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,
    ) -> BaseLLMException:
        return OpenRouterException(
            message=self._error_message(error_message),
            status_code=status_code,
            headers=headers,
        )

    def _parse_video_response(self, raw_response: httpx.Response) -> dict:
        self._raise_for_error(raw_response)
        try:
            response = raw_response.json()
        except json.JSONDecodeError:
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None
        if (
            not isinstance(response, dict)
            or not isinstance(response.get("id"), str)
            or response.get("status") not in OPENROUTER_VIDEO_STATUSES
        ):
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        return response

    def _raise_for_error(self, raw_response: httpx.Response) -> None:
        if 200 <= raw_response.status_code < 300:
            return
        raise OpenRouterException(
            message=self._error_message(raw_response.text),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _to_video_object(
        self,
        response: dict,
        custom_llm_provider: str | None,
        model: str | None,
        request_data: Mapping[str, object] | None,
    ) -> VideoObject:
        raw_id = response["id"]
        video_id = encode_video_id_with_provider(raw_id, custom_llm_provider, model) if custom_llm_provider else raw_id
        duration = request_data.get("duration") if request_data is not None else None
        size = None
        if request_data is not None:
            size = request_data.get("size") or request_data.get("resolution")

        usage_data: dict[str, object] = {}
        provider_usage = response.get("usage")
        if isinstance(provider_usage, dict):
            cost = provider_usage.get("cost")
            is_byok = provider_usage.get("is_byok")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                usage_data["cost"] = float(cost)
            if isinstance(is_byok, bool):
                usage_data["is_byok"] = is_byok
        if duration is not None:
            usage_data["duration_seconds"] = self._numeric_value(duration)
        if size is not None:
            usage_data["video_resolution"] = str(size)

        error = response.get("error")
        normalized_error = None
        if isinstance(error, str):
            normalized_error = {"message": error}
        elif isinstance(error, dict):
            normalized_error = error

        return VideoObject(
            id=video_id,
            object="video",
            status=self._status(response["status"]),
            error=normalized_error,
            seconds=str(duration) if duration is not None else None,
            size=str(size) if size is not None else None,
            model=model,
            usage=usage_data or None,
        )

    def _image_reference_url(self, reference: object) -> str:
        if isinstance(reference, str):
            return reference

        content_type = None
        content = reference
        if isinstance(reference, tuple):
            if len(reference) < 2:
                raise ValueError("OpenRouter input_reference tuples must include file content")
            content = reference[1]
            if len(reference) > 2 and isinstance(reference[2], str):
                content_type = reference[2]

        image_bytes = self._read_image_bytes(content)
        resolved_content_type = content_type or ImageEditRequestUtils.get_image_content_type(image_bytes)
        encoded_image = base64.b64encode(image_bytes).decode()
        return f"data:{resolved_content_type};base64,{encoded_image}"

    def _read_image_bytes(self, reference: object) -> bytes:
        if isinstance(reference, (bytes, bytearray)):
            return bytes(reference)
        if isinstance(reference, Path):
            return reference.read_bytes()
        if hasattr(reference, "read"):
            position = reference.tell() if hasattr(reference, "tell") else None
            if hasattr(reference, "seek"):
                reference.seek(0)
            image_bytes = reference.read()
            if position is not None and hasattr(reference, "seek"):
                reference.seek(position)
            if isinstance(image_bytes, bytes):
                return image_bytes
        raise ValueError("Unsupported input_reference type for OpenRouter video generation")

    def _coerce_duration(self, duration: object) -> int:
        numeric_duration = self._numeric_value(duration)
        if numeric_duration < 1 or not numeric_duration.is_integer():
            raise ValueError("OpenRouter video duration must be a positive whole number")
        return int(numeric_duration)

    @staticmethod
    def _numeric_value(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("OpenRouter video duration must be a positive whole number")
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("OpenRouter video duration must be a positive whole number") from exc

    @staticmethod
    def _status(status: str) -> Literal["queued", "in_progress", "completed", "failed"]:
        if status == "pending":
            return "queued"
        if status in {"in_progress", "completed"}:
            return status
        return "failed"

    @staticmethod
    def _encoded_video_id(video_id: str) -> str:
        return encode_url_path_segment(
            extract_original_video_id(video_id),
            field_name="video_id",
        )

    @staticmethod
    def _error_message(error_message: str) -> str:
        try:
            response = json.loads(error_message)
        except json.JSONDecodeError:
            return error_message
        if not isinstance(response, dict):
            return error_message
        error = response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        message = response.get("message")
        return message if isinstance(message, str) else error_message
