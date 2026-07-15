import hashlib
from collections.abc import Awaitable, Callable
from decimal import Decimal, InvalidOperation
from math import gcd
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlparse

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


class _FalQueueSubmitResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str


class _FalQueueStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    request_id: str
    response_url: str | None = None
    status_url: str | None = None
    error: str | None = None
    error_type: str | None = None


class _FalVideoFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str


class _FalVideoResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video: _FalVideoFile


class _FalPrice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    endpoint_id: str
    unit_price: Decimal
    unit: str
    currency: str


class _FalPricingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prices: tuple[_FalPrice, ...]


class _FalCostContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    authorization: str
    billable_units: Decimal
    endpoint_id: str
    request_id: str


class _PricingCache(Protocol):
    def get_cache(self, key: str) -> object | None: ...

    def set_cache(self, key: str, value: object, ttl: int) -> None: ...


_SyncMediaFetcher = Callable[[str], httpx.Response]
_AsyncMediaFetcher = Callable[[str], Awaitable[httpx.Response]]
_SyncPricingFetcher = Callable[[str, str], httpx.Response]
_AsyncPricingFetcher = Callable[[str, str], Awaitable[httpx.Response]]
_HTTP_RESPONSE_ADAPTER = TypeAdapter(
    httpx.Response,
    config=ConfigDict(arbitrary_types_allowed=True),
)
_OBJECT_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_STRING_MAPPING_ADAPTER = TypeAdapter(dict[str, str])


def _sync_media_fetcher(url: str) -> httpx.Response:
    return _HTTP_RESPONSE_ADAPTER.validate_python(safe_get(litellm.module_level_client, url, timeout=600.0))


async def _async_media_fetcher(url: str) -> httpx.Response:
    return _HTTP_RESPONSE_ADAPTER.validate_python(
        await async_safe_get(litellm.module_level_aclient, url, timeout=600.0)
    )


def _sync_pricing_fetcher(endpoint_id: str, authorization: str) -> httpx.Response:
    return _HTTP_RESPONSE_ADAPTER.validate_python(
        safe_get(
            litellm.module_level_client,
            FalAIVideoConfig.PRICING_URL,
            headers={"Authorization": authorization},
            params={"endpoint_id": endpoint_id},
            timeout=30.0,
        )
    )


async def _async_pricing_fetcher(endpoint_id: str, authorization: str) -> httpx.Response:
    return _HTTP_RESPONSE_ADAPTER.validate_python(
        await async_safe_get(
            litellm.module_level_aclient,
            FalAIVideoConfig.PRICING_URL,
            headers={"Authorization": authorization},
            params={"endpoint_id": endpoint_id},
            timeout=30.0,
        )
    )


class FalAIVideoConfig(BaseVideoConfig):
    DEFAULT_BASE_URL = "https://queue.fal.run"
    PRICING_URL = "https://api.fal.ai/v1/models/pricing"
    PRICING_CACHE_TTL_SECONDS = 600
    _QUEUE_NAMESPACES = frozenset(("comfy", "workflows"))
    _STANDARD_PARAMS = frozenset(
        (
            "extra_body",
            "extra_headers",
            "image",
            "input_reference",
            "model",
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
        sync_pricing_fetcher: _SyncPricingFetcher = _sync_pricing_fetcher,
        async_pricing_fetcher: _AsyncPricingFetcher = _async_pricing_fetcher,
        pricing_cache: _PricingCache | None = None,
    ) -> None:
        super().__init__()
        self._sync_media_fetcher = sync_media_fetcher
        self._async_media_fetcher = async_media_fetcher
        self._sync_pricing_fetcher = sync_pricing_fetcher
        self._async_pricing_fetcher = async_pricing_fetcher
        self._pricing_cache = pricing_cache or cast(_PricingCache, litellm.in_memory_llm_clients_cache)

    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "model",
            "prompt",
            "image",
            "input_reference",
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
        input_reference = video_create_optional_params.get("input_reference") or video_create_optional_params.get(
            "image"
        )
        if input_reference is not None and not isinstance(input_reference, str):
            raise ValueError("fal.ai input_reference must be a URL or data URI string")

        size = video_create_optional_params.get("size")
        seconds = video_create_optional_params.get("seconds")
        provider_params = {
            key: value
            for key, value in video_create_optional_params.items()
            if key not in self._STANDARD_PARAMS and value is not None
        }
        return {
            **({"image_url": input_reference} if input_reference is not None else {}),
            **({"duration": seconds} if seconds is not None else {}),
            **({"aspect_ratio": self._aspect_ratio(size)} if size is not None else {}),
            **provider_params,
        }

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
        return request_data, (), f"{api_base}/{encode_url_path_segments(model, field_name='model')}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,
    ) -> VideoObject:
        self._raise_for_error(raw_response)
        response = _FalQueueSubmitResponse.model_validate_json(raw_response.content)
        video_id = (
            encode_video_id_with_provider(response.request_id, custom_llm_provider, model)
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
            usage={"duration_seconds": duration_seconds} if duration_seconds is not None else {},
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, object]]:
        decoded = decode_video_id_with_provider(video_id)
        model = decoded.get("model_id")
        if model is None:
            raise ValueError("fal.ai video ID is missing its model")
        decoded_request_id = decoded.get("video_id")
        if decoded_request_id is None:
            raise ValueError("fal.ai video ID is missing its request ID")
        request_id = encode_url_path_segment(decoded_request_id, field_name="video_id")
        return f"{api_base}/{self._queue_root(model)}/requests/{request_id}/status", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_error(raw_response)
        response = _FalQueueStatusResponse.model_validate_json(raw_response.content)
        queue_root = self._queue_root_from_response(response)
        optional_params = _OBJECT_MAPPING_ADAPTER.validate_python(cast(object, logging_obj.optional_params))
        original_video_id = optional_params.get("video_id")
        original_model = (
            decode_video_id_with_provider(original_video_id).get("model_id")
            if isinstance(original_video_id, str)
            else None
        )
        response_model = original_model or queue_root
        video_id = (
            encode_video_id_with_provider(response.request_id, custom_llm_provider, response_model)
            if custom_llm_provider
            else response.request_id
        )
        error_message = response.error or response.error_type
        return VideoObject(
            id=video_id,
            object="video",
            status=self._status(response.status, error_message),
            model=response_model,
            error={"message": error_message} if error_message is not None else None,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        decoded = decode_video_id_with_provider(video_id)
        model = decoded.get("model_id")
        if model is None:
            raise ValueError("fal.ai video ID is missing its model")
        decoded_request_id = decoded.get("video_id")
        if decoded_request_id is None:
            raise ValueError("fal.ai video ID is missing its request ID")
        request_id = encode_url_path_segment(decoded_request_id, field_name="video_id")
        return f"{api_base}/{self._queue_root(model)}/requests/{request_id}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_error(raw_response)
        result = _FalVideoResult.model_validate_json(raw_response.content)
        self._track_sync_provider_cost(raw_response, logging_obj)
        media_response = self._sync_media_fetcher(result.video.url)
        media_response.raise_for_status()
        return media_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_error(raw_response)
        result = _FalVideoResult.model_validate_json(raw_response.content)
        await self._track_async_provider_cost(raw_response, logging_obj)
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
        logging_obj: LiteLLMLoggingObj,
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
        logging_obj: LiteLLMLoggingObj,
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
        logging_obj: LiteLLMLoggingObj,
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

    def _track_sync_provider_cost(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> None:
        try:
            context = self._cost_context(raw_response, logging_obj)
            if context is None:
                return
            price = self._get_cached_price(context) or self._fetch_sync_price(context)
            if price is not None:
                self._record_provider_cost(context, price, logging_obj)
        except (httpx.HTTPError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
            verbose_logger.warning("Unable to track fal.ai provider cost: %s", type(exc).__name__)

    async def _track_async_provider_cost(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> None:
        try:
            context = self._cost_context(raw_response, logging_obj)
            if context is None:
                return
            cached_price = self._get_cached_price(context)
            price = cached_price if cached_price is not None else await self._fetch_async_price(context)
            if price is not None:
                self._record_provider_cost(context, price, logging_obj)
        except (httpx.HTTPError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
            verbose_logger.warning("Unable to track fal.ai provider cost: %s", type(exc).__name__)

    def _cost_context(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> _FalCostContext | None:
        optional_params = _OBJECT_MAPPING_ADAPTER.validate_python(cast(object, logging_obj.optional_params))
        encoded_video_id = optional_params.get("video_id")
        if not isinstance(encoded_video_id, str):
            return None
        decoded_video_id = decode_video_id_with_provider(encoded_video_id)
        endpoint_id = decoded_video_id.get("model_id")
        request_id = decoded_video_id.get("video_id")
        request_headers = _STRING_MAPPING_ADAPTER.validate_python(cast(object, raw_response.request.headers))
        response_headers = _STRING_MAPPING_ADAPTER.validate_python(cast(object, raw_response.headers))
        authorization = request_headers.get("authorization")
        raw_billable_units = response_headers.get("x-fal-billable-units")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            return None
        if not isinstance(request_id, str) or not request_id:
            return None
        if not isinstance(authorization, str) or not authorization:
            return None
        if raw_billable_units is None:
            return None
        try:
            billable_units = Decimal(raw_billable_units)
        except InvalidOperation:
            return None
        if not billable_units.is_finite() or billable_units < 0:
            return None
        return _FalCostContext(
            authorization=authorization,
            billable_units=billable_units,
            endpoint_id=endpoint_id,
            request_id=request_id,
        )

    def _fetch_sync_price(self, context: _FalCostContext) -> _FalPrice | None:
        response = self._sync_pricing_fetcher(context.endpoint_id, context.authorization)
        response.raise_for_status()
        return self._cache_valid_price(context, response)

    async def _fetch_async_price(self, context: _FalCostContext) -> _FalPrice | None:
        response = await self._async_pricing_fetcher(context.endpoint_id, context.authorization)
        response.raise_for_status()
        return self._cache_valid_price(context, response)

    def _cache_valid_price(self, context: _FalCostContext, response: httpx.Response) -> _FalPrice | None:
        pricing = _FalPricingResponse.model_validate_json(response.content)
        price = next((item for item in pricing.prices if item.endpoint_id == context.endpoint_id), None)
        if price is None or price.currency.upper() != "USD":
            return None
        if not price.unit_price.is_finite() or price.unit_price < 0:
            return None
        self._pricing_cache.set_cache(
            key=self._price_cache_key(context),
            value=price,
            ttl=self.PRICING_CACHE_TTL_SECONDS,
        )
        return price

    def _get_cached_price(self, context: _FalCostContext) -> _FalPrice | None:
        cached_price = self._pricing_cache.get_cache(key=self._price_cache_key(context))
        if cached_price is None:
            return None
        try:
            return _FalPrice.model_validate(cached_price)
        except ValueError:
            return None

    def _price_cache_key(self, context: _FalCostContext) -> str:
        credential_hash = hashlib.sha256(context.authorization.encode()).hexdigest()
        return f"fal-pricing:{context.endpoint_id}:{credential_hash}"

    def _record_provider_cost(
        self,
        context: _FalCostContext,
        price: _FalPrice,
        logging_obj: LiteLLMLoggingObj,
    ) -> None:
        logging_obj.model_call_details["response_cost"] = float(context.billable_units * price.unit_price)
        logging_obj.model_call_details["provider_cost_tracking_id"] = f"fal-video-cost:{context.request_id}"
        logging_obj.model_call_details["provider_cost_tracking_model"] = context.endpoint_id

    def _queue_root(self, model: str) -> str:
        model_parts = model.split("/")
        root_length = 3 if model_parts[0] in self._QUEUE_NAMESPACES else 2
        if len(model_parts) < root_length:
            raise ValueError("fal.ai model must include an owner and model name")
        return encode_url_path_segments("/".join(model_parts[:root_length]), field_name="model")

    def _queue_root_from_response(self, response: _FalQueueStatusResponse) -> str:
        queue_url = response.response_url or response.status_url
        if queue_url is None:
            raise ValueError("fal.ai queue response did not include a response URL")
        path_parts = tuple(part for part in urlparse(queue_url).path.split("/") if part)
        try:
            requests_index = path_parts.index("requests")
        except ValueError as exc:
            raise ValueError("fal.ai queue response URL is malformed") from exc
        app_parts = path_parts[:requests_index]
        root_length = 3 if len(app_parts) >= 3 and app_parts[-3] in self._QUEUE_NAMESPACES else 2
        if len(app_parts) < root_length:
            raise ValueError("fal.ai queue response URL is missing its model")
        return "/".join(app_parts[-root_length:])

    def _aspect_ratio(self, size: object) -> str:
        if not isinstance(size, str):
            raise TypeError("fal.ai video size must be a string")
        if ":" in size:
            return size
        try:
            width_text, height_text = size.lower().split("x", maxsplit=1)
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise ValueError("fal.ai video size must use WIDTHxHEIGHT or WIDTH:HEIGHT format") from exc
        if width <= 0 or height <= 0:
            raise ValueError("fal.ai video size dimensions must be positive")
        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def _duration_seconds(self, duration: object) -> float | None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float, str)):
            return None
        normalized = duration.removesuffix("s") if isinstance(duration, str) else duration
        try:
            return float(normalized)
        except ValueError:
            return None

    def _status(self, status: str, error_message: str | None) -> str:
        if error_message is not None:
            return "failed"
        return {
            "IN_QUEUE": "queued",
            "IN_PROGRESS": "in_progress",
            "COMPLETED": "completed",
        }.get(status.upper(), "queued")
