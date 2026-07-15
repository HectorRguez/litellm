from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager


def _json_response(
    data: object,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    request_headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=data,
        headers=headers,
        request=httpx.Request(
            "GET",
            "https://queue.fal.run/fal-ai/veo3/requests/request-123",
            headers=request_headers,
        ),
    )


@pytest.fixture
def config() -> FalAIVideoConfig:
    return FalAIVideoConfig()


@pytest.fixture(autouse=True)
def clear_pricing_cache() -> None:
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_provider_config_manager_registers_fal_video_config() -> None:
    provider_config = ProviderConfigManager.get_provider_video_config(
        model="fal-ai/veo3/fast",
        provider=LlmProviders.FAL_AI,
    )

    assert isinstance(provider_config, FalAIVideoConfig)


def test_maps_standard_and_provider_specific_params(config: FalAIVideoConfig) -> None:
    mapped = config.map_openai_params(
        video_create_optional_params={
            "image": "https://example.com/first.png",
            "seconds": "8s",
            "size": "1280x720",
            "extra_headers": {"X-Test": "ignored"},
            "user": "ignored",
        },
        model="fal-ai/veo3.1/image-to-video",
        drop_params=False,
    )

    assert mapped == {
        "image_url": "https://example.com/first.png",
        "duration": "8s",
        "aspect_ratio": "16:9",
    }


def test_rejects_non_json_input_reference(config: FalAIVideoConfig) -> None:
    with pytest.raises(ValueError, match="URL or data URI"):
        config.map_openai_params(
            video_create_optional_params={"input_reference": b"image"},
            model="fal-ai/veo3.1/image-to-video",
            drop_params=False,
        )


def test_transforms_create_request(config: FalAIVideoConfig) -> None:
    data, files, url = config.transform_video_create_request(
        model="fal-ai/veo3/fast",
        prompt="A city at sunrise",
        api_base="https://queue.fal.run",
        video_create_optional_request_params={
            "duration": "8s",
            "aspect_ratio": "16:9",
            "generate_audio": True,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data == {
        "prompt": "A city at sunrise",
        "duration": "8s",
        "aspect_ratio": "16:9",
        "generate_audio": True,
    }
    assert files == ()
    assert url == "https://queue.fal.run/fal-ai/veo3/fast"


def test_rejects_unsafe_model_path(config: FalAIVideoConfig) -> None:
    with pytest.raises(ValueError, match="dot path segment"):
        config.transform_video_create_request(
            model="fal-ai/../fast",
            prompt="test",
            api_base="https://queue.fal.run",
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_create_response_encodes_provider_model_and_usage(config: FalAIVideoConfig) -> None:
    video = config.transform_video_create_response(
        model="fal-ai/veo3/fast",
        raw_response=_json_response({"request_id": "request-123", "queue_position": 0}),
        logging_obj=Mock(),
        custom_llm_provider="fal_ai",
        request_data={"duration": "8s", "aspect_ratio": "16:9"},
    )

    assert decode_video_id_with_provider(video.id) == {
        "custom_llm_provider": "fal_ai",
        "model_id": "fal-ai/veo3/fast",
        "video_id": "request-123",
    }
    assert video.status == "queued"
    assert video.seconds == "8s"
    assert video.size == "16:9"
    assert video.model == "fal-ai/veo3/fast"
    assert video.usage == {"duration_seconds": 8.0}


def test_status_and_content_requests_use_queue_app_root(config: FalAIVideoConfig) -> None:
    video_id = encode_video_id_with_provider("request/123", "fal_ai", "fal-ai/heygen/avatar5/digital-twin")

    status_url, status_data = config.transform_video_status_retrieve_request(
        video_id=video_id,
        api_base="https://queue.fal.run",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    content_url, content_data = config.transform_video_content_request(
        video_id=video_id,
        api_base="https://queue.fal.run",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert status_url == "https://queue.fal.run/fal-ai/heygen/requests/request%2F123/status"
    assert content_url == "https://queue.fal.run/fal-ai/heygen/requests/request%2F123"
    assert status_data == {}
    assert content_data == {}


@pytest.mark.parametrize(
    ("fal_status", "error", "litellm_status"),
    [
        ("IN_QUEUE", None, "queued"),
        ("IN_PROGRESS", None, "in_progress"),
        ("COMPLETED", None, "completed"),
        ("COMPLETED", "generation failed", "failed"),
    ],
)
def test_maps_queue_status_and_keeps_downloadable_id(
    config: FalAIVideoConfig,
    fal_status: str,
    error: str | None,
    litellm_status: str,
) -> None:
    original_video_id = encode_video_id_with_provider(
        "request-123",
        "fal_ai",
        "fal-ai/heygen/avatar5/digital-twin",
    )
    logging_obj = Mock()
    logging_obj.optional_params = {"video_id": original_video_id}
    video = config.transform_video_status_retrieve_response(
        raw_response=_json_response(
            {
                "status": fal_status,
                "request_id": "request-123",
                "response_url": "https://queue.fal.run/fal-ai/heygen/requests/request-123",
                "error": error,
            }
        ),
        logging_obj=logging_obj,
        custom_llm_provider="fal_ai",
    )

    assert video.status == litellm_status
    assert decode_video_id_with_provider(video.id) == {
        "custom_llm_provider": "fal_ai",
        "model_id": "fal-ai/heygen/avatar5/digital-twin",
        "video_id": "request-123",
    }
    assert video.error == ({"message": error} if error is not None else None)


def test_status_response_rejects_missing_queue_url(config: FalAIVideoConfig) -> None:
    with pytest.raises(ValueError, match="response URL"):
        config.transform_video_status_retrieve_response(
            raw_response=_json_response({"status": "COMPLETED", "request_id": "request-123"}),
            logging_obj=Mock(),
            custom_llm_provider="fal_ai",
        )


def test_content_response_downloads_video() -> None:
    def media_fetcher(url: str) -> httpx.Response:
        assert url == "https://v3.fal.media/files/video.mp4"
        return httpx.Response(
            200,
            content=b"video-bytes",
            request=httpx.Request("GET", url),
        )

    config = FalAIVideoConfig(sync_media_fetcher=media_fetcher)
    content = config.transform_video_content_response(
        raw_response=_json_response({"video": {"url": "https://v3.fal.media/files/video.mp4"}}),
        logging_obj=Mock(),
    )

    assert content == b"video-bytes"


def test_content_response_tracks_provider_reported_cost_and_caches_pricing() -> None:
    media_fetcher = Mock(
        return_value=httpx.Response(
            200,
            content=b"video-bytes",
            request=httpx.Request("GET", "https://v3.fal.media/files/video.mp4"),
        )
    )
    pricing_fetcher = Mock(
        return_value=_json_response(
            {
                "prices": [
                    {
                        "endpoint_id": "fal-ai/heygen/avatar5/digital-twin",
                        "unit_price": 0.1,
                        "unit": "seconds",
                        "currency": "USD",
                    }
                ]
            }
        )
    )
    config = FalAIVideoConfig(
        sync_media_fetcher=media_fetcher,
        sync_pricing_fetcher=pricing_fetcher,
    )
    video_id = encode_video_id_with_provider(
        "request-123",
        "fal_ai",
        "fal-ai/heygen/avatar5/digital-twin",
    )
    logging_obj = Mock()
    logging_obj.optional_params = {"video_id": video_id}
    logging_obj.model_call_details = {}
    raw_response = _json_response(
        {"video": {"url": "https://v3.fal.media/files/video.mp4"}},
        headers={"x-fal-billable-units": "21"},
        request_headers={"Authorization": "Key test-key"},
    )

    first_content = config.transform_video_content_response(raw_response, logging_obj)
    second_content = config.transform_video_content_response(raw_response, logging_obj)

    assert first_content == second_content == b"video-bytes"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(2.1)
    assert logging_obj.model_call_details["provider_cost_tracking_id"] == "fal-video-cost:request-123"
    pricing_fetcher.assert_called_once_with(
        "fal-ai/heygen/avatar5/digital-twin",
        "Key test-key",
    )


def test_content_response_does_not_fail_when_pricing_is_unavailable() -> None:
    media_fetcher = Mock(
        return_value=httpx.Response(
            200,
            content=b"video-bytes",
            request=httpx.Request("GET", "https://v3.fal.media/files/video.mp4"),
        )
    )
    pricing_fetcher = Mock(side_effect=httpx.ConnectError("pricing unavailable"))
    config = FalAIVideoConfig(
        sync_media_fetcher=media_fetcher,
        sync_pricing_fetcher=pricing_fetcher,
    )
    logging_obj = Mock()
    logging_obj.optional_params = {
        "video_id": encode_video_id_with_provider("request-123", "fal_ai", "fal-ai/heygen/avatar5/digital-twin")
    }
    logging_obj.model_call_details = {}

    content = config.transform_video_content_response(
        _json_response(
            {"video": {"url": "https://v3.fal.media/files/video.mp4"}},
            headers={"x-fal-billable-units": "21"},
            request_headers={"Authorization": "Key test-key"},
        ),
        logging_obj,
    )

    assert content == b"video-bytes"
    assert "response_cost" not in logging_obj.model_call_details


@pytest.mark.asyncio
async def test_async_content_response_downloads_video() -> None:
    async def media_fetcher(url: str) -> httpx.Response:
        assert url == "https://v3.fal.media/files/video.mp4"
        return httpx.Response(
            200,
            content=b"async-video-bytes",
            request=httpx.Request("GET", url),
        )

    config = FalAIVideoConfig(async_media_fetcher=media_fetcher)
    content = await config.async_transform_video_content_response(
        raw_response=_json_response({"video": {"url": "https://v3.fal.media/files/video.mp4"}}),
        logging_obj=Mock(),
    )

    assert content == b"async-video-bytes"


@pytest.mark.asyncio
async def test_async_content_response_tracks_provider_reported_cost() -> None:
    async_media_fetcher = AsyncMock(
        return_value=httpx.Response(
            200,
            content=b"async-video-bytes",
            request=httpx.Request("GET", "https://v3.fal.media/files/video.mp4"),
        )
    )
    async_pricing_fetcher = AsyncMock(
        return_value=_json_response(
            {
                "prices": [
                    {
                        "endpoint_id": "fal-ai/heygen/avatar5/digital-twin",
                        "unit_price": 0.1,
                        "unit": "seconds",
                        "currency": "USD",
                    }
                ]
            }
        )
    )
    config = FalAIVideoConfig(
        async_media_fetcher=async_media_fetcher,
        async_pricing_fetcher=async_pricing_fetcher,
    )
    logging_obj = Mock()
    logging_obj.optional_params = {
        "video_id": encode_video_id_with_provider("request-123", "fal_ai", "fal-ai/heygen/avatar5/digital-twin")
    }
    logging_obj.model_call_details = {}

    content = await config.async_transform_video_content_response(
        _json_response(
            {"video": {"url": "https://v3.fal.media/files/video.mp4"}},
            headers={"x-fal-billable-units": "21"},
            request_headers={"Authorization": "Key test-key"},
        ),
        logging_obj,
    )

    assert content == b"async-video-bytes"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(2.1)
    async_pricing_fetcher.assert_awaited_once_with(
        "fal-ai/heygen/avatar5/digital-twin",
        "Key test-key",
    )


def test_provider_errors_preserve_status(config: FalAIVideoConfig) -> None:
    with pytest.raises(BaseLLMException) as exc_info:
        config.transform_video_create_response(
            model="fal-ai/veo3/fast",
            raw_response=_json_response({"detail": "invalid input"}, status_code=422),
            logging_obj=Mock(),
            custom_llm_provider="fal_ai",
        )

    assert exc_info.value.status_code == 422
    assert "invalid input" in str(exc_info.value)


def test_environment_and_custom_base_url(config: FalAIVideoConfig) -> None:
    headers = config.validate_environment(
        headers={"X-Test": "value"},
        model="fal-ai/veo3/fast",
        api_key="test-key",
    )

    assert headers == {
        "Authorization": "Key test-key",
        "Content-Type": "application/json",
        "X-Test": "value",
    }
    assert (
        config.get_complete_url(
            model="fal-ai/veo3/fast",
            api_base="https://example.com/queue/",
            litellm_params={},
        )
        == "https://example.com/queue"
    )


def test_environment_reads_api_key_from_litellm_params(config: FalAIVideoConfig) -> None:
    headers = config.validate_environment(
        headers={},
        model="",
        litellm_params=GenericLiteLLMParams(api_key="params-key"),
    )

    assert headers["Authorization"] == "Key params-key"
