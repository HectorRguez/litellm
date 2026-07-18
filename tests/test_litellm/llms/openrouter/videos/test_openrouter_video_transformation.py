import base64
from io import BytesIO
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager


def _json_response(data: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://openrouter.ai/api/v1/videos/job-123"),
    )


@pytest.fixture
def config() -> OpenRouterVideoConfig:
    return OpenRouterVideoConfig()


def test_provider_config_manager_registers_openrouter_video_config() -> None:
    provider_config = ProviderConfigManager.get_provider_video_config(
        model="google/veo-3.1",
        provider=LlmProviders.OPENROUTER,
    )

    assert isinstance(provider_config, OpenRouterVideoConfig)


def test_maps_standard_and_openrouter_video_params(config: OpenRouterVideoConfig) -> None:
    mapped = config.map_openai_params(
        video_create_optional_params={
            "seconds": "4",
            "size": "1280x720",
            "input_reference": "https://example.com/frame.png",
            "aspect_ratio": "16:9",
            "generate_audio": False,
            "callback_url": "https://example.com/webhook",
            "seed": 123,
        },
        model="google/veo-3.1",
        drop_params=True,
    )

    assert mapped == {
        "aspect_ratio": "16:9",
        "callback_url": "https://example.com/webhook",
        "duration": 4,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/frame.png"},
                "frame_type": "first_frame",
            }
        ],
        "generate_audio": False,
        "seed": 123,
        "size": "1280x720",
    }


def test_explicit_openrouter_fields_take_precedence(config: OpenRouterVideoConfig) -> None:
    frame_images = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/last.png"},
            "frame_type": "last_frame",
        }
    ]

    mapped = config.map_openai_params(
        video_create_optional_params={
            "seconds": "4",
            "duration": 8,
            "input_reference": "https://example.com/first.png",
            "frame_images": frame_images,
            "input_references": [{"type": "image_url", "image_url": {"url": "https://example.com/style.png"}}],
        },
        model="google/veo-3.1",
        drop_params=True,
    )

    assert mapped["duration"] == 8
    assert mapped["frame_images"] == frame_images
    assert mapped["input_references"] == [{"type": "image_url", "image_url": {"url": "https://example.com/style.png"}}]


def test_maps_binary_input_reference_to_data_url_without_consuming_stream(
    config: OpenRouterVideoConfig,
) -> None:
    image = BytesIO(b"\x89PNG\r\n\x1a\nimage")
    image.seek(4)

    mapped = config.map_openai_params(
        video_create_optional_params={"input_reference": image},
        model="alibaba/wan-2.7",
        drop_params=True,
    )

    encoded_url = mapped["frame_images"][0]["image_url"]["url"]
    assert encoded_url == f"data:image/png;base64,{base64.b64encode(image.getvalue()).decode('utf-8')}"
    assert image.tell() == 4


@pytest.mark.parametrize("duration", [True, "0", "4.5", "invalid"])
def test_rejects_invalid_duration(config: OpenRouterVideoConfig, duration: object) -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        config.map_openai_params(
            video_create_optional_params={"seconds": duration},
            model="google/veo-3.1",
            drop_params=True,
        )


def test_transforms_create_request(config: OpenRouterVideoConfig) -> None:
    data, files, url = config.transform_video_create_request(
        model="google/veo-3.1",
        prompt="A mountain at sunset",
        api_base="https://openrouter.ai/api/v1",
        video_create_optional_request_params={
            "duration": 4,
            "resolution": "720p",
            "generate_audio": False,
            "extra_headers": {"X-Test": "ignored"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data == {
        "model": "google/veo-3.1",
        "prompt": "A mountain at sunset",
        "duration": 4,
        "resolution": "720p",
        "generate_audio": False,
    }
    assert files == ()
    assert url == "https://openrouter.ai/api/v1/videos"


def test_create_response_maps_id_status_and_request_usage(config: OpenRouterVideoConfig) -> None:
    video = config.transform_video_create_response(
        model="google/veo-3.1",
        raw_response=_json_response(
            {
                "id": "job-123",
                "polling_url": "/api/v1/videos/job-123",
                "status": "pending",
            },
            status_code=202,
        ),
        logging_obj=Mock(),
        custom_llm_provider="openrouter",
        request_data={"duration": 4, "resolution": "720p"},
    )

    decoded_id = decode_video_id_with_provider(video.id)
    assert decoded_id == {
        "custom_llm_provider": "openrouter",
        "model_id": "google/veo-3.1",
        "video_id": "job-123",
    }
    assert video.status == "queued"
    assert video.seconds == "4"
    assert video.size == "720p"
    assert video.usage == {"duration_seconds": 4.0, "video_resolution": "720p"}


@pytest.mark.parametrize(
    ("openrouter_status", "litellm_status"),
    [
        ("pending", "queued"),
        ("in_progress", "in_progress"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "failed"),
        ("expired", "failed"),
    ],
)
def test_maps_status_responses(
    config: OpenRouterVideoConfig,
    openrouter_status: str,
    litellm_status: str,
) -> None:
    video = config.transform_video_status_retrieve_response(
        raw_response=_json_response(
            {
                "id": "job-123",
                "status": openrouter_status,
                "model": "google/veo-3.1",
                "error": "generation failed" if litellm_status == "failed" else None,
                "usage": {"cost": 0.25, "is_byok": False} if openrouter_status == "completed" else None,
            }
        ),
        logging_obj=Mock(model_call_details={}),
        custom_llm_provider="openrouter",
    )

    assert video.status == litellm_status
    assert decode_video_id_with_provider(video.id)["video_id"] == "job-123"
    if openrouter_status == "completed":
        assert video.usage == {"cost": 0.25, "is_byok": False}
    if litellm_status == "failed":
        assert video.error == {"message": "generation failed"}


def test_completed_status_records_provider_cost(config: OpenRouterVideoConfig) -> None:
    logging_obj = Mock(
        model_call_details={"model": "openrouter/bytedance/seedance-2.0"}
    )

    video = config.transform_video_status_retrieve_response(
        raw_response=_json_response(
            {
                "id": "job-123",
                "status": "completed",
                "generation_id": "generation-123",
                "model": None,
                "usage": {"cost": 0.6048, "is_byok": False},
            }
        ),
        logging_obj=logging_obj,
        custom_llm_provider="openrouter",
    )

    assert video.usage == {"cost": 0.6048, "is_byok": False}
    assert logging_obj.model_call_details["response_cost"] == 0.6048
    assert (
        logging_obj.model_call_details["provider_cost_tracking_id"]
        == "openrouter-video-cost:generation-123"
    )
    assert (
        logging_obj.model_call_details["provider_cost_tracking_model"]
        == "openrouter/bytedance/seedance-2.0"
    )
    assert video._hidden_params == {
        "response_cost": 0.6048,
        "provider_cost_tracking_id": "openrouter-video-cost:generation-123",
        "provider_cost_tracking_model": "openrouter/bytedance/seedance-2.0",
    }


def test_status_and_content_requests_decode_and_escape_video_id(config: OpenRouterVideoConfig) -> None:
    video_id = encode_video_id_with_provider("job/123?unsafe=true", "openrouter", "google/veo-3.1")

    status_url, status_data = config.transform_video_status_retrieve_request(
        video_id=video_id,
        api_base="https://openrouter.ai/api/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    content_url, content_data = config.transform_video_content_request(
        video_id=video_id,
        api_base="https://openrouter.ai/api/v1",
        litellm_params=GenericLiteLLMParams(),
        headers={},
        variant="1&unsafe=true",
    )

    assert status_url == "https://openrouter.ai/api/v1/videos/job%2F123%3Funsafe%3Dtrue"
    assert content_url == (
        "https://openrouter.ai/api/v1/videos/job%2F123%3Funsafe%3Dtrue/content?index=1%26unsafe%3Dtrue"
    )
    assert status_data == {}
    assert content_data == {}


def test_content_response_returns_bytes(config: OpenRouterVideoConfig) -> None:
    result = config.transform_video_content_response(
        raw_response=httpx.Response(
            200,
            content=b"video-bytes",
            headers={"content-type": "video/mp4"},
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/videos/job-123/content"),
        ),
        logging_obj=Mock(),
    )

    assert result == b"video-bytes"


def test_provider_errors_are_preserved(config: OpenRouterVideoConfig) -> None:
    with pytest.raises(OpenRouterException, match="Insufficient credits") as exc_info:
        config.transform_video_create_response(
            model="google/veo-3.1",
            raw_response=_json_response(
                {"error": {"code": 402, "message": "Insufficient credits"}},
                status_code=402,
            ),
            logging_obj=Mock(),
            custom_llm_provider="openrouter",
        )

    assert exc_info.value.status_code == 402


def test_environment_and_custom_base_url(config: OpenRouterVideoConfig) -> None:
    headers = config.validate_environment(
        headers={"X-Title": "custom"},
        model="google/veo-3.1",
        api_key="test-key",
    )

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Title"] == "custom"
    assert (
        config.get_complete_url(
            model="google/veo-3.1",
            api_base="https://example.com/openrouter/",
            litellm_params={},
        )
        == "https://example.com/openrouter"
    )
