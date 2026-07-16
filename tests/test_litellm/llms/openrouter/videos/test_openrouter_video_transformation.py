import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager

MODEL = "google/veo-3.1-fast"


def _transport(
    captured_requests: list[httpx.Request],
    response_json: dict | None = None,
    status_code: int = 200,
    content: bytes | None = None,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if content is not None:
            return httpx.Response(status_code, content=content, headers={"content-type": "video/mp4"})
        return httpx.Response(status_code, json=response_json)

    return httpx.MockTransport(handle)


def test_openrouter_video_is_registered() -> None:
    config = ProviderConfigManager.get_provider_video_config(
        model=MODEL,
        provider=LlmProviders.OPENROUTER,
    )
    assert isinstance(config, OpenRouterVideoConfig)

    repository_root = Path(__file__).resolve().parents[5]
    metadata = json.loads((repository_root / "provider_endpoints_support.json").read_text())
    assert metadata["providers"]["openrouter"]["endpoints"]["video_generations"] is True


def test_maps_openrouter_video_params_and_binary_reference() -> None:
    config = OpenRouterVideoConfig()
    image = BytesIO(b"\x89PNG\r\n\x1a\nimage")
    image.seek(4)

    mapped = config.map_openai_params(
        video_create_optional_params={
            "seconds": "4",
            "size": "1280x720",
            "input_reference": image,
            "aspect_ratio": "16:9",
            "generate_audio": False,
            "provider": {"options": {"google-vertex": {"output_config": {"effort": "low"}}}},
        },
        model=MODEL,
        drop_params=False,
    )

    assert mapped == {
        "aspect_ratio": "16:9",
        "duration": 4,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image.getvalue()).decode()}"},
                "frame_type": "first_frame",
            }
        ],
        "generate_audio": False,
        "provider": {"options": {"google-vertex": {"output_config": {"effort": "low"}}}},
        "size": "1280x720",
    }
    assert image.tell() == 4


@pytest.mark.parametrize("duration", [True, "0", "4.5", "invalid"])
def test_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        OpenRouterVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": duration},
            model=MODEL,
            drop_params=False,
        )


def test_public_video_generation_sends_json_and_maps_response() -> None:
    captured_requests: list[httpx.Request] = []
    with httpx.Client(
        transport=_transport(
            captured_requests,
            response_json={
                "id": "job-123",
                "polling_url": "/api/v1/videos/job-123",
                "status": "pending",
            },
            status_code=202,
        )
    ) as http_client:
        response = litellm.video_generation(
            prompt="A mountain at sunset",
            model=f"openrouter/{MODEL}",
            seconds="4",
            size="1280x720",
            extra_body={
                "resolution": "720p",
                "generate_audio": False,
                "seed": 123,
            },
            api_key="test-key",
            extra_headers={"X-Title": "custom-video-client"},
            client=HTTPHandler(client=http_client),
        )

    request = captured_requests[0]
    assert request.method == "POST"
    assert request.url == "https://openrouter.ai/api/v1/videos"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-title"] == "custom-video-client"
    assert json.loads(request.content) == {
        "model": MODEL,
        "prompt": "A mountain at sunset",
        "duration": 4,
        "size": "1280x720",
        "resolution": "720p",
        "generate_audio": False,
        "seed": 123,
    }
    assert decode_video_id_with_provider(response.id) == {
        "custom_llm_provider": "openrouter",
        "model_id": MODEL,
        "video_id": "job-123",
    }
    assert response.status == "queued"
    assert response.seconds == "4"
    assert response.size == "1280x720"
    assert response.usage == {
        "duration_seconds": 4.0,
        "video_resolution": "1280x720",
    }


@pytest.mark.asyncio
async def test_public_avideo_generation_maps_provider_usage() -> None:
    captured_requests: list[httpx.Request] = []
    async_handler = AsyncHTTPHandler()
    await async_handler.client.aclose()
    async_handler.client = httpx.AsyncClient(
        transport=_transport(
            captured_requests,
            response_json={
                "id": "job-async",
                "polling_url": "/api/v1/videos/job-async",
                "status": "completed",
                "usage": {"cost": 0.25, "is_byok": False},
            },
        )
    )
    try:
        response = await litellm.avideo_generation(
            prompt="Ocean waves",
            model=f"openrouter/{MODEL}",
            seconds="6",
            extra_body={"resolution": "1080p"},
            api_key="test-key",
            client=async_handler,
        )
    finally:
        await async_handler.close()

    assert captured_requests[0].url == "https://openrouter.ai/api/v1/videos"
    assert response.status == "completed"
    assert response.usage == {
        "cost": 0.25,
        "is_byok": False,
        "duration_seconds": 6.0,
        "video_resolution": "1080p",
    }


def test_public_video_status_and_content_use_encoded_id() -> None:
    create_response = OpenRouterVideoConfig().transform_video_create_response(
        model=MODEL,
        raw_response=httpx.Response(
            202,
            json={
                "id": "job/123?unsafe=true",
                "polling_url": "/api/v1/videos/job-123",
                "status": "pending",
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/videos"),
        ),
        logging_obj=Mock(),
        custom_llm_provider="openrouter",
        request_data={"duration": 4},
    )

    status_requests: list[httpx.Request] = []
    with httpx.Client(
        transport=_transport(
            status_requests,
            response_json={
                "id": "job/123?unsafe=true",
                "polling_url": "/api/v1/videos/job-123",
                "status": "completed",
                "usage": {"cost": 0.5},
            },
        )
    ) as http_client:
        status = litellm.video_status(
            video_id=create_response.id,
            api_key="test-key",
            client=HTTPHandler(client=http_client),
        )

    assert status_requests[0].url == "https://openrouter.ai/api/v1/videos/job%2F123%3Funsafe%3Dtrue"
    assert status.status == "completed"
    assert status.usage == {"cost": 0.5}

    content_requests: list[httpx.Request] = []
    with httpx.Client(transport=_transport(content_requests, content=b"video-bytes")) as http_client:
        content = litellm.video_content(
            video_id=create_response.id,
            variant="1",
            api_key="test-key",
            client=HTTPHandler(client=http_client),
        )

    assert content_requests[0].url == ("https://openrouter.ai/api/v1/videos/job%2F123%3Funsafe%3Dtrue/content?index=1")
    assert content == b"video-bytes"


def test_openrouter_video_maps_provider_error() -> None:
    raw_response = httpx.Response(
        402,
        json={"error": {"code": 402, "message": "Insufficient credits"}},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/videos"),
    )

    with pytest.raises(OpenRouterException, match="Insufficient credits"):
        OpenRouterVideoConfig().transform_video_create_response(
            model=MODEL,
            raw_response=raw_response,
            logging_obj=Mock(),
            custom_llm_provider="openrouter",
        )


def test_openrouter_video_rejects_unsupported_list_operation() -> None:
    with pytest.raises(NotImplementedError, match="video list"):
        OpenRouterVideoConfig().transform_video_list_request(
            api_base="https://openrouter.ai/api/v1/videos",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
