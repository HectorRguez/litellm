from __future__ import annotations

from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterException(BaseLLMException):
    pass


class _OpenRouterErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: str | None = None


class _OpenRouterErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error: _OpenRouterErrorDetail | str | None = None
    message: str | None = None


def get_openrouter_api_base(api_base: str | None = None) -> str:
    return (
        api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE
    ).rstrip("/")


def get_openrouter_endpoint(api_base: str | None, endpoint_path: str) -> str:
    base_url = get_openrouter_api_base(api_base)
    normalized_path = endpoint_path.strip("/")
    if not normalized_path:
        return base_url
    if base_url.endswith(f"/{normalized_path}"):
        return base_url
    return f"{base_url}/{normalized_path}"


def get_openrouter_api_key(api_key: str | None = None) -> str:
    resolved_api_key = (
        api_key
        or litellm.api_key
        or litellm.openrouter_key
        or get_secret_str("OPENROUTER_API_KEY")
        or get_secret_str("OR_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
        )
    return resolved_api_key


def merge_openrouter_headers(
    global_headers: Mapping[str, str] | None,
    headers: Mapping[str, str] | None,
    extra_headers: Mapping[str, str] | None,
) -> dict[str, str]:  # mutable-ok: shared HTTP handler requires a mutable header dictionary
    header_items = tuple(
        item for source in (global_headers, headers, extra_headers) if source is not None for item in source.items()
    )
    return dict(header_items)  # mutable-ok: shared HTTP handler requires a mutable header dictionary


def get_openrouter_headers(
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    content_type: str | None = "application/json",
) -> dict[str, str]:  # mutable-ok: provider validation contract requires mutable headers
    default_header_items = (
        ("Authorization", f"Bearer {get_openrouter_api_key(api_key)}"),
        ("HTTP-Referer", get_secret_str("OR_SITE_URL") or "https://litellm.ai"),
        ("X-Title", get_secret_str("OR_APP_NAME") or "liteLLM"),
    )
    content_type_items = (("Content-Type", content_type),) if content_type is not None else ()
    caller_header_items = tuple(headers.items()) if headers is not None else ()
    return dict(  # mutable-ok: provider validation contract requires mutable headers
        (*default_header_items, *content_type_items, *caller_header_items)
    )


def get_openrouter_error_message(response_body: str | bytes, default: str) -> str:
    try:
        response = _OpenRouterErrorResponse.model_validate_json(response_body)
    except ValidationError:
        return default

    match response.error:
        case _OpenRouterErrorDetail(message=message) if message:
            return message
        case str() as error_message:
            return error_message
        case _:
            return response.message or default


def raise_openrouter_error(raw_response: httpx.Response) -> None:
    if 200 <= raw_response.status_code < 300:
        return

    raise OpenRouterException(
        message=get_openrouter_error_message(
            response_body=raw_response.content,
            default=raw_response.text,
        ),
        status_code=raw_response.status_code,
        headers=raw_response.headers,
    )
