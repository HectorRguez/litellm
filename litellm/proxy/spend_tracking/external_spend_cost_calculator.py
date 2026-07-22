import hashlib
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.litellm_core_utils.url_utils import async_safe_get
from litellm.proxy._types import ExternalSpendReportRequest
from litellm.secret_managers.main import get_secret_str


class ExternalSpendCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float
    metadata: dict[str, str | int | float | bool | None]


class _FalPrice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    endpoint_id: str
    unit_price: Decimal
    unit: str
    currency: str


class _FalPricingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prices: tuple[_FalPrice, ...]


class _PricingCache(Protocol):
    def get_cache(self, key: str) -> object | None: ...

    def set_cache(self, key: str, value: object, ttl: int) -> None: ...


_AsyncPricingFetcher = Callable[[str, str], Awaitable[httpx.Response]]
_HTTP_RESPONSE_ADAPTER = TypeAdapter(
    httpx.Response,
    config=ConfigDict(arbitrary_types_allowed=True),
)
_FAL_PRICING_ADAPTER = TypeAdapter(tuple[_FalPrice, ...] | _FalPricingResponse)


async def _async_fal_pricing_fetcher(
    endpoint_id: str,
    authorization: str,
) -> httpx.Response:
    return _HTTP_RESPONSE_ADAPTER.validate_python(
        await async_safe_get(
            litellm.module_level_aclient,
            ExternalSpendCostCalculator.FAL_PRICING_URL,
            headers={"Authorization": authorization},
            params={"endpoint_id": endpoint_id},
            timeout=30.0,
        )
    )


class ExternalSpendCostCalculator:
    FAL_PRICING_URL = "https://api.fal.ai/v1/models/pricing"
    FAL_PRICING_CACHE_TTL_SECONDS = 600

    def __init__(
        self,
        fal_pricing_fetcher: _AsyncPricingFetcher = _async_fal_pricing_fetcher,
        pricing_cache: _PricingCache | None = None,
    ) -> None:
        self._fal_pricing_fetcher = fal_pricing_fetcher
        self._pricing_cache = pricing_cache or litellm.in_memory_llm_clients_cache

    async def resolve(
        self,
        request: ExternalSpendReportRequest,
    ) -> ExternalSpendCost:
        return await self._resolve_fal(request)

    async def _resolve_fal(
        self,
        request: ExternalSpendReportRequest,
    ) -> ExternalSpendCost:
        authorization = self._fal_authorization()
        price = await self._fal_price(request.external_model, authorization)
        spend = Decimal(str(request.usage.quantity)) * price.unit_price
        return ExternalSpendCost(
            spend=float(spend),
            metadata={
                "billing_source": self.FAL_PRICING_URL,
                "billing_unit": price.unit,
                "billing_quantity": request.usage.quantity,
                "unit_price_usd": float(price.unit_price),
            },
        )

    async def _fal_price(
        self,
        endpoint_id: str,
        authorization: str,
    ) -> _FalPrice:
        cache_key = self._fal_cache_key(endpoint_id, authorization)
        cached = self._pricing_cache.get_cache(cache_key)
        if cached is not None:
            return _FalPrice.model_validate(cached)

        response = await self._fal_pricing_fetcher(endpoint_id, authorization)
        response.raise_for_status()
        parsed = _FAL_PRICING_ADAPTER.validate_json(response.content)
        prices = parsed if isinstance(parsed, tuple) else parsed.prices
        price = next(
            (candidate for candidate in prices if candidate.endpoint_id == endpoint_id),
            None,
        )
        if price is None:
            raise ValueError(f"Fal pricing did not return {endpoint_id}")
        if price.currency.upper() != "USD":
            raise ValueError(f"Fal pricing returned unsupported currency {price.currency}")
        if not price.unit_price.is_finite() or price.unit_price < 0:
            raise ValueError(f"Fal pricing returned invalid price for {endpoint_id}")
        self._pricing_cache.set_cache(
            cache_key,
            price.model_dump(mode="json"),
            ttl=self.FAL_PRICING_CACHE_TTL_SECONDS,
        )
        return price

    @staticmethod
    def _fal_authorization() -> str:
        api_key = get_secret_str("FAL_AI_API_KEY") or get_secret_str("FAL_KEY")
        if not api_key:
            raise ValueError("FAL_AI_API_KEY or FAL_KEY is not set")
        return api_key if api_key.startswith("Key ") else f"Key {api_key}"

    @staticmethod
    def _fal_cache_key(endpoint_id: str, authorization: str) -> str:
        credential_hash = hashlib.sha256(authorization.encode()).hexdigest()
        return f"external-fal-pricing:{endpoint_id}:{credential_hash}"


external_spend_cost_calculator = ExternalSpendCostCalculator()
