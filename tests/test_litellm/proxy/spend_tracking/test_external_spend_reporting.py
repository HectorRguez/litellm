import json
from datetime import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from litellm.caching import DualCache
from litellm.proxy._types import (
    ExternalSpendReportRequest,
    ExternalSpendUsage,
    SpendLogsPayload,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_utils import get_model_from_request
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.spend_tracking.external_spend_cost_calculator import (
    ExternalSpendCost,
    ExternalSpendCostCalculator,
)
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    report_external_spend,
)
from litellm.types.utils import UnresolvedProviderCost


def test_external_spend_report_rejects_negative_usage() -> None:
    with pytest.raises(ValidationError):
        ExternalSpendReportRequest(
            provider="fal",
            external_model="fal-ai/gemini-3.1-flash-tts",
            usage=ExternalSpendUsage(unit="billable_units", quantity=-0.01),
            request_id="request-1",
        )


def test_external_spend_report_accepts_only_fal_billable_units() -> None:
    with pytest.raises(ValidationError):
        ExternalSpendReportRequest.model_validate(
            {
                "provider": "heygen",
                "external_model": "heygen/avatar_v",
                "usage": {"unit": "seconds", "quantity": 30},
                "request_id": "heygen-video-1",
            }
        )


@pytest.mark.asyncio
async def test_external_spend_report_uses_authenticated_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import spend_management_endpoints

    report_spend = AsyncMock(return_value=True)
    resolve_cost = AsyncMock(
        return_value=ExternalSpendCost(
            spend=0.015,
            metadata={
                "billing_source": "https://api.fal.ai/v1/models/pricing",
                "billing_unit": "1000 characters",
                "billing_quantity": 0.1,
                "unit_price_usd": 0.15,
            },
        )
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj.db_spend_update_writer,
        "report_external_spend",
        report_spend,
    )
    monkeypatch.setattr(
        spend_management_endpoints.external_spend_cost_calculator,
        "resolve",
        resolve_cost,
    )
    auth = UserAPIKeyAuth(
        api_key="hashed-worker-key",
        key_alias="video-workers",
        user_id="worker-user",
        team_id="video-team",
        org_id="video-org",
    )
    request = ExternalSpendReportRequest(
        provider="fal",
        external_model="fal-ai/gemini-3.1-flash-tts",
        usage=ExternalSpendUsage(unit="billable_units", quantity=0.1),
        request_id="fal-request-1",
        end_user="course-user",
        tags=["course:course-1", "video:video-1"],
        metadata={"course_id": "course-1"},
    )

    response = await report_external_spend(request, auth)

    call = report_spend.await_args.kwargs
    payload = call["payload"]
    assert response.created is True
    assert response.status == "resolved"
    assert response.spend == 0.015
    assert response.request_id == "external:fal:fal-request-1"
    assert call["hashed_token"] == "hashed-worker-key"
    assert call["user_id"] == "worker-user"
    assert call["team_id"] == "video-team"
    assert call["org_id"] == "video-org"
    assert call["end_user_id"] == "course-user"
    assert payload["api_key"] == "hashed-worker-key"
    assert payload["user"] == "worker-user"
    assert payload["team_id"] == "video-team"
    assert payload["organization_id"] == "video-org"
    assert payload["end_user"] == "course-user"
    assert payload["request_tags"] == '["course:course-1", "video:video-1"]'
    assert json.loads(payload["metadata"])["spend_logs_metadata"]["billing_quantity"] == 0.1
    resolve_cost.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_external_spend_pricing_failure_is_recorded_as_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import spend_management_endpoints

    report_unresolved = AsyncMock(return_value=True)
    report_spend = AsyncMock()
    failed_tracking_alert = AsyncMock()
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj.db_spend_update_writer,
        "report_unresolved_provider_cost",
        report_unresolved,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj.db_spend_update_writer,
        "report_external_spend",
        report_spend,
    )
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "failed_tracking_alert", failed_tracking_alert)
    monkeypatch.setattr(
        spend_management_endpoints.external_spend_cost_calculator,
        "resolve",
        AsyncMock(side_effect=httpx.ConnectError("pricing unavailable")),
    )
    request = ExternalSpendReportRequest(
        provider="fal",
        external_model="fal-ai/gemini-3.1-flash-tts",
        usage=ExternalSpendUsage(unit="billable_units", quantity=0.1),
        request_id="fal-request-1",
        end_user="course-user",
        tags=["course:course-1", "video:video-1"],
        metadata={"course_id": "course-1"},
    )

    response = await report_external_spend(
        request,
        UserAPIKeyAuth(
            api_key="hashed-worker-key",
            user_id="worker-user",
            team_id="video-team",
            org_id="video-org",
        ),
    )

    unresolved = report_unresolved.await_args.kwargs["unresolved_cost"]
    assert response.status == "unresolved"
    assert response.spend is None
    assert response.error == "External provider pricing lookup failed: ConnectError"
    assert unresolved.tracking_id == "external:fal:fal-request-1"
    assert unresolved.evidence == {"usage_unit": "billable_units", "usage_quantity": 0.1}
    assert report_unresolved.await_args.kwargs["request_tags"] == ["course:course-1", "video:video-1"]
    report_spend.assert_not_awaited()
    failed_tracking_alert.assert_awaited_once()


def test_external_spend_model_is_not_treated_as_routing_model() -> None:
    request_data = ExternalSpendReportRequest(
        provider="fal",
        external_model="fal-ai/gemini-3.1-flash-tts",
        usage=ExternalSpendUsage(unit="billable_units", quantity=0.1),
        request_id="fal-request-1",
    ).model_dump()

    assert request_data["external_model"] == "fal-ai/gemini-3.1-flash-tts"
    assert get_model_from_request(request_data, "/spend/report") is None


@pytest.mark.asyncio
async def test_external_spend_calculator_resolves_fal_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAL_AI_API_KEY", "fal-key")
    pricing_fetcher = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "prices": [
                    {
                        "endpoint_id": "fal-ai/gemini-3.1-flash-tts",
                        "unit_price": 0.15,
                        "unit": "1000 characters",
                        "currency": "USD",
                    }
                ]
            },
            request=httpx.Request(
                "GET",
                "https://api.fal.ai/v1/models/pricing",
            ),
        )
    )
    cache = DualCache()
    calculator = ExternalSpendCostCalculator(
        fal_pricing_fetcher=pricing_fetcher,
        pricing_cache=cache,
    )
    request = ExternalSpendReportRequest(
        provider="fal",
        external_model="fal-ai/gemini-3.1-flash-tts",
        usage=ExternalSpendUsage(unit="billable_units", quantity=0.137),
        request_id="fal-request-1",
    )

    first = await calculator.resolve(request)
    second = await calculator.resolve(request)

    assert first == second
    assert first.spend == pytest.approx(0.02055)
    assert first.metadata["billing_quantity"] == 0.137
    assert first.metadata["unit_price_usd"] == 0.15
    pricing_fetcher.assert_awaited_once_with(
        "fal-ai/gemini-3.1-flash-tts",
        "Key fal-key",
    )


@pytest.mark.asyncio
async def test_external_spend_duplicate_only_applies_rollups_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_many = AsyncMock(side_effect=[1, 0])
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_spendlogs=SimpleNamespace(create_many=create_many)),
        jsonify_object=lambda value: value,
    )
    writer = DBSpendUpdateWriter()
    batch_updates = AsyncMock()
    monkeypatch.setattr(writer, "_batch_database_updates", batch_updates)
    payload = cast(SpendLogsPayload, {"request_id": "external:fal:request-1"})
    params = {
        "payload": payload,
        "response_cost": 0.015,
        "user_id": "worker-user",
        "hashed_token": "hashed-worker-key",
        "team_id": "video-team",
        "org_id": "video-org",
        "end_user_id": "course-user",
        "prisma_client": prisma_client,
        "user_api_key_cache": DualCache(),
        "litellm_proxy_budget_name": None,
    }

    first_created = await writer.report_external_spend(**params)
    duplicate_created = await writer.report_external_spend(**params)

    assert first_created is True
    assert duplicate_created is False
    assert create_many.await_count == 2
    batch_updates.assert_awaited_once()


@pytest.mark.asyncio
async def test_external_spend_accepts_batch_payload_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_many = AsyncMock(return_value=SimpleNamespace(count=1))
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_spendlogs=SimpleNamespace(create_many=create_many)),
        jsonify_object=lambda value: value,
    )
    writer = DBSpendUpdateWriter()
    batch_updates = AsyncMock()
    monkeypatch.setattr(writer, "_batch_database_updates", batch_updates)

    created = await writer.report_external_spend(
        payload=cast(SpendLogsPayload, {"request_id": "external:fal:request-1"}),
        response_cost=0.015,
        user_id="worker-user",
        hashed_token="hashed-worker-key",
        team_id="video-team",
        org_id="video-org",
        end_user_id="course-user",
        prisma_client=prisma_client,
        user_api_key_cache=DualCache(),
        litellm_proxy_budget_name=None,
    )

    assert created is True
    batch_updates.assert_awaited_once()


@pytest.mark.asyncio
async def test_unresolved_provider_cost_is_idempotent_and_excluded_from_spend() -> None:
    create_many = AsyncMock(side_effect=[1, 0])
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_errorlogs=SimpleNamespace(create_many=create_many)),
        jsonify_object=lambda value: value,
    )
    writer = DBSpendUpdateWriter()

    params = {
        "unresolved_cost": UnresolvedProviderCost(
            provider="fal",
            tracking_id="external:fal:request-1",
            model="fal-ai/gemini-3.1-flash-tts",
            reason="pricing unavailable",
            evidence={"usage_quantity": 0.1},
            metadata={"course_id": "course-1"},
        ),
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "request_tags": ["course:course-1", "video:video-1"],
        "end_user_id": "course-user",
        "user_id": "worker-user",
        "team_id": "video-team",
        "org_id": "video-org",
        "api_base": "https://api.fal.ai/v1/models/pricing",
        "prisma_client": prisma_client,
    }

    assert await writer.report_unresolved_provider_cost(**params) is True
    assert await writer.report_unresolved_provider_cost(**params) is False
    payload = create_many.await_args_list[0].kwargs["data"][0]
    assert payload["request_id"] == "unresolved-cost:external:fal:request-1"
    assert payload["exception_type"] == "UnresolvedProviderCost"
    assert payload["status_code"] == "cost_unresolved"
    assert payload["request_kwargs"]["tags"] == ["course:course-1", "video:video-1"]
    assert not hasattr(prisma_client.db, "litellm_spendlogs")
