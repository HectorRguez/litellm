from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching import DualCache
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.hooks.proxy_track_cost_callback import (
    _ProxyDBLogger,
    _update_database_and_spend_counters,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.utils import StandardLoggingPayload, UnresolvedProviderCost


@pytest.mark.parametrize(
    "call_type",
    ["video_content", "avideo_content", "video_status", "avideo_status"],
)
def test_video_provider_cost_uses_standard_logging_id(call_type: str) -> None:
    now = datetime.now(timezone.utc)
    payload = get_logging_payload(
        kwargs={
            "call_type": call_type,
            "litellm_call_id": "status-call-id",
            "litellm_params": {"metadata": {}},
            "standard_logging_object": cast(
                StandardLoggingPayload,
                {
                    "id": "openrouter-video-cost:generation-123",
                    "metadata": {},
                    "request_tags": ["video:provider-cost-audit"],
                    "hidden_params": {},
                    "model_map_information": {},
                },
            ),
        },
        response_obj={
            "id": "video-original-provider-id",
            "object": "video",
            "status": "completed",
        },
        start_time=now,
        end_time=now,
    )

    assert payload["request_id"] == "openrouter-video-cost:generation-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_id",
    [
        "fal-video-cost:fal-request-123",
        "openrouter-video-cost:generation-123",
    ],
)
async def test_update_database_provider_cost_is_inserted_idempotently(request_id: str) -> None:
    db_writer = DBSpendUpdateWriter()
    db_writer.report_external_spend = AsyncMock(side_effect=[True, False])
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma_client = MagicMock()

    def _logging_payload(**_kwargs):
        now = datetime.now()
        return {
            "request_id": request_id,
            "startTime": now,
            "endTime": now,
            "model": "provider/model",
            "custom_llm_provider": "provider",
        }

    call_kwargs = {
        "token": "hashed-worker-key",
        "user_id": "worker-user",
        "end_user_id": "course-user",
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "team_id": "video-team",
        "org_id": "video-org",
        "completion_response": {"id": "provider-cost-response"},
        "response_cost": 1.2096,
        "kwargs": {"model": "provider/model", "custom_llm_provider": "provider"},
    }

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            side_effect=_logging_payload,
        ),
        patch("litellm.proxy.db.db_spend_update_writer.asyncio.create_task") as mock_create_task,
    ):
        first_created = await db_writer.update_database(**call_kwargs)
        duplicate_created = await db_writer.update_database(**call_kwargs)

    assert first_created is True
    assert duplicate_created is False
    assert db_writer.report_external_spend.await_count == 2
    db_writer._insert_spend_log_to_db.assert_not_awaited()
    mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_external_spend_duplicate_only_applies_rollups_once() -> None:
    create_many = AsyncMock(side_effect=[1, 0])
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_spendlogs=SimpleNamespace(create_many=create_many)),
        jsonify_object=lambda value: value,
    )
    writer = DBSpendUpdateWriter()
    writer._batch_database_updates = AsyncMock()
    params = {
        "payload": cast(SpendLogsPayload, {"request_id": "external:fal:request-1"}),
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

    assert await writer.report_external_spend(**params) is True
    assert await writer.report_external_spend(**params) is False
    assert create_many.await_count == 2
    writer._batch_database_updates.assert_awaited_once()
    assert "user_api_key_cache" not in writer._batch_database_updates.await_args.kwargs


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_skips_duplicate_provider_cost() -> None:
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(return_value=False)
    increment_spend_counters = AsyncMock()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
        new_callable=AsyncMock,
    ) as mock_release_budget_reservation:
        recorded = await _update_database_and_spend_counters(
            proxy_logging_obj=proxy_logging_obj,
            increment_spend_counters=increment_spend_counters,
            user_api_key="test_api_key",
            user_id="test_user_id",
            end_user_id="test_end_user_id",
            team_id="test_team_id",
            org_id="test_org_id",
            kwargs={},
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_cost=1.2096,
            budget_reservation=budget_reservation,
            request_tags=["video:provider-cost-audit"],
        )

    assert recorded is False
    increment_spend_counters.assert_not_awaited()
    mock_release_budget_reservation.assert_awaited_once_with(
        budget_reservation=budget_reservation,
    )


@pytest.mark.asyncio
async def test_track_cost_callback_skips_duplicate_provider_cost_rollups() -> None:
    logger = _ProxyDBLogger()
    kwargs = {
        "call_type": "avideo_status",
        "model": "openrouter/bytedance/seedance-2.0",
        "litellm_params": {
            "metadata": {
                "user_api_key": "hashed-worker-key",
                "user_api_key_user_id": "worker-user",
                "tags": ["video:provider-cost-audit"],
            }
        },
        "standard_logging_object": {
            "response_cost": 1.2096,
            "request_tags": ["video:provider-cost-audit"],
            "metadata": {},
        },
    }

    with (
        patch(
            "litellm.proxy.proxy_server.increment_spend_counters",
            new_callable=AsyncMock,
        ) as mock_increment_spend_counters,
        patch(
            "litellm.proxy.proxy_server.update_cache",
            new_callable=AsyncMock,
        ) as mock_update_cache,
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging,
    ):
        mock_proxy_logging.db_spend_update_writer.update_database = AsyncMock(return_value=False)
        mock_proxy_logging.slack_alerting_instance.customer_spend_alert = AsyncMock()
        mock_proxy_logging.failed_tracking_alert = AsyncMock()

        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response={"id": "openrouter-video-cost:generation-123"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    mock_increment_spend_counters.assert_not_awaited()
    mock_update_cache.assert_not_awaited()
    mock_proxy_logging.slack_alerting_instance.customer_spend_alert.assert_not_awaited()
    mock_proxy_logging.failed_tracking_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_track_cost_callback_records_unresolved_cost_without_spend() -> None:
    logger = _ProxyDBLogger()
    unresolved = UnresolvedProviderCost(
        provider="openrouter",
        tracking_id="openrouter-video-cost:generation-123",
        model="openrouter/bytedance/seedance-2.0",
        reason="OpenRouter completed video response omitted usage.cost",
        evidence={"status": "completed"},
    )
    kwargs = {
        "call_type": "avideo_status",
        "model": unresolved.model,
        "provider_cost_unresolved": unresolved.model_dump(),
        "litellm_params": {
            "metadata": {
                "user_api_key": "hashed-worker-key",
                "user_api_key_user_id": "worker-user",
                "user_api_key_team_id": "video-team",
                "user_api_key_org_id": "video-org",
                "tags": ["course:course-1", "video:video-1"],
            },
            "user_api_key_end_user_id": "course-user",
        },
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging,
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
            new_callable=AsyncMock,
        ),
    ):
        mock_proxy_logging.db_spend_update_writer.report_unresolved_provider_cost = AsyncMock(return_value=True)
        mock_proxy_logging.db_spend_update_writer.update_database = AsyncMock()
        mock_proxy_logging.failed_tracking_alert = AsyncMock()

        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response={"id": "generation-123"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    call = mock_proxy_logging.db_spend_update_writer.report_unresolved_provider_cost.await_args.kwargs
    assert call["unresolved_cost"] == unresolved
    assert call["request_tags"] == ["course:course-1", "video:video-1"]
    assert call["end_user_id"] == "course-user"
    mock_proxy_logging.db_spend_update_writer.update_database.assert_not_awaited()
    mock_proxy_logging.failed_tracking_alert.assert_awaited_once()
