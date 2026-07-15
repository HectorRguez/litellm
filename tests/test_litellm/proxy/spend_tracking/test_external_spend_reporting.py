from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from litellm.caching import DualCache
from litellm.proxy._types import (
    ExternalSpendReportRequest,
    SpendLogsPayload,
    UserAPIKeyAuth,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    report_external_spend,
)


def test_external_spend_report_rejects_negative_spend() -> None:
    with pytest.raises(ValidationError):
        ExternalSpendReportRequest(
            provider="fal",
            model="fal-ai/gemini-3.1-flash-tts",
            spend=-0.01,
            request_id="request-1",
        )


@pytest.mark.asyncio
async def test_external_spend_report_uses_authenticated_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    report_spend = AsyncMock(return_value=True)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj.db_spend_update_writer,
        "report_external_spend",
        report_spend,
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
        model="fal-ai/gemini-3.1-flash-tts",
        spend=0.015,
        request_id="fal-request-1",
        end_user="course-user",
        tags=["course:course-1", "video:video-1"],
        metadata={"course_id": "course-1"},
    )

    response = await report_external_spend(request, auth)

    call = report_spend.await_args.kwargs
    payload = call["payload"]
    assert response.created is True
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


@pytest.mark.asyncio
async def test_external_spend_duplicate_only_applies_rollups_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_many = AsyncMock(side_effect=[SimpleNamespace(count=1), SimpleNamespace(count=0)])
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
