"""Save daily review and Undo examples from the real owner HTTP routes."""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_owner_review_brief import (
    app,
    closed_suggestion,  # noqa: F401 (fixture)
    project_review,  # noqa: F401 (fixture)
)


@pytest.mark.asyncio
async def test_the_saved_brief_keeps_the_existing_summary(project_review, owner_payload_example):
    path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot"
    async with TestClient(TestServer(app())) as client:
        old_response = await client.get(path + "?capabilities=project_summary")
        response = await client.get(path + "?capabilities=project_summary,project_review_brief")
        assert old_response.status == response.status == 200
        old, current = await old_response.json(), await response.json()

    saved = owner_payload_example("owner_snapshot", "project_review_brief", {
        "status": response.status, "body": current,
    })
    for data in (current["data"], saved["body"]["data"]):
        assert set(data["raphael_summary"]) == {"outcome", "next_action", "owner_decision", "written_at", "revision"}
        assert data["raphael_review"]["latest"]["headline"] == "The room is booked."
        assert data["raphael_review"]["history"][0]["headline"] == "The room is booked."
    assert current["data"]["raphael_summary"] == old["data"]["raphael_summary"]
    assert set(current["data"]) - set(old["data"]) == {"raphael_review"}


@pytest.mark.asyncio
async def test_the_saved_closed_list_and_undo_replies_keep_the_owner_action_fenced(closed_suggestion, owner_payload_example):
    async with TestClient(TestServer(app())) as client:
        response = await client.get("/v1/owner-workspace/decisions?capabilities=closed_suggestions_v1")
        assert response.status == 200
        listed = await response.json()
        saved_list = owner_payload_example("owner_decisions", "decision_list_closed", {"status": response.status, "body": listed})
        for view in (listed, saved_list["body"]):
            assert view["data"] == []
            [closed] = view["closed"]
            assert (closed["title"], closed["card_title"], closed["reason"]) == (
                "Book the room", "Book a room", "The room is booked.",
            )
        [item] = listed["closed"]
        path = f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close"
        body = {"reason": "The room needs another check.", "expected_closed_at": item["closed_at"]}
        first = await client.post(path, json=body)
        assert first.status == 200
        undone = await first.json()
        saved_undo = owner_payload_example("owner_decisions", "closure_undone", {"status": first.status, "body": undone})
        assert undone["data"] == {"decision_ref": item["decision_ref"], "undone": True}
        assert saved_undo["body"]["data"]["undone"] is True
        second = await client.post(path, json=body)
        assert second.status == 404
        refused = await second.json()
        saved_refusal = owner_payload_example("owner_decisions", "closure_not_waiting", {"status": second.status, "body": refused})
        assert refused["error"]["code"] == saved_refusal["body"]["error"]["code"] == "decision_not_found"
        assert refused["error"]["message"] == saved_refusal["body"]["error"]["message"] == (
            "This closure can no longer be undone."
        )
        refreshed = await client.get("/v1/owner-workspace/decisions?capabilities=closed_suggestions_v1")
        assert refreshed.status == 200
        current = await refreshed.json()
        assert current["closed"] == []
        assert [row["decision_ref"] for row in current["data"]] == [item["decision_ref"]]
