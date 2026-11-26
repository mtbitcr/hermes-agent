"""The words for each part of a batch on the Decisions answer (release card 13).

Every answer is read through the safe reader, ``GET`` of the release view, over the real release record.
"""

from __future__ import annotations

import pytest

from hermes_cli import release_ledger as ledger
from hermes_cli.owner_workspace import owner_title
from tests.gateway.test_api_server_release import (  # noqa: F401  (root and units are fixtures)
    VIEW,
    _accepted,
    _client,
    _commit,
    _ledger,
    _merge,
    root,
    units,
)

APP_PR = "https://github.com/mtbitcr/raphael-workspace/pull/7"
PLATFORM_TITLE = "Show the release decision"
APP_TITLE = "Keep the owner page fast"

BEFORE = {
    "platform": "These changes update Raphael's platform.",
    "owner_app": "These changes update your Raphael app. It restarts once, is checked, rehearses the way back,"
    " and goes forward again.",
    "both": "These changes update Raphael's platform first, then your Raphael app."
    " If either part fails, both go back.",
}
AFTER = {
    "platform": "Raphael's platform now runs these changes.",
    "owner_app": "Your Raphael app now runs these changes.",
    "both": "Raphael's platform and your Raphael app now run these changes.",
}
WENT_BACK = "It was rolled back. Its changes went back to the waiting batch. Both parts went back."


def _merge_app(label: str, title: str) -> dict:
    """Record one merged change of the owner app's repository; returns its batch."""
    return _ledger(
        ledger.record_merge,
        merge_commit=_commit(label),
        pr_url=APP_PR,
        reviewed_base=_commit(f"{label}-base"),
        reviewed_head=_commit(f"{label}-head"),
        reviewed_tree=_commit(f"{label}-tree"),
        tier=1,
        card_id=f"t_{_commit(label)[:8]}",
        title=title,
        on_main=True,
    )["batch"]


def _batch(parts: str) -> dict:
    """The waiting batch with the platform part, the owner app part or both."""
    batch = None
    if parts in ("platform", "both"):
        batch = _merge("platform", PLATFORM_TITLE)
    if parts in ("owner_app", "both"):
        batch = _merge_app("app", APP_TITLE)
    return batch


def _begin_parts(batch: dict) -> None:
    """The release run begins ``batch`` with the versions of the parts it has."""
    platform, app = ledger._parts(batch["members"])
    prev = _commit("live")
    new = platform[-1]["merge_commit"] if platform else prev
    versions = {"app_prev": _commit("app-live"), "app_new": app[-1]["merge_commit"]} if app else {}
    _ledger(ledger.begin_release, batch["batch_id"], prev=prev, new=new, **versions)


async def _view() -> dict:
    async with _client() as client:
        return await (await client.get(VIEW)).json()


async def _release_of(parts: str, outcome: str | None) -> dict:
    """The release object once the batch of ``parts`` is accepted, begun and, as asked, ended."""
    batch = _accepted(_batch(parts))
    _begin_parts(batch)
    if outcome:
        _ledger(ledger.finish_release, batch["batch_id"], outcome=outcome)
    return (await _view())["release"]


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", ["platform", "owner_app", "both"])
async def test_the_waiting_batch_says_before_accept_which_parts_it_updates(root, parts):
    _batch(parts)

    waiting = (await _view())["waiting"]

    assert waiting["parts_sentence"] == BEFORE[parts]


@pytest.mark.asyncio
async def test_the_waiting_batch_lists_its_titles_under_the_part_each_belongs_to(root):
    _batch("both")

    waiting = (await _view())["waiting"]

    assert waiting["parts"] == {
        "platform": [owner_title(PLATFORM_TITLE)],
        "owner_app": [owner_title(APP_TITLE)],
    }
    assert waiting["titles"] == [owner_title(PLATFORM_TITLE), owner_title(APP_TITLE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", ["platform", "owner_app"])
async def test_a_part_without_changes_is_empty(root, parts):
    _batch(parts)

    waiting = (await _view())["waiting"]

    empty = "owner_app" if parts == "platform" else "platform"
    assert set(waiting["parts"]) == {"platform", "owner_app"}
    assert waiting["parts"][empty] == []
    assert len(waiting["parts"][parts]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", ["platform", "owner_app", "both"])
async def test_a_released_release_says_which_parts_now_run_the_changes(root, parts):
    release = await _release_of(parts, "released")

    assert release["released_parts_sentence"] == AFTER[parts]
    assert release["parts_sentence"] == BEFORE[parts]


@pytest.mark.asyncio
async def test_a_release_lists_its_titles_under_the_part_each_belongs_to(root):
    release = await _release_of("both", "released")

    assert release["parts"] == {
        "platform": [owner_title(PLATFORM_TITLE)],
        "owner_app": [owner_title(APP_TITLE)],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["accepted", "releasing", "failed"])
async def test_a_release_that_has_not_ended_says_nothing_about_what_now_runs(root, state):
    batch = _accepted(_batch("both"))
    if state != "accepted":
        _begin_parts(batch)
    if state == "failed":
        _ledger(ledger.finish_release, batch["batch_id"], outcome="failed")

    release = (await _view())["release"]

    assert release["state"] == state
    assert release["released_parts_sentence"] == ""
    assert release["parts_sentence"] == BEFORE["both"]


@pytest.mark.asyncio
async def test_a_batch_that_went_back_with_both_parts_says_so_after_its_existing_sentence(root):
    release = await _release_of("both", "restored")

    assert release["outcome_text"] == "It was rolled back. Its changes went back to the waiting batch."
    assert release["released_parts_sentence"] == WENT_BACK


@pytest.mark.asyncio
async def test_a_batch_that_went_back_takes_its_parts_from_its_stored_versions(root):
    release = await _release_of("both", "restored")
    waiting = (await _view())["waiting"]

    # The folded batch kept no member: only its stored PREV, NEW, APP_PREV and APP_NEW name its parts.
    assert (release["titles"], release["count"]) == ([], 0)
    assert release["parts"] == {"platform": [], "owner_app": []}
    assert release["parts_sentence"] == BEFORE["both"]
    assert release["released_parts_sentence"] == WENT_BACK
    # Today's waiting batch holds the changes of both parts and says nothing different about the release.
    assert waiting["parts_sentence"] == BEFORE["both"]


@pytest.mark.asyncio
async def test_a_stored_platform_only_batch_that_went_back_is_not_read_as_both_from_today(root):
    batch = _accepted(_batch("platform"))
    _begin_parts(batch)
    _ledger(ledger.finish_release, batch["batch_id"], outcome="restored")
    _merge_app("app", APP_TITLE)  # today's waiting batch now holds a platform and an owner app change

    view = await _view()

    assert view["waiting"]["parts_sentence"] == BEFORE["both"]
    assert view["release"]["parts_sentence"] == BEFORE["platform"]
    assert view["release"]["released_parts_sentence"] == ""


@pytest.mark.asyncio
async def test_a_stored_owner_app_only_batch_that_went_back_is_read_from_its_stored_versions(root):
    release = await _release_of("owner_app", "restored")

    assert release["parts_sentence"] == BEFORE["owner_app"]
    assert release["released_parts_sentence"] == ""


@pytest.mark.asyncio
async def test_a_refused_batch_does_not_say_both_parts_went_back(root):
    release = await _release_of("both", "refused")

    assert release["outcome_text"].startswith("It was refused before anything changed.")
    assert release["parts_sentence"] == BEFORE["both"]
    assert release["released_parts_sentence"] == ""


@pytest.mark.asyncio
async def test_every_key_and_value_of_today_stays_beside_the_new_keys(root):
    _merge("a", PLATFORM_TITLE, tier=0)
    batch = _merge_app("b", APP_TITLE)

    waiting = (await _view())["waiting"]

    assert {key: waiting[key] for key in waiting if key not in ("parts", "parts_sentence")} == {
        "batch_id": batch["batch_id"],
        "status": "Waiting for your decision.",
        "titles": [owner_title(PLATFORM_TITLE), owner_title(APP_TITLE)],
        "count": 2,
        "tier": 1,
        "tier_sentence": "Tier 1, medium risk: the release goes forward once.",
        "version": batch["version"],
        "digest": batch["digest"],
        "actions": ["accept", "defer"],
    }
    assert set(waiting) == {
        "batch_id", "status", "titles", "count", "tier", "tier_sentence", "version", "digest", "actions",
        "parts", "parts_sentence",
    }

    release = await _release_of("both", "restored")
    new_keys = {"parts", "parts_sentence", "released_parts_sentence"}
    assert {key: release[key] for key in release if key not in new_keys} == {
        "batch_id": release["batch_id"],
        "state": "folded",
        "status": "Ended without being released.",
        "unit_running": False,
        "unit": "Its release unit is not running.",
        "outcome": "restored",
        "outcome_text": "It was rolled back. Its changes went back to the waiting batch.",
        "titles": [],
        "count": 0,
        "outcome_at": release["outcome_at"],
        "actions": [],
    }
    assert type(release["outcome_at"]) is int
