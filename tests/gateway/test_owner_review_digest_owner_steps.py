"""The morning message names each Project in its scope that has an open owner step, also one whose heads-up was told."""

import importlib
import json
import time

import pytest

from gateway import owner_channel_pause
from hermes_cli import projects_db
from hermes_cli.owner_project_review_owner_line import owner_line
from tests.gateway.test_owner_heads_up import HOME_CHAT, root  # noqa: F401
from tests.gateway.test_owner_heads_up import _waiting
from tests.gateway.test_owner_review_digest import (  # noqa: F401
    NOTES_LINE, QUESTION, Adapter, _clock, _headline, _pause_clock, _project, _ready, _runner, _scope, ready,
)
from tests.hermes_cli.test_owner_project_review_cycle import _at
from tests.hermes_cli.test_owner_project_review_owner_line import NAME, RID, _chat, _link

ROOM_IS_BOOKED = "Meeting Plan: The room is booked."
GUEST_LIST = "Meeting Plan: Prepare the guest list."


def _line(project, waiting=0, pending=0, approvals=None):
    """The Project's line: its name and the fixed owner line of the daily brief card, taken from its one place."""
    return f"{project['name']}: {owner_line(waiting, pending, approvals)}"


def _morning(day, *lines):
    """The whole message of a morning that has advice: the heading, the notes line and ``lines``."""
    return "\n".join([f"Your daily project review ({day})", NOTES_LINE, *lines])


@pytest.mark.asyncio
async def test_a_wait_whose_heads_up_was_sent_is_named_on_the_next_morning_and_on_the_morning_after_it(
    root, ready, monkeypatch,
):
    from gateway import owner_heads_up

    clock = _clock(monkeypatch)
    adapter = Adapter()
    runner = _runner(adapter)
    _waiting(ready, title="Choose a room", reason=QUESTION)
    assert await owner_heads_up.tick(runner) == 1
    adapter.sent.clear()
    digest = importlib.import_module("gateway.owner_review_digest")
    for number in (11, 12):
        day = f"2026-11-{number}"
        clock[0] = _at(2026, 11, number, 6, 30)
        _ready(root, ready, day, clock[0])
        assert await digest.tick(runner) == 1
        # The wait is named one time, by the fixed owner line alone, and its question is not in the message.
        assert adapter.sent == [(HOME_CHAT, _morning(day, _line(ready, waiting=1), ROOM_IS_BOOKED, GUEST_LIST))]
        adapter.sent.clear()


@pytest.mark.asyncio
async def test_a_project_with_no_owner_step_has_no_such_line(root, ready):
    other = _project(root, "Garden Party")
    _headline(root, other, "2026-11-10", int(time.time()), "The garden is ready.")
    _scope(root, ready, other)
    _waiting(ready, title="Choose a room", reason=QUESTION)
    adapter = Adapter()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(_runner(adapter)) == 1
    lines = adapter.sent[0][1].splitlines()
    # Only Meeting Plan has an owner line. Garden Party has its headline and no other line.
    assert lines[:3] == ["Your daily project review (2026-11-10)", NOTES_LINE, _line(ready, waiting=1)]
    assert len(lines) == 6
    assert set(lines[3:]) == {ROOM_IS_BOOKED, "Garden Party: The garden is ready.", GUEST_LIST}


@pytest.mark.asyncio
async def test_a_plan_change_that_waits_for_approval_is_named(root, ready):
    _chat(root, NAME, RID)
    _link(root, NAME, ready["project_id"])
    adapter = Adapter()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(_runner(adapter)) == 1
    assert adapter.sent == [(HOME_CHAT, _morning(
        "2026-11-10", _line(ready, approvals=1), ROOM_IS_BOOKED, GUEST_LIST,
    ))]


@pytest.mark.asyncio
async def test_a_paused_morning_sends_nothing(root, ready, monkeypatch):
    from gateway import owner_heads_up

    clock = _pause_clock(monkeypatch)
    adapter = Adapter()
    runner = _runner(adapter)
    _waiting(ready, title="Choose a room", reason=QUESTION)
    assert await owner_heads_up.tick(runner) == 1
    adapter.sent.clear()
    # The told wait is the only news of this morning, so the pause meets a message that exists only by this card.
    with projects_db.connect_closing(root / "projects.db") as registry:
        registry.execute("UPDATE project_review_briefs SET quiet=1")
        registry.execute("UPDATE portfolio_review_briefs SET priorities='[]'")
        registry.commit()
    owner_channel_pause.pause()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    record = json.loads((root / "owner_review_digest.json").read_text(encoding="utf-8"))
    assert record["told"]["2026-11-10"]["state"] == "paused"
    owner_channel_pause.resume()
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    # The pause does not carry over: the next morning is sent and names the wait again.
    clock[0] = _at(2026, 11, 11, 6, 30)
    _ready(root, ready, "2026-11-11", clock[0])
    assert await digest.tick(runner) == 1
    assert adapter.sent == [(HOME_CHAT, _morning("2026-11-11", _line(ready, waiting=1), ROOM_IS_BOOKED, GUEST_LIST))]
