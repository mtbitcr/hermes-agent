"""A required check cancelled on H while a newer CI run of H is still to make it again: a close and reopen of the
pull request starts a new CI run of the same head and cancels the old one, and until the new run creates the check,
the old run's cancelled check is the only one of its name. The review step then reads H's workflow runs, by
head_sha and page by page, and waits as for a pending check while GitHub shows any of them not completed, on any
page; a cancelled check with every run of H completed, and a failed check, are red as before, and a listing of H's
runs not read whole decides nothing. The review cards world, whose GitHub table here also answers H's workflow
runs."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import board  # noqa: F401  (a fixture)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    CHECK, NEWER, OLDER, _cards, _ready, _source, _state, world,
)

PULL, RUNS = f"/repos/{REPO}/pulls/41", f"/repos/{REPO}/actions/runs"


@pytest.fixture
def runs(world, monkeypatch):
    """The review cards world, whose table also answers H's workflow runs from gh["runs"], newest first, as GitHub
    lists them: 100 a page, the page the request names (the first by default), and total_count the number of all
    of them. Each is a run of H (ids from 700) with the fields it lists, which may name another head. None answers
    404, as the world's table does; gh["runs_page"] maps a page's number to the answer that page gets instead."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange  # the world's: it records every call, and answers 404 for H's workflow runs
    gh["runs"], gh["runs_page"] = None, {}

    def exchange(method, target, authorization, payload=None):
        answer = table(method, target, authorization, payload)
        if (method, urlsplit(target).path) != ("GET", RUNS) or gh["runs"] is None:
            return answer
        page = int(parse_qs(urlsplit(target).query).get("page", ["1"])[0])
        if page in gh["runs_page"]:
            return gh["runs_page"][page]
        found = [{"id": 700 + n, "head_sha": gh["head"], **run} for n, run in enumerate(gh["runs"])]
        return 200, {"total_count": len(found), "workflow_runs": found[(page - 1) * 100:page * 100]}

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _red(head, conclusion):
    """The one waiting record of a red decision on H's required check, run 900."""
    return [("delivery_review_waiting", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_waiting",
        "check_runs": [{"id": 900, "conclusion": conclusion}]})]


def _cancelled(runs, status="in_progress"):
    """A's passes: H's only check of its name is the older CI run's, cancelled in the older check suite, while
    GitHub shows the newer run of H ``status`` above the older run, completed and cancelled. Three passes; returns
    the source card, H, the source card's delivery records before the passes and its row."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"] = [("completed", "cancelled", OLDER)]
    gh["runs"] = [{"status": status, "conclusion": None}, {"status": "completed", "conclusion": "cancelled"}]
    events, source = _events(kb, tid), _source(kb, tid)
    for _ in range(3):
        _tick(kb)
    return tid, head, events, source


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
def test_a_cancelled_check_waits_as_a_pending_one_while_a_run_of_h_is_not_completed(runs, status):
    """A: each pass reads H's workflow runs once, by head_sha, right after the decision request, and stops there as
    a pending check's pass does: no waiting record, no review card, the row open, the source card unchanged, no
    POST and no other request."""
    kb, root, repo, gh = runs
    tid, head, events, source = _cancelled(runs, status)

    assert (_events(kb, tid)[len(events):], _cards(kb), _state(kb, head), _source(kb, tid)) == ([], [], "open", source)
    assert gh["queries"] == [
        (PULL, {}),
        (f"/repos/{REPO}/commits/{head}/check-runs", {"filter": ["latest"], "check_name": [CHECK], "per_page": ["100"]}),
        (RUNS, {"head_sha": [head], "per_page": ["100"]})] * 3
    assert [call for call in gh["calls"] if call[0] != "GET"] == [] and set(gh["steps"]) == {"read_checks"}


def test_the_newer_runs_green_check_then_gets_h_its_review_cards_and_the_work_never_returns(runs):
    """B, continuing from A: the newer CI run of H completes and its check, in the newer check suite, is green, so
    the next pass creates H's review card. No pass recorded a red decision, so the work never went back to its
    builder."""
    kb, root, repo, gh = runs
    tid, head, events, source = _cancelled(runs)
    gh["copies"] = [("completed", "cancelled", OLDER), ("completed", "success", NEWER)]
    gh["runs"][0].update(status="completed", conclusion="success")

    _tick(kb)

    assert len(_cards(kb)) == 1 and _state(kb, head) == "review_cards_created"
    assert [kind for kind, _ in _events(kb, tid)[len(events):] if kind == "delivery_review_waiting"] == []


@pytest.mark.parametrize("found", [
    [],
    [{"status": "completed", "conclusion": "cancelled"}],
    [{"status": "completed", "conclusion": "success"}, {"status": "completed", "conclusion": "cancelled"}],
    [{"status": "in_progress", "conclusion": None, "head_sha": "e" * 40},
     {"status": "completed", "conclusion": "cancelled"}],
], ids=["no_run", "the_cancelled_run", "every_run_completed", "a_run_of_another_head"])
def test_a_cancelled_check_is_red_when_every_run_of_h_is_completed(runs, found):
    """C: GitHub shows no run of H that is not completed, so the cancelled check is red, as before:
    red_check_waiting with its id and conclusion, once for H, no card and the source card unchanged."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"], gh["runs"] = [("completed", "cancelled", OLDER)], found
    events, source = _events(kb, tid), _source(kb, tid)

    for _ in range(3):
        _tick(kb)

    assert _events(kb, tid)[len(events):] == _red(head, "cancelled")
    assert (_cards(kb), _state(kb, head), _source(kb, tid)) == ([], "open", source)


def test_a_failed_check_is_red_even_while_a_run_of_h_is_running(runs):
    """D: a failed required check is red, as before, though GitHub shows a run of H in progress: no workflow run is
    read before the decision, red_check_waiting is recorded once for H and no card is made."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"] = [("completed", "failure", OLDER)]
    gh["runs"] = [{"status": "in_progress", "conclusion": None}, {"status": "completed", "conclusion": "failure"}]
    events, source = _events(kb, tid), _source(kb, tid)

    for _ in range(3):
        _tick(kb)

    assert _events(kb, tid)[len(events):] == _red(head, "failure")
    assert (_cards(kb), _state(kb, head), _source(kb, tid)) == ([], "open", source)
    assert [path for _, path in gh["calls"]][:4] == [PULL, f"/repos/{REPO}/commits/{head}/check-runs", PULL, RUNS]


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_a_cancelled_check_waits_while_a_run_of_h_beyond_the_first_100_is_not_completed(runs, status):
    """E: GitHub lists H's 102 runs over two pages, 100 completed ones first, then the newer run of H ``status``
    and the older run, completed and cancelled: each pass reads the first page, by head_sha, then the second, and
    stops there as a pending check's pass does: no waiting record, no review card, the row open, the source card
    unchanged, no POST and no other request."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"] = [("completed", "cancelled", OLDER)]
    gh["runs"] = [{"status": "completed", "conclusion": "success"}] * 100 + [
        {"status": status, "conclusion": None}, {"status": "completed", "conclusion": "cancelled"}]
    events, source = _events(kb, tid), _source(kb, tid)

    for _ in range(3):
        _tick(kb)

    assert (_events(kb, tid)[len(events):], _cards(kb), _state(kb, head), _source(kb, tid)) == ([], [], "open", source)
    assert gh["queries"] == [
        (PULL, {}),
        (f"/repos/{REPO}/commits/{head}/check-runs",
         {"filter": ["latest"], "check_name": [CHECK], "per_page": ["100"]}),
        (RUNS, {"head_sha": [head], "per_page": ["100"]}),
        (RUNS, {"head_sha": [head], "per_page": ["100"], "page": ["2"]})] * 3
    assert [call for call in gh["calls"] if call[0] != "GET"] == [] and set(gh["steps"]) == {"read_checks"}


def test_a_cancelled_check_is_red_when_every_run_of_h_over_two_pages_is_completed(runs):
    """F: GitHub lists H's 102 runs over two pages, every one completed, the cancelled run last, so the cancelled
    check is red, as before: red_check_waiting with its id and conclusion, once for H, no card and the source card
    unchanged."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"] = [("completed", "cancelled", OLDER)]
    gh["runs"] = [{"status": "completed", "conclusion": "success"}] * 101 + [
        {"status": "completed", "conclusion": "cancelled"}]
    events, source = _events(kb, tid), _source(kb, tid)

    for _ in range(3):
        _tick(kb)

    assert _events(kb, tid)[len(events):] == _red(head, "cancelled")
    assert (_cards(kb), _state(kb, head), _source(kb, tid)) == ([], "open", source)


@pytest.mark.parametrize("answer", [(404, None), (200, {"total_count": 102, "workflow_runs": []})],
                         ids=["page_2_unreadable", "page_2_without_runs"])
def test_a_listing_of_the_runs_of_h_not_read_whole_decides_nothing(runs, answer):
    """G: the first page lists 100 completed runs of H and a total_count of 102, but the second cannot be read or
    holds no run, so the listing is not read whole and no pass decides, red or otherwise: no waiting record, no
    review card, the row open, the source card unchanged and no POST; a later pass reads it again."""
    kb, root, repo, gh = runs
    tid, head = _ready(runs)
    gh["copies"] = [("completed", "cancelled", OLDER)]
    gh["runs"] = [{"status": "completed", "conclusion": "success"}] * 101 + [
        {"status": "completed", "conclusion": "cancelled"}]
    gh["runs_page"] = {2: answer}
    events, source = _events(kb, tid), _source(kb, tid)

    for _ in range(3):
        _tick(kb)

    assert (_events(kb, tid)[len(events):], _cards(kb), _state(kb, head), _source(kb, tid)) == ([], [], "open", source)
    assert [call for call in gh["calls"] if call[0] != "GET"] == []
