"""Card 2: with ``kanban.delivery.exact_base`` exactly true, the arm step merges an approved head only on the
base its review saw. Under one merge lock per repository, a file under the kanban root that every board and
process shares, it reads main and merges when main is the reviewed base or is already in the approved head;
else it sends nothing and records base_moved with the reviewed base, the observed main and the head. Main's
answer must name refs/heads/main. An answer that is neither 200 nor a refusal GitHub documents for a merge it
did not make is read back from GitHub once and never sent again. The merge pass compares the merge's first
parent with the main the arm checked and records delivery_merged_off_base when they differ; when git cannot
read that parent it records delivery_merged_off_base_unknown, and later passes read it again until one
delivery_merged_off_base or delivery_merged_on_base event closes that check. With the setting anything but
exactly true the arm and the merge pass are today's. Real boards and git; GitHub is the real transport with
only its exchange and App token replaced, so every call passes the allowlist, and its git origin is a local
bare repository."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _git, _spawned_worker_env, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    ARMED, OTHER, _arms, _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _published, _raw, _ready, _state, world,
)

MAIN = f"/repos/{REPO}/git/ref/heads/main"
PULL, REVIEWS, MERGE = (f"/repos/{REPO}/pulls/41{tail}" for tail in ("", "/reviews", "/merge"))
LOCK_NAME = hashlib.sha256(REPO.lower().encode()).hexdigest() + ".init.lock"
TODAY = {"delivery_id", "head", "pull_request_number", "repository", "status", "reason", "cards", "attempt"}


def _setting(home: Path, exact_base) -> None:
    """``kanban.delivery`` as a config file spells it: delivery on, and ``exact_base`` as written."""
    lines = ["kanban:", "  delivery:", "    enabled: true"] + ([f"    exact_base: {exact_base}"] if exact_base else [])
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def exact(armed, monkeypatch, tmp_path):
    """The armed world, where GitHub also answers main's ref with gh["main"], or gh["main_answer"] ("timeout":
    no answer), and runs gh["hooks"]["MAIN"] once during that read; each read of it is in gh["calls"]. GitHub's
    git origin is gh["bare"], a bare repository holding the board repository's main, as part 1's tests lay it
    out, so the merge recorder's fetch of main never reaches the network."""
    kb, root, repo, gh = armed
    from hermes_cli import kanban_delivery_github as transport

    bare = tmp_path / "remotes" / "mtbitcr" / "hermes-agent.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    _git(repo, "push", "-q", str(bare), "main")
    monkeypatch.setattr(transport, "GIT_ORIGIN", (tmp_path / "remotes").as_uri())
    inner = transport._exchange
    gh.update(main=None, main_answer=None, bare=bare)

    def exchange(method, target, authorization, payload=None):
        if (method, urlsplit(target).path) == ("GET", MAIN):
            gh["calls"].append((method, MAIN))
            gh["hooks"].pop("MAIN", lambda: None)()
            if gh["main_answer"] == "timeout":
                raise transport.GitHubTransportError("network_error")
            return gh["main_answer"] or (200, {"ref": "refs/heads/main", "object": {"sha": gh["main"]}})
        return inner(method, target, authorization, payload)

    monkeypatch.setattr(transport, "_exchange", exchange)
    return armed


def _approved_head(scene):
    """An approved head H whose review cards are done and approved by every lens: the next pass merges it.
    Built and published with exact_base off, as part 1's tests pin the start; the reviewed base is main."""
    kb, root, repo, gh = scene
    tid, head = _ready(scene)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    return tid, head, _git(repo, "rev-parse", "main")


def _moved(repo) -> str:
    """main moves on with a change H does not hold."""
    _git(repo, "commit", "-q", "--allow-empty", "-m", "merged elsewhere")
    return _git(repo, "rev-parse", "main")


def _outcomes(kb, tid, db=None):
    """The arm's outcome events on the source card: an auto_merge one or base_moved."""
    return [(kind, payload) for kind, payload in _events(kb, tid, db) if "auto_merge" in kind or "base_moved" in kind]


def _locks(root) -> list:
    """The merge lock files under the root, from the directory itself: the independent count."""
    where = root / "kanban" / "merge-locks"
    return sorted(os.listdir(where)) if where.is_dir() else []


def _intaken(head) -> list:
    """The merge commits the release intake recorded for ``head``, read from the release ledger itself."""
    from hermes_cli import release_ledger

    conn = release_ledger.connect()
    try:
        return [member["merge_commit"] for batch in release_ledger.list_batches(conn)
                for member in batch["members"] if member["reviewed_head"] == head]
    finally:
        conn.close()


def _merges(kb, tid) -> list:
    """The merge commits of the source card's delivery_merged events."""
    return [payload["merge_commit"] for kind, payload in _events(kb, tid) if kind == "delivery_merged"]


def _off_base(kb, tid) -> list:
    """The payloads of the source card's delivery_merged_off_base events."""
    return [payload for kind, payload in _events(kb, tid) if kind == "delivery_merged_off_base"]


def _off_base_unknown(kb, tid) -> list:
    """The payloads of the source card's delivery_merged_off_base_unknown events: the checks left open."""
    return [payload for kind, payload in _events(kb, tid) if kind == "delivery_merged_off_base_unknown"]


def _on_base(kb, tid) -> list:
    """The payloads of the source card's delivery_merged_on_base events."""
    return [payload for kind, payload in _events(kb, tid) if kind == "delivery_merged_on_base"]


def _fetches(monkeypatch) -> list:
    """Each fetch of main the transport makes from now on, by the checkout it fetches into."""
    from hermes_cli import kanban_delivery_github as transport

    made, fetch_main = [], transport.GitHubTransport.fetch_main

    def counted(self, checkout):
        made.append(str(checkout))
        return fetch_main(self, checkout)

    monkeypatch.setattr(transport.GitHubTransport, "fetch_main", counted)
    return made


def _failing_read(monkeypatch, merge: str, failure: str) -> dict:
    """While ``failing["now"]`` is true, the fetch of main fails as a network error does (``failure`` "fetch"),
    or git reads no first parent of ``merge`` though the fetch works ("parent_read"). failing["reads"] holds
    each git read of that first parent from now on, failed or not."""
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_github as transport

    failing, fetch_main, run_git = {"now": True, "reads": []}, transport.GitHubTransport.fetch_main, kd._run_git

    def fetch(self, checkout):
        if failure == "fetch" and failing["now"]:
            raise transport.GitHubTransportError("network_error")
        return fetch_main(self, checkout)

    def git(workdir, *args):
        if f"{merge}^1" in args:
            failing["reads"].append(args)
            if failure == "parent_read" and failing["now"]:
                return None
        return run_git(workdir, *args)

    monkeypatch.setattr(transport.GitHubTransport, "fetch_main", fetch)
    monkeypatch.setattr(kd, "_run_git", git)
    return failing


def _merge_on_put(scene, move: bool) -> dict:
    """GitHub's merge of H, made with real git in the board repository during the merge request: a two-parent
    commit with H second, on main or, when ``move``, on main after it moved on with a change H does not hold.
    GitHub then shows the pull request merged as that commit, and main at it. The dict holds the merge and its
    first parent once the merge request is sent."""
    kb, root, repo, gh = scene
    landed = {}

    def merge():
        landed["first_parent"] = _moved(repo) if move else _git(repo, "rev-parse", "main")
        _git(repo, "merge", "-q", "--no-ff", "--no-edit", gh["head"])
        landed["merge"] = gh["main"] = gh["ended"][41] = _git(repo, "rev-parse", "HEAD")

    gh["hooks"]["PUT"] = merge
    return landed


def test_the_arm_merges_when_main_is_the_reviewed_base(exact):
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")

    _tick(kb)

    assert _arms(gh) == [{"sha": head, "merge_method": "merge"}]
    assert gh["calls"][-4:] == [("GET", PULL), ("GET", REVIEWS), ("GET", MAIN), ("PUT", MERGE)]
    assert _state(kb, head) == "auto_merge_armed"
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["head"], payload["status"], payload["observed_main"]) == (
        "delivery_auto_merge_armed", head, 200, base)
    assert set(payload) == TODAY | {"observed_main"}
    assert _locks(root) == [LOCK_NAME]


def test_a_continuation_whose_head_contains_main_merges(exact):
    """Owner question 1: the publish bound H to an older base, but main is already in H, so the merge result
    is what review approved; it merges, and the next pass records the merge."""
    kb, root, repo, gh = exact
    older = _git(repo, "rev-parse", "main")
    main = _moved(repo)  # H is built on this main
    tid, head, bound = _approved_head(exact)
    assert bound == main
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "delivery_bound", {"source_task_id": tid, "head": head, "base_commit": older})
    finally:
        conn.close()
    gh["main"] = main
    _setting(root, "true")

    _tick(kb)

    assert (_arms(gh), _state(kb, head)) == ([{"sha": head, "merge_method": "merge"}], "auto_merge_armed")
    assert [payload["observed_main"] for kind, payload in _outcomes(kb, tid)] == [main]
    gh["ended"][41] = ARMED["sha"]
    _tick(kb)
    assert (_state(kb, head), len(_arms(gh))) == ("merged", 1)


@pytest.mark.parametrize("when", ["build", "review"])
def test_the_arm_sends_no_merge_and_records_base_moved_when_main_moved(exact, when):
    """main moved while H was built, or while its reviews were read: no merge, the row base_moved with the
    reviewed base, the observed main and the head, its lease untouched, and later passes send nothing."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    moved = _moved(repo)
    gh["main"] = moved if when == "build" else base
    if when == "review":
        gh["hooks"]["GET"] = lambda: gh.update(main=moved)  # during the reviews read
    _setting(root, "true")

    _tick(kb)

    assert (_arms(gh), _state(kb, head)) == ([], "base_moved")
    assert gh["calls"][-3:] == [("GET", PULL), ("GET", REVIEWS), ("GET", MAIN)]
    cards = [card["id"] for card in _raw(kb.kanban_db_path(), "SELECT id FROM tasks WHERE idempotency_key LIKE 'review:%'")]
    delivery = _raw(kb.kanban_db_path(), "SELECT id, publish_lease FROM kanban_deliveries WHERE source_head = ?", head)[0]
    assert _outcomes(kb, tid) == [("delivery_base_moved", {
        "delivery_id": delivery["id"], "head": head, "pull_request_number": 41, "repository": REPO,
        "reviewed_base": base, "observed_main": moved, "cards": cards})]
    assert delivery["publish_lease"] is None
    events, calls = _events(kb, tid), len(gh["calls"])

    _tick(kb)
    _tick(kb)

    assert (_arms(gh), _state(kb, head), _events(kb, tid), gh["calls"][calls:]) == (
        [], "base_moved", events, [("GET", PULL)] * 2)


@pytest.mark.parametrize("value", [None, "false", '"true"', "1", '"yes"'],
                         ids=["absent", "false", "quoted_true", "one", "quoted_yes"])
def test_with_the_setting_off_the_arm_merges_as_today(exact, value):
    """Off unless exactly true: with main moved, the arm still merges exactly as today, with the same calls,
    no main read, the same state and event payload, and no merge lock file."""
    from hermes_cli.config import load_config

    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = _moved(repo)
    _setting(root, value)
    assert (load_config().get("kanban", {}).get("delivery") or {}).get("exact_base", "absent") is not True

    _tick(kb)

    assert _arms(gh) == [{"sha": head, "merge_method": "merge"}]
    assert gh["calls"][-3:] == [("GET", PULL), ("GET", REVIEWS), ("PUT", MERGE)]
    assert ("GET", MAIN) not in gh["calls"]
    assert _state(kb, head) == "auto_merge_armed"
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, set(payload), payload["status"], payload["reason"]) == ("delivery_auto_merge_armed", TODAY, 200, None)
    assert _locks(root) == [] and not (root / "kanban" / "merge-locks").exists()


@pytest.mark.parametrize("answer", ["timeout", (502, None), (408, None), (429, None)],
                         ids=["timeout", "5xx", "408", "429"])
@pytest.mark.parametrize("merged", [True, False], ids=["merged_on_github", "not_merged"])
def test_an_unknown_answer_is_read_back_once_and_the_merge_is_never_sent_again(exact, answer, merged):
    """No status, or one that is neither 200 nor a refusal GitHub documents (a 5xx, a request timeout, a rate
    limit): the pull request and main are read once, inside the lock. Shown merged into main at H it is armed
    now and recorded merged by the next pass; shown not merged yet it stays unknown, never refused, and each
    later pass reads it again. The event keeps the answer's status and reason."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm=answer)
    if merged:  # GitHub merged it, but its answer was lost or unconfirmed
        gh["hooks"]["PUT"] = lambda: gh["ended"].__setitem__(41, ARMED["sha"]) or gh.update(main=ARMED["sha"])
    _setting(root, "true")

    _tick(kb)

    assert gh["calls"][-5:] == [("GET", REVIEWS), ("GET", MAIN), ("PUT", MERGE), ("GET", PULL), ("GET", MAIN)]
    [(kind, payload)] = _outcomes(kb, tid)
    status, reason = (None, "network_error") if answer == "timeout" else (answer[0], None)
    assert (payload["status"], payload["reason"], payload["read_head"], payload["observed_main"]) == (
        status, reason, head, base)
    if merged:
        assert (kind, payload["read_back"], payload["merge_commit"], payload["main_on_read"]) == (
            "delivery_auto_merge_armed", "merged_at_head", ARMED["sha"], ARMED["sha"])
    else:
        assert (kind, payload["read_back"], payload["main_on_read"]) == ("delivery_auto_merge_unknown", "not_merged", base)
        assert "merge_commit" not in payload

    _tick(kb)
    _tick(kb)

    assert (len(_arms(gh)), _state(kb, head)) == (1, "merged" if merged else "auto_merge_unknown")
    assert "delivery_auto_merge_refused" not in [kind for kind, payload in _events(kb, tid)]


def test_a_read_back_showing_another_head_is_unknown_and_never_recorded(exact):
    """The verifier's first case: the answer is lost and GitHub then shows the pull request merged into main at
    another head. The read-back binds its answer to H: unknown with read_back other_head, never armed, and no
    later pass records that merge, in the delivery or in the release intake."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm="timeout")
    gh["hooks"]["PUT"] = lambda: gh.update(pull_head="e" * 40, main=ARMED["sha"]) or gh["ended"].__setitem__(
        41, ARMED["sha"])
    _setting(root, "true")

    _tick(kb)
    _tick(kb)
    _tick(kb)

    assert (len(_arms(gh)), _state(kb, head)) == (1, "auto_merge_unknown")
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["read_back"], payload["read_head"], payload["merge_commit"], payload["main_on_read"]) == (
        "delivery_auto_merge_unknown", "other_head", "e" * 40, ARMED["sha"], ARMED["sha"])
    assert (_merges(kb, tid), _intaken(head)) == ([], [])


@pytest.mark.parametrize("answer", ["timeout", (408, None)], ids=["timeout", "408"])
def test_a_merge_that_completes_after_the_read_back_is_recorded_once_github_shows_it(exact, answer):
    """The verifier's second case: the merge request was still in flight when the read-back found the pull
    request not merged. The row is unknown, never refused, so the merge pass records the merge once GitHub shows
    it merged at H; the merge was sent once in all. A request timeout (408) is such an answer too."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm=answer)
    _setting(root, "true")

    _tick(kb)

    assert _state(kb, head) == "auto_merge_unknown"
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["read_back"], payload["read_head"]) == ("delivery_auto_merge_unknown", "not_merged", head)

    gh["ended"][41] = ARMED["sha"]
    gh["main"] = ARMED["sha"]
    _tick(kb)
    _tick(kb)

    assert (len(_arms(gh)), _state(kb, head), _merges(kb, tid)) == (1, "merged", [ARMED["sha"]])
    assert "delivery_auto_merge_refused" not in [kind for kind, payload in _events(kb, tid)]


@pytest.mark.parametrize("status", [403, 404, 405, 409, 422])
def test_a_documented_refusal_is_recorded_refused_without_a_read_back(exact, status):
    """An answer GitHub documents for a merge it did not make is a refusal: the row is refused with that status
    and nothing is read back, so no GitHub call follows the merge, which was sent once; later passes send and
    read nothing."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm=(status, None))
    _setting(root, "true")

    _tick(kb)

    assert (_arms(gh), _state(kb, head)) == ([{"sha": head, "merge_method": "merge"}], "auto_merge_refused")
    assert gh["calls"][-3:] == [("GET", REVIEWS), ("GET", MAIN), ("PUT", MERGE)]  # nothing is read after it
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["status"], payload["observed_main"]) == ("delivery_auto_merge_refused", status, base)
    assert set(payload) == TODAY | {"observed_main"}
    calls = len(gh["calls"])

    _tick(kb)
    _tick(kb)

    assert (len(_arms(gh)), _state(kb, head), gh["calls"][calls:]) == (1, "auto_merge_refused", [("GET", PULL)] * 2)


@pytest.mark.parametrize("answer, state", [
    ((200, None), "auto_merge_armed"), ((201, None), "auto_merge_refused"), ((400, None), "auto_merge_refused"),
    ((408, None), "auto_merge_refused"), ((429, None), "auto_merge_refused"), ((500, None), "auto_merge_refused"),
    ("timeout", "auto_merge_unknown")], ids=["200", "201", "400", "408", "429", "500", "timeout"])
def test_with_the_setting_off_each_merge_answer_is_recorded_as_today(exact, answer, state):
    """With the setting absent and main moved, the merge answer is classified exactly as released: armed on 200,
    unknown with no status, refused with any other status. One merge, no main read, and the merge is the last
    GitHub call of the pass, with today's event."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=_moved(repo), arm=answer)
    _setting(root, None)

    _tick(kb)

    assert (_state(kb, head), len(_arms(gh))) == (state, 1)
    assert ("GET", MAIN) not in gh["calls"] and gh["calls"][-1] == ("PUT", MERGE)
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, set(payload), payload["status"]) == (
        f"delivery_{state}", TODAY, None if answer == "timeout" else answer[0])


@pytest.mark.parametrize("value", ["true", None], ids=["on", "off"])
def test_the_merge_pass_records_only_a_merge_at_the_approved_head(exact, value):
    """An armed row (a 200 answer) whose pull request GitHub then shows merged into main at another head: with
    exact_base on, merge_step records nothing and the row stays armed, read again by each pass; with it off the
    same read is recorded merged, as today."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, value)
    _tick(kb)
    assert _state(kb, head) == "auto_merge_armed"
    gh.update(pull_head=OTHER, main=ARMED["sha"])
    gh["ended"][41] = ARMED["sha"]
    calls = len(gh["calls"])

    _tick(kb)
    _tick(kb)

    if value:
        assert (_state(kb, head), _merges(kb, tid), _intaken(head)) == ("auto_merge_armed", [], [])
        assert gh["calls"][calls:] == [("GET", PULL)] * 2  # each pass reads it again
    else:
        assert (_state(kb, head), _merges(kb, tid), _intaken(head)) == ("merged", [ARMED["sha"]], [ARMED["sha"]])
    assert len(_arms(gh)) == 1


@pytest.mark.parametrize("answer", ["timeout", (503, None)], ids=["timeout", "5xx"])
def test_a_failing_read_back_records_unknown_as_today(exact, answer):
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm=answer)
    gh["hooks"]["PUT"] = lambda: gh.update(main_answer=(503, None))  # main cannot be read back
    _setting(root, "true")

    _tick(kb)
    _tick(kb)
    _tick(kb)

    assert (len(_arms(gh)), _state(kb, head)) == (1, "auto_merge_unknown")
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["status"], payload["observed_main"]) == (
        "delivery_auto_merge_unknown", None if answer == "timeout" else 503, base)
    assert "merge_commit" not in payload and "main_on_read" not in payload
    assert gh["calls"][-2:] == [("GET", PULL)] * 2  # merge_step keeps reading the pull request, as today


def test_a_merge_lock_held_elsewhere_sends_and_records_nothing_until_it_is_free(exact, monkeypatch):
    from hermes_cli import kanban_delivery as kd
    from hermes_cli.sqlite_util import cross_process_init_lock

    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    monkeypatch.setattr(kd, "_MERGE_LOCK_SECONDS", 0.2)
    calls = len(gh["calls"])

    with cross_process_init_lock(kd._merge_lock_path(REPO), timeout_seconds=1, required=True):
        _tick(kb)
        assert (_arms(gh), _state(kb, head), _outcomes(kb, tid)) == ([], "review_cards_created", [])
        assert ("GET", MAIN) not in gh["calls"][calls:]

    _tick(kb)
    assert (_arms(gh), _state(kb, head)) == ([{"sha": head, "merge_method": "merge"}], "auto_merge_armed")


@pytest.mark.parametrize("answer", ["timeout", (404, {"message": "Not Found"}), (200, {"object": {"sha": "main"}}),
                                    "other_ref", "no_ref"],
                         ids=["timeout", "404", "malformed", "other_ref", "no_ref"])
def test_a_main_that_cannot_be_read_sends_and_records_nothing_until_it_can(exact, answer):
    """An answer naming another ref than main, or none, is no read of main either, even when its commit is the
    reviewed base: there main has really moved, so once GitHub's answer names main the row is base_moved."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    moved = _moved(repo) if answer in ("other_ref", "no_ref") else None
    if moved:  # the reviewed base under another ref's name, or under none
        answer = (200, {**({"ref": "refs/heads/some-other-branch"} if answer == "other_ref" else {}),
                        "object": {"sha": base}})
    gh.update(main=moved or base, main_answer=answer)
    _setting(root, "true")
    calls, events = len(gh["calls"]), _events(kb, tid)

    _tick(kb)
    assert ("GET", MAIN) in gh["calls"][calls:]
    assert (_arms(gh), _state(kb, head), _events(kb, tid)) == ([], "review_cards_created", events)

    gh["main_answer"] = None
    _tick(kb)
    if moved:
        assert (_arms(gh), _state(kb, head)) == ([], "base_moved")
        [(kind, payload)] = _outcomes(kb, tid)
        assert (kind, payload["reviewed_base"], payload["observed_main"]) == ("delivery_base_moved", base, moved)
    else:
        assert (_arms(gh), _state(kb, head)) == ([{"sha": head, "merge_method": "merge"}], "auto_merge_armed")


def test_a_read_back_of_main_naming_another_ref_is_a_failed_read(exact):
    """The merge answer is lost and GitHub then shows the pull request merged at H, but main's answer names
    another ref: no read of main, so the read-back failed. The row is unknown with read_back read_failed, never
    armed, and the merge was sent once."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh.update(main=base, arm="timeout")
    gh["hooks"]["PUT"] = lambda: gh["ended"].__setitem__(41, ARMED["sha"]) or gh.update(main_answer=(
        200, {"ref": "refs/heads/some-other-branch", "object": {"sha": ARMED["sha"]}}))
    _setting(root, "true")

    _tick(kb)

    assert gh["calls"][-3:] == [("PUT", MERGE), ("GET", PULL), ("GET", MAIN)]
    assert (len(_arms(gh)), _state(kb, head)) == (1, "auto_merge_unknown")
    [(kind, payload)] = _outcomes(kb, tid)
    assert (kind, payload["read_back"], payload["read_head"], payload["merge_commit"]) == (
        "delivery_auto_merge_unknown", "read_failed", head, ARMED["sha"])
    assert "main_on_read" not in payload


def test_a_merge_whose_first_parent_is_not_the_checked_main_is_recorded_off_base(exact):
    """Plan A(b): main is the reviewed base when the arm reads it under the lock, but it moves on before GitHub
    merges, so the merge's first parent is not the main the arm checked. The merge pass records one
    delivery_merged_off_base event with the merge commit, its first parent, the checked main and the reviewed
    base, and records the merge as today: delivery_merged, and the release intake holds it for H."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed = _merge_on_put(exact, move=True)

    _tick(kb)
    _tick(kb)
    _tick(kb)

    merge, moved = landed["merge"], landed["first_parent"]
    assert moved != base and _git(repo, "rev-parse", f"{merge}^1") == moved
    delivery = _raw(kb.kanban_db_path(), "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)[0]["id"]
    assert _off_base(kb, tid) == [{
        "delivery_id": delivery, "head": head, "pull_request_number": 41, "repository": REPO,
        "merge_commit": merge, "first_parent": moved, "observed_main": base, "reviewed_base": base}]
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)


def test_a_merge_on_the_checked_main_is_recorded_merged_with_no_off_base_event(exact, monkeypatch):
    """The control: GitHub merges H onto the very main the arm checked. Git holds the merge, so nothing is
    fetched, and its first parent is the checked main: no off-base event, and the merge is recorded as today."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed, fetched = _merge_on_put(exact, move=False), _fetches(monkeypatch)

    _tick(kb)
    _tick(kb)

    merge = landed["merge"]
    assert _git(repo, "rev-parse", f"{merge}^1") == landed["first_parent"] == base
    assert (_off_base(kb, tid), fetched) == ([], [])
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)


def test_a_merge_only_github_holds_is_read_after_one_fetch_of_main(exact, monkeypatch, tmp_path):
    """GitHub's merge exists only on the remote's main (made in a separate clone and pushed there), so the card's
    repository does not hold it: the merge pass fetches main once, refs/remotes/origin/main then holds the merge,
    the repository's HEAD and branches did not move, and the first parent is read from what was fetched. Main
    moved on in that clone first, so the read shows in the off-base event."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    clone, landed = tmp_path / "elsewhere", {}

    def merge_on_github():
        _git(repo, "push", "-q", str(gh["bare"]), f"{head}:refs/heads/delivery/{tid}")  # the published branch
        subprocess.run(["git", "clone", "-q", "-b", "main", str(gh["bare"]), str(clone)], check=True, capture_output=True)
        landed["first_parent"] = _moved(clone)
        _git(clone, "merge", "-q", "--no-ff", "--no-edit", head)
        landed["merge"] = gh["main"] = gh["ended"][41] = _git(clone, "rev-parse", "HEAD")
        _git(clone, "push", "-q", "origin", "main")

    gh["hooks"]["PUT"] = merge_on_github
    _tick(kb)
    merge = landed["merge"]
    held = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{merge}^{{commit}}"], capture_output=True)
    assert held.returncode != 0  # the card's repository does not hold the merge
    where, fetched = (_git(repo, "rev-parse", "HEAD"), _git(repo, "for-each-ref", "refs/heads")), _fetches(monkeypatch)

    _tick(kb)

    assert len(fetched) == 1
    assert _git(repo, "rev-parse", "refs/remotes/origin/main") == merge
    assert (_git(repo, "rev-parse", "HEAD"), _git(repo, "for-each-ref", "refs/heads")) == where
    [payload] = _off_base(kb, tid)
    assert (payload["merge_commit"], payload["first_parent"], payload["observed_main"]) == (
        merge, landed["first_parent"], base)
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)


def test_a_merge_git_cannot_find_after_the_fetch_is_recorded_merged_with_its_check_left_open(exact, monkeypatch, caplog):
    """GitHub names a merge commit that neither the card's repository nor the fetched main holds: its first parent
    stays unreadable after the one fetch, so one warning names the delivery and the merge commit, and the merge is
    recorded as today with one delivery_merged_off_base_unknown event, which leaves the check open with the merge
    commit, the checked main and the reviewed base. The next pass fetches and reads again, warns again and adds
    nothing."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    _tick(kb)
    gh["ended"][41] = ARMED["sha"]
    fetched = _fetches(monkeypatch)
    caplog.set_level(logging.WARNING, logger="hermes_cli.kanban_delivery")

    def warned():
        return [record.getMessage() for record in caplog.records
                if record.name == "hermes_cli.kanban_delivery" and ARMED["sha"] in record.getMessage()]

    _tick(kb)

    delivery = _raw(kb.kanban_db_path(), "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)[0]["id"]
    assert len(fetched) == 1 and len(warned()) == 1 and f"delivery {delivery}:" in warned()[0]
    assert (_off_base(kb, tid), _state(kb, head)) == ([], "merged")
    assert (_merges(kb, tid), _intaken(head)) == ([ARMED["sha"]], [ARMED["sha"]])
    assert _off_base_unknown(kb, tid) == [{
        "delivery_id": delivery, "head": head, "pull_request_number": 41, "repository": REPO,
        "merge_commit": ARMED["sha"], "observed_main": base, "reviewed_base": base}]

    _tick(kb)

    assert len(fetched) == 2 and len(warned()) == 2 and f"delivery {delivery}:" in warned()[1]
    assert (len(_off_base_unknown(kb, tid)), _off_base(kb, tid), _on_base(kb, tid)) == (1, [], [])
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == (
        "merged", [ARMED["sha"]], [ARMED["sha"]], 1)


@pytest.mark.parametrize("failure", ["fetch", "parent_read"])
def test_an_off_base_merge_read_after_a_failed_read_is_recorded_off_base_once(exact, monkeypatch, tmp_path, failure):
    """GitHub's merge exists only on the remote's main, which moved on before it, so its first parent is not the
    checked main. In the pass that records the merge, the fetch of main fails as a network error does, or git
    reads no first parent though the fetch works: the merge is recorded as today, and the check left open. Once
    the read works, a later pass records one delivery_merged_off_base event with the merge commit, its first
    parent, the checked main and the reviewed base, which closes the check: one more pass neither fetches nor
    reads, and adds nothing. The merge is sent once and the release intake holds it once."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    clone, landed = tmp_path / "elsewhere", {}

    def merge_on_github():
        _git(repo, "push", "-q", str(gh["bare"]), f"{head}:refs/heads/delivery/{tid}")  # the published branch
        subprocess.run(["git", "clone", "-q", "-b", "main", str(gh["bare"]), str(clone)], check=True, capture_output=True)
        landed["first_parent"] = _moved(clone)
        _git(clone, "merge", "-q", "--no-ff", "--no-edit", head)
        landed["merge"] = gh["main"] = gh["ended"][41] = _git(clone, "rev-parse", "HEAD")
        _git(clone, "push", "-q", "origin", "main")

    gh["hooks"]["PUT"] = merge_on_github
    _tick(kb)
    merge, moved = landed["merge"], landed["first_parent"]
    held = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{merge}^{{commit}}"], capture_output=True)
    assert moved != base and held.returncode != 0  # the card's repository does not hold the merge
    failing = _failing_read(monkeypatch, merge, failure)
    fetched = _fetches(monkeypatch)

    _tick(kb)

    assert (len(fetched), _state(kb, head), _merges(kb, tid), _intaken(head)) == (1, "merged", [merge], [merge])
    assert (_off_base(kb, tid), len(_off_base_unknown(kb, tid)), len(_arms(gh))) == ([], 1, 1)

    failing["now"] = False
    _tick(kb)
    _tick(kb)

    delivery = _raw(kb.kanban_db_path(), "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)[0]["id"]
    assert _off_base(kb, tid) == [{
        "delivery_id": delivery, "head": head, "pull_request_number": 41, "repository": REPO,
        "merge_commit": merge, "first_parent": moved, "observed_main": base, "reviewed_base": base}]
    assert (len(_off_base_unknown(kb, tid)), _on_base(kb, tid)) == (1, [])
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)
    events, fetches, reads = _events(kb, tid), len(fetched), len(failing["reads"])

    _tick(kb)

    assert (_events(kb, tid), len(fetched), len(failing["reads"])) == (events, fetches, reads)


def test_an_on_base_merge_read_after_a_failed_read_closes_its_check_on_base(exact, monkeypatch):
    """The control: GitHub merges H onto the very main the arm checked, but in the pass that records the merge
    git reads no first parent, so the check is left open. Once git reads it, a later pass finds the checked main
    and records one delivery_merged_on_base event with the merge commit, its first parent, the checked main and
    the reviewed base, which closes the check: later passes neither fetch nor read it, and add nothing."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed = _merge_on_put(exact, move=False)
    _tick(kb)
    merge = landed["merge"]
    failing = _failing_read(monkeypatch, merge, "parent_read")
    fetched = _fetches(monkeypatch)

    _tick(kb)

    assert (_state(kb, head), _merges(kb, tid), _intaken(head)) == ("merged", [merge], [merge])
    assert (_off_base(kb, tid), len(_off_base_unknown(kb, tid)), _on_base(kb, tid)) == ([], 1, [])

    failing["now"] = False
    _tick(kb)

    delivery = _raw(kb.kanban_db_path(), "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)[0]["id"]
    assert landed["first_parent"] == base
    assert _on_base(kb, tid) == [{
        "delivery_id": delivery, "head": head, "pull_request_number": 41, "repository": REPO,
        "merge_commit": merge, "first_parent": base, "observed_main": base, "reviewed_base": base}]
    assert (_off_base(kb, tid), len(_off_base_unknown(kb, tid))) == ([], 1)
    events, fetches, reads = _events(kb, tid), len(fetched), len(failing["reads"])

    _tick(kb)
    _tick(kb)

    assert (_events(kb, tid), len(fetched), len(failing["reads"])) == (events, fetches, reads)
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)


def test_with_the_setting_off_an_open_check_is_not_read_until_it_is_on_again(exact, monkeypatch):
    """A check left open under exact_base stays as it is while the setting is off: no fetch of main, no git read
    of the first parent and no event. With the setting exactly true again, the next pass reads the first parent
    and closes the check with one event as it says: delivery_merged_off_base, since main moved on before GitHub
    merged."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed = _merge_on_put(exact, move=True)
    _tick(kb)
    merge = landed["merge"]
    failing = _failing_read(monkeypatch, merge, "parent_read")
    _tick(kb)
    assert (_state(kb, head), _off_base(kb, tid), len(_off_base_unknown(kb, tid))) == ("merged", [], 1)
    failing["now"] = False
    _setting(root, None)
    events, fetched, reads = _events(kb, tid), _fetches(monkeypatch), len(failing["reads"])

    _tick(kb)
    _tick(kb)

    assert (_events(kb, tid), fetched, len(failing["reads"])) == (events, [], reads)

    _setting(root, "true")
    _tick(kb)

    [payload] = _off_base(kb, tid)
    assert (payload["merge_commit"], payload["first_parent"], payload["observed_main"], payload["reviewed_base"]) == (
        merge, landed["first_parent"], base, base)
    assert (len(_off_base_unknown(kb, tid)), _on_base(kb, tid)) == (1, [])
    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)


def test_with_the_setting_off_at_record_time_an_off_base_merge_is_recorded_as_today(exact, monkeypatch):
    """Armed with exact_base on, then GitHub merges H onto a main that moved on; the setting is off when the merge
    pass records it: no git read and no fetch, so no off-base event, and the merge is recorded as today."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed = _merge_on_put(exact, move=True)
    _tick(kb)
    assert _state(kb, head) == "auto_merge_armed"
    _setting(root, None)
    fetched, reads, run_git = _fetches(monkeypatch), [], kd._run_git
    monkeypatch.setattr(kd, "_run_git", lambda workdir, *args: reads.append(args) or run_git(workdir, *args))

    _tick(kb)

    merge = landed["merge"]
    assert _git(repo, "rev-parse", f"{merge}^1") == landed["first_parent"] != base
    assert (_off_base(kb, tid), fetched, [args for args in reads if f"{merge}^1" in args]) == ([], [], [])
    assert (_state(kb, head), _merges(kb, tid), _intaken(head)) == ("merged", [merge], [merge])


def test_an_off_base_merge_whose_intake_waits_a_pass_is_recorded_off_base_once(exact, monkeypatch):
    """The first merge pass records the off-base event with its claim, but its intake does not settle the row,
    which stays armed; the pass that records the merge finds that event for this delivery and merge commit and
    adds none."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    gh["main"] = base
    _setting(root, "true")
    landed, intake, waited = _merge_on_put(exact, move=True), kd._intake, []

    def unsettled_once(db_path, row, merge):
        waited.append(merge)
        return None if len(waited) == 1 else intake(db_path, row, merge)

    monkeypatch.setattr(kd, "_intake", unsettled_once)
    _tick(kb)
    _tick(kb)

    merge = landed["merge"]
    assert (waited, _state(kb, head), _merges(kb, tid), len(_off_base(kb, tid))) == ([merge], "auto_merge_armed", [], 1)

    _tick(kb)
    _tick(kb)

    assert (_state(kb, head), _merges(kb, tid), _intaken(head), len(_arms(gh))) == ("merged", [merge], [merge], 1)
    assert [payload["first_parent"] for payload in _off_base(kb, tid)] == [landed["first_parent"]]


def test_the_merge_lock_is_one_root_file_from_the_worker_environment_too(exact, monkeypatch):
    """Under the environment the dispatcher gives a worker of proj-a (its board pinned, its card set, a profile
    home under the root), the merge lock is the root's file; arms on two other boards leave exactly one."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db, other_db, root_db = (root / "kanban" / "boards" / "proj-a" / "kanban.db",
                                 root / "kanban" / "boards" / "other" / "kanban.db", root / "kanban.db")
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    tid, head = _approved(kb, repo, "beta", board="other")
    _published(kb, repo, gh, tid, head, board="other")
    home, home_head = _approved(kb, repo, "gamma")
    _published_as(root_db, gh, home, home_head, 44)
    gh["main"] = _git(repo, "rev-parse", "main")
    for db, name in ((other_db, "other"), (root_db, kb.DEFAULT_BOARD)):
        ticked = kb.connect(db_path=db)
        try:
            kb.dispatch_once(ticked, board=name, spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()
        _done(db)
    _reviews(gh, *_lenses(kb, head, db=other_db))
    _reviews(gh, *_lenses(kb, home_head, db=root_db), number=44)
    from_root = kd._merge_lock_path(REPO)
    conn = kb.connect(board="proj-a")
    try:
        claimed = kb.claim_task(conn, kb.create_task(conn, title="plain work", assignee=IMPLEMENTER))
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, claimed.id, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert (env["HERMES_KANBAN_TASK"], env["HERMES_KANBAN_DB"], env["HERMES_KANBAN_BOARD"], env["HERMES_HOME"]) == (
        claimed.id, str(own_db), "proj-a", str(profile_home))
    _setting(profile_home, "true")  # the worker reads its own profile's configuration
    _setting(root, "true")

    _become(env, monkeypatch)
    assert kd._merge_lock_path(REPO) == from_root == root / "kanban" / "merge-locks" / LOCK_NAME[:-len(".init.lock")]
    for db, name in ((other_db, "other"), (root_db, kb.DEFAULT_BOARD)):
        ticked = kb.connect(db_path=db)
        try:
            kb.dispatch_once(ticked, board=name, spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()

    assert sorted(body["sha"] for body in _arms(gh)) == sorted([head, home_head])
    count = "SELECT COUNT(*) AS n FROM kanban_deliveries WHERE pull_request_state = 'auto_merge_armed'"
    assert [_raw(db, count)[0]["n"] for db in (other_db, own_db, root_db)] == [1, 0, 1]
    assert _locks(root) == [LOCK_NAME]
    assert not (profile_home / "kanban").exists()


_CHILD = """
import contextlib, http.client, json, os, sys, time
from pathlib import Path
port, db, name, log, go = sys.argv[1:6]
from hermes_cli import kanban_db as kb, kanban_delivery_github as transport, sqlite_util
transport._connect = lambda: http.client.HTTPConnection("127.0.0.1", int(port), timeout=20)
def token(self):
    self._token = "token"
    return self._token
transport.GitHubTransport._installation_token = token
spans, real = [], sqlite_util.cross_process_init_lock
@contextlib.contextmanager
def traced(path, **options):
    with real(path, **options):
        span = {"path": str(path), "enter": time.time()}
        try:
            yield
        finally:
            span["exit"] = time.time()
            spans.append(span)
sqlite_util.cross_process_init_lock = traced
Path(log + ".ready").touch()
while not os.path.exists(go):
    time.sleep(0.005)
conn = kb.connect(db_path=Path(db))
try:
    kb.dispatch_once(conn, board=name, spawn_fn=lambda *a, **k: 1)
finally:
    conn.close()
Path(log).write_text(json.dumps([span for span in spans if "merge-locks" in span["path"]]))
"""


def test_two_processes_on_two_boards_merge_one_and_record_the_other_base_moved(exact, tmp_path):
    """Two dispatcher processes, each ticking its own board, arm on one repository at once. GitHub (a local
    server) answers the first merge slowly and then moves main, so an unlocked second arm would read the old
    main during it. The lock spans never overlap, exactly one merge is sent, and the second holder reads
    the moved main and records base_moved."""
    kb, root, repo, gh = exact
    kb.create_board("alpha")
    kb.create_board("beta")
    dbs = {name: root / "kanban" / "boards" / name / "kanban.db" for name in ("alpha", "beta")}
    heads = {}
    for name, number in (("alpha", 41), ("beta", 42)):
        tid, head = _approved(kb, repo, name, board=name)
        _published_as(dbs[name], gh, tid, head, number)
        gh["head"] = head  # the table's CI read for the review cards
        conn = kb.connect(board=name)
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "delivery_bound", {
                    "source_task_id": tid, "head": head, "base_commit": _git(repo, "rev-parse", "main")})
        finally:
            conn.close()
        ticked = kb.connect(db_path=dbs[name])
        try:
            kb.dispatch_once(ticked, board=name, spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()
        _done(dbs[name])
        _reviews(gh, *_lenses(kb, head, db=dbs[name]), number=number)
        heads[number] = (name, tid, head)
    base, merge_commit = _git(repo, "rev-parse", "main"), "c" * 40
    hub = {"main": base, "merges": [], "reads": []}

    class GitHub(BaseHTTPRequestHandler):
        def _answer(self, status, data=None):
            body = b"" if data is None else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, parts = urlsplit(self.path).path, urlsplit(self.path).path.split("/")
            if path == MAIN:
                hub["reads"].append((time.time(), hub["main"]))
                return self._answer(200, {"ref": "refs/heads/main", "object": {"sha": hub["main"]}})
            if len(parts) >= 6 and parts[4] == "pulls" and int(parts[5]) in heads:
                number = int(parts[5])
                if path.endswith("/reviews"):
                    return self._answer(200, gh["reviews"][number])
                return self._answer(200, {"number": number, "state": "open", "merged": False,
                                          "head": {"sha": heads[number][2]}})
            return self._answer(404)

        def do_PUT(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            hub["merges"].append((urlsplit(self.path).path, body, time.time()))
            time.sleep(2 if len(hub["merges"]) == 1 else 0)  # the first merge is slow
            hub["main"] = merge_commit
            self._answer(200, {"merged": True})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), GitHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _setting(root, "true")
    script, go = tmp_path / "child.py", tmp_path / "go"
    script.write_text(_CHILD, encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    logs = {name: tmp_path / f"{name}.json" for name in dbs}
    try:
        children = [subprocess.Popen([sys.executable, str(script), str(server.server_address[1]), str(dbs[name]),
                                      name, str(logs[name]), str(go)], env=env, cwd=str(tmp_path),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for name in dbs]
        deadline = time.time() + 120
        while not all(Path(str(log) + ".ready").exists() for log in logs.values()) and time.time() < deadline:
            time.sleep(0.02)
        go.touch()
        outputs = [child.communicate(timeout=120)[0].decode(errors="replace") for child in children]
    finally:
        server.shutdown()
        server.server_close()
    assert [child.returncode for child in children] == [0, 0], outputs

    spans = sorted((span for log in logs.values() for span in json.loads(log.read_text())), key=lambda s: s["enter"])
    assert len(spans) == 2 and spans[0]["exit"] <= spans[1]["enter"]  # one holder at a time
    assert len(hub["merges"]) == 1
    [(path, body, sent)] = hub["merges"]
    first = int(path.split("/")[-2])
    second = next(number for number in heads if number != first)
    assert body == {"sha": heads[first][2], "merge_method": "merge"}
    assert [main for _, main in hub["reads"]] == [base, merge_commit]  # the second holder read the moved main
    for number, state in ((first, "auto_merge_armed"), (second, "base_moved")):
        name, tid, head = heads[number]
        assert _state(kb, head, dbs[name]) == state
    name, tid, head = heads[second]
    [(kind, payload)] = _outcomes(kb, tid, dbs[name])
    assert (kind, payload["reviewed_base"], payload["observed_main"], payload["head"]) == (
        "delivery_base_moved", base, merge_commit, head)
    assert _locks(root) == [LOCK_NAME]
