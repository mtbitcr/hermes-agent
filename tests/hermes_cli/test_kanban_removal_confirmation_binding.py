"""The permanent confirmation is bound to the statement that was SHOWN.

Every test here drives a REAL surface — the shipped ``kanban boards rm``
command through its real argparse tree, or the shipped dashboard
``DELETE /boards/{slug}`` route through a real ASGI client — and asserts
against durable state read outside the module under test.

What is being pinned down:

* A calling surface cannot mint a confirmation. ``confirm_permanent_removal``
  is the only thing that makes one and it only makes one from the exact
  statement-bound line the operator was shown.
* The CLI really reads stdin. A closed stdin, an empty answer and a wrong
  answer all refuse, and the board is left completely intact.
* The non-interactive path is the statement-bound line, not a bare flag.
* The dashboard can OBTAIN the statement to display (a disclosure
  response), and a returned, statement-bound payload reaches the same
  common driver the CLI uses.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area,
    board_with_multi_commit_advance,
    create_fenced_board,
    read_only,
    ready_task,
    record_git_receipt,
    record_run_advance,
    register_row,
)


# ---------------------------------------------------------------------------
# Surfaces, loaded the way production loads them
# ---------------------------------------------------------------------------

def _dashboard_client():
    """A real ASGI client over the shipped kanban dashboard router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    plugin_file = (
        _WORKTREE / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_kanban_plugin_confirm_test", plugin_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


class _ClosedStdin(io.StringIO):
    """A stdin that is CLOSED — what an unattended invocation really has."""

    def readline(self, *args, **kwargs):
        raise ValueError("I/O operation on closed file")


@contextlib.contextmanager
def _stdin(payload):
    """Put *payload* on ``sys.stdin`` for the duration of a real CLI call."""
    original = sys.stdin
    sys.stdin = io.StringIO(payload) if isinstance(payload, str) else payload
    try:
        yield
    finally:
        sys.stdin = original


def _run_cli(argv: list, *, stdin="") -> "tuple[int, str]":
    """Run ``hermes kanban …`` through the real argparse tree and dispatch."""
    import argparse

    from hermes_cli import kanban as kanban_cli

    wrap = argparse.ArgumentParser(prog="hermes-test", add_help=False)
    top = wrap.add_subparsers(dest="_top")
    parser = kanban_cli.build_parser(top)
    args = parser.parse_args(argv)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        with _stdin(stdin):
            code = kanban_cli.kanban_command(args)
    return code, buffer.getvalue()


def _board_is_fully_intact(slug: str, task_id: str) -> None:
    """The board, its store, its task row and its lifecycle are untouched."""
    assert kb.board_dir(slug).exists(), "the board directory was removed"
    store = kb.kanban_db_path(board=slug)
    assert store.exists(), "the board store was removed"
    with read_only(store) as conn:
        row = conn.execute(
            "SELECT id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    assert row is not None, "the board's task row was destroyed"
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.LIVE.value
    assert kb.get_removal_phase_record(slug) is None, (
        "a refused removal must not have begun one"
    )


@pytest.fixture
def live_board(fence_home):
    """A fenced board carrying one real task, ready to be removed."""
    slug = "confirm-me"
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    conn.close()
    return slug, task_id


# ---------------------------------------------------------------------------
# The confirmation is never minted by the caller
# ---------------------------------------------------------------------------

def test_disclosure_alone_confirms_nothing(live_board):
    """Obtaining the statement to display is not confirming it."""
    slug, task_id = live_board

    disclosure = kb.permanent_removal_disclosure(slug)

    assert disclosure.statement_text.strip(), "no statement to show the operator"
    assert slug in disclosure.required_response
    assert disclosure.statement_digest[:16] in disclosure.required_response
    _board_is_fully_intact(slug, task_id)


def test_a_wrong_answer_mints_no_confirmation(live_board):
    """Only the exact statement-bound line yields a confirmation."""
    slug, _task_id = live_board
    disclosure = kb.permanent_removal_disclosure(slug)

    for answer in ("yes", "y", "YES", "", None, True, disclosure.required_response[:-1]):
        checked = kb.confirm_permanent_removal(
            slug, response=answer, disclosure=disclosure,
        )
        assert not checked.confirmed, f"{answer!r} was accepted as an answer"
        assert checked.confirmation is None
        assert checked.refusal is not None
        assert checked.message

    accepted = kb.confirm_permanent_removal(
        slug, response=disclosure.required_response, disclosure=disclosure,
    )
    assert accepted.confirmed, accepted.message
    assert accepted.confirmation.statement_digest == disclosure.statement_digest


def test_an_answer_for_a_different_board_is_refused(fence_home):
    """A statement-bound line binds to ONE board's statement."""
    create_fenced_board("board-a")
    create_fenced_board("board-b")
    for_a = kb.permanent_removal_disclosure("board-a")

    checked = kb.confirm_permanent_removal("board-b", response=for_a.required_response)

    assert not checked.confirmed
    assert checked.refusal is kb.PermanentConfirmationRefusal.MISMATCHED_RESPONSE


def test_a_digest_from_another_statement_is_refused(fence_home):
    """A payload echoing a digest this board no longer has is not a confirmation."""
    create_fenced_board("board-c")
    create_fenced_board("board-d")
    right = kb.permanent_removal_disclosure("board-c")
    wrong = kb.permanent_removal_disclosure("board-d")

    checked = kb.confirm_permanent_removal(
        "board-c",
        response=right.required_response,
        shown_digest=wrong.statement_digest,
    )

    assert not checked.confirmed
    assert checked.refusal is kb.PermanentConfirmationRefusal.MISMATCHED_STATEMENT


def test_the_driver_refuses_a_confirmation_the_surface_made_up(live_board):
    """A hand-built "confirmed" object never removes a board."""
    slug, task_id = live_board

    result = kb.remove_board_fenced(
        slug,
        mode="permanent",
        permanent_confirmation=kb.PermanentRemovalConfirmation(
            confirmed=True, statement_digest="0" * 64, confirmed_by="a-surface",
        ),
    )

    assert not result.success, result.message
    assert result.refusal_reason == "unbound-confirmation"
    _board_is_fully_intact(slug, task_id)


def test_the_driver_refuses_permanent_mode_with_no_confirmation(live_board):
    """No confirmation at all is a refusal, not a default."""
    slug, task_id = live_board

    result = kb.remove_board_fenced(slug, mode="permanent")

    assert not result.success, result.message
    assert result.refusal_reason == "no-confirmation"
    _board_is_fully_intact(slug, task_id)


# ---------------------------------------------------------------------------
# The CLI really reads stdin
# ---------------------------------------------------------------------------

def test_cli_delete_with_closed_stdin_refuses_and_leaves_the_board(live_board):
    """`boards rm <slug> --delete` with stdin closed removes nothing."""
    slug, task_id = live_board

    code, output = _run_cli(
        ["boards", "rm", slug, "--delete"], stdin=_ClosedStdin(),
    )

    assert code == 1, output
    assert kb.PermanentConfirmationRefusal.NO_RESPONSE.value in output
    # The statement was still SHOWN — refusing is not a reason to hide it.
    assert "Permanent removal of board" in output
    assert kb.permanent_removal_disclosure(slug).required_response in output
    _board_is_fully_intact(slug, task_id)


def test_cli_delete_with_eof_stdin_refuses_and_leaves_the_board(live_board):
    """An empty (EOF) stdin is an absent answer, so the removal refuses."""
    slug, task_id = live_board

    code, output = _run_cli(["boards", "rm", slug, "--delete"], stdin="")

    assert code == 1, output
    assert kb.PermanentConfirmationRefusal.NO_RESPONSE.value in output
    _board_is_fully_intact(slug, task_id)


def test_cli_delete_with_a_yes_answer_refuses(live_board):
    """"yes" is not the statement-bound line, so it is not a confirmation."""
    slug, task_id = live_board

    code, output = _run_cli(["boards", "rm", slug, "--delete"], stdin="yes\n")

    assert code == 1, output
    assert kb.PermanentConfirmationRefusal.MISMATCHED_RESPONSE.value in output
    _board_is_fully_intact(slug, task_id)


def test_cli_delete_reads_the_echoed_statement_line_from_stdin(live_board):
    """The interactive read is real: the echoed line drives the removal."""
    slug, _task_id = live_board
    required = kb.permanent_removal_disclosure(slug).required_response

    code, output = _run_cli(
        ["boards", "rm", slug, "--delete"], stdin=f"{required}\n",
    )

    assert code == 0, output
    assert "permanently removed" in output
    assert not kb.board_dir(slug).exists()
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.HARD_REMOVED.value
    record = kb.get_removal_phase_record(slug)
    assert record.phase is kb.RemovalPhase.DONE
    assert kb.get_permanent_removal_receipt(slug, record.removal_id) is not None


def test_cli_non_interactive_path_is_the_statement_line_not_a_flag(live_board):
    """--confirm takes the exact line; there is no bare --yes to find."""
    slug, task_id = live_board
    required = kb.permanent_removal_disclosure(slug).required_response

    wrong, output = _run_cli(
        ["boards", "rm", slug, "--delete", "--confirm", "yes"],
        stdin=_ClosedStdin(),
    )
    assert wrong == 1, output
    _board_is_fully_intact(slug, task_id)

    with pytest.raises(SystemExit):
        _run_cli(["boards", "rm", slug, "--delete", "--yes"], stdin="")

    code, output = _run_cli(
        ["boards", "rm", slug, "--delete", "--confirm", required],
        stdin=_ClosedStdin(),
    )
    assert code == 0, output
    assert not kb.board_dir(slug).exists()


def test_cli_archive_needs_no_confirmation_and_reads_no_stdin(live_board):
    """A reversible removal is not a permanent one: no statement, no prompt."""
    slug, _task_id = live_board

    code, output = _run_cli(["boards", "rm", slug], stdin=_ClosedStdin())

    assert code == 0, output
    assert "Permanent removal of board" not in output
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value


# ---------------------------------------------------------------------------
# The dashboard route reaches the same driver, statement-bound
# ---------------------------------------------------------------------------

def test_dashboard_hard_delete_without_confirmation_returns_the_disclosure(
    live_board,
):
    """A first request OBTAINS the statement to display, and removes nothing."""
    slug, task_id = live_board
    client = _dashboard_client()

    response = client.delete(f"/api/plugins/kanban/boards/{slug}?delete=true")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "confirmation-required"
    disclosure = detail["disclosure"]
    expected = kb.permanent_removal_disclosure(slug)
    assert disclosure["statement"] == expected.statement_text
    assert disclosure["statement_digest"] == expected.statement_digest
    assert disclosure["required_response"] == expected.required_response
    assert disclosure["out_categories"], "the statement named no OUT categories"
    _board_is_fully_intact(slug, task_id)


def test_dashboard_hard_delete_with_an_unbound_payload_is_refused(live_board):
    """A bare "yes" body is not a statement-bound confirmation."""
    slug, task_id = live_board
    client = _dashboard_client()

    response = client.request(
        "DELETE",
        f"/api/plugins/kanban/boards/{slug}?delete=true",
        json={"confirm": "yes"},
    )

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["error"] == (
        kb.PermanentConfirmationRefusal.MISMATCHED_RESPONSE.value
    )
    assert detail["board_intact"] is True
    _board_is_fully_intact(slug, task_id)


def test_dashboard_hard_delete_with_a_mismatched_digest_is_refused(live_board):
    """Echoing the right line with the wrong statement digest is refused."""
    slug, task_id = live_board
    client = _dashboard_client()
    required = kb.permanent_removal_disclosure(slug).required_response

    response = client.request(
        "DELETE",
        f"/api/plugins/kanban/boards/{slug}?delete=true",
        json={"confirm": required, "statement_digest": "f" * 64},
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"] == (
        kb.PermanentConfirmationRefusal.MISMATCHED_STATEMENT.value
    )
    _board_is_fully_intact(slug, task_id)


def test_dashboard_hard_delete_with_the_returned_statement_removes_the_board(
    live_board,
):
    """The two-step flow reaches the SAME driver the CLI reaches."""
    slug, _task_id = live_board
    client = _dashboard_client()

    first = client.delete(f"/api/plugins/kanban/boards/{slug}?delete=true")
    assert first.status_code == 409
    disclosure = first.json()["detail"]["disclosure"]

    second = client.request(
        "DELETE",
        f"/api/plugins/kanban/boards/{slug}?delete=true",
        json={
            "confirm": disclosure["required_response"],
            "statement_digest": disclosure["statement_digest"],
        },
    )

    assert second.status_code == 200, second.text
    body = second.json()["result"]
    assert body["mode"] == kb.RemovalMode.PERMANENT.value
    assert body["action"] == "deleted"
    assert body["phase"] == kb.RemovalPhase.DONE.value
    assert body["steps"], "the driver reported no roll-forward steps"
    assert not kb.board_dir(slug).exists()
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.HARD_REMOVED.value
    assert kb.get_permanent_removal_receipt(slug, body["removal_id"]) is not None


# ---------------------------------------------------------------------------
# The statement is built from THIS BOARD'S frozen prospective inventory
# ---------------------------------------------------------------------------
#
# A statement built only from the board name, a static live-work notice and
# the static OUT-category prose tells an operator nothing about what THEY
# are about to lose: its digest is the same for a board with a repository,
# a linked work area and recorded commits as for a board with nothing at
# all. So the statement is built from the exact prospective inventory and
# the digest binds that snapshot.

def _rich_board(slug: str, repo: Path, *, absorbed_heads=None) -> dict:
    """A board with a real repo, a multi-commit advance and a transcript."""
    setup = board_with_multi_commit_advance(
        slug, repo, extra_commits=2, absorbed_heads=absorbed_heads,
    )
    conn = kb.connect(board=slug)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            (f"session-of-{setup['task']}", setup["task"]),
        )
    conn.close()
    setup["session_id"] = f"session-of-{setup['task']}"
    return setup


@pytest.fixture
def rich_board(fence_home, tmp_path):
    return _rich_board("statement-subject", tmp_path / "repo")


def test_the_statement_names_this_boards_repository_reference_and_commits(
    rich_board,
):
    """The operator sees what SURVIVES, by exact identity, for this board."""
    setup = rich_board
    statement = kb.permanent_removal_disclosure(setup["slug"]).statement_text

    assert str(setup["repo"]) in statement, "the repository is not named"
    assert setup["branch"] in statement, "the reference is not named"
    assert setup["base"] in statement, "the base commit is not named"
    assert setup["head"] in statement, "the head commit is not named"
    assert setup["task"] in statement, "the task identity is not named"


def test_the_statement_names_the_work_area_and_registration_removed(
    rich_board,
):
    """And what is DESTROYED or DEREGISTERED, also by exact identity."""
    setup = rich_board
    statement = kb.permanent_removal_disclosure(setup["slug"]).statement_text

    assert str(setup["work_area"]) in statement, "the work area is not named"
    assert str(setup["registration"]) in statement, (
        "the registration that is deregistered is not named"
    )
    assert str(kb.board_dir(setup["slug"])) in statement
    assert str(kb.kanban_db_path(board=setup["slug"])) in statement


def test_the_statement_names_the_transcript_identities_and_their_location(
    rich_board,
):
    """C6 is kept and disclosed, so the operator is told where it is."""
    setup = rich_board
    disclosure = kb.permanent_removal_disclosure(setup["slug"])

    transcripts = disclosure.prospective_inventory["transcripts"]
    sessions = transcripts["sessions"]
    assert sessions, "the inventory names no transcript identity at all"
    assert {s["session_id"] for s in sessions} == {setup["session_id"]}
    assert transcripts["store"], "the transcript store's location is missing"
    assert setup["session_id"] in disclosure.statement_text
    assert str(transcripts["store"]) in disclosure.statement_text


def test_the_statement_marks_commit_list_completeness_unverified(rich_board):
    """Several commits in one advance: the statement says so, honestly.

    Only the advance's head is derivable from the recorded run receipts,
    so the intermediate commits of that same advance are not on the list.
    The statement must mark completeness UNVERIFIED rather than present
    the list as complete.
    """
    setup = rich_board
    assert setup["intermediate"], "the fixture made no intermediate commit"
    disclosure = kb.permanent_removal_disclosure(setup["slug"])

    references = disclosure.prospective_inventory["survives"]["references"]
    assert references, "the inventory names no reference at all"
    reference, = [r for r in references if r["task"] == setup["task"]]
    assert setup["head"] in reference["commits"]
    for commit in setup["intermediate"]:
        assert commit not in reference["commits"], commit
    completeness = reference["commit_list_completeness"]
    assert completeness["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert completeness["reasons"], "the marking gives no reason"
    assert completeness["creation_time_provenance_ledger"] == "absent"
    assert reference["commit_list_source"] == kb.COMMIT_LIST_SOURCE
    # …and the operator reads that verdict in the statement itself.
    assert kb.RECEIPT_VERDICT_UNVERIFIED in disclosure.statement_text


def test_an_absorbed_other_board_head_is_disclosed_not_claimed(
    fence_home, tmp_path
):
    """A head another board's advance produced is disclosed, never claimed."""
    repo = tmp_path / "shared"
    origin = board_with_multi_commit_advance(
        "absorbing-origin", repo, extra_commits=1,
    )
    absorbing = _rich_board(
        "absorbing-subject", repo,
        absorbed_heads=[
            {"task": origin["task"], "head_commit": origin["head"]}
        ],
    )

    disclosure = kb.permanent_removal_disclosure(absorbing["slug"])

    references = disclosure.prospective_inventory["survives"]["references"]
    assert references, "the inventory names no reference at all"
    reference, = [r for r in references if r["task"] == absorbing["task"]]
    assert reference["absorbed_heads"] == [origin["head"]]
    assert origin["head"] not in reference["commits"], (
        "a head another board's advance produced was claimed as this "
        "board's own"
    )
    assert origin["head"] in disclosure.statement_text
    assert absorbing["head"] in reference["commits"]


def test_two_boards_with_different_inventories_get_different_digests(
    fence_home, tmp_path
):
    """The digest is not the same for a board with work and one without.

    This is the defect in one assertion: a statement whose digest is
    byte-identical for a board with a repository, a reference, recorded
    commits and a linked work area, and for a board with none of it, binds
    nothing about the board.
    """
    with_work = _rich_board("digest-with-work", tmp_path / "repo")
    create_fenced_board("digest-without-work")

    rich = kb.permanent_removal_disclosure(with_work["slug"])
    bare = kb.permanent_removal_disclosure("digest-without-work")

    assert rich.statement_digest != bare.statement_digest
    assert rich.required_response != bare.required_response
    assert rich.statement.inventory_digest() != bare.statement.inventory_digest()
    # And the answer for one is not the answer for the other.
    crossed = kb.confirm_permanent_removal(
        with_work["slug"], response=bare.required_response,
    )
    assert not crossed.confirmed
    assert crossed.refusal is kb.PermanentConfirmationRefusal.MISMATCHED_RESPONSE


def test_a_confirmation_does_not_survive_a_change_to_the_inventory(
    rich_board, tmp_path
):
    """Frozen means frozen: a changed prospective inventory refuses the answer."""
    setup = rich_board
    stale = kb.permanent_removal_disclosure(setup["slug"])

    # A REAL change to what the removal would touch: a second task with its
    # own linked work area, reference and recorded advance.
    conn = kb.connect(board=setup["slug"])
    second = kb.create_task(conn, title="work added after the disclosure")
    branch = f"hermes/{second}"
    work_area, head = add_linked_work_area(setup["repo"], second, branch=branch)
    record_run_advance(
        conn, second, branch=branch, base_commit=setup["base"],
        head_commit=head,
    )
    record_git_receipt(
        conn, second, workspace_path=work_area, branch_name=branch,
        base_commit=setup["base"], head_commit=head,
    )
    conn.close()

    fresh = kb.permanent_removal_disclosure(setup["slug"])
    assert fresh.statement_digest != stale.statement_digest
    assert str(work_area) in fresh.statement_text
    assert str(work_area) not in stale.statement_text

    checked = kb.confirm_permanent_removal(
        setup["slug"], response=stale.required_response,
    )
    assert not checked.confirmed, (
        "an answer bound to a smaller prospective inventory was accepted"
    )
    assert kb.board_dir(setup["slug"]).exists()

    # The driver refuses a confirmation minted against the stale statement
    # too, so the board is not removed on an answer to an older question.
    minted = kb.confirm_permanent_removal(
        setup["slug"], response=stale.required_response, disclosure=stale,
    )
    assert minted.confirmed, minted.message
    result = kb.remove_board_fenced(
        setup["slug"], mode="permanent",
        permanent_confirmation=minted.confirmation,
    )
    assert not result.success, result.message
    assert result.refusal_reason == "unbound-confirmation"
    assert kb.board_dir(setup["slug"]).exists()
    assert register_row(setup["slug"])["lifecycle"] == kb.BoardLifecycle.LIVE.value


def test_the_cli_prints_the_whole_statement_including_the_inventory(
    rich_board,
):
    """The command-line surface really shows the operator the inventory."""
    setup = rich_board
    disclosure = kb.permanent_removal_disclosure(setup["slug"])

    code, output = _run_cli(
        ["boards", "rm", setup["slug"], "--delete"], stdin=_ClosedStdin(),
    )

    assert code == 1, output
    assert disclosure.statement_text in output, (
        "the CLI showed something other than the exact statement"
    )
    for named in (
        str(setup["repo"]), setup["branch"], setup["base"], setup["head"],
        str(setup["work_area"]), str(setup["registration"]),
        setup["session_id"],
    ):
        assert named in output, f"the CLI did not show {named}"
    assert kb.board_dir(setup["slug"]).exists()


def test_the_route_returns_the_inventory_as_structured_data(rich_board):
    """The 409 disclosure carries the frozen snapshot, not just prose."""
    setup = rich_board
    client = _dashboard_client()

    response = client.delete(
        f"/api/plugins/kanban/boards/{setup['slug']}?delete=true"
    )

    assert response.status_code == 409, response.text
    disclosure = response.json()["detail"]["disclosure"]
    expected = kb.permanent_removal_disclosure(setup["slug"])
    inventory = disclosure["prospective_inventory"]
    assert inventory == expected.prospective_inventory
    assert disclosure["prospective_inventory_digest"] == (
        expected.statement.inventory_digest()
    )
    references = inventory["survives"]["references"]
    assert references, "the route disclosed no reference at all"
    reference, = [r for r in references if r["task"] == setup["task"]]
    assert reference["repository"] == str(setup["repo"])
    assert reference["reference"] == setup["branch"]
    assert reference["head_commit"] == setup["head"]
    registrations = inventory["removed"]["work_area_registrations"]
    assert registrations, "the route disclosed no registration at all"
    assert {r["work_area"] for r in registrations} == {str(setup["work_area"])}
    assert {r["registration"] for r in registrations} == {
        str(setup["registration"])
    }
    assert inventory["survives"]["commit_list_source"] == kb.COMMIT_LIST_SOURCE
    assert kb.board_dir(setup["slug"]).exists()


def test_the_route_removes_the_board_on_the_inventory_bound_line(rich_board):
    """The two-step flow still reaches the same driver, inventory and all."""
    setup = rich_board
    client = _dashboard_client()

    first = client.delete(
        f"/api/plugins/kanban/boards/{setup['slug']}?delete=true"
    )
    assert first.status_code == 409
    disclosure = first.json()["detail"]["disclosure"]

    second = client.request(
        "DELETE",
        f"/api/plugins/kanban/boards/{setup['slug']}?delete=true",
        json={
            "confirm": disclosure["required_response"],
            "statement_digest": disclosure["statement_digest"],
        },
    )

    assert second.status_code == 200, second.text
    body = second.json()["result"]
    assert body["phase"] == kb.RemovalPhase.DONE.value
    assert not kb.board_dir(setup["slug"]).exists()
    assert not setup["work_area"].exists()
    # C7 is kept: the shared repository and the reference survive.
    assert setup["repo"].exists()
    assert kb.get_permanent_removal_receipt(
        setup["slug"], body["removal_id"]
    ) is not None
