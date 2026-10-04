"""Delivery records, the approval proof and the integration-card hook.

When a review-required build card is approved from the review lane and
``kanban.delivery.enabled`` is true, :func:`record_approved_delivery` records
ONE delivery for the source card and its approved head and creates ONE
integration card, assigned to ``kanban.delivery.integration_profile``.
``complete_task`` calls it once, from its approval branch, inside the
approval's own transaction, so the approval, the record and the card commit
or roll back together. With the setting off, or no integration profile, it
returns before touching anything and the approval behaves exactly as before.

The approval proof is the T1 fence of :mod:`hermes_cli.kanban_delivery_fences`
applied to facts read only from the kernel's own records: the task row, its
event log, its runs and the git repository the card was built in. Worker
prose (handover text, comments, summaries, run metadata) is never read.

:func:`publish_delivery` is T1 itself, run by the integration card's current
run: the same approval facts, the publish ledger kept on the delivery row and
the remote facts read through :mod:`hermes_cli.kanban_delivery_github` decide
whether the approved head becomes the delivery's one pull request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_delivery_fences import Decision, decide_publish, load_policy

logger = logging.getLogger(__name__)

# The integration card's idempotency key (plan section 5) is this prefix, the
# owner-confirmed value, + source card id + ":" + approved head, so one source
# card and head can only ever name one card.
DELIVERY_KEY_PREFIX = "delivery:"

# The creator every kernel-made integration card carries.
_CREATED_BY = "kanban-delivery"

_SHA = re.compile(r"[0-9a-f]{40}")

# Nothing is published at approval: no pull request is recorded and no remote
# branch is read, so only the approval half of T1 can decide.
_NO_LEDGER = {
    "pull_request": None, "head": None, "state": None, "head_is_ancestor": False,
}
_NO_REMOTE = {"branch_head": None, "open_pulls": []}


def delivery_key(source_task_id: str, head: str) -> str:
    return DELIVERY_KEY_PREFIX + source_task_id + ":" + head


# The integration card's body (plan section 5) is this fixed kernel text: the
# steps in order and where it stops. Only the source card id, the approved head
# and the delivery key are filled in, so no worker prose can reach the card.
_INTEGRATION_CARD_BODY = (
    "Kernel integration card for an approved build.\n"
    "Approved source card: {source}\n"
    "Approved head: {head}\n"
    "Delivery key: {key}\n"
    "\n"
    "Steps, in this order:\n"
    "1. T1 kanban_delivery_publish: publish the approved head.\n"
    "2. T2 kanban_delivery_read_checks: read the required checks on the head.\n"
    "3. T3 kanban_delivery_rerun_flaky: only for an eligible flaky failure, the one\n"
    "   rerun, then T2 again.\n"
    "4. T4 kanban_delivery_request_lenses: once the checks pass, request the lenses.\n"
    "5. T6 kanban_delivery_merge: once both lenses approve the head, merge it.\n"
    "6. T7 kanban_delivery_handoff: record the handoff of the merge.\n"
    "\n"
    "Where it stops:\n"
    "- T1 refused (not approved, head mismatch, foreign branch, unknown pull\n"
    "  request): block (needs_input) for Delivery and infrastructure and the\n"
    "  owner. The source card is untouched.\n"
    "- Checks pending: wait (blocked, transient); nothing moves.\n"
    "- GitHub started zero jobs: stop and block for Delivery and infrastructure\n"
    "  and the owner.\n"
    "- Ineligible failure, or a failure after the one rerun: stop and complete as\n"
    "  returned for changes with the evidence; the source card goes back to its\n"
    "  implementer.\n"
    "- Cancelled, timed out or infrastructure failure: stop and block\n"
    "  (needs_input) for Delivery and infrastructure, not the authors.\n"
    "- The review-labels gate (CI files touched): stop and block for the owner.\n"
    "- A lens asks for changes: complete as returned for changes; the source card\n"
    "  returns to Software engineering with the findings document.\n"
    "- Identical findings twice on an identical head: stop for an owner decision.\n"
    "- Head moved: all evidence and approvals for the head are void; stop and\n"
    "  block for Delivery and infrastructure.\n"
    "- Merge refused (not clean, or a GitHub refusal): stop and block for\n"
    "  Delivery and infrastructure.\n"
    "- Handoff error: retry; block as transient.\n"
)


def integration_card_body(source_task_id: str, head: str) -> str:
    """The integration card's fixed kernel text for ``source_task_id`` at ``head``."""
    return _INTEGRATION_CARD_BODY.format(
        source=source_task_id, head=head, key=delivery_key(source_task_id, head),
    )


def delivery_settings() -> tuple[bool, Optional[str]]:
    """``(enabled, integration_profile)`` from ``kanban.delivery``; off on any doubt."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("kanban", {}).get("delivery") or {}
        enabled = section.get("enabled") is True
        profile = section.get("integration_profile")
    except Exception:
        return False, None
    profile = profile.strip() if isinstance(profile, str) else ""
    return enabled, profile or None


def _sha(value) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if _SHA.fullmatch(text) else None


def _repository(workspace_path) -> Optional[Path]:
    """The card's worktree or, once the clean worktree is removed after the
    approval, the repository the kernel cut it from (``<repo>/.worktrees/<id>``)."""
    path = Path(workspace_path) if workspace_path else None
    if path is None or not path.is_absolute():
        return None
    for candidate in (path, path.parent.parent if path.parent.name == ".worktrees" else None):
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def _run_git(workdir: Path, *args: str) -> Optional[subprocess.CompletedProcess]:
    """One git command in the repository at ``workdir`` only; ``None`` if it cannot run.

    git is resolved as an absolute path from the absolute PATH entries only,
    and the command runs with ``-C`` set to ``workdir``, standard input closed
    and a 30-second timeout. The environment is this process's without any
    GIT_* variable, so GIT_DIR, GIT_OBJECT_DIRECTORY and the like cannot answer
    from another repository, and without system or global configuration.
    """
    search = os.pathsep.join(
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if os.path.isabs(entry)
    )
    git = shutil.which("git", path=search) if search else None
    if git is None or not os.path.isabs(git):
        return None
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    try:
        return subprocess.run(
            [git, "-C", str(workdir), *args],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_proof(
    workdir: Optional[Path], head: Optional[str], base: Optional[str],
) -> tuple[bool, bool]:
    """``(head_exists, base_is_ancestor)`` in the repository at ``workdir`` only,
    each check through :func:`_run_git`. A check that cannot run confirms nothing.
    """
    if workdir is None or head is None:
        return False, False

    def confirms(*args: str) -> bool:
        result = _run_git(workdir, *args)
        return result is not None and result.returncode == 0

    head_exists = confirms("cat-file", "-e", f"{head}^{{commit}}")
    return head_exists, head_exists and base is not None and confirms(
        "merge-base", "--is-ancestor", base, head,
    )


def approval_facts(
    conn: sqlite3.Connection, task_id: str, bound_head: Optional[str],
) -> dict:
    """The T1 approval facts for ``task_id``, from kernel records and git only."""
    row = conn.execute(
        "SELECT status, head_commit, base_commit, workspace_path FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    status = row["status"] if row is not None else None
    head = _sha(row["head_commit"]) if row is not None else None
    base = _sha(row["base_commit"]) if row is not None else None

    # The same cycle rule as task_review_states: the latest handover R, the
    # latest completion C after it, and no invalidating event between them.
    # The reviewer's own claim out of review is part of the cycle it reviews.
    requested = kb._latest_review_handover_id(conn, task_id)
    completed = conn.execute(
        "SELECT id, run_id FROM task_events "
        "WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    invalidations = []
    if requested is not None:
        kinds = kb._REVIEW_APPROVAL_INVALIDATING_EVENT_KINDS
        for event in conn.execute(
            "SELECT id, kind, payload FROM task_events WHERE task_id = ? "
            f"AND id > ? AND kind IN ({','.join('?' for _ in kinds)})",
            (task_id, requested, *kinds),
        ):
            if event["kind"] == "claimed":
                try:
                    payload = json.loads(event["payload"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("source_status") == "review":
                    continue
            invalidations.append(int(event["id"]))
    completed_id = int(completed["id"]) if completed is not None else None
    approved = (
        requested is not None
        and completed_id is not None
        and completed_id > requested
        and not any(event_id < completed_id for event_id in invalidations)
    )
    reopened = completed_id is not None and any(
        event_id > completed_id for event_id in invalidations
    )

    reviewer = None
    if completed is not None and completed["run_id"] is not None:
        run = conn.execute(
            "SELECT profile FROM task_runs WHERE id = ? AND task_id = ?",
            (completed["run_id"], task_id),
        ).fetchone()
        reviewer = run["profile"] if run is not None else None
    implementer, _ = kb._latest_review_provenance(conn, task_id)

    workdir = _repository(row["workspace_path"]) if row is not None else None
    head_exists, base_is_ancestor = _git_proof(workdir, head, base)
    return {
        "source_done": status == "done",
        "approved_by_review_lane": approved,
        "head_commit": head,
        "review_head": _sha(kb._latest_review_head_provenance(conn, task_id)),
        "bound_head": _sha(bound_head),
        "reviewer": reviewer,
        "implementer": implementer,
        "reopened_after_approval": reopened,
        "head_exists": head_exists,
        "base": base,
        "base_is_ancestor": base_is_ancestor,
    }


def prove_approval(
    conn: sqlite3.Connection, task_id: str, bound_head: Optional[str],
) -> Decision:
    """T1 on the kernel's records: may ``bound_head`` be delivered for ``task_id``?"""
    return decide_publish(
        approval_facts(conn, task_id, bound_head), dict(_NO_LEDGER), dict(_NO_REMOTE),
    )


def record_approved_delivery(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Record the delivery of an approved card's head and create its one integration card.

    Returns the integration card id, or ``None`` when delivery is off, the card
    carries no review requirement, or the approval proof refuses. A second call
    for the same source card and head returns the card the first one created.
    A card that only carries the delivery key is never adopted: the call raises
    instead, so the approval, the record and any new card roll back together.
    """
    enabled, profile = delivery_settings()
    if not enabled or profile is None:
        return None
    with kb.write_txn(conn, allow_nested=True):
        row = conn.execute(
            "SELECT requires_review, head_commit FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if row is None or not row["requires_review"]:
            return None
        head = _sha(row["head_commit"])
        proof = prove_approval(conn, task_id, head)
        if not proof.allowed:
            logger.warning(
                "kanban delivery: no delivery for %s: %s (%s)",
                task_id, proof.code, proof.detail,
            )
            return None
        approval = conn.execute(
            "SELECT run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO kanban_deliveries "
            "(source_task_id, source_head, approval_run_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, head, approval["run_id"], int(time.time())),
        )
        record = conn.execute(
            "SELECT id, integration_task_id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (task_id, head),
        ).fetchone()
        if record["integration_task_id"]:
            return record["integration_task_id"]
        body = integration_card_body(task_id, head)
        integration_id = kb.create_task(
            conn,
            title=f"Integrate {task_id} at {head[:12]}",
            body=body,
            assignee=profile,
            created_by=_CREATED_BY,
            owned_paths=[],
            idempotency_key=delivery_key(task_id, head),
        )
        # create_task answers a key that is already taken with that card, whoever
        # made it. Nothing is adopted by its key alone: the card must be this
        # delivery's own kernel card, or the approval, the record and any new
        # card roll back together.
        card = kb.get_task(conn, integration_id)
        if card is None or (
            card.created_by, card.assignee, card.owned_paths, card.body,
        ) != (_CREATED_BY, kb._canonical_assignee(profile), [], body):
            raise RuntimeError(
                f"kanban delivery: the integration card {integration_id} for "
                f"{task_id} at {head} is not this delivery's kernel card"
            )
        conn.execute(
            "UPDATE kanban_deliveries SET integration_task_id = ? WHERE id = ?",
            (integration_id, record["id"]),
        )
        return integration_id


# ---------------------------------------------------------------------------
# T1 kanban_delivery_publish
# ---------------------------------------------------------------------------

# The one branch T1 publishes for a source card (plan section 4): this prefix
# and the source card id, so every run of every delivery of a card names it.
_BRANCH_PREFIX = "delivery/"

# The pull request's base: main, the repository's only protected branch.
_PULL_REQUEST_BASE = "main"

# The reviewed per-repository policy (C1): a repository it does not list is
# never published.
_POLICY_FILE = Path(__file__).with_name("kanban_delivery_policy.json")

# The GitHub repository the card's own repository names as its origin.
_ORIGIN = re.compile(
    r"(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"([A-Za-z0-9-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?"
)

# The pull request's title and body are fixed kernel text with ids only, so
# no worker prose reaches GitHub.
_PULL_REQUEST_TITLE = "Delivery of {source} at {head}"
_PULL_REQUEST_BODY = (
    "Kernel delivery of an approved build.\n"
    "Approved source card: {source}\n"
    "Approved head: {head}\n"
    "Delivery key: {key}\n"
    "Integration card: {card}\n"
)


class PublishRefused(Exception):
    """T1 refused. ``code`` is the fixed reason the tool reports; nothing was stored."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.detail = detail


def delivery_branch(source_task_id: str) -> str:
    return _BRANCH_PREFIX + source_task_id


def _origin_repository(workdir: Optional[Path]) -> Optional[str]:
    """``owner/name`` of the GitHub repository the card's own repository pushes to."""
    if workdir is None:
        return None
    result = _run_git(workdir, "config", "--get", "remote.origin.url")
    if result is None or result.returncode != 0:
        return None
    match = _ORIGIN.fullmatch(result.stdout.decode("utf-8", "replace").strip())
    return match.group(1) if match else None


def _policy_repositories() -> frozenset[str]:
    """The repositories the reviewed policy lists; none when it cannot be read."""
    try:
        policy = load_policy(_POLICY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    return frozenset(policy.required_checks)


def _bind(conn: sqlite3.Connection, task_id: str, run_id: int) -> tuple:
    """C1 and C2: delivery is on, ``task_id`` is the kernel integration card a
    delivery record names, ``run_id`` is its current run, and the source card's
    repository is in the policy. Returns ``(record, workdir, repository)``."""
    enabled, _ = delivery_settings()
    if not enabled:
        raise PublishRefused("delivery_disabled", "kanban.delivery.enabled is not true")
    records = conn.execute(
        "SELECT * FROM kanban_deliveries WHERE integration_task_id = ?", (task_id,),
    ).fetchall()
    card = conn.execute(
        "SELECT created_by, idempotency_key FROM tasks WHERE id = ?", (task_id,),
    ).fetchall()
    if len(records) != 1 or len(card) != 1 or (
        card[0]["created_by"], card[0]["idempotency_key"],
    ) != (_CREATED_BY, delivery_key(records[0]["source_task_id"], records[0]["source_head"])):
        raise PublishRefused(
            "not_integration_card", f"{task_id} is not the integration card of a delivery",
        )
    current = conn.execute(
        "SELECT 1 FROM tasks t "
        "JOIN task_runs r ON r.id = t.current_run_id AND r.task_id = t.id "
        "WHERE t.id = ? AND r.id = ? AND t.status = 'running' "
        "AND r.status = 'running' AND r.ended_at IS NULL",
        (task_id, run_id),
    ).fetchall()
    if not current:
        raise PublishRefused("not_current_run", f"run {run_id} is not the current run of {task_id}")
    record = records[0]
    source = conn.execute(
        "SELECT workspace_path FROM tasks WHERE id = ?", (record["source_task_id"],),
    ).fetchall()
    workdir = _repository(source[0]["workspace_path"]) if source else None
    repository = _origin_repository(workdir)
    if repository is None or repository not in _policy_repositories():
        raise PublishRefused(
            "repository_not_in_policy",
            f"{repository or 'no GitHub origin'} is not in the delivery policy",
        )
    return record, workdir, repository


def publish_available(task_id: str, run_id: int) -> bool:
    """C1 and C2 for the check function: may run ``run_id`` of ``task_id`` see T1?"""
    try:
        with kb.connect_closing(board=os.environ.get("HERMES_KANBAN_BOARD")) as conn:
            _bind(conn, task_id, run_id)
    except Exception:
        return False
    return True


def _snapshot(conn: sqlite3.Connection, task_id: str, run_id: int, source: str) -> tuple:
    """What must be unchanged for a result to be stored: every delivery record
    of the source card, and the integration card's run."""
    records = [tuple(row) for row in conn.execute(
        "SELECT * FROM kanban_deliveries WHERE source_task_id = ? ORDER BY id", (source,),
    )]
    run = [tuple(row) for row in conn.execute(
        "SELECT t.status, t.current_run_id, r.status, r.ended_at FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = ? AND r.task_id = t.id WHERE t.id = ?",
        (run_id, task_id),
    )]
    return records, run


def _lease_refusal(error, code: str, detail: str) -> Exception:
    """A lease that GitHub refuses is the fence's ``code``; any other transport
    failure is reported by its reason alone."""
    if error.reason == "push_lease_mismatch":
        return PublishRefused(code, detail)
    return PublishRefused("transport_failed", error.reason)


def publish_delivery(task_id: str, run_id: int) -> dict:
    """T1: publish the approved head of the delivery whose integration card is
    ``task_id`` as its one pull request, for the card's current run ``run_id``.

    Returns ``{state, head, pull_request_number, branch}`` or raises
    :class:`PublishRefused`. Every fact is captured first, the GitHub work runs
    outside every transaction and connection, and the ledger and the events are
    stored only if the delivery records and the run are unchanged (C4).
    """
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    board = os.environ.get("HERMES_KANBAN_BOARD")
    with kb.connect_closing(board=board) as conn:
        record, workdir, repository = _bind(conn, task_id, run_id)
        source = record["source_task_id"]
        if conn.execute(
            "SELECT 1 FROM kanban_deliveries WHERE source_task_id = ? AND id > ?",
            (source, record["id"]),
        ).fetchall():
            raise PublishRefused("superseded", f"a newer approved head of {source} has its own delivery")
        approval = approval_facts(conn, source, record["source_head"])
        # The approval half of T1 answers before GitHub is touched.
        early = decide_publish(approval, dict(_NO_LEDGER), dict(_NO_REMOTE))
        if not early.allowed:
            raise PublishRefused(early.code, early.detail)
        head = approval["head_commit"]
        approval_event = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'completed'", (source,),
        ).fetchall()[0][0]
        recorded = conn.execute(
            "SELECT pull_request_number, pull_request_head, pull_request_state "
            "FROM kanban_deliveries WHERE source_task_id = ? AND pull_request_number IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchall()
        ledger = dict(_NO_LEDGER)
        if recorded:
            recorded_number, old, recorded_state = recorded[0]
            ledger = {
                "pull_request": recorded_number, "head": old, "state": recorded_state,
                "head_is_ancestor": _git_proof(workdir, head, _sha(old))[1],
            }
        snapshot = _snapshot(conn, task_id, run_id, source)

    branch = delivery_branch(source)
    pulls_path = f"/repos/{repository}/pulls"
    try:
        github = GitHubTransport("publish", repository)
        listed = github.request("GET", pulls_path, query={
            "state": "open", "head": f"{repository.split('/')[0]}:{branch}", "per_page": 100,
        })
        pulls = listed["data"]
        if listed["status"] != 200 or not isinstance(pulls, list) or not all(
            isinstance(pull, dict) for pull in pulls
        ):
            raise PublishRefused("pulls_unreadable", f"GitHub answered {listed['status']}")
        open_pulls = [{
            "number": pull.get("number"),
            "head": pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None,
        } for pull in pulls]
        # No allowlisted endpoint reads a branch, so the recorded pull request's
        # head stands for it, and with nothing recorded the leased push below
        # finds the branch absent (pushed), at H (up to date) or foreign.
        remote = {
            "branch_head": open_pulls[0]["head"] if open_pulls else None,
            "open_pulls": open_pulls,
        }
        decision = decide_publish(approval, ledger, remote)
        if not decision.allowed:
            raise PublishRefused(decision.code, decision.detail)
        if decision.code in ("push_and_create", "adopt_and_create"):
            try:
                pushed = github.push(workdir, branch, head, expected="")
            except GitHubTransportError as error:
                raise _lease_refusal(
                    error, "foreign_branch", f"{branch} holds a head other than {head}",
                ) from None
            state = "adopted" if pushed["state"] == "up_to_date" else "created"
            created = github.request("POST", pulls_path, body={
                "title": _PULL_REQUEST_TITLE.format(source=source, head=head),
                "head": branch,
                "base": _PULL_REQUEST_BASE,
                "body": _PULL_REQUEST_BODY.format(
                    source=source, head=head, key=delivery_key(source, head), card=task_id,
                ),
            })
            pull = created["data"] if isinstance(created["data"], dict) else {}
            number = pull.get("number")
            pulled = pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None
            if created["status"] != 201 or type(number) is not int or number < 1 or pulled != head:
                raise PublishRefused(
                    "pull_request_not_created", f"GitHub answered {created['status']}",
                )
        elif decision.code == "push_fast_forward":
            try:
                github.push(workdir, branch, head, expected=ledger["head"])
            except GitHubTransportError as error:
                raise _lease_refusal(
                    error, "head_moved", f"{branch} no longer holds {ledger['head']}",
                ) from None
            number, state = ledger["pull_request"], "fast_forwarded"
        else:  # already_published or adopt_fast_forward: GitHub already holds H
            number = ledger["pull_request"]
            state = "already_published" if decision.code == "already_published" else "fast_forwarded"
    except GitHubTransportError as error:
        raise PublishRefused("transport_failed", error.reason) from None

    with kb.connect_closing(board=board) as conn:
        with kb.write_txn(conn):
            if _snapshot(conn, task_id, run_id, source) != snapshot:
                raise PublishRefused(
                    "stale_run", "the delivery record or the run changed; nothing was stored",
                )
            if state != "already_published":
                conn.execute(
                    "UPDATE kanban_deliveries SET pull_request_number = ?, pull_request_head = ?, "
                    "pull_request_state = 'open', pull_request_branch = ? WHERE id = ?",
                    (number, head, branch, record["id"]),
                )
                kb._append_event(conn, task_id, "delivery_bound", {
                    "source_task_id": source,
                    "head": head,
                    "base_commit": approval["base"],
                    "reviewer": approval["reviewer"],
                    "implementer": approval["implementer"],
                    "approval_event_id": approval_event,
                    "base_is_ancestor": approval["base_is_ancestor"],
                }, run_id=run_id)
            kb._append_event(conn, task_id, "delivery_published", {
                "repository": repository,
                "branch": branch,
                "pull_request_number": number,
                "head": head,
                "state": state,
            }, run_id=run_id)
    return {
        "state": "created" if state == "adopted" else state,
        "head": head,
        "pull_request_number": number,
        "branch": branch,
    }
