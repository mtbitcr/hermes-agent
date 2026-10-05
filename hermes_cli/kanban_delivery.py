"""Delivery records, the approval proof and the dispatcher's publish step.

When a review-required build card is approved from the review lane and
``kanban.delivery.enabled`` is true, :func:`record_approved_delivery` records
ONE delivery for the source card and its approved head, and nothing else: no
card is created. ``complete_task`` calls it once, from its approval branch,
inside the approval's own transaction, so the approval and the record commit
or roll back together. With the setting off it returns before touching
anything and the approval behaves exactly as before.

The approval proof is the T1 fence of :mod:`hermes_cli.kanban_delivery_fences`
applied to facts read only from the kernel's own records: the task row, its
event log, its runs and the git repository the card was built in. Worker
prose (handover text, comments, summaries, run metadata) is never read.

:func:`publish_step` is T1 itself, run by ``dispatch_once`` after the tick
lock is released, on the board that was ticked; no model runs any part of it.
It leases at most one delivery row per tick, and the same approval facts, the
publish ledger kept on the delivery row and the remote facts read through
:mod:`hermes_cli.kanban_delivery_github` decide whether the approved head
becomes the delivery's one pull request.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_delivery_fences import Decision, decide_publish, load_policy

logger = logging.getLogger(__name__)

# The delivery key the pull request carries (plan section 5) is this prefix, the
# owner-confirmed value, + source card id + ":" + approved head, so one source
# card and head can only ever name one delivery.
DELIVERY_KEY_PREFIX = "delivery:"

_SHA = re.compile(r"[0-9a-f]{40}")

# Nothing is published at approval: no pull request is recorded and no remote
# branch is read, so only the approval half of T1 can decide.
_NO_LEDGER = {
    "pull_request": None, "head": None, "state": None, "head_is_ancestor": False,
}
_NO_REMOTE = {"branch_head": None, "open_pulls": []}


def delivery_key(source_task_id: str, head: str) -> str:
    return DELIVERY_KEY_PREFIX + source_task_id + ":" + head


def delivery_settings() -> bool:
    """``kanban.delivery.enabled``; off on any doubt."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("kanban", {}).get("delivery") or {}
        return section.get("enabled") is True
    except Exception:
        return False


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


def record_approved_delivery(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    """Record the delivery of an approved card's head; no card is created.

    Returns the delivery row id, or ``None`` when delivery is off, the card
    carries no review requirement, or the approval proof refuses. A second call
    for the same source card and head returns the row the first one recorded.
    Every older row of the same source card is marked returned for changes, so
    only the newest approved head is published, as a fast-forward of the one
    pull request when an older head already has it.
    """
    if not delivery_settings():
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
            "SELECT id FROM kanban_deliveries WHERE source_task_id = ? AND source_head = ?",
            (task_id, head),
        ).fetchone()
        conn.execute(
            "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
            "WHERE source_task_id = ? AND id < ?",
            (task_id, record["id"]),
        )
        return record["id"]


# ---------------------------------------------------------------------------
# T1: the dispatcher's publish step
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
)

# How long one step holds a delivery row. A pass that crashes or hangs loses
# the row once its lease runs out, and a later pass takes it over.
_LEASE_SECONDS = 600

# Refusals that can pass on their own: the step logs them and stores nothing,
# and the row is tried again once its lease runs out. Every other refusal
# parks the row and is recorded once on the source card.
_RETRIED = frozenset({
    "delivery_disabled", "transport_failed", "pulls_unreadable",
    "lease_lost", "record_changed", "source_changed",
})


class PublishRefused(Exception):
    """T1 refused. ``code`` is the fixed reason; nothing was stored."""

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


def _bind(conn: sqlite3.Connection, delivery_id: int, holder: str) -> tuple:
    """C1 and C2: delivery is on, ``delivery_id`` is a delivery row with no pull
    request recorded whose publish lease ``holder`` holds, and the source card's
    repository is in the policy. Returns ``(record, workdir, repository)``."""
    if not delivery_settings():
        raise PublishRefused("delivery_disabled", "kanban.delivery.enabled is not true")
    records = conn.execute(
        "SELECT * FROM kanban_deliveries WHERE id = ? AND publish_lease = ? "
        "AND pull_request_number IS NULL",
        (delivery_id, holder),
    ).fetchall()
    if len(records) != 1:
        raise PublishRefused(
            "lease_lost", f"this pass no longer holds the lease of delivery {delivery_id}",
        )
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


def _snapshot(conn: sqlite3.Connection, source: str) -> list:
    """What must be unchanged for a result to be stored: every delivery row of
    the source card, its publish lease included."""
    return [tuple(row) for row in conn.execute(
        "SELECT * FROM kanban_deliveries WHERE source_task_id = ? ORDER BY id", (source,),
    )]


def _source_state(conn: sqlite3.Connection, source: str) -> tuple:
    """What the approval of a publish is proven on: the source card's status,
    its head and its latest event revision."""
    task = [tuple(row) for row in conn.execute(
        "SELECT status, head_commit FROM tasks WHERE id = ?", (source,),
    )]
    return task, kb.task_event_revision(conn, source)


@contextlib.contextmanager
def _one_read(conn: sqlite3.Connection):
    """One SQLite read transaction, so every read inside sees the same committed
    state. It keeps nothing, so it ends with a rollback."""
    conn.execute("BEGIN")
    try:
        yield
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")


def _lease_refusal(error, code: str, detail: str) -> Exception:
    """A lease that GitHub refuses is the fence's ``code``; any other transport
    failure is reported by its reason alone."""
    if error.reason == "push_lease_mismatch":
        return PublishRefused(code, detail)
    return PublishRefused("transport_failed", error.reason)


def _take_lease(db_path: Path) -> Optional[tuple]:
    """Lease the oldest delivery row still to publish: no pull request
    recorded, not returned for changes, not parked, and no lease that still
    holds. Returns ``(delivery_id, holder)``, or ``None``."""
    holder, now = secrets.token_hex(16), int(time.time())
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        row = conn.execute(
            "SELECT id FROM kanban_deliveries WHERE pull_request_number IS NULL "
            "AND pull_request_state IS NULL AND publish_refusal IS NULL "
            "AND (publish_lease_until IS NULL OR publish_lease_until <= ?) "
            "ORDER BY id LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE kanban_deliveries SET publish_lease = ?, publish_lease_until = ? WHERE id = ?",
            (holder, now + _LEASE_SECONDS, row["id"]),
        )
    return row["id"], holder


def _park(db_path: Path, delivery_id: int, holder: str, refusal: PublishRefused) -> None:
    """Park a delivery row this pass still holds: it is not tried again, and
    its refusal is recorded once on the source card."""
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        row = conn.execute(
            "SELECT source_task_id, source_head FROM kanban_deliveries "
            "WHERE id = ? AND publish_lease = ? AND pull_request_number IS NULL",
            (delivery_id, holder),
        ).fetchone()
        if row is None:
            return  # another pass took the row over; it is not this pass's to park
        conn.execute(
            "UPDATE kanban_deliveries SET publish_refusal = ?, publish_lease = NULL, "
            "publish_lease_until = NULL WHERE id = ?",
            (refusal.code, delivery_id),
        )
        kb._append_event(conn, row["source_task_id"], "delivery_refused", {
            "delivery_id": delivery_id,
            "head": row["source_head"],
            "code": refusal.code,
            "detail": refusal.detail,
        })


def publish_step(db_path: Optional[Path]) -> Optional[dict]:
    """The dispatcher's delivery step for the board whose file is ``db_path``:
    lease at most one approved delivery row and publish it.

    Returns what :func:`publish_delivery` returns, or ``None``. A refusal that
    can pass on its own is logged and the row is tried again once its lease
    runs out; any other refusal parks the row. Raises no error, so the tick
    goes on.
    """
    if db_path is None or not delivery_settings():
        return None
    try:
        leased = _take_lease(db_path)
        if leased is None:
            return None
        try:
            return publish_delivery(db_path, *leased)
        except PublishRefused as refusal:
            logger.warning(
                "kanban delivery: delivery %s not published: %s (%s)",
                leased[0], refusal.code, refusal.detail,
            )
            if refusal.code not in _RETRIED:
                _park(db_path, *leased, refusal)
    except Exception:
        # The tick goes on; a row this pass leased is tried again once its lease runs out.
        logger.exception("kanban delivery: the publish step failed")
    return None


def publish_delivery(db_path: Path, delivery_id: int, holder: str) -> dict:
    """T1: publish the approved head of delivery row ``delivery_id``, whose
    publish lease ``holder`` holds, as the source card's one pull request.

    Returns ``{state, head, pull_request_number, branch}`` or raises
    :class:`PublishRefused`. Every fact is captured first, the GitHub work runs
    outside every transaction and connection, and the ledger and the events are
    stored only if the source card's delivery rows, the lease included, and the
    source card are unchanged (C4), and only once GitHub, read again after the
    publish, shows the pull request open at H against main on the branch at H.
    """
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    # The admission, the approval and ledger reads and the snapshot are one read, so a
    # takeover after the admission cannot put the new lease into the snapshot that the
    # final write compares against. The read ends before any GitHub call.
    with kb.connect_closing(db_path=db_path) as conn, _one_read(conn):
        record, workdir, repository = _bind(conn, delivery_id, holder)
        source = record["source_task_id"]
        if conn.execute(
            "SELECT 1 FROM kanban_deliveries WHERE source_task_id = ? AND id > ?",
            (source, record["id"]),
        ).fetchall():
            raise PublishRefused("superseded", f"a newer approved head of {source} has its own delivery")
        # Captured before the approval is proven, so the final write sees any later change.
        approved_source = _source_state(conn, source)
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
        snapshot = _snapshot(conn, source)

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
        remote = {
            "branch_head": github.branch_head(branch),  # None when GitHub has no such branch
            "open_pulls": open_pulls,
        }
        decision = decide_publish(approval, ledger, remote)
        if not decision.allowed:
            raise PublishRefused(decision.code, decision.detail)
        if decision.code in ("push_and_create", "adopt_and_create"):
            state = "adopted"  # the branch already holds H, so nothing is pushed
            if decision.code == "push_and_create":
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
                    source=source, head=head, key=delivery_key(source, head),
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
        else:  # adopt_pull_request, adopt_fast_forward or already_published: GitHub holds H
            number = ledger["pull_request"] or open_pulls[0]["number"]
            state = "fast_forwarded" if decision.code == "adopt_fast_forward" else "adopted"
        # Read again, still outside every transaction: only the pull request open at H
        # against main, on the branch at H, is a success.
        again = github.request("GET", f"{pulls_path}/{number}")
        reread = again["data"] if again["status"] == 200 and isinstance(again["data"], dict) else {}
        reread_head = reread["head"].get("sha") if isinstance(reread.get("head"), dict) else None
        reread_base = reread["base"].get("ref") if isinstance(reread.get("base"), dict) else None
        if (reread.get("state"), reread_head, reread_base, github.branch_head(branch)) != (
            "open", head, _PULL_REQUEST_BASE, head,
        ):
            raise PublishRefused(
                "head_moved",
                f"pull request {number} is not open at {head} against {_PULL_REQUEST_BASE}, "
                f"or {branch} is not at {head}",
            )
    except GitHubTransportError as error:
        raise PublishRefused("transport_failed", error.reason) from None

    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if _snapshot(conn, source) != snapshot:
            raise PublishRefused(
                "record_changed",
                "a delivery row of the source card changed or its lease was taken over; "
                "nothing was stored",
            )
        if _source_state(conn, source) != approved_source:
            raise PublishRefused(
                "source_changed",
                "the source card changed after its approval was proven; nothing was stored",
            )
        conn.execute(
            "UPDATE kanban_deliveries SET pull_request_number = ?, pull_request_head = ?, "
            "pull_request_state = 'open', pull_request_branch = ?, publish_lease = NULL, "
            "publish_lease_until = NULL WHERE id = ?",
            (number, head, branch, delivery_id),
        )
        kb._append_event(conn, source, "delivery_bound", {
            "source_task_id": source,
            "head": head,
            "base_commit": approval["base"],
            "reviewer": approval["reviewer"],
            "implementer": approval["implementer"],
            "approval_event_id": approval_event,
            "base_is_ancestor": approval["base_is_ancestor"],
        })
        kb._append_event(conn, source, "delivery_published", {
            "repository": repository,
            "branch": branch,
            "pull_request_number": number,
            "head": head,
            "state": state,
        })
    return {"state": state, "head": head, "pull_request_number": number, "branch": branch}
