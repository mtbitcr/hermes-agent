"""Delivery records, the approval proof and the dispatcher's publish step.

When a review-required build card is approved from the review lane and
``kanban.delivery.enabled`` is true, :func:`record_approved_delivery` records
ONE delivery for the source card and its approved head; no card is created.
``complete_task`` calls it once, from its approval branch, inside the
approval's own transaction, so the approval and the record commit
or roll back together. With the setting off it
returns before touching anything and the approval behaves exactly as before.

The approval proof is the T1 fence of :mod:`hermes_cli.kanban_delivery_fences`
applied to facts read only from the kernel's own records: the task row, its
event log, its runs and the git repository the card was built in. Worker
prose (handover text, comments, summaries, run metadata) is never read.

:func:`publish_delivery` is T1 itself, run by :func:`publish_step` from
``dispatch_once``: the same approval facts, the publish ledger kept on the delivery row and
the remote facts read through :mod:`hermes_cli.kanban_delivery_github` decide
whether the approved head becomes the delivery's one pull request.

:func:`review_step`, run by the same step, waits for CI on each published head:
on green it creates that head's review cards; while a required check is red it
waits, and a later pass reads CI again.
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
from hermes_cli.kanban_risk_tier import RISK_NOT_RECORDED, UNRECORDED_RISK_TIER, effective_risk_tier
from hermes_cli.kanban_delivery_fences import Decision, _required_check, decide_publish, load_policy

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
        enabled = section.get("enabled") is True
    except Exception:
        return False
    return enabled


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
    Every other row of the source card is returned for changes, and a new
    approval re-arms the row of its head, parked or returned for changes, while
    no pull request is recorded on it.
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
        # A new approval of a head with no pull request recorded re-arms its row
        # and ends any lease on it, so a pass still holding one stores nothing.
        conn.execute(
            "UPDATE kanban_deliveries SET approval_run_id = ?, pull_request_state = NULL, "
            "publish_refusal = NULL, publish_lease = NULL, publish_lease_until = NULL "
            "WHERE source_task_id = ? AND source_head = ? AND pull_request_number IS NULL "
            "AND approval_run_id IS NOT ?",
            (approval["run_id"], task_id, head, approval["run_id"]),
        )
        record = conn.execute(
            "SELECT id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (task_id, head),
        ).fetchone()
        # The approval is the source card's latest, so every other head of it,
        # older or newer, is returned for changes.
        conn.execute(
            "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
            "WHERE source_task_id = ? AND source_head != ?",
            (task_id, head),
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

# How long one pass holds a delivery row. A pass that crashes or hangs loses
# the row once its lease runs out, and a later pass takes it over.
_LEASE_SECONDS = 600

# Refusals that can pass on their own: nothing is stored, and the row is tried
# again once its lease runs out. Any other refusal parks the row, recorded once.
_RETRIED = frozenset({"delivery_disabled", "transport_failed", "lease_lost", "source_changed"})

# Transport reasons that are GitHub's own answer, one the step cannot use: a
# successful status whose body is not a JSON object or list, or a token of
# another scope. They are refusals. Every other reason is a request that did not
# complete, the transport's own refusal before any request, or a failure it does
# not tell apart from a temporary one, so it is ``transport_failed``.
_REFUSED_REASONS = frozenset({"bad_response", "token_scope_mismatch"})


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


def _held(conn: sqlite3.Connection, delivery_id: int, holder: str):
    """Delivery row ``delivery_id`` while ``holder``'s lease holds: the row names
    this holder, its deadline is later than a fresh clock and no pull request is
    recorded. Otherwise raises ``lease_lost``."""
    records = conn.execute(
        "SELECT * FROM kanban_deliveries WHERE id = ? AND publish_lease = ? "
        "AND publish_lease_until > ? AND pull_request_number IS NULL",
        (delivery_id, holder, int(time.time())),
    ).fetchall()
    if len(records) != 1:
        raise PublishRefused("lease_lost", f"the lease of delivery {delivery_id} ran out or was taken over")
    return records[0]


def _bind(conn: sqlite3.Connection, delivery_id: int, holder: str) -> tuple:
    """C1 and C2: delivery is on, ``holder``'s lease on delivery row
    ``delivery_id`` holds, and the source card's
    repository is in the policy. Returns ``(record, workdir, repository)``."""
    if not delivery_settings():
        raise PublishRefused("delivery_disabled", "kanban.delivery.enabled is not true")
    record = _held(conn, delivery_id, holder)
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
    """What must be unchanged for a result to be stored: every delivery record
    of the source card, its publish lease included."""
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


def _transport_refusal(error) -> PublishRefused:
    """A transport failure, classified by its reason."""
    if error.reason in _REFUSED_REASONS:
        return PublishRefused(error.reason, "GitHub answered with a body the step cannot use")
    return PublishRefused("transport_failed", error.reason)


def _lease_refusal(error, code: str, detail: str) -> Exception:
    """A lease that GitHub refuses is the fence's ``code``; any other transport
    failure is classified by its reason."""
    if error.reason == "push_lease_mismatch":
        return PublishRefused(code, detail)
    return _transport_refusal(error)


def _answered(answer: dict) -> dict:
    """GitHub's answer, read as evidence only when it is not a temporary one: a
    rate limit (429) or an outage (5xx) proves nothing, so it stores nothing and
    the row is tried again once its lease runs out. Any other answer the step
    cannot use is a refusal."""
    if answer["status"] == 429 or answer["status"] >= 500:
        raise PublishRefused("transport_failed", f"GitHub answered {answer['status']}")
    return answer


def _branch_head(github, branch: str) -> Optional[str]:
    """The commit ``branch`` holds on GitHub, or ``None`` when GitHub answers 404.
    Read through ``request``: the transport's ``branch_head`` raises one reason
    for a temporary answer and for a malformed one alike."""
    answer = _answered(github.request("GET", f"/repos/{github.repository}/git/ref/heads/{branch}"))
    if answer["status"] == 404:
        return None
    target = answer["data"].get("object") if isinstance(answer["data"], dict) else None
    sha = target.get("sha") if isinstance(target, dict) else None
    if answer["status"] != 200 or not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise PublishRefused("branch_unreadable", f"GitHub answered {answer['status']}")
    return sha


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
    """Park a delivery row while this pass's lease on it holds: it is not tried
    again, and its refusal is recorded once on the source card."""
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        row = _held(conn, delivery_id, holder)
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
    """The dispatcher's delivery step for the board whose file is ``db_path``: act on the
    published heads whose CI is done, then lease at most one approved delivery row and
    publish it. Raises no error, so the tick goes on."""
    if db_path is None or not delivery_settings():
        return None
    review_step(db_path)
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
                with contextlib.suppress(PublishRefused):  # a lease that ran out parks nothing
                    _park(db_path, *leased, refusal)
    except Exception:
        # The tick goes on; a row this pass leased is tried again once its lease runs out.
        logger.exception("kanban delivery: the publish step failed")
    return None


def publish_delivery(db_path: Path, delivery_id: int, holder: str) -> dict:
    """T1: publish the approved head of delivery row ``delivery_id`` as its one
    pull request, while ``holder``'s publish lease on the row holds.

    Returns ``{state, head, pull_request_number, branch}`` or raises
    :class:`PublishRefused`. Every fact is captured first, the GitHub work runs
    outside every transaction and connection, and the ledger and the events are
    stored only if the delivery records, the lease and the source card are
    unchanged (C4), and only once GitHub, read again after any push or creation,
    shows the pull request open at H on the branch at H.
    """
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    def hold() -> None:  # right before each change on GitHub, on its own connection
        if not delivery_settings():  # switched off after the admission: change nothing
            raise PublishRefused("delivery_disabled", "kanban.delivery.enabled is not true")
        with kb.connect_closing(db_path=db_path) as conn:
            _held(conn, delivery_id, holder)

    # The admission, the approval and ledger reads and the snapshot are one read, so a
    # takeover after the admission cannot put the new lease into the snapshot that
    # the final write compares against. The read ends before any GitHub call.
    with kb.connect_closing(db_path=db_path) as conn, _one_read(conn):
        record, workdir, repository = _bind(conn, delivery_id, holder)
        source = record["source_task_id"]
        # Each approval returns every other head of the source card for changes, so
        # only the row of its latest approval is still pending.
        if record["pull_request_state"] is not None:
            raise PublishRefused("superseded", f"a later approval of {source} returned this head for changes")
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
        listed = _answered(github.request("GET", pulls_path, query={
            "state": "open", "head": f"{repository.split('/')[0]}:{branch}", "per_page": 100,
        }))
        pulls = listed["data"]
        if listed["status"] != 200 or not isinstance(pulls, list) or not all(
            isinstance(pull, dict) for pull in pulls
        ):
            raise PublishRefused("pulls_unreadable", f"GitHub answered {listed['status']}")
        open_pulls = [{
            "number": pull.get("number"),
            "head": pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None,
            "base": pull["base"].get("ref") if isinstance(pull.get("base"), dict) else None,
        } for pull in pulls]
        # A pull request of the delivery branch that targets another base, or none, is not the
        # delivery's own: nothing is updated, adopted or bound, before any decision.
        if any(pull["base"] != _PULL_REQUEST_BASE for pull in open_pulls):
            raise PublishRefused("wrong_base", "an open pull request of the delivery branch does not target main")
        remote = {
            "branch_head": _branch_head(github, branch),  # None when GitHub has no such branch
            "open_pulls": open_pulls,
        }
        decision = decide_publish(approval, ledger, remote)
        if not decision.allowed:
            raise PublishRefused(decision.code, decision.detail)
        if decision.code in ("push_and_create", "adopt_and_create"):
            state = "adopted"  # the branch already holds H, so nothing is pushed
            if decision.code == "push_and_create":
                try:
                    hold()
                    pushed = github.push(workdir, branch, head, expected="")
                except GitHubTransportError as error:
                    raise _lease_refusal(
                        error, "foreign_branch", f"{branch} holds a head other than {head}",
                    ) from None
                state = "adopted" if pushed["state"] == "up_to_date" else "created"
            hold()
            created = _answered(github.request("POST", pulls_path, body={
                "title": _PULL_REQUEST_TITLE.format(source=source, head=head),
                "head": branch,
                "base": _PULL_REQUEST_BASE,
                "body": _PULL_REQUEST_BODY.format(
                    source=source, head=head, key=delivery_key(source, head),
                ),
            }))
            pull = created["data"] if isinstance(created["data"], dict) else {}
            number = pull.get("number")
            pulled = pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None
            if created["status"] != 201 or type(number) is not int or number < 1 or pulled != head:
                raise PublishRefused(
                    "pull_request_not_created", f"GitHub answered {created['status']}",
                )
        elif decision.code == "push_fast_forward":
            try:
                hold()
                github.push(workdir, branch, head, expected=ledger["head"])
            except GitHubTransportError as error:
                raise _lease_refusal(
                    error, "head_moved", f"{branch} no longer holds {ledger['head']}",
                ) from None
            number, state = ledger["pull_request"], "fast_forwarded"
        elif decision.code == "adopt_pull_request":  # no ledger, and one pull request open at H
            number, state = open_pulls[0]["number"], "adopted"
        else:  # already_published or adopt_fast_forward: GitHub already holds H
            number = ledger["pull_request"]
            state = "already_published" if decision.code == "already_published" else "fast_forwarded"
        if decision.code in (
            "push_and_create", "adopt_and_create", "push_fast_forward", "adopt_pull_request", "adopt_fast_forward",
        ):
            # GitHub changed or holds an unrecorded pull request, so both are read again, outside every
            # transaction: only the pull request open at H against main on the branch at H is a success.
            again = _answered(github.request("GET", f"{pulls_path}/{number}"))
            reread = again["data"] if again["status"] == 200 and isinstance(again["data"], dict) else {}
            reread_head = reread["head"].get("sha") if isinstance(reread.get("head"), dict) else None
            reread_base = reread["base"].get("ref") if isinstance(reread.get("base"), dict) else None
            if (reread.get("state"), reread_head, reread_base, _branch_head(github, branch)) != (
                    "open", head, _PULL_REQUEST_BASE, head):
                raise PublishRefused(
                    "head_moved",
                    f"pull request {number} or {branch} is not open at {head} after the publish",
                )
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None

    with kb.connect_closing(db_path=db_path) as conn:
        with kb.write_txn(conn):
            _held(conn, delivery_id, holder)
            if _snapshot(conn, source) != snapshot:
                raise PublishRefused(
                    "lease_lost", "the delivery records or the lease changed; nothing was stored",
                )
            if _source_state(conn, source) != approved_source:
                raise PublishRefused(
                    "source_changed",
                    "the source card changed after its approval was proven; nothing was stored",
                )
            if state != "already_published":
                conn.execute(
                    "UPDATE kanban_deliveries SET pull_request_number = ?, pull_request_head = ?, "
                    "pull_request_state = 'open', pull_request_branch = ?, publish_lease = NULL, "
                    "publish_lease_until = NULL WHERE id = ?",
                    (number, head, branch, record["id"]),
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
    return {
        "state": "created" if state == "adopted" else state,
        "head": head,
        "pull_request_number": number,
        "branch": branch,
    }


# ---------------------------------------------------------------------------
# Card 2: the review cards of a published head, once its CI is done
# ---------------------------------------------------------------------------

# This prefix + source card id + ":" + head + ":" + responsibility is a review
# card's idempotency key, so each head gets each of its review cards once.
REVIEW_KEY_PREFIX = "review:"

# The review command's own fields (plan section 2(c)); the reviewer and the
# route are the team policy's, resolved on each pass.
_REVIEW_CREATOR, _REVIEW_TIER = "kanban-delivery", "routine"

# Tiers 0 and 1 get one combined card; tier 2, or no recorded tier, gets a
# correctness (R15) and a security (R12) card (owner decision, 2026-10-04). Only
# the recorded tier decides: code nothing calls yet gets its tier before delivery.
_COMBINED = {"R15": "correctness and security"}
_SPLIT = {"R15": "correctness", "R12": "security"}

# Fixed kernel text with ids only, like the pull request's.
_REVIEW_TITLE = "Review ({lens}) of {source} at {head}"
_REVIEW_BODY = (
    "Kernel review of a published delivery: review pull request {number} on the code host at the "
    "head below, not a local worktree, and post the review on that commit.\nReview: {lens}\n"
    "Source card: {source}\nHead: {head}\nBase commit: {base}\nRepository: {repository}\n"
    "Pull request: {number}\nBranch: {branch}\nRisk tier: {tier}\nCI on {head}: passed ({checks})\n"
)


def review_key(source_task_id: str, head: str, responsibility: str) -> str:
    return f"{REVIEW_KEY_PREFIX}{source_task_id}:{head}:{responsibility}"


def _board_slug(db_path: Path) -> str:
    """The board whose file is ``db_path``, whatever board the environment names."""
    path = Path(db_path)
    return path.parent.name if path.parent.parent.name == "boards" else kb.DEFAULT_BOARD


def _review_route() -> tuple:
    """``(reviewer, route)`` as the team policy resolves them now, or ``(None, None)``."""
    reviewer = kb.policy_resolved_reviewer()
    if reviewer is None:
        return None, None
    try:
        policy = kb._model_policy()
        chosen = policy.resolve_task_assignment(reviewer, policy.normalize_execution_tier(_REVIEW_TIER))
    except Exception:
        return None, None
    return reviewer, {"provider_override": chosen.provider, "model_override": chosen.model,
                      "reasoning_effort": chosen.reasoning_effort}


def _page(github, path: str, key: str, **query) -> tuple:
    """``(objects, total_count)`` of GitHub's one answer of up to 100 objects under ``key``; no other
    page is read. An answer that is not a 200 listing objects of distinct ids is a refusal."""
    answer = _answered(github.request("GET", path, query=dict(query, per_page=100)))
    data = answer["data"] if answer["status"] == 200 and isinstance(answer["data"], dict) else {}
    page, count = data.get(key), data.get("total_count")
    if type(count) is not int or not isinstance(page, list) or not all(
            isinstance(item, dict) and type(item.get("id")) is int for item in page) or (
            len({item["id"] for item in page}) != len(page)):
        raise PublishRefused("ci_unreadable", f"GitHub answered {answer['status']} for {path}")
    return page, count


def _open_at_head(github, repository: str, row: dict) -> bool:
    """The pull request, read through the fenced pull endpoint, is open at the row's head."""
    answer = _answered(github.request("GET", f"/repos/{repository}/pulls/{row['pull_request_number']}"))
    pull = answer["data"] if answer["status"] == 200 and isinstance(answer["data"], dict) else {}
    pulled = pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None
    return (pull.get("state"), pulled) == ("open", row["pull_request_head"])


def _source_read(conn: sqlite3.Connection, source: str) -> tuple:
    """The source card's event revision, status, head and risk tier. The owner workspace
    writes a tier with no event, so the tier is read itself."""
    revision = kb.task_event_revision(conn, source)
    task = conn.execute("SELECT status, head_commit, risk_tier FROM tasks WHERE id = ?", (source,)).fetchone()
    return (revision, *(task or (None, None, None)))


def _unchanged(conn: sqlite3.Connection, row: dict, read: tuple) -> bool:
    """The delivery row still names the pull request, head, state and lease this pass read,
    and the source card is as ``read``, its :func:`_source_read` before the network calls."""
    keys = ("pull_request_number", "pull_request_head", "pull_request_state", "publish_lease")
    now = conn.execute(f"SELECT {', '.join(keys)} FROM kanban_deliveries WHERE id = ?", (row["id"],)).fetchall()
    return [tuple(record) for record in now] == [tuple(row[key] for key in keys)] and (
        _source_read(conn, row["source_task_id"]) == read)


def _outcome(row: dict, **details) -> dict:
    """An outcome's event payload: the delivery and the head it applied to."""
    return {"delivery_id": row["id"], "head": row["pull_request_head"],
            "pull_request_number": row["pull_request_number"], **details}


def _waiting(conn: sqlite3.Connection, row: dict, code: str, **details) -> None:
    """Why the open row still waits, recorded on the source card once for its head and
    ``code``, inside the caller's write transaction."""
    waited = {(json.loads(payload).get("head"), json.loads(payload).get("code")) for (payload,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_review_waiting'",
        (row["source_task_id"],))}
    if (row["pull_request_head"], code) not in waited:
        kb._append_event(conn, row["source_task_id"], "delivery_review_waiting", _outcome(row, code=code, **details))


def review_step(db_path: Path) -> None:
    """Card 2 on the board whose file is ``db_path``: each published head whose CI is
    green gets its review cards; one whose CI still runs or is red waits, and a later
    pass reads CI again. With no reviewer from the team policy nothing is read or
    created. Raises no error, so the tick goes on."""
    try:
        reviewer, route = _review_route() if delivery_settings() else (None, None)
        if reviewer is None:
            return
        with kb.connect_closing(db_path=db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM kanban_deliveries WHERE pull_request_number IS NOT NULL "
                "AND pull_request_state = 'open' ORDER BY id")]
    except Exception:
        logger.exception("kanban delivery: the review step failed")
        return
    for row in rows:
        try:
            _review_delivery(db_path, row, reviewer, route)
        except PublishRefused as refusal:
            logger.warning("kanban delivery: delivery %s not reviewed: %s (%s)",
                           row["id"], refusal.code, refusal.detail)
        except Exception:
            logger.exception("kanban delivery: delivery %s not reviewed", row["id"])


def _review_delivery(db_path: Path, row: dict, reviewer: str, route: dict) -> Optional[str]:
    """Card 2 for one published row: GitHub is read outside every transaction, and an
    outcome or a waiting reason is stored only while the row and its done source card
    are as this pass read them before GitHub. The required check on H, in one request read
    right before the pull request is read again, alone decides green, red or wait; a red
    decision's waiting record holds the id and conclusion of each red required check run,
    as that request answered them. Returns the state stored, if any."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        read = _source_read(conn, source)
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (source,)).fetchone()
        bound = [event[0] for event in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_bound' "
            "ORDER BY id DESC", (source,))]
    if (read[1], _sha(read[2])) != ("done", head):
        return None  # the source card is not done at H: nothing to decide for it
    repository = _origin_repository(_repository(task["workspace_path"])) if task is not None else None
    try:
        policy = load_policy(_POLICY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise PublishRefused("policy_unreadable", "the delivery policy cannot be read") from None
    if repository not in policy.required_checks or _sha(head) != head:
        raise PublishRefused("repository_not_in_policy", f"{repository or 'no GitHub origin'} is not in the policy")
    base = None
    for event in bound:  # the base the publish bound H to
        with contextlib.suppress(ValueError, TypeError, KeyError):
            payload = json.loads(event)
            if payload["head"] == head:
                base = payload["base_commit"]
                break
    try:
        github = GitHubTransport("read_checks", repository)
        if not _open_at_head(github, repository, row):
            return None  # the pull request moved off H or closed: no card for a stale head
        required = policy.required_checks[repository]
        runs, count = _page(github, f"/repos/{repository}/commits/{head}/check-runs", "check_runs", filter="latest")
        if count > 100 or len(runs) != count:  # the one decision request, read last: whole on its one page
            raise PublishRefused("ci_unreadable", f"{count} check runs on {head}, {len(runs)} in the answer")
        named = {name: [run for run in runs if run.get("name") == name] for name in required}
        if not named or any(len(found) != 1 for found in named.values()):
            return None  # a required check that is not in the answer exactly once decides nothing
        outcomes = {_required_check(name, head, {"check_runs": found, "statuses": []}) for name, found in named.items()}
        if outcomes & {"pending", "stale"} or not _open_at_head(github, repository, row):
            return None  # CI on H still runs, or the pull request left H while CI was read: a later step tries again
        if outcomes == {"success"}:
            return _create_review_cards(db_path, row, read, repository, base, required, reviewer, route)
        failed = [{"id": run["id"], "conclusion": run.get("conclusion")} for (run,) in named.values()
                  if run.get("conclusion") != "success"]  # the red required checks, as the decision request read them
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):  # red: no rerun, no rework, no POST
            if delivery_settings() and _unchanged(conn, row, read):
                _waiting(conn, row, "red_check_waiting", check_runs=failed)
        return None
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None


def _create_review_cards(db_path, row, read, repository, base, checks, reviewer, route) -> Optional[str]:
    """H's review cards as the review command makes them, each past the kernel's route guard,
    created with the row's outcome and its event in one transaction, with the tier and project
    the source card has there. No card is adopted by its key: when a key of H already names a
    card, none is created and the conflict is recorded once for H."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read):
            return None
        task = conn.execute("SELECT risk_tier, project_id FROM tasks WHERE id = ?", (source,)).fetchone()
        try:
            tier, recorded = effective_risk_tier(task["risk_tier"]), task["risk_tier"] is not None
        except ValueError:  # not a tier: tier 2, never a lower one
            tier, recorded = UNRECORDED_RISK_TIER, False
        text = {"source": source, "head": head, "base": base, "repository": repository, "checks": ", ".join(checks),
                "number": row["pull_request_number"], "branch": row["pull_request_branch"],
                "tier": tier if recorded else f"{tier}. {RISK_NOT_RECORDED}"}
        lenses = _COMBINED if tier < 2 else _SPLIT
        taken = sorted(card for responsibility in lenses for (card,) in conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?", (review_key(source, head, responsibility),)))
        if taken:
            _waiting(conn, row, "review_key_conflict", cards=taken)
            return None
        cards = []
        for responsibility, lens in lenses.items():
            title = _REVIEW_TITLE.format(lens=lens, **text)
            card = kb.create_task(
                conn, title=title, body=_REVIEW_BODY.format(lens=lens, **text), assignee=reviewer,
                responsibility=responsibility, created_by=_REVIEW_CREATOR, workspace_kind="worktree",
                execution_tier=_REVIEW_TIER, board=_board_slug(db_path), project_id=task["project_id"],
                project_source_task_id=source, owned_paths=[], parents=[source],
                idempotency_key=review_key(source, head, responsibility), **route)
            kb.authorize_executable_transition(conn, card)  # its route lock, or the card parked with why
            cards.append(card)
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'review_cards_created', "
                     "publish_lease = NULL, publish_lease_until = NULL WHERE id = ?", (row["id"],))
        kb._append_event(conn, source, "delivery_review_cards_created", _outcome(row, cards=cards))
        return "review_cards_created"
