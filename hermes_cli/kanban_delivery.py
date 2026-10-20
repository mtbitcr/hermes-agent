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
on green it creates that head's review cards; when a required check is red it
returns the work once to its builder through one continuation card naming the
failed jobs and tests, and nothing reruns a check. An open head whose pull request GitHub
shows closed, merged or not, outside the delivery step is marked closed outside,
its final state. :func:`arm_step` then arms GitHub
auto-merge once on each head whose required review cards are done and approve it,
returns the work of each head whose review requests changes through that same card,
and closes once each pull request a rework card replaced; GitHub does the merge.
:func:`merge_step` reads each armed pull request, and once GitHub shows it merged
into main adds the merge to the release decision through
:mod:`hermes_cli.release_intake` and marks the row merged, its final state.

With ``kanban.delivery.exact_base`` exactly true, :func:`worktree_start` starts a
new card branch of a policy repository at the main GitHub holds now, fetched
through the publish step's transport; a failed fetch starts it at HEAD as before
and :func:`base_fallback` records why, after the base.

With the same setting, :func:`review_step` reads main before any check of a published head, as the arm
reads it: when main is neither the reviewed base nor in the head, it makes no review card and records
the row base_moved, so the update step brings the head up to date before any test wait or review.

With the same setting, :func:`arm_step` merges only on the base review saw: under one merge lock per
repository, a file under the kanban root that every board and process shares, it reads main (an answer
naming another ref than main is no read), merges when main is the reviewed base or is already in the
approved head (:func:`base_allows_merge`), and else sends nothing and records the row base_moved with what it
compared. A merge answer that is neither 200 nor one of the refusals GitHub documents for a merge it did not
make is settled by reading the pull request and main back, still under the lock: armed only when the pull
request is merged into main at exactly the approved head, else unknown, never refused, and the merge step
reads it on every pass; the merge is never sent again. The merge step, too, records only a merge at the
approved head, and records delivery_merged_off_base beside it when the merge's first parent is not the main
the arm checked. When git cannot read that parent, it records delivery_merged_off_base_unknown instead, and
each later pass reads the parent again until one delivery_merged_off_base or delivery_merged_on_base event
closes that check.
"""

from __future__ import annotations

import contextlib
import hashlib
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
from hermes_cli.kanban_delivery_fences import (
    _PASSING, TEST_JOB_STEPS, Decision, _required_check, decide_publish, load_policy,
)

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


def exact_base_settings() -> bool:
    """``kanban.delivery.exact_base``: a new card branch starts at the current main;
    off unless exactly true, and on any doubt."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("kanban", {}).get("delivery") or {}
        exact_base = section.get("exact_base") is True
    except Exception:
        return False
    return exact_base


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
        # and ends any lease on it, so a pass still holding one stores nothing. A head
        # parked behind an armed pull request stays parked: a later card owns its release.
        conn.execute(
            "UPDATE kanban_deliveries SET approval_run_id = ?, pull_request_state = NULL, "
            "publish_refusal = NULL, publish_lease = NULL, publish_lease_until = NULL "
            "WHERE source_task_id = ? AND source_head = ? AND pull_request_number IS NULL "
            "AND approval_run_id IS NOT ? AND IFNULL(publish_refusal, '') != 'auto_merge_armed'",
            (approval["run_id"], task_id, head, approval["run_id"]),
        )
        record = conn.execute(
            "SELECT id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (task_id, head),
        ).fetchone()
        # The approval is the source card's latest, so every other head of it,
        # older or newer, is returned for changes; a head GitHub may hold armed, or merged, stays as it is,
        # and so does one whose pull request GitHub showed closed outside the delivery step.
        conn.execute(
            "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
            "WHERE source_task_id = ? AND source_head != ? AND IFNULL(pull_request_state, '') NOT IN (?, ?, ?, ?, ?)",
            (task_id, head, *_ARM_HELD, _MERGED, _CLOSED_OUTSIDE),
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
# The one account whose GitHub review confirms a done review card before auto-merge is armed.
_REVIEWER_BOT = "raphael-reviewer-mtbitcr[bot]"

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

# GitHub moves a pull request to a head pushed to its branch a few seconds after the push, so after a clean
# update's push the pull request and the branch are read up to six times, these pauses apart, until one monotonic
# deadline _SETTLE_SECONDS after the first read: no read or pause starts once it is reached, a pause asks for at
# most the time left, and only reads answered before it are a success. The pauses add up to less than
# _SETTLE_SECONDS, so a sixth pair of quick reads starts before the deadline. The clock and the sleep are seams:
# tests wait in no real time.
_SETTLE_PAUSES = (2, 4, 8, 16, 29)
_SETTLE_SECONDS = 60
_clock, _sleep = time.monotonic, time.sleep

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


def _branch_head(github, branch: str, *, named: bool = False) -> Optional[str]:
    """The commit ``branch`` holds on GitHub, or ``None`` when GitHub answers 404.
    Read through ``request``: the transport's ``branch_head`` raises one reason
    for a temporary answer and for a malformed one alike. When ``named`` (card 2's
    reads of main), an answer whose ref is not exactly ``refs/heads/<branch>`` is unreadable too."""
    answer = _answered(github.request("GET", f"/repos/{github.repository}/git/ref/heads/{branch}"))
    if answer["status"] == 404:
        return None
    target = answer["data"].get("object") if isinstance(answer["data"], dict) else None
    sha = target.get("sha") if isinstance(target, dict) else None
    if answer["status"] != 200 or not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise PublishRefused("branch_unreadable", f"GitHub answered {answer['status']}")
    if named and answer["data"].get("ref") != f"refs/heads/{branch}":
        raise PublishRefused("branch_unreadable", "GitHub answered for another ref")
    return sha


def _take_lease(db_path: Path, adopt: Optional[int] = None) -> Optional[tuple]:
    """Lease the oldest delivery row still to publish: no pull request
    recorded, not returned for changes, not parked unless it is row ``adopt``
    parked with head_moved, and no lease that still holds. Returns
    ``(delivery_id, holder)``, or ``None``."""
    holder, now = secrets.token_hex(16), int(time.time())
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        row = conn.execute(
            "SELECT id FROM kanban_deliveries WHERE pull_request_number IS NULL "
            "AND pull_request_state IS NULL AND (publish_refusal IS NULL "
            "OR publish_refusal = 'head_moved' AND id = ?) "
            "AND (publish_lease_until IS NULL OR publish_lease_until <= ?) "
            "ORDER BY id LIMIT 1",
            (adopt, now),
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


def _adoptable(db_path: Path) -> Optional[int]:
    """The oldest delivery row parked with head_moved, its lease run out, that GitHub now shows published:
    the source card's recorded pull request open at the row's head and its branch at that head, read through
    the fenced publish transport. Only such a row may be leased again, to be adopted. Returns its id, or ``None``."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    with kb.connect_closing(db_path=db_path) as conn:
        rows = [dict(row) for row in conn.execute(
            f"SELECT d.id, d.source_task_id, d.source_head AS pull_request_head, l.pull_request_number, "
            f"{_RECEIPT.format('l')} AS repository FROM kanban_deliveries AS d JOIN kanban_deliveries AS l "
            "ON l.id = (SELECT MAX(id) FROM kanban_deliveries WHERE source_task_id = d.source_task_id "
            "AND pull_request_number IS NOT NULL) WHERE d.publish_refusal = 'head_moved' "
            "AND d.pull_request_number IS NULL AND d.pull_request_state IS NULL "
            "AND (d.publish_lease_until IS NULL OR d.publish_lease_until <= ?) ORDER BY d.id",
            (int(time.time()),))]
    policy = _policy_repositories() if rows else frozenset()
    for row in rows:
        if row["repository"] not in policy:  # no record of the repository, or one the policy does not list
            continue
        with contextlib.suppress(PublishRefused, GitHubTransportError):  # unreadable: not adopted on this pass
            github = GitHubTransport("publish", row["repository"])
            if _open_at_head(github, row["repository"], row) and _branch_head(
                    github, delivery_branch(row["source_task_id"])) == row["pull_request_head"]:
                return row["id"]
    return None


def publish_step(db_path: Optional[Path]) -> Optional[dict]:
    """The dispatcher's delivery step for the board whose file is ``db_path``: act on the
    published heads whose CI is done, then lease at most one approved delivery row, or one
    parked with head_moved that GitHub now shows published, and publish it. Raises no
    error, so the tick goes on."""
    if db_path is None or not delivery_settings():
        return None
    review_step(db_path)
    merge_step(db_path)
    arm_step(db_path)
    if exact_base_settings():  # card 3: each row left base_moved gets its update card on main, or goes back
        from hermes_cli.kanban_delivery_update import update_step

        update_step(db_path)
    try:
        adopt = _adoptable(db_path)
        leased = _take_lease(db_path, adopt)
        if leased is None:
            return None
        try:
            return publish_delivery(db_path, *leased)
        except PublishRefused as refusal:
            logger.warning(
                "kanban delivery: delivery %s not published: %s (%s)",
                leased[0], refusal.code, refusal.detail,
            )
            # A row not adopted after all stays parked with head_moved; only another refusal parks it again.
            if refusal.code not in _RETRIED and (leased[0] != adopt or refusal.code != "head_moved"):
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
    shows the pull request open at H on the branch at H. A row parked with
    head_moved is only adopted, where GitHub already holds H, with one event
    naming that refusal.
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
        update = None
        if approval["bound_head"] not in (None, approval["head_commit"]):
            # Card B: a head other than H is admitted only as a clean update of H's approval, on the kernel's
            # record of that update, its records of the update heads before it and git's parents of the head,
            # all judged by the fence.
            clean = _clean_record(conn, source, approval["bound_head"])
            update = {"record": clean, "parents": _parents(workdir, approval["bound_head"]),
                      "chain": _update_chain(conn, source, clean)}
        # The approval half of T1 answers before GitHub is touched: for a recorded clean update, H's approval.
        early = decide_publish({**approval, "bound_head": approval["head_commit"]} if update and update["record"]
                               else approval, dict(_NO_LEDGER), dict(_NO_REMOTE))
        if not early.allowed:
            raise PublishRefused(early.code, early.detail)
        head = approval["head_commit"] if update is None else approval["bound_head"]
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
            if recorded_state in _ARM_HELD:  # no new head on it: a later card owns its release
                raise PublishRefused("auto_merge_armed", f"pull request {recorded_number} may have auto-merge armed")
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
        decision = decide_publish(approval, ledger, remote, update)
        if not decision.allowed:
            raise PublishRefused(decision.code, decision.detail)
        # A row parked with head_moved is only adopted, where GitHub already holds its head: nothing is pushed.
        if record["publish_refusal"] is not None and decision.code != "adopt_fast_forward":
            raise PublishRefused("head_moved", f"GitHub does not show {head} on {branch} and its pull request")
        if update is not None:  # the fence's own words: the head it admitted as a clean update of H's approval
            logger.info("kanban delivery: delivery %s: %s", delivery_id, decision.detail)
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
            number, state = ledger["pull_request"], "fast_forwarded" if update is None else "clean_update"
        elif decision.code == "adopt_pull_request":  # no ledger, and one pull request open at H
            number, state = open_pulls[0]["number"], "adopted"
        else:  # already_published or adopt_fast_forward: GitHub already holds H
            number = ledger["pull_request"]
            state = "already_published" if decision.code == "already_published" else "fast_forwarded"
            state = "clean_update" if update is not None else state
        if decision.code in (
            "push_and_create", "adopt_and_create", "push_fast_forward", "adopt_pull_request", "adopt_fast_forward",
        ):
            # GitHub changed or holds an unrecorded pull request, so both are read again, outside every
            # transaction: only the pull request open at H against main on the branch at H is a success.
            # After a clean update's push GitHub moves the pull request a little later, so it is read again, until
            # the deadline: no read starts once it is reached, and only a pair answered before it is a success.
            pauses = _SETTLE_PAUSES if update is not None and decision.code == "push_fast_forward" else ()
            deadline = _clock() + _SETTLE_SECONDS if pauses else float("inf")  # no wait: one pair, as before
            moved = PublishRefused(
                "head_moved", f"pull request {number} or {branch} is not open at {head} after the publish",
            )
            for pause in (*pauses, None):
                if _clock() >= deadline:
                    raise moved
                again = _answered(github.request("GET", f"{pulls_path}/{number}"))
                reread = again["data"] if again["status"] == 200 and isinstance(again["data"], dict) else {}
                reread_head = reread["head"].get("sha") if isinstance(reread.get("head"), dict) else None
                reread_base = reread["base"].get("ref") if isinstance(reread.get("base"), dict) else None
                if _clock() >= deadline:
                    raise moved
                shown = (reread.get("state"), reread_head, reread_base, _branch_head(github, branch))
                left = deadline - _clock()
                if left > 0 and shown == ("open", head, _PULL_REQUEST_BASE, head):
                    break
                if pause is None or left <= 0:
                    raise moved
                _sleep(min(pause, left))
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
                    "publish_lease_until = NULL, publish_refusal = NULL WHERE id = ?",
                    (number, head, branch, record["id"]),
                )
                if record["publish_refusal"] is not None:  # adopted: the one event naming the refusal it parked with
                    kb._append_event(conn, source, "delivery_adopted", {
                        "delivery_id": record["id"],
                        "head": head,
                        "pull_request_number": number,
                        "refusal": record["publish_refusal"],
                        "refusal_event_id": conn.execute(
                            "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'delivery_refused' "
                            "AND json_extract(payload, '$.delivery_id') = ?", (source, record["id"]),
                        ).fetchone()[0],
                    })
                # A clean update is bound to the main it was built on, the one its record names.
                kb._append_event(conn, source, "delivery_bound", {
                    "source_task_id": source,
                    "head": head,
                    "base_commit": approval["base"] if update is None else update["record"]["observed_main"],
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

# Each tier's review cards, by responsibility: the one set read both where a
# head's cards are made and where its merge requires their approvals, so the two
# never differ. Tiers 0 and 1 get one combined card; tier 2, or no recorded tier,
# gets one security (R12) card alone, since the card review before delivery
# already judged the correctness of the same tree. Only the recorded tier
# decides: code nothing calls yet gets its tier before delivery.
_LENSES = {0: {"R15": "correctness and security"}, 1: {"R15": "correctness and security"}, 2: {"R12": "security"}}

# Fixed kernel text: the title names the source card by its own title, and the body carries the ids.
_REVIEW_TITLE = "Review ({lens}) of {title}"
_REVIEW_BODY = (
    "Kernel review of a published delivery: review pull request {number} on the code host at the "
    "head below, not a local worktree, and post the review on that commit.\nReview: {lens}\n"
    "Source card: {source}\nHead: {head}\nBase commit: {base}\nRepository: {repository}\n"
    "Pull request: {number}\nBranch: {branch}\nRisk tier: {tier}\nCI on {head}: passed ({checks})\n"
)
# Card B, truth in reporting: added to the body of a clean update's cards only, from the kernel's record of it, so the
# approval of the approved head is never reported as a review of the updated head.
_REVIEW_CLEAN_UPDATE = (
    "Clean update: the fence admitted the updated head {head}, on main {main}, as a clean update of the approval of "
    "the approved head {approved}. The original card approval remains bound to {approved}.\n"
)
# Added once the card has its id: the line that ties the reviewer's GitHub review to this one card.
_REVIEW_LINE = "Start your GitHub review body with this exact line:\nReview card {card}\n"
# GitHub showed the open row's pull request closed, merged or not, outside the delivery step: final. An approval
# of another head of its source card leaves it as it is, the review step never reads that row again, and the merge
# step, which reads armed or unknown arms only, never records it.
_CLOSED_OUTSIDE = "closed_outside"


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


def _pull(github, repository: str, row: dict) -> dict:
    """The row's pull request, read through the fenced pull endpoint: GitHub's object, or {} for any other answer."""
    answer = _answered(github.request("GET", f"/repos/{repository}/pulls/{row['pull_request_number']}"))
    return answer["data"] if answer["status"] == 200 and isinstance(answer["data"], dict) else {}


def _open_at_head(github, repository: str, row: dict, pull: Optional[dict] = None) -> bool:
    """The pull request, read through the fenced pull endpoint unless ``pull`` is that read, is open at the
    row's head."""
    pull = _pull(github, repository, row) if pull is None else pull
    pulled = pull["head"].get("sha") if isinstance(pull.get("head"), dict) else None
    return (pull.get("state"), pulled) == ("open", row["pull_request_head"])


def _source_read(conn: sqlite3.Connection, source: str) -> tuple:
    """The source card's event revision, status, head and risk tier. The owner workspace
    writes a tier with no event, so the tier is read itself."""
    revision = kb.task_event_revision(conn, source)
    task = conn.execute("SELECT status, head_commit, risk_tier FROM tasks WHERE id = ?", (source,)).fetchone()
    return (revision, *(task or (None, None, None)))


def _clean_record(conn: sqlite3.Connection, source: str, head: str) -> Optional[dict]:
    """Card B: the kernel's record of ``head`` as an update of an approved head of ``source``, the source card's
    latest delivery_updated event naming it as its updated head, which the update step writes in the transaction
    that checks the card's owned paths against what main changed. None when there is none."""
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_updated' "
                                   "ORDER BY id DESC", (source,)):
        with contextlib.suppress(ValueError, TypeError):
            record = json.loads(payload)
            if isinstance(record, dict) and record.get("update_head") == head:
                return record
    return None


def _update_chain(conn: sqlite3.Connection, source: str, record: Optional[dict]) -> list:
    """Card B: the kernel's records (:func:`_clean_record`) of each earlier update head that ``record`` names as its
    previous_update_head, newest first, which the fence reads back to the approved head. The walk ends at a record
    that names none, a head with no record or a head named twice."""
    chain, named = [], set()
    previous = record.get("previous_update_head") if isinstance(record, dict) else None
    while isinstance(previous, str) and previous not in named:
        named.add(previous)
        link = _clean_record(conn, source, previous)
        if link is None:
            break
        chain.append(link)
        previous = link.get("previous_update_head")
    return chain


def _parents(workdir: Optional[Path], head: str) -> list:
    """git's parents of ``head`` in the card's own repository, first parent first; [] when git cannot say."""
    listed = _run_git(workdir, "rev-list", "--parents", "-n", "1", head) if workdir is not None else None
    if listed is None or listed.returncode != 0:
        return []
    return listed.stdout.decode("utf-8", "replace").split()[1:]


def _delivers(conn: sqlite3.Connection, source: str, read: tuple, head: str) -> bool:
    """The source card, as ``read``, is done at ``head``, or done at the approved head the kernel's record
    (:func:`_clean_record`) names ``head`` a clean update of; that head is never reported as reviewed."""
    if read[1] != "done":
        return False
    record = _clean_record(conn, source, head) if _sha(read[2]) != head else None
    return _sha(read[2]) == head or record is not None and record.get("clean") is True and (
        record.get("head") == _sha(read[2]))


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


def _closed_outside(db_path: Path, row: dict, merged: bool) -> Optional[str]:
    """GitHub shows the row's pull request closed, merged or not, outside the delivery step. One write
    transaction marks the row closed outside only while it is still open at the same head and pull request,
    with one event on the source card naming them and whether GitHub shows it merged; a row changed
    meanwhile gets nothing. Returns the state stored, if any."""
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not conn.execute(
                "UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ? AND pull_request_state = 'open' "
                "AND pull_request_head = ? AND pull_request_number = ?",
                (_CLOSED_OUTSIDE, row["id"], row["pull_request_head"], row["pull_request_number"])).rowcount:
            return None  # the row moved on, or names another head or pull request, while GitHub was read
        kb._append_event(conn, row["source_task_id"], f"delivery_{_CLOSED_OUTSIDE}", _outcome(row, merged=merged))
    return _CLOSED_OUTSIDE


def _review_pull(db_path: Path, github, repository: str, row: dict) -> tuple:
    """A review pass's read of the row's pull request, through the fenced pull endpoint: ``(open at H, the state
    stored)``. GitHub's pull request object with the row's own number, state closed and a boolean merged marks the
    row closed outside (:func:`_closed_outside`); any other answer stores nothing."""
    pull = _pull(github, repository, row)
    if (pull.get("number"), pull.get("state")) == (row["pull_request_number"], "closed") and type(
            pull.get("merged")) is bool:  # closed outside the delivery step: final, never read again
        return False, _closed_outside(db_path, row, pull["merged"])
    return _open_at_head(github, repository, row, pull), None


def _review_base(db_path: Path, github, repository: str, row: dict, read: tuple) -> tuple:
    """Card A: with ``kanban.delivery.exact_base``, a review pass reads main as the arm reads it, before any check of
    H, since GitHub starts no CI run on a pull request that conflicts with main. ``(H is reviewed, the state
    stored)``, always reviewed with the setting off: a main neither the reviewed base nor in H gets no review card,
    only the arm's base_moved record with no cards, so the update step brings H up to date first; no main shown
    stores nothing."""
    if not exact_base_settings():
        return True, None
    source, head = row["source_task_id"], row["pull_request_head"]
    main = _branch_head(github, _PULL_REQUEST_BASE, named=True)
    if main is None:
        return False, None
    with kb.connect_closing(db_path=db_path) as conn:
        reviewed_base, workdir = _reviewed(conn, source, head)
    if base_allows_merge(workdir, reviewed_base, main, head):
        return True, None
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read):
            return False, None  # the row or its source card changed while GitHub was read: nothing is stored
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (_BASE_MOVED, row["id"]))
        kb._append_event(conn, source, f"delivery_{_BASE_MOVED}", _outcome(
            row, repository=repository, reviewed_base=reviewed_base, observed_main=main, cards=[]))
    return False, _BASE_MOVED


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
    are as this pass read them before GitHub. The required checks on H, each by its name
    in one request read right before the pull request is read again, alone decide green,
    red or wait; a red decision's waiting record holds the id and conclusion of each red
    required check run, as those requests answered them, before the work is returned. An
    open row whose pull request any read of this
    pass (:func:`_review_pull`: the first, the one after CI, the one before a
    return) shows closed, merged or not, is marked closed outside, and nothing more is read,
    recorded or sent. Card A: with exact_base, main is read after the first pull request read
    and before any check of H (:func:`_review_base`). Returns the state or the red outcome stored, if any."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        read = _source_read(conn, source)
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (source,)).fetchone()
        bound = [event[0] for event in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_bound' "
            "ORDER BY id DESC", (source,))]
        delivers = _delivers(conn, source, read, head)
    if not delivers or row["pull_request_state"] != "open":
        return None  # the source card is not done at H or its clean update, or the row is past open
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
        at_head, stored = _review_pull(db_path, github, repository, row)
        if not at_head:
            return stored  # the pull request closed outside the delivery step, or moved off H: no card for a stale head
        on_base, stored = _review_base(db_path, github, repository, row, read)
        if not on_base:
            return stored  # GitHub shows no main, or H's change goes to the update step first: no check of H is read
        required = policy.required_checks[repository]
        named = {}
        for name in required:  # each required check by its name, so H's other CI runs never fill its one page
            runs, count = _page(github, f"/repos/{repository}/commits/{head}/check-runs", "check_runs",
                                filter="latest", check_name=name)
            if count > 100 or len(runs) != count:  # each decision request, read last: whole on its one page
                raise PublishRefused("ci_unreadable",
                                     f"{count} check runs of {name} on {head}, {len(runs)} in the answer")
            named[name] = [run for run in runs if run.get("name") == name]
        for name, found in named.items():  # a later CI run of H holds a check again: the newest check suite decides
            suites = [run["check_suite"].get("id") if isinstance(run.get("check_suite"), dict) else None
                      for run in found]
            if len(found) > 1 and all(type(suite) is int for suite in suites):  # a rerun keeps its run's older suite
                named[name] = [run for run in found if run["check_suite"]["id"] == max(suites)]
        if not named or any(len(found) != 1 for found in named.values()):
            return None  # a required check missing, or doubled in its newest suite or with no suite, decides nothing
        outcomes = {_required_check(name, head, {"check_runs": found, "statuses": []}) for name, found in named.items()}
        if outcomes & {"pending", "stale"}:
            return None  # CI on H still runs: a later step tries again
        at_head, stored = _review_pull(db_path, github, repository, row)
        if not at_head:
            return stored  # the pull request closed outside the delivery step, or left H, while CI was read
        if outcomes == {"success"}:
            return _create_review_cards(db_path, row, read, repository, base, required, reviewer, route)
        failed = [{"id": run["id"], "conclusion": run.get("conclusion")} for (run,) in named.values()
                  if run.get("conclusion") != "success"]  # the red required checks, as the decision requests read them
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
            if not delivery_settings() or not _unchanged(conn, row, read):
                return None
            _waiting(conn, row, "red_check_waiting", check_runs=failed)
            read = _source_read(conn, source)  # the record just written moved the source card's revision
        return _return_red_check(db_path, row, read, github, repository, {
            run["id"]: (name, run.get("conclusion")) for name, (run,) in named.items()
            if run.get("conclusion") != "success"})
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None


# The one title tests.yml gives every annotation a failed slice job writes on its own check run: first
# the count of failed tests ("count N"), then each failed test id. No other annotation is read. Owner
# rule 4: "count" and a number, 0 included (a setup error's), is the count and never a test id.
_FAILED_TEST, _FAILED_COUNT = "Failed test", re.compile(r"count ([0-9]+)")
_RED_REWORK_BODY = (
    "Rework of {source}: a required check is red on head {head} of pull request {number} on the code host, "
    "and the delivery returned the work to its builder ({reason}). {tests} "
    "The job's CI log has the complete list."
)
_FINDINGS_MAX = 8000  # required behaviour 3: a review return's findings, 8,000 characters in all
_REVIEW_RETURN_BODY = (
    "Rework of {source}: the review of head {head} of pull request {number} on the code host requested changes "
    "on review cards {cards}, and the delivery returned the work to its builder. Continue from head {head}. "
    "The findings each returned review card's latest run recorded:\n{findings}"
)


def _recorded(conn: sqlite3.Connection, source: str, code: str) -> list:
    """The payloads of the source card's waiting records with ``code``, oldest first."""
    payloads = [json.loads(payload) for (payload,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_review_waiting' ORDER BY id",
        (source,))]
    return [payload for payload in payloads if payload.get("code") == code]


def _failed_jobs(github, repository: str, head: str, red: dict) -> Optional[list]:
    """The failed latest jobs of H's workflow run that holds a red required check run (an Actions job is its
    check run), each failed slice job with the test ids its own check run's annotations of the fixed title
    name. Owner rule of round 1: that run is a run of H (head_sha exactly H) on the one page read whose jobs
    with a red required check run's id are each its own (head_sha H, run_id its id), completed, and of the
    name and conclusion the decision requests read (``red``: each id's name and conclusion). None only while
    GitHub shows that run still in progress: its own status exactly queued or in_progress, or, unless the run
    says completed, one of its own jobs'. Read only to fill the return's reason, from its own jobs completed
    with a name and a conclusion not passing: no job ([]) when no run is that run, or an answer cannot be
    read or is not whole on its one page; any other field, missing, malformed or contradicting, is dropped
    and never waits."""
    from hermes_cli.kanban_delivery_github import GitHubTransportError

    try:
        runs, _ = _page(github, f"/repos/{repository}/actions/runs", "workflow_runs", head_sha=head)
        for run in runs:
            if run.get("head_sha") != head:
                continue  # a run of another head is not H's
            page, count = _page(github, f"/repos/{repository}/actions/runs/{run['id']}/jobs", "jobs", filter="latest")
            jobs = [job for job in page if job.get("head_sha") == head and type(job.get("run_id")) is int
                    and job["run_id"] == run["id"]]  # the run's own jobs of H
            own = {job["id"] for job in jobs}
            checks = [job for job in page if job["id"] in red]
            if not checks or any(job["id"] not in own or (job.get("name"), job.get("status"), job.get("conclusion"))
                                 != (red[job["id"]][0], "completed", red[job["id"]][1]) for job in checks):
                continue  # the run holds no red required check run of H, or one other than the decision read it
            running = ("queued", "in_progress")  # exactly; a run that says completed contradicts a job said running
            if run.get("status") in running or (run.get("status") != "completed" and any(
                    job.get("status") in running for job in jobs)):
                return None  # GitHub shows the run still running (ci.yaml's timing report runs after the summary
                # check): return later
            failed = [job for job in jobs if job.get("status") == "completed" and isinstance(job.get("name"), str)
                      and isinstance(job.get("conclusion"), str) and job["conclusion"] not in _PASSING]
            if count != len(page):
                return []  # not every job of the run whole on its one page
            for job in failed:
                if job["name"] in TEST_JOB_STEPS:
                    job["failed_tests"] = _failed_tests(github, repository, job["id"])
            return failed
    except (PublishRefused, GitHubTransportError):
        pass
    return []


def _failed_tests(github, repository: str, job: int) -> list:
    """The test ids a failed slice job's own check run names under the fixed title, after its one count (a
    blank message is no test id); none when that answer cannot be read, is malformed, or names more tests
    than its count."""
    from hermes_cli.kanban_delivery_github import GitHubTransportError

    try:
        answer = _answered(github.request(
            "GET", f"/repos/{repository}/check-runs/{job}/annotations", query={"per_page": 100}))
    except (PublishRefused, GitHubTransportError):
        return []
    if answer["status"] != 200 or not isinstance(answer["data"], list):
        return []
    notes = [note["message"] for note in answer["data"]
             if note["title"] == _FAILED_TEST and isinstance(note["message"], str) and note["message"].strip()]
    counts = [int(found[1]) for found in map(_FAILED_COUNT.fullmatch, notes) if found]
    tests = [note for note in notes if not _FAILED_COUNT.fullmatch(note)]
    return tests if len(counts) == 1 and len(set(tests)) == len(tests) <= counts[0] else []


def _return_red_check(db_path: Path, row: dict, read: tuple, github, repository: str, red: dict):
    """A red required check on H returns the work once for H: red_check_returned and one continuation card
    for the builder, whose reason names the failed jobs of H's run and the failed tests CI names
    (:func:`_failed_jobs`), or the names of ``red``'s red required checks when no such job remains. Only GitHub
    showing H's run still in progress makes the pass wait. Earlier passes' red_check_rerun and
    red_check_rerun_outcome records of H neither block nor repeat the return. Nothing is recorded unless the
    pull request, read after H's run, jobs and annotations, is still open at H; that read showing it closed
    outside the delivery step marks the open row closed outside instead (:func:`_review_pull`). Returns the
    outcome recorded, if any."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        if any(record["head"] == head for record in _recorded(conn, source, "red_check_returned")):
            return None  # H's work went back to its builder already
    jobs = _failed_jobs(github, repository, head, red)
    if jobs is None:
        return None
    reason = (f"failed jobs: {', '.join(job['name'] for job in jobs)}" if jobs
              else f"red required check: {', '.join(name for name, _ in red.values())}")
    branch = _return_branch(db_path, row)  # owner rule 2: the branch comes first
    at_head, stored = _review_pull(db_path, github, repository, row)
    if not at_head:
        return stored  # owner rule 1: the pull request left H or closed while H's run was read
    tests = [test for job in jobs for test in job.get("failed_tests", ())]
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read) or any(
                record["head"] == head for record in _recorded(conn, source, "red_check_returned")):
            return None
        _waiting(conn, row, "red_check_returned", reason=reason, tests=tests,
                 **_continuation(conn, db_path, row, _red_body(row, reason, tests), branch))
    return "red_check_returned"


def _red_body(row: dict, reason: str, tests: list) -> str:
    """A red check continuation's body: H, the pull request, the reason and the failed tests as CI named them."""
    named = f"CI named these failed tests: {', '.join(tests)}." if tests else "No failed test was named."
    return _RED_REWORK_BODY.format(source=row["source_task_id"], head=row["pull_request_head"],
                                   number=row["pull_request_number"], reason=reason, tests=named)


def _continued(conn: sqlite3.Connection, source: str, head: str) -> tuple:
    """Owner rule 3: the identity of the source, H and "continuation", and the card the source card's own
    review_followup_recorded events name for it with that card's status, done or archived included; the
    card and status are None while none is recorded."""
    identity = kb._review_followup_identity_key(reviewed_task_id=source, candidate=f"{head} continuation")
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id = ? "
                                   "AND kind = 'review_followup_recorded' ORDER BY id DESC", (source,)):
        recorded = json.loads(payload)
        if recorded.get("identity") == identity and recorded.get("followup_task_id"):
            item = recorded["followup_task_id"]
            status = conn.execute("SELECT status FROM tasks WHERE id = ?", (item,)).fetchone()
            return identity, item, status[0] if status else None
    return identity, None, None


def _return_branch(db_path: Path, row: dict) -> Optional[tuple]:
    """Owner rule 2: H's continuation branch, named from the source card and H only, made at H before the write
    transaction or reused when it points at H already, in the continuation's repository. By the owner's rule of
    round 2 that is the primary folder the project registry records for the source card's project, whose origin
    names the repository the delivery recorded and which holds H. When the folder cannot be read, either fact
    is false or the name points at another commit, the pass is refused and records nothing. No branch is ever
    deleted: a pass stopped before its commit leaves only this branch at H, which the next pass reuses. None
    when H's continuation is recorded already, as that card has its branch."""
    from hermes_cli import projects_db

    source, head = row["source_task_id"], row["pull_request_head"]
    branch, folder = f"delivery-return/{source}-{head[:12]}", None
    with kb.connect_closing(db_path=db_path) as conn:
        if _continued(conn, source, head)[1] is not None:
            return None
        project = kb.get_task(conn, source).project_id
        (repository,) = conn.execute(f"SELECT {_RECEIPT.format('d')} FROM kanban_deliveries AS d WHERE d.id = ?",
                                     (row["id"],)).fetchone()
    if project:
        with contextlib.suppress(Exception), projects_db.connect_closing() as registry:  # kb.create_task's registry
            found = projects_db.get_project(registry, project)
            folder = found.primary_path if found is not None else None
    root = Path(folder) if folder and os.path.isabs(folder) else None
    if root is None or not repository or _origin_repository(root) != repository or not _git_proof(root, head, None)[0]:
        raise PublishRefused("no_repository", f"the project folder of {source} does not hold {head} of {repository}")
    _run_git(root, "branch", branch, head)  # git refuses a name that exists; the read below decides
    found = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if found is None or found.returncode != 0 or _sha(found.stdout.decode()) != head:
        raise PublishRefused("branch_not_at_head", f"{branch} is not at {head}")
    return root, branch


def _builder_route(conn: sqlite3.Connection, source: str, builder: Optional[str]) -> dict:
    """The route ``builder`` last ran the card ``source`` on, under the card's route fields: the provider, model and
    effort that the runtime receipt of its latest run of the card records (``runtime_receipt`` in the run's
    metadata, as the kernel stamps it). An effort a card cannot hold, as the receipt's "provider-default" for a run
    that set none of its own, is NULL, which on a card means just that: the worker profile's own effort. No lock
    binds a route with no effort, so a governed card parks on it. Empty when it has no run of the card, or that
    run has no receipt naming a provider and model."""
    row = conn.execute("SELECT * FROM task_runs WHERE task_id = ? AND profile = ? ORDER BY started_at DESC, id DESC "
                       "LIMIT 1", (source, builder)).fetchone()
    run = kb.Run.from_row(row) if row is not None else None
    receipt = run.metadata.get("runtime_receipt") if run is not None and isinstance(run.metadata, dict) else None
    if not isinstance(receipt, dict):
        return {}
    provider, model, effort = (receipt.get(key) for key in ("provider", "model", "reasoning_effort"))
    try:
        effort = kb.normalize_reasoning_effort(effort) if isinstance(effort, str) else None
    except ValueError:
        effort = None
    if not all(isinstance(part, str) and part.strip() for part in (provider, model)):
        return {}
    return {"provider_override": provider, "model_override": model, "reasoning_effort": effort}


def _continuation(conn: sqlite3.Connection, db_path: Path, row: dict, body: str, branch: Optional[tuple]) -> dict:
    """H's one continuation card, for a review or a red check return: a task_links child of the source for
    its builder, with its owned paths, tiers, route and review requirement, the route the builder's own where the
    receipt of its latest run of the source records one (:func:`_builder_route`), titled through owner_title and
    recorded under the identity of the source, H and "continuation". Owner rule 3: a recorded one is used for
    its lifetime, and one not done or archived gets ``body`` as one comment. A new card is made in the source
    card's project, which anchors it under the project's primary folder, on ``branch``, the folder and branch
    :func:`_return_branch` made at H, so H is its recorded base; anchored anywhere else, it refuses the pass.
    Owner rule 1 of round 3: the card records H as its expected base, and the branch is read once more here, in
    the caller's write transaction with no network call before its commit; not at H, no card is made. Returns
    the return event's fields: the card as ``rework`` and, when it was created parked, the reason as ``parked``."""
    from hermes_cli.owner_workspace import owner_title

    source, head = row["source_task_id"], row["pull_request_head"]
    identity, item, status = _continued(conn, source, head)
    if item is not None:
        if status not in ("done", "archived", None):
            kb.add_comment(conn, item, _REVIEW_CREATOR, body)
        return {"rework": item}
    task = kb.get_task(conn, source)
    governed = conn.execute("SELECT execution_tier, provider_override, model_override, reasoning_effort, "
                            "model_policy_lock, owner_receipt_bound FROM tasks WHERE id = ?", (source,)).fetchone()
    route = {key: governed[key] for key in ("execution_tier", "provider_override", "model_override",
                                            "reasoning_effort")}
    implementer = kb._latest_review_provenance(conn, source)[0]
    route.update(_builder_route(conn, source, implementer))  # an approved review leaves the reviewer's on the source
    root, name = branch
    found = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    if found is None or found.returncode != 0 or _sha(found.stdout.decode()) != head:
        raise PublishRefused("branch_not_at_head", f"{name} is not at {head} in the write transaction")
    lock = parked = None
    # Owner rule 1: the source card's tier and its builder's route, else the source's own, locked for the builder;
    # unadmitted, the card parks. Owner rule 3 of round 3: its owner_receipt_bound too, and that route as it is:
    # locked, the card keeps that route's effort with that lock, which the pin of a new card would move. When a
    # governed source's route cannot be locked, the card is created parked with the reason and stays governed, never
    # a manual card: it keeps that tier and route as stored, which create_task would strip or refuse, and, governed
    # by nothing else, the source's lock, which binds no route with no tier.
    try:
        lock = kb.mint_policy_lock(implementer, route["provider_override"], route["model_override"],
                                   route["reasoning_effort"], route["execution_tier"])
    except ValueError as error:
        parked = str(error) if kb.task_is_policy_governed(governed) else None
    item = kb.create_task(
        conn, title=owner_title(task.title), body=body, assignee=implementer, parents=[source], tenant=task.tenant,
        workspace_kind="worktree", project_id=task.project_id, branch_name=name, owned_paths=task.owned_paths,
        risk_tier=task.risk_tier, requires_review=task.requires_review, model_policy_lock=lock,
        receipt_owned=bool(governed["owner_receipt_bound"]), board=_board_slug(db_path), **({} if parked else route))
    conn.execute("UPDATE tasks SET base_commit = ? WHERE id = ?", (head, item))  # the claim keeps it
    if Path(kb.get_task(conn, item).workspace_path or "") != root / ".worktrees" / item:
        raise PublishRefused("no_repository", f"the project of {source} no longer anchors its cards in {root}")
    if lock:  # a pin's record stays only where the pin kept that effort
        effort = kb.normalize_reasoning_effort(route["reasoning_effort"])
        conn.execute("UPDATE tasks SET pinned_effort = CASE WHEN reasoning_effort IS ? THEN pinned_effort END, "
                     "reasoning_effort = ?, model_policy_lock = ? WHERE id = ?", (effort, effort, lock, item))
    if parked:
        alone = not (governed["execution_tier"] or governed["owner_receipt_bound"])
        conn.execute("UPDATE tasks SET execution_tier = ?, provider_override = ?, model_override = ?, "
                     "reasoning_effort = ?, model_policy_lock = ? WHERE id = ?",
                     (*route.values(), governed["model_policy_lock"] if alone else None, item))
        kb._pause_unpinnable_task(conn, item, parked)  # as the readiness guard parks a card it cannot lock
    else:
        kb.authorize_executable_transition(conn, item)  # its route lock, or none for a manual source's card
    kb._append_event(conn, source, "review_followup_recorded",
                     {"followup_task_id": item, "implementer": implementer, "identity": identity})
    return {"rework": item, **({"parked": parked} if parked else {})}


def _create_review_cards(db_path, row, read, repository, base, checks, reviewer, route) -> Optional[str]:
    """H's review cards as the review command makes them, each past the kernel's route guard,
    created with the row's outcome and its event in one transaction, with the tier and project
    the source card has there. No card is adopted by its key: when a key of H already names a
    card, none is created and the conflict is recorded once for H."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read):
            return None
        task = conn.execute("SELECT risk_tier, project_id, title FROM tasks WHERE id = ?", (source,)).fetchone()
        try:
            tier, recorded = effective_risk_tier(task["risk_tier"]), task["risk_tier"] is not None
        except ValueError:  # not a tier: tier 2, never a lower one
            tier, recorded = UNRECORDED_RISK_TIER, False
        text = {"source": source, "head": head, "base": base, "repository": repository, "checks": ", ".join(checks),
                "number": row["pull_request_number"], "branch": row["pull_request_branch"], "title": task["title"],
                "tier": tier if recorded else f"{tier}. {RISK_NOT_RECORDED}"}
        # Card B: H and main come only from the kernel's clean-update record of this head, as _delivers reads it.
        record = _clean_record(conn, source, head) if _sha(read[2]) != head else None
        main = _sha(record.get("observed_main")) if record is not None and record.get("clean") is True and (
            record.get("head") == _sha(read[2])) else None
        clean = _REVIEW_CLEAN_UPDATE.format(head=head, main=main, approved=record["head"]) if main else ""
        lenses = _LENSES[tier]
        taken = sorted(card for responsibility in lenses for (card,) in conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?", (review_key(source, head, responsibility),)))
        if taken:
            _waiting(conn, row, "review_key_conflict", cards=taken)
            return None
        cards = []
        for responsibility, lens in lenses.items():
            title = _REVIEW_TITLE.format(lens=lens, **text)
            lock = None
            # Unadmitted routes still park through the existing readiness guard.
            with contextlib.suppress(ValueError):
                lock = kb.mint_policy_lock(
                    reviewer, route["provider_override"], route["model_override"],
                    route["reasoning_effort"], _REVIEW_TIER)
            card = kb.create_task(
                conn, title=title, body=_REVIEW_BODY.format(lens=lens, **text) + clean, assignee=reviewer,
                responsibility=responsibility, created_by=_REVIEW_CREATOR, workspace_kind="worktree",
                execution_tier=_REVIEW_TIER, board=_board_slug(db_path), project_id=task["project_id"],
                project_source_task_id=source, owned_paths=[], parents=[source],
                idempotency_key=review_key(source, head, responsibility),
                risk_tier=tier, model_policy_lock=lock, **route)
            conn.execute("UPDATE tasks SET body = body || ? WHERE id = ?", (_REVIEW_LINE.format(card=card), card))
            kb.authorize_executable_transition(conn, card)  # its route lock, or the card parked with why
            cards.append(card)
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'review_cards_created', "
                     "publish_lease = NULL, publish_lease_until = NULL WHERE id = ?", (row["id"],))
        kb._append_event(conn, source, "delivery_review_cards_created", _outcome(row, cards=cards))
        return "review_cards_created"


# ---------------------------------------------------------------------------
# Card 5: a fully approved head merged; a replaced pull request closed
# ---------------------------------------------------------------------------

# Each mutation is first recorded as pending, in the write transaction that reads its
# evidence a last time, and only the pass that wrote that sends it; its answer is then
# recorded once. A head's arm, its merge, ends armed (200), refused (with GitHub's other status) or unknown (no
# status), and is never sent again: a new head is a new row. GitHub may hold a pending,
# armed or unknown arm, so no new head is pushed to that pull request. A close ends
# replaced or close_refused (with GitHub's status), its delivery_close_pending event
# the record that it was sent.
_PENDING, _ARMED, _ARM_REFUSED, _ARM_UNKNOWN = (
    "auto_merge_pending", "auto_merge_armed", "auto_merge_refused", "auto_merge_unknown")
_ARM_HELD = (_PENDING, _ARMED, _ARM_UNKNOWN)
_REPLACED, _CLOSE_REFUSED = "replaced", "close_refused"
# Card 2 (kanban.delivery.exact_base): main is neither the reviewed base nor in H, so nothing was sent; the
# row waits for part 3. The merge lock is waited for this long, then the next pass tries again.
_BASE_MOVED = "base_moved"
# Card 3: a base_moved row whose update card, its builder's, carries the change on current main now, to its own
# review. Final; its pull request closes once the update is published (:func:`_replaced`).
_UPDATED = "updated"
_MERGE_LOCK_SECONDS = 30
_MERGE_REFUSALS = frozenset({403, 404, 405, 409, 422})  # the answers GitHub documents for a merge it did not make
_REVIEW_PAGES = 10  # owner rule 2 of round 3: a pull request's reviews are read up to 10 pages of 100
# A row's repository: the publish step's own record of the pull request it published,
# never the current git origin. No record, no repository, and nothing is sent.
_RECEIPT = ("(SELECT json_extract(payload, '$.repository') FROM task_events WHERE task_id = {0}.source_task_id "
            "AND kind = 'delivery_published' AND json_extract(payload, '$.head') = {0}.pull_request_head "
            "AND json_extract(payload, '$.pull_request_number') = {0}.pull_request_number ORDER BY id DESC LIMIT 1)")


def arm_step(db_path: Path) -> None:
    """Card 5 on the board whose file is ``db_path``: merge the pull request once at each
    head whose required review cards are done and approve it, and close once the pull
    request of each head a rework card continues. GitHub merges exactly the approved head. Raises no error,
    so the tick goes on. A head whose review requests changes returns to its builder instead."""
    try:
        with kb.connect_closing(db_path=db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT d.*, {_RECEIPT.format('d')} AS repository FROM kanban_deliveries AS d "
                "WHERE d.pull_request_number IS NOT NULL AND d.pull_request_state = 'review_cards_created' "
                "ORDER BY d.id")]
            replaced = _replaced(conn)
    except Exception:
        logger.exception("kanban delivery: the auto-merge step failed")
        return
    for act, items in ((_arm_delivery, rows), (_close_replaced, replaced)):
        for row in items:
            try:
                act(db_path, row)
            except PublishRefused as refusal:
                logger.warning("kanban delivery: delivery %s: %s (%s)", row["id"], refusal.code, refusal.detail)
            except Exception:
                logger.exception("kanban delivery: delivery %s not armed or closed", row["id"])


def _replaced(conn: sqlite3.Connection, old_id: Optional[int] = None) -> list:
    """Each published row (or row ``old_id``) whose pull request, its repository and number,
    another card's work continues: that card is a task_links child of the row's source card,
    its recorded base is the row's head or it is the row's update card (card 3), which only the
    kernel's delivery_updated event of the row and its head names, and its own
    published pull request is another one of the same repository. The row is still the latest
    on its pull request, and no close of it was sent; a merged row is final (card 10's owner
    rule 2), and so is a row closed outside."""
    old, new, later = (_RECEIPT.format(name) for name in ("old", "new", "later"))
    return [dict(row) for row in conn.execute(
        f"SELECT old.*, {old} AS repository, MIN(rework.id) AS replaced_by FROM kanban_deliveries AS old "
        "JOIN tasks AS rework ON (rework.base_commit = old.pull_request_head OR EXISTS (SELECT 1 FROM task_events "
        "WHERE task_id = old.source_task_id AND kind = 'delivery_updated' AND json_extract(payload, '$.delivery_id') "
        "= old.id AND json_extract(payload, '$.head') = old.pull_request_head "
        "AND json_extract(payload, '$.update_task_id') = rework.id)) "
        "AND rework.id != old.source_task_id "
        "JOIN task_links AS link ON link.parent_id = old.source_task_id AND link.child_id = rework.id "
        "JOIN kanban_deliveries AS new ON new.source_task_id = rework.id AND new.pull_request_number IS NOT NULL "
        f"AND new.pull_request_number != old.pull_request_number AND {new} = {old} "
        "WHERE old.pull_request_number IS NOT NULL AND (? IS NULL OR old.id = ?) "
        "AND IFNULL(old.pull_request_state, '') NOT IN (?, ?) "
        "AND NOT EXISTS (SELECT 1 FROM kanban_deliveries AS later WHERE later.id > old.id "
        f"AND later.pull_request_number = old.pull_request_number AND {later} = {old}) "
        "AND NOT EXISTS (SELECT 1 FROM task_events WHERE task_id = old.source_task_id "
        "AND kind = 'delivery_close_pending' AND json_extract(payload, '$.delivery_id') = old.id) "
        "GROUP BY old.id ORDER BY old.id", (old_id, old_id, _MERGED, _CLOSED_OUTSIDE))]


def _approving_cards(conn: sqlite3.Connection, row: dict, read: tuple, returned: bool = False) -> list:
    """The review cards the source card's tier requires for the row's head, when the source
    is done at that head and each of them exists, is done and returned no changes; else [].
    Each is its id, key, creator, status, own head and base, and event revision. With
    ``returned``, a card that returned changes counts too."""
    source, head = row["source_task_id"], row["pull_request_head"]
    if not _delivers(conn, source, read, head):
        return []
    try:
        tier = effective_risk_tier(read[3])
    except ValueError:  # not a tier: tier 2, never a lower one
        tier = UNRECORDED_RISK_TIER
    cards = [conn.execute("SELECT id, idempotency_key, created_by, status, head_commit, base_commit FROM tasks "
                          "WHERE idempotency_key = ? AND created_by = ?",
                          (review_key(source, head, responsibility), _REVIEW_CREATOR)).fetchone()
             for responsibility in _LENSES[tier]]
    if any(card is None or card["status"] != "done" or not returned and conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'changes_requested'", (card["id"],)).fetchone()
           for card in cards):
        return []
    return [(*card, kb.task_event_revision(conn, card["id"])) for card in cards]


def _lenses_approve(reviews: list, head: str, cards: list) -> bool:
    """Owner rule 1 of round 2. Of the reviews, oldest first as GitHub lists them, only the
    reviewer bot's count: its latest approves H, and for each card so does its latest review
    whose first line is exactly that card's line. No lens stands in for another."""
    own = [review for review in reviews if isinstance(review, dict) and isinstance(review.get("user"), dict)
           and review["user"].get("login") == _REVIEWER_BOT]
    return all(found and (found[-1].get("state"), found[-1].get("commit_id")) == ("APPROVED", head) for found in (
        own, *([review for review in own if review.get("body") == f"Review card {card[0]}"] for card in cards)))


def _requests_changes(reviews: list, head: str, card: str) -> bool:
    """Required behaviour 1: the reviewer bot's latest review on H of ``card`` (its first line) requests
    changes. No other account's review counts, and the bot's later approval stands (safety rule 4)."""
    own = [review for review in reviews if isinstance(review, dict) and isinstance(review.get("user"), dict)
           and review["user"].get("login") == _REVIEWER_BOT and review.get("body") == f"Review card {card}"
           and review.get("commit_id") == head]
    return bool(own) and own[-1].get("state") == "CHANGES_REQUESTED"


def _findings(conn: sqlite3.Connection, cards: list) -> str:
    """Each returned card's findings, as its latest run recorded them (``findings`` in the run's
    metadata, as kanban_complete records it), bounded to 8,000 characters in all."""
    parts = []
    for card in cards:
        run = kb.latest_run(conn, card)
        found = run.metadata.get("findings") if run is not None and isinstance(run.metadata, dict) else None
        items = found if isinstance(found, list) else [] if found is None else [found]
        lines = [item if isinstance(item, str) else json.dumps(item, sort_keys=True) for item in items]
        parts.append("\n".join([f"Review card {card}:", *(lines or ["No findings were recorded."])]))
    return "\n".join(parts)[:_FINDINGS_MAX]


def _return_review(db_path: Path, row: dict, evidence: tuple, github, returned: list) -> Optional[str]:
    """Required behaviour 1: after every GitHub read the pull request is read again; open at H, one write
    transaction reads the evidence again, records review_returned once for H and makes H's one continuation
    card. Nothing is sent to GitHub; the row keeps its state, so a later approval of H by every lens arms it."""
    from hermes_cli.kanban_delivery_github import GitHubTransportError

    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        if any(record["head"] == head for record in _recorded(conn, source, "review_returned")):
            return None  # H's work went back to its builder already
    branch = _return_branch(db_path, row)  # owner rule 2: the branch comes first
    try:
        if not _open_at_head(github, row["repository"], row):
            return None  # safety rule 1: the pull request left H or closed while its reviews were read
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or _arm_evidence(conn, row) != evidence or any(
                record["head"] == head for record in _recorded(conn, source, "review_returned")):
            return None  # the evidence changed while GitHub was read, or H returned already: nothing is made
        body = _REVIEW_RETURN_BODY.format(source=source, head=head, number=row["pull_request_number"],
                                          cards=", ".join(returned), findings=_findings(conn, returned))
        _waiting(conn, row, "review_returned", cards=returned, **_continuation(conn, db_path, row, body, branch))
    return "review_returned"


def _sent(github, call, *args, **kwargs) -> dict:
    """A mutation's answer, or no status when none came: it is never sent again either way."""
    from hermes_cli.kanban_delivery_github import GitHubTransportError

    try:
        return getattr(github, call)(*args, **kwargs)
    except GitHubTransportError as error:
        return {"status": None, "data": None, "reason": error.reason}


def _arm_delivery(db_path: Path, row: dict) -> Optional[str]:
    """Merge the pull request at the row's head H, once (owner rules 1 to 3 of round 2). Every GitHub
    read comes first: the pull request, open at H, then its reviews, which approve H lens by
    lens. One write transaction then reads the complete evidence again, finds the pull request
    unclaimed and reserves the arm under this attempt. The arm, GitHub's merge of exactly H, is the one
    call after it, and its status line alone, its body never read, is written to this attempt only: armed on
    200, else refused with that status, or unknown with no status. A full tenth page of reviews that GitHub does not
    show to be the last refuses H once instead, and a review requesting changes returns H (:func:`_return_review`).
    With exact_base, card 2 holds the repository's merge lock from a read of main, which must name main, to the
    recorded answer; refused is then only a refusal GitHub documents for a merge it did not make (_MERGE_REFUSALS),
    and any other answer but 200, with or without a status, is read back once (:func:`_read_back`): armed at
    exactly H, else unknown, never refused, and never sent again."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    source, head, number, repository = (row[key] for key in (
        "source_task_id", "pull_request_head", "pull_request_number", "repository"))
    with kb.connect_closing(db_path=db_path) as conn:
        evidence = _arm_evidence(conn, row)
        claimed = _claimed(conn, repository, number)
    if claimed or not evidence[3] or repository not in _policy_repositories() or not delivery_settings():
        return None  # claimed, a required card is missing or open, or no repository recorded
    reviews, whole = [], False
    try:
        github = GitHubTransport("publish", repository)
        if not _open_at_head(github, repository, row):
            return None  # the pull request left H or is closed: nothing to arm
        for page in range(1, _REVIEW_PAGES + 1):  # oldest first; the last is not full or names no next page
            listed = _answered(github.request("GET", f"/repos/{repository}/pulls/{number}/reviews",
                                              query={"per_page": 100, "page": page}))
            if listed["status"] != 200 or not isinstance(listed["data"], list):
                return None  # a page GitHub does not list: nothing is armed or recorded on this pass
            reviews, whole = reviews + listed["data"], len(listed["data"]) < 100 or listed.get("next_page") is False
            if whole:
                break
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None
    returned = [card[0] for card in evidence[3] if whole and _requests_changes(reviews, head, card[0])]
    if returned:
        return _return_review(db_path, row, evidence, github, returned)
    if not evidence[2] or whole and not _lenses_approve(reviews, head, evidence[2]):
        return None  # a card returned changes, or a lens or the bot's latest review does not approve H
    attempt, exact, moved, observed = secrets.token_hex(8), whole and exact_base_settings(), None, {}
    with contextlib.ExitStack() as held:
        if exact:  # card 2: from main's read to the answer's record, one repository's arms run one at a time
            from hermes_cli.sqlite_util import InitLockUnavailable, cross_process_init_lock

            try:
                held.enter_context(cross_process_init_lock(
                    _merge_lock_path(repository), timeout_seconds=_MERGE_LOCK_SECONDS, required=True))
            except InitLockUnavailable:
                logger.warning("kanban delivery: delivery %s: the merge lock is held elsewhere", row["id"])
                return None  # nothing is sent or recorded: the next pass tries again
            try:
                main = _branch_head(github, _PULL_REQUEST_BASE, named=True)
            except GitHubTransportError as error:
                raise _transport_refusal(error) from None
            if main is None:
                return None  # GitHub shows no main: nothing is sent or recorded on this pass
            with kb.connect_closing(db_path=db_path) as conn:
                reviewed_base, workdir = _reviewed(conn, source, head)
            observed = {"observed_main": main}
            if not base_allows_merge(workdir, reviewed_base, main, head):
                moved = {"reviewed_base": reviewed_base, **observed}
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
            if not delivery_settings() or _arm_evidence(conn, row) != evidence or _claimed(conn, repository, number):
                return None  # the evidence changed while GitHub was read: nothing is sent
            if not whole:  # a full tenth page, a next page named or unknown: H is refused once and never sent
                conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (_ARM_REFUSED, row["id"]))
                kb._append_event(conn, source, f"delivery_{_ARM_REFUSED}", _outcome(
                    row, repository=repository, status=None, reason="review_history_over_10_pages",
                    cards=[card[0] for card in evidence[2]]))
                return _ARM_REFUSED
            if moved:  # main left the base review saw and is not in H: nothing is sent, the lease is untouched
                conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (_BASE_MOVED, row["id"]))
                kb._append_event(conn, source, f"delivery_{_BASE_MOVED}", _outcome(
                    row, repository=repository, **moved, cards=[card[0] for card in evidence[2]]))
                return _BASE_MOVED
            conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = ? WHERE id = ?",
                         (_PENDING, attempt, row["id"]))
        answer = _sent(github, "request", "PUT", f"/repos/{repository}/pulls/{number}/merge",
                       body={"sha": head, "merge_method": "merge"})
        status, reason = answer["status"], answer.get("reason")
        state = _ARM_UNKNOWN if status is None else _ARMED if status == 200 else _ARM_REFUSED
        if exact and status != 200 and status not in _MERGE_REFUSALS:
            state, read = _read_back(github, repository, number, head)  # not a documented refusal: read once
            observed = {**observed, **read}
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
            if conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = NULL "
                            "WHERE id = ? AND publish_lease = ?", (state, row["id"], attempt)).rowcount:
                kb._append_event(conn, source, f"delivery_{state}", _outcome(
                    row, repository=repository, status=status, reason=reason,
                    cards=[card[0] for card in evidence[2]], attempt=attempt, **observed))
    return state


def _merge_lock_path(repository: str) -> Path:
    """The repository's merge lock: under the kanban root, which no board or profile changes, so every
    dispatcher and process on the host computes the same file."""
    return kb.kanban_home() / "kanban" / "merge-locks" / hashlib.sha256(repository.lower().encode()).hexdigest()


def _reviewed(conn: sqlite3.Connection, source: str, head: str) -> tuple:
    """The base the publish bound ``head`` to, as :func:`_intake` reads it, and the card's own repository."""
    task = conn.execute("SELECT workspace_path FROM tasks WHERE id = ?", (source,)).fetchone()
    bound = conn.execute(
        "SELECT json_extract(payload, '$.base_commit') FROM task_events WHERE task_id = ? "
        "AND kind = 'delivery_bound' AND json_extract(payload, '$.head') = ? ORDER BY id DESC LIMIT 1",
        (source, head)).fetchone()
    return (_sha(bound[0]) if bound is not None else None), (_repository(task[0]) if task is not None else None)


def base_allows_merge(workdir: Optional[Path], reviewed_base: Optional[str], main: str, head: str) -> bool:
    """Card 2's rule (owner question 1): main is the base review saw, or main is already in the approved
    head, so the merge result is what review approved. Containment is git's answer in the card's own
    repository; a git failure, no repository or a commit git does not know is not contained."""
    if _sha(main) != main:
        return False
    if main == _sha(reviewed_base):
        return True
    if workdir is None or _sha(head) != head:
        return False
    contained = _run_git(Path(workdir), "merge-base", "--is-ancestor", main, head)
    return contained is not None and contained.returncode == 0


def _read_back(github, repository: str, number: int, head: str) -> tuple:
    """An unconfirmed merge answer settled by one read of the pull request and one of main, never by a second
    merge: shown merged into main at exactly H it is armed (merge_step records it); else it is unknown, which
    merge_step reads on every pass: not merged yet, a merge or an open head other than H, or a read that failed
    or cannot be used. read_back names which, with the head, merge commit and main the read showed. Never
    refused. No failure escapes: the row is pending."""
    pull, main = {"status": None, "data": None}, None
    try:
        pull = github.request("GET", f"/repos/{repository}/pulls/{number}")
        main = _branch_head(github, _PULL_REQUEST_BASE, named=True)
    except Exception:
        pass  # a read failed: what the pull request read showed is still named
    shown = _pull_head(pull, number)
    read = {key: value for key, value in (
        ("read_head", shown), ("merge_commit", _merge_commit(pull, number)), ("main_on_read", main)) if value}
    if main is None or shown is None:
        return _ARM_UNKNOWN, {"read_back": "read_failed", **read}
    if shown != head:
        return _ARM_UNKNOWN, {"read_back": "other_head", **read}
    if "merge_commit" in read:
        return _ARMED, {"read_back": "merged_at_head", **read}
    return _ARM_UNKNOWN, {"read_back": "not_merged", **read}


def _pull_head(answer: dict, number: int) -> Optional[str]:
    """The head commit of GitHub's answer when it is a read of pull request ``number``."""
    pull = answer["data"] if answer["status"] == 200 and isinstance(answer["data"], dict) else {}
    shown = pull.get("head") if pull.get("number") == number and isinstance(pull.get("head"), dict) else {}
    return _sha(shown.get("sha"))


def _arm_evidence(conn: sqlite3.Connection, row: dict) -> tuple:
    """All the arm stands on, read at once: the delivery row (its repository receipt, number,
    head, state and claim), the source card's read, its approving and its done review cards."""
    now = conn.execute(f"SELECT d.*, {_RECEIPT.format('d')} AS repository FROM kanban_deliveries AS d "
                       "WHERE d.id = ?", (row["id"],)).fetchone()
    if now is None or any(now[key] != row[key] for key in (
            "source_task_id", "pull_request_number", "pull_request_head", "pull_request_state", "repository")):
        return (None, None, [], [])
    read = _source_read(conn, row["source_task_id"])
    return tuple(now), read, _approving_cards(conn, row, read), _approving_cards(conn, row, read, returned=True)


def _claimed(conn: sqlite3.Connection, repository: str, number: int) -> bool:
    """Owner rule 3 of round 2: an arm, a close or a merge record of this pull request holds its one action
    claim, the attempt in publish_lease, which a published row holds for nothing else."""
    return conn.execute(f"SELECT 1 FROM kanban_deliveries AS d WHERE d.publish_lease IS NOT NULL "
                        f"AND d.pull_request_number = ? AND {_RECEIPT.format('d')} = ?",
                        (number, repository)).fetchone() is not None


def _close_replaced(db_path: Path, row: dict) -> None:
    """Close, once, the pull request of a head a rework card continues (owner rules 2 and 3 of
    round 2). GitHub must show it open at that head; one write transaction then reads the
    replacement proof again, finds the pull request unclaimed and reserves the close
    under this attempt, and the one call setting its state to closed follows. Done only when
    the answer shows that pull request closed; else a close refusal, which keeps an armed hold."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    repository, number = row["repository"], row["pull_request_number"]
    with kb.connect_closing(db_path=db_path) as conn:
        proof = _close_proof(conn, row)
        claimed = _claimed(conn, repository, number)
    if claimed or not proof or repository not in _policy_repositories() or not delivery_settings():
        return
    try:
        github = GitHubTransport("publish", repository)
        if not _open_at_head(github, repository, row):
            return  # the pull request left the replaced head or is closed: nothing to close
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None
    attempt = secrets.token_hex(8)
    held = row["pull_request_state"] if row["pull_request_state"] in _ARM_HELD else None
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or _close_proof(conn, row) != proof or _claimed(conn, repository, number):
            return
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = ? WHERE id = ?",
                     (held or "close_pending", attempt, row["id"]))
        kb._append_event(conn, row["source_task_id"], "delivery_close_pending", _outcome(
            row, repository=repository, replaced_by=row["replaced_by"], attempt=attempt))
    answer = _sent(github, "request", "PATCH", f"/repos/{repository}/pulls/{number}", body={"state": "closed"})
    data = answer["data"] if isinstance(answer["data"], dict) else {}
    closed = (answer["status"], data.get("number"), data.get("state")) == (200, number, "closed")
    merge = _merge_commit(answer, number) if closed else None
    if merge is not None:  # card 10's owner rule 3: GitHub merged it; merged once recorded, else read again
        _merged(db_path, row, merge, attempt, held or _ARM_UNKNOWN)
        return
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = NULL "
                        "WHERE id = ? AND publish_lease = ?",
                        (_REPLACED if closed else held or _CLOSE_REFUSED, row["id"], attempt)).rowcount:
            kb._append_event(conn, row["source_task_id"], f"delivery_{_REPLACED if closed else _CLOSE_REFUSED}",
                             _outcome(row, repository=repository, replaced_by=row["replaced_by"],
                                      status=answer["status"], closed=closed, attempt=attempt))


def _close_proof(conn: sqlite3.Connection, row: dict) -> tuple:
    """The replacement proof, read at once, by design not covering every change of the source or
    continuation evidence (owner rule 1 of round 3): the row still replaced as ``row`` holds it
    (its repository, number, head and state), its source card's status, head and tier, and the
    continuing card's status and head and its deliveries' heads, repositories and numbers."""
    if _replaced(conn, row["id"]) != [row]:
        return ()
    source, rework = (conn.execute("SELECT status, head_commit, risk_tier FROM tasks WHERE id = ?", (task,)).fetchone()
                      for task in (row["source_task_id"], row["replaced_by"]))
    deliveries = conn.execute(f"SELECT d.source_head, {_RECEIPT.format('d')}, d.pull_request_number, "
                              "d.pull_request_head FROM kanban_deliveries AS d WHERE d.source_task_id = ? "
                              "ORDER BY d.id", (row["replaced_by"],)).fetchall()
    return tuple(source or ()), tuple(rework or ())[:2], [tuple(delivery) for delivery in deliveries]


# ---------------------------------------------------------------------------
# Card 10: each merged change added to the release decision
# ---------------------------------------------------------------------------

# GitHub merged the armed pull request into main and the release record holds the merge: final.
_MERGED = "merged"
# Card 2's check of a merge's first parent against the main the arm checked. delivery_merged_off_base records
# that they differ; when git cannot read the parent, delivery_merged_off_base_unknown leaves the check open, and
# each later pass reads the parent again until one delivery_merged_off_base or delivery_merged_on_base closes it.
_OFF_BASE, _OFF_BASE_UNKNOWN, _ON_BASE = (
    "delivery_merged_off_base", "delivery_merged_off_base_unknown", "delivery_merged_on_base")


def merge_step(db_path: Path) -> None:
    """Card 10 on the board whose file is ``db_path``: read the pull request of each armed or
    unknown arm and, once GitHub shows it merged into main, add the merge to the waiting
    release decision and mark the row merged. A failure leaves the row as it was, so the next
    pass reads it again. With exact_base (card 2), each first-parent check an earlier pass left open is read
    again first (:func:`_recheck`). Raises no error, so the tick goes on."""
    checks = _open_checks(db_path) if exact_base_settings() else []  # a check this pass opens waits for the next
    try:
        with kb.connect_closing(db_path=db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT d.*, {_RECEIPT.format('d')} AS repository FROM kanban_deliveries AS d "
                "WHERE d.pull_request_number IS NOT NULL AND d.pull_request_state IN (?, ?) ORDER BY d.id",
                (_ARMED, _ARM_UNKNOWN))]
    except Exception:
        logger.exception("kanban delivery: the merge step failed")
        return
    for check in checks:
        try:
            _recheck(db_path, check)
        except Exception:
            logger.exception("kanban delivery: the first parent of merge %s of delivery %s is not checked",
                             check["merge_commit"], check["id"])
    for row in rows:
        try:
            _record_merge(db_path, row)
        except PublishRefused as refusal:
            logger.warning("kanban delivery: delivery %s: %s (%s)", row["id"], refusal.code, refusal.detail)
        except Exception:
            logger.exception("kanban delivery: the merge of delivery %s is not recorded", row["id"])


def _record_merge(db_path: Path, row: dict) -> Optional[str]:
    """One armed row: its pull request, read through the fenced pull endpoint, merged into main
    gives the merge commit and the confirmation that it is on main. The rest is what review
    approved, never recomputed from the merge: H, the base the publish bound H to, and H's tree
    read from the card's own repository. The release record ignores a repeat, so a merge whose
    row could not be marked is still recorded once, however often it is seen. With exact_base (card 2), only a
    merge at exactly H counts: one at another head is not recorded, and the row stays as it is; one whose first
    parent is not the main the arm checked is recorded all the same, with one delivery_merged_off_base event,
    and one whose first parent git cannot read with one delivery_merged_off_base_unknown event, a check that
    :func:`merge_step` reads again on later passes."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    head, number, repository = (row[key] for key in ("pull_request_head", "pull_request_number", "repository"))
    with kb.connect_closing(db_path=db_path) as conn:
        claimed = _claimed(conn, repository, number)
    if claimed or repository not in _policy_repositories() or _sha(head) != head:
        return None  # card 10's owner rule 1: another action holds it, read later; or no receipt in the policy
    try:
        answer = _answered(GitHubTransport("read_checks", repository).request(
            "GET", f"/repos/{repository}/pulls/{number}"))
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None
    merge = _merge_commit(answer, number)
    exact = merge is not None and exact_base_settings()
    if merge is None or exact and _pull_head(answer, number) != head:
        return None  # not merged into main, or (card 2) merged at another head than H: a later pass reads it again
    off_base = _off_base(db_path, row, merge) if exact else None  # read before the claim's write, never in it
    attempt = secrets.token_hex(8)
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if _claimed(conn, repository, number) or not conn.execute(
                "UPDATE kanban_deliveries SET publish_lease = ? WHERE id = ? AND pull_request_state = ?",
                (attempt, row["id"], row["pull_request_state"])).rowcount:
            return None  # another action took the pull request, or the row moved on, while GitHub was read
        if off_base:
            _record_off_base(conn, row, merge, *off_base)
    return _merged(db_path, row, merge, attempt, row["pull_request_state"])


def _off_base(db_path: Path, row: dict, merge: str) -> Optional[tuple]:
    """Card 2: the event the first-parent check of the merge at H records, and its details: the main the arm
    checked (the observed_main of the latest armed or unknown record of this delivery and H), the reviewed base
    and, for delivery_merged_off_base, the first parent when it is not the checked main; or
    delivery_merged_off_base_unknown when git cannot read it (:func:`_first_parent`), which leaves the check
    open. Else None: no main was checked, or the merge is on it. Raises nothing the fetch raises."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        checked = conn.execute(
            "SELECT json_extract(payload, '$.observed_main') FROM task_events WHERE task_id = ? AND kind IN (?, ?) "
            "AND json_extract(payload, '$.delivery_id') = ? AND json_extract(payload, '$.head') = ? "
            "ORDER BY id DESC LIMIT 1", (source, f"delivery_{_ARMED}", f"delivery_{_ARM_UNKNOWN}", row["id"], head),
        ).fetchone()
        reviewed_base, workdir = _reviewed(conn, source, head)
    checked = _sha(checked[0]) if checked is not None else None
    if checked is None:
        return None
    parent = _first_parent(row, workdir, merge)
    if parent is None:
        return _OFF_BASE_UNKNOWN, {"observed_main": checked, "reviewed_base": reviewed_base}
    if parent == checked:
        return None  # merged on the main the arm checked
    return _OFF_BASE, {"first_parent": parent, "observed_main": checked, "reviewed_base": reviewed_base}


def _first_parent(row: dict, workdir: Optional[Path], merge: str) -> Optional[str]:
    """Card 2: the first parent of ``merge`` as git reads it in the card's repository ``workdir``, after one fetch
    of main when git does not hold the merge; None, which one warning names, when there is no repository or git
    still cannot read it. Raises nothing the fetch raises."""
    from hermes_cli.kanban_delivery_github import GitHubTransport

    def read() -> Optional[str]:
        found = _run_git(workdir, "rev-parse", "--verify", "--quiet", f"{merge}^1") if workdir else None
        return _sha(os.fsdecode(found.stdout)) if found is not None and found.returncode == 0 else None

    parent = read()
    if parent is None and workdir is not None:
        try:  # git does not hold the merge: main is fetched once, into refs/remotes/origin/main only
            GitHubTransport("publish", row["repository"]).fetch_main(workdir)
        except Exception:
            pass  # the fetch failed: git is read once more all the same
        parent = read()
    if parent is None:
        logger.warning("kanban delivery: delivery %s: the first parent of merge %s is unreadable", row["id"], merge)
    return parent


def _record_off_base(conn: sqlite3.Connection, row: dict, merge: str, kind: str, details: dict) -> None:
    """In the caller's write transaction: the first-parent check's ``kind`` event for this delivery and merge
    commit, unless one is there already or the check is closed, by a delivery_merged_off_base or
    delivery_merged_on_base event. So each delivery and merge commit has at most one open check and one
    event that closes it."""
    if conn.execute("SELECT 1 FROM task_events WHERE task_id = ? AND kind IN (?, ?, ?) "
                    "AND json_extract(payload, '$.delivery_id') = ? AND json_extract(payload, '$.merge_commit') = ?",
                    (row["source_task_id"], kind, _OFF_BASE, _ON_BASE, row["id"], merge)).fetchone() is None:
        kb._append_event(conn, row["source_task_id"], kind, _outcome(
            row, repository=row["repository"], merge_commit=merge, **details))


def _open_checks(db_path: Path) -> list:
    """Card 2: each first-parent check a delivery_merged_off_base_unknown event left open on a merged row, as
    that row with the event's merge commit, checked main and reviewed base; a check is open while no
    delivery_merged_off_base or delivery_merged_on_base event of that delivery and merge commit closes it.
    No check when they cannot be read, which one error names, so the pass records merges all the same."""
    try:
        with kb.connect_closing(db_path=db_path) as conn:
            return [dict(check) for check in conn.execute(
                f"SELECT d.*, {_RECEIPT.format('d')} AS repository, json_extract(e.payload, '$.merge_commit') "
                "AS merge_commit, json_extract(e.payload, '$.observed_main') AS observed_main, "
                "json_extract(e.payload, '$.reviewed_base') AS reviewed_base FROM kanban_deliveries AS d "
                "JOIN task_events AS e ON e.task_id = d.source_task_id AND e.kind = ? "
                "AND json_extract(e.payload, '$.delivery_id') = d.id WHERE d.pull_request_state = ? AND NOT EXISTS ("
                "SELECT 1 FROM task_events AS c WHERE c.task_id = d.source_task_id AND c.kind IN (?, ?) "
                "AND json_extract(c.payload, '$.delivery_id') = d.id AND json_extract(c.payload, '$.merge_commit') "
                "= json_extract(e.payload, '$.merge_commit')) ORDER BY e.id",
                (_OFF_BASE_UNKNOWN, _MERGED, _OFF_BASE, _ON_BASE))]
    except Exception:
        logger.exception("kanban delivery: the open first-parent checks are not read")
        return []


def _recheck(db_path: Path, check: dict) -> None:
    """Card 2: one open check read again, the merge's first parent read as :func:`_record_merge` reads it. Still
    unreadable, the check stays open for the next pass; else one event closes it, in a write transaction that
    first finds it still open: delivery_merged_off_base when the parent is not the check's checked main,
    delivery_merged_on_base when it is, each with the merge commit, the first parent, that checked main and the
    reviewed base. The row and the release record stay as they are; no GitHub API is called, only main fetched."""
    merge = check["merge_commit"]
    if _sha(merge) != merge:
        return  # not a merge commit this step recorded: git is not read
    with kb.connect_closing(db_path=db_path) as conn:
        workdir = _reviewed(conn, check["source_task_id"], check["pull_request_head"])[1]
    parent = _first_parent(check, workdir, merge)
    if parent is None:
        return
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        _record_off_base(conn, check, merge, _ON_BASE if parent == check["observed_main"] else _OFF_BASE, {
            "first_parent": parent, "observed_main": check["observed_main"], "reviewed_base": check["reviewed_base"]})


def _merge_commit(answer: dict, number: int) -> Optional[str]:
    """The merge commit of GitHub's answer when it shows pull request ``number`` merged into main."""
    pull = answer["data"] if answer["status"] == 200 and isinstance(answer["data"], dict) else {}
    base = pull["base"].get("ref") if isinstance(pull.get("base"), dict) else None
    merge = _sha(pull.get("merge_commit_sha"))
    return merge if (pull.get("number"), pull.get("merged"), base) == (number, True, _PULL_REQUEST_BASE) else None


def _merged(db_path: Path, row: dict, merge: str, attempt: str, kept: str) -> str:
    """Record the merge for the action holding ``attempt`` and settle that attempt only (card 10's owner rules
    2 and 3): recorded, merged, final and unclaimed; else ``kept``, a state the merge pass reads, unclaimed."""
    recorded = None
    try:
        recorded = _intake(db_path, row, merge)
    finally:
        state = kept if recorded is None else _MERGED
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
            settled = conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = NULL "
                                   "WHERE id = ? AND publish_lease = ?", (state, row["id"], attempt)).rowcount
            if settled and state == _MERGED:
                kb._append_event(conn, row["source_task_id"], f"delivery_{_MERGED}", _outcome(
                    row, repository=row["repository"], merge_commit=merge, recorded=recorded, attempt=attempt))
    return state


def _intake(db_path: Path, row: dict, merge: str) -> Optional[bool]:
    """The merge through the release intake with the reviewed values :func:`_record_merge` names: whether it
    was new, or None when the reviewed base or tree is unreadable."""
    from hermes_cli import release_intake

    source, head, number, repository = (row[key] for key in (
        "source_task_id", "pull_request_head", "pull_request_number", "repository"))
    with kb.connect_closing(db_path=db_path) as conn:
        task = conn.execute("SELECT title, risk_tier, workspace_path FROM tasks WHERE id = ?", (source,)).fetchone()
        bound = conn.execute(
            "SELECT json_extract(payload, '$.base_commit') FROM task_events WHERE task_id = ? "
            "AND kind = 'delivery_bound' AND json_extract(payload, '$.head') = ? ORDER BY id DESC LIMIT 1",
            (source, head)).fetchone()
    workdir = _repository(task["workspace_path"]) if task is not None else None
    tree = _run_git(workdir, "rev-parse", "--verify", "--quiet", f"{head}^{{tree}}") if workdir else None
    reviewed_tree = _sha(tree.stdout.decode()) if tree is not None and tree.returncode == 0 else None
    reviewed_base = _sha(bound[0]) if bound is not None else None
    if reviewed_base is None or reviewed_tree is None:
        logger.warning("kanban delivery: delivery %s: the reviewed base or tree of %s is unreadable", row["id"], head)
        return None
    return release_intake.record_merged_change(
        repository=repository, number=number, merge_commit=merge, reviewed_base=reviewed_base,
        reviewed_head=head, reviewed_tree=reviewed_tree, tier=task["risk_tier"], card_id=source,
        title=task["title"], board=_board_slug(db_path))["recorded"]


# A new card branch's start (kanban.delivery.exact_base). When the fetch of main fails, the fixed reason, the
# commit HEAD holds, the worktree path and the branch go on disk before the worktree exists, in one small file
# per worktree path in the repository's git common directory: outside every working tree and no ref, so the
# bound checkout's owner sees nothing, and there for a fresh dispatcher until record_worktree_base has
# committed that worktree's base and, in the same transaction, why.
_REASON = re.compile(r"[a-z_]{1,64}")  # a transport reason code, never an error's text
_BASE_FALLBACK_DIR = "hermes-kanban-base-fallback"  # the pending causes' directory in a git common directory
_PENDING_BOUND = 65536  # bytes: a reason code, a commit, a worktree path and a branch name
_PENDING_FIELDS = {"reason", "base_commit", "target", "branch"}


def _worktree_key(path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _pending_path(checkout: Path, key: str) -> Optional[Path]:
    """The pending cause's file for the worktree path ``key`` in the git common directory of the repository
    ``checkout`` belongs to; None when git cannot tell (it could not run or timed out, exited nonzero, or did not
    answer exactly one absolute path): where a cause would wait is then unknown, never a sign that none does."""
    found = _run_git(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
    lines = os.fsdecode(found.stdout).splitlines() if found is not None and found.returncode == 0 else []
    if len(lines) != 1 or not os.path.isabs(lines[0]):
        return None
    return Path(lines[0]).resolve(strict=False) / _BASE_FALLBACK_DIR / hashlib.sha256(os.fsencode(key)).hexdigest()


def _pending_cause(path: Path, worktree: Path) -> Optional[dict]:
    """What ``path`` holds for the worktree at ``worktree``, read with a hard size bound: None when nothing is
    pending for it (no file, or one naming another worktree path or another branch, an earlier preparation's);
    the cause when it is exactly what :func:`worktree_start` writes; ``{}`` for anything else, a prepared
    fallback that cannot be verified."""
    try:
        if path.is_symlink() or path.exists() and not path.is_file():
            return {}
        with open(path, "rb") as file:
            data = file.read(_PENDING_BOUND + 1)
        cause = json.loads(data) if len(data) <= _PENDING_BOUND else None
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError):
        return {}
    if not isinstance(cause, dict) or set(cause) != _PENDING_FIELDS or not all(isinstance(v, str) for v in cause.values()):
        return {}
    if cause["target"] != _worktree_key(worktree):
        return None
    found = _run_git(worktree, "symbolic-ref", "--quiet", "HEAD")
    branch = os.fsdecode(found.stdout).strip() if found is not None and found.returncode == 0 else None
    if branch is not None and branch != f"refs/heads/{cause['branch']}":
        return None
    if branch is None or not (_SHA.fullmatch(cause["base_commit"]) and _REASON.fullmatch(cause["reason"])):
        return {}
    return cause


def worktree_start(repo_root, target, branch) -> str:
    """Where the kernel starts the NEW card branch ``branch`` for the worktree at ``target``: ``HEAD`` as always
    or, with ``kanban.delivery.exact_base`` exactly true and ``repo_root``'s origin a repository of the
    delivery policy, the commit main holds on GitHub now, fetched through the publish step's transport
    into ``refs/remotes/origin/main`` only. When that fetch fails the branch starts at the commit HEAD holds,
    returned only once the transport's fixed reason waits on disk for :func:`base_fallback`, so no fallback
    worktree exists without its cause; should either fail, this raises and no worktree is made. An earlier
    preparation's cause for ``target`` is cleared first. Opens no database."""
    checkout, key = Path(repo_root), _worktree_key(target)
    path = _pending_path(checkout, key)
    if path is not None:
        try:
            path.unlink()
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError:
            raise RuntimeError("kanban delivery: an earlier fallback cause of this worktree could not be cleared") from None
    if not exact_base_settings():
        return "HEAD"
    repository = _origin_repository(checkout)
    if repository is None or repository not in _policy_repositories():
        return "HEAD"
    reason = "fetch_failed"
    try:
        from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

        try:
            return GitHubTransport("publish", repository).fetch_main(repo_root)
        except GitHubTransportError as error:
            if _REASON.fullmatch(str(error.reason)):
                reason = error.reason
    except Exception:  # any other failure starts at HEAD too, and its text is never kept
        pass
    head = _run_git(checkout, "rev-parse", "--verify", "HEAD^{commit}")
    commit = _sha(os.fsdecode(head.stdout)) if head is not None and head.returncode == 0 else None
    cause = json.dumps({"reason": reason, "base_commit": commit, "target": key, "branch": branch})
    kept = commit is not None and path is not None and len(cause) <= _PENDING_BOUND
    if kept:
        try:
            path.parent.mkdir(exist_ok=True)
            kb._atomic_write_text(path, cause)
        except OSError:
            kept = False
    if not kept:
        raise RuntimeError("kanban delivery: the fetch of main failed and its cause could not be kept; no worktree was made")
    logger.warning("kanban delivery: the fetch of main failed (%s); the new card branch starts at HEAD", reason)
    return commit


@contextlib.contextmanager
def base_fallback(workspace_path):
    """Entered by ``record_worktree_base`` just before its write transaction, in the same ``with``, for the
    worktree at ``workspace_path``; yields the recorder it runs right after its UPDATE has set that base, in
    that transaction and on its connection, opening none. Whether a cause is pending for that worktree is found
    here, before the transaction, with one of three outcomes: git names where it would be and none is there for
    this worktree, so the base goes alone; one is there, and when it is this branch's and names the commit
    recorded the recorder appends that commit and the fixed reason, while one that cannot be verified makes the
    recorder raise; or git cannot tell where it would be, which is no proof that none is, so the recorder raises
    one fixed error that holds no git output, path or error text. Either refusal rolls the base back, so no
    worker starts and that prepared fallback never becomes ordinary work. The cause is retired only once that
    transaction has committed, and only from where git named it: a rollback, a failed commit or an unknown
    location keeps it for the retry, in this process or a fresh one."""
    workspace = Path(workspace_path)
    path = _pending_path(workspace, _worktree_key(workspace))
    pending = _pending_cause(path, workspace) if path is not None else None

    def record(conn: sqlite3.Connection, task_id: str, base_commit: str) -> None:
        if path is None:  # unknown, not absent: a cause may wait where git could not look
            raise kb.WorktreeScopeError("whether a fallback cause is pending for this worktree cannot be established")
        if pending is None:
            return
        if pending.get("base_commit") != base_commit:
            raise kb.WorktreeScopeError("the pending fallback cause of this worktree cannot be verified")
        kb._append_event(conn, task_id, "worktree_base_fallback", {"base_commit": base_commit, "reason": pending["reason"]})

    yield record
    if pending is not None:  # reached only on a clean exit, so after write_txn's commit
        with contextlib.suppress(OSError):
            path.unlink()
