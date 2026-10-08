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
names the failed tests and reruns the failed jobs once if ``decide_rerun``
allows it, or else returns the work once to its builder through one continuation
card, and a later pass reads CI again. :func:`arm_step` then arms GitHub
auto-merge once on each head whose required review cards are done and approve it,
returns the work of each head whose review requests changes through that same card,
and closes once each pull request a rework card replaced; GitHub does the merge.
:func:`merge_step` reads each armed pull request, and once GitHub shows it merged
into main adds the merge to the release decision through
:mod:`hermes_cli.release_intake` and marks the row merged, its final state.
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
from hermes_cli.kanban_delivery_fences import (
    _PASSING, TEST_JOB_STEPS, Decision, _required_check, decide_publish, decide_rerun, load_policy,
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
        # older or newer, is returned for changes; a head GitHub may hold armed, or merged, stays as it is.
        conn.execute(
            "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
            "WHERE source_task_id = ? AND source_head != ? AND IFNULL(pull_request_state, '') NOT IN (?, ?, ?, ?)",
            (task_id, head, *_ARM_HELD, _MERGED),
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
    merge_step(db_path)
    arm_step(db_path)
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

# Fixed kernel text: the title names the source card by its own title, and the body carries the ids.
_REVIEW_TITLE = "Review ({lens}) of {title}"
_REVIEW_BODY = (
    "Kernel review of a published delivery: review pull request {number} on the code host at the "
    "head below, not a local worktree, and post the review on that commit.\nReview: {lens}\n"
    "Source card: {source}\nHead: {head}\nBase commit: {base}\nRepository: {repository}\n"
    "Pull request: {number}\nBranch: {branch}\nRisk tier: {tier}\nCI on {head}: passed ({checks})\n"
)
# Added once the card has its id: the line that ties the reviewer's GitHub review to this one card.
_REVIEW_LINE = "Start your GitHub review body with this exact line:\nReview card {card}\n"


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
    as that request answered them, before the red check is rerun or returned. Owner rule 2
    of round 3: a recorded refused or unanswered rerun of H with no return recorded for H
    returns the work before CI on H is read, whatever it shows. Returns the state or the
    red outcome stored, if any."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        read = _source_read(conn, source)
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (source,)).fetchone()
        bound = [event[0] for event in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_bound' "
            "ORDER BY id DESC", (source,))]
        refused = _refused_rerun(conn, source, head)
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
        if refused is not None:  # owner rule 2 of round 3: green, pending or red, CI on H does not change it
            return _rerun_refused(db_path, row, read, github, repository, *refused)
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
        with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
            if not delivery_settings() or not _unchanged(conn, row, read):
                return None
            _waiting(conn, row, "red_check_waiting", check_runs=failed)
            read = _source_read(conn, source)  # the record just written moved the source card's revision
        return _rerun_or_return(db_path, row, read, github, repository, policy, {run["id"] for run in failed})
    except GitHubTransportError as error:
        raise _transport_refusal(error) from None


# The one title tests.yml gives every annotation a failed slice job writes on its own check run: first
# the count of failed tests ("count N"), then each failed test id. No other annotation is read. Owner
# rule 4: "count" and a number, 0 included (a setup error's), is the count and never a test id.
_FAILED_TEST, _FAILED_COUNT = "Failed test", re.compile(r"count ([0-9]+)")
_RED_REWORK_BODY = (
    "Rework of {source}: a required check is red on head {head} of pull request {number} on the code host, "
    "and the delivery returned the work to its builder ({reason}: {detail}). {tests} "
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


def _failed_jobs(github, repository: str, head: str, red: set) -> Optional[list]:
    """Every latest job of H's workflow run that holds a red required check run (an Actions job is its
    check run), each failed slice job with the test ids and the count its own check run's annotations of
    the fixed title name; None when no run of H on the one page read holds one, or while that run runs."""
    runs, _ = _page(github, f"/repos/{repository}/actions/runs", "workflow_runs", head_sha=head)
    for run in runs:
        jobs, count = _page(github, f"/repos/{repository}/actions/runs/{run['id']}/jobs", "jobs", filter="latest")
        if not red & {job["id"] for job in jobs}:
            continue
        if count != len(jobs):  # every job of the run, whole on its one page
            raise PublishRefused("ci_unreadable", f"{count} jobs in run {run['id']}, {len(jobs)} in the answer")
        if any(job.get("status") != "completed" for job in jobs):
            return None  # the run still runs (ci.yaml's timing report runs after the summary check): decide later
        for job in jobs:
            if job.get("name") in TEST_JOB_STEPS and job.get("conclusion") not in _PASSING:
                answer = _answered(github.request(
                    "GET", f"/repos/{repository}/check-runs/{job['id']}/annotations", query={"per_page": 100}))
                if answer["status"] != 200 or not isinstance(answer["data"], list):
                    raise PublishRefused("ci_unreadable", f"GitHub answered {answer['status']} for job {job['id']}")
                notes = [note["message"] for note in answer["data"]
                         if note["title"] == _FAILED_TEST and isinstance(note["message"], str)]
                counts = [int(found[1]) for found in map(_FAILED_COUNT.fullmatch, notes) if found]
                job.update(failed_tests=[note for note in notes if not _FAILED_COUNT.fullmatch(note)],
                           failed_count=counts[0] if len(counts) == 1 else None)
        return jobs
    return None


def _rerun_or_return(db_path: Path, row: dict, read: tuple, github, repository: str, policy, red: set):
    """A red required check on H: ``decide_rerun``, given the latest jobs of H's run and the failed tests
    each failed slice names, either reruns the failed jobs once, recorded as red_check_rerun before the
    rerun calls are sent and as red_check_rerun_outcome with their answers after them, or returns the
    work: red_check_returned and one continuation card for the builder, once for H, also when a rerun call was
    refused (rerun_refused) or got no answer (rerun_unknown). Nothing is recorded or sent unless the pull
    request, read after H's run, jobs and annotations, is still open at H. Until GitHub shows a newer
    attempt than the rerun whose recorded outcome shows every call accepted, the pass waits; after a
    refused or unanswered one, the pass reads the pull request again (owner rule 4). Returns the outcome
    recorded, if any."""
    from hermes_cli.kanban_delivery_github import GitHubTransport

    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        reruns, returned = _recorded(conn, source, "red_check_rerun"), _recorded(conn, source, "red_check_returned")
        answered = [record for record in _recorded(conn, source, "red_check_rerun_outcome") if record["head"] == head]
    if any(record["head"] == head for record in returned):
        return None  # H's work went back to its builder already
    jobs = _failed_jobs(github, repository, head, red)
    if jobs is None:
        return None
    attempt = max((job["run_attempt"] for job in jobs if type(job.get("run_attempt")) is int), default=0)
    if answered and any(record["head"] == head and attempt <= record["attempt"] for record in reruns):
        return None  # H's rerun was accepted and GitHub has not started it yet
    decision = decide_rerun(policy, repository, head, jobs, {"reruns": [record["head"] for record in reruns]})
    if decision.allowed and decision.code != "rerun":
        return None
    branch = None if decision.allowed else _return_branch(db_path, row)  # owner rule 2: the branch comes first
    if not _open_at_head(github, repository, row):
        return None  # owner rule 1: the pull request left H or closed while H's run was read
    code = "red_check_rerun" if decision.allowed else "red_check_returned"
    tests = [test for job in jobs for test in job.get("failed_tests") or () if isinstance(test, str)]
    ids = sorted({match["job_id"] for match in decision.matches})
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read) or any(
                record["head"] == head for record in _recorded(conn, source, code)):
            return None
        if decision.allowed:
            _waiting(conn, row, code, jobs=ids, tests=tests, attempt=attempt)
        else:
            _waiting(conn, row, code, reason=decision.code, tests=tests,
                     **_continuation(conn, db_path, row, _red_body(row, decision, tests), branch))
    if not decision.allowed:
        return code
    rerun = GitHubTransport("rerun_flaky", repository)  # sent once: only by the pass that recorded the rerun
    statuses = [_sent(rerun, "request", "POST", f"/repos/{repository}/actions/jobs/{job}/rerun")["status"]
                for job in ids]
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        _waiting(conn, row, "red_check_rerun_outcome", jobs=ids, statuses=statuses)
        read = _source_read(conn, source)  # the record just written moved the source card's revision
    if not _rerun_reason(statuses):
        return code
    return _rerun_refused(db_path, row, read, github, repository, {"jobs": ids, "statuses": statuses}, tests) or code


def _rerun_reason(statuses: list) -> Optional[str]:
    """Why a rerun's calls did not all start a newer attempt: a call refused, or one with no answer."""
    return "rerun_refused" if set(statuses) - {201, None} else "rerun_unknown" if None in statuses else None


def _refused_rerun(conn: sqlite3.Connection, source: str, head: str) -> Optional[tuple]:
    """H's recorded rerun outcome and the failed tests its rerun record names, when a rerun call was refused or
    got no answer and no red_check_returned is recorded for H; else None."""
    answered = [record for record in _recorded(conn, source, "red_check_rerun_outcome") if record["head"] == head]
    if not answered or not _rerun_reason(answered[0]["statuses"]) or any(
            record["head"] == head for record in _recorded(conn, source, "red_check_returned")):
        return None
    return answered[0], next(record["tests"] for record in _recorded(conn, source, "red_check_rerun")
                             if record["head"] == head)


def _rerun_refused(db_path: Path, row: dict, read: tuple, github, repository: str, sent: dict, tests: list):
    """Owner rule 4: H's rerun outcome ``sent`` shows a refused or unanswered call, so the pull request is read
    again and, open at H, red_check_returned is recorded once for H with H's continuation. A failed read
    raises and only ends the pass; a later pass reads it again. Returns the outcome recorded, if any."""
    source, head = row["source_task_id"], row["pull_request_head"]
    branch = _return_branch(db_path, row)
    if not _open_at_head(github, repository, row):
        return None  # safety rule 1: the pull request left H or closed
    reason = _rerun_reason(sent["statuses"])
    refusal = Decision(False, reason, f"the rerun calls of jobs {sent['jobs']} were answered {sent['statuses']} "
                                      "(None: no answer)")
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or not _unchanged(conn, row, read) or any(
                record["head"] == head for record in _recorded(conn, source, "red_check_returned")):
            return None
        _waiting(conn, row, "red_check_returned", reason=reason, tests=tests,
                 **_continuation(conn, db_path, row, _red_body(row, refusal, tests), branch))
    return "red_check_returned"


def _red_body(row: dict, decision, tests: list) -> str:
    """A red check continuation's body: H, the pull request, the reason and the failed tests as CI named them."""
    named = f"CI named these failed tests: {', '.join(tests)}." if tests else "No failed test was named."
    return _RED_REWORK_BODY.format(source=row["source_task_id"], head=row["pull_request_head"],
                                   number=row["pull_request_number"], reason=decision.code,
                                   detail=decision.detail, tests=named)


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


def _continuation(conn: sqlite3.Connection, db_path: Path, row: dict, body: str, branch: Optional[tuple]) -> dict:
    """H's one continuation card, for a review or a red check return: a task_links child of the source for
    its builder, with its owned paths, tiers, route and review requirement, titled through owner_title and
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
    governed = conn.execute("SELECT execution_tier, model_policy_lock, owner_receipt_bound FROM tasks WHERE id = ?",
                            (source,)).fetchone()
    implementer = kb._latest_review_provenance(conn, source)[0]
    root, name = branch
    found = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    if found is None or found.returncode != 0 or _sha(found.stdout.decode()) != head:
        raise PublishRefused("branch_not_at_head", f"{name} is not at {head} in the write transaction")
    lock = parked = None
    # Owner rule 1: the source card's own route and tier, locked for the builder; unadmitted, the card parks.
    # Owner rule 3 of round 3: its owner_receipt_bound too. When a governed source's route cannot be locked, the
    # card is created parked with the reason and stays governed, never a manual card: it keeps the source's
    # provider, which create_task refuses with no model, and, governed by nothing else, the source's lock, which
    # binds no route with no tier.
    try:
        lock = kb.mint_policy_lock(implementer, task.provider_override, task.model_override,
                                   task.reasoning_effort, task.execution_tier)
    except ValueError as error:
        parked = str(error) if kb.task_is_policy_governed(governed) else None
    item = kb.create_task(
        conn, title=owner_title(task.title), body=body, assignee=implementer, parents=[source], tenant=task.tenant,
        workspace_kind="worktree", project_id=task.project_id, branch_name=name, owned_paths=task.owned_paths,
        risk_tier=task.risk_tier, execution_tier=task.execution_tier, requires_review=task.requires_review,
        provider_override=None if parked else task.provider_override, model_override=task.model_override,
        reasoning_effort=task.reasoning_effort, model_policy_lock=lock,
        receipt_owned=bool(governed["owner_receipt_bound"]), board=_board_slug(db_path))
    conn.execute("UPDATE tasks SET base_commit = ? WHERE id = ?", (head, item))  # the claim keeps it
    if Path(kb.get_task(conn, item).workspace_path or "") != root / ".worktrees" / item:
        raise PublishRefused("no_repository", f"the project of {source} no longer anchors its cards in {root}")
    if parked:
        conn.execute("UPDATE tasks SET provider_override = ?, model_policy_lock = CASE WHEN execution_tier IS NULL "
                     "AND NOT owner_receipt_bound THEN ? END WHERE id = ?",
                     (task.provider_override, governed["model_policy_lock"], item))
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
        lenses = _COMBINED if tier < 2 else _SPLIT
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
                conn, title=title, body=_REVIEW_BODY.format(lens=lens, **text), assignee=reviewer,
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
# Card 5: auto-merge armed on a fully approved head; a replaced pull request closed
# ---------------------------------------------------------------------------

# Each mutation is first recorded as pending, in the write transaction that reads its
# evidence a last time, and only the pass that wrote that sends it; its answer is then
# recorded once. A head's arm ends armed, refused (with GitHub's status) or unknown (no
# answer), and is never sent again: a new head is a new row. GitHub may hold a pending,
# armed or unknown arm, so no new head is pushed to that pull request. A close ends
# replaced or close_refused (with GitHub's status), its delivery_close_pending event
# the record that it was sent.
_PENDING, _ARMED, _ARM_REFUSED, _ARM_UNKNOWN = (
    "auto_merge_pending", "auto_merge_armed", "auto_merge_refused", "auto_merge_unknown")
_ARM_HELD = (_PENDING, _ARMED, _ARM_UNKNOWN)
_REPLACED, _CLOSE_REFUSED = "replaced", "close_refused"
_REVIEW_PAGES = 10  # owner rule 2 of round 3: a pull request's reviews are read up to 10 pages of 100
# A row's repository: the publish step's own record of the pull request it published,
# never the current git origin. No record, no repository, and nothing is sent.
_RECEIPT = ("(SELECT json_extract(payload, '$.repository') FROM task_events WHERE task_id = {0}.source_task_id "
            "AND kind = 'delivery_published' AND json_extract(payload, '$.head') = {0}.pull_request_head "
            "AND json_extract(payload, '$.pull_request_number') = {0}.pull_request_number ORDER BY id DESC LIMIT 1)")


def arm_step(db_path: Path) -> None:
    """Card 5 on the board whose file is ``db_path``: arm GitHub auto-merge once on each
    head whose required review cards are done and approve it, and close once the pull
    request of each head a rework card continues. GitHub does the merge. Raises no error,
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
    its recorded base is the row's head and its own published pull request is another one of
    the same repository. The row is still the latest on its pull request, and no close of it
    was sent; a merged row is final (card 10's owner rule 2)."""
    old, new, later = (_RECEIPT.format(name) for name in ("old", "new", "later"))
    return [dict(row) for row in conn.execute(
        f"SELECT old.*, {old} AS repository, MIN(rework.id) AS replaced_by FROM kanban_deliveries AS old "
        "JOIN tasks AS rework ON rework.base_commit = old.pull_request_head AND rework.id != old.source_task_id "
        "JOIN task_links AS link ON link.parent_id = old.source_task_id AND link.child_id = rework.id "
        "JOIN kanban_deliveries AS new ON new.source_task_id = rework.id AND new.pull_request_number IS NOT NULL "
        f"AND new.pull_request_number != old.pull_request_number AND {new} = {old} "
        "WHERE old.pull_request_number IS NOT NULL AND (? IS NULL OR old.id = ?) AND old.pull_request_state IS NOT ? "
        "AND NOT EXISTS (SELECT 1 FROM kanban_deliveries AS later WHERE later.id > old.id "
        f"AND later.pull_request_number = old.pull_request_number AND {later} = {old}) "
        "AND NOT EXISTS (SELECT 1 FROM task_events WHERE task_id = old.source_task_id "
        "AND kind = 'delivery_close_pending' AND json_extract(payload, '$.delivery_id') = old.id) "
        "GROUP BY old.id ORDER BY old.id", (old_id, old_id, _MERGED))]


def _approving_cards(conn: sqlite3.Connection, row: dict, read: tuple, returned: bool = False) -> list:
    """The review cards the source card's tier requires for the row's head, when the source
    is done at that head and each of them exists, is done and returned no changes; else [].
    Each is its id, key, creator, status, own head and base, and event revision. With
    ``returned``, a card that returned changes counts too."""
    source, head = row["source_task_id"], row["pull_request_head"]
    if (read[1], _sha(read[2])) != ("done", head):
        return []
    try:
        tier = effective_risk_tier(read[3])
    except ValueError:  # not a tier: tier 2, never a lower one
        tier = UNRECORDED_RISK_TIER
    cards = [conn.execute("SELECT id, idempotency_key, created_by, status, head_commit, base_commit FROM tasks "
                          "WHERE idempotency_key = ? AND created_by = ?",
                          (review_key(source, head, responsibility), _REVIEW_CREATOR)).fetchone()
             for responsibility in (_COMBINED if tier < 2 else _SPLIT)]
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
    """Arm auto-merge on the row's head H, once (owner rules 1 to 3 of round 2). Every GitHub
    read comes first: the pull request, open at H, then its reviews, which approve H lens by
    lens. One write transaction then reads the complete evidence again, finds the pull request
    unclaimed and reserves the arm under this attempt. The arm is the one call after it, and
    its answer is written to this attempt only. A full tenth page of reviews that GitHub does not
    show to be the last refuses H once instead, and a review requesting changes returns H (:func:`_return_review`)."""
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
    attempt = secrets.token_hex(8)
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not delivery_settings() or _arm_evidence(conn, row) != evidence or _claimed(conn, repository, number):
            return None  # the evidence changed while GitHub was read: nothing is sent
        if not whole:  # a full tenth page, a next page named or unknown: H is refused once and never sent
            conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (_ARM_REFUSED, row["id"]))
            kb._append_event(conn, source, f"delivery_{_ARM_REFUSED}", _outcome(
                row, repository=repository, status=None, reason="review_history_over_10_pages",
                cards=[card[0] for card in evidence[2]]))
            return _ARM_REFUSED
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = ? WHERE id = ?",
                     (_PENDING, attempt, row["id"]))
    answer = _sent(github, "arm_auto_merge", number, head)
    data = answer["data"] if isinstance(answer["data"], dict) else {}
    result = data.get("data") if isinstance(data.get("data"), dict) else {}
    state = _ARM_UNKNOWN if answer["status"] is None else _ARMED if answer["status"] == 200 and (
        "errors" not in data and isinstance(result.get("enablePullRequestAutoMerge"), dict)) else _ARM_REFUSED
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if conn.execute("UPDATE kanban_deliveries SET pull_request_state = ?, publish_lease = NULL "
                        "WHERE id = ? AND publish_lease = ?", (state, row["id"], attempt)).rowcount:
            kb._append_event(conn, source, f"delivery_{state}", _outcome(
                row, repository=repository, status=answer["status"], reason=answer.get("reason"),
                cards=[card[0] for card in evidence[2]], attempt=attempt))
    return state


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


def merge_step(db_path: Path) -> None:
    """Card 10 on the board whose file is ``db_path``: read the pull request of each armed or
    unknown arm and, once GitHub shows it merged into main, add the merge to the waiting
    release decision and mark the row merged. A failure leaves the row as it was, so the next
    pass reads it again. Raises no error, so the tick goes on."""
    try:
        with kb.connect_closing(db_path=db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT d.*, {_RECEIPT.format('d')} AS repository FROM kanban_deliveries AS d "
                "WHERE d.pull_request_number IS NOT NULL AND d.pull_request_state IN (?, ?) ORDER BY d.id",
                (_ARMED, _ARM_UNKNOWN))]
    except Exception:
        logger.exception("kanban delivery: the merge step failed")
        return
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
    row could not be marked is still recorded once, however often it is seen."""
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
    if merge is None:
        return None  # not merged into main: a later pass reads it again
    attempt = secrets.token_hex(8)
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if _claimed(conn, repository, number) or not conn.execute(
                "UPDATE kanban_deliveries SET publish_lease = ? WHERE id = ? AND pull_request_state = ?",
                (attempt, row["id"], row["pull_request_state"])).rowcount:
            return None  # another action took the pull request, or the row moved on, while GitHub was read
    return _merged(db_path, row, merge, attempt, row["pull_request_state"])


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
