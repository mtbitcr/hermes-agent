"""Card 3 (kanban.delivery.exact_base): a delivery row the arm left base_moved, brought up to date on main.

Main moved past an approved head H, so card 2's arm sent nothing and recorded base_moved with the main it read.
That approval covers H on its reviewed base only; it never merges anything else. Here the update is computed
without touching any checkout: ``git merge-tree --write-tree`` of main and H, and ``git commit-tree`` of that
tree with H and main as its two parents. Before anything is created, the update must change the same set of
paths relative to main as H did relative to its reviewed base, and nothing outside the source card's owned
paths. Then one update card is made per source card, H and main: a task_links child of the source card for its
builder, its branch at the updated head, its recorded base main's commit, handed over to review on the kernel's
own record, so it is reviewed, approved and merged as any card is. The row is then final, ``updated``, and
:func:`hermes_cli.kanban_delivery._replaced` closes its pull request once the update is published. A conflict or
another set of paths goes back to the builder through H's continuation card instead, and no card goes to review.
:func:`hermes_cli.kanban_delivery.publish_step` calls :func:`update_step` once per pass, with the setting on only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_delivery as kd

logger = logging.getLogger(__name__)

# One update card per source card, old head and main commit; kanban_delivery._replaced reads the same prefix.
_IDENTITY = "update:{source}:{head}:{main}"
_BRANCH = "delivery-update/{source}-{head}-{main}"
_AUTHOR = ("-c", f"user.name={kd._REVIEW_CREATOR}", "-c", "user.email=kanban-delivery@localhost")
_MESSAGE = ("{head} of {source} brought up to date on main {main}\n\nThe change approved at {head}, reviewed on "
            "{base}, merged with main at {main}: git merge-tree --write-tree of main and the approved head.\n")
_MOVED = ("Main moved past the approved head {head} of {source} (pull request #{number} of {repository}), "
          "reviewed on {base}: main is at {main}.")
_UPDATE_BODY = _MOVED + (
    " This card carries that change brought up to date on main as {update}, on branch {branch}: git merge-tree "
    "--write-tree of main and the approved head, committed with the approved head and main as its two parents. "
    "It changes the same paths relative to main as the approved change did relative to its base, all within the "
    "source card's owned paths. The approval of {head} does not cover it: only this card's own review delivers it.")
_RETURN_BODY = _MOVED + " {why} Bring the change up to date on main and hand it over for review again."
_WHY = {
    "conflict": "Merging main and the approved head conflicts, so no update was made.",
    "paths": ("The update on main would change another set of paths relative to main than the approved change "
              "did relative to its base, or paths outside the source card's owned paths, so no update was made."),
}


def update_step(db_path: Path) -> None:
    """Card 3 on the board whose file is ``db_path``: each row the arm left base_moved gets its one update card
    on the main the arm read, or goes back to its builder. Only with ``kanban.delivery.exact_base`` exactly true;
    otherwise a base_moved row stays where it is. Raises no error, so the tick goes on."""
    if not kd.delivery_settings() or not kd.exact_base_settings():
        return
    try:
        with kb.connect_closing(db_path=db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT d.*, {kd._RECEIPT.format('d')} AS repository FROM kanban_deliveries AS d "
                "WHERE d.pull_request_number IS NOT NULL AND d.pull_request_state = ? ORDER BY d.id",
                (kd._BASE_MOVED,))]
    except Exception:
        logger.exception("kanban delivery: the update step failed")
        return
    for row in rows:
        try:
            _update(db_path, row)
        except kd.PublishRefused as refusal:
            logger.warning("kanban delivery: delivery %s not updated: %s (%s)", row["id"], refusal.code,
                           refusal.detail)
        except Exception:
            logger.exception("kanban delivery: delivery %s not updated", row["id"])


def _update(db_path: Path, row: dict) -> Optional[str]:
    """One base_moved row: the update of H on the main its base_moved event records, checked before anything is
    created, then the update card, or H back to its builder. Returns what was done, if anything."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        moved = _moved(conn, row)
        task = kb.get_task(conn, source)
    main, base = (kd._sha(moved.get(key)) for key in ("observed_main", "reviewed_base"))
    if kd._sha(head) != head or main is None or base is None or moved.get("repository") != row["repository"]:
        raise kd.PublishRefused("no_main", f"delivery {row['id']} records no main of its repository it moved to")
    root = _folder(task.project_id, row["repository"])
    if root is None or not all(kd._git_proof(root, commit, None)[0] for commit in (head, base, main)):
        raise kd.PublishRefused("no_repository", f"the project folder of {source} does not hold {head}, {base} "
                                f"and {main} of {row['repository']}")
    facts = {"head": head, "base": base, "main": main, "source": source}
    key = _IDENTITY.format(**facts)
    with kb.connect_closing(db_path=db_path) as conn:
        made = conn.execute("SELECT id FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()
    if made is not None:  # made already: nothing is computed or made again
        return _updated(db_path, row, moved, key, None)
    merged = kd._run_git(root, "merge-tree", "--write-tree", main, head)
    if merged is None or merged.returncode not in (0, 1):
        raise kd.PublishRefused("no_update", f"git merge-tree of {main} and {head} did not run")
    if merged.returncode == 1:
        return _returned(db_path, row, moved, "conflict")
    tree = kd._sha(merged.stdout.decode().split("\n", 1)[0])
    approved, update = _paths(root, base, head), _paths(root, main, tree)
    if tree is None or approved is None or update is None:
        raise kd.PublishRefused("no_update", f"the paths of {head} on {main} cannot be read")
    if update != approved or not all(_owned(path, task.owned_paths) for path in update):
        return _returned(db_path, row, moved, "paths")
    branch = _BRANCH.format(source=source, head=head[:12], main=main[:12])
    commit = _branch(root, branch, tree, head, main, _MESSAGE.format(**facts))
    return _updated(db_path, row, moved, key, (root, branch, commit))


def _moved(conn, row: dict) -> dict:
    """The row's latest delivery_base_moved event at its head: the main the arm read and the reviewed base."""
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id = ? "
                                   "AND kind = 'delivery_base_moved' ORDER BY id DESC", (row["source_task_id"],)):
        moved = json.loads(payload)
        if moved.get("delivery_id") == row["id"] and moved.get("head") == row["pull_request_head"]:
            return moved
    return {}


def _folder(project: Optional[str], repository: Optional[str]) -> Optional[Path]:
    """The primary folder the project registry records for the source card's project, as the continuation's,
    when its origin names the repository the delivery recorded."""
    from hermes_cli import projects_db

    folder = None
    if project:
        with contextlib.suppress(Exception), projects_db.connect_closing() as registry:  # kb.create_task's registry
            found = projects_db.get_project(registry, project)
            folder = found.primary_path if found is not None else None
    root = Path(folder) if folder and os.path.isabs(folder) else None
    if root is None or not repository or kd._origin_repository(root) != repository:
        return None
    return root


def _paths(root: Path, old: str, new: str) -> Optional[frozenset]:
    """The paths git changes from ``old`` to ``new``, each tree-ish, renames as two paths."""
    changed = kd._run_git(root, "diff-tree", "-r", "--name-only", "--no-renames", "-z", old, new)
    if changed is None or changed.returncode != 0:
        return None
    return frozenset(path for path in changed.stdout.decode("utf-8", "surrogateescape").split("\0") if path)


def _owned(path: str, owned: Optional[list]) -> bool:
    """The kernel's owned path rule: "." owns the repository, every other entry one exact file or subtree.
    An unscoped card (None) or a read-only one owns nothing that can be shown."""
    return bool(owned) and ("." in owned or any(path == owner or path.startswith(f"{owner}/") for owner in owned))


def _branch(root: Path, name: str, tree: str, head: str, main: str, message: str) -> str:
    """The updated head on ``name``: the commit of ``tree`` with H and main as its parents, made before the write
    transaction, or the one a pass stopped before its commit left on the name. No checkout is touched."""
    found = kd._run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    if found is None or found.returncode != 0:
        made = kd._run_git(root, *_AUTHOR, "commit-tree", tree, "-p", head, "-p", main, "-m", message)
        commit = kd._sha(made.stdout.decode()) if made is not None and made.returncode == 0 else None
        if commit is None:
            raise kd.PublishRefused("no_update", f"git commit-tree of {tree} did not run")
        kd._run_git(root, "branch", name, commit)  # git refuses a name that exists; the read below decides
        found = kd._run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    commit = kd._sha(found.stdout.decode()) if found is not None and found.returncode == 0 else None
    if commit is None or _commit(root, commit) != (tree, [head, main]):
        raise kd.PublishRefused("branch_not_at_update", f"{name} is not the update of {head} on {main}")
    return commit


def _commit(root: Path, commit: str) -> tuple:
    """A commit's tree and parents."""
    tree = kd._run_git(root, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}")
    parents = kd._run_git(root, "rev-list", "--parents", "-n", "1", commit)
    if tree is None or parents is None or tree.returncode != 0 or parents.returncode != 0:
        return None, None
    return kd._sha(tree.stdout.decode()), parents.stdout.decode().split()[1:]


def _still_moved(conn, row: dict) -> bool:
    """The row is still base_moved at its head and pull request, with the setting still on."""
    return kd.delivery_settings() and kd.exact_base_settings() and conn.execute(
        "SELECT 1 FROM kanban_deliveries WHERE id = ? AND pull_request_state = ? AND pull_request_head = ? "
        "AND pull_request_number = ?", (row["id"], kd._BASE_MOVED, row["pull_request_head"],
                                        row["pull_request_number"])).fetchone() is not None


def _updated(db_path: Path, row: dict, moved: dict, key: str, update: Optional[tuple]) -> Optional[str]:
    """One write transaction: the update card, unless the identity holds one already, and the row final,
    updated, while it is still base_moved."""
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not _still_moved(conn, row):
            return None
        if update is not None and conn.execute("SELECT 1 FROM tasks WHERE idempotency_key = ?",
                                               (key,)).fetchone() is None:
            item = _card(conn, db_path, row, moved, key, *update)
            kb._append_event(conn, row["source_task_id"], "delivery_updated", kd._outcome(
                row, repository=row["repository"], reviewed_base=moved["reviewed_base"],
                observed_main=moved["observed_main"], update_task_id=item, update_head=update[2],
                update_branch=update[1]))
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (kd._UPDATED, row["id"]))
    return kd._UPDATED


def _card(conn, db_path: Path, row: dict, moved: dict, key: str, root: Path, branch: str, commit: str) -> str:
    """The update card: a task_links child of the source card for its builder, in its project on ``branch``,
    which is read once more here at the updated head, main's commit its recorded base, and handed over to review
    as the kernel's request_review records a handover: the reviewer of the source's handover holds it read-only,
    and the builder, its owned paths and the updated head ride on its review_requested event."""
    from hermes_cli.owner_workspace import owner_title

    source, main = row["source_task_id"], moved["observed_main"]
    found = kd._run_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if found is None or found.returncode != 0 or kd._sha(found.stdout.decode()) != commit:
        raise kd.PublishRefused("branch_not_at_update", f"{branch} is not at {commit} in the write transaction")
    task = kb.get_task(conn, source)
    implementer, reviewer = kb._latest_review_provenance(conn, source)
    body = _UPDATE_BODY.format(head=row["pull_request_head"], source=source, number=row["pull_request_number"],
                               repository=row["repository"], base=moved["reviewed_base"], main=main,
                               update=commit, branch=branch)
    item = kb.create_task(
        conn, title=owner_title(task.title), body=body, assignee=implementer, parents=[source], tenant=task.tenant,
        workspace_kind="worktree", project_id=task.project_id, branch_name=branch, owned_paths=task.owned_paths,
        risk_tier=task.risk_tier, requires_review=task.requires_review, idempotency_key=key,
        board=kd._board_slug(db_path))
    if Path(kb.get_task(conn, item).workspace_path or "") != root / ".worktrees" / item:
        raise kd.PublishRefused("no_repository", f"the project of {source} no longer anchors its cards in {root}")
    conn.execute("UPDATE tasks SET status = 'review', assignee = ?, owned_paths = '[]', integrates_parent_heads = 0, "
                 "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, head_commit = ?, base_commit = ? "
                 "WHERE id = ?", (reviewer, commit, main, item))
    if not kb.authorize_executable_transition(conn, item, park=False):  # the reviewer's route, as it enters review
        raise kd.PublishRefused("route_not_admitted", f"the route of {item} for {reviewer} is not admitted")
    kb._append_event(conn, item, "review_requested", {
        "summary": f"update of {row['pull_request_head'][:12]} on main {main[:12]}", "implementer": implementer,
        "reviewer": reviewer, kb._REVIEW_SCOPE_PATHS_KEY: task.owned_paths,
        kb._REVIEW_SCOPE_INTEGRATES_KEY: False, kb._REVIEW_SCOPE_HEAD_KEY: commit})
    return item


def _returned(db_path: Path, row: dict, moved: dict, reason: str) -> Optional[str]:
    """H back to its builder through its continuation card, on the continuation branch made at H first as card
    2's return does; the row is returned for changes, with one event naming why and the card. No card goes to
    review."""
    branch = kd._return_branch(db_path, row)
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not _still_moved(conn, row):
            return None
        body = _RETURN_BODY.format(head=row["pull_request_head"], source=row["source_task_id"],
                                   number=row["pull_request_number"], repository=row["repository"],
                                   base=moved["reviewed_base"], main=moved["observed_main"], why=_WHY[reason])
        returned = kd._continuation(conn, db_path, row, body, branch)
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' WHERE id = ?",
                     (row["id"],))
        kb._append_event(conn, row["source_task_id"], "delivery_update_returned", kd._outcome(
            row, repository=row["repository"], reason=reason, reviewed_base=moved["reviewed_base"],
            observed_main=moved["observed_main"], **returned))
    return "returned_for_changes"
