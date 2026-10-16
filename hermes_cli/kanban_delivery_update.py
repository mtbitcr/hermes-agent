"""Card 3 (kanban.delivery.exact_base): a delivery row the arm left base_moved, brought up to date on main.

Main moved past an approved head H, so card 2's arm sent nothing and recorded base_moved. That approval covers H
on its reviewed base only; it never merges anything else. Here main is read as GitHub holds it now and its objects
fetched with the fenced fetch worktree_start uses, into refs/remotes/origin/main of the source card's project folder
only, so no checkout moves; a failed fetch leaves the row base_moved, and the next pass reads main again. The
update is then computed without touching any checkout: ``git merge-tree --write-tree`` of that main and H, and
``git commit-tree`` of that tree with H and main as its two parents. Before anything is created, the update must
change the same set of paths relative to main as H did relative to its reviewed base, and nothing outside the source
card's owned paths. When H continues an earlier delivery, its reviewed base the head of a delivery of a task_links
parent of its card or of a card its card replaced, the change is measured from where the chain started, which main
and H must both hold. Then one update card is made per source card, H and main, through the card creation of
:func:`hermes_cli.kanban_delivery._continuation`: a task_links child of the source card for its builder, with the
source's tier, route, effort, policy lock, owner receipt binding, risk tier and review requirement, on a branch at
the updated head. Its builder hands that head over unchanged through the normal completion, so the kernel's own
reviewer nomination and route pins apply, and only its own approval merges it. As that completion hands over only
committed changes since the card's recorded base, the base _continuation records, the updated head, is then moved
to main in the same transaction: the handover is exactly the approved paths on main. The kernel's
``delivery_updated`` event of the source card, written in the transaction that creates the card, is the one record
of it: an existing update card is accepted only through that event, never through a task's caller-chosen fields.
The row is then final, ``updated``, and :func:`hermes_cli.kanban_delivery._replaced` closes its pull request once
the update is published. Another set of paths goes back to the builder through H's continuation card instead. So
does a conflict, unless every conflicted file is inside the approved paths, git's merged tree passes the same checks,
and it holds each of those files with git's conflict markers, or as H's version of a file main deleted: then the
update card is made on that tree the same way, its body names those files, the deleted ones apart, for its builder to
resolve and nothing else, and its delivery_updated event names them as ``conflicts``, the deleted ones also as
``deleted_on_main``.
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
from hermes_cli.kanban_replaced_cards import replaced_task_ids

logger = logging.getLogger(__name__)

_BRANCH = "delivery-update/{source}-{head}-{main}"
_AUTHOR = ("-c", f"user.name={kd._REVIEW_CREATOR}", "-c", "user.email=kanban-delivery@localhost")
_MESSAGE = ("{head} of {source} brought up to date on main {main}\n\nThe change approved at {head}, reviewed on "
            "{base}, merged with main at {main}: git merge-tree --write-tree of main and the approved head.\n")
_MOVED = ("Main moved past the approved head {head} of {source} (pull request #{number} of {repository}), "
          "reviewed on {base}: main is at {main}.")
_UPDATE_BODY = _MOVED + (
    " This card carries that change brought up to date on main as {update}, on branch {branch}: git merge-tree "
    "--write-tree of main and the approved head, committed with the approved head and main as its two parents. Its "
    "recorded base is main, so it changes the same paths relative to main as the approved change did relative to its "
    "base, all within the source card's owned paths. Hand {update} over unchanged for review: the approval of {head} "
    "does not cover it, and only this card's own review delivers it.")
_CONFLICT_BODY = _MOVED + (
    " Merging it with the approved change conflicts in these files: {conflicts}. This card's branch {branch} at "
    "{update} is on current main with the conflict marked. Resolve the marked conflicts in those files only, change "
    "nothing else, and hand over for review.")
_DELETED_BODY = _MOVED + (
    " Merging it with the approved change conflicts in these files, which main deleted and the approved change "
    "modified: {deleted}. This card's branch {branch} at {update} is on current main and holds the approved change's "
    "version of them, with no conflict markers: keep or delete each of them.{marked} Change nothing else, and hand "
    "over for review.")
_ALSO_MARKED = (" The merge also conflicts in these files, which the branch holds with the conflict marked: {marked}. "
                "Resolve the marked conflicts in those files.")
_RETURN_BODY = _MOVED + " {why} Bring the change up to date on main and hand it over for review again."
_WHY = {
    "conflict": "Merging main and the approved head conflicts, so no update was made.",
    "paths": ("The update on main would change another set of paths relative to main than the approved change "
              "did relative to its base, or paths outside the source card's owned paths, so no update was made."),
}


def update_step(db_path: Path) -> None:
    """Card 3 on the board whose file is ``db_path``: each row the arm left base_moved gets its one update card
    on main as GitHub holds it now, or goes back to its builder. Only with ``kanban.delivery.exact_base`` exactly
    true; otherwise a base_moved row stays where it is. Raises no error, so the tick goes on."""
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
    """One base_moved row: main read and fetched now, the update of H on it checked before anything is created,
    then the update card, or H back to its builder. Returns what was done, if anything."""
    source, head = row["source_task_id"], row["pull_request_head"]
    with kb.connect_closing(db_path=db_path) as conn:
        moved = _moved(conn, row)
        task = kb.get_task(conn, source)
    base = kd._sha(moved.get("reviewed_base"))
    if kd._sha(head) != head or base is None or moved.get("repository") != row["repository"]:
        raise kd.PublishRefused("no_main", f"delivery {row['id']} records no reviewed base of its repository")
    root = _folder(task.project_id, row["repository"])
    if root is None:
        raise kd.PublishRefused("no_repository", f"the project folder of {source} is no checkout of "
                                f"{row['repository']}")
    main = _main(root, row["repository"])
    if not all(kd._git_proof(root, commit, None)[0] for commit in (head, base, main)):
        raise kd.PublishRefused("no_repository", f"the project folder of {source} does not hold {head}, {base} "
                                f"and {main} of {row['repository']}")
    facts = {"head": head, "base": base, "main": main, "source": source}
    with kb.connect_closing(db_path=db_path) as conn:
        made = _made(conn, row, main)
    if made is not None:  # the kernel's record of the update card: nothing is computed or made again
        return _updated(db_path, row, facts, None)
    merged = kd._run_git(root, "merge-tree", "--write-tree", "--name-only", "-z", main, head)
    if merged is None or merged.returncode not in (0, 1):
        raise kd.PublishRefused("no_update", f"git merge-tree of {main} and {head} did not run")
    # Exit 1 is a conflict: the conflicted files follow the merged tree, raw and each ended by a NUL, read as _paths
    # reads its names, up to the empty field before git's messages. Its update is made only as a clean one is, with
    # those files inside the approved paths and each held in the merged tree as _held reads it; anything else sends it
    # back as a conflict.
    fields = merged.stdout.decode("utf-8", "surrogateescape").split("\0")
    reason = "conflict" if merged.returncode == 1 else "paths"
    with kb.connect_closing(db_path=db_path) as conn:
        start = _approved_base(conn, source, base)
    if start != base:  # H continues an earlier delivery of the same work: approved from where the chain started
        if start is None or not (kd._git_proof(root, start, None)[0] and kd._git_proof(root, main, start)[1]
                                 and kd._git_proof(root, head, start)[1]):
            return _returned(db_path, row, facts, reason)
        facts["approved_base"] = start
    tree = kd._sha(fields[0])
    approved, update = _paths(root, start, head), _paths(root, main, tree)
    if tree is None or approved is None or update is None:
        raise kd.PublishRefused("no_update", f"the paths of {head} on {main} cannot be read")
    if update != approved or not all(_owned(path, task.owned_paths) for path in update):
        return _returned(db_path, row, facts, reason)
    if merged.returncode == 1:
        conflicts = fields[1:fields.index("", 1)] if "" in fields[1:] else []
        # A name that is no UTF-8 holds a lone surrogate here, which neither the event nor the body can store as text.
        if not conflicts or not approved.issuperset(conflicts) or any(
                path != path.encode("utf-8", "replace").decode("utf-8") for path in conflicts):
            return _returned(db_path, row, facts, "conflict")
        held = [_held(root, tree, head, main, path) for path in conflicts]
        if None in held:
            return _returned(db_path, row, facts, "conflict")
        facts["conflicts"] = conflicts
        if "deleted_on_main" in held:
            facts["deleted_on_main"] = [path for path, kind in zip(conflicts, held) if kind == "deleted_on_main"]
    branch = _BRANCH.format(source=source, head=head[:12], main=main[:12])
    commit = _branch(root, branch, tree, head, main, _MESSAGE.format(**facts))
    return _updated(db_path, row, facts, (root, branch, commit))


def _moved(conn, row: dict) -> dict:
    """The row's latest delivery_base_moved event at its head: the reviewed base and the main the arm read."""
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id = ? "
                                   "AND kind = 'delivery_base_moved' ORDER BY id DESC", (row["source_task_id"],)):
        moved = json.loads(payload)
        if moved.get("delivery_id") == row["id"] and moved.get("head") == row["pull_request_head"]:
            return moved
    return {}


def _approved_base(conn, source: str, base: str) -> Optional[str]:
    """Where the change reviewed on ``base`` for ``source`` starts: while ``base`` is the head of a delivery of a
    task_links parent of the card or of a card it replaced, an earlier delivery the card continues, that delivery's
    reviewed base, then on from that card, at most 8 links back. ``base`` itself when no such delivery is at it;
    None when a link records no reviewed base or the chain goes on past 8 links."""
    for link in range(9):
        cards = [parent for (parent,) in conn.execute("SELECT parent_id FROM task_links WHERE child_id = ?",
                                                      (source,))] + replaced_task_ids(conn, source)
        earlier = conn.execute(
            "SELECT source_task_id FROM kanban_deliveries WHERE pull_request_head = ? AND source_task_id IN "
            f"({', '.join('?' * len(cards))}) ORDER BY id DESC LIMIT 1", (base, *cards)).fetchone() if cards else None
        if earlier is None:
            return base
        if link == 8:
            break
        source, base = earlier[0], kd._reviewed(conn, earlier[0], base)[0]
        if base is None:
            return None
    return None  # a ninth link: where the chain starts is not read, so the work goes back


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


def _main(root: Path, repository: str) -> str:
    """Main as GitHub holds it now, its objects fetched into ``root`` with the fenced fetch worktree_start uses:
    into refs/remotes/origin/main only, so HEAD, branches, index and worktree never move. A failed fetch refuses
    the pass, so the row stays base_moved and the next pass reads main again."""
    from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError

    try:
        main = kd._sha(GitHubTransport("publish", repository).fetch_main(root))
    except GitHubTransportError:
        main = None
    if main is None:
        raise kd.PublishRefused("no_main", f"main of {repository} was not fetched")
    return main


def _made(conn, row: dict, main: str) -> Optional[dict]:
    """The kernel's own record of the row's update card on ``main``: the source card's delivery_updated event of
    this delivery, H and main, written in the transaction that created the card. None while there is none; no
    field a caller can choose on a task, as its idempotency key, ever stands for it."""
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_updated' "
                                   "ORDER BY id", (row["source_task_id"],)):
        made = json.loads(payload)
        if (made.get("delivery_id"), made.get("head"), made.get("observed_main")) == (
                row["id"], row["pull_request_head"], main) and made.get("update_task_id"):
            return made
    return None


def _paths(root: Path, old: str, new: str) -> Optional[frozenset]:
    """The paths git changes from ``old`` to ``new``, each tree-ish, renames as two paths."""
    changed = kd._run_git(root, "diff-tree", "-r", "--name-only", "--no-renames", "-z", old, new)
    if changed is None or changed.returncode != 0:
        return None
    return frozenset(path for path in changed.stdout.decode("utf-8", "surrogateescape").split("\0") if path)


def _held(root: Path, tree: str, head: str, main: str, path: str) -> Optional[str]:
    """How git's merged ``tree`` of main and H holds the conflicted ``path``, read through git objects and never from
    the exit code alone: "deleted_on_main" when main holds nothing there and the tree exactly H's blob, as git leaves
    a file main deleted and H modified; "marked" when the tree's blob holds git's conflict markers, a line starting
    with each of <<<<<<<, ======= and >>>>>>>; None for anything else. Git takes everything after the first colon of
    ``<commit or tree>:<path>`` as a literal path, so no name needs escaping."""
    def read(*args: str) -> Optional[bytes]:
        found = kd._run_git(root, *args)
        return found.stdout if found is not None and found.returncode == 0 else None

    blob = read("cat-file", "blob", f"{tree}:{path}")
    if blob is None:
        return None
    on_main = kd._run_git(root, "rev-parse", "--verify", "--quiet", f"{main}:{path}")
    in_head, in_tree = (read("rev-parse", "--verify", "--quiet", f"{commit}:{path}") for commit in (head, tree))
    if on_main is not None and on_main.returncode == 1 and in_head is not None and in_head == in_tree:
        return "deleted_on_main"
    lines = blob.split(b"\n")
    if all(any(line.startswith(marker) for line in lines) for marker in (b"<<<<<<<", b"=======", b">>>>>>>")):
        return "marked"
    return None


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


def _updated(db_path: Path, row: dict, facts: dict, update: Optional[tuple]) -> Optional[str]:
    """One write transaction, while the row is still base_moved: unless the kernel's delivery_updated event of
    this delivery, H and main records the update card already, the card is made through _continuation's card
    creation at the updated head, so for the source's builder with the source's governance, its recorded base then
    main, and that event names it; then the row is final, updated."""
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not _still_moved(conn, row):
            return None
        if _made(conn, row, facts["main"]) is None:
            if update is None:
                return None
            root, branch, commit = update
            deleted = facts.get("deleted_on_main", [])
            marked = ", ".join(path for path in facts.get("conflicts", ()) if path not in deleted)
            body = (_DELETED_BODY if deleted else _CONFLICT_BODY if "conflicts" in facts else _UPDATE_BODY).format(
                head=facts["head"], source=facts["source"], number=row["pull_request_number"],
                repository=row["repository"], base=facts["base"], main=facts["main"], update=commit, branch=branch,
                conflicts=", ".join(facts.get("conflicts", ())), deleted=", ".join(deleted),
                marked=_ALSO_MARKED.format(marked=marked) if marked else "")
            made = kd._continuation(conn, db_path, {**row, "pull_request_head": commit}, body, (root, branch))
            # The kernel's scoped completion hands over only committed changes since the recorded base, so the
            # unchanged updated head is handed over against main, which it holds: exactly the approved paths.
            conn.execute("UPDATE tasks SET base_commit = ? WHERE id = ? AND base_commit = ?",
                         (facts["main"], made["rework"], commit))
            kb._append_event(conn, row["source_task_id"], "delivery_updated", kd._outcome(
                row, repository=row["repository"], reviewed_base=facts["base"], observed_main=facts["main"],
                update_task_id=made["rework"], update_head=commit, update_branch=branch,
                **({"approved_base": facts["approved_base"]} if "approved_base" in facts else {}),
                **({"conflicts": facts["conflicts"]} if "conflicts" in facts else {}),
                **({"deleted_on_main": facts["deleted_on_main"]} if "deleted_on_main" in facts else {}),
                **({"parked": made["parked"]} if made.get("parked") else {})))
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (kd._UPDATED, row["id"]))
    return kd._UPDATED


def _returned(db_path: Path, row: dict, facts: dict, reason: str) -> Optional[str]:
    """H back to its builder through its continuation card, on the continuation branch made at H first as card
    2's return does; the row is returned for changes. The card's body says why, and _continuation's own
    review_followup_recorded event names the card, so no other event is written. No card goes to review."""
    branch = kd._return_branch(db_path, row)
    with kb.connect_closing(db_path=db_path) as conn, kb.write_txn(conn):
        if not _still_moved(conn, row):
            return None
        body = _RETURN_BODY.format(head=facts["head"], source=facts["source"], number=row["pull_request_number"],
                                   repository=row["repository"], base=facts["base"], main=facts["main"],
                                   why=_WHY[reason])
        kd._continuation(conn, db_path, row, body, branch)
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' WHERE id = ?",
                     (row["id"],))
    return "returned_for_changes"
