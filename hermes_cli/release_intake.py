"""The merged record step: one merged change into the release record (release card 10).

The delivery step calls :func:`record_merged_change` once GitHub, read through its
transport, shows a delivery's pull request merged into main. The record carries the
merge commit and the pull request, the base, head and tree review approved (never
values recomputed from the merge: a later release check judges those against git),
and the card's recorded risk tier, plain title and board. A repeat of the same merge
writes nothing.

The release store is opened through :func:`hermes_cli.release_ledger.connect`, which
resolves the kanban root, never a board file or a profile home, so a dispatched
worker's environment reaches the same store as the dispatcher.
"""

from __future__ import annotations

from typing import Any

from hermes_cli import release_ledger

# The release record keeps merges of the platform repository only, compared byte for byte:
# a release run takes NEW, the last change's merge commit, in the platform checkout
# (release_cmd), which holds no commit of another repository.
_PLATFORM_REPOSITORY = "mtbitcr/hermes-agent"


def record_merged_change(
    *,
    repository: str,
    number: int,
    merge_commit: str,
    reviewed_base: str,
    reviewed_head: str,
    reviewed_tree: str,
    tier: Any,
    card_id: str,
    title: str,
    board: str,
) -> dict[str, Any]:
    """Add the merge of pull request ``number`` of ``repository`` to the waiting release
    decision. Called only after the caller's GitHub read showed it merged into main,
    which is the ``on_main`` confirmation. Returns :func:`release_ledger.record_merge`'s
    answer; ``recorded`` is False for a repeat. Any failure raises, and nothing is written.

    Only a platform merge is recorded: a merge of any other repository writes nothing
    and answers ``recorded`` False, with no member and no batch, since the owner app
    deploy releases it."""
    if repository != _PLATFORM_REPOSITORY:
        return {"recorded": False, "member": None, "batch": None}
    conn = release_ledger.connect()
    try:
        return release_ledger.record_merge(
            conn,
            merge_commit=merge_commit,
            pr_url=f"https://github.com/{repository}/pull/{number}",
            reviewed_base=reviewed_base,
            reviewed_head=reviewed_head,
            reviewed_tree=reviewed_tree,
            tier=tier,
            card_id=card_id,
            title=title,
            board=board,
            on_main=True,
        )
    finally:
        conn.close()
