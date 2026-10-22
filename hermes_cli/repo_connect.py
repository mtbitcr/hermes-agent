"""Connect an owner's GitHub repository to a project: clone main and bind it as the project's primary folder.

Nothing imports this until the owner route card. connect_repository first refuses, writing nothing, a name the
transport refuses, an owner other than the account the owner delivery list admits, a project that already has a
primary folder, and a repository that already has a folder or a project. It then reads main through a read_checks
transport: GitHub answers 200 with main's commit, 409 for an empty repository and 404 when main is missing. Both of
the last count as empty, and any other answer refuses. An empty repository gets its first commit from repo_bootstrap.
Main is then cloned into <kanban root>/repositories/<owner>/<name>, bound as the project's primary folder and made
the default workdir of the project's board, as projects_cmd makes it when it binds a board. A failure after the
clone removes the new folder and its record.

The kanban root is shared by every profile, so card workers of every profile reach the clone. The projects store is
the calling profile's own.
"""
from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError, _is_sha
from hermes_cli.kanban_delivery_owner_policy import _INSTALLATION_ACCOUNT
from hermes_cli.repo_bootstrap import bootstrap


class RepositoryConnectError(GitHubTransportError):
    """A connect refused before anything is written, named by a fixed reason code as the transport's refusals are."""


def _folder(repository: str) -> Path:
    """The clone's folder, named lowercased: one folder for a repository however its name is spelled."""
    owner, name = repository.lower().split("/")
    return kb.kanban_home() / "kanban" / "repositories" / owner / name


def connect_repository(project_slug: str, repository: str) -> dict:
    """Clone `repository` and bind it as the primary folder of the project `project_slug`. Returns the receipt,
    {"repository", "folder", "bootstrapped", "commit"}: whether the first commit was pushed here, and the commit
    main held when it was cloned."""
    transport = GitHubTransport("read_checks", repository)  # refuses a name it does not take: bad_repository
    if repository.split("/")[0] != _INSTALLATION_ACCOUNT:
        raise RepositoryConnectError("foreign_account")
    folder = _folder(repository)
    key = pdb._primary_path_key(str(folder))
    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, project_slug)
        if project is None:
            raise RepositoryConnectError("unknown_project")
        if project.primary_path or any(f.is_primary for f in project.folders):
            raise RepositoryConnectError("project_has_primary_folder")
        if os.path.lexists(folder):
            raise RepositoryConnectError("repository_has_folder")
        if any(pdb._primary_path_key(f.path) == key
               for other in pdb.list_projects(conn, include_archived=True) for f in other.folders):
            raise RepositoryConnectError("repository_has_project")

    answer = transport.request("GET", f"/repos/{repository}/git/ref/heads/main")
    empty = answer["status"] in (404, 409)
    if not empty:
        target = answer["data"].get("object") if answer["status"] == 200 and isinstance(answer["data"], dict) else None
        if not isinstance(target, dict) or not _is_sha(target.get("sha")):
            raise RepositoryConnectError("main_unreadable")
    if empty:
        bootstrap(repository)

    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()  # the claim: of two connects of one repository, one makes the folder
    except FileExistsError:
        raise RepositoryConnectError("repository_has_folder") from None
    try:
        commit = transport.clone_main(folder)
        with pdb.connect_closing() as conn:
            path = pdb.add_folder(conn, project.id, str(folder))
            bound = pdb.get_project(conn, project.id)
        if bound is None or bound.primary_path != path:  # a primary bound meanwhile stays the primary
            raise RepositoryConnectError("project_has_primary_folder")
        board = kb._normalize_board_slug(bound.board_slug)
        if board and (board == kb.DEFAULT_BOARD or kb.board_exists(board)):
            kb.write_board_metadata(board, default_workdir=path)
    except BaseException:
        with contextlib.suppress(Exception), pdb.connect_closing() as conn:
            pdb.remove_folder(conn, project.id, str(folder))
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return {"repository": repository, "folder": path, "bootstrapped": empty, "commit": commit}
