"""The owner connects their own GitHub repository to a project (card A5), served by the API server.

Three routes, behind the owner authentication of the release routes. The view answers whether the project holds a
repository, and which. The check writes nothing: it reads main and answers the repository, the required checks,
whether main is empty, and the digest of that delivery entry. The connect is bound to the digest the check gives
now; any other digest is refused and nothing is written. It then validates the entry, so an entry the owner list
may not name, such as a platform repository, is refused before anything is written. It runs ``connect_repository``
unless the project's primary folder already holds the repository, then always records the confirmed entry in the
owner delivery list. Each step is idempotent, so a retry completes a step an earlier answer missed and a repeat
writes nothing. The receipt is read from the clone, so a repeat answers the first receipt, also after a restart. No
answer holds the clone's folder or a refusal's reason code; each refusal is one plain sentence.

The check and the connect accept only a repository name in lowercase letters, and every step uses it exactly as given,
as the transport checks the repository its token names exactly. A name with a capital letter is refused before any
call, clone or record; a repository whose name on GitHub has capital letters is renamed there first.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from gateway.platforms.api_server import _openai_error, _owner_workspace_toolset_enabled
from hermes_cli import projects_db as pdb
from hermes_cli import repo_connect
from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError, _is_sha
from hermes_cli.kanban_delivery_owner_policy import _code_policy, entry_digest, record_confirmed_entry, validate_entry
from hermes_cli.repo_bootstrap import _EMAIL, _MESSAGE, _NAME

logger = logging.getLogger(__name__)

_REQUIRED_CHECKS = ("quality",)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FIELDS = {"check": {"repository"}, "connect": {"repository", "digest"}}
# One sentence for each refusal of connect_repository; a transport refusal not named here gets _TRANSPORT.
_REFUSALS = {
    "foreign_account": "Only a repository of the account the platform delivers to can be connected.",
    "unknown_project": "This project could not be found.",
    "project_has_primary_folder": "This project already has a main folder, so no repository was connected.",
    "repository_has_folder": "This repository already has a folder on the server, so it was not connected again.",
    "repository_has_project": "This repository already belongs to another project.",
    "main_unreadable": "The main branch of this repository could not be read.",
}
_TRANSPORT = "GitHub did not complete the request for this repository."
_CHANGED = "The repository check changed after it was shown, so nothing was connected."
_NOT_LISTED = "This repository cannot be added to the delivery list, so the connection was not completed."
_LOWERCASE = ("Use the repository name in lowercase letters. "
              "If its name on GitHub has capital letters, rename it on GitHub first.")
# One connect at a time in this gateway: a second click waits, then finds the repository the first one bound,
# and the owner list is never rewritten by two records at once.
_CONNECTING = threading.Lock()


def route(adapter: Any, action: str):
    """The handler of one repository route, behind the owner authentication of the release routes."""

    async def handle(request: web.Request) -> web.Response:
        auth_err = adapter._check_auth(request)
        if auth_err:
            return auth_err
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return _refusal(
                    404, "Owner workspace is not enabled for this profile", "owner_workspace_not_enabled"
                )
            slug = request.match_info["slug"]
            if action == "view":
                view = await asyncio.to_thread(_view, slug)
                if view is None:
                    return _refusal(404, _REFUSALS["unknown_project"], "project_not_found")
                return web.json_response(view)
            body, err = await adapter._read_json_body(request)
            repository, digest = body.get("repository"), body.get("digest")
            shown = action == "check" or (type(digest) is str and _DIGEST.fullmatch(digest))
            if err or set(body) != _FIELDS[action] or type(repository) is not str or not shown:
                return _refusal(400, "Invalid repository request", "invalid_argument")
            GitHubTransport("read_checks", repository)  # refuses a name it does not take, as it was sent
            if repository != repository.lower():  # only a lowercase name, then used exactly as given
                return _refusal(400, _LOWERCASE, "invalid_argument")
            if action == "check":
                return web.json_response({
                    "object": "hermes.owner_workspace.repository_check",
                    "repository": repository,
                    "required_checks": list(_REQUIRED_CHECKS),
                    "main_empty": await asyncio.to_thread(_main_empty, GitHubTransport("read_checks", repository)),
                    "digest": entry_digest(repository, list(_REQUIRED_CHECKS)),
                })
            if digest != entry_digest(repository, list(_REQUIRED_CHECKS)):
                return _refusal(409, _CHANGED, "repository_changed")
            bootstrapped = await asyncio.to_thread(_connect, slug, repository, digest)
            if bootstrapped is None:
                return _refusal(409, _NOT_LISTED, "repository_not_listed")
            return web.json_response({
                "object": "hermes.owner_workspace.repository_receipt",
                "repository": repository,
                "bootstrapped": bootstrapped,
            })
        except GitHubTransportError as error:  # RepositoryConnectError included
            logger.warning("[api_server] owner repository %s refused: %s", action, error.reason)
            if error.reason == "bad_repository":
                return _refusal(400, "This is not a GitHub repository name.", "invalid_argument")
            return _refusal(409, _REFUSALS.get(error.reason, _TRANSPORT), "repository_refused")
        except Exception:
            logger.exception("[api_server] owner repository %s failed", action)
            return _refusal(
                503,
                "The repository connection is unavailable right now; please try again shortly.",
                "owner_workspace_unavailable",
            )

    return handle


def _view(slug: str) -> Optional[dict[str, Any]]:
    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, slug)
    if project is None:
        return None
    held = _held(project)
    return {"object": "hermes.owner_workspace.repository", "state": "connected" if held else "none",
            "repository": held}


def _main_empty(transport: GitHubTransport) -> bool:
    """Whether main is empty, read as connect_repository reads it: 404 and 409 are empty, a commit is not."""
    answer = transport.request("GET", f"/repos/{transport.repository}/git/ref/heads/main")
    if answer["status"] in (404, 409):
        return True
    target = answer["data"].get("object") if answer["status"] == 200 and isinstance(answer["data"], dict) else None
    if not isinstance(target, dict) or not _is_sha(target.get("sha")):
        raise repo_connect.RepositoryConnectError("main_unreadable")
    return False


def _connect(slug: str, repository: str, digest: str) -> Optional[bool]:
    """Validate the entry, connect unless the project's primary folder already holds the repository, then record
    the confirmed entry: every step runs and each is idempotent, so a retry completes the step an earlier answer
    missed. Returns the receipt's bootstrapped, read from the clone; None when the entry is not listed."""
    with _CONNECTING:
        decision = validate_entry(repository, list(_REQUIRED_CHECKS), _code_policy())
        if decision.allowed:  # a refused entry is never cloned, bound or recorded
            with pdb.connect_closing() as conn:
                project = pdb.get_project(conn, slug)
            if project is None or _held(project) != repository:
                repo_connect.connect_repository(slug, repository)
            decision = record_confirmed_entry(repository, list(_REQUIRED_CHECKS), digest)
        if decision.code in ("confirmed", "already_confirmed"):  # a bound clone alone is no finished connect
            return _bootstrapped(repo_connect._folder(repository))
    logger.warning("[api_server] owner repository connect not listed: %s", decision.code)
    return None


def _bootstrapped(folder: Path) -> bool:
    """Whether main in the clone has one root commit, and that commit has the author, the committer and the
    message of the first commit hermes_cli/repo_bootstrap.py writes in an empty repository."""
    roots = _git(folder, "rev-list", "--max-parents=0", "refs/heads/main").split()
    if len(roots) != 1:
        return False
    header, _, message = _git(folder, "cat-file", "commit", roots[0]).partition("\n\n")
    lines, identity = header.splitlines(), f"{_NAME} <{_EMAIL}> "
    return (message == f"{_MESSAGE}\n" and any(line.startswith(f"author {identity}") for line in lines)
            and any(line.startswith(f"committer {identity}") for line in lines))


def _git(folder: Path, *args: str) -> str:
    """The output of a git command that reads the clone; a failure raises, so the connect answers 503."""
    return subprocess.run(
        ["git", "-C", str(folder), *args], capture_output=True, check=True, encoding="utf-8", errors="replace",
    ).stdout


def _held(project: Any) -> Optional[str]:
    """The repository whose clone is the project's primary folder, named as its folder is: in lowercase letters, the
    only name the routes accept. None when the primary folder is not such a clone."""
    if not project.primary_path:
        return None
    path = Path(project.primary_path)
    repository = f"{path.parent.name}/{path.name}"
    clone = pdb._primary_path_key(str(repo_connect._folder(repository)))
    return repository if clone == pdb._primary_path_key(project.primary_path) else None


def _refusal(status: int, message: str, code: str) -> web.Response:
    return web.json_response(_openai_error(message, code=code), status=status)
