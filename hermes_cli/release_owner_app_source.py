"""The owner app's merge commit on the release host, fetched and archived (owner app release plan, card 2).

The release builds the owner app's image from one exact merge commit of mtbitcr/raphael-workspace, on the release
host itself. fetch_source fetches that commit into the platform's own bare copy of the repository, under the kanban
root that every profile and board shares, through the transport's release_source step: the existing GitHub App's
one read-only step, whose token holds contents read on this one repository. As in fetch_main, the credential lives
only in the environment of the one git that fetches, in a fresh private repository removed after it, so none reaches
the copy, a file or a log. The commit is then written as a git archive into a fresh private build folder, the
build's input.

Nothing calls this module yet: the release run starts it at the owner app's build (card 9). Every import is made at
load, and no secret, host address or key file's content is copied, printed or logged.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hermes_cli.kanban_db import kanban_home
from hermes_cli.kanban_delivery_github import (
    _COMMIT_REPOSITORY, GitHubTransport, GitHubTransportError, _git_base_env, _is_sha, _run_git,
)

REPOSITORY = _COMMIT_REPOSITORY  # the owner app, the one repository this module reads: the transport's own binding
STEP = "release_source"  # the transport's read-only step: contents read and nothing else


def copy_path() -> Path:
    """The platform's bare copy of the owner app: under the kanban root, never a profile home or a board, so a
    release started from any of them finds the one copy. Resolved through kanban_home, never kanban_db_path(): that
    honours HERMES_KANBAN_DB, which the dispatcher pins to a worker's own board."""
    return kanban_home() / "kanban" / "release-source" / "raphael-workspace.git"


def fetch_source(commit: str) -> Path:
    """Fetch the owner app's `commit`, 40 lowercase hex, into the copy, and return the git archive of it: a tar file
    alone in a fresh private build folder, which the caller removes when the build is done. A short, symbolic or
    relative commit is refused (bad_commit) before any token, folder or git exists."""
    if not _is_sha(commit):
        raise GitHubTransportError("bad_commit")
    transport = GitHubTransport(STEP, REPOSITORY)  # the App's key under the platform root, never a profile's
    copy = copy_path()
    try:
        copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # the owner app's source is private
        home = tempfile.mkdtemp(prefix="hermes-release-source-")  # mode 0700, for this module's own gits
    except OSError:
        raise GitHubTransportError("copy_failed") from None
    try:
        env = _git_base_env(home)
        try:
            Path(env["GIT_CONFIG_GLOBAL"]).touch(exist_ok=False)  # the global configuration: empty
        except OSError:
            raise GitHubTransportError("copy_failed") from None
        # Made on first use with an empty --template=, so it holds what init writes; a copy made before is kept.
        made = _run_git(["git", "init", "--bare", "--quiet", "--template=", str(copy)], env)
        if made is None or made.returncode != 0:
            raise GitHubTransportError("copy_failed")
        transport.fetch_commit(copy, commit)  # the only git with the credential, in the transport's own folder
        try:
            build = tempfile.mkdtemp(prefix="hermes-release-build-")  # mode 0700: the build's input stays private
        except OSError:
            raise GitHubTransportError("archive_failed") from None
        archive = os.path.join(build, f"{commit}.tar")
        written = _run_git(["git", "--git-dir", str(copy), "archive", "--format=tar", f"--output={archive}", commit],
                           env)
        if written is None or written.returncode != 0:
            shutil.rmtree(build, ignore_errors=True)
            raise GitHubTransportError("archive_failed")
    finally:
        shutil.rmtree(home, ignore_errors=True)  # the empty configuration this module's gits read
    return Path(archive)
