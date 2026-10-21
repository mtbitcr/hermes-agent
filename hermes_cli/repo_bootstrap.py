"""The first commit of an empty connected repository: a README and its minimal quality check.

Nothing imports this until the connect card. Run once at connect, after owner decision D2, bootstrap
builds one commit from fixed content in a fresh temporary repository and pushes it to main through the
transport's bootstrap step, the only one whose token may write workflows. The push is leased on "": main
must not exist yet, so a repository whose main already exists is refused (push_lease_mismatch) and
nothing is overwritten.

The repository is made with an empty --template= and every git runs in the transport's own environment
(_git_base_env), so no hook, user configuration, identity or other host variable reaches the commit:
its author and committer are fixed.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError, _commit_of, _git_base_env, _run_git

_NAME, _EMAIL = "Raphael", "raphael@users.noreply.github.com"  # the author and the committer, always
_MESSAGE = "Add a README and the quality check"
_CHECK = "scripts/ci.sh"  # mode 100755, the one file that is not 100644
# The checkout .github/workflows/tests-os.yml pins, so the first commit runs only an action already reviewed.
_CHECKOUT = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2"
_FILES = {
    _CHECK: "#!/usr/bin/env bash\nset -euo pipefail\ntest -s README.md\n",
    ".github/workflows/ci.yml": f"""name: ci

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: {_CHECKOUT}
        with:
          persist-credentials: false
      - run: bash {_CHECK}
""",
}


def bootstrap(repository: str, *, key_path: str | Path | None = None) -> dict:
    """Push the first commit to main of `repository`, which must not have main yet, and return the push's
    result, {"state", "branch", "head"}."""
    transport = GitHubTransport("bootstrap", repository, key_path=key_path)  # the repository is checked first
    files = {"README.md": f"# {repository.split('/')[1]}\n\nThe quality check runs `{_CHECK}`.\n", **_FILES}
    try:
        home = tempfile.mkdtemp(prefix="hermes-repo-bootstrap-")  # mode 0700, for this one commit
    except OSError:
        raise GitHubTransportError("bootstrap_failed") from None
    try:
        work = os.path.join(home, "first")
        env = _git_base_env(home)
        env.update(GIT_AUTHOR_NAME=_NAME, GIT_AUTHOR_EMAIL=_EMAIL, GIT_COMMITTER_NAME=_NAME, GIT_COMMITTER_EMAIL=_EMAIL)
        try:
            Path(env["GIT_CONFIG_GLOBAL"]).touch(exist_ok=False)  # the global configuration: empty
            for path, text in files.items():
                Path(work, path).parent.mkdir(parents=True, exist_ok=True)
                Path(work, path).write_bytes(text.encode())
        except OSError:
            raise GitHubTransportError("bootstrap_failed") from None
        # An empty --template= copies no sample hook or other file: the repository holds what init writes.
        # The check is added as executable whatever the file system records.
        for argv in (["git", "init", "--quiet", "--template=", work],
                     ["git", "-C", work, "add", "--", *(path for path in files if path != _CHECK)],
                     ["git", "-C", work, "add", "--chmod=+x", "--", _CHECK],
                     ["git", "-C", work, "commit", "--quiet", "-m", _MESSAGE]):
            result = _run_git(argv, env)
            if result is None or result.returncode != 0:
                raise GitHubTransportError("bootstrap_failed")
        sha = _commit_of(_run_git(["git", "-C", work, "rev-parse", "--verify", "HEAD^{commit}"], env))
        if not sha:
            raise GitHubTransportError("bootstrap_failed")
        pushed = transport.push(work, "main", sha, expected="")  # the lease "": main must not exist yet
        # git lets a main that already holds this very commit pass the lease as up to date: a repeat all the same.
        if pushed["state"] == "up_to_date":
            raise GitHubTransportError("push_lease_mismatch")
        return pushed
    finally:
        shutil.rmtree(home, ignore_errors=True)  # the repository and its commit, pushed or not
