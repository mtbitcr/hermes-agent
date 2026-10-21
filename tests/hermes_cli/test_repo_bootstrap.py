"""The first commit of an empty connected repository, with its minimal quality check (card A3).

A local bare repository stands in for GitHub's git side, with GIT_ORIGIN pointed at file://, and the
transport tests' stand-in plays its token endpoint, recording each token request.
"""
import importlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from tests.hermes_cli.test_kanban_delivery_github import (  # noqa: F401  (github is a fixture)
    BASIC, REPO, TOKEN, _carries, _files_text, _git, _record_git, _record_mkdtemp, _remote_head, _remote_refs,
    _repositories, _subcommand, _transport_module, github,
)

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = {"contents": "write", "workflows": "write"}
NAME, EMAIL = "Raphael", "raphael@users.noreply.github.com"


def _bootstrap_module():
    """Import inside each test, so a missing module fails that test rather than the whole file."""
    return importlib.import_module("hermes_cli.repo_bootstrap")


def _remote(monkeypatch, github, tmp_path):
    """The transport pointed at the stand-in, an empty bare remote as its GIT_ORIGIN, and a card
    repository with three commits to fill the remote with."""
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, commits = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    return mod, work, bare, commits


def test_the_bootstrap_mint_requests_exactly_contents_and_workflows_write(github, monkeypatch):
    mod = _transport_module(monkeypatch, github)
    for step in ("bootstrap", "publish", "merge"):  # one token each: a read of main mints it
        assert mod.GitHubTransport(step, REPO, key_path=github["key_file"]).branch_head("main") is None

    bootstrap, publish, merge = github["token_requests"]
    assert bootstrap == {"repositories": ["hermes-agent"], "permissions": BOOTSTRAP}
    assert publish == merge == {"repositories": ["hermes-agent"],
                                "permissions": {"contents": "write", "pull_requests": "write"}}
    assert [step for step, permissions in mod.STEP_PERMISSIONS.items() if "workflows" in permissions] == ["bootstrap"]


def test_an_empty_remote_gets_exactly_the_three_files_on_main(github, monkeypatch, tmp_path):
    mod, work, bare, _ = _remote(monkeypatch, github, tmp_path)
    made = _record_mkdtemp(monkeypatch)
    # A host whose own identity, configuration, hooks and template would all reach a plain commit.
    home, hooks, template = tmp_path / "home", tmp_path / "hooks", tmp_path / "template"
    for folder in (home, hooks, template / "hooks"):
        folder.mkdir(parents=True)
    for hook in (hooks / "pre-commit", hooks / "post-commit", template / "hooks" / "post-commit"):
        hook.write_text(f"#!/bin/sh\ntouch {tmp_path / 'hooked'}\n")
        hook.chmod(0o755)
    (home / ".gitconfig").write_text(f"[user]\n\tname = Somebody\n\temail = somebody@example.invalid\n"
                                     f"[core]\n\thooksPath = {hooks}\n")
    with monkeypatch.context() as host:
        host.setenv("HOME", str(home))
        host.setenv("GIT_TEMPLATE_DIR", str(template))
        for variable in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
            host.setenv(variable, "somebody@example.invalid")
        result = _bootstrap_module().bootstrap(REPO, key_path=github["key_file"])

    head = _remote_head(bare, "main")
    assert result == {"state": "pushed", "branch": "main", "head": head}
    assert _remote_refs(bare) == ["refs/heads/main"]
    assert github["token_requests"] == [{"repositories": ["hermes-agent"], "permissions": BOOTSTRAP}]
    listed = [line.split("\t") for line in _git("--git-dir", str(bare), "ls-tree", "-r", "main").splitlines()]
    assert {path: meta.split()[0] for meta, path in listed} == {
        "README.md": "100644", "scripts/ci.sh": "100755", ".github/workflows/ci.yml": "100644"}
    assert _git("--git-dir", str(bare), "log", "--format=%P|%an|%ae|%cn|%ce", "main").splitlines() == [
        f"|{NAME}|{EMAIL}|{NAME}|{EMAIL}"]  # one commit, with no parent
    assert not (tmp_path / "hooked").exists()
    assert made and not [folder for folder in made if folder.exists()]  # the commit's repository is gone

    def show(path):
        return _git("--git-dir", str(bare), "show", f"main:{path}")

    readme = show("README.md").splitlines()
    assert readme[0] == "# hermes-agent" and [line for line in readme[1:] if line] == [
        "The quality check runs `scripts/ci.sh`."]
    assert show("scripts/ci.sh").splitlines() == ["#!/usr/bin/env bash", "set -euo pipefail", "test -s README.md"]
    pinned = re.search(r"actions/checkout@[0-9a-f]{40}", (ROOT / ".github/workflows/tests-os.yml").read_text())
    assert yaml.safe_load(show(".github/workflows/ci.yml")) == {
        "name": "ci",
        True: {"pull_request": None, "push": {"branches": ["main"]}},  # YAML 1.1 reads the key on as True
        "permissions": {"contents": "read"},
        "jobs": {"quality": {"runs-on": "ubuntu-latest", "timeout-minutes": 10, "steps": [
            {"uses": pinned.group(), "with": {"persist-credentials": False}}, {"run": "bash scripts/ci.sh"}]}}}

    checkout = tmp_path / "checkout"  # the check passes on the commit it arrives in, and fails without README.md
    _git("clone", "-q", "--branch", "main", str(bare), str(checkout))
    assert subprocess.run(["bash", "scripts/ci.sh"], cwd=checkout).returncode == 0
    (checkout / "README.md").unlink()
    assert subprocess.run(["bash", "scripts/ci.sh"], cwd=checkout, capture_output=True).returncode != 0


def test_a_remote_whose_main_already_exists_is_refused_and_keeps_its_commit(github, monkeypatch, tmp_path):
    mod, work, bare, (first, *_) = _remote(monkeypatch, github, tmp_path)
    _git("push", "-q", str(bare), f"{first}:refs/heads/main", cwd=work)

    with pytest.raises(mod.GitHubTransportError) as refused:
        _bootstrap_module().bootstrap(REPO, key_path=github["key_file"])
    assert refused.value.reason == "push_lease_mismatch"
    assert _remote_refs(bare) == ["refs/heads/main"] and _remote_head(bare, "main") == first


def test_a_second_bootstrap_reports_no_success_even_on_the_same_commit_and_main_keeps_the_first(
        github, monkeypatch, tmp_path):
    """Review finding one: once main exists, a second bootstrap is refused whether it builds the very commit main
    holds or another one. The bootstrap's own commit dates are pinned, so equal dates build an equal commit with no
    clock involved, and the real push records each commit it is offered."""
    mod, _, bare, _ = _remote(monkeypatch, github, tmp_path)
    boot = _bootstrap_module()
    push, offered = mod.GitHubTransport.push, []

    def recorded_push(self, worktree, branch, sha, *, expected):  # straight through to the real push
        offered.append((branch, sha, expected))
        return push(self, worktree, branch, sha, expected=expected)

    def bootstrap(when):
        """One bootstrap whose commit is dated *when*, and what it reported: its result, or the reason it raised."""
        monkeypatch.setattr(boot, "_git_base_env", lambda home: {
            **mod._git_base_env(home), "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})
        try:
            return boot.bootstrap(REPO, key_path=github["key_file"])
        except mod.GitHubTransportError as exc:
            return exc.reason

    monkeypatch.setattr(mod.GitHubTransport, "push", recorded_push)
    once, later = "2026-10-21T12:00:00+00:00", "2026-10-21T12:00:01+00:00"
    outcomes = [bootstrap(when) for when in (once, once, later)]  # once, the same commit again, then another

    first, again, other = (sha for _, sha, _ in offered)
    assert again == first != other  # offered the very commit main holds, then a different one
    assert offered == [("main", first, ""), ("main", first, ""), ("main", other, "")]  # each leased on ""
    assert outcomes == [{"state": "pushed", "branch": "main", "head": first}, "push_lease_mismatch",
                        "push_lease_mismatch"]  # neither repeat reports success
    assert _remote_refs(bare) == ["refs/heads/main"] and _remote_head(bare, "main") == first
    # Both dates as pinned, read back raw (epoch seconds and offset) as every git prints them: the strict ISO form of
    # %aI and %cI differs between gits, whose newer releases print a zero offset as Z rather than +00:00.
    pinned = datetime.fromisoformat(once)
    raw = f"{int(pinned.timestamp())} {pinned:%z}"  # 1792584000 +0000
    assert _git("--git-dir", str(bare), "log", "--date=raw", "--format=%ad %cd", "main") == f"{raw} {raw}"


def test_clone_main_gives_mains_commit_with_no_credential_and_refuses_a_folder_that_is_not_empty(
        github, monkeypatch, tmp_path):
    mod, work, bare, (first, second, third) = _remote(monkeypatch, github, tmp_path)
    _git("push", "-q", str(bare), f"{second}:refs/heads/main", f"{third}:refs/heads/delivery/card-1", cwd=work)
    made = _record_mkdtemp(monkeypatch)
    calls = _record_git(monkeypatch)
    transport = mod.GitHubTransport("read_checks", REPO, key_path=github["key_file"])
    empty = tmp_path / "empty"
    empty.mkdir()

    for clone in (tmp_path / "missing" / "clone", empty):
        assert transport.clone_main(clone) == second
        assert _git("rev-parse", "HEAD", cwd=clone) == second
        assert _git("symbolic-ref", "HEAD", cwd=clone) == "refs/heads/main"
        assert _git("for-each-ref", "--format=%(refname)", cwd=clone).split() == [
            "refs/heads/main", "refs/remotes/origin/main"]  # main alone
        assert _git("status", "--porcelain", cwd=clone) == ""
        assert _git("remote", "get-url", "origin", cwd=clone) == f"https://github.com/{REPO}.git"
        config = _git("config", "--list", "--local", cwd=clone)
        assert "extraheader" not in config and "credential" not in config
        assert transport.fetch_main(clone) == second  # a clone has the HEAD fetch_main needs
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and BASIC not in text, path
    assert not [folder for folder in made if folder.exists()]  # each private copy is gone
    carrying = [argv for argv, env in calls if _carries(argv, env)]
    assert len(carrying) == 4 and {_subcommand(argv) for argv in carrying} == {"fetch"}
    assert all(argv[argv.index("--git-dir") + 1].startswith(tuple(map(str, made))) for argv in carrying)

    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("kept\n")
    tokens, gits = len(github["token_requests"]), len(calls)
    for occupied in (full, tmp_path / "missing" / "clone", full / "keep.txt"):
        with pytest.raises(mod.GitHubTransportError) as refused:
            mod.GitHubTransport("read_checks", REPO, key_path=github["key_file"]).clone_main(occupied)
        assert refused.value.reason == "clone_target_not_empty"
    assert [path.name for path in full.iterdir()] == ["keep.txt"] and (full / "keep.txt").read_text() == "kept\n"
    assert len(github["token_requests"]) == tokens and len(calls) == gits  # refused before a token or a git
    with pytest.raises(mod.GitHubTransportError) as denied:
        mod.GitHubTransport("handoff", REPO, key_path=github["key_file"]).clone_main(tmp_path / "other")
    assert denied.value.reason == "step_not_permitted" and not (tmp_path / "other").exists()
