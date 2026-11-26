"""check_patch says whether a card's patch applies on a repository, and writes nothing.

Every test builds a temporary git repository under ``tmp_path``; nothing reads the real
repository or the network.
"""

from __future__ import annotations

import hashlib
import os
import runpy
import subprocess
import types
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "hermes_cli" / "kanban_scope_patch.py"
scope_patch = types.SimpleNamespace(**runpy.run_path(str(_MODULE_PATH)))


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return done.stdout


def _snapshot(root: Path) -> dict[str, str | None]:
    """Every path under root, .git included, with a digest of its bytes."""
    seen: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            seen[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return seen


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "base")
    return root


@pytest.fixture
def good_patch(repo, tmp_path):
    """A patch made against the repository's current content, kept outside the repository."""
    (repo / "a.txt").write_text("one\nTWO\nthree\n", encoding="utf-8")
    (repo / "new.txt").write_text("fresh\n", encoding="utf-8")
    _git(repo, "add", "-N", "new.txt")
    patch = tmp_path / "good.patch"
    patch.write_text(_git(repo, "diff"), encoding="utf-8")
    _git(repo, "reset", "-q")
    (repo / "new.txt").unlink()
    _git(repo, "checkout", "--", "a.txt")
    return patch


def test_a_good_patch_applies(repo, good_patch):
    assert scope_patch.check_patch(repo, good_patch) == "applies"


def test_a_patch_made_against_other_content_is_refused_with_git_text(repo, good_patch):
    (repo / "a.txt").write_text("x\ny\nz\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "other content")

    answer = scope_patch.check_patch(repo, good_patch)

    assert answer != "applies"
    assert "a.txt: patch does not apply" in answer


def test_a_patch_for_a_file_that_is_not_there_is_refused(repo, good_patch):
    _git(repo, "rm", "-q", "a.txt")
    _git(repo, "commit", "-q", "-m", "gone")

    assert scope_patch.check_patch(repo, good_patch) != "applies"


def test_a_missing_patch_file_is_a_refusal_not_an_error(repo, tmp_path):
    answer = scope_patch.check_patch(repo, tmp_path / "nope.patch")

    assert answer != "applies"
    assert answer


def test_a_relative_patch_path_is_read_from_the_callers_directory(repo, good_patch, monkeypatch):
    monkeypatch.chdir(good_patch.parent)

    assert scope_patch.check_patch(repo, good_patch.name) == "applies"


def test_the_root_and_patch_may_be_given_as_text(repo, good_patch):
    assert scope_patch.check_patch(str(repo), str(good_patch)) == "applies"


def test_no_byte_of_the_repository_changes_when_the_patch_applies(repo, good_patch):
    before = _snapshot(repo)

    assert scope_patch.check_patch(repo, good_patch) == "applies"

    assert _snapshot(repo) == before
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\ntwo\nthree\n"
    assert not (repo / "new.txt").exists()


def test_no_byte_of_the_repository_changes_when_the_patch_is_refused(repo, good_patch):
    (repo / "a.txt").write_text("x\ny\nz\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "other content")
    before = _snapshot(repo)

    assert scope_patch.check_patch(repo, good_patch) != "applies"

    assert _snapshot(repo) == before
