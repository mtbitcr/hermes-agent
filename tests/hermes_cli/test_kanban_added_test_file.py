"""Tests for the added-test-file check: one path, nine ordered checks, a fixed code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_added_test_file as mod
from hermes_cli import kanban_db as kb
from hermes_cli.owner_text import check_owner_text
from hermes_constants import get_hermes_home

BLOB = "ab" * 20
BASE = "c0ffee" + "0" * 34
WORKTREE = "/work/tree"
TEST_PATH = "tests/hermes_cli/test_thing.py"
OWN = ("hermes_cli/widget.py",)
BAD_PATH = "tests/../x.py"
CODES = (
    "unknown_card",
    "no_write_scope",
    "whole_repository",
    "not_canonical",
    "not_test_file",
    "already_owned",
    "second_addition",
    "held_by_open_card",
    "not_at_base",
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "store" / "kanban.db"
    db.parent.mkdir()
    kb.init_db(db_path=db)
    c = kb.connect(db)
    yield c
    c.close()


def db_file(c) -> Path:
    return Path(c.execute("PRAGMA database_list").fetchone()[2])


def add_card(c, task_id="t1", owned=OWN, *, raw_owned=None, project="proj", status="running",
             base=BASE, worktree=WORKTREE):
    stored = raw_owned if raw_owned is not None else (None if owned is None else json.dumps(list(owned)))
    with kb.write_txn(c):
        c.execute(
            "INSERT INTO tasks (id, title, status, created_at, project_id, owned_paths, base_commit,"
            " workspace_path) VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
            (task_id, "card " + task_id, status, project, stored, base, worktree),
        )
    return task_id


def add_event(c, task_id="t1", kind="owned_path_added"):
    with kb.write_txn(c):
        c.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, NULL, 1)",
            (task_id, kind),
        )


class Reader:
    def __init__(self, result=("100644", BLOB)):
        self.result = result
        self.calls = []

    def __call__(self, worktree, base_commit, path):
        self.calls.append((worktree, base_commit, path))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def never_read():
    return Reader(AssertionError("the reader must not run"))


def check(c, path=TEST_PATH, *, task_id="t1", reader=None):
    return mod.check_added_test_file(c, task_id, path, Reader() if reader is None else reader)


def holder(c, owned=("tests",), status="ready", project="proj", task_id="t2"):
    add_card(c, task_id, owned, project=project, status=status)


def _unknown(c):
    return "nope", TEST_PATH, never_read()


def _no_write_scope(c):
    add_card(c, owned=[])
    return "t1", TEST_PATH, never_read()


def _whole_repository(c):
    add_card(c, owned=None)
    return "t1", TEST_PATH, never_read()


def _not_canonical(c):
    add_card(c)
    return "t1", "tests/../tests/test_a.py", never_read()


def _not_test_file(c):
    add_card(c)
    return "t1", "tests/hermes_cli/helpers.py", never_read()


def _already_owned(c):
    add_card(c, owned=["tests/hermes_cli"])
    return "t1", TEST_PATH, never_read()


def _second_addition(c):
    add_card(c)
    add_event(c)
    return "t1", TEST_PATH, never_read()


def _held(c):
    add_card(c)
    holder(c, ["tests/hermes_cli"])
    return "t1", TEST_PATH, never_read()


def _not_at_base(c):
    add_card(c)
    return "t1", TEST_PATH, Reader(None)


def _ok(c):
    add_card(c)
    return "t1", TEST_PATH, Reader()


SCENARIOS = {
    "unknown_card": _unknown,
    "no_write_scope": _no_write_scope,
    "whole_repository": _whole_repository,
    "not_canonical": _not_canonical,
    "not_test_file": _not_test_file,
    "already_owned": _already_owned,
    "second_addition": _second_addition,
    "held_by_open_card": _held,
    "not_at_base": _not_at_base,
    "ok": _ok,
}


class Repo:
    def __init__(self, work, env, root):
        self.work = work
        self.env = env
        self.root = root
        self.base = self.head = None

    def git(self, *args, stdin=None):
        done = subprocess.run(["git", "-C", str(self.work), *args], env=self.env, input=stdin,
                              capture_output=True)
        assert done.returncode == 0, done.stderr
        return done.stdout.decode()

    def put(self, path, data=b"", mode="100644", oid=None):
        oid = oid or self.git("hash-object", "-w", "--stdin", stdin=data).strip()
        self.git("update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}")

    def commit(self, message, parent=None):
        tree = self.git("write-tree").strip()
        extra = ["-p", parent] if parent else []
        oid = self.git("commit-tree", tree, "-m", message, *extra).strip()
        self.git("update-ref", "HEAD", oid)
        return oid

    def oracle(self, path, rev=None):
        return self.git("rev-parse", f"{rev or self.base}:{path}").strip()


BASE_FILES = {
    "tests/test_alpha.py": b"alpha\n",
    "tests/hermes_cli/test_thing.py": b"thing\n",
    "tests/contracts/owner_payloads/example.json": b"{}\n",
    "tests/test_empty.py": b"",
    "tests/test_nul.py": b"\x00\x01binary\xff",
    "tests/test_café.py": b"cafe\n",
    "tests/test_日本語.py": b"nihongo\n",
    'tests/test_q"uote.py': b"quote\n",
    "tests/test_line\u2028sep.py": b"separator\n",
    "tests/test_next\u0085line.py": b"next line\n",
    "tests/test_gone.py": b"gone\n",
    "tests/conftest.py": b"# shared\n",
    "tests/test_dir.py/inner.txt": b"inner\n",
}


SEPARATED = f"release{os.pathsep}stable"
LAYOUTS = pytest.mark.parametrize(
    "repo", [None, SEPARATED], indirect=True, ids=["plain layout", "below a separator-named folder"]
)


@pytest.fixture
def repo(tmp_path, monkeypatch, request):
    """The repository; an indirect parameter names the folder below tmp_path that holds the repository folder."""
    holder = tmp_path / request.param if getattr(request, "param", None) else tmp_path
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "xdg"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="A U Thor",
        GIT_AUTHOR_EMAIL="author@example.invalid",
        GIT_AUTHOR_DATE="2026-01-01T00:00:00+0000",
        GIT_COMMITTER_NAME="C O Mitter",
        GIT_COMMITTER_EMAIL="committer@example.invalid",
        GIT_COMMITTER_DATE="2026-01-01T00:00:00+0000",
    )
    work = holder / "wörk tree 日本"
    work.mkdir(parents=True)
    r = Repo(work, env, tmp_path)
    r.git("init", "-q")
    for path, data in BASE_FILES.items():
        r.put(path, data)
    r.put("tests/test_exec.py", b"#!/bin/sh\n", "100755")
    r.put("tests/test_link.py", b"test_alpha.py", "120000")
    r.put("tests/test_module.py", mode="160000", oid="1" * 40)
    r.base = r.commit("base")
    r.put("tests/test_alpha.py", b"alpha two\n")
    r.put("tests/test_new.py", b"new\n")
    r.git("update-index", "--force-remove", "tests/test_gone.py")
    r.head = r.commit("head", parent=r.base)
    return r


def tree_snapshot(root: Path):
    seen = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            stat = full.lstat()
            seen[str(full.relative_to(root))] = (
                hashlib.sha256(full.read_bytes()).hexdigest(), stat.st_size, stat.st_mtime_ns,
            )
        for name in dirnames:
            full = Path(dirpath) / name
            seen[str(full.relative_to(root)) + "/"] = full.lstat().st_mtime_ns
    return seen


# --- point 7: the function, its parameters and its frozen result ---------------------------


def test_p7_result_is_frozen_with_ok_code_and_blob(conn):
    add_card(conn)
    ok = check(conn)
    refused = check(conn, "tests/hermes_cli/helpers.py")
    assert (ok.ok, ok.code, ok.blob) == (True, "ok", BLOB)
    assert (refused.ok, refused.code, refused.blob) == (False, "not_test_file", None)
    assert isinstance(ok.ok, bool) and isinstance(ok.code, str) and isinstance(ok.blob, str)
    for field, value in (("ok", False), ("code", "x"), ("blob", None)):
        with pytest.raises(AttributeError):
            setattr(ok, field, value)


def test_p7_parameters_are_conn_task_id_path_and_read_base_entry(conn):
    add_card(conn)
    reader = Reader()
    got = mod.check_added_test_file(conn=conn, task_id="t1", path=TEST_PATH, read_base_entry=reader)
    assert got.ok
    assert reader.calls == [(WORKTREE, BASE, TEST_PATH)]


def test_p7_reads_tuple_rows_and_row_objects_alike(conn):
    add_card(conn)
    holder(conn, ["tests/hermes_cli"])
    add_card(conn, "t3", ["src/a.py"], project="other")
    plain = sqlite3.connect(db_file(conn))
    try:
        for c in (conn, plain):
            assert check(c).code == "held_by_open_card"
            assert check(c, "tests/x/test_y.py", task_id="t3").code == "ok"
    finally:
        plain.close()


# --- point 8: the default reader ---------------------------------------------------------


def fake_run(stdout=b"", returncode=0, raises=None):
    def run(argv, **kwargs):
        if raises is not None:
            raise raises
        out = stdout.decode() if kwargs.get("text") or kwargs.get("encoding") else stdout
        return subprocess.CompletedProcess(argv, returncode, stdout=out, stderr=b"")

    return run


LINE = f"100644 blob {BLOB}\ttests/test_alpha.py\n".encode()


def test_p8_default_reader_runs_git_ls_tree_as_an_argument_list_without_shell_in_30_seconds(repo, monkeypatch):
    blob = repo.oracle("tests/test_alpha.py")
    calls = []
    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    entry = mod._git_read_base_entry(str(repo.work), repo.base, "tests/test_alpha.py")
    assert entry == ("100644", blob)
    ((args, kwargs),) = calls
    argv = args[0] if args else kwargs["args"]
    assert argv == ["git", "-C", str(repo.work), "ls-tree", repo.base, "--", "tests/test_alpha.py"]
    assert not kwargs.get("shell")
    assert kwargs["timeout"] == 30


@LAYOUTS
def test_p8_default_reader_runs_rev_parse_before_ls_tree_only_when_the_parent_holds_the_path_separator(
    repo, monkeypatch
):
    separated = repo.work.parent != repo.root
    assert has_separator_parent(repo.work) == separated, "the layout decides the path separator"
    blob = repo.oracle("tests/test_alpha.py")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CEILING_DIRECTORIES", "GIT_TRACE2_EVENT"):
        monkeypatch.setenv(name, str(repo.root / "elsewhere"))
    calls = []
    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    assert mod._git_read_base_entry(str(repo.work), repo.base, "tests/test_alpha.py") == ("100644", blob)
    rev_parse = ["git", "-C", str(repo.work), "rev-parse", "--show-toplevel"]
    ls_tree = ["git", "-C", str(repo.work), "ls-tree", repo.base, "--", "tests/test_alpha.py"]
    assert [args[0] if args else kwargs["args"] for args, kwargs in calls] == (
        [rev_parse, ls_tree] if separated else [ls_tree]
    )
    expected_env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    expected_env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_TRACE2="0", GIT_TRACE2_EVENT="0", GIT_TRACE2_PERF="0")
    if not separated:
        expected_env["GIT_CEILING_DIRECTORIES"] = os.path.realpath(repo.root)
    for _args, kwargs in calls:
        assert not kwargs.get("shell")
        assert kwargs["timeout"] == 30
        assert kwargs["env"] == expected_env
        assert kwargs["capture_output"] and kwargs["encoding"] == "utf-8" and kwargs["errors"] == "surrogateescape"


def test_p8_default_reader_gives_mode_and_id_for_exactly_one_line_for_that_path(monkeypatch):
    monkeypatch.setattr(mod.subprocess, "run", fake_run(LINE))
    assert mod._git_read_base_entry("/w", BASE, "tests/test_alpha.py") == ("100644", BLOB)
    other_mode = LINE.replace(b"100644", b"100755")
    monkeypatch.setattr(mod.subprocess, "run", fake_run(other_mode))
    assert mod._git_read_base_entry("/w", BASE, "tests/test_alpha.py") == ("100755", BLOB)


OTHER_RESULTS = {
    "no line": fake_run(b""),
    "two lines": fake_run(LINE + LINE.replace(b"alpha", b"beta")),
    "a line for another path": fake_run(LINE.replace(b"alpha", b"alpha.bak")),
    "a non-zero exit": fake_run(LINE, returncode=128),
    "a time-out": fake_run(raises=subprocess.TimeoutExpired("git", 30)),
    "an os error": fake_run(raises=FileNotFoundError("git")),
    "a permission error": fake_run(raises=PermissionError("git")),
    "no tab": fake_run(f"100644 blob {BLOB} tests/test_alpha.py\n".encode()),
    "two fields": fake_run(b"100644 blob\ttests/test_alpha.py\n"),
    "a folder": fake_run(f"040000 tree {BLOB}\ttests/test_alpha.py\n".encode()),
    "a submodule": fake_run(f"160000 commit {BLOB}\ttests/test_alpha.py\n".encode()),
}


@pytest.mark.parametrize("run", list(OTHER_RESULTS.values()), ids=list(OTHER_RESULTS))
def test_p8_default_reader_gives_none_for_each_other_result(monkeypatch, run):
    monkeypatch.setattr(mod.subprocess, "run", run)
    assert mod._git_read_base_entry("/w", BASE, "tests/test_alpha.py") is None


def test_p8_default_reader_gives_none_when_git_cannot_start(repo, monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert mod._git_read_base_entry(str(repo.work), repo.base, "tests/test_alpha.py") is None


@pytest.mark.parametrize("which", ["worktree", "base_commit", "path"])
def test_p8_default_reader_gives_none_for_a_nul_in_an_argument(repo, which):
    args = {"worktree": str(repo.work), "base_commit": repo.base, "path": "tests/test_alpha.py"}
    args[which] += "\x00x"
    assert mod._git_read_base_entry(**args) is None


@pytest.mark.parametrize("where", ["missing", "not a repository"])
def test_p8_default_reader_gives_none_for_a_worktree_that_is_not_a_repository(repo, tmp_path, where):
    work = tmp_path / "nothing"
    if where != "missing":
        work.mkdir()
    assert mod._git_read_base_entry(str(work), repo.base, "tests/test_alpha.py") is None


def test_p8_default_reader_gives_none_for_a_base_commit_that_does_not_exist(repo):
    assert mod._git_read_base_entry(str(repo.work), "f" * 40, "tests/test_alpha.py") is None


def scripted_run(calls, first):
    """A fake runner: ``first`` answers rev-parse, and ls-tree gives LINE; every command is noted in ``calls``."""

    def run(argv, **kwargs):
        calls.append(argv)
        return (first if "rev-parse" in argv else fake_run(LINE))(argv, **kwargs)

    return run


FIRST_ANSWERS = [
    "a non-zero exit", "no output", "an enclosing folder", "a folder that does not exist", "a time-out", "an os error",
    "a NUL in the stored folder",
]


@pytest.mark.parametrize("answer", FIRST_ANSWERS)
def test_p8_default_reader_gives_none_and_never_starts_ls_tree_when_the_first_command_does_not_show_the_folder(
    tmp_path, monkeypatch, answer
):
    stored = tmp_path / SEPARATED / "checkout"
    stored.mkdir(parents=True)
    first = {
        "a non-zero exit": fake_run(f"{stored}\n".encode(), returncode=128),
        "no output": fake_run(b""),
        "an enclosing folder": fake_run(f"{stored.parent}\n".encode()),
        "a folder that does not exist": fake_run(f"{stored.parent / 'gone'}\n".encode()),
        "a time-out": fake_run(raises=subprocess.TimeoutExpired("git", 30)),
        "an os error": fake_run(raises=FileNotFoundError("git")),
        "a NUL in the stored folder": fake_run(f"{stored}\n".encode()),
    }[answer]
    worktree = str(stored) + ("\x00x" if answer == "a NUL in the stored folder" else "")
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", scripted_run(calls, first))
    assert mod._git_read_base_entry(worktree, BASE, "tests/test_alpha.py") is None
    assert [argv[3] for argv in calls] == ([] if "NUL" in answer else ["rev-parse"]), "ls-tree is never started"


@pytest.mark.parametrize("name", ["checkout", "checkout\n"], ids=["a plain name", "a name that ends in a newline"])
def test_p8_default_reader_runs_ls_tree_second_when_the_first_command_names_the_stored_folder(
    tmp_path, monkeypatch, name
):
    stored = tmp_path / SEPARATED / name
    stored.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", scripted_run(calls, fake_run(f"{stored}\n".encode())))
    assert mod._git_read_base_entry(str(stored), BASE, "tests/test_alpha.py") == ("100644", BLOB)
    assert [argv[3] for argv in calls] == ["rev-parse", "ls-tree"]


# --- point 9: one test for each check ------------------------------------------------------


def test_p9a_unknown_card_when_no_card_has_the_id(conn):
    add_card(conn, "t1")
    got = check(conn, task_id="t-other", reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "unknown_card", None)


@pytest.mark.parametrize("stored", ["[]", "[ ]"])
def test_p9b_no_write_scope_when_the_owned_paths_are_empty(conn, stored):
    add_card(conn, raw_owned=stored)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "no_write_scope", None)


@pytest.mark.parametrize("stored", [None, '["."]', '[ "." ]'], ids=["null", "dot", "spaced dot"])
def test_p9c_whole_repository_when_the_owned_paths_are_null_or_hold_a_dot(conn, stored):
    add_card(conn, owned=None, raw_owned=stored)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "whole_repository", None)


NOT_CANONICAL = [
    "../tests/test_a.py", "tests/../tests/test_a.py", "tests/test_*.py", "tests/test_?.py",
    "tests/[ab]/test_a.py", "tests/test_a.py/", "tests/", ".git", ".git/hooks/test_a.py",
    "/tests/test_a.py", "tests\\test_a.py", "./tests/test_a.py", "tests//test_a.py",
    "tests/./test_a.py", " tests/test_a.py", "tests/test_a.py ", "tests/test_a.py\n",
    "tests/test_a\x00.py", "tests/test_a\x1f.py", "tests/test_a\x7f.py", "tests:x/test_a.py",
    "", "   ", "tests/" + "a" * 520 + "/test_a.py", None, 0, b"tests/test_a.py",
]


@pytest.mark.parametrize("path", NOT_CANONICAL, ids=repr)
def test_p9d_not_canonical_when_the_path_is_not_a_canonical_relative_posix_path(conn, path):
    add_card(conn)
    got = check(conn, path, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "not_canonical", None)


NOT_TEST_FILES = [
    "tests/conftest.py", "tests/hermes_cli/conftest.py", "tests/__init__.py",
    "tests/hermes_cli/__init__.py", "tests/helpers.py", "tests/hermes_cli/helpers/util.py",
    "tests-js/test_a.py", "tests-js/a.test.ts", "hermes_cli/widget.py", "test_a.py",
    "scripts/test_run.py", "tests", ".", "Tests/test_a.py", "tests/test_a.txt", "tests/test_a.pyc",
    "tests/Test_a.py", "tests/testa.py", "tests/a_tests.py", "tests/test_a.py/inner.py",
    "tests/test_dir/helper.py", "tests/contracts/owner_payloads/x.JSON",
    "tests/contracts/owner_payloads/x.jsonl", "tests/contracts/owner_payloads/x.json.bak",
    "tests/contracts/owner_payloads_extra/x.json", "tests/contracts/owner_payloads",
    "tests/contracts/other/x.json", "tests/x.json", "contracts/owner_payloads/x.json",
    "owner_payloads/x.json", "tests-js/contracts/owner_payloads/x.json",
]


@pytest.mark.parametrize("path", NOT_TEST_FILES)
def test_p9e_not_test_file_for_a_path_that_is_not_one_of_the_two_forms(conn, path):
    add_card(conn)
    got = check(conn, path, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "not_test_file", None)


TEST_FILES = [
    "tests/test_a.py", TEST_PATH, "tests/deep/er/x_test.py", "tests/test_.py", "tests/_test.py",
    "tests/contracts/owner_payloads/example.json", "tests/contracts/owner_payloads/nested/example.json",
    "tests/contracts/owner_payloads/test_x.py",
]


@pytest.mark.parametrize("path", TEST_FILES)
def test_p9e_both_forms_are_admitted_by_the_glob_and_the_prefix_rule(conn, path):
    add_card(conn)
    reader = Reader()
    got = check(conn, path, reader=reader)
    assert (got.ok, got.code, got.blob) == (True, "ok", BLOB)
    assert reader.calls == [(WORKTREE, BASE, path)]


@pytest.mark.parametrize(
    "owned",
    [[TEST_PATH], ["tests/hermes_cli"], ["tests"], ["src/a.py", "tests/hermes_cli"]],
    ids=["the path", "its folder", "the top folder", "among others"],
)
def test_p9f_already_owned_for_an_owned_path_and_for_a_path_below_an_owned_folder(conn, owned):
    add_card(conn, owned=owned)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "already_owned", None)


def test_p9f_a_lookalike_prefix_does_not_own_the_path(conn):
    add_card(conn, owned=["tests/hermes_cl", "tests/hermes_cli/test_thing.p", "tests/hermes_cli/test_thing.py2"])
    assert check(conn).code == "ok"


def test_p9g_second_addition_for_a_card_with_an_owned_path_added_event(conn):
    add_card(conn)
    add_event(conn)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "second_addition", None)


def test_p9g_other_event_kinds_and_other_cards_events_do_not_count(conn):
    add_card(conn)
    add_card(conn, "t2", ["src/b.py"], project="other")
    add_event(conn, "t1", "claimed")
    add_event(conn, "t1", "owned_path_added_x")
    add_event(conn, "t2")
    assert check(conn).code == "ok"


@pytest.mark.parametrize("count, code", [(63, "ok"), (64, "second_addition"), (65, "whole_repository")])
def test_p9g_a_list_of_64_paths_is_a_second_addition(conn, count, code):
    add_card(conn, owned=[f"src/m{i:03d}.py" for i in range(count)])
    reader = Reader() if code == "ok" else never_read()
    assert check(conn, reader=reader).code == code


@pytest.mark.parametrize("status", ["triage", "todo", "ready", "running", "blocked"])
def test_p9h_held_by_an_open_card_of_the_same_project_that_owns_an_overlapping_path(conn, status):
    add_card(conn)
    holder(conn, ["tests/hermes_cli"], status=status)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "held_by_open_card", None)


@pytest.mark.parametrize("status", ["done", "archived"])
def test_p9h_a_done_or_archived_card_does_not_hold_even_with_a_whole_repository_scope(conn, status):
    add_card(conn)
    holder(conn, None, status=status)
    assert check(conn).code == "ok"


OVERLAPPING_SCOPES = {
    "the path": dict(owned=[TEST_PATH]),
    "its folder": dict(owned=["tests/hermes_cli"]),
    "the top folder": dict(owned=["tests"]),
    "a dot": dict(owned=["."]),
    "null": dict(owned=None),
    "a case variant": dict(owned=["Tests/Hermes_CLI"]),
    "a path below it": dict(owned=[TEST_PATH + "/inner.txt"]),
    "malformed json": dict(owned=None, raw_owned='{"x": 1}'),
    "not json": dict(owned=None, raw_owned="not json"),
    "65 paths": dict(owned=[f"src/m{i:03d}.py" for i in range(65)]),
    "among others": dict(owned=["src/a.py", "tests/hermes_cli"]),
}


@pytest.mark.parametrize("scope", list(OVERLAPPING_SCOPES.values()), ids=list(OVERLAPPING_SCOPES))
def test_p9h_overlap_is_the_kernel_rule_and_unreadable_scopes_fail_closed(conn, scope):
    add_card(conn)
    add_card(conn, "t2", status="ready", **scope)
    assert check(conn, reader=never_read()).code == "held_by_open_card"


def test_p9h_a_unicode_variant_of_the_path_overlaps(conn):
    add_card(conn)
    holder(conn, ["tests/test_cafe\u0301.py"])
    assert check(conn, "tests/test_café.py", reader=never_read()).code == "held_by_open_card"


@pytest.mark.parametrize(
    "scope",
    [["tests/hermes_cl"], ["tests/hermes_cli_other"], [TEST_PATH + "x"], ["src/other.py"], []],
    ids=["lookalike prefix", "lookalike folder", "longer name", "elsewhere", "read-only"],
)
def test_p9h_a_scope_that_does_not_overlap_does_not_hold(conn, scope):
    add_card(conn)
    holder(conn, scope)
    assert check(conn).code == "ok"


@pytest.mark.parametrize(
    "mine, theirs",
    [("other", "proj"), ("proj", "other"), ("proj", None), (None, "proj"), (None, None), ("", "")],
    ids=["other project", "other project 2", "holder without project", "card without project",
         "neither has a project", "empty project ids"],
)
def test_p9h_only_a_card_of_the_same_project_holds(conn, mine, theirs):
    add_card(conn, project=mine)
    holder(conn, ["tests"], project=theirs)
    assert check(conn).code == "ok"


def test_p9h_the_card_never_holds_against_itself(conn):
    add_card(conn, owned=["TESTS/hermes_cli/test_thing.py"])
    assert check(conn).code == "ok"


def test_p9h_one_open_holder_among_closed_ones_is_enough(conn):
    add_card(conn)
    holder(conn, ["tests"], status="done", task_id="t2")
    holder(conn, ["tests"], status="archived", task_id="t3")
    holder(conn, ["tests/hermes_cli"], status="blocked", task_id="t4")
    assert check(conn, reader=never_read()).code == "held_by_open_card"


@pytest.mark.parametrize(
    "entry",
    [("120000", BLOB), ("040000", BLOB), ("160000", BLOB), ("100755", BLOB), ("100664", BLOB), None],
    ids=["symbolic link", "folder", "submodule", "executable", "other mode", "no entry"],
)
def test_p9i_not_at_base_for_each_entry_that_is_not_a_plain_file(conn, entry):
    add_card(conn)
    reader = Reader(entry)
    got = check(conn, reader=reader)
    assert (got.ok, got.code, got.blob) == (False, "not_at_base", None)
    assert reader.calls == [(WORKTREE, BASE, TEST_PATH)]


@pytest.mark.parametrize(
    "base, worktree",
    [(None, WORKTREE), ("", WORKTREE), (BASE, None), (BASE, ""), (None, None)],
    ids=["no base", "empty base", "no worktree", "empty worktree", "neither"],
)
def test_p9i_a_card_without_a_base_commit_or_a_worktree_gives_not_at_base_and_no_read(conn, base, worktree):
    add_card(conn, base=base, worktree=worktree)
    got = check(conn, reader=never_read())
    assert (got.ok, got.code, got.blob) == (False, "not_at_base", None)


def test_p9_ok_when_every_check_passes_with_the_blob_the_reader_gave(conn):
    add_card(conn)
    other_blob = "12" * 20
    got = check(conn, reader=Reader(("100644", other_blob)))
    assert (got.ok, got.code, got.blob) == (True, "ok", other_blob)


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_p9_each_scenario_gives_its_code_and_a_none_blob_unless_ok(conn, code):
    task_id, path, reader = SCENARIOS[code](conn)
    got = check(conn, path, task_id=task_id, reader=reader)
    assert got.code == code
    assert got.ok is (code == "ok")
    assert (got.blob is None) is (code != "ok")


# --- 17j: a path that fails two or more checks gives the code of the first ----------------


def _order_unknown(c):
    return "nope", BAD_PATH, never_read()


def _order_no_write_scope(c):
    add_card(c, owned=[])
    add_event(c)
    holder(c)
    return "t1", BAD_PATH, never_read()


def _order_whole_repository(c):
    add_card(c, owned=["."])
    add_event(c)
    holder(c)
    return "t1", BAD_PATH, never_read()


def _order_not_canonical(c):
    add_card(c, owned=["tests"])
    add_event(c)
    holder(c)
    return "t1", BAD_PATH, never_read()


def _order_not_test_file(c):
    add_card(c, owned=["tests"])
    add_event(c)
    holder(c)
    return "t1", "tests/helpers.py", never_read()


def _order_already_owned(c):
    add_card(c, owned=["tests"])
    add_event(c)
    holder(c)
    return "t1", TEST_PATH, never_read()


def _order_second_addition(c):
    add_card(c)
    add_event(c)
    holder(c)
    return "t1", TEST_PATH, never_read()


def _order_held(c):
    add_card(c)
    holder(c)
    return "t1", TEST_PATH, never_read()


ORDER = {
    "unknown_card": _order_unknown,
    "no_write_scope": _order_no_write_scope,
    "whole_repository": _order_whole_repository,
    "not_canonical": _order_not_canonical,
    "not_test_file": _order_not_test_file,
    "already_owned": _order_already_owned,
    "second_addition": _order_second_addition,
    "held_by_open_card": _order_held,
    "not_at_base": _not_at_base,
}


@pytest.mark.parametrize("code", CODES)
def test_17j_a_path_that_fails_two_or_more_checks_gives_the_first_and_runs_no_later_read(conn, code):
    task_id, path, reader = ORDER[code](conn)
    got = check(conn, path, task_id=task_id, reader=reader)
    assert (got.ok, got.code, got.blob) == (False, code, None)


# --- point 17 a-i: the cases as the owner listed them ----------------------------------------


def triple(result):
    return result.ok, result.code, result.blob


def test_17a_an_existing_test_file_below_tests_passes_and_gives_its_blob(conn):
    add_card(conn)
    reader = Reader(("100644", BLOB))
    assert triple(check(conn, "tests/hermes_cli/test_thing.py", reader=reader)) == (True, "ok", BLOB)
    assert reader.calls == [(WORKTREE, BASE, "tests/hermes_cli/test_thing.py")]


def test_17b_a_saved_example_json_below_owner_payloads_passes(conn):
    add_card(conn)
    path = "tests/contracts/owner_payloads/example.json"
    assert triple(check(conn, path)) == (True, "ok", BLOB)


@pytest.mark.parametrize(
    "path",
    ["tests/conftest.py", "tests/hermes_cli/conftest.py", "tests/__init__.py", "tests/hermes_cli/__init__.py",
     "tests/hermes_cli/helpers.py", "tests-js/test_a.py", "tests-js/a.test.ts", "hermes_cli/test_widget.py"],
)
def test_17c_conftest_package_file_helper_tests_js_and_outside_tests_give_not_test_file(conn, path):
    add_card(conn)
    assert triple(check(conn, path, reader=never_read())) == (False, "not_test_file", None)


@pytest.mark.parametrize(
    "entry",
    [("120000", BLOB), ("040000", BLOB), ("160000", BLOB), ("100755", BLOB), None],
    ids=["symbolic link", "folder", "submodule", "executable", "missing"],
)
def test_17d_a_link_folder_submodule_executable_and_missing_path_give_not_at_base(conn, entry):
    add_card(conn)
    assert triple(check(conn, reader=Reader(entry))) == (False, "not_at_base", None)


@pytest.mark.parametrize(
    "scope, code",
    [(dict(owned=[TEST_PATH]), "already_owned"), (dict(owned=["tests/hermes_cli"]), "already_owned"),
     (dict(owned=None), "whole_repository"), (dict(owned=["."]), "whole_repository")],
    ids=["owned path", "below an owned folder", "NULL", "dot"],
)
def test_17e_an_owned_path_a_path_below_an_owned_folder_null_and_dot_give_their_codes(conn, scope, code):
    add_card(conn, **scope)
    assert triple(check(conn, reader=never_read())) == (False, code, None)


def test_17f_a_card_with_one_owned_path_added_event_gives_second_addition(conn):
    add_card(conn)
    add_event(conn)
    assert triple(check(conn, reader=never_read())) == (False, "second_addition", None)


def test_17f_a_card_with_64_paths_gives_second_addition(conn):
    add_card(conn, owned=[f"src/m{i:02d}.py" for i in range(64)])
    assert triple(check(conn, reader=never_read())) == (False, "second_addition", None)


@pytest.mark.parametrize(
    "path", ["tests/../test_a.py", "tests/test_*.py", "tests/test_a.py/", "tests/hermes_cli/", ".git"]
)
def test_17g_dot_dot_glob_trailing_slash_and_dot_git_give_not_canonical(conn, path):
    add_card(conn)
    assert triple(check(conn, path, reader=never_read())) == (False, "not_canonical", None)


def test_17h_an_open_card_of_the_same_project_that_owns_an_overlapping_path_gives_held_by_open_card(conn):
    add_card(conn)
    holder(conn, ["tests/hermes_cli"], status="ready")
    assert triple(check(conn, reader=never_read())) == (False, "held_by_open_card", None)


def test_17h_a_done_card_that_owns_the_same_path_does_not_hold(conn):
    add_card(conn)
    holder(conn, [TEST_PATH], status="done")
    assert triple(check(conn)) == (True, "ok", BLOB)


def test_17i_a_card_with_an_empty_list_gives_no_write_scope(conn):
    add_card(conn, owned=[])
    assert triple(check(conn, reader=never_read())) == (False, "no_write_scope", None)


def test_17i_an_unknown_id_gives_unknown_card(conn):
    add_card(conn)
    assert triple(check(conn, task_id="t9", reader=never_read())) == (False, "unknown_card", None)


# --- point 10: the event kind is a constant and this card writes no such event --------------


def event_count(c, kind="owned_path_added"):
    return c.execute("SELECT COUNT(*) FROM task_events WHERE kind = ?", (kind,)).fetchone()[0]


def test_p10_the_event_kind_is_a_constant_of_the_module():
    assert mod.OWNED_PATH_ADDED_EVENT_KIND == "owned_path_added"


def test_p10_the_check_counts_events_of_that_kind_and_no_other(conn):
    add_card(conn)
    for kind in ("owned_path_added_x", "Owned_Path_Added", "owned_path_adde", "claimed", "owner_retry"):
        add_event(conn, "t1", kind)
    assert check(conn).code == "ok"
    add_event(conn, "t1", mod.OWNED_PATH_ADDED_EVENT_KIND)
    assert check(conn, reader=never_read()).code == "second_addition"


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_p10_no_call_writes_an_owned_path_added_event(conn, code):
    task_id, path, reader = SCENARIOS[code](conn)
    before = event_count(conn)
    check(conn, path, task_id=task_id, reader=reader)
    assert event_count(conn) == before


# --- point 11: the closed table of sentences, and 17k --------------------------------------


def test_p11_the_table_has_one_sentence_for_each_code_of_point_9_and_no_other():
    assert set(mod.SENTENCES) == {*CODES, "ok"}
    assert len(set(mod.SENTENCES.values())) == len(CODES) + 1


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_17k_each_sentence_passes_the_owner_text_check_unchanged_and_holds_no_slash(code):
    sentence = mod.SENTENCES[code]
    assert check_owner_text(sentence, "planner_prose") == sentence
    assert "/" not in sentence


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_p11_a_sentence_is_one_plain_sentence_that_names_no_path_id_or_commit(code):
    sentence = mod.SENTENCES[code]
    assert sentence == sentence.strip() and sentence[0].isupper()
    assert sentence.endswith(".") and sentence.count(".") == 1
    assert not any(ch.isdigit() or ch in "\\_:\"'`<>{}[]" for ch in sentence)
    assert not re.search(r"[0-9a-f]{7,}", sentence)
    assert code not in sentence and code.replace("_", " ") not in sentence


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_p11_every_code_the_function_gives_has_a_sentence(conn, code):
    task_id, path, reader = SCENARIOS[code](conn)
    got = check(conn, path, task_id=task_id, reader=reader)
    assert got.code == code
    assert mod.SENTENCES[got.code]


# --- point 12: no other check and no other form --------------------------------------------


@pytest.mark.parametrize("status", ["triage", "todo", "ready", "running", "blocked", "done", "archived"])
def test_p12_the_status_of_the_card_itself_is_not_checked(conn, status):
    add_card(conn, status=status)
    assert triple(check(conn)) == (True, "ok", BLOB)


@pytest.mark.parametrize("kind", ["work", "control", "recommendation"])
def test_p12_the_kind_of_the_card_itself_is_not_checked(conn, kind):
    add_card(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET task_kind = ? WHERE id = 't1'", (kind,))
    assert triple(check(conn)) == (True, "ok", BLOB)


@pytest.mark.parametrize(
    "path",
    ["tests/fixtures/data.json", "tests/contracts/owner_payloads/example.yaml", "tests/test_a.js",
     "tests/test_a.sh", "tests/a.test.py", "tests/snapshots/a.snap", "tests/__snapshots__/a.ambr",
     "scripts/run_tests.sh", "pytest.ini", "docs/test_a.py", "tests/README.md", "tests/data/sample.txt"],
)
def test_p12_no_other_form_is_admitted(conn, path):
    add_card(conn)
    assert triple(check(conn, path, reader=never_read())) == (False, "not_test_file", None)


# --- 17l: the store and the worktree are the same before and after each call ----------------


def store_state(c):
    db = db_file(c)
    return (
        hashlib.sha256(db.read_bytes()).hexdigest(),
        hashlib.sha256(c.serialize()).hexdigest(),
        c.total_changes,
        c.in_transaction,
        sorted(p.name for p in db.parent.iterdir()),
    )


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_17l_the_store_bytes_are_the_same_before_and_after_each_call(conn, code):
    task_id, path, reader = SCENARIOS[code](conn)
    before = store_state(conn)
    assert check(conn, path, task_id=task_id, reader=reader).code == code
    assert store_state(conn) == before


@pytest.mark.parametrize("code", [*CODES, "ok"])
def test_17l_a_connection_that_refuses_every_write_still_gets_every_answer(conn, code):
    task_id, path, reader = SCENARIOS[code](conn)
    conn.execute("PRAGMA query_only = ON")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM tasks")
    assert check(conn, path, task_id=task_id, reader=reader).code == code


def test_17l_a_read_only_file_connection_gets_the_answer(conn):
    add_card(conn)
    holder(conn, ["tests/hermes_cli"])
    ro = sqlite3.connect(f"file:{db_file(conn)}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM tasks")
        assert check(ro, reader=never_read()).code == "held_by_open_card"
        assert check(ro, "tests/other/test_b.py").code == "ok"
    finally:
        ro.close()


def real(c, path, task_id="t1"):
    return mod.check_added_test_file(c, task_id, path)


def real_card(c, repo, **kwargs):
    return add_card(c, base=repo.base, worktree=str(repo.work), **kwargs)


def has_separator_parent(stored) -> bool:
    return os.pathsep in os.path.dirname(os.path.realpath(stored))


TRACE_LOGS = ("trace2-normal.log", "trace2-event.log", "trace2-perf.log")


def configure_tracing_into_the_worktree(repo):
    """The temporary HOME names one Trace2 log inside the worktree for each of the three targets."""
    home = Path(os.environ["HOME"])
    assert home.is_relative_to(repo.root), "the fixture keeps HOME inside the temporary folder"
    for key, name in zip(("normalTarget", "eventTarget", "perfTarget"), TRACE_LOGS):
        repo.git("config", "--file", str(home / ".gitconfig"), f"trace2.{key}", str(repo.work / name))
    plain = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "-C", str(repo.work), "ls-tree", repo.base, "--", "tests/test_alpha.py"],
                   env=plain, check=True, capture_output=True)
    logs = [repo.work / name for name in TRACE_LOGS]
    assert all(log.is_file() and log.stat().st_size > 0 for log in logs), "a plain git read writes the logs"
    for log in logs:
        log.unlink()


@LAYOUTS
@pytest.mark.parametrize(
    "tracing", [False, True], ids=["no tracing configured", "tracing configured into the worktree"]
)
@pytest.mark.parametrize("path", ["tests/test_alpha.py", "tests/test_missing.py", "tests/test_exec.py"])
def test_17l_the_worktree_is_the_same_before_and_after_a_call_with_the_default_reader(
    conn, repo, path, tracing
):
    real_card(conn, repo)
    stray = repo.work / "tests" / "stray.txt"
    stray.parent.mkdir()
    stray.write_text("stray\n", encoding="utf-8")
    if tracing:
        configure_tracing_into_the_worktree(repo)
    before = tree_snapshot(repo.work)
    real(conn, path)
    assert tree_snapshot(repo.work) == before


# --- 17m: the default reader and the function, with a real temporary git repository ---------


def test_17m_the_default_reader_gives_mode_and_blob_for_a_file(repo):
    for path in ("tests/test_alpha.py", TEST_PATH, "tests/contracts/owner_payloads/example.json"):
        assert mod._git_read_base_entry(str(repo.work), repo.base, path) == ("100644", repo.oracle(path))


@LAYOUTS
@pytest.mark.parametrize("replaced", [False, True], ids=["no replacement ref", "a replacement ref for the base"])
def test_17m_the_default_reader_reads_the_base_commit_and_not_the_head(conn, repo, replaced):
    path = "tests/test_alpha.py"
    base_blob, head_blob, gone_blob = repo.oracle(path), repo.oracle(path, repo.head), repo.oracle("tests/test_gone.py")
    assert base_blob != head_blob
    if replaced:
        repo.git("replace", repo.base, repo.head)
        assert repo.oracle(path) == head_blob, "a plain git read of the base now answers from the replacement"
    assert mod._git_read_base_entry(str(repo.work), repo.base, path) == ("100644", base_blob)
    assert mod._git_read_base_entry(str(repo.work), repo.head, path) == ("100644", head_blob)
    assert mod._git_read_base_entry(str(repo.work), repo.base, "tests/test_gone.py") == ("100644", gone_blob)
    assert mod._git_read_base_entry(str(repo.work), repo.base, "tests/test_new.py") is None
    real_card(conn, repo)
    assert triple(real(conn, path)) == (True, "ok", base_blob)
    assert triple(real(conn, "tests/test_gone.py")) == (True, "ok", gone_blob)
    assert triple(real(conn, "tests/test_new.py")) == (False, "not_at_base", None)


@pytest.mark.parametrize("path", ["tests/test_missing.py", "tests/test_new.py", "nope", "tests/test_alpha.pyc"])
def test_17m_the_default_reader_gives_none_for_a_missing_path(repo, path):
    assert mod._git_read_base_entry(str(repo.work), repo.base, path) is None


@pytest.mark.parametrize(
    "path",
    ["tests", "tests/hermes_cli", "tests/hermes_cli/", "tests/contracts/owner_payloads", "tests/test_dir.py",
     "tests/test_dir.py/", "tests/test_module.py"],
)
def test_17m_the_default_reader_gives_none_for_a_folder(repo, path):
    assert mod._git_read_base_entry(str(repo.work), repo.base, path) is None


@pytest.mark.parametrize("path, mode", [("tests/test_exec.py", "100755"), ("tests/test_link.py", "120000")])
def test_17m_the_default_reader_gives_the_mode_of_an_executable_file_and_of_a_link(repo, path, mode):
    assert mod._git_read_base_entry(str(repo.work), repo.base, path) == (mode, repo.oracle(path))


@pytest.mark.parametrize(
    "path",
    ["tests/test_café.py", "tests/test_日本語.py", "tests/test_line\u2028sep.py", "tests/test_next\u0085line.py"],
    ids=["accent", "japanese", "line separator", "next line"],
)
def test_17m_the_default_reader_reads_a_name_that_git_prints_as_it_is(repo, path):
    repo.git("config", "core.quotePath", "false")
    assert mod._git_read_base_entry(str(repo.work), repo.base, path) == ("100644", repo.oracle(path))


@pytest.mark.parametrize(
    "path",
    ["tests/test_alpha.py", TEST_PATH, "tests/contracts/owner_payloads/example.json", "tests/test_empty.py",
     "tests/test_nul.py", "tests/test_café.py", "tests/test_日本語.py", 'tests/test_q"uote.py',
     "tests/test_line\u2028sep.py", "tests/test_next\u0085line.py", "tests/test_gone.py"],
    ids=["file", "nested file", "owner payload", "empty file", "binary file", "accent", "japanese",
         "double quote", "line separator", "next line", "removed after the base"],
)
def test_17m_the_function_gives_ok_and_the_blob_of_the_card_base(conn, repo, path):
    real_card(conn, repo)
    assert triple(real(conn, path)) == (True, "ok", repo.oracle(path))


@pytest.mark.parametrize(
    "path",
    ["tests/test_exec.py", "tests/test_link.py", "tests/test_module.py", "tests/test_dir.py",
     "tests/test_missing.py", "tests/test_new.py", "tests/test_cafe\u0301.py"],
    ids=["executable", "link", "submodule", "folder", "missing", "new after the base", "other unicode form"],
)
def test_17m_the_function_refuses_a_path_that_is_not_a_plain_file_of_the_base(conn, repo, path):
    real_card(conn, repo)
    got = real(conn, path)
    assert (got.ok, got.blob) == (False, None)
    assert got.code == "not_at_base"


def test_17m_a_linked_worktree_of_the_card_reads_the_same_base(conn, repo, tmp_path):
    linked = tmp_path / "linked"
    repo.git("worktree", "add", "-q", "--detach", str(linked), repo.base)
    add_card(conn, base=repo.base, worktree=str(linked))
    before = tree_snapshot(linked)
    assert triple(real(conn, "tests/test_alpha.py")) == (True, "ok", repo.oracle("tests/test_alpha.py"))
    assert triple(real(conn, "tests/test_new.py")) == (False, "not_at_base", None)
    assert tree_snapshot(linked) == before


NESTED_FOLDERS = ("pkg", f"a{os.pathsep}b/pkg", "日本 é/pkg")


@pytest.fixture
def nested(repo):
    """The repository with an ordinary folder for each of NESTED_FOLDERS; the base holds a test file below it."""
    for folder in NESTED_FOLDERS:
        repo.put(f"{folder}/tests/test_nested.py", f"nested in {folder}\n".encode())
        (repo.work / folder).mkdir(parents=True)
    repo.head = repo.commit("nested test files", parent=repo.head)
    return repo


@LAYOUTS
def test_17m_an_ordinary_folder_inside_a_repository_is_not_a_checkout_at_the_stored_location(conn, nested):
    stored = nested.work / "pkg"
    assert has_separator_parent(stored) == (nested.work.parent != nested.root), "the layout decides the separator"
    card = kb.create_task(
        conn, title="folder", owned_paths=["hermes_cli/widget.py"], workspace_kind="worktree",
        workspace_path=str(stored), initial_status="running",
    )
    resolved = kb.resolve_workspace(kb.get_task(conn, card))
    assert resolved == stored and not (stored / ".git").exists(), "the kernel takes the ordinary folder as the worktree"
    base = kb.record_worktree_base(conn, card, resolved)
    assert base == nested.head, "the kernel records the base of the enclosing repository for an ordinary folder"
    assert nested.oracle("pkg/tests/test_nested.py", base), "the nested file is in the base below the folder"
    assert nested.git("ls-tree", base, "--", "tests/test_nested.py") == "", "the proposed path is not in the base"
    assert triple(mod.check_added_test_file(conn, card, "tests/test_nested.py")) == (False, "not_at_base", None)
    assert mod._git_read_base_entry(str(stored), base, "tests/test_nested.py") is None


@LAYOUTS
def test_17m_a_linked_worktree_made_by_the_kernel_inside_the_main_checkout_is_a_checkout(conn, repo):
    card = kb.create_task(
        conn, title="linked", owned_paths=["hermes_cli/widget.py"], workspace_kind="worktree",
        workspace_path=str(repo.work), initial_status="running",
    )
    stored = kb.resolve_workspace(kb.get_task(conn, card))
    assert stored != repo.work and stored.is_relative_to(repo.work) and (stored / ".git").is_file()
    assert has_separator_parent(stored) == (repo.work.parent != repo.root), "the layout decides the path separator"
    base = kb.record_worktree_base(conn, card, stored)
    assert base == repo.head
    branch = Repo(stored, repo.env, repo.root)
    branch.put("tests/test_after_base.py", b"after the base\n")
    branch.commit("after the base", parent=base)
    assert branch.oracle("tests/test_after_base.py", "HEAD"), "the card's branch holds a file that the base does not"
    before = tree_snapshot(stored)
    assert triple(real(conn, "tests/test_new.py", card)) == (True, "ok", repo.oracle("tests/test_new.py", repo.head))
    assert triple(real(conn, "tests/test_alpha.py", card)) == (True, "ok", repo.oracle("tests/test_alpha.py", repo.head))
    assert triple(real(conn, "tests/test_gone.py", card)) == (False, "not_at_base", None)
    assert triple(real(conn, "tests/test_missing.py", card)) == (False, "not_at_base", None)
    assert triple(real(conn, "tests/test_after_base.py", card)) == (False, "not_at_base", None)
    assert tree_snapshot(stored) == before


@pytest.mark.parametrize(
    "repo", [f"{SEPARATED}/リリース{os.pathsep}安定版 ünï"], indirect=True, ids=["a multibyte folder below it"]
)
def test_17m_a_checkout_below_a_multibyte_folder_named_with_the_path_separator_gives_ok(conn, repo):
    assert has_separator_parent(repo.work) and not repo.work.parent.name.isascii()
    real_card(conn, repo)
    assert triple(real(conn, "tests/test_alpha.py")) == (True, "ok", repo.oracle("tests/test_alpha.py"))
    assert triple(real(conn, "tests/test_new.py")) == (False, "not_at_base", None)


# --- point 13: the limits the owner accepts --------------------------------------------------


def test_p13a_a_test_file_new_on_the_branch_and_not_in_the_base_is_refused(conn, repo):
    real_card(conn, repo)
    assert repo.oracle("tests/test_new.py", repo.head)
    assert triple(real(conn, "tests/test_new.py")) == (False, "not_at_base", None)


def test_p13b_the_check_does_not_read_the_test_content_or_the_disk(conn, repo):
    real_card(conn, repo)
    ondisk = repo.work / "tests" / "test_ondisk.py"
    ondisk.parent.mkdir()
    ondisk.write_text("def test_it():\n    assert True\n", encoding="utf-8")
    assert not (repo.work / "tests" / "test_empty.py").exists()
    assert triple(real(conn, "tests/test_empty.py")) == (True, "ok", repo.oracle("tests/test_empty.py"))
    assert triple(real(conn, "tests/test_nul.py")) == (True, "ok", repo.oracle("tests/test_nul.py"))
    assert triple(real(conn, "tests/test_ondisk.py")) == (False, "not_at_base", None)


@LAYOUTS
def test_p13c_git_variables_that_name_another_repository_do_not_move_the_read(conn, repo, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "-C", str(other), "init", "-q"], env=repo.env, check=True)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(other / ".git" / "objects"))
    real_card(conn, repo)
    assert triple(real(conn, "tests/test_alpha.py")) == (True, "ok", repo.oracle("tests/test_alpha.py"))
    assert triple(real(conn, "tests/test_other.py")) == (False, "not_at_base", None)


# --- the cases for each input and each read ------------------------------------------------


def test_input_an_empty_store_gives_unknown_card(conn):
    assert triple(check(conn, reader=never_read())) == (False, "unknown_card", None)


@pytest.mark.parametrize(
    "task_id", ["", "0", "-1", "T1", " t1", "t1 ", "t1\x00", "t\x001", "t-日本語", None], ids=repr
)
def test_input_a_task_id_that_names_no_card_gives_unknown_card(conn, task_id):
    add_card(conn)
    assert triple(check(conn, task_id=task_id, reader=never_read())) == (False, "unknown_card", None)


@pytest.mark.parametrize("task_id", ["0", "-1", "カード-日本語", "t\x001"], ids=repr)
def test_input_a_stored_card_id_of_zero_negative_multibyte_or_nul_text_is_found(conn, task_id):
    add_card(conn, task_id)
    assert triple(check(conn, task_id=task_id)) == (True, "ok", BLOB)


@pytest.mark.parametrize(
    "path, code",
    [("tests/test_日本語.py", "ok"), ("tests/日本語/test_a.py", "ok"), ("tests/test_a\x00.py", "not_canonical"),
     ("", "not_canonical"), (None, "not_canonical"), (0, "not_canonical"), (-1, "not_canonical"),
     ("0", "not_test_file"), ("-1", "not_test_file")],
    ids=repr,
)
def test_input_a_path_that_is_empty_missing_zero_negative_multibyte_or_has_a_nul(conn, path, code):
    add_card(conn)
    assert check(conn, path).code == code


@pytest.mark.parametrize("raw", ['["tests/日本語"]', '["tests/\\u65e5\\u672c\\u8a9e"]'], ids=["utf-8", "escaped"])
def test_input_multibyte_text_in_the_stored_scope_is_read(conn, raw):
    add_card(conn, raw_owned=raw)
    assert check(conn, "tests/日本語/test_a.py", reader=never_read()).code == "already_owned"
    assert check(conn, "tests/test_日本語.py").code == "ok"


UNREADABLE_SCOPES = [
    "", "null", "not json", "{}", '"tests"', "[1]", '[""]', '["../x"]', '["tests/a\\u0000b.py"]',
    "[" + ",".join(f'"src/m{i:02d}.py"' for i in range(65)) + "]",
]


@pytest.mark.parametrize("raw", UNREADABLE_SCOPES, ids=lambda raw: raw[:24])
def test_input_a_scope_text_the_kernel_cannot_read_is_the_whole_repository(conn, raw):
    add_card(conn, raw_owned=raw)
    assert triple(check(conn, reader=never_read())) == (False, "whole_repository", None)


def test_input_a_card_of_the_previous_version_with_no_project_base_commit_or_worktree(conn):
    add_card(conn, project=None, base=None, worktree=None)
    assert triple(check(conn, reader=never_read())) == (False, "not_at_base", None)
    add_card(conn, "t2", project=None, base=BASE, worktree=WORKTREE)
    assert triple(check(conn, task_id="t2")) == (True, "ok", BLOB)


def test_input_stored_base_commit_and_worktree_reach_the_reader_unchanged(conn):
    add_card(conn, base="bäse-日本\x00z", worktree="/work/日本 tree\x00z")
    reader = Reader()
    assert check(conn, reader=reader).ok
    assert reader.calls == [("/work/日本 tree\x00z", "bäse-日本\x00z", TEST_PATH)]


# --- the stored workspace: the input that the default reader takes its ceiling from ----------


def stored_answer(conn, stored, path, base):
    task_id = f"t{conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] + 1}"
    add_card(conn, task_id, base=base, worktree=None if stored is None else str(stored))
    return triple(real(conn, path, task_id))


def refused_at_base(conn, stored, path, base):
    assert stored_answer(conn, stored, path, base) == (False, "not_at_base", None)


def answered_ok(conn, repo, stored, path, base):
    assert stored_answer(conn, stored, path, base) == (True, "ok", repo.oracle(path, base))


@pytest.mark.parametrize("stored", ["", None], ids=["empty", "missing"])
def test_input_an_empty_or_missing_stored_workspace_gives_not_at_base(conn, repo, monkeypatch, stored):
    monkeypatch.chdir(repo.work)
    refused_at_base(conn, stored, "tests/test_alpha.py", repo.base)


def test_input_the_default_reader_gives_none_for_an_empty_workspace_and_does_not_read_the_current_folder(
    repo, monkeypatch
):
    monkeypatch.chdir(repo.work)
    assert mod._git_read_base_entry("", repo.base, "tests/test_alpha.py") is None


def a_file(path):
    path.write_text("not a folder\n", encoding="utf-8")
    return str(path)


MISSING_WORKSPACES = {
    "a folder that does not exist": lambda repo, tmp: str(tmp / "nothing"),
    "a folder below a folder that does not exist": lambda repo, tmp: str(tmp / "nothing" / "deeper"),
    "a folder that does not exist below a checkout": lambda repo, tmp: str(repo.work / "gone" / "deeper"),
    "a file": lambda repo, tmp: a_file(tmp / "a file"),
    "a path longer than the system allows": lambda repo, tmp: str(tmp) + ("/" + "d" * 100) * 50,
}


@pytest.mark.parametrize("make", list(MISSING_WORKSPACES.values()), ids=list(MISSING_WORKSPACES))
def test_input_a_stored_workspace_that_does_not_exist_or_is_not_a_folder_gives_not_at_base(
    conn, repo, tmp_path, make
):
    refused_at_base(conn, make(repo, tmp_path), "tests/test_alpha.py", repo.base)


@pytest.mark.parametrize("where", ["after the checkout", "between two folders", "alone"])
def test_input_a_nul_in_the_stored_workspace_gives_not_at_base_and_never_raises(conn, repo, where):
    stored = {
        "after the checkout": f"{repo.work}\x00",
        "between two folders": f"{repo.work.parent}\x00/{repo.work.name}",
        "alone": "\x00",
    }[where]
    refused_at_base(conn, stored, "tests/test_alpha.py", repo.base)


def test_input_a_stored_workspace_below_a_multibyte_folder_name_is_a_checkout_only_when_it_is_one(
    conn, nested, tmp_path
):
    linked = tmp_path / "日本語 é" / "linked"
    nested.git("worktree", "add", "-q", "--detach", str(linked), nested.base)
    answered_ok(conn, nested, linked, "tests/test_alpha.py", nested.base)
    refused_at_base(conn, linked, "tests/test_new.py", nested.base)
    refused_at_base(conn, nested.work / "日本 é" / "pkg", "tests/test_nested.py", nested.head)


def test_input_a_stored_workspace_below_a_folder_named_with_the_path_separator_is_a_checkout_only_when_it_is_one(
    conn, nested, tmp_path
):
    refused_at_base(conn, nested.work / NESTED_FOLDERS[1], "tests/test_nested.py", nested.head)
    separated = tmp_path / f"a{os.pathsep}b"
    nested.git("worktree", "add", "-q", "--detach", str(separated), nested.base)
    answered_ok(conn, nested, separated, "tests/test_alpha.py", nested.base)
    parent = tmp_path / SEPARATED / "linked"
    grandparent = tmp_path / SEPARATED / "inner" / "linked"
    for linked in (parent, grandparent):
        nested.git("worktree", "add", "-q", "--detach", str(linked), nested.base)
        assert has_separator_parent(linked)
        answered_ok(conn, nested, linked, "tests/test_alpha.py", nested.base)
        refused_at_base(conn, linked, "tests/test_new.py", nested.base)
        refused_at_base(conn, linked, "tests/test_missing.py", nested.base)
    assert os.pathsep not in grandparent.parent.name and os.pathsep in grandparent.parent.parent.name
    to_checkout = tmp_path / "link to the separated checkout"
    to_checkout.symlink_to(parent, target_is_directory=True)
    answered_ok(conn, nested, to_checkout, "tests/test_alpha.py", nested.base)
    refused_at_base(conn, to_checkout, "tests/test_new.py", nested.base)
    above = tmp_path / "link above the separated checkout"
    above.symlink_to(parent.parent, target_is_directory=True)
    answered_ok(conn, nested, above / "linked", "tests/test_alpha.py", nested.base)
    refused_at_base(conn, above / "linked", "tests/test_new.py", nested.base)
    to_folder = tmp_path / "link to the ordinary folder"
    to_folder.symlink_to(nested.work / NESTED_FOLDERS[1], target_is_directory=True)
    assert has_separator_parent(to_folder)
    refused_at_base(conn, to_folder, "tests/test_nested.py", nested.head)
    leaving = tmp_path / SEPARATED / "link to the plain checkout"
    leaving.symlink_to(nested.work, target_is_directory=True)
    assert not has_separator_parent(leaving) and os.pathsep in str(leaving)
    answered_ok(conn, nested, leaving, "tests/test_alpha.py", nested.base)
    refused_at_base(conn, leaving, "tests/test_new.py", nested.base)


def test_input_a_symbolic_link_as_the_stored_workspace_is_followed_to_where_it_leads(conn, nested, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(nested.work / "pkg", target_is_directory=True)
    refused_at_base(conn, link, "tests/test_nested.py", nested.head)
    above = tmp_path / "link above"
    above.symlink_to(nested.work, target_is_directory=True)
    refused_at_base(conn, above / "pkg", "tests/test_nested.py", nested.head)
    answered_ok(conn, nested, above, "tests/test_alpha.py", nested.base)
    linked = tmp_path / "linked"
    nested.git("worktree", "add", "-q", "--detach", str(linked), nested.base)
    to_linked = tmp_path / "link to the worktree"
    to_linked.symlink_to(linked, target_is_directory=True)
    answered_ok(conn, nested, to_linked, "tests/test_alpha.py", nested.base)
    refused_at_base(conn, to_linked, "tests/test_new.py", nested.base)


def test_input_a_stored_workspace_that_is_not_in_canonical_form_is_read_where_it_leads(conn, nested):
    work = str(nested.work)
    for stored in (work + "/", work + "/.", f"{nested.work.parent}//{nested.work.name}", work + "/pkg/.."):
        answered_ok(conn, nested, stored, "tests/test_alpha.py", nested.base)
        refused_at_base(conn, stored, "tests/test_new.py", nested.base)
    for stored in (work + "/pkg/", work + "/pkg/.", work + "/pkg/../pkg", work + "//pkg"):
        refused_at_base(conn, stored, "tests/test_nested.py", nested.head)


def test_input_a_relative_stored_workspace_is_read_from_the_current_folder(conn, nested, monkeypatch):
    monkeypatch.chdir(nested.work.parent)
    name = nested.work.name
    for stored in (name, f"./{name}", f"{name}/pkg/.."):
        answered_ok(conn, nested, stored, "tests/test_alpha.py", nested.base)
        refused_at_base(conn, stored, "tests/test_new.py", nested.base)
    refused_at_base(conn, f"{name}/pkg", "tests/test_nested.py", nested.head)


def test_input_a_closed_store_raises_and_gives_no_answer(conn):
    add_card(conn)
    conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        check(conn, reader=never_read())


def test_input_a_store_without_the_tasks_table_raises(tmp_path):
    bare = sqlite3.connect(tmp_path / "bare.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            check(bare, reader=never_read())
    finally:
        bare.close()


def test_input_a_store_without_the_events_table_raises_where_the_events_are_read(tmp_path):
    bare = sqlite3.connect(tmp_path / "bare.db")
    try:
        bare.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, project_id TEXT, owned_paths TEXT,"
            " base_commit TEXT, workspace_path TEXT)"
        )
        bare.execute(
            "INSERT INTO tasks VALUES ('t1', 'running', 'proj', '[\"hermes_cli/widget.py\"]', ?, ?)",
            (BASE, WORKTREE),
        )
        assert check(bare, "tests/helpers.py", reader=never_read()).code == "not_test_file"
        with pytest.raises(sqlite3.OperationalError):
            check(bare, reader=never_read())
    finally:
        bare.close()


@pytest.mark.parametrize(
    "error", [OSError("boom"), RuntimeError("boom"), subprocess.TimeoutExpired("git", 30)],
    ids=lambda error: type(error).__name__,
)
def test_input_a_reader_that_raises_is_not_turned_into_an_answer(conn, error):
    add_card(conn)
    with pytest.raises(type(error)):
        check(conn, reader=Reader(error))


# --- the store path under the dispatcher worker env -----------------------------------------


def raw_rows(db, sql, *args):
    c = sqlite3.connect(db)
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def raw_exec(db, sql, *args):
    c = sqlite3.connect(db)
    try:
        with c:
            c.execute(sql, args)
    finally:
        c.close()


def seed_board(db, repo, *, own):
    c = kb.connect(db)
    try:
        if own:
            add_card(c, "card-1", ["hermes_cli/widget.py"], base=repo.base, worktree=str(repo.work))
            add_card(c, "holder-1", ["tests/hermes_cli"], status="ready")
        else:
            add_card(c, "card-1", [TEST_PATH])
    finally:
        c.close()


def pin_worker_env(monkeypatch, root, repo, own_db):
    profile = root / "profiles" / "builder"
    profile.mkdir(parents=True)
    worker = {
        "HERMES_HOME": str(profile),
        "HERMES_PROFILE": "builder",
        "HERMES_SESSION_SOURCE": "kanban",
        "HERMES_KANBAN_TASK": "card-1",
        "HERMES_KANBAN_RUN_ID": "1",
        "HERMES_KANBAN_CLAIM_LOCK": "claim-lock",
        "HERMES_KANBAN_BRANCH": "kanban/card-1",
        "HERMES_KANBAN_WORKSPACE": str(repo.work),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(root / "kanban" / "workspaces"),
        "HERMES_KANBAN_DB": str(own_db),
        "HERMES_KANBAN_BOARD": "alpha",
        "TERMINAL_CWD": str(repo.work),
    }
    for name, value in worker.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("board", ["alpha", "beta"], ids=["the worker's own board", "another board"])
def test_store_path_the_check_reaches_the_card_store_and_git_under_the_dispatcher_worker_env(
    repo, monkeypatch, board
):
    root = get_hermes_home()
    dbs = {}
    for slug in ("alpha", "beta"):
        dbs[slug] = kb.kanban_db_path(board=slug)
        dbs[slug].parent.mkdir(parents=True, exist_ok=True)
        kb.init_db(db_path=dbs[slug])
        seed_board(dbs[slug], repo, own=slug == board)
    decoy = dbs["beta" if board == "alpha" else "alpha"]
    pin_worker_env(monkeypatch, root, repo, dbs["alpha"])
    assert kb.kanban_db_path(board="beta") == dbs["alpha"], "the worker env pins every board to its own store"
    assert get_hermes_home() != root

    store = dbs[board]
    open_holders = "SELECT COUNT(*) FROM tasks WHERE project_id = 'proj' AND id != 'card-1' AND status NOT IN ('done', 'archived')"
    assert raw_rows(store, open_holders) == [(1,)]
    assert raw_rows(decoy, open_holders) == [(0,)]
    conn = kb.connect(store)
    decoy_conn = kb.connect(decoy)
    try:
        assert real(conn, TEST_PATH, "card-1").code == "held_by_open_card"
        assert real(decoy_conn, TEST_PATH, "card-1").code == "already_owned"

        raw_exec(store, "UPDATE tasks SET status = 'done' WHERE id = 'holder-1'")
        assert raw_rows(store, open_holders) == [(0,)]
        listed = subprocess.run(
            ["git", "-C", str(repo.work), "ls-tree", repo.base, "--", TEST_PATH],
            capture_output=True, text=True, env=repo.env, check=True,
        ).stdout.splitlines()
        assert len(listed) == 1
        got = real(conn, TEST_PATH, "card-1")
        assert (got.ok, got.code, got.blob) == (True, "ok", listed[0].split()[2])

        raw_exec(
            store, "INSERT INTO task_events (task_id, kind, payload, created_at)"
            " VALUES ('card-1', 'owned_path_added', NULL, 1)",
        )
        assert raw_rows(store, "SELECT COUNT(*) FROM task_events WHERE kind = 'owned_path_added'") == [(1,)]
        assert raw_rows(decoy, "SELECT COUNT(*) FROM task_events WHERE kind = 'owned_path_added'") == [(0,)]
        assert real(conn, TEST_PATH, "card-1").code == "second_addition"
        assert real(decoy_conn, TEST_PATH, "card-1").code == "already_owned"
    finally:
        conn.close()
        decoy_conn.close()


# --- end to end with cards made by the kernel and a real worktree --------------------------


def test_e2e_cards_made_by_the_kernel_and_a_real_worktree(conn, repo):
    mine = kb.create_task(
        conn, title="mine", owned_paths=["hermes_cli/widget.py"], workspace_kind="worktree",
        workspace_path=str(repo.work), initial_status="running",
    )
    other_tree = repo.work.parent / "other tree"
    other_tree.mkdir()
    other = kb.create_task(
        conn, title="other", owned_paths=["tests/hermes_cli"], workspace_kind="worktree",
        workspace_path=str(other_tree), initial_status="running",
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET project_id = 'proj', base_commit = ? WHERE id = ?", (repo.base, mine))
        conn.execute("UPDATE tasks SET project_id = 'proj' WHERE id = ?", (other,))
    assert mod.check_added_test_file(conn, mine, TEST_PATH).code == "held_by_open_card"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (other,))
    got = mod.check_added_test_file(conn, mine, TEST_PATH)
    assert (got.ok, got.code, got.blob) == (True, "ok", repo.oracle(TEST_PATH))
    assert mod.check_added_test_file(conn, mine, "tests/test_new.py").code == "not_at_base"
    assert mod.check_added_test_file(conn, mine, "hermes_cli/widget.py").code == "not_test_file"
