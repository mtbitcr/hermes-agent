"""The scope hints search lists the tests and readers of the paths a proposed card will own.

Every test builds a temporary repository tree under ``tmp_path``; nothing reads the
real repository or the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_scope_hints as hints

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "kanban_scope_hints.py"

OWNED = "hermes_cli/widget.py"
KEY = "gadget_mode"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """A small repository: one owned module, its pins and readers, and traps."""
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    _write(root / ".git" / "config", f"[x]\n\tpath = {OWNED}\n\tkey = '{KEY}'\n")
    _write(root / "node_modules" / "dep.py", "from hermes_cli.widget import build\n")
    _write(root / "hermes_cli" / "__init__.py", "")
    _write(root / OWNED, f"def build():\n    return '{KEY}'\n")
    _write(root / "hermes_cli" / "reader.py", "from hermes_cli.widget import build\n")
    _write(root / "hermes_cli" / "unrelated.py", "import os\n")
    _write(root / "tests" / "__init__.py", "")
    _write(root / "tests" / "test_pin_key.py",
           f"import hermes_cli.widget\n\ndef test_it():\n    assert x == \"{KEY}\"\n")
    _write(root / "tests" / "test_pin_plain.py", "from hermes_cli import widget\n")
    _write(root / "tests" / "test_unrelated.py", 'import os\nOTHER = "gadget_mode_extra"\n')
    _write(root / "tests" / "test_unquoted.py", f"# {KEY} is only named in a comment\n")
    _write(root / "scripts" / "tool.py", f"from hermes_cli.widget import build\nP = '{OWNED}'\n")
    _write(root / "docs" / "notes.md", f"See `{OWNED}`; the key is '{KEY}'.\n")
    _write(root / "config" / "settings.json", f'{{"{KEY}": true}}\n')
    # Files outside the repository, reached through symlinks inside it.
    _write(outside / "test_outside.py", f"from hermes_cli.widget import build\nK = '{KEY}'\n")
    _write(outside / "linked_dir" / "deep.py", f"import hermes_cli.widget\nK = '{KEY}'\n")
    (root / "tests" / "test_link.py").symlink_to(outside / "test_outside.py")
    (root / "tests" / "link_dir").symlink_to(outside / "linked_dir", target_is_directory=True)
    return root


def _paths(report) -> list[str]:
    return [h.path for h in report.hints]


def _rules(report) -> dict[str, tuple[int, ...]]:
    return {h.path: h.rules for h in report.hints}


def _output_paths(stdout: str) -> list[str]:
    return [line.split("\t")[0] for line in stdout.splitlines() if line and not line.startswith("#")]


def _load_script():
    spec = importlib.util.spec_from_file_location("kanban_scope_hints_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    saved = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


def _snapshot(root: Path) -> list[tuple[str, str, int]]:
    """Sorted path, sha256 and mtime_ns of every entry, the root and symlinks included."""
    entries = [(".", "dir", root.lstat().st_mtime_ns)]
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath, name)
            if path.is_symlink():
                digest = "link:" + os.readlink(path)
            elif path.is_dir():
                digest = "dir"
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((path.relative_to(root).as_posix(), digest, path.lstat().st_mtime_ns))
    return sorted(entries)


def test_test_file_importing_owned_module_is_listed_and_unrelated_one_is_not(repo):
    report = hints.find_hints(repo, [OWNED])
    listed = _paths(report)
    assert "tests/test_pin_key.py" in listed          # import hermes_cli.widget
    assert "tests/test_pin_plain.py" in listed        # from hermes_cli import widget
    assert "tests/test_unrelated.py" not in listed
    assert "tests/test_unquoted.py" not in listed
    assert _rules(report)["tests/test_pin_plain.py"] == (1,)
    # Nothing is read through a symlink that leaves the repository, nor under skipped folders.
    assert "tests/test_link.py" not in listed
    assert not any(p.startswith(("tests/link_dir", "node_modules", ".git")) for p in listed)

    # The same through the script, in a subprocess, with the root found from the current folder.
    env = {"PATH": os.environ.get("PATH", "")}
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--owned", OWNED],
        cwd=repo / "hermes_cli", env=env, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    printed = _output_paths(run.stdout)
    assert "tests/test_pin_key.py" in printed and "tests/test_pin_plain.py" in printed
    assert "tests/test_unrelated.py" not in printed


HELPER_LABEL = "rule 1 (test imports a pinning test)"


@pytest.fixture
def helper_repo(tmp_path):
    """tests/a.py pins the owned module; b imports a; c imports b; d imports both."""
    root = tmp_path / "helper_repo"
    _write(root / ".git" / "config", "")
    _write(root / OWNED, "def build():\n    return 1\n")
    _write(root / "tests" / "__init__.py", "")
    _write(root / "tests" / "a.py", "import hermes_cli.widget\n\ndef helper():\n    return 1\n")
    _write(root / "tests" / "b.py", "from tests.a import helper\n\ndef test_b():\n    helper()\n")
    _write(root / "tests" / "c.py", "from tests import b\n")
    _write(root / "tests" / "d.py", "import hermes_cli.widget\nfrom tests.a import helper\n")
    return root


def _lines(report) -> dict[str, str]:
    return {line.split("\t")[0]: line.split("\t")[1] for line in hints.format_report(report) if "\t" in line}


def test_test_importing_a_pinning_test_is_listed_with_its_own_label(helper_repo):
    report = hints.find_hints(helper_repo, [OWNED])
    lines = _lines(report)
    assert lines["tests/a.py"] == "rule 1 (test imports owned module)"
    assert lines["tests/b.py"] == HELPER_LABEL


def test_helper_step_is_one_step_only(helper_repo):
    assert "tests/c.py" not in _paths(hints.find_hints(helper_repo, [OWNED]))


def test_file_already_listed_directly_is_not_listed_twice(helper_repo):
    report = hints.find_hints(helper_repo, [OWNED])
    assert _paths(report).count("tests/d.py") == 1
    assert _lines(report)["tests/d.py"] == "rule 1 (test imports owned module)"
    assert _lines(report)["tests/a.py"] == "rule 1 (test imports owned module)"


def test_output_without_a_test_helper_is_unchanged(repo):
    report = hints.find_hints(repo, [OWNED], [KEY])
    assert HELPER_LABEL not in "\n".join(hints.format_report(report))
    assert not any(getattr(h, "imports_pinning_test", False) for h in report.hints)
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 3}
    assert _lines(report)["tests/test_pin_plain.py"] == "rule 1 (test imports owned module)"


def test_module_importing_name_from_owned_module_is_listed_as_reader(repo):
    report = hints.find_hints(repo, [OWNED])
    rules = _rules(report)
    assert rules["hermes_cli/reader.py"] == (2,)
    assert 2 in rules["scripts/tool.py"]
    assert "hermes_cli/unrelated.py" not in rules
    # A test file that imports the module is a pin (rule 1), not a reader.
    assert all(2 not in r for p, r in rules.items() if p.startswith("tests/"))

    # A relative import counts too.
    _write(repo / "hermes_cli" / "relative.py", "from .widget import build\n")
    assert _rules(hints.find_hints(repo, [OWNED]))["hermes_cli/relative.py"] == (2,)

    # An owned package __init__.py: the absolute and the relative import of a name from it
    # are both readers, although the relative one never spells the package name.
    _write(repo / "pkg" / "__init__.py", "def build():\n    return 1\n")
    _write(repo / "pkg" / "absolute.py", "from pkg import build\n")
    _write(repo / "pkg" / "reader.py", "from . import build\n")
    _write(repo / "pkg" / "sub" / "deep.py", "from .. import build\n")
    pkg = hints.find_hints(repo, ["pkg/__init__.py"])
    assert _rules(pkg) == {
        "pkg/absolute.py": (2,), "pkg/reader.py": (2,), "pkg/sub/deep.py": (2,),
    }
    assert pkg.counts[2] == 3


def test_already_owned_file_is_never_listed(repo):
    owned = [OWNED, "hermes_cli/reader.py", "tests/test_pin_key.py", "scripts"]
    report = hints.find_hints(repo, owned, [KEY])
    listed = _paths(report)
    for path in (OWNED, "hermes_cli/reader.py", "tests/test_pin_key.py", "scripts/tool.py"):
        assert path not in listed
    assert "tests/test_pin_plain.py" in listed
    assert "docs/notes.md" in listed


def test_key_lists_exactly_the_files_with_that_quoted_key_first_and_prints_counts(repo, capsys):
    # A text file whose quoted key comes after more than 2 MiB of whitespace is still found.
    _write(repo / "data" / "large.json", " " * (2 * 1024 * 1024 + 7) + f'{{"{KEY}": true}}\n')
    report = hints.find_hints(repo, [OWNED], [KEY])
    with_key = [h.path for h in report.hints if 4 in h.rules]
    assert sorted(with_key) == [
        "config/settings.json", "data/large.json", "docs/notes.md", "tests/test_pin_key.py",
    ]
    # Key matches come first, then the rest; each group sorted by path.
    assert _paths(report) == [
        "config/settings.json", "data/large.json", "docs/notes.md", "tests/test_pin_key.py",
        "hermes_cli/reader.py", "scripts/tool.py", "tests/test_pin_plain.py",
    ]
    # Every rule that matched is named, not only the first.
    assert _rules(report) == {
        "config/settings.json": (4,),
        "data/large.json": (4,),
        "docs/notes.md": (3, 4),
        "tests/test_pin_key.py": (1, 4),
        "hermes_cli/reader.py": (2,),
        "scripts/tool.py": (2, 3),
        "tests/test_pin_plain.py": (1,),
    }
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 4}

    # Through the script: same order, every rule on the line, one count per rule.
    script = _load_script()
    code = script.main(["--root", str(repo), "--owned", OWNED, "--key", KEY])
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == _paths(report)
    by_path = {line.split("\t")[0]: line.split("\t")[1] for line in out if line and not line.startswith("#")}
    assert "rule 1" in by_path["tests/test_pin_key.py"] and "rule 4" in by_path["tests/test_pin_key.py"]
    counts = [line for line in out if line.startswith("#")]
    assert len(counts) == 4
    assert [line.split(":")[-1].strip() for line in counts] == ["2", "2", "2", "4"]
    assert all(f"rule {n}" in line for n, line in zip((1, 2, 3, 4), counts))

    # Without a key all four counts are still printed, rule 4 at zero.
    assert script.main(["--root", str(repo), OWNED]) == 0
    counts = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert len(counts) == 4 and counts[3].endswith(": 0")


def test_search_writes_nothing(repo, capsys):
    before = _snapshot(repo)
    hints.find_hints(repo, [OWNED, "scripts"], [KEY, "absent"])
    script = _load_script()
    assert script.main(["--root", str(repo), "--owned", OWNED, "--key", KEY]) == 0
    capsys.readouterr()
    assert _snapshot(repo) == before

    # Owned paths and roots that leave the repository are refused, still writing nothing.
    for bad in ("../outside/test_outside.py", "tests/test_link.py", "tests/link_dir/deep.py"):
        with pytest.raises(ValueError):
            hints.find_hints(repo, [bad])
        assert script.main(["--root", str(repo), "--owned", bad]) == 2
    with pytest.raises(ValueError):
        hints.find_hints(repo / "missing", [OWNED])
    assert script.main(["--root", str(repo / "missing"), OWNED]) == 2
    capsys.readouterr()
    assert _snapshot(repo) == before


def test_script_started_by_a_person_writes_nothing_into_its_repository(tmp_path):
    source = Path(hints.__file__).resolve().parent
    copy = tmp_path / "copy"
    for rel, original in (
        ("hermes_cli/__init__.py", source / "__init__.py"),
        ("hermes_cli/kanban_scope_hints.py", Path(hints.__file__)),
        ("scripts/kanban_scope_hints.py", SCRIPT),
        ("tests/hermes_cli/test_kanban_scope_hints.py", Path(__file__)),
    ):
        _write(copy / rel, original.read_text(encoding="utf-8"))
    before = _snapshot(copy)
    # No special environment variable: neither bytecode setting may be set in the child.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    run = subprocess.run(
        [sys.executable, "scripts/kanban_scope_hints.py", "--root", ".", "hermes_cli/kanban_scope_hints.py"],
        cwd=copy, env=env, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert "scripts/kanban_scope_hints.py" in _output_paths(run.stdout)
    assert _snapshot(copy) == before
    assert not list(copy.rglob("__pycache__")) and not list(copy.rglob("*.pyc"))


def test_owned_folder_is_read_like_an_owned_file_by_each_rule(tmp_path, capsys):
    root = tmp_path / "repo"
    _write(root / "pkg" / "__init__.py", "")
    _write(root / "pkg" / "widget.py", "def build():\n    return 1\n")
    _write(root / "tests" / "test_pin.py", "import pkg.widget\n")                    # rule 1
    _write(root / "lib" / "reader.py", "from pkg.widget import build\n")             # rule 2
    _write(root / "docs" / "notes.md", "See 'pkg/widget.py' for details.\n")        # rule 3
    _write(root / "config" / "settings.json", f'{{"{KEY}": 1}}\n')                   # rule 4
    _write(root / "lib" / "unrelated.py", "import os\n")
    expected = {
        "config/settings.json": (4,),
        "docs/notes.md": (3,),
        "lib/reader.py": (2,),
        "tests/test_pin.py": (1,),
    }

    by_file = hints.find_hints(root, ["pkg/widget.py", "pkg/__init__.py"], [KEY])
    assert _rules(by_file) == expected
    assert by_file.counts == {1: 1, 2: 1, 3: 1, 4: 1}
    for folder in ("pkg", "pkg/", str(root / "pkg")):
        by_folder = hints.find_hints(root, [folder], [KEY])
        assert _paths(by_folder) == _paths(by_file)
        assert _rules(by_folder) == expected
        assert by_folder.counts == by_file.counts

    # The whole repository as the owned scope owns every file: nothing to list, all counts zero.
    for whole in (".", "./", str(root)):
        report = hints.find_hints(root, [whole], [KEY])
        assert report.hints == () and report.counts == {1: 0, 2: 0, 3: 0, 4: 0}
    script = _load_script()
    assert script.main(["--root", str(root), "--owned", ".", "--key", KEY]) == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == []
    assert [line.split(":")[-1].strip() for line in out] == ["0", "0", "0", "0"]


def test_file_the_search_cannot_read_is_named_and_counted(repo, capsys):
    (repo / "data").mkdir()
    (repo / "data" / "bad.txt").write_bytes(b'\xff\xfe invalid utf-8 "' + KEY.encode() + b'"\n')
    (repo / "data" / "binary.bin").write_bytes(b'\0\1\2 "' + KEY.encode() + b'"\n')
    report = hints.find_hints(repo, [OWNED], [KEY])
    assert report.unreadable == ("data/bad.txt",)
    assert "data/bad.txt" not in _paths(report)
    assert report.counts[4] == 3

    script = _load_script()
    assert script.main(["--root", str(repo), "--owned", OWNED, "--key", KEY]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line for line in out if "bad.txt" in line] == ["unreadable\tdata/bad.txt"]
    assert [line for line in out if line.startswith("# unreadable")] == ["# unreadable: 1"]
    # The key matches are still first, and the four rule counts are still printed.
    assert _output_paths("\n".join(out))[:3] == ["config/settings.json", "docs/notes.md", "tests/test_pin_key.py"]
    assert len([line for line in out if line.startswith("# rule")]) == 4


def _run_cli_unprivileged(args: list[str], build, locked: dict[str, int]) -> subprocess.CompletedProcess:
    """Build a repository under /tmp with *build(repo)*, set the modes in *locked* (path
    relative to the repository), and run the real script as a user that obeys them (root
    ignores modes, so drop to uid 65534). The modes are restored before the cleanup."""
    base = Path(tempfile.mkdtemp(dir="/tmp"))
    repo_dir = base / "repo"
    try:
        os.chmod(base, 0o755)
        for rel, original in (
            ("hermes_cli/__init__.py", None),
            ("hermes_cli/kanban_scope_hints.py", Path(hints.__file__)),
            ("scripts/kanban_scope_hints.py", SCRIPT),
        ):
            _write(base / "tool" / rel, "" if original is None else original.read_text(encoding="utf-8"))
        repo_dir.mkdir()
        build(repo_dir)
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames:
                os.chmod(Path(dirpath, name), 0o755)
            for name in filenames:
                os.chmod(Path(dirpath, name), 0o644)
        for rel, mode in locked.items():
            os.chmod(repo_dir / rel, mode)

        def drop() -> None:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)

        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
        return subprocess.run(
            [sys.executable, str(base / "tool" / "scripts" / "kanban_scope_hints.py"), "--root", str(repo_dir), *args],
            cwd=base, env=env, capture_output=True, text=True, timeout=60,
            preexec_fn=drop if os.geteuid() == 0 else None,
        )
    finally:
        for rel in locked:
            os.chmod(repo_dir / rel, 0o755)
        shutil.rmtree(base, ignore_errors=True)


def test_directory_the_search_cannot_list_is_named_and_counted():
    def build(repo: Path) -> None:
        _write(repo / "owned.py", "VALUE = 1\n")
        _write(repo / "open" / "pin.txt", f'"{KEY}"\n')
        _write(repo / "locked" / "pin.txt", f'"{KEY}"\n')

    run = _run_cli_unprivileged(["--owned", "owned.py", "--key", KEY], build, {"locked": 0o000})
    assert run.returncode == 0, run.stderr
    out = run.stdout.splitlines()
    assert [line for line in out if line.startswith("unreadable")] == ["unreadable\tlocked/"]
    assert [line for line in out if line.startswith("# unreadable")] == ["# unreadable: 1"]
    assert [p for p in _output_paths(run.stdout) if p != "unreadable"] == ["open/pin.txt"]
    assert [line.split(":")[0] for line in out if line.startswith("# rule")] == [
        f"# rule {rule} ({name})" for rule, name in hints.RULE_NAMES.items()
    ]


def test_owned_directory_the_search_cannot_list_is_named_and_counted():
    def build(repo: Path) -> None:
        _write(repo / "locked" / "widget.py", "VALUE = 1\n")
        _write(repo / "notes.txt", "See 'locked/widget.py' for details.\n")

    run = _run_cli_unprivileged(["--owned", "locked"], build, {"locked": 0o111})
    assert run.returncode == 0, run.stderr
    out = run.stdout.splitlines()
    assert [line for line in out if line.startswith("unreadable")] == ["unreadable\tlocked/"]
    assert [line for line in out if line.startswith("# unreadable")] == ["# unreadable: 1"]
