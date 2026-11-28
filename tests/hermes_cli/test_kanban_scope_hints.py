"""The scope hints search lists the tests and readers of the paths a proposed card will own.

Every test builds a temporary repository tree under ``tmp_path``; nothing reads the
real repository or the network.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
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
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 3, 5: 0, 6: 0, 7: 0}
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
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 4, 5: 0, 6: 0, 7: 0}

    # Through the script: same order, every rule on the line, one count per rule.
    script = _load_script()
    code = script.main(["--root", str(repo), "--owned", OWNED, "--key", KEY])
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == _paths(report)
    by_path = {line.split("\t")[0]: line.split("\t")[1] for line in out if line and not line.startswith("#")}
    assert "rule 1" in by_path["tests/test_pin_key.py"] and "rule 4" in by_path["tests/test_pin_key.py"]
    counts = [line for line in out if line.startswith("#")]
    assert len(counts) == 7
    assert [line.split(":")[-1].strip() for line in counts] == ["2", "2", "2", "4", "0", "0", "0"]
    assert all(f"rule {n}" in line for n, line in zip((1, 2, 3, 4, 5, 6, 7), counts))

    # Without a key all seven counts are still printed, rule 4 at zero.
    assert script.main(["--root", str(repo), OWNED]) == 0
    counts = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert len(counts) == 7 and counts[3].endswith(": 0")


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
    assert by_file.counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0}
    for folder in ("pkg", "pkg/", str(root / "pkg")):
        by_folder = hints.find_hints(root, [folder], [KEY])
        assert _paths(by_folder) == _paths(by_file)
        assert _rules(by_folder) == expected
        assert by_folder.counts == by_file.counts

    # The whole repository as the owned scope owns every file: nothing to list, all counts zero.
    for whole in (".", "./", str(root)):
        report = hints.find_hints(root, [whole], [KEY])
        assert report.hints == () and report.counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    script = _load_script()
    assert script.main(["--root", str(root), "--owned", ".", "--key", KEY]) == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == []
    assert [line.split(":")[-1].strip() for line in out] == ["0", "0", "0", "0", "0", "0", "0"]


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
    # The key matches are still first, and the seven rule counts are still printed.
    assert _output_paths("\n".join(out))[:3] == ["config/settings.json", "docs/notes.md", "tests/test_pin_key.py"]
    assert len([line for line in out if line.startswith("# rule")]) == 7


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


def test_helper_with_a_key_hit_keeps_its_rule_4_row_and_the_counts(helper_repo):
    _write(helper_repo / "tests" / "b.py", f'from tests.a import helper\nK = "{KEY}"\n')
    report = hints.find_hints(helper_repo, [OWNED], [KEY])
    rows = [line for line in hints.format_report(report) if line.startswith("tests/b.py\t")]
    assert rows == [f'tests/b.py\trule 4 (quoted key: "{KEY}")', f"tests/b.py\t{HELPER_LABEL}"]
    assert [h.rules for h in report.hints if h.path == "tests/b.py"] == [(4,), ()]
    assert report.counts == {1: 2, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}


def test_helper_naming_the_owned_path_keeps_its_rule_3_row_and_the_counts(helper_repo):
    _write(helper_repo / "tests" / "b.py", f'from tests.a import helper\nP = "{OWNED}"\n')
    report = hints.find_hints(helper_repo, [OWNED])
    rows = [line for line in hints.format_report(report) if line.startswith("tests/b.py\t")]
    assert rows == ["tests/b.py\trule 3 (names owned path)", f"tests/b.py\t{HELPER_LABEL}"]
    assert [h.rules for h in report.hints if h.path == "tests/b.py"] == [(3,), ()]
    assert report.counts == {1: 2, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}


SAVED = "tests/contracts/owner_payloads"
SAVED_LABEL = "rule 5 (saved example)"
FAMILY_LABEL = "rule 5 (saved example: every example of the family)"


@pytest.fixture
def saved_repo(tmp_path):
    """An owned module and saved examples in two families; tests are added by each test."""
    root = tmp_path / "saved_repo"
    _write(root / ".git" / "config", "")
    _write(root / OWNED, "def build():\n    return 1\n")
    _write(root / "tests" / "__init__.py", "")
    for name in ("fam/one", "fam/two", "other/three"):
        _write(root / SAVED / f"{name}.json", "{}\n")
    _write(root / SAVED / "fam" / "notes.txt", "not an example\n")
    return root


def _saved_rows(report) -> dict[str, str]:
    return {k: v for k, v in _lines(report).items() if k.startswith(SAVED)}


def test_saved_example_named_by_two_string_literals_is_listed_with_rule_5(saved_repo):
    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n"
           "    owner_payload_example('fam', \"one\", {})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_rows(report) == {f"{SAVED}/fam/one.json": SAVED_LABEL}
    assert _rules(report)[f"{SAVED}/fam/one.json"] == (5,)
    assert [h for h in report.hints if h.path.endswith("one.json")][0].keys == ()


def test_family_bound_to_a_module_level_string_and_keyword_arguments_are_read(saved_repo):
    _write(saved_repo / "tests" / "test_name.py",
           "import hermes_cli.widget\n\nFAMILY = 'fam'\nKIND = 'two'\n\n"
           "def test_a(fx, helpers):\n"
           "    helpers.owner_payload_example(family=FAMILY, record_kind=KIND, payload={})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_rows(report) == {f"{SAVED}/fam/two.json": SAVED_LABEL}


def test_name_that_is_not_one_string_literal_is_not_read(saved_repo):
    _write(saved_repo / "tests" / "test_names.py",
           "import hermes_cli.widget\n\nFAMILY = 'fam'\nKIND = make()\nOTHER = 'other'\nOTHER = 'fam'\n\n"
           "def test_a(owner_payload_example, family):\n"
           "    owner_payload_example(family, 'one', {})\n"
           "    owner_payload_example(OTHER, 'three', {})\n"
           "    owner_payload_example(FAMILY, KIND, {}) if False else None\n"
           "    for FAMILY in ('other',):\n"
           "        owner_payload_example(FAMILY, 'three', {})\n")
    assert _saved_rows(hints.find_hints(saved_repo, [OWNED])) == {}


def test_record_kind_that_is_not_a_literal_lists_every_json_file_of_that_family(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\nFAMILY = 'fam'\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    return {k: owner_payload_example(FAMILY, k, a) for k, a in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_rows(report) == {
        f"{SAVED}/fam/one.json": FAMILY_LABEL,
        f"{SAVED}/fam/two.json": FAMILY_LABEL,
    }


def _saved_lines(report) -> list[str]:
    return sorted(line for line in hints.format_report(report) if line.startswith(SAVED))


def test_repeated_exact_calls_give_one_row_per_call(saved_repo):
    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n"
           "    owner_payload_example('fam', 'one', {})\n    owner_payload_example('fam', 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_lines(report) == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"] * 2
    assert report.counts[5] == 2


def test_repeated_family_only_calls_give_one_row_per_call_and_file(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\nFAMILY = 'fam'\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    a = {k: owner_payload_example(FAMILY, k, v) for k, v in answers.items()}\n"
           "    b = {k: owner_payload_example(FAMILY, k, v) for k, v in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_lines(report) == [f"{SAVED}/fam/one.json\t{FAMILY_LABEL}"] * 2 + [
        f"{SAVED}/fam/two.json\t{FAMILY_LABEL}"] * 2
    assert report.counts[5] == 4


def test_an_exact_row_never_replaces_a_family_only_row_of_the_same_path(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    owner_payload_example('fam', 'one', {})\n"
           "    return {k: owner_payload_example('fam', k, a) for k, a in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_lines(report) == [
        f"{SAVED}/fam/one.json\t{SAVED_LABEL}",
        f"{SAVED}/fam/one.json\t{FAMILY_LABEL}",
        f"{SAVED}/fam/two.json\t{FAMILY_LABEL}",
    ]
    assert report.counts[5] == 3


def test_family_that_is_not_a_literal_lists_nothing(saved_repo):
    _write(saved_repo / "tests" / "test_f.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, fam, answers):\n"
           "    owner_payload_example(fam, 'one', {})\n"
           "    return {k: owner_payload_example(fam, k, a) for k, a in answers.items()}\n")
    assert _saved_rows(hints.find_hints(saved_repo, [OWNED])) == {}


def test_missing_literal_file_family_folder_and_unsafe_names_are_not_listed(saved_repo):
    _write(saved_repo / "tests" / "test_missing.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    owner_payload_example('fam', 'absent', {})\n"
           "    owner_payload_example('nofolder', 'one', {})\n"
           "    owner_payload_example('..', 'one', {})\n"
           "    owner_payload_example('fam/../other', 'three', {})\n"
           "    owner_payload_example('other', '../other/three', {})\n"
           "    return {k: owner_payload_example('nofolder', k, a) for k, a in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_rows(report) == {}
    assert report.counts[5] == 0


def test_test_file_that_is_not_listed_is_not_read(saved_repo):
    _write(saved_repo / "tests" / "test_unlisted.py",
           "def test_a(owner_payload_example):\n    owner_payload_example('other', 'three', {})\n")
    _write(saved_repo / "lib" / "caller.py", "owner_payload_example('other', 'three', {})\n")
    _write(saved_repo / "tests" / "test_listed.py", "import hermes_cli.widget\n")
    assert _saved_rows(hints.find_hints(saved_repo, [OWNED])) == {}


def test_a_saved_example_the_card_owns_is_not_listed(saved_repo):
    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n    owner_payload_example('fam', 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED, f"{SAVED}/fam/one.json"])
    assert _saved_rows(report) == {}
    folder = hints.find_hints(saved_repo, [OWNED, f"{SAVED}/fam"])
    assert _saved_rows(folder) == {}


def test_saved_example_with_its_own_rule_4_row_keeps_it_and_gets_a_separate_rule_5_row(saved_repo):
    _write(saved_repo / SAVED / "fam" / "one.json", '{"unit_running": true}\n')
    plain = hints.find_hints(saved_repo, [OWNED], ["unit_running"])
    plain_lines = hints.format_report(plain)
    own = f'{SAVED}/fam/one.json\trule 4 (quoted key: "unit_running")'
    assert plain_lines[0] == own and plain.counts == {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}

    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n    owner_payload_example('fam', 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED], ["unit_running"])
    lines = hints.format_report(report)
    assert lines[:2] == [own, f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert [h.rules for h in report.hints[:2]] == [(4,), (5,)]
    assert [h.keys for h in report.hints[:2]] == [("unit_running",), ()]
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0}
    # Besides the rule 5 row and its count line, the new test file's own row is the only change.
    rest = [line for line in lines if SAVED_LABEL not in line and not line.startswith("# rule 5")]
    assert [line for line in rest if "test_lit.py" not in line and not line.startswith("# rule 1")] == [
        line for line in plain_lines if not line.startswith("# rule 5") and not line.startswith("# rule 1")]


def test_helper_row_that_calls_owner_payload_example_is_read_too(helper_repo):
    _write(helper_repo / SAVED / "fam" / "one.json", "{}\n")
    _write(helper_repo / "tests" / "b.py",
           "from tests.a import helper\n\ndef test_b(owner_payload_example):\n"
           "    owner_payload_example('fam', 'one', helper())\n")
    report = hints.find_hints(helper_repo, [OWNED])
    assert _lines(report)["tests/b.py"] == HELPER_LABEL
    assert _lines(report)[f"{SAVED}/fam/one.json"] == SAVED_LABEL
    assert report.counts == {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}


def test_rule_5_has_its_own_count_line_after_the_four_of_today(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    owner_payload_example('other', 'three', {})\n"
           "    return {k: owner_payload_example('fam', k, a) for k, a in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 0, 5: 3, 6: 0, 7: 0}
    assert hints.RULE_NAMES[5] == "saved example"
    counts = [line for line in hints.format_report(report) if line.startswith("#")]
    assert counts == [
        "# rule 1 (test imports owned module): 1",
        "# rule 2 (imports name from owned module): 0",
        "# rule 3 (names owned path): 0",
        "# rule 4 (quoted key): 0",
        "# rule 5 (saved example): 3",
        "# rule 6 (whole-answer comparison): 0",
        "# rule 7 (calls a named function): 0",
    ]
    assert hints.find_hints(saved_repo, ["."]).counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}


def _one_row(saved_repo, source: str) -> dict[str, str]:
    _write(saved_repo / "tests" / "test_scope.py", "import hermes_cli.widget\n\n" + source)
    return _saved_rows(hints.find_hints(saved_repo, [OWNED]))


def test_production_module_with_an_old_rule_hit_is_not_read_for_saved_examples(saved_repo):
    _write(saved_repo / "lib" / "reader.py",
           "from hermes_cli.widget import build\n\n"
           "def run(owner_payload_example):\n    owner_payload_example('fam', 'one', build())\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _lines(report)["lib/reader.py"] == "rule 2 (imports name from owned module)"
    assert _saved_rows(report) == {}
    assert report.counts[5] == 0


def test_a_literal_name_that_is_a_safe_single_component_is_read_even_with_a_space(saved_repo):
    _write(saved_repo / SAVED / "fam" / "with space.json", "{}\n")
    _write(saved_repo / SAVED / "my fam" / "one.json", "{}\n")
    rows = _one_row(
        saved_repo,
        "def test_a(owner_payload_example):\n    owner_payload_example('fam', 'with space', {})\n"
        "    owner_payload_example('my fam', 'one', {})\n")
    assert rows == {f"{SAVED}/fam/with space.json": SAVED_LABEL, f"{SAVED}/my fam/one.json": SAVED_LABEL}


def test_empty_dot_separator_nul_and_absolute_names_are_not_read(saved_repo):
    rows = _one_row(
        saved_repo,
        "def test_a(owner_payload_example):\n"
        "    owner_payload_example('fam', '', {})\n"
        "    owner_payload_example('fam', '.', {})\n"
        "    owner_payload_example('.', 'fam', {})\n"
        "    owner_payload_example('', 'one', {})\n"
        "    owner_payload_example('fam', '..\\\\fam\\\\one', {})\n"
        "    owner_payload_example('fam', 'one\\0', {})\n"
        "    owner_payload_example('/tmp', 'one', {})\n"
        "    owner_payload_example('fam', '/one', {})\n")
    assert rows == {}


def test_a_saved_example_alias_whose_resolved_target_is_owned_is_not_listed_or_counted(saved_repo):
    (saved_repo / SAVED / "fam" / "alias.json").symlink_to(saved_repo / SAVED / "fam" / "one.json")
    source = ("def test_a(owner_payload_example):\n    owner_payload_example('fam', 'alias', {})\n"
              "    owner_payload_example('fam', 'two', {})\n")
    _write(saved_repo / "tests" / "test_scope.py", "import hermes_cli.widget\n\n" + source)
    report = hints.find_hints(saved_repo, [OWNED, f"{SAVED}/fam/one.json"])
    assert _saved_rows(report) == {f"{SAVED}/fam/two.json": SAVED_LABEL}
    assert report.counts[5] == 1
    # The alias is listed when its target is not owned.
    assert f"{SAVED}/fam/alias.json" in _saved_rows(hints.find_hints(saved_repo, [OWNED]))


def test_a_family_folder_is_checked_before_it_is_enumerated_and_its_files_are_checked_too(saved_repo, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "x.json", "{}\n")
    _write(saved_repo / "node_modules" / "skipped" / "y.json", "{}\n")
    (saved_repo / SAVED / "away").symlink_to(outside, target_is_directory=True)
    (saved_repo / SAVED / "skipdir").symlink_to(saved_repo / "node_modules" / "skipped", target_is_directory=True)
    (saved_repo / SAVED / "again").symlink_to(saved_repo / SAVED / "fam", target_is_directory=True)
    _write(saved_repo / SAVED / "node_modules" / "z.json", "{}\n")
    (saved_repo / SAVED / "fam" / "skipfile.json").symlink_to(saved_repo / "node_modules" / "skipped" / "y.json")
    source = ("def test_a(owner_payload_example, answers):\n"
              "    owner_payload_example('away', 'x', {})\n"
              "    owner_payload_example('skipdir', 'y', {})\n"
              "    owner_payload_example('node_modules', 'z', {})\n"
              "    owner_payload_example('fam', 'skipfile', {})\n"
              "    return {k: owner_payload_example(f, k, a) for f in ('away',) for k, a in answers.items()}\n"
              "    return {k: owner_payload_example('away', k, a) for k, a in answers.items()}\n"
              "    return {k: owner_payload_example('skipdir', k, a) for k, a in answers.items()}\n"
              "    return {k: owner_payload_example('node_modules', k, a) for k, a in answers.items()}\n")
    assert _one_row(saved_repo, source) == {}
    # A family folder that is an alias of an owned folder lists no file of it.
    _write(saved_repo / "tests" / "test_scope.py",
           "import hermes_cli.widget\n\ndef test_a(owner_payload_example, answers):\n"
           "    return {k: owner_payload_example('again', k, a) for k, a in answers.items()}\n")
    owned = hints.find_hints(saved_repo, [OWNED, f"{SAVED}/fam"])
    assert _saved_rows(owned) == {}
    shown = hints.find_hints(saved_repo, [OWNED])
    assert set(_saved_rows(shown)) == {f"{SAVED}/again/one.json", f"{SAVED}/again/two.json"}


@pytest.fixture
def listed_folders(monkeypatch):
    """The folders whose files ``Path.glob`` is asked to list."""
    seen: list[Path] = []
    real_glob = Path.glob

    def spy(self, pattern, *args, **kwargs):
        seen.append(self)
        return real_glob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", spy)
    return seen


def _family_call(family: str) -> str:
    return f"def test_a(owner_payload_example, kind):\n    owner_payload_example({family!r}, kind, {{}})\n"


def test_a_family_folder_that_is_owned_skipped_or_leads_out_is_not_enumerated(saved_repo, tmp_path, listed_folders):
    outside = tmp_path / "outside"
    _write(outside / "x.json", "{}\n")
    _write(saved_repo / "node_modules" / "skipped" / "y.json", "{}\n")
    _write(saved_repo / SAVED / "node_modules" / "z.json", "{}\n")
    (saved_repo / SAVED / "away").symlink_to(outside, target_is_directory=True)
    (saved_repo / SAVED / "skipdir").symlink_to(saved_repo / "node_modules" / "skipped", target_is_directory=True)
    (saved_repo / SAVED / "again").symlink_to(saved_repo / SAVED / "fam", target_is_directory=True)
    owned = (OWNED, f"{SAVED}/fam")
    for family in ("fam", "again", "node_modules", "away", "skipdir"):
        _, lines = _rule_5_rows(saved_repo, _family_call(family), owned)
        assert _saved_only(lines) == {}, family
    assert listed_folders == []


def test_an_eligible_family_folder_is_still_enumerated_and_gives_the_same_rows(saved_repo, tmp_path, listed_folders):
    (saved_repo / SAVED / "again").symlink_to(saved_repo / SAVED / "fam", target_is_directory=True)
    _, lines = _rule_5_rows(saved_repo, _family_call("fam"))
    assert _saved_only(lines) == FAMILY_ROWS
    assert listed_folders == [saved_repo / SAVED / "fam"]
    _, lines = _rule_5_rows(saved_repo, _family_call("again"))
    assert _saved_only(lines) == {f"{SAVED}/again/one.json": FAMILY_LABEL, f"{SAVED}/again/two.json": FAMILY_LABEL}
    assert listed_folders == [saved_repo / SAVED / "fam", saved_repo / SAVED / "again"]


# Card 2, one narrow rule: an argument is resolved only by a string literal at the call, or by a
# plain name that the whole file binds once, at module level, to a string literal. No scope rule.
NOT_RESOLVED = "rule 5 (saved example not resolved)"
TEST_SCOPE = "tests/test_scope.py"
ONE_ROW = {f"{SAVED}/fam/one.json": SAVED_LABEL}
FAMILY_ROWS = {f"{SAVED}/fam/one.json": FAMILY_LABEL, f"{SAVED}/fam/two.json": FAMILY_LABEL}


def _rule_5_rows(saved_repo, source: str, owned=(OWNED,)):
    """The report and the rule 5 lines (saved examples and not resolved) for one test file."""
    compile(source, "<case>", "exec")  # every case is valid Python; nothing is run
    _write(saved_repo / "tests" / "test_scope.py", "import hermes_cli.widget\n\n" + source)
    report = hints.find_hints(saved_repo, list(owned))
    lines = [line for line in hints.format_report(report)
             if not line.startswith("#") and (line.startswith(SAVED) or NOT_RESOLVED in line)]
    return report, lines


def _saved_only(lines: list[str]) -> dict[str, str]:
    return dict(line.split("\t") for line in lines if line.startswith(SAVED))


def _not_resolved(lines: list[str]) -> list[str]:
    return [line for line in lines if NOT_RESOLVED in line]


def _at(source: str) -> str:
    """The expected not-resolved row for the one call in *source* (two lines come before it)."""
    number = next(i for i, text in enumerate(source.split("\n"), 3) if "owner_payload_example(" in text)
    return f"{TEST_SCOPE}\t{NOT_RESOLVED}: line {number}"


def test_two_string_literals_give_the_saved_example_row(saved_repo):
    source = "def test_a(owner_payload_example):\n    owner_payload_example('fam', 'one', {})\n"
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert report.counts[5] == 1


def test_a_module_name_bound_once_to_a_literal_gives_the_same_row(saved_repo):
    source = ("FAMILY = 'fam'\nKIND: str = 'one'\n\ndef test_a(owner_payload_example):\n"
              "    owner_payload_example(FAMILY, KIND, {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert report.counts[5] == 1


# The eight cases of the three review rounds: each binds the name a second time, in another
# place of the file, and the search does not ask which binding the call sees.
EIGHT_CASES = {
    "helper parameter":
        "{n} = '{v}'\n\ndef helper({n}):\n    return {n}\n\ndef test_a(owner_payload_example):\n    {call}\n",
    "loop variable":
        "{n} = '{v}'\n\ndef test_a(owner_payload_example):\n    {call}\n\n"
        "def test_b():\n    for {n} in range(3):\n        print({n})\n",
    "match capture":
        "{n} = '{v}'\n\ndef test_b(x):\n    match x:\n        case {n}:\n            pass\n\n"
        "def test_a(owner_payload_example):\n    {call}\n",
    "global declaration":
        "{n} = '{v}'\n\ndef setup():\n    global {n}\n\ndef test_a(owner_payload_example):\n    {call}\n",
    "class body calling before assigning":
        "{n} = '{v}'\n\nclass T:\n    {call}\n    {n} = 'other'\n",
    "annotation with no value":
        "{n} = '{v}'\n\ndef test_a(owner_payload_example):\n    {n}: str\n    {call}\n",
    "class default":
        "{n} = '{v}'\n\nclass T:\n    {n} = 'other'\n\n"
        "    def test_a(self, owner_payload_example, seen={n}):\n        {call}\n",
    "nonlocal binding":
        "{n} = '{v}'\n\ndef outer(owner_payload_example):\n    {n} = 'other'\n\n"
        "    def inner():\n        nonlocal {n}\n        {call}\n",
}


@pytest.mark.parametrize("case", EIGHT_CASES)
def test_each_of_the_eight_cases_leaves_the_family_not_resolved_and_names_no_example(saved_repo, case):
    source = EIGHT_CASES[case].format(n="FAMILY", v="fam", call="owner_payload_example(FAMILY, 'one', {})")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    assert report.counts[5] == 1
    assert not [h for h in report.hints if h.path.startswith(SAVED)]


@pytest.mark.parametrize("case", EIGHT_CASES)
def test_each_of_the_eight_cases_leaves_the_record_kind_not_resolved_so_the_family_is_listed(saved_repo, case):
    source = EIGHT_CASES[case].format(n="KIND", v="one", call="owner_payload_example('fam', KIND, {})")
    report, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == FAMILY_ROWS
    assert _not_resolved(lines) == []
    assert report.counts[5] == 2


# One more binding of any kind, in any scope, takes the name out; the first line is the one
# module-level literal and {n} is the name.
SECOND_BINDINGS = {
    "second assignment": "{n} = 'fam'\n",
    "augmented assignment": "{n} += 'x'\n",
    "annotated assignment": "{n}: str = 'fam'\n",
    "named expression": "def f():\n    return ({n} := 1)\n",
    "parameter": "def f({n}=None):\n    pass\n",
    "keyword-only parameter": "def f(*, {n}):\n    pass\n",
    "variadic parameter": "def f(*{n}):\n    pass\n",
    "lambda parameter": "f = lambda {n}: {n}\n",
    "loop target": "for {n} in 'ab':\n    pass\n",
    "starred target": "*{n}, last = 'ab'\n",
    "with target": "def f(cm):\n    with cm as {n}:\n        pass\n",
    "except name": "try:\n    pass\nexcept OSError as {n}:\n    pass\n",
    "match capture": "def f(x):\n    match x:\n        case [*{n}]:\n            pass\n",
    "match mapping rest": "def f(x):\n    match x:\n        case {{'k': 1, **{n}}}:\n            pass\n",
    "comprehension target": "L = [{n} for {n} in 'ab']\n",
    "import": "import json as {n}\n",
    "import from": "from os import path as {n}\n",
    "function": "def {n}():\n    pass\n",
    "class": "class {n}:\n    pass\n",
    "global declaration": "def f():\n    global {n}\n",
    "nonlocal declaration": "def f():\n    {n} = 1\n    def g():\n        nonlocal {n}\n",
    "deletion": "del {n}\n",
    "type parameter": "def f[{n}](x):\n    pass\n",
}


@pytest.mark.parametrize("binding", [
    pytest.param(name, marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters need Python 3.12"))
    if name == "type parameter" else name
    for name in SECOND_BINDINGS])
def test_a_second_binding_of_any_kind_leaves_the_name_not_resolved(saved_repo, binding):
    source = ("FAMILY = 'fam'\n" + SECOND_BINDINGS[binding].format(n="FAMILY")
              + "\ndef test_a(owner_payload_example):\n    owner_payload_example(FAMILY, 'one', {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    assert report.counts[5] == 1


def test_a_name_bound_twice_with_the_same_literal_is_not_resolved(saved_repo):
    source = "FAMILY = 'fam'\nFAMILY = 'fam'\n\ndef t(owner_payload_example):\n    owner_payload_example(FAMILY, 'one', {})\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]


@pytest.mark.parametrize("module_part", [
    "if True:\n    FAMILY = 'fam'\n",
    "try:\n    FAMILY = 'fam'\nexcept OSError:\n    pass\n",
    "def setup():\n    FAMILY = 'fam'\n",
    "class Holder:\n    FAMILY = 'fam'\n",
    "for _ in range(1):\n    FAMILY = 'fam'\n",
])
def test_a_single_binding_that_is_not_at_module_level_does_not_resolve(saved_repo, module_part):
    source = module_part + "\ndef t(owner_payload_example):\n    owner_payload_example(FAMILY, 'one', {})\n"
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    assert report.counts[5] == 1


@pytest.mark.parametrize("value", ["5", "None", "b'fam'", "True", "('fam',)", "['fam']", "{'fam'}"])
def test_a_binding_that_is_not_a_text_does_not_resolve(saved_repo, value):
    source = f"FAMILY = {value}\n\ndef t(owner_payload_example):\n    owner_payload_example(FAMILY, 'one', {{}})\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]


@pytest.mark.parametrize("value", ["'f' + 'am'", "f'fam'", "f'{OTHER}'", "make()", "OTHER", "'fam'.upper()",
                                   "names['a']", "'fam' if OTHER else 'other'", "str('fam')"])
def test_a_binding_by_any_other_expression_does_not_resolve(saved_repo, value):
    source = (f"OTHER = 'fam'\nFAMILY = {value}\n\n"
              "def t(owner_payload_example):\n    owner_payload_example(FAMILY, 'one', {})\n")
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]


@pytest.mark.parametrize("argument", [
    "'fam'.upper()", "f'fam'", "names['a']", "self.family", "'f' + 'am'", "make()", "UNBOUND", "*args",
])
def test_an_argument_that_is_neither_a_literal_nor_a_plain_name_is_not_resolved(saved_repo, argument):
    source = f"def t(owner_payload_example):\n    owner_payload_example({argument}, 'one', {{}})\n"
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    assert report.counts[5] == 1


def test_a_call_without_a_second_argument_lists_the_family_and_without_a_first_names_no_example(saved_repo):
    source = "def t(owner_payload_example):\n    owner_payload_example('fam')\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == FAMILY_ROWS
    source = "def t(owner_payload_example):\n    owner_payload_example()\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]


def test_a_name_that_is_only_read_in_other_places_still_resolves(saved_repo):
    source = ("FAMILY = 'fam'\nKIND = 'one'\n\ndef helper():\n    return FAMILY, KIND\n\n"
              "def t(owner_payload_example):\n    print(FAMILY)\n    owner_payload_example(FAMILY, KIND, {})\n")
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]


def test_a_resolved_family_with_a_record_kind_that_is_not_resolved_lists_the_family_rows(saved_repo):
    source = ("FAMILY = 'fam'\n\ndef t(owner_payload_example, kinds):\n"
              "    for kind in kinds:\n        owner_payload_example(FAMILY, kind, {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{path}\t{FAMILY_LABEL}" for path in sorted(FAMILY_ROWS)]
    assert report.counts[5] == 2
    # A record kind bound twice is not resolved either; a literal kind wins over the folder rows.
    source = ("KIND = 'one'\nKIND = 'two'\n\ndef t(owner_payload_example):\n"
              "    owner_payload_example('fam', KIND, {})\n    owner_payload_example('other', 'three', {})\n")
    _, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == {**FAMILY_ROWS, f"{SAVED}/other/three.json": SAVED_LABEL}


def test_an_unresolved_call_gives_one_row_for_the_test_file_with_the_line_of_the_call(saved_repo):
    source = ("import json\n\ndef t(owner_payload_example, family):\n"
              "    owner_payload_example(family, 'one', {})\n"
              "    owner_payload_example('fam', 'two', {})\n"
              "    owner_payload_example(family, 'three', {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert _not_resolved(lines) == [f"{TEST_SCOPE}\t{NOT_RESOLVED}: line 6", f"{TEST_SCOPE}\t{NOT_RESOLVED}: line 8"]
    assert _saved_only(lines) == {f"{SAVED}/fam/two.json": SAVED_LABEL}
    own = [line for line in hints.format_report(report) if line.startswith(TEST_SCOPE)]
    assert own[0] == f"{TEST_SCOPE}\trule 1 (test imports owned module)" and len(own) == 3
    assert [h.rules for h in report.hints if h.path == TEST_SCOPE] == [(1,), (5,), (5,)]
    assert not [h for h in report.hints if h.path.endswith("one.json") or h.path.endswith("three.json")]


def test_the_rule_5_count_line_equals_the_number_of_its_rows(saved_repo):
    source = ("FAMILY = 'fam'\n\n"
              "def t(owner_payload_example, family, kinds):\n"
              "    owner_payload_example('other', 'three', {})\n"
              "    for kind in kinds:\n        owner_payload_example(FAMILY, kind, {})\n"
              "    owner_payload_example(family, 'one', {})\n")
    report, _ = _rule_5_rows(saved_repo, source)
    out = hints.format_report(report)
    rows = [line for line in out if not line.startswith("#") and "rule 5" in line]
    assert len(rows) == 4 and sum(1 for h in report.hints if 5 in h.rules) == 4
    assert [line for line in out if line.startswith("# rule 5")] == ["# rule 5 (saved example): 4"]
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 0, 5: 4, 6: 0, 7: 0}


def test_a_test_file_that_is_not_listed_gets_no_not_resolved_row(saved_repo):
    _write(saved_repo / "tests" / "test_unlisted.py", "def t(o, f):\n    o.owner_payload_example(f, 'one', {})\n")
    _write(saved_repo / "lib" / "reader.py", "from hermes_cli.widget import build\nowner_payload_example(f, 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert not any(NOT_RESOLVED in line for line in hints.format_report(report))
    assert report.counts[5] == 0


def test_a_wildcard_import_makes_a_plain_name_not_resolved_but_not_a_literal(saved_repo):
    source = ("FAMILY = 'fam'\nfrom pkg.extras import *\n\ndef t(owner_payload_example):\n"
              "    owner_payload_example(FAMILY, 'one', {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    assert report.counts[5] == 1
    assert not [h for h in report.hints if h.path.startswith(SAVED)]
    source = ("FAMILY = 'fam'\nfrom pkg.extras import *\n\ndef t(owner_payload_example):\n"
              "    owner_payload_example('fam', 'one', {})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]


def test_keywords_resolve_both_inputs_next_to_a_starred_expansion(saved_repo):
    source = ("def t(owner_payload_example):\n"
              "    owner_payload_example(*(), family='fam', record_kind='one', payload={})\n")
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    source = ("def t(owner_payload_example):\n"
              "    owner_payload_example('fam', *(), record_kind='one', payload={})\n")
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert report.counts[5] == 1


def test_a_slot_after_a_starred_expansion_or_in_a_double_star_is_not_resolved(saved_repo):
    source = "def t(owner_payload_example):\n    owner_payload_example(*(), 'fam', 'one', {})\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    source = "def t(owner_payload_example):\n    owner_payload_example('fam', *(), 'one', {})\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == FAMILY_ROWS and _not_resolved(lines) == []
    source = "def t(owner_payload_example):\n    owner_payload_example('fam', **{'record_kind': 'one'})\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == FAMILY_ROWS and _not_resolved(lines) == []
    source = "def t(owner_payload_example):\n    owner_payload_example(**{'family': 'fam'}, record_kind='one')\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]


def test_a_slot_given_by_a_keyword_and_by_the_positional_prefix_is_not_resolved(saved_repo):
    source = "def t(owner_payload_example):\n    owner_payload_example('fam', family='other', record_kind='one')\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)]
    source = "def t(owner_payload_example):\n    owner_payload_example('fam', 'one', record_kind='two')\n"
    _, lines = _rule_5_rows(saved_repo, source)
    assert _saved_only(lines) == FAMILY_ROWS and _not_resolved(lines) == []


def test_two_unresolved_calls_on_one_line_give_two_rows_and_a_count_of_two(saved_repo):
    source = "def t(owner_payload_example, f):\n    owner_payload_example(f, 'one', {}); owner_payload_example(f, 'two', {})\n"
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [_at(source)] * 2
    assert report.counts[5] == 2
    assert "# rule 5 (saved example): 2" in hints.format_report(report)


def test_a_fixture_name_that_python_normalizes_is_read(saved_repo):
    source = ("def t(ｏwner_payload_example):\n    ｏwner_payload_example('fam', 'one', {})\n")
    assert "owner_payload_example" not in source
    report, lines = _rule_5_rows(saved_repo, source)
    assert lines == [f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert report.counts[5] == 1


def test_a_listed_test_file_with_a_byte_order_mark_keeps_its_rule_5_row(tmp_path, capsys):
    script = _load_script()
    source = (f"# {OWNED}\n\n"
              "def test_a(owner_payload_example):\n"
              "    owner_payload_example('fam', 'one', {})\n")
    results = {}
    for name, prefix in (("plain", b""), ("bom", b"\xef\xbb\xbf")):
        root = tmp_path / name
        _write(root / ".git" / "config", "")
        _write(root / OWNED, "def build():\n    return 1\n")
        _write(root / SAVED / "fam" / "one.json", "{}\n")
        target = root / "tests" / "test_lit.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(prefix + source.encode("utf-8"))
        assert script.main(["--root", str(root), "--owned", OWNED]) == 0
        results[name] = capsys.readouterr().out.splitlines()
    assert results["bom"] == results["plain"]
    assert "tests/test_lit.py\trule 3 (names owned path)" in results["bom"]
    assert f"{SAVED}/fam/one.json\t{SAVED_LABEL}" in results["bom"]
    assert "# rule 5 (saved example): 1" in results["bom"]


# Card 3, rule 6: each whole-answer comparison (== or != beside a dict display, a set display, or a
# call of the plain name set or frozenset) in the test files that rule 5 reads gives one row.
RULE_6 = "rule 6 (whole-answer comparison)"
WHOLE = "tests/test_whole.py"
WHOLE_HEAD = ("import hermes_cli.widget\n\n"
              "def _pin(owner_payload_example):\n"
              "    owner_payload_example('fam', 'one', {})\n\n")
FORMS = ["{'a': 1}", "{1, 2}", "set(x)", "frozenset(x)"]


def _six(saved_repo, source: str, name: str = "test_whole.py", head: str = WHOLE_HEAD, owned=(OWNED,)):
    """Write one listed test file that rule 5 reads; return the report, its rule 6 lines and the
    rows expected for the source lines that end with ``# hit``."""
    text = head + source
    _write(saved_repo / "tests" / name, text)
    report = hints.find_hints(saved_repo, list(owned))
    rows = [line for line in hints.format_report(report) if RULE_6 in line and not line.startswith("#")]
    expected = [f"tests/{name}\t{RULE_6}: line {n}" for n, t in enumerate(text.split("\n"), 1)
                if t.rstrip().endswith("# hit")]
    return report, rows, expected


def _six_hints(report) -> list:
    return [h for h in report.hints if 6 in h.rules]


def test_rule_6_is_named_in_rule_names_and_has_its_count_line_last():
    assert hints.RULE_NAMES[6] == "whole-answer comparison"
    assert list(hints.RULE_NAMES) == [1, 2, 3, 4, 5, 6, 7]
    report = hints.Report((), {rule: 0 for rule in hints.RULE_NAMES})
    assert hints.format_report(report)[-2] == "# rule 6 (whole-answer comparison): 0"


@pytest.mark.parametrize("op", ["==", "!="])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("form", FORMS)
def test_each_form_on_either_side_with_either_operator_gives_one_row_with_its_line(saved_repo, form, side, op):
    expr = f"x {op} {form}" if side == "right" else f"{form} {op} x"
    report, rows, expected = _six(saved_repo, f"def test_a(x):\n    assert {expr}  # hit\n")
    assert len(expected) == 1 and rows == expected
    assert report.counts[6] == 1
    assert [h.rules for h in report.hints if h.path == WHOLE] == [(1,), (6,)]


@pytest.mark.parametrize("expr", [
    "1 < x == {'a': 1}",
    "x == y < 3 != frozenset(z)",
    "{1, 2} == x < y",
    "set(y) != x < 3 < z",
])
def test_a_chained_comparison_with_one_counted_operator_gives_one_row(saved_repo, expr):
    report, rows, expected = _six(saved_repo, f"def test_a(x, y, z):\n    assert {expr}  # hit\n")
    assert len(rows) == 1 and rows == expected
    assert report.counts[6] == 1


def test_a_chained_comparison_with_two_counted_operators_is_still_one_node_and_one_row(saved_repo):
    report, rows, expected = _six(saved_repo, "def test_a(x, y):\n    assert {1} == x != {2}  # hit\n")
    assert rows == expected and len(rows) == 1 and report.counts[6] == 1


NOT_COUNTED = {
    "name bound to a dict": "A = {'a': 1}\n\ndef test_a(x):\n    assert x == A\n",
    "name bound to a set": "A = {1, 2}\n\ndef test_a(x):\n    assert A != x\n",
    "name bound to a frozenset call": "A = frozenset([1])\n\ndef test_a(x):\n    assert x == A\n",
    "dict comprehension": "def test_a(x, y):\n    assert x == {k: 1 for k in y}\n",
    "set comprehension": "def test_a(x, y):\n    assert {k for k in y} != x\n",
    "list comprehension": "def test_a(x, y):\n    assert x == [k for k in y]\n",
    "generator expression": "def test_a(x, y):\n    assert x == (k for k in y)\n",
    "list": "def test_a(x):\n    assert x == [1, 2]\n",
    "empty list": "def test_a(x):\n    assert [] != x\n",
    "tuple": "def test_a(x):\n    assert x == (1, 2)\n",
    "empty tuple": "def test_a(x):\n    assert x != ()\n",
    "call of dict": "def test_a(x):\n    assert x == dict(a=1)\n",
    "call through builtins.set": "import builtins\n\ndef test_a(x, y):\n    assert x == builtins.set(y)\n",
    "call through an attribute frozenset": "def test_a(x, y, b):\n    assert b.frozenset(y) != x\n",
    "call of a call of set": "def test_a(x, y, z):\n    assert x == set(y)(z)\n",
    "call on a dict display": "def test_a(x):\n    assert x == {'a': 1}.get('a')\n",
    "other names": "def test_a(x, y):\n    assert x == Set(y) and x == sorted(y) and x == list(y)\n",
    "subscript and attribute": "def test_a(x, y):\n    assert x == y['a'] and x != y.items\n",
    "a display that is no operand of a comparison": (
        "def test_a(x, f):\n    y = {'a': 1}\n    f(x, {1}, set(x))\n    assert x\n    return {1} | set(x)\n"),
    "comparison of two plain operands": "def test_a(x, y):\n    assert x == y != 3\n",
    "empty file body": "",
}


@pytest.mark.parametrize("case", list(NOT_COUNTED))
def test_a_form_that_the_closed_list_does_not_hold_gives_no_row(saved_repo, case):
    report, rows, _ = _six(saved_repo, NOT_COUNTED[case])
    assert rows == [] and _six_hints(report) == []
    assert report.counts[6] == 0
    assert any(h.path == WHOLE for h in report.hints)


@pytest.mark.parametrize("expr", [
    "{'a': 1} < x", "x < {'a': 1}", "{'a': 1} > x", "x <= {'a': 1}",
    "x in {'a': 1}", "x not in {'a': 1}", "x in {1, 2}", "x in set(y)",
    "x is {}", "x is not {'a': 1}", "{} is x", "x is set(y)", "x is not frozenset(y)",
    "{1} <= x", "x >= frozenset(y)",
    "{'a': 1} < x == y", "x == y in {1}", "x != y is {}", "x == y < set(z)", "y is {} is z",
])
def test_a_display_beside_another_operator_gives_no_row_even_in_a_chain(saved_repo, expr):
    report, rows, _ = _six(saved_repo, f"def test_a(x, y, z):\n    assert {expr}\n")
    assert rows == [] and report.counts[6] == 0


def test_a_comparison_at_any_depth_gives_one_row_each(saved_repo):
    source = (
        "import pytest\n\n"
        "class TestA:\n"
        "    def test_a(self, x, y):\n"
        "        def inner():\n"
        "            return x == {'a': 1}  # hit\n"
        "        assert [v for v in y if v != set(x)]  # hit\n"
        "        assert {k: v == frozenset(y) for k, v in x.items()}  # hit\n"
        "        assert (lambda: x == {1})()  # hit\n"
        "        with pytest.raises(ValueError, match=str(x != {2})):  # hit\n"
        "            pass\n"
        "        assert inner(), x == {3}  # hit\n"
        "\n"
        "@pytest.mark.parametrize('v', [{1} == {1}])  # hit\n"
        "def test_b(v, w=({} == {})):  # hit\n"
        "    assert v\n"
        "\n"
        "async def test_c(x):\n"
        "    async with x as y:\n"
        "        assert y != {4}  # hit\n"
        "\n"
        "assert f'{set() == 1}'  # hit\n"
        "TOP = 3 == set()  # hit\n"
    )
    report, rows, expected = _six(saved_repo, source)
    assert len(expected) == 11 and rows == expected
    assert report.counts[6] == 11


def test_nested_and_same_line_comparisons_give_one_row_per_node(saved_repo):
    head_lines = WHOLE_HEAD.count("\n")
    source = ("def test_a(x, f):\n"
              "    assert f(x == {1}) == {2}\n"
              "    assert x == {1}; assert x != {2}\n")
    report, rows, _ = _six(saved_repo, source)
    first, second = head_lines + 2, head_lines + 3
    assert rows == [f"{WHOLE}\t{RULE_6}: line {first}"] * 2 + [f"{WHOLE}\t{RULE_6}: line {second}"] * 2
    assert report.counts[6] == 4


def test_a_row_of_rule_6_has_the_shape_of_the_rule_5_row_that_holds_a_line(saved_repo):
    source = ("def test_a(x, family, owner_payload_example):\n"
              "    owner_payload_example(family, 'one', {})\n"
              "    assert x == {'a': 1}\n")
    report, _, _ = _six(saved_repo, source)
    base = WHOLE_HEAD.count("\n")
    assert [(h.path, h.rules) for h in report.hints if h.path == WHOLE] == [(WHOLE, (1,)), (WHOLE, (5,)), (WHOLE, (6,))]
    five, six = [h for h in report.hints if h.path == WHOLE][1:]
    assert five == hints.Hint(WHOLE, (5,), unresolved_line=base + 2)
    assert six == hints.Hint(WHOLE, (6,), unresolved_line=base + 3)
    assert (six.keys, six.imports_pinning_test, six.every_example) == (five.keys, five.imports_pinning_test, five.every_example)
    lines = [line for line in hints.format_report(report) if line.startswith(WHOLE)]
    assert lines == [
        f"{WHOLE}\trule 1 (test imports owned module)",
        f"{WHOLE}\t{NOT_RESOLVED}: line {base + 2}",
        f"{WHOLE}\t{RULE_6}: line {base + 3}",
    ]


def test_rules_1_to_5_keep_their_rows_labels_and_counts_when_a_file_also_has_rule_6_rows(saved_repo):
    plain = ("def test_a(x, family, owner_payload_example):\n"
             "    owner_payload_example(family, 'one', {})\n"
             "    owner_payload_example('other', 'three', {})\n")
    before, _, _ = _six(saved_repo, plain)
    lines_before = hints.format_report(before)
    after, rows, expected = _six(saved_repo, plain + "    assert x == {'a': 1}  # hit\n    assert x != set()  # hit\n")
    lines_after = hints.format_report(after)
    assert len(rows) == 2 and rows == expected
    assert [h for h in after.hints if 6 not in h.rules] == list(before.hints)
    assert [line for line in lines_after if RULE_6 not in line and not line.startswith("# rule 6")] == [
        line for line in lines_before if not line.startswith("# rule 6")]
    assert {k: v for k, v in after.counts.items() if k != 6} == {k: v for k, v in before.counts.items() if k != 6}
    assert before.counts[6] == 0 and after.counts[6] == 2
    unresolved = f"{WHOLE}\t{NOT_RESOLVED}: line {WHOLE_HEAD.count(chr(10)) + 2}"
    assert lines_after.index(unresolved) < lines_after.index(rows[0]) < lines_after.index(rows[1])


def test_a_test_file_with_a_syntax_error_gives_no_rule_6_row_and_rules_1_to_5_are_as_before(saved_repo):
    good = ("def test_good(x):\n    assert x == {'a': 1}  # hit\n")
    broken = (f"# {OWNED}\nimport hermes_cli.widget\n\ndef test_bad(x, owner_payload_example):\n"
              "    owner_payload_example('fam', 'one', {})\n    assert x == {'a': 1}\n    assert (\n")
    with pytest.raises(SyntaxError):
        compile(broken, "<case>", "exec")
    _write(saved_repo / "tests" / "test_broken.py", broken)
    report, rows, expected = _six(saved_repo, good)
    assert rows == expected and len(rows) == 1 and report.counts[6] == 1
    assert [h.rules for h in report.hints if h.path == "tests/test_broken.py"] == [(3,)]
    assert report.counts == {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0}
    only_broken = hints.find_hints(saved_repo, [OWNED, WHOLE])
    assert [h for h in only_broken.hints if 6 in h.rules] == []
    assert only_broken.counts == {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}
    assert "# rule 6 (whole-answer comparison): 0" in hints.format_report(only_broken)


def test_a_file_with_a_nul_byte_gives_no_rule_6_row_and_does_not_stop_the_search(saved_repo):
    body = "def test_a(x):\n    assert x == {'a': 1}\n"
    probe = hints.BINARY_PROBE_BYTES
    late = WHOLE_HEAD + "# " + "x" * probe + "\n" + body + "# nul \0 here\n"
    _write(saved_repo / "tests" / "test_late_nul.py", late)
    early = WHOLE_HEAD + body + "# nul \0 here\n"
    _write(saved_repo / "tests" / "test_early_nul.py", early)
    report, rows, expected = _six(saved_repo, "def test_ok(x):\n    assert x == set()  # hit\n")
    assert rows == expected and len(rows) == 1
    assert report.counts[6] == 1
    assert not [h for h in report.hints if h.path in ("tests/test_late_nul.py", "tests/test_early_nul.py") and 6 in h.rules]


def test_multibyte_text_keeps_the_line_of_the_row(saved_repo):
    source = ("# 日本語のコメント ключ ✓\n"
              "def test_a(x):\n"
              "    label = '値 ключ 🚀'; assert x == {'ключ': '値'}  # hit\n"
              "    assert x != {'🚀'}  # hit\n")
    report, rows, expected = _six(saved_repo, source)
    assert len(expected) == 2 and rows == expected and report.counts[6] == 2


def test_a_listed_test_file_with_no_comparison_has_a_zero_count_and_no_row(saved_repo):
    report, rows, _ = _six(saved_repo, "def test_a(x):\n    assert x\n")
    assert rows == [] and report.counts[6] == 0
    assert "# rule 6 (whole-answer comparison): 0" in hints.format_report(report)
    empty = hints.find_hints(saved_repo, [OWNED, WHOLE])
    assert empty.counts[6] == 0 and not [h for h in empty.hints if 6 in h.rules]


def test_a_listed_test_file_with_a_byte_order_mark_gets_the_same_rule_6_rows(tmp_path):
    source = f"# {OWNED}\n" + WHOLE_HEAD + "def test_a(x):\n    assert x == {'a': 1}\n    assert x != set()\n"
    results = {}
    for name, prefix in (("plain", b""), ("bom", b"\xef\xbb\xbf")):
        root = tmp_path / name
        _write(root / ".git" / "config", "")
        _write(root / OWNED, "def build():\n    return 1\n")
        _write(root / SAVED / "fam" / "one.json", "{}\n")
        target = root / "tests" / "test_whole.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(prefix + source.encode("utf-8"))
        report = hints.find_hints(root, [OWNED])
        results[name] = ([h for h in report.hints if 6 in h.rules], report.counts[6])
    assert results["bom"] == results["plain"]
    assert len(results["bom"][0]) == 2 and results["bom"][1] == 2


def test_rule_6_reads_only_the_test_files_that_rule_5_reads(saved_repo):
    compare = "def test_a(x):\n    assert x == {'a': 1}\n"
    call = "def _pin(o):\n    o.owner_payload_example('fam', 'one', {})\n"
    read = "import hermes_cli.widget\n" + call + compare
    # Read (every listed test file, whether or not it calls owner_payload_example): a listed test
    # file with and without the call, and a helper row.
    _write(saved_repo / "tests" / "test_read.py", read)
    _write(saved_repo / "tests" / "a.py", "import hermes_cli.widget\n")
    _write(saved_repo / "tests" / "test_pin_a.py", "import tests.a\n")
    _write(saved_repo / "tests" / "test_helper.py", "from tests.a import helper\n" + call + compare)
    # Not read: not listed; not under tests/; owned by the card.
    _write(saved_repo / "tests" / "test_no_call.py", "import hermes_cli.widget\n" + compare)
    _write(saved_repo / "tests" / "test_unlisted.py", call + compare)
    _write(saved_repo / "lib" / "reader.py", "from hermes_cli.widget import build\n" + call + compare)
    _write(saved_repo / "tests" / "test_owned.py", read)
    report = hints.find_hints(saved_repo, [OWNED, "tests/test_owned.py"])
    assert sorted({h.path for h in report.hints if 6 in h.rules}) == [
        "tests/test_helper.py", "tests/test_no_call.py", "tests/test_read.py"]
    assert report.counts[6] == 3
    assert not [h for h in report.hints if 5 in h.rules and h.unresolved_line]
    assert _lines(report)[f"{SAVED}/fam/one.json"] == SAVED_LABEL


def test_the_count_line_of_rule_6_equals_the_number_of_its_rows(saved_repo, capsys):
    _write(saved_repo / "tests" / "test_other.py",
           "import hermes_cli.widget\n\ndef t(o, x):\n    o.owner_payload_example('fam', 'one', {})\n"
           "    assert x == frozenset()\n")
    report, rows, expected = _six(saved_repo, (
        "def test_a(x):\n    assert x == {'a': 1}  # hit\n    assert x != {2}  # hit\n    assert x == set(x)  # hit\n"))
    assert rows[1:] == expected and len(rows) == 4
    out = hints.format_report(report)
    all_rows = [line for line in out if RULE_6 in line and not line.startswith("#")]
    assert len(all_rows) == 4 == report.counts[6] == len(_six_hints(report))
    assert [line for line in out if line.startswith("# rule 6")] == ["# rule 6 (whole-answer comparison): 4"]
    assert out[-2] == "# rule 6 (whole-answer comparison): 4"
    assert report.counts == {1: 2, 2: 0, 3: 0, 4: 0, 5: 2, 6: 4, 7: 0}
    # The reader prints the same lines.
    script = _load_script()
    assert script.main(["--root", str(saved_repo), "--owned", OWNED]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == out


def test_the_search_runs_and_imports_no_test_file_that_it_reads_for_rule_6(saved_repo, tmp_path):
    marker = tmp_path / "ran.txt"
    source = (f"open({str(marker)!r}, 'w').write('ran')\nimport no_such_module_for_rule_6\nraise SystemExit(3)\n"
              "def test_a(x):\n    assert x == {'a': 1}  # hit\n")
    _write(saved_repo / "tests" / "test_whole.py", WHOLE_HEAD + source)
    snapshot, before = _snapshot(saved_repo), set(sys.modules)
    report = hints.find_hints(saved_repo, [OWNED])
    assert report.counts[6] == 1
    assert not marker.exists()
    assert not [m for m in set(sys.modules) - before if "test_whole" in m or "no_such_module_for_rule_6" in m]
    assert _snapshot(saved_repo) == snapshot


def test_the_rule_6_rows_of_a_helper_row_come_after_its_helper_row_and_rule_5_rows_stay_put(saved_repo):
    _write(saved_repo / "tests" / "a.py", "import hermes_cli.widget\n")
    _write(saved_repo / "tests" / "test_pin_a.py", "import tests.a\n")
    _write(saved_repo / "tests" / "b.py",
           "from tests.a import helper\n\ndef t(o, f, x):\n    o.owner_payload_example(f, 'one', {})\n"
           "    assert x == {1}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    own = [line for line in hints.format_report(report) if line.startswith("tests/b.py")]
    assert own == [
        f"tests/b.py\t{NOT_RESOLVED}: line 4",
        f"tests/b.py\t{HELPER_LABEL}",
        f"tests/b.py\t{RULE_6}: line 5",
    ]
    assert report.counts[6] == 1


# Card 4, rule 7: each call of a named function (``--symbol NAME``): the plain name ``NAME(...)`` or
# an attribute that ends in ``NAME`` (``x.NAME(...)``). A closed list; no name is resolved.
RULE_7 = "rule 7 (calls a named function"
COUNT_7 = "# rule 7 (calls a named function): "
SYM = "target_fn"


@pytest.fixture
def calls_repo(tmp_path):
    root = tmp_path / "calls_repo"
    _write(root / ".git" / "config", "")
    _write(root / "hermes_cli" / "__init__.py", "")
    _write(root / OWNED, "def build():\n    return 1\n")
    return root


def _src(text: str, symbol: str = SYM) -> str:
    return text.replace("NAME", symbol)


def _row_line(path: str, symbol: str, line: int) -> str:
    return f"{path}\t{RULE_7}: {symbol}): line {line}"


def _expected_rows(path: str, source: str, symbol: str = SYM) -> list[str]:
    """One row per ``# hit`` line, two for ``# hit x2``."""
    rows = []
    for number, line in enumerate(source.splitlines(), 1):
        times = 2 if "# hit x2" in line else 1 if "# hit" in line else 0
        rows.extend([_row_line(path, symbol, number)] * times)
    return rows


def _call_rows(report) -> list[str]:
    return [line for line in hints.format_report(report) if f"{RULE_7}: " in line and not line.startswith("#")]


def _count_7(report) -> str:
    return next(line for line in hints.format_report(report) if line.startswith("# rule 7"))


def _without_rule_7(lines: list[str]) -> list[str]:
    return [line for line in lines if f"{RULE_7}: " not in line and not line.startswith("# rule 7")]


def _calls(root, source: str, *, path: str = "lib/mod.py", symbols=(SYM,), owned=(OWNED,), raw: bytes | None = None):
    target = root / path
    if raw is None:
        _write(target, source)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return hints.find_hints(root, list(owned), symbols=list(symbols))


def test_rule_7_is_named_and_has_its_count_line_last():
    assert hints.RULE_NAMES[7] == "calls a named function"
    report = hints.Report((), {rule: 0 for rule in hints.RULE_NAMES})
    assert hints.format_report(report)[-1] == COUNT_7 + "0"


@pytest.mark.parametrize("path", ["tests/test_calls.py", "lib/mod.py"])
def test_a_plain_call_and_two_attribute_calls_give_one_row_each_with_their_line(calls_repo, path):
    source = _src("import os\n\nNAME(1)  # hit\nmod.NAME(2)  # hit\nobj.part.NAME(3)  # hit\n"
                  "other(4)\nmod.other(5)\nNAME_2(6)\nxNAME(7)\nobj.NAME_2(8)\n")
    report = _calls(calls_repo, source, path=path)
    assert _call_rows(report) == _expected_rows(path, source) and len(_call_rows(report)) == 3
    assert report.counts[7] == 3 and _count_7(report) == COUNT_7 + "3"
    assert _paths(report) == [path, path, path]
    assert all(h.rules == (7,) for h in report.hints)


def test_a_call_in_a_nested_function_a_class_a_lambda_and_a_comprehension_gives_a_row(calls_repo):
    source = _src(
        "def outer():\n"
        "    def inner():\n"
        "        NAME(1)  # hit\n"
        "    class K:\n"
        "        attr = mod.NAME(2)  # hit\n"
        "        def m(self):\n"
        "            return self.NAME(3)  # hit\n"
        "    return lambda: NAME(4)  # hit\n"
        "values = [NAME(i) for i in items]  # hit\n"
        "pairs = {k: obj.NAME(v) for k, v in d}  # hit\n"
        "gen = (x for x in NAME(5))  # hit\n"
        "if cond(NAME(6)):  # hit\n"
        "    pass\n"
        "async def coro():\n"
        "    return await NAME(7)  # hit\n"
        "text = f'{NAME(8)}'  # hit\n")
    report = _calls(calls_repo, source)
    assert _call_rows(report) == _expected_rows("lib/mod.py", source) and report.counts[7] == 10


NOT_CALLS = {
    "import alias": "from m import NAME as other\nother(1)\n",
    "variable holding the function": "f = NAME\nf(1)\n",
    "name passed as an argument": "map(NAME, items)\n",
    "decorator without parentheses": "@NAME\ndef f():\n    pass\n",
    "attribute decorator without parentheses": "@mod.NAME\ndef f():\n    pass\n",
    "getattr": "getattr(obj, 'NAME')()\n",
    "definition": "def NAME():\n    pass\n",
    "class definition": "class NAME:\n    pass\n",
    "import": "import NAME\nfrom m import NAME\n",
    "attribute without a call": "x = obj.NAME\ny = obj.part.NAME\n",
    "string": "s = 'NAME(1)'\nt = \"mod.NAME(2)\"\n",
    "docstring": "def f():\n    '''Calls NAME(1) and mod.NAME(2).'''\n",
    "comment": "# NAME(1)\nx = 1  # mod.NAME(2)\n",
    "subscript function": "NAME[0](1)\nobj.NAME[0](2)\n",
    "boolean function": "(NAME or other)(1)\n",
    "attribute of the name": "NAME.attr(1)\n",
    "keyword argument": "f(NAME=1)\n",
    "assignment": "NAME = 1\n",
    "other name": "NAME_2(1)\nxNAME(2)\nobj.NAME_2(3)\n",
}


@pytest.mark.parametrize("source", list(NOT_CALLS.values()), ids=list(NOT_CALLS))
def test_a_form_that_the_closed_list_does_not_hold_gives_no_rule_7_row(calls_repo, source):
    source = _src(source)
    compile(source, "<case>", "exec")
    report = _calls(calls_repo, source)
    assert _call_rows(report) == [] and report.counts[7] == 0 and report.hints == ()
    assert _count_7(report) == COUNT_7 + "0"


@pytest.mark.parametrize("decorator", ["@NAME(1)", "@mod.NAME(1)"])
@pytest.mark.parametrize("target", ["def f():\n    pass\n", "class C:\n    pass\n"])
def test_a_decorator_with_parentheses_is_a_call_and_gives_one_row(calls_repo, decorator, target):
    source = _src("x = 1\n" + decorator + "  # hit\n" + target)
    report = _calls(calls_repo, source)
    assert _call_rows(report) == _expected_rows("lib/mod.py", source) and report.counts[7] == 1


def test_a_call_of_a_call_counts_the_inner_call_only(calls_repo):
    source = _src("NAME(1)(2)  # hit\n")
    report = _calls(calls_repo, source)
    assert _call_rows(report) == _expected_rows("lib/mod.py", source) and report.counts[7] == 1


@pytest.mark.parametrize("line", [
    "NAME(1); NAME(2)  # hit x2",
    "x = NAME(1) + mod.NAME(2)  # hit x2",
    "NAME(NAME(1))  # hit x2",
    "a.NAME(1).NAME(2)  # hit x2",
])
def test_two_calls_on_one_line_give_two_rows(calls_repo, line):
    source = _src("x = 0\n" + line + "\n")
    report = _calls(calls_repo, source)
    assert _call_rows(report) == [_row_line("lib/mod.py", SYM, 2)] * 2 and report.counts[7] == 2


def test_a_call_that_runs_over_several_lines_gives_one_row_with_the_line_of_the_node(calls_repo):
    source = _src("x = 0\nobj.\\\n    NAME(  # hit\n        1,\n        2)\n")
    report = _calls(calls_repo, source)
    assert report.counts[7] == 1
    assert _call_rows(report) == [_row_line("lib/mod.py", SYM, 2)]


def test_a_call_in_an_owned_file_or_owned_folder_gives_no_row(calls_repo):
    _write(calls_repo / "hermes_cli" / "owned_dir" / "inner.py", _src("NAME(1)\nmod.NAME(2)\n"))
    _write(calls_repo / OWNED, _src("def build():\n    return NAME(1)\n"))
    _write(calls_repo / "lib" / "mod.py", _src("NAME(1)  # outside\n"))
    report = hints.find_hints(calls_repo, [OWNED, "hermes_cli/owned_dir"], symbols=[SYM])
    assert _call_rows(report) == [_row_line("lib/mod.py", SYM, 1)] and report.counts[7] == 1
    assert _paths(report) == ["lib/mod.py"]
    whole = hints.find_hints(calls_repo, ["."], symbols=[SYM])
    assert whole.hints == () and whole.counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}


def test_a_file_with_a_syntax_error_gives_no_rule_7_row_and_rules_1_to_6_are_as_before(calls_repo):
    broken = _src(f"# {OWNED}\nimport hermes_cli.widget\nNAME(1)\ndef (:\n")
    with pytest.raises(SyntaxError):
        compile(broken, "<case>", "exec")
    _write(calls_repo / "tests" / "test_broken.py", broken)
    _write(calls_repo / "lib" / "ok.py", _src("NAME(1)  # hit\n"))
    plain = hints.find_hints(calls_repo, [OWNED])
    report = hints.find_hints(calls_repo, [OWNED], symbols=[SYM])
    assert _call_rows(report) == [_row_line("lib/ok.py", SYM, 1)] and report.counts[7] == 1
    assert "tests/test_broken.py" in _paths(plain) and "tests/test_broken.py" in _paths(report)
    assert not [h for h in report.hints if h.path == "tests/test_broken.py" and 7 in h.rules]
    assert _without_rule_7(hints.format_report(report)) == _without_rule_7(hints.format_report(plain))


@pytest.mark.parametrize("path", ["tests/test_bom.py", "lib/mod.py"])
def test_a_file_with_a_leading_byte_order_mark_and_a_call_gives_its_row(calls_repo, path):
    source = _src("x = 0\nNAME(1)  # hit\nmod.NAME(2)  # hit\n")
    with pytest.raises((SyntaxError, ValueError)):
        ast.parse("\ufeff" + source)
    report = _calls(calls_repo, "", path=path, raw=b"\xef\xbb\xbf" + source.encode("utf-8"))
    assert _call_rows(report) == _expected_rows(path, source) and report.counts[7] == 2


def test_a_file_with_a_byte_order_mark_and_a_syntax_error_gives_no_row(calls_repo):
    raw = b"\xef\xbb\xbf" + _src("NAME(1)\ndef (:\n").encode("utf-8")
    report = _calls(calls_repo, "", path="lib/mod.py", raw=raw)
    assert _call_rows(report) == [] and report.counts[7] == 0 and report.hints == ()


def test_a_byte_order_mark_gives_the_same_rule_7_row_and_a_symbol_changes_no_row_of_rules_1_to_6(calls_repo):
    source = f"# {OWNED}\nimport hermes_cli.widget\n\ndef test_a(x):\n    assert x == {{'a': 1}}\n    NAME(1)\n"
    results = {}
    for name, prefix in (("plain", b""), ("bom", b"\xef\xbb\xbf")):
        root = calls_repo.parent / name
        _write(root / ".git" / "config", "")
        _write(root / OWNED, "def build():\n    return 1\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_x.py").write_bytes(prefix + _src(source).encode("utf-8"))
        with_symbol = hints.find_hints(root, [OWNED], symbols=[SYM])
        without = hints.find_hints(root, [OWNED])
        assert _without_rule_7(hints.format_report(with_symbol)) == _without_rule_7(hints.format_report(without))
        results[name] = _call_rows(with_symbol)
    assert results["bom"] == results["plain"] == [_row_line("tests/test_x.py", SYM, 6)]


def test_two_symbols_give_the_rows_of_each_and_a_repeated_symbol_gives_each_row_one_time(calls_repo):
    source = ("alpha_fn(1)  # hit\nbeta_fn(2)\nobj.alpha_fn(3)  # hit\nmod.beta_fn(4)\ngamma_fn(5)\n")
    expected = [_row_line("lib/mod.py", "alpha_fn", 1), _row_line("lib/mod.py", "beta_fn", 2),
                _row_line("lib/mod.py", "alpha_fn", 3), _row_line("lib/mod.py", "beta_fn", 4)]
    once = _calls(calls_repo, source, symbols=["alpha_fn", "beta_fn"])
    assert _call_rows(once) == expected and once.counts[7] == 4
    again = hints.find_hints(calls_repo, [OWNED], symbols=["alpha_fn", "beta_fn", "alpha_fn", "beta_fn"])
    assert _call_rows(again) == expected and again.counts[7] == 4 and again.hints == once.hints
    reverse = hints.find_hints(calls_repo, [OWNED], symbols=["beta_fn", "alpha_fn", "beta_fn"])
    assert _call_rows(reverse) == expected and reverse.counts[7] == 4
    only = hints.find_hints(calls_repo, [OWNED], symbols=["alpha_fn"])
    assert _call_rows(only) == [expected[0], expected[2]] and only.counts[7] == 2


def test_a_symbol_that_is_never_called_gives_no_row_and_the_count_zero(calls_repo):
    report = _calls(calls_repo, _src("NAME(1)\nmod.NAME(2)\n"), symbols=["never_called"])
    assert _call_rows(report) == [] and report.counts[7] == 0 and report.hints == ()
    assert _count_7(report) == COUNT_7 + "0"


def test_an_empty_file_and_an_empty_symbol_list_give_no_row(calls_repo):
    report = _calls(calls_repo, "", symbols=[SYM])
    assert report.hints == () and report.counts[7] == 0
    assert hints.find_hints(calls_repo, [OWNED], symbols=[]).counts[7] == 0
    assert hints.find_hints(calls_repo, [OWNED], symbols=()).counts[7] == 0


REFUSED = ["", "a.b", "1x", "class", "ﬁnd", "None", "a b", " a", "a\n", "a\x00b", "a-b", "a(", "ｆind"]


@pytest.fixture
def no_file_is_read(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a file was read before the symbol was refused")

    monkeypatch.setattr(hints, "_examine", refuse)
    monkeypatch.setattr(hints, "_walk_files", refuse)


@pytest.mark.parametrize("symbol", REFUSED)
@pytest.mark.parametrize("owned", [[OWNED], ["."]])
def test_a_symbol_that_is_not_a_plain_nfkc_name_is_refused_before_a_file_is_read(calls_repo, no_file_is_read, symbol, owned):
    with pytest.raises(ValueError):
        hints.find_hints(calls_repo, owned, symbols=[symbol])
    with pytest.raises(ValueError):
        hints.find_hints(calls_repo, owned, symbols=[SYM, symbol])


@pytest.mark.parametrize("symbol", [s for s in REFUSED if "\x00" not in s])
def test_a_refused_symbol_gives_exit_2_and_nothing_on_standard_output_through_the_script(calls_repo, symbol):
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(calls_repo), "--owned", OWNED, "--symbol", SYM, "--symbol", symbol],
        env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 2 and run.stdout == ""
    assert run.stderr.startswith("kanban-scope-hints:")


def test_the_script_refuses_a_symbol_with_owned_dot_and_prints_nothing(calls_repo, capsys):
    script = _load_script()
    assert script.main(["--root", str(calls_repo), "--owned", ".", "--symbol", "a.b"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "a.b" in captured.err


@pytest.mark.parametrize("symbol", ["größe", "名前", "_", "__x__", "match", "print", "x1"])
def test_a_non_ascii_symbol_in_nfkc_form_and_soft_keywords_are_admitted_and_found(calls_repo, symbol):
    source = f"import os\n{symbol}(1)  # hit\nobj.{symbol}(2)  # hit\n"
    report = _calls(calls_repo, source, symbols=[symbol])
    assert _call_rows(report) == _expected_rows("lib/mod.py", source, symbol) and report.counts[7] == 2


def test_multibyte_text_before_a_call_keeps_the_line_of_the_row(calls_repo):
    source = "# größe 名前 ünï\ns = 'ß名\U0001f600'\n\ngröße(1)  # hit\n"
    report = _calls(calls_repo, source, symbols=["größe"])
    assert _call_rows(report) == [_row_line("lib/mod.py", "größe", 4)]


def test_a_call_written_in_a_compatibility_form_is_found_by_its_nfkc_symbol(calls_repo):
    source = "\ufb01nd(1)\nobj.\ufb01nd(2)\n"
    assert unicodedata.normalize("NFKC", "\ufb01nd") == "find"
    report = _calls(calls_repo, source, symbols=["find"])
    assert _call_rows(report) == [_row_line("lib/mod.py", "find", 1), _row_line("lib/mod.py", "find", 2)]


def test_with_no_symbol_the_output_has_no_rule_7_row_and_the_count_line_zero(calls_repo, capsys):
    _write(calls_repo / "lib" / "mod.py", _src("NAME(1)\nmod.NAME(2)\n"))
    report = hints.find_hints(calls_repo, [OWNED])
    assert report.hints == () and report.counts[7] == 0
    assert hints.format_report(report)[-1] == COUNT_7 + "0"
    assert hints.find_hints(calls_repo, [OWNED], ()).counts[7] == 0
    assert hints.find_hints(calls_repo, [OWNED]) == hints.find_hints(calls_repo, [OWNED], symbols=())
    script = _load_script()
    assert script.main(["--root", str(calls_repo), "--owned", OWNED]) == 0
    out = capsys.readouterr().out.splitlines()
    assert not [line for line in out if f"{RULE_7}: " in line]
    assert out[-1] == COUNT_7 + "0"


def _everything_repo(root: Path) -> None:
    _write(root / ".git" / "config", f"[x]\n\tpath = {OWNED}\n\tkey = '{KEY}'\n")
    _write(root / "hermes_cli" / "__init__.py", "")
    _write(root / OWNED, f"def build():\n    return '{KEY}'\n")
    _write(root / SAVED / "fam" / "one.json", "{}\n")
    _write(root / SAVED / "fam" / "two.json", "{}\n")
    _write(root / "hermes_cli" / "reader.py", _src("from hermes_cli.widget import build\nNAME(1)\n"))
    _write(root / "scripts" / "tool.py", _src(f"from hermes_cli.widget import build\nP = '{OWNED}'\nobj.NAME(2)\n"))
    _write(root / "tests" / "test_pin.py", _src(
        f"import hermes_cli.widget\n\ndef test_a(owner_payload_example, f, x):\n"
        f"    owner_payload_example('fam', 'one', {{}})\n    owner_payload_example(f, 'one', {{}})\n"
        f"    assert x == {{'a': 1}}\n    assert x != set()\n    NAME(3)\n    assert x == '{KEY}'\n"))
    _write(root / "tests" / "a.py", "import hermes_cli.widget\n")
    _write(root / "tests" / "test_pin_a.py", "import tests.a\n")
    _write(root / "tests" / "b.py", _src("from tests.a import helper\nmod.NAME(4)\n"))
    _write(root / "docs" / "notes.md", f"See `{OWNED}`; NAME(5) '{KEY}'.\n")
    _write(root / "lib" / "only_calls.py", _src("def f(o, x):\n    o.owner_payload_example('fam', 'one', {})\n"
                                               "    assert x == {1}\n    return NAME(6)\n"))


def test_rules_1_to_6_keep_their_rows_labels_and_counts_with_a_symbol_and_without(tmp_path):
    root = tmp_path / "all"
    _everything_repo(root)
    plain = hints.find_hints(root, [OWNED], [KEY])
    with_symbol = hints.find_hints(root, [OWNED], [KEY], symbols=[SYM])
    assert plain.counts[7] == 0 and with_symbol.counts[7] == 5
    assert {k: v for k, v in with_symbol.counts.items() if k != 7} == {k: v for k, v in plain.counts.items() if k != 7}
    assert _without_rule_7(hints.format_report(with_symbol)) == _without_rule_7(hints.format_report(plain))
    assert [h for h in with_symbol.hints if h.rules != (7,)] == list(plain.hints)
    assert plain.counts[5] >= 2 and plain.counts[6] >= 2
    with_two = hints.find_hints(root, [OWNED], [KEY], symbols=[SYM, "other_name"])
    assert _without_rule_7(hints.format_report(with_two)) == _without_rule_7(hints.format_report(plain))


def test_a_file_that_only_rule_7_names_is_not_a_listed_file_for_rules_5_and_6(tmp_path):
    root = tmp_path / "all"
    _everything_repo(root)
    plain = hints.find_hints(root, [OWNED], [KEY])
    assert "lib/only_calls.py" not in _paths(plain)
    report = hints.find_hints(root, [OWNED], [KEY], symbols=[SYM])
    assert [h.rules for h in report.hints if h.path == "lib/only_calls.py"] == [(7,)]
    assert report.counts[5] == plain.counts[5] and report.counts[6] == plain.counts[6]


def test_the_rows_of_rule_7_of_a_file_come_after_each_other_row_of_that_file_in_node_order(tmp_path):
    root = tmp_path / "all"
    _everything_repo(root)
    _write(root / "tests" / "b.py", _src(
        "from tests.a import helper\nNAME(0)\n\ndef t(o, f, x):\n    o.owner_payload_example(f, 'one', {})\n"
        "    assert x == {1}\n    NAME(1); NAME(2)\n    return NAME(3)\n"))
    report = hints.find_hints(root, [OWNED], symbols=[SYM])
    own = [line for line in hints.format_report(report) if line.startswith("tests/b.py")]
    assert own == [
        f"tests/b.py\t{NOT_RESOLVED}: line 5",
        f"tests/b.py\t{HELPER_LABEL}",
        f"tests/b.py\t{RULE_6}: line 6",
        _row_line("tests/b.py", SYM, 2),
        _row_line("tests/b.py", SYM, 7),
        _row_line("tests/b.py", SYM, 7),
        _row_line("tests/b.py", SYM, 8),
    ]
    lines = [line for line in hints.format_report(report) if line.startswith(("tests/test_pin.py", "scripts/tool.py"))]
    pin = [line for line in lines if line.startswith("tests/test_pin.py")]
    assert pin[-1].startswith("tests/test_pin.py\trule 7") and all("rule 7" not in line for line in pin[:-1])
    tool = [line for line in lines if line.startswith("scripts/tool.py")]
    assert tool[0] == f"scripts/tool.py\trule 2 (imports name from owned module); rule 3 (names owned path)"
    assert tool[1] == _row_line("scripts/tool.py", SYM, 3) and len(tool) == 2


def test_a_row_of_rule_7_has_its_symbol_and_line(calls_repo):
    report = _calls(calls_repo, _src("x = 1\nNAME(1)\n"))
    assert report.hints == (hints.Hint("lib/mod.py", (7,), symbol=SYM, unresolved_line=2),)


def test_the_count_line_of_rule_7_equals_the_number_of_its_rows(tmp_path, capsys):
    root = tmp_path / "all"
    _everything_repo(root)
    report = hints.find_hints(root, [OWNED], symbols=[SYM])
    rows = [h for h in report.hints if 7 in h.rules]
    assert len(rows) == report.counts[7] == len(_call_rows(report)) == 5
    assert [line for line in hints.format_report(report) if line.startswith("# rule 7")] == [COUNT_7 + "5"]
    script = _load_script()
    assert script.main(["--root", str(root), "--owned", OWNED, "--symbol", SYM]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == hints.format_report(report) and printed[-1] == COUNT_7 + "5"
    assert script.main(["--root", str(root), "--owned", OWNED, "--symbol", SYM, "--symbol", "other_name", "--symbol", SYM]) == 0
    assert capsys.readouterr().out.splitlines() == printed


def test_rule_7_reads_only_the_python_files_that_the_scan_reads(calls_repo, tmp_path):
    _write(calls_repo / "scripts" / "run.sh", _src("NAME(1)\n"))
    _write(calls_repo / "lib" / "stub.pyi", _src("NAME(1)\n"))
    _write(calls_repo / "docs" / "notes.md", _src("NAME(1)\n"))
    _write(calls_repo / "node_modules" / "dep.py", _src("NAME(1)\n"))
    _write(calls_repo / "venv" / "dep.py", _src("NAME(1)\n"))
    _write(tmp_path / "outside" / "out.py", _src("NAME(1)\n"))
    (calls_repo / "lib" / "link.py").parent.mkdir(exist_ok=True)
    (calls_repo / "lib" / "link.py").symlink_to(tmp_path / "outside" / "out.py")
    (calls_repo / "lib" / "dangling.py").symlink_to(tmp_path / "outside" / "missing.py")
    _write(calls_repo / "lib" / "real.py", _src("NAME(1)\n"))
    report = hints.find_hints(calls_repo, [OWNED], symbols=[SYM])
    assert _call_rows(report) == [_row_line("lib/real.py", SYM, 1)] and report.counts[7] == 1


def test_a_binary_file_with_a_nul_byte_and_a_file_that_is_not_utf_8_give_no_row(calls_repo):
    _write(calls_repo / "lib" / "good.py", _src("NAME(1)\n"))
    (calls_repo / "lib" / "nul.py").write_bytes(b"x = 1\n\x00\n" + _src("NAME(1)\n").encode())
    (calls_repo / "lib" / "latin.py").write_bytes(b"# \xe9\n" + _src("NAME(1)\n").encode())
    report = hints.find_hints(calls_repo, [OWNED], symbols=[SYM])
    assert _call_rows(report) == [_row_line("lib/good.py", SYM, 1)] and report.counts[7] == 1
    assert report.unreadable == ("lib/latin.py",)
    assert hints.format_report(report)[-1] == "# unreadable: 1"
    assert hints.find_hints(calls_repo, [OWNED]).unreadable == ("lib/latin.py",)


def test_a_nul_character_after_the_probe_gives_no_row_and_does_not_stop_the_search(calls_repo):
    _write(calls_repo / "lib" / "good.py", _src("NAME(1)\n"))
    padding = "# " + "x" * 9000 + "\n"
    (calls_repo / "lib" / "late_nul.py").write_bytes((padding + _src("NAME(1)\nNAME(2)\n") + "s = '\x00'\n").encode())
    report = hints.find_hints(calls_repo, [OWNED], symbols=[SYM])
    assert _call_rows(report) == [_row_line("lib/good.py", SYM, 1)] and report.counts[7] == 1


def test_the_search_runs_and_imports_no_file_that_it_reads_for_rule_7(calls_repo, tmp_path):
    marker = tmp_path / "ran.txt"
    source = _src(f"open({str(marker)!r}, 'w').write('ran')\nimport no_such_module_for_rule_7\nraise SystemExit(3)\nNAME(1)\n")
    _write(calls_repo / "lib" / "runs.py", source)
    snapshot, before = _snapshot(calls_repo), set(sys.modules)
    report = hints.find_hints(calls_repo, [OWNED], symbols=[SYM])
    assert report.counts[7] == 1
    assert not marker.exists()
    assert not [m for m in set(sys.modules) - before if "runs" in m or "no_such_module_for_rule_7" in m]
    assert _snapshot(calls_repo) == snapshot


def test_the_script_passes_each_symbol_and_writes_nothing_into_the_repository(calls_repo, capsys):
    _write(calls_repo / "lib" / "mod.py", "alpha_fn(1)\nobj.beta_fn(2)\n")
    snapshot = _snapshot(calls_repo)
    script = _load_script()
    assert script.main(["--root", str(calls_repo), "--owned", OWNED, "--symbol", "alpha_fn", "--symbol", "beta_fn"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == [_row_line("lib/mod.py", "alpha_fn", 1), _row_line("lib/mod.py", "beta_fn", 2)]
    assert out[-1] == COUNT_7 + "2"
    assert _snapshot(calls_repo) == snapshot


def test_the_call_of_find_hints_in_the_script_is_listed_with_its_line_on_a_copy_of_the_repository_files(tmp_path):
    root = tmp_path / "copy"
    module = Path(hints.__file__).resolve()
    _write(root / ".git" / "config", "")
    for source in (module, SCRIPT):
        _write(root / source.relative_to(module.parents[1]), source.read_text(encoding="utf-8"))
    script_text = (root / "scripts" / "kanban_scope_hints.py").read_text(encoding="utf-8")
    lines = [n.lineno for n in ast.walk(ast.parse(script_text))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "find_hints"]
    assert len(lines) == 1
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--owned", "hermes_cli/kanban_scope_hints.py",
         "--symbol", "find_hints"],
        env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    rows = [line for line in run.stdout.splitlines() if line.startswith("scripts/kanban_scope_hints.py\t") and RULE_7 in line]
    assert rows == [f"scripts/kanban_scope_hints.py\t{RULE_7}: find_hints): line {lines[0]}"]
    assert run.stdout.splitlines()[-1] == COUNT_7 + "1"
