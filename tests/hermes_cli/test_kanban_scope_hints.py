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
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 3, 5: 0}
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
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 4, 5: 0}

    # Through the script: same order, every rule on the line, one count per rule.
    script = _load_script()
    code = script.main(["--root", str(repo), "--owned", OWNED, "--key", KEY])
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == _paths(report)
    by_path = {line.split("\t")[0]: line.split("\t")[1] for line in out if line and not line.startswith("#")}
    assert "rule 1" in by_path["tests/test_pin_key.py"] and "rule 4" in by_path["tests/test_pin_key.py"]
    counts = [line for line in out if line.startswith("#")]
    assert len(counts) == 5
    assert [line.split(":")[-1].strip() for line in counts] == ["2", "2", "2", "4", "0"]
    assert all(f"rule {n}" in line for n, line in zip((1, 2, 3, 4, 5), counts))

    # Without a key all five counts are still printed, rule 4 at zero.
    assert script.main(["--root", str(repo), OWNED]) == 0
    counts = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert len(counts) == 5 and counts[3].endswith(": 0")


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
    assert by_file.counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 0}
    for folder in ("pkg", "pkg/", str(root / "pkg")):
        by_folder = hints.find_hints(root, [folder], [KEY])
        assert _paths(by_folder) == _paths(by_file)
        assert _rules(by_folder) == expected
        assert by_folder.counts == by_file.counts

    # The whole repository as the owned scope owns every file: nothing to list, all counts zero.
    for whole in (".", "./", str(root)):
        report = hints.find_hints(root, [whole], [KEY])
        assert report.hints == () and report.counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    script = _load_script()
    assert script.main(["--root", str(root), "--owned", ".", "--key", KEY]) == 0
    out = capsys.readouterr().out.splitlines()
    assert _output_paths("\n".join(out)) == []
    assert [line.split(":")[-1].strip() for line in out] == ["0", "0", "0", "0", "0"]


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
    # The key matches are still first, and the five rule counts are still printed.
    assert _output_paths("\n".join(out))[:3] == ["config/settings.json", "docs/notes.md", "tests/test_pin_key.py"]
    assert len([line for line in out if line.startswith("# rule")]) == 5


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
    assert report.counts == {1: 2, 2: 0, 3: 0, 4: 1, 5: 0}


def test_helper_naming_the_owned_path_keeps_its_rule_3_row_and_the_counts(helper_repo):
    _write(helper_repo / "tests" / "b.py", f'from tests.a import helper\nP = "{OWNED}"\n')
    report = hints.find_hints(helper_repo, [OWNED])
    rows = [line for line in hints.format_report(report) if line.startswith("tests/b.py\t")]
    assert rows == ["tests/b.py\trule 3 (names owned path)", f"tests/b.py\t{HELPER_LABEL}"]
    assert [h.rules for h in report.hints if h.path == "tests/b.py"] == [(3,), ()]
    assert report.counts == {1: 2, 2: 0, 3: 1, 4: 0, 5: 0}


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


def test_a_literal_reading_wins_over_the_family_folder_and_a_file_is_listed_once(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    owner_payload_example('fam', 'one', {})\n"
           "    return {k: owner_payload_example('fam', k, a) for k, a in answers.items()}\n")
    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n"
           "    owner_payload_example('fam', 'one', {})\n    owner_payload_example('fam', 'two', {})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert _saved_rows(report) == {
        f"{SAVED}/fam/one.json": SAVED_LABEL,
        f"{SAVED}/fam/two.json": SAVED_LABEL,
    }
    assert [p for p in _paths(report) if p.startswith(SAVED)] == [f"{SAVED}/fam/one.json", f"{SAVED}/fam/two.json"]


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
    assert plain_lines[0] == own and plain.counts == {1: 0, 2: 0, 3: 0, 4: 1, 5: 0}

    _write(saved_repo / "tests" / "test_lit.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example):\n    owner_payload_example('fam', 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED], ["unit_running"])
    lines = hints.format_report(report)
    assert lines[:2] == [own, f"{SAVED}/fam/one.json\t{SAVED_LABEL}"]
    assert [h.rules for h in report.hints[:2]] == [(4,), (5,)]
    assert [h.keys for h in report.hints[:2]] == [("unit_running",), ()]
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 1, 5: 1}
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
    assert report.counts == {1: 2, 2: 0, 3: 0, 4: 0, 5: 1}


def test_rule_5_has_its_own_count_line_after_the_four_of_today(saved_repo):
    _write(saved_repo / "tests" / "test_loop.py",
           "import hermes_cli.widget\n\n"
           "def test_a(owner_payload_example, answers):\n"
           "    owner_payload_example('other', 'three', {})\n"
           "    return {k: owner_payload_example('fam', k, a) for k, a in answers.items()}\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 0, 5: 3}
    assert hints.RULE_NAMES[5] == "saved example"
    counts = [line for line in hints.format_report(report) if line.startswith("#")]
    assert counts == [
        "# rule 1 (test imports owned module): 1",
        "# rule 2 (imports name from owned module): 0",
        "# rule 3 (names owned path): 0",
        "# rule 4 (quoted key): 0",
        "# rule 5 (saved example): 3",
    ]
    assert hints.find_hints(saved_repo, ["."]).counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}


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


@pytest.mark.parametrize("binding", SECOND_BINDINGS)
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
    assert report.counts == {1: 1, 2: 0, 3: 0, 4: 0, 5: 4}


def test_a_test_file_that_is_not_listed_gets_no_not_resolved_row(saved_repo):
    _write(saved_repo / "tests" / "test_unlisted.py", "def t(o, f):\n    o.owner_payload_example(f, 'one', {})\n")
    _write(saved_repo / "lib" / "reader.py", "from hermes_cli.widget import build\nowner_payload_example(f, 'one', {})\n")
    report = hints.find_hints(saved_repo, [OWNED])
    assert not any(NOT_RESOLVED in line for line in hints.format_report(report))
    assert report.counts[5] == 0
