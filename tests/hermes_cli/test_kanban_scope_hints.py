"""The scope hints search lists the tests and readers of the paths a proposed card will own.

Every test builds a temporary repository tree under ``tmp_path``; nothing reads the
real repository or the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
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
    spec.loader.exec_module(module)
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
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--owned", OWNED],
        cwd=repo / "hermes_cli", env=env, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    printed = _output_paths(run.stdout)
    assert "tests/test_pin_key.py" in printed and "tests/test_pin_plain.py" in printed
    assert "tests/test_unrelated.py" not in printed


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


def test_already_owned_file_is_never_listed(repo):
    owned = [OWNED, "hermes_cli/reader.py", "tests/test_pin_key.py", "scripts"]
    report = hints.find_hints(repo, owned, [KEY])
    listed = _paths(report)
    for path in (OWNED, "hermes_cli/reader.py", "tests/test_pin_key.py", "scripts/tool.py"):
        assert path not in listed
    assert "tests/test_pin_plain.py" in listed
    assert "docs/notes.md" in listed


def test_key_lists_exactly_the_files_with_that_quoted_key_first_and_prints_counts(repo, capsys):
    report = hints.find_hints(repo, [OWNED], [KEY])
    with_key = [h.path for h in report.hints if 4 in h.rules]
    assert sorted(with_key) == ["config/settings.json", "docs/notes.md", "tests/test_pin_key.py"]
    # Key matches come first, then the rest; each group sorted by path.
    assert _paths(report) == [
        "config/settings.json", "docs/notes.md", "tests/test_pin_key.py",
        "hermes_cli/reader.py", "scripts/tool.py", "tests/test_pin_plain.py",
    ]
    # Every rule that matched is named, not only the first.
    assert _rules(report) == {
        "config/settings.json": (4,),
        "docs/notes.md": (3, 4),
        "tests/test_pin_key.py": (1, 4),
        "hermes_cli/reader.py": (2,),
        "scripts/tool.py": (2, 3),
        "tests/test_pin_plain.py": (1,),
    }
    assert report.counts == {1: 2, 2: 2, 3: 2, 4: 3}

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
    assert [line.split(":")[-1].strip() for line in counts] == ["2", "2", "2", "3"]
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
