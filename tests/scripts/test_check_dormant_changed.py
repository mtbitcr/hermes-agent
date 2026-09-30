"""scripts/ci/check_dormant_changed.py fails a PR that adds or changes a test file listed in
tests/fork_dormant_skips.txt.

CI never collects a listed file, so a change to one would otherwise look tested when it is not. The
contract is pinned on throwaway repos with placeholder file names; the real list is never read.
"""
import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_dormant_changed.py"
LIST = "tests/fork_dormant_skips.txt"


def _load():
    spec = importlib.util.spec_from_file_location("check_dormant_changed", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin"})


def _commit(repo, files, remove=()):
    """Write ``files`` ({path: text}), delete ``remove``, commit everything (the first call inits the repo)."""
    if not (repo / ".git").exists():
        _git(repo, "init", "-q", "-b", "main")
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    for rel in remove:
        (repo / rel).unlink()
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "commit")


def _run(repo, monkeypatch, capsys):
    """Run the check for the PR ``HEAD~1..HEAD``; return (exit code, stdout)."""
    monkeypatch.chdir(repo)
    rc = _load().main(["--base", "HEAD~1", "--head", "HEAD"])
    return rc, capsys.readouterr().out


def _named(out):
    """The paths the failure lines name; every line must say why the change is untested."""
    lines = out.splitlines()
    assert all("CI never runs it while it stays listed" in line for line in lines), out
    return [line.split(":", 1)[0] for line in lines]


def test_changed_listed_test_file_fails_and_is_named(tmp_path, monkeypatch, capsys):
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n", LIST: "tests/test_placeholder_a.py\n"})
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): assert 1\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert _named(out) == ["tests/test_placeholder_a.py"]


def test_changed_unlisted_test_file_passes(tmp_path, monkeypatch, capsys):
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n",
                       "tests/test_placeholder_b.py": "def test_b(): ...\n", LIST: "tests/test_placeholder_b.py\n"})
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): assert 1\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 0
    assert "test_placeholder_a" not in out


def test_pr_that_unlists_and_changes_a_file_passes(tmp_path, monkeypatch, capsys):
    """The PR is judged by its own list: taking the file off the list is the fix."""
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n", LIST: "tests/test_placeholder_a.py\n"})
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): assert 1\n", LIST: "# burned down\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 0
    assert "test_placeholder_a" not in out


def test_comments_and_blank_lines_in_the_list_are_ignored(tmp_path, monkeypatch, capsys):
    listing = ("# header comment\n#tests/test_placeholder_a.py\n   # tests/test_placeholder_a.py\n\n"
               "tests/test_placeholder_b.py\n   \n# trailing comment\n")
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n",
                       "tests/test_placeholder_b.py": "def test_b(): ...\n", LIST: listing})
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): assert 1\n",
                       "tests/test_placeholder_b.py": "def test_b(): assert 1\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert _named(out) == ["tests/test_placeholder_b.py"]


def test_pr_that_changes_no_test_file_passes(tmp_path, monkeypatch, capsys):
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n", "agent/placeholder.py": "X = 1\n",
                       LIST: "tests/test_placeholder_a.py\n"})
    _commit(tmp_path, {"agent/placeholder.py": "X = 2\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 0
    assert "test_placeholder_a" not in out


def test_rename_or_copy_onto_a_listed_path_fails_under_the_new_path(tmp_path, monkeypatch, capsys):
    old, src = "def test_old():\n    assert True\n", "def test_src():\n    assert True\n"
    _commit(tmp_path, {"tests/test_placeholder_old.py": old, "tests/test_placeholder_src.py": src,
                       LIST: "tests/test_placeholder_old.py\ntests/test_placeholder_renamed.py\n"
                             "tests/test_placeholder_copied.py\n"})
    # With rename/copy detection on, a plain diff reports these as R/C rather than as additions.
    _git(tmp_path, "config", "diff.renames", "copies")
    _commit(tmp_path, {"tests/test_placeholder_renamed.py": old, "tests/test_placeholder_copied.py": src,
                       "tests/test_placeholder_src.py": src + "# edited\n"},
            remove=["tests/test_placeholder_old.py"])
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert _named(out) == ["tests/test_placeholder_copied.py", "tests/test_placeholder_renamed.py"]


def test_entries_match_whole_test_file_paths_like_the_ci_runner(tmp_path, monkeypatch, capsys):
    """Entries are stripped; in scripts/run_tests_parallel.py a directory entry drops nothing and a
    non-``test_*.py`` file is never discovered, so neither is flagged."""
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): ...\n", "tests/helper_placeholder.py": "X = 1\n",
                       "tests/sub/test_placeholder_c.py": "def test_c(): ...\n",
                       LIST: "  tests/test_placeholder_a.py  \ntests/helper_placeholder.py\ntests/sub\n"})
    _commit(tmp_path, {"tests/test_placeholder_a.py": "def test_a(): assert 1\n", "tests/helper_placeholder.py": "X = 2\n",
                       "tests/sub/test_placeholder_c.py": "def test_c(): assert 1\n"})
    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert _named(out) == ["tests/test_placeholder_a.py"]
