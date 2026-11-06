"""Tests for the launch-time stale-bytecode sweep (checkout fingerprint guard).

Bug class: the checkout's ``.py`` files change (``hermes update``, manual
``git pull``, ZIP update) while ``__pycache__`` retains bytecode compiled
from the previous revision; the next process to import trusts the stale
``.pyc`` and dies with ``cannot import name ...`` (#6207, #60242).

The launch-time guard compares the current checkout fingerprint against the
last-validated stamp and sweeps ``__pycache__`` once when they diverge —
covering paths no update-time clear can reach (manual pulls, pre-hardening
updaters).
"""

from pathlib import Path

from hermes_cli import main as hermes_main
from hermes_cli import main_web_build


def _make_repo(tmp_path: Path, sha: str = "a" * 40) -> Path:
    """Minimal git checkout layout that _read_git_revision_fingerprint groks."""
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    return repo


def _make_pycache(repo: Path, subdir: str = "hermes_cli") -> Path:
    cache = repo / subdir / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "main.cpython-311.pyc").write_bytes(b"stale")
    return cache


def test_sweep_clears_pycache_when_checkout_changed(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, sha="b" * 40)
    cache = _make_pycache(repo)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    # Stamp records a different (older) fingerprint.
    (repo / main_web_build._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "a" * 40, encoding="utf-8"
    )

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not cache.exists()
    # Stamp updated to the current fingerprint.
    recorded = (repo / main_web_build._BYTECODE_FINGERPRINT_FILE).read_text(encoding="utf-8")
    assert recorded.strip().endswith("b" * 40)


def test_sweep_leaves_the_release_environment_under_hermes_runtime_intact(monkeypatch, tmp_path):
    # The real folder of release.venv_dir lives under .hermes-runtime, and a release copies it
    # while it moves the checkout: a sweep started then must not delete its bytecode mid-copy.
    repo = _make_repo(tmp_path, sha="b" * 40)
    cache = _make_pycache(repo)
    environment_cache = _make_pycache(
        repo, ".hermes-runtime/sync2e/hermes-env/lib/python3.11/site-packages/cryptography/hazmat"
    )
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    (repo / main_web_build._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "a" * 40, encoding="utf-8"
    )

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not cache.exists()
    assert (environment_cache / "main.cpython-311.pyc").read_bytes() == b"stale"







# ---------------------------------------------------------------------------
# Plugin-update sibling site: __pycache__ under ~/.hermes/plugins/<name>
# ---------------------------------------------------------------------------

def test_clear_plugin_bytecode_removes_nested_caches(tmp_path):
    from hermes_cli import plugins_cmd

    plugin = tmp_path / "myplugin"
    top = plugin / "__pycache__"
    nested = plugin / "sub" / "__pycache__"
    top.mkdir(parents=True)
    nested.mkdir(parents=True)
    (top / "a.pyc").write_bytes(b"stale")
    (nested / "b.pyc").write_bytes(b"stale")

    removed = plugins_cmd._clear_plugin_bytecode(plugin)

    assert removed == 2
    assert not top.exists()
    assert not nested.exists()


