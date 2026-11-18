"""The skill delivery at the end of each release: the full sync result, its record and the owner line.

The keyword test runs the real sync in a child process against a temporary role home and
built-in tree, set through ``HERMES_HOME`` and ``HERMES_BUNDLED_SKILLS``. The record tests
reach the module as ``ledger.<name>``, so a missing name fails that test on its own.
"""

import argparse
import contextlib
import errno
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.skill_utils import ESSENTIAL_SKILLS
from hermes_cli import (
    build_info, kanban_db, profiles, release_cmd, release_host_actions, release_runner, release_unit,
)
from hermes_cli import owner_project_summary_events as summary_events
from hermes_cli import release_ledger as ledger
from hermes_cli.profiles import seed_profile_skills
from hermes_cli.subcommands.release import build_release_parser
from hermes_constants import mark_named_profile_deleted
# The run's and the recovery's own test hosts and helpers, as those files build them.
from tests.hermes_cli import test_release_recover as recover_tests
from tests.hermes_cli import test_release_run as run_tests

ESSENTIAL = sorted(ESSENTIAL_SKILLS)[0]
LEGACY_KEYS = ("copied", "updated", "skipped", "user_modified", "cleaned", "suppressed", "relocated",
               "total_bundled", "optional_provenance_backfilled", "shadowed_by_external", "skipped_opt_out")
# The child's code string of the call without the keyword, byte for byte as before the keyword.
BASE_CODE = ("import json; from tools.skills_sync import sync_skills; r = sync_skills(quiet=True); "
             "r = {k: r[k] for k in ('copied', 'updated', 'skipped', 'user_modified', 'cleaned', "
             "'suppressed', 'relocated', 'total_bundled', 'optional_provenance_backfilled', "
             "'shadowed_by_external', 'skipped_opt_out') if k in r}; print(json.dumps(r))")


@pytest.fixture
def role(tmp_path, monkeypatch):
    """One role's temporary home plus the release's built-in and optional trees."""
    home, bundled, optional = tmp_path / "home", tmp_path / "bundled", tmp_path / "optional"
    for path in (home / "skills", bundled, optional):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional))
    return SimpleNamespace(home=home, skills=home / "skills", bundled=bundled, optional=optional)


def _write_skill(root: Path, rel: str, body: str) -> Path:
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: test skill\n---\n{body}\n", encoding="utf-8")
    return folder


# ── The keyword: full_result ────────────────────────────────────────────────────────────

def test_full_result_returns_the_whole_sync_result_and_the_call_without_it_is_unchanged(
        role, monkeypatch, capsys):
    _write_skill(role.bundled, "tools/alpha", "v1")
    _write_skill(role.bundled, "tools/beta", "v1")

    plain = seed_profile_skills(role.home, quiet=True)
    assert isinstance(plain, dict) and plain
    assert list(plain) == [key for key in LEGACY_KEYS if key in plain]
    assert not {"kept", "refused", "not_seeded_by_marker"} & set(plain)

    full = seed_profile_skills(role.home, quiet=True, full_result=True)
    assert isinstance(full, dict)
    assert isinstance(full["kept"], list) and isinstance(full["refused"], list)
    assert isinstance(full["not_seeded_by_marker"], int)
    assert set(plain) <= set(full)
    assert capsys.readouterr().out == ""

    # Without the keyword the child process, the return value and every message stay as before.
    calls = []

    def exits_non_zero(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 3, stdout="", stderr="boom\n")

    monkeypatch.setattr(profiles.subprocess, "run", exits_non_zero)
    assert seed_profile_skills(role.home) is None
    assert capsys.readouterr().out == "⚠ Skill seeding returned exit code 3\n  boom\n"
    (args, kwargs), = calls
    assert args == ([sys.executable, "-c", BASE_CODE],)
    assert kwargs == {"env": {**os.environ, "HERMES_HOME": str(role.home)},
                      "cwd": str(Path(profiles.__file__).parent.parent.resolve()),
                      "capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace",
                      "timeout": 60}
    assert seed_profile_skills(role.home, quiet=True, full_result=True) is None
    assert capsys.readouterr().out == ""

    def times_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(profiles.subprocess, "run", times_out)
    assert seed_profile_skills(role.home) is None
    assert capsys.readouterr().out == "⚠ Skill seeding timed out (60s)\n"
    assert seed_profile_skills(role.home, quiet=True, full_result=True) is None
    assert capsys.readouterr().out == ""


def test_full_result_refuses_a_deleted_role_home_in_the_child(served):
    """The child refuses a tombstoned role home before the sync can create anything in it, its
    tree gone or still there, so a stale role list cannot bring a deleted role back. A live
    role home still gets its result."""
    _write_skill(served.bundled, "tools/alpha", "v1")
    live, gone, left = served.home("live"), served.home("gone"), served.home("left")
    assert isinstance(seed_profile_skills(live, quiet=True, full_result=True), dict)
    assert (live / "skills" / "tools" / "alpha" / "SKILL.md").is_file()
    for home in (gone, left):
        mark_named_profile_deleted(home)
    shutil.rmtree(gone)
    before = sorted(left.rglob("*"))

    assert seed_profile_skills(gone, quiet=True, full_result=True) is None
    assert not gone.exists()
    assert seed_profile_skills(left, quiet=True, full_result=True) is None
    assert sorted(left.rglob("*")) == before


# ── The record: record_skill_delivery and last_skill_delivery ───────────────────────────

PAYLOAD = {
    "roles": [{"role": "default", "status": "refreshed", "updated": 1,
               "kept": [{"name": "alpha", "reason": "changed"}], "refused": [], "not_seeded_by_marker": 0}],
    "refreshed": 1, "failed": 0,
}
FAILED_PAYLOAD = {
    "roles": [{"role": "default", "status": "failed", "updated": 0, "kept": [], "refused": [],
               "not_seeded_by_marker": 0, "reason": "no result"}],
    "refreshed": 0, "failed": 1,
}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root at ``<tmp>/.hermes``; nothing can resolve to the real home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def conn(root):
    connection = ledger.connect()
    yield connection
    connection.close()


def _commit(label: str) -> str:
    """A 40-hex id shaped like a git commit."""
    return hashlib.sha1(label.encode()).hexdigest()


def _accepted(conn, label: str, board: str = "") -> int:
    """Merge one change, a card of ``board`` when one is named, and accept the batch it joined;
    returns the batch id."""
    batch = ledger.record_merge(
        conn, merge_commit=_commit(label), pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
        reviewed_base=_commit(f"{label}-base"), reviewed_head=_commit(f"{label}-head"),
        reviewed_tree=_commit(f"{label}-tree"), tier=1, card_id=f"t_{_commit(label)[:8]}", board=board,
        on_main=True,
    )["batch"]
    return ledger.decide_release(
        conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
        expected_version=batch["version"], decision_ref="decisions-page:release",
    )["batch_id"]


def _begin(conn, batch_id: int, label: str) -> None:
    ledger.begin_release(conn, batch_id, prev=ledger.last_released(conn) or _commit("live"), new=_commit(label))


def _released(conn, label: str) -> int:
    batch_id = _accepted(conn, label)
    _begin(conn, batch_id, label)
    ledger.finish_release(conn, batch_id, outcome="released")
    return batch_id


def _counted(store: Path) -> tuple[int, int]:
    """(all events, skills_delivered events), read straight from the store file."""
    independent = sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)
    try:
        return independent.execute(
            "SELECT COUNT(*), COALESCE(SUM(kind = 'skills_delivered'), 0) FROM release_events").fetchone()
    finally:
        independent.close()


def _sized(size: int) -> dict:
    """A payload whose compact JSON is exactly ``size`` bytes."""
    payload = {"roles": [{"role": ""}], "refreshed": 0, "failed": 0}
    payload["roles"][0]["role"] = "r" * (size - len(json.dumps(payload, separators=(",", ":"))))
    return payload


def test_record_skill_delivery_writes_one_row_and_last_skill_delivery_reads_the_newest(conn):
    assert ledger.last_skill_delivery(conn) is None
    first = _released(conn, "first")

    assert ledger.record_skill_delivery(conn, first, PAYLOAD) is None
    rows = conn.execute(
        "SELECT batch_id, payload FROM release_events WHERE kind = 'skills_delivered'").fetchall()
    assert [(row["batch_id"], json.loads(row["payload"])) for row in rows] == [(first, PAYLOAD)]
    assert ledger.last_skill_delivery(conn) == PAYLOAD

    second = _released(conn, "second")
    ledger.record_skill_delivery(conn, second, FAILED_PAYLOAD)
    assert _counted(ledger.ledger_path())[1] == 2
    assert ledger.last_skill_delivery(conn) == FAILED_PAYLOAD


def test_record_skill_delivery_refuses_an_unreleased_batch_an_oversized_payload_and_a_worker(
        conn, monkeypatch):
    store = ledger.ledger_path()
    batch_id = _accepted(conn, "accepted")
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, batch_id, PAYLOAD)
    _begin(conn, batch_id, "accepted")
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, batch_id, PAYLOAD)
    ledger.finish_release(conn, batch_id, outcome="failed")
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, batch_id, PAYLOAD)
    ledger.finish_release(conn, batch_id, outcome="restored")
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, batch_id, PAYLOAD)
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, 999, PAYLOAD)
    assert _counted(store)[1] == 0

    released = _released(conn, "released")
    before = _counted(store)
    with pytest.raises(ValueError):
        ledger.record_skill_delivery(conn, released, _sized(65537))
    assert _counted(store) == before
    ledger.record_skill_delivery(conn, released, _sized(65536))
    assert ledger.last_skill_delivery(conn) == _sized(65536)

    before = _counted(store)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0badc0de")
    with pytest.raises(PermissionError):
        ledger.record_skill_delivery(conn, released, PAYLOAD)
    assert _counted(store) == before


def test_under_the_worker_environment_the_record_is_the_roots_and_the_write_is_refused(root, monkeypatch):
    operator = ledger.connect()
    try:
        released = _released(operator, "released")
        ledger.record_skill_delivery(operator, released, PAYLOAD)
    finally:
        operator.close()
    store = root / "kanban" / "release_ledger.db"
    before = _counted(store)
    assert before[1] == 1

    # Exactly what the dispatcher injects into a worker it spawns.
    profile_home = root / "profiles" / "worker"
    board_dir = root / "kanban" / "boards" / "worker-board"
    profile_home.mkdir(parents=True)
    board_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0badc0de")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_dir / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board_dir / "workspaces"))

    assert ledger.ledger_path() == store
    worker = ledger.connect()
    try:
        with pytest.raises(PermissionError):
            ledger.record_skill_delivery(worker, released, FAILED_PAYLOAD)
        assert ledger.last_skill_delivery(worker) == PAYLOAD
    finally:
        worker.close()

    # Independent count, read straight from the one root store.
    assert _counted(store) == before
    assert not [path for path in profile_home.rglob("*") if "release" in path.name]
    assert not [path for path in board_dir.rglob("*") if "release" in path.name]


# ── The step: _deliver_skills ───────────────────────────────────────────────────────────

class _Actions:
    """The one part of the host actions the step uses: the served homes, the root first."""

    def __init__(self, homes=(), error=None):
        self.homes, self.error = list(homes), error

    def _served_homes(self):
        if self.error is not None:
            raise self.error
        return list(self.homes)


@pytest.fixture
def told(monkeypatch):
    """Every text the step sends the owner, in order."""
    sent = []
    monkeypatch.setattr(release_cmd, "_tell_owner", lambda text: sent.append(text))
    return sent


@pytest.fixture
def served(root, tmp_path, monkeypatch):
    """The release's built-in trees, and the root's role homes: ``home(name)`` makes one."""
    bundled, optional = tmp_path / "bundled", tmp_path / "optional"
    for path in (bundled, optional, root / "skills"):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional))

    def home(name: str) -> Path:
        path = root / "profiles" / name
        (path / "skills").mkdir(parents=True)
        return path

    return SimpleNamespace(root=root, bundled=bundled, optional=optional, home=home)


def _refreshed_line(batch_id: int, role: str, updated: int = 0, kept: str = "none", refused: str = "none",
                    unseeded: int = 0) -> str:
    return (f"Skills after release {batch_id}, role {role}: refreshed. {updated} updated; "
            f"kept: {kept}; refused: {refused}; {unseeded} not seeded because of the marker.")


def _failed_line(batch_id: int, role: str, updated: int = 0, kept: str = "none", refused: str = "none",
                 unseeded: int = 0, unavailable: bool = False) -> str:
    items = f"{updated} updated; kept: {kept}; refused: {refused}; {unseeded} not seeded because of the marker."
    if unavailable:
        return (f"Skills after release {batch_id}, role {role}: not refreshed (the role no longer exists). "
                f"{items} The release itself is done.")
    return (f"Skills after release {batch_id}, role {role}: not refreshed (the refresh did not finish). "
            f"{items} It keeps its earlier skills until the next release; the release itself is done.")


def _entry_line(batch_id: int, entry: dict) -> str:
    """The owner line rendered from one ledger entry alone: its status, its reason and its four items."""
    kept, refused = (", ".join(f"{item['name']} ({item['reason']})" for item in entry[key]) or "none"
                     for key in ("kept", "refused"))
    items = (entry["updated"], kept, refused, entry["not_seeded_by_marker"])
    if entry["status"] == "refreshed":
        return _refreshed_line(batch_id, entry["role"], *items)
    return _failed_line(batch_id, entry["role"], *items, unavailable=entry["reason"] == "unavailable")


def _only_the_error_type(out: str, name: str, *hidden: str) -> None:
    """The journal holds one line, naming the error type and nothing of its message or path."""
    assert len(out.splitlines()) == 1 and name in out
    assert "/" not in out and "\\" not in out
    for text in hidden:
        assert text not in out


def test_the_role_list_that_cannot_be_read_records_nothing_and_tells_the_owner_one_line(
        conn, told, capsys):
    released = _released(conn, "released")
    store = ledger.ledger_path()
    before = _counted(store)

    error = OSError("cannot list /srv/secret/profiles")
    assert release_cmd._deliver_skills(released, _Actions(error=error)) is None

    _only_the_error_type(capsys.readouterr().out, "OSError", "cannot list", "secret")
    assert _counted(store) == before
    assert ledger.last_skill_delivery(conn) is None
    assert told == [f"Skills after release {released}: the role list could not be read, "
                    "so no role was refreshed. The release itself is done."]


def test_each_role_gets_one_entry_and_one_line_and_a_role_without_a_result_is_failed(
        conn, served, told, capsys, monkeypatch, tmp_path):
    """The root role holds the marker and gets all four items from the real sync; the sync of
    ``broken`` exits non-zero and the one of ``slow`` passes its time limit."""
    released = _released(conn, "released")
    broken, slow = served.home("broken"), served.home("slow")
    for rel in ("tools/alpha", "tools/beta", "tools/gamma"):
        _write_skill(served.bundled, rel, "v1")
    assert sorted(seed_profile_skills(served.root, quiet=True)["copied"]) == ["alpha", "beta", "gamma"]
    (served.root / profiles.NO_BUNDLED_SKILLS_MARKER).write_text("opted out\n", encoding="utf-8")
    copies = served.root / "skills" / "tools"
    (copies / "beta" / "SKILL.md").write_text("the owner's own words\n", encoding="utf-8")
    shutil.copytree(copies / "gamma", copies / "gamma.bak")
    (tmp_path / "owner-category").mkdir()
    (served.root / "skills" / "core").symlink_to(tmp_path / "owner-category", target_is_directory=True)
    # The release: three changed built-ins, two new ones the marker holds back, and the essential.
    for rel in ("tools/alpha", "tools/beta", "tools/gamma"):
        _write_skill(served.bundled, rel, "v2")
    for rel in ("research/arxiv", "media/gif-search", f"core/{ESSENTIAL}"):
        _write_skill(served.bundled, rel, "new in this release")

    real_run = subprocess.run

    def child(*args, **kwargs):
        if kwargs["env"]["HERMES_HOME"] == str(broken):
            return subprocess.CompletedProcess(args[0], 1, stdout="", stderr=f"Traceback: {broken}\n")
        if kwargs["env"]["HERMES_HOME"] == str(slow):
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(profiles.subprocess, "run", child)
    assert release_cmd._deliver_skills(released, _Actions([served.root, broken, slow])) is None

    assert capsys.readouterr().out == ""
    row = ledger.last_skill_delivery(conn)
    kept = row["roles"][0]["kept"]
    assert sorted(kept, key=lambda item: item["name"]) == [
        {"name": "beta", "reason": "changed"}, {"name": "gamma", "reason": "backup present"}]
    no_result = {"status": "failed", "updated": 0, "kept": [], "refused": [], "not_seeded_by_marker": 0,
                 "reason": "no result"}
    assert row == {
        "roles": [{"role": "default", "status": "refreshed", "updated": 1, "kept": kept,
                   "refused": [{"name": ESSENTIAL, "reason": "link"}], "not_seeded_by_marker": 2},
                  {"role": "broken", **no_result}, {"role": "slow", **no_result}],
        "refreshed": 1, "failed": 2,
    }
    named = ", ".join(f"{item['name']} ({item['reason']})" for item in kept)
    assert told == ["\n".join([_refreshed_line(released, "default", 1, named, f"{ESSENTIAL} (link)", 2),
                               _failed_line(released, "broken"), _failed_line(released, "slow")])]
    for text in (json.dumps(row), told[0]):
        assert "official" not in text and "/" not in text and str(tmp_path) not in text


def test_a_failed_roles_line_holds_the_four_items_of_its_entry(conn, served, told, capsys, monkeypatch):
    """Owner fact 4 for failed roles: the sync of ``broken`` exits non-zero, the one of ``slow``
    passes its time limit and the call for ``raising`` raises. Each role's owner line is the one
    its ledger entry renders from its status and its four items, with no path."""
    released = _released(conn, "released")
    broken, slow, raising = served.home("broken"), served.home("slow"), served.home("raising")
    started = _children(monkeypatch, {str(served.root): FOUR_ITEMS, str(broken): 1,
                                      str(slow): subprocess.TimeoutExpired(["python"], 60)})
    seeded = release_cmd.seed_profile_skills

    def seed(home, **options):
        if home == raising:
            raise RuntimeError(f"the sync broke in {home}")
        return seeded(home, **options)

    monkeypatch.setattr(release_cmd, "seed_profile_skills", seed)
    assert release_cmd._deliver_skills(released, _Actions([served.root, broken, slow, raising])) is None

    _only_the_error_type(capsys.readouterr().out, "RuntimeError", "the sync broke")
    assert started == [str(served.root), str(broken), str(slow)]
    row = ledger.last_skill_delivery(conn)
    assert row == {
        "roles": [_four_items_entry("default"), {"role": "broken", **FAILED, "reason": "no result"},
                  {"role": "slow", **FAILED, "reason": "no result"},
                  {"role": "raising", **FAILED, "reason": "RuntimeError"}],
        "refreshed": 1, "failed": 3,
    }
    assert told == ["\n".join(_entry_line(released, entry) for entry in row["roles"])]
    assert "/" not in told[0] and "official" not in told[0]


def test_a_role_deleted_while_an_earlier_role_syncs_is_not_brought_back(conn, served, told, capsys, monkeypatch):
    """The role list is read once, and ``victim`` is deleted as ``hermes profile delete`` does it,
    its tombstone and then its tree, while the root's real sync runs. No sync starts in its home,
    the home stays gone, and its entry and its line say that the role no longer exists."""
    released = _released(conn, "released")
    victim = served.home("victim")
    _write_skill(served.bundled, "tools/alpha", "v1")
    real_run, started = subprocess.run, []

    def child(*args, **kwargs):
        started.append(kwargs["env"]["HERMES_HOME"])
        if kwargs["env"]["HERMES_HOME"] == str(served.root):
            mark_named_profile_deleted(victim)
            shutil.rmtree(victim)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(profiles.subprocess, "run", child)
    assert release_cmd._deliver_skills(released, _Actions([served.root, victim])) is None

    assert capsys.readouterr().out == ""
    assert not victim.exists()
    assert started == [str(served.root)]
    assert ledger.last_skill_delivery(conn) == {
        "roles": [{"role": "default", "status": "refreshed", "updated": 0, "kept": [], "refused": [],
                   "not_seeded_by_marker": 0},
                  {"role": "victim", "status": "failed", "updated": 0, "kept": [], "refused": [],
                   "not_seeded_by_marker": 0, "reason": "unavailable"}],
        "refreshed": 1, "failed": 1,
    }
    assert told == ["\n".join([_refreshed_line(released, "default"),
                               _failed_line(released, "victim", unavailable=True)])]


def test_a_record_that_cannot_be_written_prints_only_the_error_type_and_still_tells_the_owner(
        conn, served, told, capsys, monkeypatch):
    _write_skill(served.bundled, "tools/alpha", "v1")
    actions = _Actions([served.root])
    store = ledger.ledger_path()
    batch_id = _accepted(conn, "accepted")

    def delivered(error_type: str, *hidden: str) -> None:
        before = _counted(store)
        assert release_cmd._deliver_skills(batch_id, actions) is None
        _only_the_error_type(capsys.readouterr().out, error_type, *hidden)
        assert _counted(store) == before
        assert told == [_refreshed_line(batch_id, "default") + "\nThe skill record could not be saved."]
        told.clear()

    delivered("ValueError", "not released")  # the batch is not released
    _begin(conn, batch_id, "accepted")
    ledger.finish_release(conn, batch_id, outcome="released")
    with monkeypatch.context() as worker:
        worker.setenv("HERMES_KANBAN_TASK", "t_0badc0de")
        delivered("PermissionError", "owner-only")  # a board worker's environment

    def unopened():
        raise sqlite3.OperationalError(f"unable to open database file {store}")

    monkeypatch.setattr(ledger, "connect", unopened)
    delivered("OperationalError", "unable to open")
    assert ledger.last_skill_delivery(conn) is None


# ── The step in a release run and in a recovery ─────────────────────────────────────────

# A marked role's whole sync result as its child prints it with full_result: the four items.
FOUR_ITEMS = {
    "copied": [], "updated": ["alpha"], "user_modified": ["beta"], "total_bundled": 6,
    "kept": [{"name": "beta", "reason": "changed"}, {"name": "gamma", "reason": "backup present"}],
    "refused": [{"name": ESSENTIAL, "reason": "link"}], "not_seeded_by_marker": 2,
}
# A role whose sync gave no result, before its reason.
FAILED = {"status": "failed", "updated": 0, "kept": [], "refused": [], "not_seeded_by_marker": 0}
# How a run ends without releasing: its host, the batch's outcome and the run's exit status.
NOT_RELEASED = {
    "refused": (lambda: run_tests.FakeHost(origin=run_tests.PREV), "refused", 1),
    "restored": (lambda: run_tests.FakeHost(unhealthy={run_tests.NEW}), "restored",
                 release_unit.RESTORED_EXIT_STATUS),
    "failed": (lambda: run_tests.FakeHost(unhealthy={run_tests.NEW, run_tests.PREV}), "failed", 1),
    "stopped before the cutover": (  # SIGINT while the run waits for the drain
        lambda: run_tests.FakeHost(open_reads=3, at={("G10", 2): lambda: run_tests._stop_signal(signal.SIGINT)}),
        "refused", 1),
}


def _four_items_entry(role: str) -> dict:
    return {"role": role, "status": "refreshed", "updated": 1, "kept": FOUR_ITEMS["kept"],
            "refused": FOUR_ITEMS["refused"], "not_seeded_by_marker": 2}


def _four_items_line(batch_id: int, role: str) -> str:
    return _refreshed_line(batch_id, role, 1, "beta (changed), gamma (backup present)", f"{ESSENTIAL} (link)", 2)


@pytest.fixture
def run_root(root, tmp_path):
    """The root home with a run's release settings, as tests/hermes_cli/test_release_run.py
    writes them for its FakeHost and its Boundary."""
    settings = {**run_tests.SETTINGS, "venv_dir": str(run_tests._platform_environment(tmp_path, root))}
    (root / "config.yaml").write_text(yaml.safe_dump({"release": settings}))
    return root


@pytest.fixture
def recover_root(root, monkeypatch):
    """The root home with a recovery's release settings, as tests/hermes_cli/test_release_recover.py
    writes them; no wait between two reads of a readback sleeps."""
    (root / "config.yaml").write_text(yaml.safe_dump({"release": recover_tests.SETTINGS}))
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)
    return root


@pytest.fixture
def steps(monkeypatch, told):
    """The step's calls, each noted as it starts with the record's outcomes, the pause and the
    owner's lines at that moment, then made; the start of the journeys is noted as "journeys"."""
    noted, real = [], release_cmd._deliver_skills

    def step(batch_id, actions):
        noted.append({"batch_id": batch_id, "actions": actions, "told": list(told),
                      "outcomes": [batch["outcome"] for batch in run_tests._batches()],
                      "pause": run_tests._sentinel()})
        real(batch_id, actions)

    monkeypatch.setattr(release_cmd, "_deliver_skills", step)
    monkeypatch.setattr(release_cmd, "_start_journeys", lambda: noted.append("journeys"))
    return noted


def _children(monkeypatch, answers: dict) -> list[str]:
    """Each role's sync child, answered at the process boundary by its home: a dict is printed by
    a child that exits 0, a number is a child's non-zero exit, an exception is raised by the call
    (a TimeoutExpired is the time limit). Any other process goes to the ``subprocess.run`` in place
    before. Returns the homes whose child started, in order."""
    started, before = [], subprocess.run

    def run(argv, **options):
        if argv[1:2] != ["-c"] or "from tools.skills_sync import sync_skills" not in argv[2]:
            return before(argv, **options)
        home = options["env"]["HERMES_HOME"]
        started.append(home)
        answer = answers[home]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, int):
            return subprocess.CompletedProcess(argv, answer, "", f"Traceback: {home}\n")
        return subprocess.CompletedProcess(argv, 0, json.dumps(answer) + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    return started


def _still_released(code: int, host, batch_id: int) -> None:
    """The run as it is without the step: exit 0, the batch released, its pause lifted, the
    checkout left at NEW, and no outcome or resume after them."""
    assert code == 0
    assert [(batch["batch_id"], batch["outcome"]) for batch in run_tests._batches()] == [(batch_id, "released")]
    assert run_tests._sentinel() is None
    assert (host.head, [result.outcome for result in host.results]) == (run_tests.NEW, ["released"])
    assert host.journal[-2:] == ["outcome released", "resume"]
    assert (host.journal.count("outcome released"), host.journal.count("resume")) == (1, 1)


def test_a_released_run_delivers_once_after_its_outcome_and_resume_and_before_the_journeys(
        run_root, steps, told, monkeypatch, capsys):
    """Each served home gets one entry: the root's sync brings the four items, the one of
    ``broken`` exits non-zero and the one of ``slow`` passes its time limit. The release stays as
    it was: released, exit 0, its pause lifted, nothing rolled back."""
    batch_id = run_tests._accepted(1)["batch_id"]
    broken, slow = run_root / "profiles" / "broken", run_root / "profiles" / "slow"
    for home in (broken, slow):
        home.mkdir(parents=True)
    host = run_tests.FakeHost()
    host._served_homes = lambda: [run_root, broken, slow]
    started = _children(monkeypatch, {str(run_root): FOUR_ITEMS, str(broken): 1,
                                      str(slow): subprocess.TimeoutExpired(["python"], 60)})

    code = run_tests._run(monkeypatch, host)

    released = f"Release {batch_id} was released."
    assert [note if note == "journeys" else "step" for note in steps] == ["step", "journeys"]
    call = steps[0]
    assert call.pop("actions") is host
    assert call == {"batch_id": batch_id, "told": [released], "outcomes": ["released"], "pause": None}
    _still_released(code, host, batch_id)
    assert started == [str(run_root), str(broken), str(slow)]
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) == {
            "roles": [_four_items_entry("default"), {"role": "broken", **FAILED, "reason": "no result"},
                      {"role": "slow", **FAILED, "reason": "no result"}],
            "refreshed": 1, "failed": 2,
        }
    assert told == [released, "\n".join([_four_items_line(batch_id, "default"),
                                         _failed_line(batch_id, "broken"), _failed_line(batch_id, "slow")])]
    assert capsys.readouterr().out.endswith(f"Batch {batch_id} was released.\n")


@pytest.mark.parametrize("case", list(NOT_RELEASED))
def test_a_run_that_does_not_release_never_delivers(run_root, steps, monkeypatch, case):
    host, outcome, code = NOT_RELEASED[case]
    batch_id = run_tests._accepted(1)["batch_id"]
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)  # no readback wait sleeps

    assert run_tests._run(monkeypatch, host()) == code

    assert [(batch["batch_id"], batch["outcome"]) for batch in run_tests._batches()][0] == (batch_id, outcome)
    assert steps == []
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) is None


def test_a_recovery_that_ends_released_delivers_once_after_its_outcome_and_resume(
        recover_root, steps, told, monkeypatch):
    batch_id = recover_tests._stopped_midway()
    recover_tests._snapshots(recover_root, f"config/{recover_tests.NEW}")  # published before the cutover
    host = recover_tests.FakeHost()  # at NEW, which reads back
    host._served_homes = lambda: [recover_root]
    started = _children(monkeypatch, {str(recover_root): FOUR_ITEMS})

    assert recover_tests._command(monkeypatch, host, "recover", str(batch_id)) == 0

    released = f"Release {batch_id} was released."
    assert [note if note == "journeys" else "step" for note in steps] == ["step"]
    call = steps[0]
    assert call.pop("actions") is host
    assert call == {"batch_id": batch_id, "told": [released], "outcomes": ["released"], "pause": None}
    assert (recover_tests._batch(batch_id)["outcome"], recover_tests._sentinel(), host.calls) == (
        "released", None, [])
    assert started == [str(recover_root)]
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) == {
            "roles": [_four_items_entry("default")], "refreshed": 1, "failed": 0}
    assert told == [released, _four_items_line(batch_id, "default")]


def test_a_recovery_that_ends_released_tells_a_failed_role_its_four_items(recover_root, told, monkeypatch):
    """The sync of ``coder`` exits non-zero: its line, like the root's, is the one its ledger
    entry renders, with its four items."""
    batch_id = recover_tests._stopped_midway()
    recover_tests._snapshots(recover_root, f"config/{recover_tests.NEW}")  # published before the cutover
    coder = recover_root / "profiles" / "coder"
    coder.mkdir(parents=True)
    host = recover_tests.FakeHost()  # at NEW, which reads back
    host._served_homes = lambda: [recover_root, coder]
    started = _children(monkeypatch, {str(recover_root): FOUR_ITEMS, str(coder): 1})

    assert recover_tests._command(monkeypatch, host, "recover", str(batch_id)) == 0

    assert started == [str(recover_root), str(coder)]
    with contextlib.closing(ledger.connect()) as conn:
        row = ledger.last_skill_delivery(conn)
    assert row == {"roles": [_four_items_entry("default"), {"role": "coder", **FAILED, "reason": "no result"}],
                   "refreshed": 1, "failed": 1}
    assert told == [f"Release {batch_id} was released.",
                    "\n".join(_entry_line(batch_id, entry) for entry in row["roles"])]


def test_a_recovery_that_ends_restored_never_delivers(recover_root, steps, monkeypatch):
    batch_id = recover_tests._stopped_midway()
    recover_tests._snapshots(recover_root, f"config/{recover_tests.NEW}")  # published before the cutover
    host = recover_tests._midway(recover_tests.NEW)  # NEW never reads back; PREV does after the restore

    code = recover_tests._command(monkeypatch, host, "recover", str(batch_id))

    assert (code, recover_tests._batch(batch_id)["outcome"]) == (release_unit.RESTORED_EXIT_STATUS, "restored")
    assert steps == []
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) is None


def test_a_run_whose_every_role_fails_still_ends_released(run_root, told, monkeypatch, capsys):
    """The root's sync exits non-zero, the one of ``coder`` passes its time limit and the one of
    ``writer`` is interrupted: every role is failed, and the run ends as it does without the step.
    Only the interrupt, an error inside the step, reaches the journal, by its type alone."""
    batch_id = run_tests._accepted(1)["batch_id"]
    coder, writer = run_root / "profiles" / "coder", run_root / "profiles" / "writer"
    for home in (coder, writer):
        home.mkdir(parents=True)
    host = run_tests.FakeHost()
    host._served_homes = lambda: [run_root, coder, writer]
    _children(monkeypatch, {str(run_root): 1, str(coder): subprocess.TimeoutExpired(["python"], 60),
                            str(writer): KeyboardInterrupt()})

    code = run_tests._run(monkeypatch, host)

    _still_released(code, host, batch_id)
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) == {
            "roles": [{"role": "default", **FAILED, "reason": "no result"},
                      {"role": "coder", **FAILED, "reason": "no result"},
                      {"role": "writer", **FAILED, "reason": "KeyboardInterrupt"}],
            "refreshed": 0, "failed": 3,
        }
    assert told == [f"Release {batch_id} was released.",
                    "\n".join(_failed_line(batch_id, role) for role in ("default", "coder", "writer"))]
    assert capsys.readouterr().out.endswith(
        f"Batch {batch_id} was released.\nThe skills of role writer were not refreshed (KeyboardInterrupt).\n")


def test_a_run_whose_role_list_cannot_be_read_still_ends_released(run_root, told, monkeypatch, capsys):
    batch_id = run_tests._accepted(1)["batch_id"]
    host = run_tests.FakeHost()

    def unreadable():
        raise PermissionError(errno.EACCES, "Permission denied", "/srv/secret/.hermes/profiles")

    host._served_homes = unreadable
    started = _children(monkeypatch, {})

    code = run_tests._run(monkeypatch, host)

    _still_released(code, host, batch_id)
    out = capsys.readouterr().out
    assert out.endswith(f"Batch {batch_id} was released.\n"
                        "The skills were not delivered: the role list could not be read (PermissionError).\n")
    assert "secret" not in out and "Permission denied" not in out
    assert started == []
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) is None
    assert told == [f"Release {batch_id} was released.",
                    f"Skills after release {batch_id}: the role list could not be read, so no role was"
                    " refreshed. The release itself is done."]


def test_a_run_whose_skill_record_cannot_be_saved_still_ends_released(run_root, told, monkeypatch, capsys):
    batch_id = run_tests._accepted(1)["batch_id"]
    host = run_tests.FakeHost()
    host._served_homes = lambda: [run_root]
    _children(monkeypatch, {str(run_root): FOUR_ITEMS})

    def unsaved(conn, batch_id, payload):
        raise sqlite3.OperationalError("disk I/O error on /srv/secret/.hermes/release_ledger.db")

    monkeypatch.setattr(ledger, "record_skill_delivery", unsaved)

    code = run_tests._run(monkeypatch, host)

    _still_released(code, host, batch_id)
    out = capsys.readouterr().out
    assert out.endswith(f"Batch {batch_id} was released.\nThe skill record could not be saved (OperationalError).\n")
    assert "secret" not in out and "disk I/O" not in out
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) is None
    assert told == [f"Release {batch_id} was released.",
                    _four_items_line(batch_id, "default") + "\nThe skill record could not be saved."]


def test_the_step_imports_nothing_after_the_units_stop(run_root, told, monkeypatch):
    """K6 for the step, as tests/hermes_cli/test_release_run.py checks it for the journeys start:
    it runs after the cutover, with the checkout moved to NEW, and it lists the roles, reads each
    role's result, writes its row and tells the owner with the modules loaded before. Each role's
    sync is its own process, answered here at that boundary."""
    batch_id = run_tests._accepted(1)["batch_id"]
    coder = run_root / "profiles" / "coder"
    coder.mkdir(parents=True)
    args = run_tests._parsed_run()
    boundary = run_tests.Boundary(run_root)
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # the health address answers
    direct = SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer))
    monkeypatch.setattr(subprocess, "run", boundary.run)
    started = _children(monkeypatch, {str(run_root): FOUR_ITEMS, str(coder): FOUR_ITEMS})
    monkeypatch.setattr(build_info, "get_code_identity", lambda **_: {"sha": boundary.head})
    monkeypatch.setattr(release_host_actions, "_DIRECT", direct)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    run_tests.Clock().install(monkeypatch)

    assert args.func(args) == 0
    assert started == [str(run_root), str(coder)]
    assert boundary.head == run_tests.NEW and boundary.modules_at_stop is not None
    assert set(sys.modules) - boundary.modules_at_stop == set()
    with contextlib.closing(ledger.connect()) as conn:
        assert ledger.last_skill_delivery(conn) == {
            "roles": [_four_items_entry("default"), _four_items_entry("coder")], "refreshed": 2, "failed": 0}
    assert told == [f"Release {batch_id} was released.",
                    "\n".join([_four_items_line(batch_id, "default"), _four_items_line(batch_id, "coder")])]


# ── Existing readers ────────────────────────────────────────────────────────────────────

def test_existing_readers_ignore_a_skills_delivered_row(conn, capsys):
    """``hermes release status`` and the owner summary's release events read as before. The row
    is the newest event and belongs to the older batch, the one holding a card of the board: the
    last outcome stays the newer batch's, nothing more is counted, and the row only moves the
    summary's release cursor past it."""
    slug = "website"
    with kanban_db.connect_closing(summary_events._board_path(slug), board=slug, create=True):
        pass
    older = _accepted(conn, "older", board=slug)
    _begin(conn, older, "older")
    ledger.finish_release(conn, older, outcome="released")
    newer = _released(conn, "newer")
    project = SimpleNamespace(board_slug=slug)
    parser = argparse.ArgumentParser()
    build_release_parser(parser.add_subparsers())
    args = parser.parse_args(["release", "status"])

    assert args.func(args) == 0
    status = capsys.readouterr().out
    events = summary_events.read_events(project, summary_events.Cursors())
    assert status.splitlines()[-1] == f"Last outcome: batch {newer} was released."
    assert (events.material, events.urgent) == (0, 1)

    ledger.record_skill_delivery(conn, older, PAYLOAD)

    assert args.func(args) == 0
    assert capsys.readouterr().out == status
    after = summary_events.read_events(project, summary_events.Cursors())
    assert (after.material, after.urgent) == (0, 1)
    assert after.cursors.release > events.cursors.release
    assert summary_events.read_events(project, events.cursors) == summary_events.Batch(
        events.cursors._replace(release=after.cursors.release))
