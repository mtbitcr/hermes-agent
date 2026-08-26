"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.dashboard_auth import audit as audit_module
from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError, audit_log


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Redirect $HERMES_HOME and ~ to a tmp dir for the duration of the test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some code paths fall back to Path.home() — patch that too.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", user_id="u1",
        email="a@b.com", ip="1.2.3.4",
    )

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    second = json.loads(lines[1])
    assert second["event"] == "login_success"
    assert second["provider"] == "nous"
    assert second["user_id"] == "u1"
    assert second["email"] == "a@b.com"
    assert "ts" in second  # ISO-8601 timestamp


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"


# --------------------------------------------------------------------------
# Item 32G-A: strict=True fail-closed audit writes
# --------------------------------------------------------------------------


def test_strict_false_default_never_raises_on_write_failure(profile_home, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)
    # Every existing call site relies on this: a broken logger must not
    # break auth.
    audit_log(AuditEvent.LOGIN_START, provider="nous")


def test_strict_true_raises_audit_write_error_on_failure(profile_home, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audit_module.os, "open", _boom)
    with pytest.raises(AuditWriteError):
        audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="raphael-workspace-token")


def test_strict_true_succeeds_silently_when_write_succeeds(profile_home):
    audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="raphael-workspace-token")
    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists()
    assert json.loads(path.read_text().strip().splitlines()[-1])["event"] == "token_auth_success"


@pytest.mark.skipif(audit_module.os.name == "nt", reason="POSIX modes only")
def test_audit_tightens_owned_log_and_parent_permissions(profile_home):
    logs = profile_home / "logs"
    logs.mkdir(mode=0o755)
    path = logs / "dashboard-auth.log"
    path.write_text("")
    path.chmod(0o644)

    audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="workspace")

    assert stat.S_IMODE(logs.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# --------------------------------------------------------------------------
# Item 32TK: a batch is a record of what happened, not of what was attempted
# --------------------------------------------------------------------------


def _entries(*profiles):
    return [
        (
            AuditEvent.TOKEN_AUTH_SUCCESS,
            {"profile": profile, "decision": "allow", "action": "assign"},
        )
        for profile in profiles
    ]


def _allowed_profiles(records):
    return [
        record["profile"]
        for record in records
        if record.get("decision") == "allow"
    ]


def _log_path(profile_home):
    return profile_home / "logs" / "dashboard-auth.log"


def _records(profile_home):
    """Read through the canonical reader, at an explicit path.

    ``monkeypatch.undo()`` in a crash-injection test also undoes the
    ``HERMES_HOME`` redirect the fixtures installed, so a path-less read after
    it silently answers from a different, empty log — and asserts nothing.
    """
    return audit_module.read_audit_records(_log_path(profile_home))


def _naive_line_records(profile_home):
    """What an ordinary one-JSON-object-per-line consumer of this log sees.

    Exactly the reading this log's documented format invites, and the one the
    canonical reader must agree with about what happened.
    """
    path = _log_path(profile_home)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        nested = record.get("records")
        if isinstance(nested, list):
            records.extend(item for item in nested if isinstance(item, dict))
        else:
            records.append(record)
    return records


def test_a_journalled_batch_is_invisible_until_it_is_committed(profile_home):
    batch = audit_module.begin_audit_batch(_entries("first", "second"))

    # The bytes are durable, but nothing happened as far as any reader is
    # concerned — the operation they describe has not been committed.
    path = _log_path(profile_home)
    assert len(path.read_text().strip().splitlines()) == 2
    assert _allowed_profiles(_records(profile_home)) == []

    audit_module.commit_audit_batch(batch)
    assert _allowed_profiles(_records(profile_home)) == ["first", "second"]


def test_a_prepared_entry_never_reads_as_an_allow_to_a_line_consumer(profile_home):
    """The production format is safe for the reading this log's docs invite.

    A prepared entry is not a record, so the raw line must not carry the auth
    event or the decision it has not earned — and must not contain that text
    even for a consumer that greps rather than parses.
    """
    batch = audit_module.begin_audit_batch(_entries("first", "second"))

    raw = _log_path(profile_home).read_text(encoding="utf-8")
    assert '"decision":"allow"' not in raw
    assert f'"event":"{AuditEvent.TOKEN_AUTH_SUCCESS.value}"' not in raw
    assert _allowed_profiles(_naive_line_records(profile_home)) == []
    assert {record["event"] for record in _naive_line_records(profile_home)} == {
        AuditEvent.BATCH_PREPARED.value,
    }

    # Committing is what writes the records, and both readers then agree.
    audit_module.commit_audit_batch(batch)
    assert _allowed_profiles(_naive_line_records(profile_home)) == [
        "first", "second",
    ]
    assert _naive_line_records(profile_home)[-2:] == _records(profile_home)


def test_a_batch_a_restart_never_committed_stays_invisible(profile_home):
    audit_module.begin_audit_batch(_entries("abandoned"))
    # "Restart": nothing in memory survives; the file is the whole state.
    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []

    # A later, complete operation is unaffected by the abandoned one.
    audit_module.commit_audit_batch(audit_module.begin_audit_batch(_entries("later")))
    assert _allowed_profiles(_records(profile_home)) == ["later"]
    assert _allowed_profiles(_naive_line_records(profile_home)) == ["later"]


def test_a_short_write_leaves_no_effective_allow_record(profile_home):
    real_write = audit_module.os.write

    def short_write(fd, data):
        # Enough bytes for the first complete entry, and no more.
        return real_write(fd, data[: data.index(b"\n") + 1])

    with pytest.MonkeyPatch.context() as patched:
        # A scoped context, not ``monkeypatch.undo()``: undo would also revert
        # the fixtures' HERMES_HOME redirect and silently repoint every later
        # read and write at a different, empty log.
        patched.setattr(audit_module.os, "write", short_write)
        with pytest.raises(AuditWriteError):
            audit_module.begin_audit_batch(_entries("first", "second"))

    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []


def test_a_short_write_of_the_commit_line_leaves_no_allow_record(profile_home):
    """A batch's records are one line, so a partial commit is not a record."""
    batch = audit_module.begin_audit_batch(_entries("first", "second"))
    real_write = audit_module.os.write

    def short_write(fd, data):
        return real_write(fd, data[: len(data) // 2])

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "write", short_write)
        with pytest.raises(AuditWriteError):
            audit_module.commit_audit_batch(batch)

    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []


def test_an_fsync_failure_after_the_bytes_landed_leaves_no_allow_record(
    profile_home,
):
    def _boom(_fd):
        raise OSError("fsync failed")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "fsync", _boom)
        with pytest.raises(AuditWriteError):
            audit_module.begin_audit_batch(_entries("first", "second"))

    # The whole batch really is on disk — and still asserts nothing.
    assert len(_log_path(profile_home).read_text().strip().splitlines()) == 2
    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []


def test_a_failed_commit_is_removed_again_not_left_asserting(profile_home):
    """An ``fsync`` failure is a failure, so the caller rolls back — and so
    must the line that would otherwise outlive the operation claiming it
    succeeded."""
    batch = audit_module.begin_audit_batch(_entries("first"))
    before = _log_path(profile_home).read_text(encoding="utf-8")

    def _boom(_fd):
        raise OSError("fsync failed")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "fsync", _boom)
        with pytest.raises(AuditWriteError):
            audit_module.commit_audit_batch(batch)

    assert _log_path(profile_home).read_text(encoding="utf-8") == before
    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []

    # And the batch is still committable once the disk cooperates.
    audit_module.commit_audit_batch(batch)
    assert _allowed_profiles(_records(profile_home)) == ["first"]


_SIBLING_LINE = (
    '{"ts":"2026-01-01T00:00:00+00:00","event":"login_success",'
    '"provider":"nous","user_id":"sibling"}\n'
)


def _append_uncooperatively(profile_home):
    """Append as a writer that ignores this module's advisory lock.

    Buffered file I/O on purpose: ``os.write`` is patched in these tests, and
    the point of this writer is that it is NOT one of them.
    """
    with open(_log_path(profile_home), "ab") as handle:
        handle.write(_SIBLING_LINE.encode("utf-8"))


def test_a_failed_commit_never_truncates_a_concurrent_writer(profile_home):
    """Rollback removes only this call's own bytes — and removes them even
    once they are no longer the file's tail.

    A writer that ignores the advisory lock appends between the commit write
    and its failing ``fsync``, so truncating back would destroy that writer's
    record. This call's own byte range is blanked in place instead: the
    interleaved record survives untouched, and the reverted batch stops
    asserting anything to either reader.
    """
    batch = audit_module.begin_audit_batch(_entries("first"))
    real_fsync = audit_module.os.fsync
    calls = []

    def _boom(fd):
        calls.append(fd)
        if len(calls) > 1:
            # The rollback's own fsync must really sync.
            return real_fsync(fd)
        _append_uncooperatively(profile_home)
        raise OSError("fsync failed")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "fsync", _boom)
        with pytest.raises(AuditWriteError):
            audit_module.commit_audit_batch(batch)

    events = [record["event"] for record in _records(profile_home)]
    assert "login_success" in events
    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []

    # Still recoverable: the journalled batch commits normally afterwards.
    audit_module.commit_audit_batch(batch)
    assert _allowed_profiles(_records(profile_home)) == ["first"]


def test_a_commit_that_cannot_be_removed_reports_rollback_uncertainty(
    profile_home,
):
    """The one residual case is propagated, never swallowed.

    Neither the truncate (the file grew) nor the in-place blank (the disk
    refuses the write) can complete, so the log may still be asserting a batch
    the caller is about to revert. That is a different fact from "the commit
    was never written", and the caller is told which one it got — while every
    existing ``except AuditWriteError`` still fails closed.
    """
    batch = audit_module.begin_audit_batch(_entries("first"))
    real_write = audit_module.os.write
    writes = []

    def _write(fd, data):
        writes.append(fd)
        if len(writes) > 1:
            # The in-place blank cannot land either.
            raise OSError("no space left on device")
        return real_write(fd, data)

    def _fsync_boom(_fd):
        _append_uncooperatively(profile_home)
        raise OSError("fsync failed")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "write", _write)
        patched.setattr(audit_module.os, "fsync", _fsync_boom)
        with pytest.raises(audit_module.AuditRollbackUncertain) as raised:
            audit_module.commit_audit_batch(batch)

    assert isinstance(raised.value, AuditWriteError)


def test_a_clean_rollback_is_not_reported_as_uncertain(profile_home):
    """Removal that IS proven keeps the plain, narrower failure."""
    batch = audit_module.begin_audit_batch(_entries("first"))
    real_fsync = audit_module.os.fsync
    calls = []

    def _boom(fd):
        calls.append(fd)
        if len(calls) > 1:
            return real_fsync(fd)
        raise OSError("fsync failed")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audit_module.os, "fsync", _boom)
        with pytest.raises(AuditWriteError) as raised:
            audit_module.commit_audit_batch(batch)

    assert not isinstance(raised.value, audit_module.AuditRollbackUncertain)


def test_concurrent_batches_are_committed_independently(profile_home):
    """Interleaved lines from two operations resolve to their own outcomes."""
    losing = audit_module.begin_audit_batch(_entries("rolled-back"))
    winning = audit_module.begin_audit_batch(_entries("committed"))
    audit_module.commit_audit_batch(winning)

    assert _allowed_profiles(_records(profile_home)) == ["committed"]
    assert _allowed_profiles(_naive_line_records(profile_home)) == ["committed"]

    # The loser committing later does not retroactively hide the winner.
    audit_module.commit_audit_batch(losing)
    assert sorted(_allowed_profiles(_records(profile_home))) == [
        "committed", "rolled-back",
    ]


def test_a_truncated_line_never_swallows_the_record_after_it(profile_home, monkeypatch):
    path = _log_path(profile_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"event":"token_auth_suc', encoding="utf-8")

    audit_module.commit_audit_batch(audit_module.begin_audit_batch(_entries("intact")))

    assert _allowed_profiles(_records(profile_home)) == ["intact"]
    assert _allowed_profiles(_naive_line_records(profile_home)) == ["intact"]


def test_a_pre_upgrade_journalled_entry_still_needs_its_marker(profile_home):
    """An upgrade must not turn yesterday's uncommitted batch into a record."""
    path = _log_path(profile_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"ts":"2026-01-01T00:00:00+00:00","event":"token_auth_success",'
        '"profile":"never-committed","decision":"allow","action":"assign",'
        '"audit_batch":"oldbatch1"}\n'
        '{"ts":"2026-01-01T00:00:01+00:00","event":"token_auth_success",'
        '"profile":"old-committed","decision":"allow","action":"assign",'
        '"audit_batch":"oldbatch2"}\n'
        '{"ts":"2026-01-01T00:00:02+00:00","event":"audit_batch_committed",'
        '"audit_batch":"oldbatch2"}\n',
        encoding="utf-8",
    )

    assert _allowed_profiles(_records(profile_home)) == ["old-committed"]

    # And the new format keeps working alongside it.
    audit_module.commit_audit_batch(audit_module.begin_audit_batch(_entries("new")))
    assert _allowed_profiles(_records(profile_home)) == ["old-committed", "new"]


def test_an_ordinary_entry_needs_no_batch_marker(profile_home):
    """Every existing one-shot call site keeps reading back exactly as before."""
    audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", user_id="u1")
    assert [
        record["event"] for record in _records(profile_home)
    ] == ["login_success"]
    assert [
        record["event"] for record in _naive_line_records(profile_home)
    ] == ["login_success"]


# --------------------------------------------------------------------------
# Item 32TK: the write window is serialized across PROCESSES, not just threads
#
# These use real, separately-spawned interpreters. A thread in this process
# would prove nothing: ``_write_lock`` already orders those, and the gap being
# closed is precisely the one an in-process lock cannot see.
# --------------------------------------------------------------------------

_REPO_ROOT = Path(audit_module.__file__).resolve().parents[2]

posix_locking = pytest.mark.skipif(
    audit_module._fcntl is None, reason="POSIX flock only"
)

# Holds an exclusive OS lock on the log, announces it, then lets go.
_HOLDER = """
import fcntl, os, sys, time

path, ready, hold = sys.argv[1], sys.argv[2], float(sys.argv[3])
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
open(ready, "w").close()
time.sleep(hold)
fcntl.flock(fd, fcntl.LOCK_UN)
os.close(fd)
"""

# Waits for a go-ahead, then makes one ORDINARY (non-strict) audit append —
# the cheapest, commonest writer, and the one most likely to slip into another
# writer's window.
_APPENDER = """
import os, sys, time

from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log

go, marker = sys.argv[1], sys.argv[2]
while not os.path.exists(go):
    time.sleep(0.005)
audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", user_id=marker)
"""

# A burst of both writer kinds, as fast as one process can issue them.
_BURST = """
import sys

from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log

tag, count = sys.argv[1], int(sys.argv[2])
for index in range(count):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        strict=index % 2 == 0,
        provider="nous",
        user_id="{}-{}".format(tag, index),
    )
"""


def _spawn(source, args, profile_home):
    """Run ``source`` in a genuinely separate interpreter on the same log."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return subprocess.Popen(
        [sys.executable, "-c", source, *(str(arg) for arg in args)],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(path, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


@posix_locking
@pytest.mark.parametrize("writer", ["non_strict", "strict", "prepare", "commit"])
def test_every_audit_writer_waits_for_another_process_lock(
    profile_home, tmp_path, writer,
):
    """No writer here may append while another process holds the log.

    Not just the strict commit whose rollback needs the window: an ordinary
    ``login_success`` line landing inside a commit's fsync window is exactly
    what would strand a rolled-back batch, so it has to queue too.
    """
    log = _log_path(profile_home)
    log.parent.mkdir(parents=True, exist_ok=True)
    ready = tmp_path / "holder-ready"
    hold = 0.7
    # Prepared BEFORE the holder starts — otherwise this leg would time the
    # journal write rather than the commit.
    prepared = (
        audit_module.begin_audit_batch(_entries("held")) if writer == "commit" else None
    )

    child = _spawn(_HOLDER, [log, ready, hold], profile_home)
    try:
        assert _wait_for(ready), "the other process never took the lock"
        started = time.monotonic()
        if writer == "non_strict":
            audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", user_id="held")
        elif writer == "strict":
            audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, profile="held")
        elif writer == "prepare":
            audit_module.begin_audit_batch(_entries("held"))
        else:
            audit_module.commit_audit_batch(prepared)
        waited = time.monotonic() - started
    finally:
        _, stderr = child.communicate(timeout=30)

    assert child.returncode == 0, stderr
    assert waited >= hold / 2, (
        f"the {writer} writer appended after {waited:.3f}s while another "
        f"process held the log for {hold}s — it did not serialize"
    )


@posix_locking
def test_no_other_process_can_append_inside_a_failed_commit_rollback(
    profile_home, tmp_path,
):
    """The residual gap, closed: a failed commit can always undo itself.

    Rollback may only remove its own bytes while they are still the file's
    tail, so a line appended by anyone else between the commit write and its
    failing fsync used to pin the reverted batch in the log for good. The
    other process here is real and genuinely trying to append throughout that
    window — its record does land, just after the window, which is what makes
    the "nothing interleaved" assertion mean something.
    """
    batch = audit_module.begin_audit_batch(_entries("rolled-back"))
    before = _log_path(profile_home).read_text(encoding="utf-8")
    go = tmp_path / "go"
    during = []
    real_fsync = audit_module.os.fsync
    calls = []

    def _boom(fd):
        calls.append(fd)
        if len(calls) > 1:
            # The rollback's own fsync must really sync.
            return real_fsync(fd)
        # The commit line is on disk and its fsync is about to fail. Release
        # the other process and give it far longer than an append needs.
        go.write_text("go", encoding="utf-8")
        time.sleep(0.5)
        during.append(_log_path(profile_home).read_text(encoding="utf-8"))
        raise OSError("fsync failed")

    child = _spawn(_APPENDER, [go, "sibling"], profile_home)
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(audit_module.os, "fsync", _boom)
            with pytest.raises(AuditWriteError):
                audit_module.commit_audit_batch(batch)
    finally:
        # Never strand the child if the commit failed somewhere unexpected.
        go.write_text("go", encoding="utf-8")
        _, stderr = child.communicate(timeout=30)

    assert child.returncode == 0, stderr
    assert during, "the fsync failure never fired"
    assert "sibling" not in during[0], (
        "another process appended inside the commit's write/fsync/rollback "
        "window — the rollback could not have removed its own line"
    )

    after = _log_path(profile_home).read_text(encoding="utf-8")
    assert after.startswith(before)
    # Exactly the other process's one line was added, and the commit line the
    # fsync failure disowned is gone.
    extra = [line for line in after[len(before):].splitlines() if line.strip()]
    assert len(extra) == 1
    assert json.loads(extra[0])["user_id"] == "sibling"
    assert AuditEvent.BATCH_COMMITTED.value not in after

    # The reverted batch asserts nothing, to either reader, and the other
    # process's record survived intact.
    assert _allowed_profiles(_records(profile_home)) == []
    assert _allowed_profiles(_naive_line_records(profile_home)) == []
    assert [record.get("user_id") for record in _records(profile_home)] == ["sibling"]


@posix_locking
def test_concurrent_processes_lose_no_ordinary_audit_lines(profile_home):
    """Serializing the window must not cost a line, mangle one, or deadlock."""
    _log_path(profile_home).parent.mkdir(parents=True, exist_ok=True)
    tags = ["p0", "p1", "p2", "p3"]
    per_process = 12

    children = [_spawn(_BURST, [tag, per_process], profile_home) for tag in tags]
    for child in children:
        _, stderr = child.communicate(timeout=60)
        assert child.returncode == 0, stderr

    expected = {f"{tag}-{index}" for tag in tags for index in range(per_process)}
    # Every line is intact JSON on its own line: no write landed inside
    # another's, and none was dropped.
    lines = [
        line
        for line in _log_path(profile_home).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == len(expected)
    for line in lines:
        json.loads(line)
    assert {record["user_id"] for record in _records(profile_home)} == expected
