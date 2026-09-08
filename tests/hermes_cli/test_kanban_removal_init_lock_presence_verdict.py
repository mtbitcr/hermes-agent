"""The init-lock receipt check must gate its PASS on its own reading.

Reviewer finding: the ``init-lock-retained`` check's verdict was derived
from the apply journal ALONE.  The file-presence read happened inline in
the observed f-string, never gated the verdict and never reached the
evidence — so a receipt could report ``PASS`` for a "retained" file
while its own observed text said ``file present at receipt time: False``.

The removal here is driven through the shipped CLI entry point (the same
route :mod:`tests.hermes_cli.test_kanban_removal_init_lock_and_transcript`
uses), never by calling the receipt builder directly, and the presence
read is fault-injected at the filesystem boundary for that ONE path.
"""

from __future__ import annotations

import contextlib
import errno
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_kanban_removal_init_lock_and_transcript import (
    _board,
    _remove_via_cli,
)


def _init_lock_path(slug: str) -> Path:
    """The materialised init-lock sibling of the board's register lock."""
    lock = kb.register_lock_path(slug)
    return lock.with_name(lock.name + ".init.lock")


@contextlib.contextmanager
def presence_reads_absent(monkeypatch, target: Path):
    """Make every filesystem presence read of *target* report absent.

    A fault injected at the OS boundary, not at the module under test:
    the removal still runs its real apply, still journals the retention,
    and still acquires the real cross-process init lock (which opens the
    file rather than testing for it). Only "is it there?" lies — which is
    exactly the reading whose answer the receipt must be honest about.
    """
    target = Path(target)
    real_exists = Path.exists
    real_is_file = Path.is_file
    real_stat = Path.stat

    def exists(self, *args, **kwargs):
        if self == target:
            return False
        return real_exists(self, *args, **kwargs)

    def is_file(self, *args, **kwargs):
        if self == target:
            return False
        return real_is_file(self, *args, **kwargs)

    def stat(self, *args, **kwargs):
        if self == target:
            raise FileNotFoundError(
                errno.ENOENT, "injected: no such file", str(target)
            )
        return real_stat(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "exists", exists)
        m.setattr(Path, "is_file", is_file)
        m.setattr(Path, "stat", stat)
        yield


def _init_lock_check(slug: str) -> dict:
    record = kb.get_removal_phase_record(slug)
    assert record is not None, f"{slug} has no removal phase record"
    receipt = kb.get_permanent_removal_receipt(slug, record.removal_id)
    assert receipt is not None, "no permanent-removal receipt was recorded"
    checks = [
        check for check in receipt["verification"]["checks"]
        if check["check"] == "init-lock-retained"
    ]
    assert len(checks) == 1, "no init-lock-retained check on the receipt"
    return checks[0]


def test_init_lock_check_is_not_pass_when_its_own_read_says_absent(
    fence_home, tmp_path, monkeypatch
):
    """Journalled retention plus an absent-reading file is NOT a PASS.

    The retention journal entry is written for real (the injection does
    not touch it), so this is precisely the combination the old verdict
    called PASS: a receipt claiming a retained file it could not see.
    """
    slug = "init-lock-absent-read"
    _board(slug, tmp_path / "repo")

    with presence_reads_absent(monkeypatch, _init_lock_path(slug)):
        code, output = _remove_via_cli(slug)

    check = _init_lock_check(slug)
    assert check["verdict"] != kb.RECEIPT_VERDICT_PASS, (
        f"the receipt claimed PASS while its own reading said the "
        f"init-lock was absent: {check['observed']} (cli exit {code})"
    )
    assert check["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED, output
    # The reading that produced the verdict is in the evidence, not only
    # in prose, and the prose no longer calls an absent file expected.
    assert check["evidence"]["file_present"] is False
    assert check["evidence"]["file_is_regular_file"] is False
    assert "expected" not in check["observed"].lower(), check["observed"]


def test_init_lock_check_passes_when_the_file_is_really_there(
    fence_home, tmp_path
):
    """The happy path is unchanged: present, journalled, regular file, PASS."""
    slug = "init-lock-present-read"
    _board(slug, tmp_path / "repo")

    code, output = _remove_via_cli(slug)
    assert code == 0, output

    path = _init_lock_path(slug)
    assert path.is_file(), (
        "the register-lock discipline did not re-materialise the init-lock, "
        "so this test is no longer exercising the retained-by-design case"
    )
    check = _init_lock_check(slug)
    assert check["verdict"] == kb.RECEIPT_VERDICT_PASS, check["observed"]
    assert check["evidence"]["retention_journalled"] is True
    assert check["evidence"]["file_present"] is True
    assert check["evidence"]["file_is_regular_file"] is True
    assert check["evidence"]["identity"] == str(path)


def test_a_directory_at_the_init_lock_identity_is_not_a_pass(
    fence_home, tmp_path, monkeypatch
):
    """Present but not a regular file: the type check must fail the PASS.

    The expected materialised identity is a FILE. Something else at that
    exact path is present, so a presence-only test would pass it.
    """
    slug = "init-lock-wrong-type"
    _board(slug, tmp_path / "repo")
    target = _init_lock_path(slug)
    real_is_file = Path.is_file
    real_stat = Path.stat

    def is_file(self, *args, **kwargs):
        if self == target:
            return False
        return real_is_file(self, *args, **kwargs)

    def stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == target:
            # A directory's mode at the same path: present, wrong type.
            return type(result)(
                (0o040755, *tuple(result)[1:])
            )
        return result

    with monkeypatch.context() as m:
        m.setattr(Path, "is_file", is_file)
        m.setattr(Path, "stat", stat)
        code, output = _remove_via_cli(slug)

    check = _init_lock_check(slug)
    assert check["verdict"] != kb.RECEIPT_VERDICT_PASS, (
        f"a non-file at the init-lock identity produced PASS: "
        f"{check['observed']} (cli exit {code}, {output[:200]})"
    )
    assert check["evidence"]["file_is_regular_file"] is False
