"""Records of each attempt to send a failed scheduled report again.

Runs are delivered through the real ``cron.scheduler._deliver_result``, whose standalone sender is
faked to answer each chat as scripted. Each run's failed chats are then claimed, finished and fenced
through ``cron.delivery_record`` and read back through the re-send view of the run history. Nothing
here sends a report again: this is only the record-store half of the re-send.
"""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler
from gateway.config import Platform, PlatformConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED = "Telegram send failed: Forbidden: bot was blocked by the user"
REPLIES = {
    "delivered": {"success": True, "platform": "telegram", "message_id": "s1"},
    "failed": {"error": BLOCKED},
    # A refusal that still carries a message id may have been sent, so it reads unknown.
    "unknown": {"error": BLOCKED, "message_id": "m7"},
}
REPORT = "Numbers are up."
DAY = 86400.0
NOW = 1_790_000_000.0


def _store():
    return importlib.import_module("cron.delivery_record")


def _chat(position):
    return f"-100100000{position:04d}"


def _run(execution_id, *outcomes, text=REPORT):
    """Deliver one run to one Telegram chat per outcome, in order, and return its record."""
    store = _store()
    replies = {_chat(position): outcome for position, outcome in enumerate(outcomes)}

    async def standalone(platform, pconfig, chat_id, message, thread_id=None, media_files=None, **_kwargs):
        return dict(REPLIES[replies[chat_id]], chat_id=chat_id)

    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda _platform: None
    job = {"id": "job-report", "name": "Daily report", "deliver": ",".join(f"telegram:{chat}" for chat in replies)}
    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", new=standalone),
        store.recording(execution_id, job["id"]),
    ):
        scheduler._deliver_result(job, text)
    record = store.load(execution_id)
    assert [target["state"] for target in record["targets"]] == list(outcomes)
    return record


def _row(execution_id, status="completed"):
    return {"id": execution_id, "status": status}


def _view(execution_id, status="completed"):
    return _store().history_deliveries([_row(execution_id, status)])[0]


def _claim(execution_id, request_id, status="completed"):
    return _store().claim_resend(_row(execution_id, status), request_id)


def _finish(attempt, *results, error=None):
    """Finish ``attempt`` with one (position, state, reason) result per chat."""
    return _store().finish_resend(
        attempt["attempt_id"],
        [{"position": position, "state": state, "reason": reason} for position, state, reason in results],
        error=error,
    )


def _at(monkeypatch, moment):
    monkeypatch.setattr(_store(), "_clock", lambda: moment)


def _iso(moment):
    return datetime.fromtimestamp(moment, timezone.utc).isoformat()


def _listed(attempt, *, finished_at=None, state="in_progress"):
    """How the history row lists ``attempt``."""
    return {
        "attempt_id": attempt["attempt_id"],
        "request_id": attempt["request_id"],
        "requested_at": attempt["requested_at"],
        "finished_at": finished_at,
        "state": state,
    }


def _attempt_rows():
    conn = sqlite3.connect(_store()._path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM attempts ORDER BY rowid")]
    finally:
        conn.close()


# --- claim -----------------------------------------------------------------------------------------

def test_claim_marks_only_the_failed_chats_of_an_eligible_run_in_progress(monkeypatch):
    _at(monkeypatch, NOW)
    _run("exec-1", "delivered", "failed", "unknown")
    before = _view("exec-1")
    assert before["resend"] == {"eligible": True, "reason": None, "attempts": []}

    _at(monkeypatch, NOW + 60)
    claim = _claim("exec-1", "req-1")

    attempt = claim["attempt"]
    assert (claim["claimed"], claim["reason"]) == (True, None)
    assert attempt["attempt_id"]
    assert (attempt["request_id"], attempt["requested_at"], attempt["finished_at"], attempt["state"]) == (
        "req-1", "2026-09-21T14:14:20+00:00", None, "in_progress",
    )
    assert attempt["chats"] == [{"position": 1, "state": "in_progress", "reason": None}]
    after = _view("exec-1")
    assert (after["state"], after["targets"]) == (before["state"], before["targets"])
    assert after["resend"] == {"eligible": False, "reason": "in_progress", "attempts": [_listed(attempt)]}


@pytest.mark.parametrize(("outcomes", "status", "moment", "reason"), [
    (("failed",), "claimed", NOW, "in_progress"),
    (("failed",), "running", NOW, "in_progress"),
    (("unknown",), "completed", NOW, "outcome_unknown"),
    (("delivered", "unknown"), "completed", NOW, "outcome_unknown"),
    (("delivered",), "completed", NOW, "already_delivered"),
    (("failed",), "completed", NOW + 7 * DAY, "output_expired"),
])
def test_claim_refuses_every_run_the_view_refuses_and_writes_nothing(monkeypatch, outcomes, status, moment, reason):
    _at(monkeypatch, NOW)
    _run("exec-1", *outcomes)
    _at(monkeypatch, moment)
    before = _view("exec-1", status)
    assert before["resend"] == {"eligible": False, "reason": reason, "attempts": []}

    claim = _claim("exec-1", "req-1", status)

    assert claim == {"claimed": False, "reason": reason, "attempt": None}
    assert _view("exec-1", status) == before
    assert _attempt_rows() == []


def test_claim_refuses_a_run_without_a_record_or_a_kept_text(monkeypatch):
    store = _store()
    assert _claim("exec-1", "req-1") == {"claimed": False, "reason": "not_recorded", "attempt": None}
    assert not store._path().exists()

    monkeypatch.setattr(store, "MAX_SAVED_TEXT_BYTES", 64)
    _at(monkeypatch, NOW)
    _run("exec-big", "failed", text="X" * 65)

    assert _view("exec-big")["resend"] == {"eligible": False, "reason": "output_expired", "attempts": []}
    assert _claim("exec-big", "req-1") == {"claimed": False, "reason": "output_expired", "attempt": None}
    assert _claim("exec-missing", "req-1") == {"claimed": False, "reason": "not_recorded", "attempt": None}
    assert _attempt_rows() == []


def test_the_seven_days_count_from_the_failed_delivery_not_from_an_attempt(monkeypatch):
    _at(monkeypatch, NOW)
    _run("exec-1", "failed")
    _at(monkeypatch, NOW + 6 * DAY)
    attempt = _claim("exec-1", "req-1")["attempt"]
    assert _finish(attempt, (0, "failed", "platform_refused")) is True

    _at(monkeypatch, NOW + 7 * DAY - 1)
    assert _view("exec-1")["resend"]["eligible"] is True
    _at(monkeypatch, NOW + 7 * DAY)
    assert _view("exec-1")["resend"] == {
        "eligible": False, "reason": "output_expired",
        "attempts": [_listed(attempt, finished_at=_iso(NOW + 6 * DAY), state="failed")],
    }
    assert _claim("exec-1", "req-2") == {"claimed": False, "reason": "output_expired", "attempt": None}


# --- repeat ----------------------------------------------------------------------------------------

def test_a_repeated_request_id_returns_its_first_attempt_without_claiming_again(monkeypatch):
    _at(monkeypatch, NOW)
    _run("exec-1", "failed")
    first = _claim("exec-1", "req-1")["attempt"]

    assert _claim("exec-1", "req-1") == {"claimed": False, "reason": None, "attempt": first}

    assert _finish(first, (0, "failed", "platform_refused")) is True
    again = _claim("exec-1", "req-1")
    assert (again["claimed"], again["reason"]) == (False, None)
    assert (again["attempt"]["attempt_id"], again["attempt"]["state"]) == (first["attempt_id"], "failed")
    assert _view("exec-1")["resend"]["eligible"] is True
    with pytest.raises(ValueError):
        _claim("exec-1", "")  # without a request id a repeat could not be told apart
    assert len(_attempt_rows()) == 1

    second = _claim("exec-1", "req-2")["attempt"]
    assert second["attempt_id"] != first["attempt_id"]
    assert _finish(second, (0, "delivered", None)) is True
    repeated = _claim("exec-1", "req-2")
    assert (repeated["claimed"], repeated["reason"]) == (False, None)
    assert (repeated["attempt"]["attempt_id"], repeated["attempt"]["state"]) == (second["attempt_id"], "delivered")
    assert _claim("exec-1", "req-3") == {"claimed": False, "reason": "already_delivered", "attempt": None}
    assert [row["request_id"] for row in _attempt_rows()] == ["req-1", "req-2"]


# --- concurrency -----------------------------------------------------------------------------------

def test_two_simultaneous_claims_give_exactly_one_attempt(monkeypatch):
    _at(monkeypatch, NOW)
    _run("exec-1", "failed")
    start = threading.Barrier(2)
    claims = {}

    def claim(request_id):
        start.wait(10)
        claims[request_id] = _claim("exec-1", request_id)

    threads = [threading.Thread(target=claim, args=(request_id,)) for request_id in ("req-a", "req-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    [won] = [claim["attempt"] for claim in claims.values() if claim["claimed"]]
    [lost] = [claim for claim in claims.values() if not claim["claimed"]]
    assert lost == {"claimed": False, "reason": "in_progress", "attempt": None}
    assert _view("exec-1")["resend"]["attempts"] == [_listed(won)]


CHILD = """
import json
import os
import sys
import time

from cron import delivery_record

execution_id, request_id, result, go, delay, hold = sys.argv[1:]
rules = delivery_record._resend_refusal


def slow_rules(*args, **kwargs):
    # Widen the gap between the eligibility read and the write: a claim that is not one
    # compare-and-set write lets every racer through it.
    reason = rules(*args, **kwargs)
    time.sleep(float(delay))
    return reason


delivery_record._resend_refusal = slow_rules
with open(result + ".ready", "w", encoding="utf-8"):
    pass
deadline = time.monotonic() + 60
while not os.path.exists(go):
    if time.monotonic() > deadline:
        sys.exit("never told to claim")
    time.sleep(0.005)
claim = delivery_record.claim_resend({"id": execution_id, "status": "completed"}, request_id)
with open(result + ".tmp", "w", encoding="utf-8") as handle:
    json.dump({
        "claimed": claim["claimed"],
        "reason": claim["reason"],
        "attempt_id": (claim["attempt"] or {}).get("attempt_id"),
    }, handle)
os.replace(result + ".tmp", result)
if hold == "hold":
    time.sleep(3600)  # the send that a restart cuts off
"""


class _Claimer:
    """A separate process that claims the re-send of ``execution_id`` once ``go`` exists."""

    def __init__(self, tmp_path, execution_id, request_id, go, *, delay=0.0, hold=False):
        script = tmp_path / "claimer.py"
        script.write_text(CHILD, encoding="utf-8")
        self.result = tmp_path / f"{request_id}.json"
        self.ready = tmp_path / f"{request_id}.json.ready"
        self.errors = tmp_path / f"{request_id}.stderr"
        with open(self.errors, "wb") as stderr:
            self.process = subprocess.Popen(
                [sys.executable, str(script), execution_id, request_id, str(self.result), str(go),
                 str(delay), "hold" if hold else "exit"],
                cwd=tmp_path,
                env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )

    def wait_for(self, path):
        deadline = time.monotonic() + 60
        while not path.exists():
            if self.process.poll() is not None and not path.exists():
                pytest.fail("claimer ended early: " + self.errors.read_text(encoding="utf-8", errors="replace")[-3000:])
            if time.monotonic() > deadline:
                pytest.fail(f"claimer never wrote {path.name}")
            time.sleep(0.02)

    def claim(self):
        self.wait_for(self.result)
        return json.loads(self.result.read_text(encoding="utf-8"))

    def stop(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=30)


def test_simultaneous_claims_from_separate_processes_give_exactly_one_attempt(tmp_path):
    _run("exec-1", "failed")
    go = tmp_path / "go"
    claimers = [_Claimer(tmp_path, "exec-1", f"req-{index}", go, delay=0.3) for index in range(4)]
    try:
        for claimer in claimers:
            claimer.wait_for(claimer.ready)
        go.touch()
        claims = [claimer.claim() for claimer in claimers]
    finally:
        for claimer in claimers:
            claimer.stop()

    [won] = [claim for claim in claims if claim["claimed"]]
    assert [claim["reason"] for claim in claims if not claim["claimed"]] == ["in_progress"] * 3
    assert [row["attempt_id"] for row in _attempt_rows()] == [won["attempt_id"]]
    assert [attempt["state"] for attempt in _view("exec-1")["resend"]["attempts"]] == ["in_progress"]


# --- finish ----------------------------------------------------------------------------------------

def test_finish_records_each_chat_result_the_times_and_a_redacted_error(monkeypatch):
    store = _store()
    _at(monkeypatch, NOW)
    _run("exec-1", "delivered", "failed", "failed")
    _at(monkeypatch, NOW + 60)
    attempt = _claim("exec-1", "req-1")["attempt"]
    assert [chat["position"] for chat in attempt["chats"]] == [1, 2]
    token = "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
    error = RuntimeError(f"POST https://api.telegram.org/bot123456789:{token}/sendMessage: {BLOCKED}")

    _at(monkeypatch, NOW + 120)
    recorded = _finish(
        attempt,
        (0, "failed", "platform_refused"),  # not a chat of this attempt: ignored
        (1, "delivered", None),
        (2, "failed", "platform_refused"),
        error=error,
    )

    assert recorded is True
    [stored] = store.load("exec-1")["attempts"]
    assert stored["chats"] == [
        {"position": 1, "state": "delivered", "reason": None},
        {"position": 2, "state": "failed", "reason": "platform_refused"},
    ]
    assert (stored["requested_at"], stored["finished_at"], stored["state"]) == (
        _iso(NOW + 60), _iso(NOW + 120), "failed",
    )
    assert BLOCKED in stored["error"]
    assert token not in stored["error"]
    assert token not in json.dumps(_attempt_rows())
    view = _view("exec-1")
    assert view["resend"] == {
        "eligible": True, "reason": None,
        "attempts": [_listed(attempt, finished_at=_iso(NOW + 120), state="failed")],
    }
    assert "Forbidden" not in json.dumps(view)


def test_a_chat_without_a_clear_result_reads_unknown_and_is_never_claimed_again(monkeypatch):
    store = _store()
    _at(monkeypatch, NOW)
    _run("exec-1", "failed", "failed", "failed")
    first = _claim("exec-1", "req-1")["attempt"]

    # Raw error text is no reason code, "sent" is no outcome, and chat 2 has no result at all.
    assert _finish(first, (0, "failed", BLOCKED), (1, "sent", "timeout")) is True

    [stored] = store.load("exec-1")["attempts"]
    assert stored["chats"] == [
        {"position": 0, "state": "failed", "reason": None},
        {"position": 1, "state": "unknown", "reason": None},
        {"position": 2, "state": "unknown", "reason": "interrupted"},
    ]
    assert stored["state"] == "failed"
    second = _claim("exec-1", "req-2")["attempt"]
    assert second["chats"] == [{"position": 0, "state": "in_progress", "reason": None}]
    assert _finish(second, (0, "delivered", None)) is True
    assert _view("exec-1")["resend"] == {
        "eligible": False, "reason": "outcome_unknown",
        "attempts": [
            _listed(first, finished_at=_iso(NOW), state="failed"),
            _listed(second, finished_at=_iso(NOW), state="delivered"),
        ],
    }
    assert _claim("exec-1", "req-3") == {"claimed": False, "reason": "outcome_unknown", "attempt": None}


def test_once_every_failed_chat_is_delivered_the_run_reads_already_delivered(monkeypatch):
    _at(monkeypatch, NOW)
    _run("exec-1", "delivered", "failed")
    before = _view("exec-1")
    attempt = _claim("exec-1", "req-1")["attempt"]

    assert _finish(attempt, (1, "delivered", None)) is True

    after = _view("exec-1")
    assert (after["state"], after["targets"]) == (before["state"], before["targets"])
    assert after["resend"] == {
        "eligible": False, "reason": "already_delivered",
        "attempts": [_listed(attempt, finished_at=_iso(NOW), state="delivered")],
    }
    assert _claim("exec-1", "req-2") == {"claimed": False, "reason": "already_delivered", "attempt": None}


def test_finish_writes_only_an_attempt_still_in_progress(monkeypatch):
    store = _store()
    _at(monkeypatch, NOW)
    _run("exec-1", "failed")
    attempt = _claim("exec-1", "req-1")["attempt"]
    assert _finish(attempt, (0, "failed", "platform_refused")) is True

    _at(monkeypatch, NOW + 60)
    assert _finish(attempt, (0, "delivered", None)) is False
    assert _finish({"attempt_id": "no-such-attempt"}, (0, "delivered", None)) is False

    [stored] = store.load("exec-1")["attempts"]
    assert (stored["state"], stored["finished_at"]) == ("failed", _iso(NOW))
    assert stored["chats"] == [{"position": 0, "state": "failed", "reason": "platform_refused"}]


# --- start-up fence --------------------------------------------------------------------------------

def test_start_up_fence_turns_every_attempt_in_progress_unknown_and_ineligible(monkeypatch):
    store = _store()
    _at(monkeypatch, NOW)
    _run("exec-cut", "delivered", "failed")
    _run("exec-done", "failed")
    cut = _claim("exec-cut", "req-cut")["attempt"]
    done = _claim("exec-done", "req-done")["attempt"]
    assert _finish(done, (0, "failed", "platform_refused")) is True

    _at(monkeypatch, NOW + 300)
    assert store.fence_resends() == 1

    [fenced] = store.load("exec-cut")["attempts"]
    assert (fenced["state"], fenced["finished_at"]) == ("unknown", _iso(NOW + 300))
    assert fenced["chats"] == [{"position": 1, "state": "unknown", "reason": "interrupted"}]
    assert _view("exec-cut")["resend"] == {
        "eligible": False, "reason": "outcome_unknown",
        "attempts": [_listed(cut, finished_at=_iso(NOW + 300), state="unknown")],
    }
    assert _claim("exec-cut", "req-again") == {"claimed": False, "reason": "outcome_unknown", "attempt": None}
    assert _claim("exec-cut", "req-cut")["attempt"]["state"] == "unknown"
    assert _finish(cut, (1, "delivered", None)) is False
    assert _view("exec-done")["resend"] == {
        "eligible": True, "reason": None,
        "attempts": [_listed(done, finished_at=_iso(NOW), state="failed")],
    }
    assert store.fence_resends() == 0


def test_start_up_fence_without_a_store_creates_nothing():
    store = _store()

    assert store.fence_resends() == 0
    assert not store._path().exists()


def test_a_restart_mid_resend_leaves_it_unknown_and_ineligible(tmp_path):
    store = _store()
    _run("exec-1", "failed")
    go = tmp_path / "go"
    go.touch()
    claimer = _Claimer(tmp_path, "exec-1", "req-1", go, hold=True)
    try:
        claim = claimer.claim()
        assert claimer.process.poll() is None
    finally:
        claimer.stop()
    assert claim["claimed"] is True
    assert _view("exec-1")["resend"]["reason"] == "in_progress"

    assert store.fence_resends() == 1

    resend = _view("exec-1")["resend"]
    assert (resend["eligible"], resend["reason"]) == (False, "outcome_unknown")
    assert [(attempt["attempt_id"], attempt["state"]) for attempt in resend["attempts"]] == [
        (claim["attempt_id"], "unknown"),
    ]
    assert _claim("exec-1", "req-2") == {"claimed": False, "reason": "outcome_unknown", "attempt": None}


# --- history view ----------------------------------------------------------------------------------

def test_history_rows_list_each_attempt_in_order_and_runs_without_attempts_read_as_before(monkeypatch):
    store = _store()
    _at(monkeypatch, NOW)
    _run("exec-1", "delivered", "failed", "unknown")
    _run("exec-plain", "delivered", "failed", "unknown")
    _at(monkeypatch, NOW + 60)
    first = _claim("exec-1", "req-1")["attempt"]
    _at(monkeypatch, NOW + 120)
    assert _finish(first, (1, "failed", "platform_refused"), error=BLOCKED) is True
    _at(monkeypatch, NOW + 180)
    second = _claim("exec-1", "req-2")["attempt"]
    _at(monkeypatch, NOW + 240)
    assert _finish(second, (1, "delivered", None)) is True

    resent, plain, unrecorded = store.history_deliveries(
        [_row("exec-1"), _row("exec-plain"), {"id": "exec-none", "status": "completed", "delivery_outcome": "failed"}],
        now=NOW + 300,
    )

    targets = [
        {"label": "Telegram", "state": "delivered", "reason": None},
        {"label": "Telegram 2", "state": "failed", "reason": "platform_refused"},
        {"label": "Telegram 3", "state": "unknown", "reason": "error_after_handover"},
    ]
    assert plain == {"state": "failed", "targets": targets, "resend": {"eligible": True, "reason": None, "attempts": []}}
    assert unrecorded == {
        "state": "not_recorded", "targets": [],
        "resend": {"eligible": False, "reason": "not_recorded", "attempts": []},
    }
    assert resent == {
        "state": "failed",
        "targets": targets,
        "resend": {
            "eligible": False, "reason": "outcome_unknown",
            "attempts": [
                _listed(first, finished_at=_iso(NOW + 120), state="failed"),
                _listed(second, finished_at=_iso(NOW + 240), state="delivered"),
            ],
        },
    }
    shown = json.dumps(resent)
    for hidden in ("exec-1", REPORT, "Forbidden", *(_chat(position) for position in range(3))):
        assert hidden not in shown


OLD_SCHEMA = """
CREATE TABLE records (
  execution_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  created_at REAL NOT NULL,
  text TEXT,
  attachments TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX idx_records_created ON records(created_at DESC, execution_id DESC);
CREATE TABLE targets (
  execution_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  platform TEXT NOT NULL,
  chat_id TEXT NOT NULL,
  thread_id TEXT,
  state TEXT NOT NULL CHECK(state IN ('pending','delivered','failed','unknown')),
  reason TEXT,
  sent_text TEXT,
  updated_at REAL NOT NULL,
  PRIMARY KEY (execution_id, position)
);
"""


def _held(path):
    """Everything a store written before re-send attempts existed holds."""
    conn = sqlite3.connect(path)
    try:
        return {
            "schema": conn.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE tbl_name IN ('records','targets') ORDER BY name"
            ).fetchall(),
            "records": conn.execute("SELECT * FROM records ORDER BY execution_id").fetchall(),
            "targets": conn.execute("SELECT * FROM targets ORDER BY execution_id, position").fetchall(),
        }
    finally:
        conn.close()


def test_resend_records_never_change_what_the_store_already_holds(monkeypatch):
    store = _store()
    path = store._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO records (execution_id, job_id, created_at, text, attachments) VALUES (?, ?, ?, ?, '[]')",
            ("exec-old", "job-report", NOW, REPORT),
        )
        conn.executemany(
            "INSERT INTO targets (execution_id, position, platform, chat_id, thread_id, state, reason, "
            "sent_text, updated_at) VALUES ('exec-old', ?, 'telegram', ?, NULL, ?, ?, NULL, ?)",
            [(0, _chat(0), "delivered", None, NOW), (1, _chat(1), "failed", "platform_refused", NOW)],
        )
        conn.commit()
    finally:
        conn.close()
    held = _held(path)

    _at(monkeypatch, NOW + 60)
    assert _view("exec-old") == {
        "state": "failed",
        "targets": [
            {"label": "Telegram", "state": "delivered", "reason": None},
            {"label": "Telegram 2", "state": "failed", "reason": "platform_refused"},
        ],
        "resend": {"eligible": True, "reason": None, "attempts": []},
    }
    first = _claim("exec-old", "req-1")["attempt"]
    assert _finish(first, (1, "failed", "platform_refused"), error=BLOCKED) is True
    assert _claim("exec-old", "req-2")["claimed"] is True
    assert store.fence_resends() == 1

    assert _held(path) == held
    assert [row["request_id"] for row in _attempt_rows()] == ["req-1", "req-2"]


def test_attempts_are_pruned_only_with_their_record(monkeypatch):
    store = _store()
    monkeypatch.setattr(store, "MAX_RECORDS", 1)
    _at(monkeypatch, NOW - 8 * DAY)
    _run("exec-old", "failed")
    assert _claim("exec-old", "req-old")["claimed"] is True
    _at(monkeypatch, NOW)
    _run("exec-recent", "failed")
    assert _claim("exec-recent", "req-recent")["claimed"] is True

    _at(monkeypatch, NOW + 1)
    _run("exec-today", "failed")

    assert store.load("exec-old") is None
    assert [row["execution_id"] for row in _attempt_rows()] == ["exec-recent"]
