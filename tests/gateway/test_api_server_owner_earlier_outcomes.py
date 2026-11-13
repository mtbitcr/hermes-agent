"""Every earlier proposal of a conversation reports what became of it.

Only the newest exchange of an owner conversation carried an outcome, so an
earlier proposal that was approved and applied read exactly like one the
owner never approved: "A newer proposal replaced this one". The platform now
records, in the transaction that consumes or releases a proposal, that it was
applied or released, and the history answer reports each earlier proposal
turn as applied, not applied, replaced or unknown.

The conversation is made with production code throughout, as in
test_api_server_owner_applied_release.py: 1 a proposal approved and applied,
2 a question, 3 a proposal never approved, 4 a proposal approved and refused
for real by the native plan path, 5 a proposal approved and applied. Every
read is made after a reload: a fresh adapter on the same stores.
"""

import asyncio
import contextlib
import copy
import functools
import hashlib
import json
import sqlite3
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_cli import owner_workspace as ow, projects_db
from tests.gateway.test_api_server import _owner_existing_proposal
from tests.gateway.test_api_server_owner_applied_release import (
    _API_KEY,
    _CLAIM_ID,
    _CONVERSATION,
    _KEY,
    _OPERATION,
    _PLANNER,
    _RESPONSE_ID,
    _conversation_row,
    _crashed_run,
    _history,
    _owner_run_patches,
    _proposal_run,
    _receipts_db,
    _recovery_job,
    _restarted_twice,
)
from tests.gateway.test_api_server_runs import (  # noqa: F401  (an autouse fixture)
    _create_runs_app,
    _make_adapter,
    _resolved_owner_task_routes,
)

_QUESTION = {"schema_version": 1, "kind": "question", "message": "Which milestone comes next?"}
_FIFTH = "resp_earlier_outcome_turn_5"
# The catch-up's marker: the name of its row in ``owner_outcome_tracking`` and
# the ``run_id`` of every mark it writes. Stored, so it never changes.
_CATCH_UP = "owner-outcome-catch-up"
_NEW_TABLES = ("owner_proposal_outcomes", "owner_outcome_tracking")


def _key(response_id: str) -> str:
    return "conversation-" + hashlib.sha256(response_id.encode("utf-8")).hexdigest()


def _proposal(turn: int) -> dict:
    proposal = _owner_existing_proposal(request_title=f"Add milestone {turn}")
    proposal["changes"][0]["title"] = f"Prepare milestone {turn}"
    return proposal


def _publish(store, previous: str, turn: int, request: str, reply: dict) -> str:
    """Publish turn ``turn`` after ``previous`` as a real owner turn is published."""
    response_id = f"resp_earlier_outcome_turn_{turn}"
    (stored,) = store._conn.execute(
        "SELECT data FROM responses WHERE profile = ? AND response_id = ?",
        (_PLANNER, previous),
    ).fetchone()
    history = json.loads(stored)["conversation_history"] + [
        {"role": "user", "content": request},
        {"role": "assistant", "content": json.dumps(reply)},
    ]
    assert store.publish_owner_turn(
        profile=_PLANNER,
        conversation=_CONVERSATION,
        response_id=response_id,
        data={
            "response": {"id": response_id, "created_at": turn},
            "conversation_history": history,
        },
        owner_proposal=reply["kind"] == "project_change_proposal",
        reservation_id=None,
        expected_previous_response_id=previous,
    ) is True
    return response_id


def _approval(body: dict, response_id: str, turn: int) -> dict:
    """The run body approving the proposal of ``turn``, built as the first one is."""
    approval = copy.deepcopy(body)
    authority = approval["owner_proposal_authority"]
    authority.update(
        response_id=response_id,
        claim_id="claim_" + str(turn) * 32,
        idempotency_key=_key(response_id),
    )
    add = _proposal(turn)["changes"][0]
    change = {key: value for key, value in add.items() if key != "existing_parent_refs"}
    change["existing_parents"] = []
    authority["payload"].update(
        idempotency_key=_key(response_id),
        request_title=f"Add milestone {turn}",
        changes=[change],
    )
    return approval


async def _approved(adapter, body: dict, handler=None) -> tuple[str, str]:
    """Approve through ``POST /v1/runs``; the run id and how the run ended."""
    headers = {
        "Idempotency-Key": body["owner_proposal_authority"]["idempotency_key"],
        "Authorization": f"Bearer {_API_KEY}",
    }
    with _owner_run_patches(adapter, handler):
        async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
            started = await cli.post("/v1/runs", json=body, headers=headers)
            assert started.status == 202, await started.text()
            run_id = (await started.json())["run_id"]
            for _ in range(200):
                if adapter._run_statuses.get(run_id, {}).get("status") in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.05)
    return run_id, adapter._run_statuses[run_id]["status"]


def _refusing(created: dict):
    """The real native plan path refusing for real, as in the sibling file.

    The Project is archived while the plan awaits confirmation, so the native
    path refuses with ``conflict`` and commits its ``ok: false`` receipt.
    """
    from tools import owner_workspace_tools

    native = owner_workspace_tools._handle_project_plan

    def _archived(*_args, **_kw):
        with contextlib.closing(projects_db.connect()) as conn:
            assert projects_db.archive_project(conn, created["project_id"])
        return {"approved": True, "reason": None}

    def _refused(args, **kw):
        with patch.object(ow, "_confirm", side_effect=_archived):
            return native(args, **kw)

    return _refused


async def _five_turns(*, crashed_first: bool = False):
    """The five-turn conversation, reloaded; the adapter and the three runs.

    With ``crashed_first`` the first change is applied but the gateway dies
    before its completion and the recovery writer releases the claim.
    """
    adapter = _make_adapter(api_key=_API_KEY)
    created, body = _proposal_run(adapter)
    if crashed_first:
        first = await _crashed_run(adapter, body, commits=True)
        adapter = _restarted_twice()
        profile, job = _recovery_job(adapter._response_store, first)
        assert adapter._response_store.fail_orphaned_owner_run(
            profile, first, job["owner"],
        ) is True
    else:
        first, status = await _approved(adapter, body)
        assert status == "completed"
    store = adapter._response_store
    second = _publish(store, _RESPONSE_ID, 2, "What about B?", _QUESTION)
    third = _publish(store, second, 3, "Add C.", _proposal(3))
    fourth = _publish(store, third, 4, "Instead add D.", _proposal(4))
    refused, status = await _approved(
        adapter, _approval(body, fourth, 4), _refusing(created),
    )
    assert status == "failed"
    with contextlib.closing(projects_db.connect()) as conn:
        assert projects_db.restore_project(conn, created["project_id"])
    fifth = _publish(store, fourth, 5, "Add E.", _proposal(5))
    applied, status = await _approved(adapter, _approval(body, fifth, 5))
    assert status == "completed"
    return _restarted_twice(), (first, refused, applied)


def _outcomes(history: dict) -> list:
    return [turn.get("proposal_outcome") for turn in history["data"]]


def _marks(store) -> list[tuple]:
    return store._conn.execute(
        "SELECT turn_ordinal, outcome, run_id FROM owner_proposal_outcomes "
        "WHERE profile = ? AND name = ? ORDER BY turn_ordinal",
        (_PLANNER, _CONVERSATION),
    ).fetchall()


def _before_the_catch_up(store, *, drop: bool = False) -> None:
    """The store as production holds it before the catch-up ran: both new
    tables empty or, with ``drop``, a store from before either table."""
    for table in _NEW_TABLES:
        store._conn.execute(f"DROP TABLE {table}" if drop else f"DELETE FROM {table}")
    store._conn.commit()


def _marker(store) -> list[tuple]:
    return store._conn.execute(
        "SELECT profile, name, from_ordinal FROM owner_outcome_tracking WHERE name = ?",
        (_CATCH_UP,),
    ).fetchall()


def _rows(store, *tables: str) -> dict:
    return {
        table: sorted(store._conn.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
        for table in tables
    }


class _FailsAfterTheMark:
    """The store connection, failing the first step after the outcome mark."""

    def __init__(self, conn):
        self.conn = conn
        self.steps = []

    def _step(self, step: str) -> None:
        if self.steps and self.steps[-1].startswith("INSERT INTO owner_proposal_outcomes"):
            raise sqlite3.OperationalError("disk I/O error")
        self.steps.append(step)

    def execute(self, sql, *args):
        self._step(" ".join(sql.split()))
        return self.conn.execute(sql, *args)

    def commit(self):
        self._step("COMMIT")
        return self.conn.commit()

    def __getattr__(self, name):
        return getattr(self.conn, name)


@pytest.mark.asyncio
async def test_earlier_applied_proposal_reads_applied_after_a_newer_turn():
    adapter, (first, refused, applied) = await _five_turns()

    status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history)[0] == "applied"
    # The newest exchange is told by the snapshot flags, as before.
    assert "proposal_outcome" not in history["data"][4]
    assert history["proposal_consumed"] is True
    assert history["latest_response_id"] == _FIFTH
    assert _marks(adapter._response_store) == [
        (1, "applied", first), (4, "released", refused), (5, "applied", applied),
    ]


@pytest.mark.asyncio
async def test_earlier_refused_proposal_reads_not_applied():
    adapter, (_first, refused, _applied) = await _five_turns()
    fourth = "resp_earlier_outcome_turn_4"
    with _receipts_db() as conn:
        [(result_json,)] = conn.execute(
            "SELECT result_json FROM owner_workspace_receipts WHERE idempotency_key = ? "
            "AND operation = ? AND status = 'committed'",
            (_key(fourth), _OPERATION),
        ).fetchall()
    assert json.loads(result_json)["ok"] is False

    status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history)[3] == "not_applied"
    # Read from the refusal receipt under the key the release recorded.
    assert adapter._response_store._conn.execute(
        "SELECT outcome, operation, idempotency_key, run_id "
        "FROM owner_proposal_outcomes WHERE profile = ? AND name = ? AND turn_ordinal = 4",
        (_PLANNER, _CONVERSATION),
    ).fetchone() == ("released", _OPERATION, _key(fourth), refused)


@pytest.mark.asyncio
async def test_earlier_released_but_committed_proposal_reads_applied():
    adapter, (first, _refused, _applied) = await _five_turns(crashed_first=True)

    status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history) == ["applied", None, "replaced", "not_applied", None]
    # Recorded released; only its committed success receipt shows it applied.
    assert _marks(adapter._response_store)[0] == (1, "released", first)


@pytest.mark.asyncio
async def test_earlier_unapproved_proposal_reads_replaced():
    adapter, _runs = await _five_turns()
    store = adapter._response_store

    status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history)[2] == "replaced"
    assert 3 not in [mark[0] for mark in _marks(store)]
    # Tracked from turn 2, the first turn published as an owner turn.
    assert store._conn.execute(
        "SELECT from_ordinal FROM owner_outcome_tracking WHERE profile = ? AND name = ?",
        (_PLANNER, _CONVERSATION),
    ).fetchall() == [(2,)]


@pytest.mark.asyncio
async def test_proposal_before_tracking_reads_unknown():
    adapter, _runs = await _five_turns()
    store = adapter._response_store
    # A conversation from before this change that the catch-up, already run,
    # could not prove: neither its marks nor its tracking row exist.
    for table in _NEW_TABLES:
        store._conn.execute(
            f"DELETE FROM {table} WHERE profile = ? AND name = ?",
            (_PLANNER, _CONVERSATION),
        )
    store._conn.commit()
    row = _conversation_row(store)
    restarted = _restarted_twice()

    status, history = await _history(restarted)

    assert status == 200, history
    assert _outcomes(history) == ["unknown", None, "unknown", "unknown", None]
    assert history["proposal_consumed"] is True
    reloaded = restarted._response_store
    assert _conversation_row(reloaded) == row
    for table in _NEW_TABLES:
        assert reloaded._conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE profile = ? AND name = ?",
            (_PLANNER, _CONVERSATION),
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_newest_turn_has_no_proposal_outcome_and_snapshot_flags_unchanged():
    adapter, (_first, _refused, applied) = await _five_turns()
    store = adapter._response_store

    snapshot = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)

    # Only earlier proposal turns carry the key: not the question, not the newest.
    assert ["proposal_outcome" in turn for turn in snapshot["data"]] == [
        True, False, True, True, False,
    ]
    assert set(snapshot) == {
        "head_response_id", "latest_response_id", "proposal_consumed",
        "proposal_claimed", "active_run_id", "completed_run_id",
        "released_run_id", "conversation_closed", "truncated", "incomplete",
        "pending", "recovery", "data",
    }
    assert snapshot["head_response_id"] == snapshot["latest_response_id"] == _FIFTH
    assert snapshot["proposal_consumed"] is True
    assert snapshot["proposal_claimed"] is False
    assert snapshot["completed_run_id"] == applied
    assert snapshot["released_run_id"] is None and snapshot["active_run_id"] is None
    # Without any record the answer is the same, the key apart.
    store._conn.execute("DELETE FROM owner_proposal_outcomes")
    store._conn.execute("DELETE FROM owner_outcome_tracking")
    store._conn.commit()
    unrecorded = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)
    for turn in (*snapshot["data"], *unrecorded["data"]):
        turn.pop("proposal_outcome", None)
    assert unrecorded == snapshot


@pytest.mark.asyncio
async def test_unreadable_receipts_store_reads_unknown_not_failed(tmp_path):
    adapter, _runs = await _five_turns()
    unreadable = tmp_path / "unreadable-executor"
    unreadable.mkdir()
    (unreadable / "projects.db").write_bytes(b"not a database " * 128)

    with patch.object(ow, "owner_executor_home", return_value=unreadable):
        status, history = await _history(adapter)

    assert status == 200, history
    # Only the released turn needed the receipts store; the rest keep their answer.
    assert _outcomes(history) == ["applied", None, "replaced", "unknown", None]


@pytest.mark.asyncio
async def test_history_load_opens_the_receipts_store_at_most_once():
    # Two released marks: turn 1 applied, turn 4 refused.
    adapter, _runs = await _five_turns(crashed_first=True)
    store = adapter._response_store
    receipts = (ow.owner_executor_home() / "projects.db").read_bytes()
    changes = store._conn.total_changes
    opened, statements = [], []
    connect = ow.sqlite3.connect

    def _traced(*args, **kw):
        conn = connect(*args, **kw)
        if "projects.db" in str(args[0]):
            opened.append(str(args[0]))
            conn.set_trace_callback(statements.append)
        return conn

    with patch.object(ow.sqlite3, "connect", side_effect=_traced):
        status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history) == ["applied", None, "replaced", "not_applied", None]
    assert len(opened) == 1 and opened[0].endswith("?mode=ro")
    # Its table check and one batched read, for both released turns.
    assert len(statements) == 2
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
    # The history read writes nothing, in either store.
    assert store._conn.total_changes == changes
    assert (ow.owner_executor_home() / "projects.db").read_bytes() == receipts


@pytest.mark.asyncio
async def test_applied_mark_never_downgrades_to_released():
    adapter = _make_adapter(api_key=_API_KEY)
    _created, body = _proposal_run(adapter)
    crashed = await _crashed_run(adapter, body, commits=False)
    adapter = _restarted_twice()
    store = adapter._response_store
    profile, job = _recovery_job(store, crashed)
    assert store.fail_orphaned_owner_run(profile, crashed, job["owner"]) is True
    assert _marks(store) == [(1, "released", crashed)]

    retried, status = await _approved(adapter, body)

    assert status == "completed"
    assert _marks(store) == [(1, "applied", retried)]
    applied = store._conn.execute("SELECT * FROM owner_proposal_outcomes").fetchall()
    # A late release of the same proposal leaves the applied record as it was.
    with store._conversation_lock:
        store._conn.execute("BEGIN IMMEDIATE")
        store._record_owner_proposal_outcome_locked(
            _PLANNER, _CONVERSATION, _RESPONSE_ID, "released",
            _OPERATION, _KEY, "f" * 64, crashed,
        )
        store._conn.commit()
    assert store._conn.execute("SELECT * FROM owner_proposal_outcomes").fetchall() == applied
    _publish(store, _RESPONSE_ID, 2, "What about B?", _QUESTION)
    status, history = await _history(adapter)
    assert status == 200, history
    assert _outcomes(history) == ["applied", None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer", "outcome"),
    [
        ("completion", "applied"),
        ("complete", "applied"),
        ("consume", "applied"),
        ("release", "released"),
        ("orphan", "released"),
    ],
)
async def test_records_written_in_the_same_transaction(writer, outcome):
    adapter = _make_adapter(api_key=_API_KEY)
    _created, body = _proposal_run(adapter)
    if writer == "consume":
        store = adapter._response_store
        write = functools.partial(
            store.mark_owner_proposal_consumed, _PLANNER, _CONVERSATION, _RESPONSE_ID,
        )
    else:
        # A claimed run nobody finished; with "completion" its change committed.
        run_id = await _crashed_run(adapter, body, commits=writer == "completion")
        adapter = _restarted_twice()
        store = adapter._response_store
        profile, job = _recovery_job(store, run_id)
        claim = (_PLANNER, _CONVERSATION, _RESPONSE_ID, _CLAIM_ID, run_id)
        write = {
            "completion": functools.partial(
                adapter._reconcile_orphaned_native_run,
                {"profile": profile, "job_key": run_id, "payload": job},
            ),
            "complete": functools.partial(store.complete_owner_claim, *claim),
            "release": functools.partial(store.release_owner_claim, *claim),
            "orphan": functools.partial(
                store.fail_orphaned_owner_run, profile, run_id, job["owner"],
            ),
        }[writer]
    row = _conversation_row(store)
    connection = store._conn
    store._conn = faulty = _FailsAfterTheMark(connection)
    try:
        with pytest.raises((sqlite3.OperationalError, RuntimeError)):
            write()
    finally:
        store._conn = connection

    # The mark was written after the conversation update with no commit
    # between them, and the failure after it left neither.
    assert faulty.steps[-1].startswith("INSERT INTO owner_proposal_outcomes")
    updated = max(
        index for index, step in enumerate(faulty.steps)
        if step.startswith("UPDATE conversations")
    )
    assert "COMMIT" not in faulty.steps[updated:]
    assert not connection.in_transaction
    assert _conversation_row(store) == row
    assert _marks(store) == []

    assert write() is True
    assert _conversation_row(store) != row
    assert [mark[:2] for mark in _marks(store)] == [(1, outcome)]


@pytest.mark.asyncio
async def test_catch_up_marks_a_surviving_applied_proposal():
    adapter, _runs = await _five_turns()
    store = adapter._response_store
    recorded = store._conn.execute(
        "SELECT turn_ordinal, response_id, proposal_digest FROM owner_proposal_outcomes "
        "WHERE outcome = 'applied' ORDER BY turn_ordinal",
    ).fetchall()
    _before_the_catch_up(store)

    adapter = _restarted_twice()
    store = adapter._response_store
    status, history = await _history(adapter)

    assert status == 200, history
    # Turn 1 by its committed success receipt, turn 5 as the consumed proposal:
    # the same turns and proposals step 1 recorded. The refusal of turn 4 and
    # the never approved turn 3 prove nothing here, so both stay unknown.
    assert _outcomes(history) == ["applied", None, "unknown", "unknown", None]
    assert history["proposal_consumed"] is True
    assert _marks(store) == [(1, "applied", _CATCH_UP), (5, "applied", _CATCH_UP)]
    assert store._conn.execute(
        "SELECT turn_ordinal, response_id, proposal_digest FROM owner_proposal_outcomes "
        "ORDER BY turn_ordinal",
    ).fetchall() == recorded
    assert store._conn.execute(
        "SELECT operation, idempotency_key, authority_digest FROM owner_proposal_outcomes "
        "WHERE turn_ordinal = 1",
    ).fetchone() == (_OPERATION, _KEY, None)
    # Tracking is left as it was: no conversation row, only the marker.
    assert store._conn.execute(
        "SELECT profile, name, from_ordinal FROM owner_outcome_tracking",
    ).fetchall() == [("", _CATCH_UP, 0)]


@pytest.mark.asyncio
async def test_catch_up_leaves_an_evicted_proposal_unknown():
    adapter, _runs = await _five_turns()
    store = adapter._response_store
    # A store from before step 1 whose turn 1 response the stored-response cap
    # evicted. Its committed success receipt remains, under a key that no
    # stored row names any more.
    _before_the_catch_up(store, drop=True)
    store._conn.execute(
        "DELETE FROM responses WHERE profile = ? AND response_id = ?",
        (_PLANNER, _RESPONSE_ID),
    )
    store._conn.commit()
    with _receipts_db() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM owner_workspace_receipts "
            "WHERE idempotency_key = ? AND status = 'committed'",
            (_KEY,),
        ).fetchone() == (1,)

    adapter = _restarted_twice()
    status, history = await _history(adapter)

    assert status == 200, history
    assert _outcomes(history) == ["unknown", None, "unknown", "unknown", None]
    # Only the consumed fifth proposal is proven; nothing is guessed.
    assert _marks(adapter._response_store) == [(5, "applied", _CATCH_UP)]


@pytest.mark.asyncio
async def test_catch_up_runs_once_and_is_idempotent():
    adapter, _runs = await _five_turns()
    _before_the_catch_up(adapter._response_store)

    store = _restarted_twice()._response_store
    once = _rows(store, *_NEW_TABLES)

    assert _marks(store) == [(1, "applied", _CATCH_UP), (5, "applied", _CATCH_UP)]
    assert _marker(store) == [("", _CATCH_UP, 0)]
    # Run again without its marker, it writes the same rows and changes none.
    store._conn.execute("DELETE FROM owner_outcome_tracking WHERE name = ?", (_CATCH_UP,))
    store._conn.commit()
    store = _restarted_twice()._response_store
    assert _rows(store, *_NEW_TABLES) == once
    # Once its marker is set it never runs again, also not after a restart.
    store._conn.execute("DELETE FROM owner_proposal_outcomes WHERE turn_ordinal = 1")
    store._conn.commit()
    adapter = _restarted_twice()
    assert _marks(adapter._response_store) == [(5, "applied", _CATCH_UP)]
    status, history = await _history(adapter)
    assert status == 200, history
    assert _outcomes(history)[0] == "unknown"


@pytest.mark.asyncio
async def test_catch_up_never_writes_conversations_or_receipts():
    adapter, _runs = await _five_turns()
    store = adapter._response_store
    _before_the_catch_up(store)
    tables = [
        name for (name,) in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if name not in _NEW_TABLES
    ]
    assert {"conversations", "responses", "run_idempotency"} <= set(tables)
    before = _rows(store, *tables)
    receipts = (ow.owner_executor_home() / "projects.db").read_bytes()
    opened, statements = [], []
    connect = ow.sqlite3.connect

    def _traced(*args, **kw):
        conn = connect(*args, **kw)
        if "projects.db" in str(args[0]):
            opened.append(str(args[0]))
            conn.set_trace_callback(statements.append)
        return conn

    with patch.object(ow.sqlite3, "connect", side_effect=_traced):
        store = _restarted_twice()._response_store

    # Its marks and its marker are all it wrote, in either store.
    assert _marks(store) == [(1, "applied", _CATCH_UP), (5, "applied", _CATCH_UP)]
    assert _rows(store, *tables) == before
    assert (ow.owner_executor_home() / "projects.db").read_bytes() == receipts
    # One read-only open of the receipts store, which it only read.
    assert len(opened) == 1 and opened[0].endswith("?mode=ro")
    assert statements and all(
        sql.lstrip().upper().startswith("SELECT") for sql in statements
    )
