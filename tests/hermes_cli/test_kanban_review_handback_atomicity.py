"""A review handback is all three records or none of them.

``submit_review_findings`` composes three durable facts: the findings
attachment, the ``request_changes`` transition (task status + assignee +
``changes_requested`` event) and the ``review_findings_delivered`` event that
later handbacks compare against. Any partial outcome is a real failure mode
rather than a tidiness concern:

* transition WITHOUT the delivered event — the task is handed back to the
  implementer with the document attached but no delivery record, so the next
  reviewer's identical findings are not recognised as a repeat and, worse,
  ``_review_findings_history`` never learns the delivery happened at all; and
* delivered event WITHOUT the transition — the audit trail claims a handback
  that never occurred, and the reviewer's retry is read as a REPEAT of that
  phantom handback, sticky-blocking the card after zero real handbacks.

The delivered-event write is forced to fail here with a real SQLite trigger
that aborts only that one insert. Nothing else is stubbed: the store, the
transition, the compensation and the retry are all the production path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _finding(**overrides) -> dict:
    base = {
        "severity": "blocking",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "off-by-one in the loop bound",
        "impact": "drops the last item in the batch",
        "smallest_fix": "use `<=` instead of `<` in the range check",
        "candidate_digest": "digest-1",
    }
    base.update(overrides)
    return base


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _hand_off_to_review(conn, title="reviewed task", *, reviewer="reviewer"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    implementation = kb.claim_task(conn, tid, claimer="worker:1")
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", reviewer=reviewer,
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, tid, claimer=f"{reviewer}:1")
    assert review is not None
    return tid, review


def _events(conn, tid, kind):
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _stored_blobs(task_id) -> list[str]:
    directory = kb.task_attachments_dir(task_id)
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir())


_ABORT_DELIVERED_TRIGGER = """
CREATE TRIGGER hermes_test_abort_delivered
BEFORE INSERT ON task_events
WHEN NEW.kind = 'review_findings_delivered'
BEGIN
    SELECT RAISE(ABORT, 'simulated storage fault on the delivered event');
END
"""

_ABORT_DELIVERED_AND_REMOVED_TRIGGER = """
CREATE TRIGGER hermes_test_abort_delivered_and_removed
BEFORE INSERT ON task_events
WHEN NEW.kind IN ('review_findings_delivered', 'attachment_removed')
BEGIN
    SELECT RAISE(ABORT, 'simulated storage fault on delivered/removed events');
END
"""


def _break_delivered_event_write(conn) -> None:
    conn.execute(_ABORT_DELIVERED_TRIGGER)
    conn.commit()


def _break_delivered_and_removed_event_write(conn) -> None:
    conn.execute(_ABORT_DELIVERED_AND_REMOVED_TRIGGER)
    conn.commit()


def _repair_delivered_event_write(conn) -> None:
    conn.execute("DROP TRIGGER hermes_test_abort_delivered")
    conn.commit()


def _repair_delivered_and_removed_event_write(conn) -> None:
    conn.execute("DROP TRIGGER hermes_test_abort_delivered_and_removed")
    conn.commit()


def test_delivered_event_failure_rolls_the_whole_handback_back(kanban_home):
    """The failing write is the LAST one, which is exactly why it matters.

    On the broken composition the transition had already committed by the
    time the delivered event failed, so the card was handed back with the
    findings attached and no delivery record — the worst of the two partial
    states, and unrecoverable by a retry.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        _break_delivered_event_write(conn)

        with pytest.raises(sqlite3.Error):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        # Nothing landed: not the transition, not its event, not the
        # delivery record, not the attachment row, not the blob.
        assert _events(conn, tid, "review_findings_delivered") == []
        assert _events(conn, tid, "changes_requested") == [], (
            "the request-changes transition committed without its delivered "
            "event; the implementer now has the findings with no delivery "
            "record behind them"
        )
        assert kb.list_attachments(conn, tid) == []
        assert _stored_blobs(tid) == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", (
            f"the review run was ended by a failed handback; task is "
            f"{task.status!r}"
        )
        assert task.current_run_id == review.current_run_id
        assert task.assignee == "reviewer"

        # And the reviewer's retry, once storage is healthy, is a FIRST
        # handback rather than a repeat of a phantom one.
        _repair_delivered_event_write(conn)
        retry = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert retry["outcome"] == "handed_back", retry
        assert len(_events(conn, tid, "changes_requested")) == 1
        delivered = _events(conn, tid, "review_findings_delivered")
        assert len(delivered) == 1
        assert delivered[0]["attachment_id"] == retry["attachment_id"]
        assert _events(conn, tid, "review_findings_repeated") == []
        assert len(kb.list_attachments(conn, tid)) == 1
        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.status == "ready"
        assert after.assignee == "worker"


def test_successful_handback_lands_transition_and_delivery_together(kanban_home):
    """The success path is one atomic step, observable as one event id run.

    ``changes_requested`` and ``review_findings_delivered`` must both exist
    and share the same run id — a delivered event recorded in a separate
    later transaction is precisely what could go missing.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="atomic success")
        result = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert result["outcome"] == "handed_back"

        rows = conn.execute(
            "SELECT kind, run_id FROM task_events WHERE task_id = ? "
            "AND kind IN ('changes_requested', 'review_findings_delivered') "
            "ORDER BY id",
            (tid,),
        ).fetchall()
        kinds = [r["kind"] for r in rows]
        assert kinds == ["changes_requested", "review_findings_delivered"]
        assert {r["run_id"] for r in rows} == {review.current_run_id}


def test_a_refused_transition_still_compensates_the_attachment(kanban_home):
    """The refusal path keeps its existing all-or-nothing guarantee.

    ``role_transition_route`` refuses the handback on migrated owner work.
    The restructured composition must roll the attachment back there too,
    and must still report the refusal structurally rather than raising.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="refused handback")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET owner_receipt_bound = 1 WHERE id = ?", (tid,),
            )

        refused = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert refused["outcome"] == "error", refused
        assert refused.get("reason")
        assert _events(conn, tid, "review_findings_delivered") == []
        assert _events(conn, tid, "changes_requested") == []
        assert kb.list_attachments(conn, tid) == []
        assert _stored_blobs(tid) == []
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == review.current_run_id


def test_request_changes_alone_is_unchanged(kanban_home):
    """The standalone transition keeps its own contract.

    ``request_changes`` is a public mutator in its own right (the
    ``hermes kanban request-changes`` command). Factoring its body out so a
    composed handback can share one transaction with it must not change what
    it does on its own, refusals included.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="plain request changes")

        assert kb.request_changes(conn, tid, reason="   ") == (
            False, "reason is required",
        )
        assert kb.request_changes(
            conn, tid, reason="fix it", expected_run_id=review.current_run_id + 999,
        ) == (False, "run_id mismatch")

        ok, implementer = kb.request_changes(
            conn, tid, reason="fix the loop bound",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, "worker")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "worker"
        assert task.current_run_id is None
        changes = _events(conn, tid, "changes_requested")
        assert len(changes) == 1
        assert changes[0]["reason"] == "fix the loop bound"
        assert changes[0]["implementer"] == "worker"
        assert changes[0]["reviewer"] == "reviewer"
        # A standalone transition delivers no findings document.
        assert _events(conn, tid, "review_findings_delivered") == []

        # And it refuses a second time: the review run is over.
        assert kb.request_changes(conn, tid, reason="again") == (
            False, "task is not in an active review run",
        )


def test_both_delivered_and_removed_events_blocked_leaves_no_orphan(kanban_home):
    """When the compensating ``attachment_removed`` event is ALSO rejected,
    no stale attachment row or blob may survive.

    The existing test_delivered_event_failure test only blocks the
    ``review_findings_delivered`` insert, so the compensating
    ``delete_attachment`` — which appends ``attachment_removed`` — always
    succeeds and hides the real atomicity gap. This test blocks BOTH event
    kinds, proving that the handback is truly all-or-nothing even when
    compensation through ``delete_attachment`` cannot write its own event.

    A surviving attachment row + blob is dangerous: the owner sees a
    ``review_findings.json`` that looks like a delivered handback even
    though no delivery and no request-changes transition ever happened.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        _break_delivered_and_removed_event_write(conn)

        with pytest.raises(sqlite3.Error):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        # The transition must not have landed.
        assert _events(conn, tid, "changes_requested") == [], (
            "the request-changes transition committed despite the storage "
            "fault — the implementer sees a handback with no delivery record"
        )
        assert _events(conn, tid, "review_findings_delivered") == [], (
            "a review_findings_delivered event survived the fault"
        )
        # The attachment row and blob must not survive: a stale
        # review_findings.json visible to the owner looks like a
        # delivered handback even though no delivery ever happened.
        assert kb.list_attachments(conn, tid) == [], (
            "a stale attachment row survived — the owner sees a "
            "review_findings.json that was never delivered"
        )
        assert _stored_blobs(tid) == [], (
            "a stale blob survived on disk — the review_findings.json "
            "file is owner-visible even though no handback occurred"
        )
        # No attached receipt should remain either.
        assert _events(conn, tid, "attached") == [], (
            "an 'attached' receipt survived for a document that was "
            "never delivered — the audit trail is inconsistent"
        )

        # Task must still be running with the review run intact.
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", (
            f"the review run was ended by a failed handback; task is "
            f"{task.status!r}"
        )
        assert task.current_run_id == review.current_run_id
        assert task.assignee == "reviewer"

        # After the fault clears, the reviewer's retry is a FIRST handback
        # (not a repeat of a phantom one).
        _repair_delivered_and_removed_event_write(conn)
        retry = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert retry["outcome"] == "handed_back", retry
        assert len(_events(conn, tid, "changes_requested")) == 1
        delivered = _events(conn, tid, "review_findings_delivered")
        assert len(delivered) == 1
        assert delivered[0]["attachment_id"] == retry["attachment_id"]
        assert _events(conn, tid, "review_findings_repeated") == []
        assert len(kb.list_attachments(conn, tid)) == 1
        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.status == "ready"
        assert after.assignee == "worker"


def test_oversized_document_refused_and_leaves_nothing_behind(kanban_home, monkeypatch):
    """The staged-blob path enforces KANBAN_ATTACHMENT_MAX_BYTES.

    The old code delegated to ``store_attachment_bytes``, which checked the
    cap before writing. The restructured single-transaction path must check
    the cap itself — before staging the blob — so an oversized document is
    refused without writing any file, row, event, or transition.
    """
    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 64)
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)

        with pytest.raises(kb.AttachmentTooLarge):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        assert kb.list_attachments(conn, tid) == []
        assert _stored_blobs(tid) == []
        assert _events(conn, tid, "changes_requested") == []
        assert _events(conn, tid, "review_findings_delivered") == []
        assert _events(conn, tid, "attached") == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == review.current_run_id


def test_concurrent_collision_winner_document_survives(kanban_home):
    """Two concurrent handbacks on separate connections must not destroy
    each other's staged blob, and exactly one of them wins the single
    review run's request-changes transition.

    Each caller submits a DISTINCT document (different finding text and a
    different ``candidate_digest``) so the committed row can be proven
    byte-for-byte to be the WINNER's document and not the loser's -- two
    callers racing over the identical document could never tell whose
    bytes actually landed.
    """
    barrier = threading.Barrier(2, timeout=30)
    barrier_hits = 0
    barrier_lock = threading.Lock()

    with kb.connect() as setup_conn:
        tid, review = _hand_off_to_review(setup_conn)

    dest_dir = kb.task_attachments_dir(tid)
    real_open = kb.os.open

    def _synced_open(path, flags, *args, **kwargs):
        nonlocal barrier_hits
        # Call the real syscall first: a colliding O_EXCL create must raise
        # FileExistsError here and propagate untouched, never reaching the
        # barrier below -- only a caller that actually reserved a name
        # (a successful O_EXCL create in this task's attachments directory)
        # is made to wait for the other caller to reserve its own name too.
        fd = real_open(path, flags, *args, **kwargs)
        if flags & os.O_EXCL and Path(path).parent == dest_dir:
            with barrier_lock:
                barrier_hits += 1
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return fd

    documents = [
        {
            "finding": _finding(
                problem="caller-0's unique off-by-one in the batch loop",
                lines="10-20",
                candidate_digest="digest-A",
            ),
            "digest": "digest-A",
        },
        {
            "finding": _finding(
                problem="caller-1's unique unchecked null in the parser",
                lines="30-40",
                candidate_digest="digest-B",
            ),
            "digest": "digest-B",
        },
    ]

    results = [None, None]

    def worker(idx):
        try:
            with kb.connect() as conn:
                r = kb.submit_review_findings(
                    conn, tid,
                    findings=[documents[idx]["finding"]],
                    candidate_digest=documents[idx]["digest"],
                    expected_run_id=review.current_run_id,
                )
                results[idx] = r
        except Exception as exc:
            results[idx] = {"outcome": "exception", "reason": str(exc)}

    kb.os.open = _synced_open
    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=35)
    finally:
        kb.os.open = real_open

    assert barrier_hits == 2, (
        f"the concurrency barrier was reached {barrier_hits} time(s) instead of 2 — "
        f"the implementation did not perform two exclusive (O_EXCL) creates in the "
        f"task's attachments directory, so this test proves nothing about exclusive "
        f"stage reservation"
    )

    for i, t in enumerate(threads):
        assert not t.is_alive(), f"worker thread {i} never terminated"

    assert results[0] is not None and results[1] is not None, results

    winners = [i for i, r in enumerate(results) if r.get("outcome") == "handed_back"]
    losers = [i for i, r in enumerate(results) if r.get("outcome") != "handed_back"]
    assert len(winners) == 1, f"expected exactly one winner: {results}"
    assert len(losers) == 1, f"expected exactly one loser: {results}"
    winner_idx = winners[0]
    loser_idx = losers[0]
    winner_result = results[winner_idx]
    winner_digest = documents[winner_idx]["digest"]
    loser_problem = documents[loser_idx]["finding"]["problem"]
    loser_digest = documents[loser_idx]["digest"]

    with kb.connect() as conn:
        attachments = kb.list_attachments(conn, tid)
        assert len(attachments) == 1, (
            f"expected exactly 1 attachment row, got {len(attachments)}"
        )
        att = attachments[0]
        delivered = _events(conn, tid, "review_findings_delivered")
        assert len(delivered) == 1
        assert delivered[0]["attachment_id"] == att.id == winner_result["attachment_id"]
        assert delivered[0]["candidate_digest"] == winner_digest

        blob_path = Path(att.stored_path)
        assert blob_path.exists(), (
            "the winner's staged blob was deleted by the loser's cleanup — "
            "the attachment row points at nothing"
        )
        committed = kb.read_attachment_bytes(att)
        assert len(committed) == att.size, (
            f"blob size {len(committed)} != row size {att.size}"
        )

        doc = json.loads(committed)
        assert doc["candidate_digest"] == winner_digest, (
            "the committed document's digest is not the winner's — the "
            "loser's document survived instead"
        )
        assert doc["candidate_digest"] != loser_digest

        # The expected bytes are built INDEPENDENTLY from the winner's own
        # input — never by re-serialising the committed bytes, which would
        # only prove they are canonical JSON of themselves and would accept
        # any same-shape corruption of the winner's content. The document is
        # fully deterministic from its input (schema_version, candidate_digest
        # and the validated findings, each with a content-derived
        # fingerprint), so no field has to be substituted from the readback
        # and every byte is compared. The serialisation below mirrors the
        # production submit path exactly (``json.dumps(document,
        # ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")``).
        expected_doc = kb.build_review_findings_document(
            [documents[winner_idx]["finding"]],
            candidate_digest=winner_digest,
        )
        expected_bytes = json.dumps(
            expected_doc, ensure_ascii=False, indent=2, sort_keys=True,
        ).encode("utf-8")
        assert committed == expected_bytes, (
            "the committed blob is not byte-for-byte the winner's own "
            "document: expected "
            f"{expected_bytes!r} but the surviving row reads back {committed!r}"
        )
        assert loser_problem.encode("utf-8") not in committed, (
            "the loser's finding text leaked into the committed blob"
        )
        assert loser_digest.encode("utf-8") not in committed, (
            "the loser's digest leaked into the committed blob"
        )


def test_mid_write_failure_leaves_nothing_behind(kanban_home, monkeypatch):
    """A partial blob write that raises mid-way must not leave an orphan file.

    The blob is written through the SAME exclusive descriptor that reserved
    its name (``_write_exclusive_stage_blob`` never reopens the name via
    ``Path.write_bytes``), so the fault must be injected at the ``os.write``
    layer that descriptor actually uses.
    """
    call_count = 0
    original_write = kb.os.write

    def _failing_write(fd, data):
        nonlocal call_count
        call_count += 1
        original_write(fd, data[:17])
        raise OSError("simulated short write")

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        dest_dir = kb.task_attachments_dir(tid)
        monkeypatch.setattr(kb.os, "write", _failing_write)

        with pytest.raises(OSError, match="simulated short write"):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        assert call_count >= 1, "the write monkeypatch was never reached"
        assert dest_dir.exists()
        assert sorted(p.name for p in dest_dir.iterdir()) == [], (
            "an orphan blob survived a failed write — the file was created "
            "but never cleaned up"
        )
        assert kb.list_attachments(conn, tid) == []
        assert _events(conn, tid, "changes_requested") == []
        assert _events(conn, tid, "review_findings_delivered") == []
        assert _events(conn, tid, "attached") == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", (
            f"task should still be running, got {task.status!r}"
        )
        assert task.current_run_id == review.current_run_id


def test_post_create_path_boundary_failure_leaves_no_orphan(
    kanban_home, monkeypatch,
):
    """A fault at the FIRST operation after the exclusive create — before the
    write, before the close — must not orphan the reserved blob.

    ``os.open(..., O_CREAT | O_EXCL)`` returning means the file already
    exists on disk. Any statement executed between that moment and the
    moment a cleanup handler owns both the descriptor and the reserved name
    is a gap: a plain ``dest_dir / candidate`` path join is itself an
    operation that can raise, and one performed after the create leaves an
    untracked ``review_findings.json`` behind with the descriptor still open.

    The fault here is armed by the successful exclusive create and fires on
    the FIRST path-level operation on the reserved path afterwards:

    * a helper that joins ``dest_dir / candidate`` again after the create
      takes the hit inside that gap (the reserved name is not yet held by
      any cleanup handler), and orphans the blob; while
    * a helper that computes the path BEFORE the create and enters its
      cleanup region immediately performs no such join, so the first
      post-create path operation on the reserved path is the caller's
      ``staged_path.resolve()``, which is inside
      ``submit_review_findings``'s own ``_unlink_staged_blob`` region.

    Either way the handback must be all-or-nothing. Neither the write nor
    the close is touched (those are covered by
    ``test_mid_write_failure_leaves_nothing_behind`` and
    ``test_close_failure_after_exclusive_create_leaves_no_orphan``).
    """
    real_open = kb.os.open
    real_div = Path.__truediv__
    real_resolve = Path.resolve

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        # Capture the attachments directory UP FRONT and assert against this
        # captured path: resolving it later (after any patching) could point
        # somewhere else entirely and make an orphan look like a clean result.
        dest_dir = kb.task_attachments_dir(tid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_name = kb.REVIEW_FINDINGS_ATTACHMENT_FILENAME
        reserved = real_div(dest_dir, safe_name)

        state = {"armed": False, "post_create_joins": 0, "fired": None}

        def _arming_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            # The create SUCCEEDED: from this instant on, the file exists.
            if flags & os.O_EXCL and str(path) == str(reserved):
                state["armed"] = True
            return fd

        def _trapped_div(self, other):
            if (
                state["armed"]
                and state["fired"] is None
                and isinstance(other, str)
                and str(real_div(self, other)) == str(reserved)
            ):
                state["post_create_joins"] += 1
                state["fired"] = "path-join after the exclusive create"
                raise RuntimeError("simulated post-create boundary fault")
            return real_div(self, other)

        def _trapped_resolve(self, *args, **kwargs):
            if (
                state["armed"]
                and state["fired"] is None
                and str(self) == str(reserved)
            ):
                state["fired"] = "resolve() of the reserved path"
                raise RuntimeError("simulated post-create boundary fault")
            return real_resolve(self, *args, **kwargs)

        monkeypatch.setattr(kb.os, "open", _arming_open)
        monkeypatch.setattr(Path, "__truediv__", _trapped_div)
        monkeypatch.setattr(Path, "resolve", _trapped_resolve)
        try:
            result = kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )
        finally:
            # Disarm the trap for the assertions below WITHOUT calling
            # monkeypatch.undo(): undo() would also revert the kanban_home
            # fixture's HERMES_HOME and silently redirect every later lookup.
            state["armed"] = False

        assert state["fired"] is not None, (
            "the post-create boundary fault never fired — the reserved path "
            "was never touched after the exclusive create, so this test "
            "proves nothing"
        )
        assert result["outcome"] == "error", result

        # Nothing survives the fault: no blob under the captured directory,
        # no row, no receipt, no transition, no delivery record.
        assert dest_dir.exists()
        assert sorted(p.name for p in dest_dir.iterdir()) == [], (
            "an orphan blob survived a fault at the post-create boundary "
            f"({state['fired']}); the exclusive create had already made the "
            f"file but nothing owned it yet "
            f"(post-create joins: {state['post_create_joins']})"
        )
        assert kb.list_attachments(conn, tid) == []
        assert _events(conn, tid, "attached") == []
        assert _events(conn, tid, "changes_requested") == []
        assert _events(conn, tid, "review_findings_delivered") == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", (
            f"task should still be running, got {task.status!r}"
        )
        assert task.current_run_id == review.current_run_id
        assert task.assignee == "reviewer"


def test_close_failure_after_exclusive_create_leaves_no_orphan(
    kanban_home, monkeypatch,
):
    """A failure AFTER the exclusive create (e.g. on close) must not orphan
    the reserved blob, and must leave no durable record behind either.

    The old ``_exclusive_stage_path`` helper closed the reserving descriptor
    and returned only the ``Path`` -- the caller's ``staged_path = ...``
    assignment had not run yet if anything raised between that close and
    the return, so cleanup received ``None`` and the reserved file was
    orphaned. ``_write_exclusive_stage_blob`` must instead track its own
    reserved path internally and unlink it on ANY failure after the create,
    including a failure of the close itself.
    """
    call_count = 0
    original_close = kb.os.close

    def _failing_close(fd):
        nonlocal call_count
        call_count += 1
        original_close(fd)
        raise OSError("simulated close fault")

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        dest_dir = kb.task_attachments_dir(tid)
        monkeypatch.setattr(kb.os, "close", _failing_close)

        with pytest.raises(OSError, match="simulated close fault"):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        assert call_count >= 1, "the close monkeypatch was never reached"
        assert dest_dir.exists()
        assert sorted(p.name for p in dest_dir.iterdir()) == [], (
            "the reserved blob was orphaned by a post-create close failure"
        )
        assert kb.list_attachments(conn, tid) == []
        assert _events(conn, tid, "changes_requested") == []
        assert _events(conn, tid, "review_findings_delivered") == []
        assert _events(conn, tid, "attached") == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", (
            f"task should still be running, got {task.status!r}"
        )
        assert task.current_run_id == review.current_run_id
