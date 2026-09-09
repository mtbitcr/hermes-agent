"""Typed review-findings handback (kanban kernel).

Covers the whole ``submit_review_findings`` surface end to end: the typed
document schema + validator, the durable handback through the existing
``request_changes`` transition, resolved-finding dropping, the
identical-findings-twice owner-decision block, the clean-pass approval, the
kernel-supplied reviewer contract in ``build_worker_context``, and the CLI +
dashboard operator surfaces.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _finding(**overrides) -> dict:
    base = {
        "severity": "major",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "off-by-one in the loop bound",
        "impact": "drops the last item in the batch",
        "smallest_fix": "use `<=` instead of `<` in the range check",
        "candidate_digest": "digest-1",
    }
    base.update(overrides)
    return base


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _hand_off_to_review(conn, title="reviewed task", *, reviewer="reviewer"):
    """implementer claims -> request_review -> reviewer claims from review."""
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


# ---------------------------------------------------------------------------
# 1. Typed document schema + validator
# ---------------------------------------------------------------------------


def test_document_schema_has_all_seven_fields_and_validator_rejects_malformed():
    doc = kb.build_review_findings_document(
        [_finding()], candidate_digest="digest-1",
    )
    assert doc["schema_version"] == kb.REVIEW_FINDINGS_SCHEMA_VERSION
    assert doc["candidate_digest"] == "digest-1"
    assert len(doc["findings"]) == 1
    finding = doc["findings"][0]
    for field in (
        "severity", "file", "lines", "problem", "impact", "smallest_fix",
        "candidate_digest",
    ):
        assert field in finding and finding[field]
    assert finding["fingerprint"]

    # A stored document round-trips through the parser.
    reparsed = kb.parse_review_findings_document(json.loads(json.dumps(doc)))
    assert reparsed == doc

    # Rewording impact doesn't change the fingerprint (deliberately excluded
    # from the fingerprint so model-reworded rationale doesn't mint a new
    # identity for the same defect).
    reworded = kb.build_review_findings_document(
        [_finding(impact="a different way of describing the same drop")],
        candidate_digest="digest-1",
    )
    assert reworded["findings"][0]["fingerprint"] == finding["fingerprint"]

    # Changing the problem text DOES change the fingerprint.
    different = kb.build_review_findings_document(
        [_finding(problem="a completely different defect")],
        candidate_digest="digest-1",
    )
    assert different["findings"][0]["fingerprint"] != finding["fingerprint"]

    # Missing field.
    with pytest.raises(kb.ReviewFindingsError, match="missing required field"):
        kb.build_review_findings_document(
            [{k: v for k, v in _finding().items() if k != "smallest_fix"}],
            candidate_digest="digest-1",
        )

    # Unknown severity.
    with pytest.raises(kb.ReviewFindingsError, match="severity"):
        kb.build_review_findings_document(
            [_finding(severity="catastrophic")], candidate_digest="digest-1",
        )

    # Non-string digest.
    with pytest.raises(kb.ReviewFindingsError):
        kb.build_review_findings_document(
            [_finding(candidate_digest=123)], candidate_digest="digest-1",
        )

    # Per-finding digest disagreeing with the document header.
    with pytest.raises(kb.ReviewFindingsError, match="candidate_digest"):
        kb.build_review_findings_document(
            [_finding(candidate_digest="other-digest")], candidate_digest="digest-1",
        )


# ---------------------------------------------------------------------------
# 2. End-to-end handback through the real kernel entry point
# ---------------------------------------------------------------------------


def test_handback_end_to_end_attaches_document_and_next_worker_sees_it(kanban_home):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)

        result = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert result["outcome"] == "handed_back"
        assert result["implementer"] == "worker"
        attachment_id = result["attachment_id"]

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "worker"
        assert task.current_run_id is None

        # The document is ONE attachment on the implementer's task.
        attachments = kb.list_attachments(conn, tid)
        assert len(attachments) == 1
        assert attachments[0].id == attachment_id
        assert attachments[0].filename == kb.REVIEW_FINDINGS_ATTACHMENT_FILENAME
        stored_doc = json.loads(kb.read_attachment_bytes(attachments[0]))
        parsed = kb.parse_review_findings_document(stored_doc)
        assert len(parsed["findings"]) == 1
        assert parsed["findings"][0]["problem"] == _finding()["problem"]

        # Routed back through the real request_changes transition.
        changes = _events(conn, tid, kind="changes_requested")
        assert len(changes) == 1
        assert str(attachment_id) in changes[0][1]["reason"]
        assert kb.REVIEW_FINDINGS_ATTACHMENT_FILENAME in changes[0][1]["reason"]
        delivered = _events(conn, tid, kind="review_findings_delivered")
        assert len(delivered) == 1
        assert delivered[0][1]["attachment_id"] == attachment_id

        # The NEXT worker's context already carries the findings document.
        next_run = kb.claim_task(conn, tid, claimer="worker:2")
        assert next_run is not None
        context = kb.build_worker_context(conn, tid)
        assert kb.REVIEW_FINDINGS_ATTACHMENT_FILENAME in context
        assert "## Attachments" in context


def test_next_worker_context_inlines_the_latest_findings_document(kanban_home):
    """The next implementer must START with the findings, not with a path.

    The attachment list only names an absolute host path, which a remote
    worker cannot read at all — so a context that merely mentions
    ``review_findings.json`` carries no actionable problem/impact/fix. The
    LATEST document's field text has to be in the prompt itself; a
    superseded cycle's must not be.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        first = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(
                candidate_digest="digest-1",
                file="src/first_cycle.py",
                problem="the FIRST cycle's defect text",
                smallest_fix="the FIRST cycle's fix text",
            )],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert first["outcome"] == "handed_back"

        implementer = kb.claim_task(conn, tid, claimer="worker:2")
        assert implementer is not None
        assert kb.request_review(
            conn, tid, summary="round two", reviewer="reviewer",
            expected_run_id=implementer.current_run_id,
        )
        review2 = kb.claim_review_task(conn, tid, claimer="reviewer:2")
        assert review2 is not None
        latest = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(
                candidate_digest="digest-2",
                severity="blocking",
                file="src/handler.py",
                lines="120-134",
                problem="the retry loop never resets its backoff",
                impact="a flapping upstream pins the worker at the max delay",
                smallest_fix="reset the delay to the base value on success",
            )],
            candidate_digest="digest-2",
            expected_run_id=review2.current_run_id,
        )
        assert latest["outcome"] == "handed_back"

        next_run = kb.claim_task(conn, tid, claimer="worker:3")
        assert next_run is not None
        context = kb.build_worker_context(conn, tid)

        for field_text in (
            "blocking",
            "src/handler.py",
            "120-134",
            "the retry loop never resets its backoff",
            "a flapping upstream pins the worker at the max delay",
            "reset the delay to the base value on success",
        ):
            assert field_text in context, (
                f"{field_text!r} is not in the next worker's context — the "
                "handback carries no actionable finding"
            )

        # Only the LATEST delivery is inlined; the superseded cycle is not.
        assert "the FIRST cycle's defect text" not in context
        assert "the FIRST cycle's fix text" not in context

        # Still bounded — the whole prompt, not just this section.
        assert len(context) < 32 * 1024


# ---------------------------------------------------------------------------
# 3. A re-raised finding is audited as re-raised — and still delivered
# ---------------------------------------------------------------------------


def test_re_raised_finding_is_audited_and_still_reaches_the_implementer(
    kanban_home,
):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        first = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert first["outcome"] == "handed_back"
        resolved_fingerprint = first["fingerprints"][0]

        # Implementer "fixes" it and resubmits for review — new candidate.
        implementer = kb.claim_task(conn, tid, claimer="worker:2")
        assert implementer is not None
        assert kb.request_review(
            conn, tid, summary="fixed", reviewer="reviewer",
            expected_run_id=implementer.current_run_id,
        )
        review2 = kb.claim_review_task(conn, tid, claimer="reviewer:2")
        assert review2 is not None

        # Reviewer re-raises the SAME fingerprint against the NEW candidate,
        # alongside a genuinely NEW finding. The re-raise is AUDITED, but both
        # findings are handed back: history never suppresses a live finding.
        second = kb.submit_review_findings(
            conn, tid,
            findings=[
                _finding(candidate_digest="digest-2"),
                _finding(
                    candidate_digest="digest-2",
                    file="src/other.py",
                    problem="a distinct, never-before-seen defect",
                ),
            ],
            candidate_digest="digest-2",
            expected_run_id=review2.current_run_id,
        )
        assert second["outcome"] == "handed_back"
        assert second["re_raised_fingerprints"] == [resolved_fingerprint]
        assert resolved_fingerprint in second["fingerprints"]
        new_fingerprint = next(
            f for f in second["fingerprints"] if f != resolved_fingerprint
        )

        re_raised_events = _events(conn, tid, kind="review_finding_re_raised")
        assert len(re_raised_events) == 1
        assert re_raised_events[0][1]["fingerprint"] == resolved_fingerprint

        # Two documents total: the first handback's, and this second one —
        # which carries BOTH findings, the re-raised one included.
        attachments = kb.list_attachments(conn, tid)
        assert len(attachments) == 2
        second_doc = json.loads(kb.read_attachment_bytes(attachments[-1]))
        stored_fingerprints = {f["fingerprint"] for f in second_doc["findings"]}
        assert stored_fingerprints == {resolved_fingerprint, new_fingerprint}

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"


def test_a_persistent_defect_re_raised_on_a_new_candidate_is_never_dropped(
    kanban_home,
):
    """History can inform the audit trail; it can never veto the live review.

    A later ``review_requested`` plus a different candidate digest is not
    evidence that a defect is gone — the implementer only CLAIMED it was.
    When the current reviewer explicitly re-raises the same defect it must
    reach the implementer, and a task with an outstanding finding must
    never be approved.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        first = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert first["outcome"] == "handed_back"
        persistent = first["fingerprints"][0]

        # The implementer reworks the code — the snapshot really does change,
        # so the next review runs against a NEW candidate digest — and asks
        # for another look.
        implementer = kb.claim_task(conn, tid, claimer="worker:2")
        assert implementer is not None
        assert kb.request_review(
            conn, tid, summary="believed fixed", reviewer="reviewer",
            expected_run_id=implementer.current_run_id,
        )
        review2 = kb.claim_review_task(conn, tid, claimer="reviewer:2")
        assert review2 is not None

        # The defect is still there. The live reviewer re-raises it against
        # the new candidate, and it is the ONLY finding.
        second = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-2")],
            candidate_digest="digest-2",
            expected_run_id=review2.current_run_id,
        )
        assert second["outcome"] == "handed_back", (
            "a finding the current review raises was suppressed by history; "
            f"outcome={second['outcome']!r}"
        )
        assert persistent in second["fingerprints"]

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status != "done", (
            "a task with an outstanding, currently-raised finding was approved"
        )
        assert task.status == "ready"
        assert task.assignee == "worker"
        assert _events(conn, tid, kind="completed") == []

        # The handback really carried the re-raised finding to the
        # implementer, and it went through a second real request_changes.
        attachments = kb.list_attachments(conn, tid)
        assert len(attachments) == 2
        latest_doc = json.loads(kb.read_attachment_bytes(attachments[-1]))
        assert {f["fingerprint"] for f in latest_doc["findings"]} == {persistent}
        assert len(_events(conn, tid, kind="changes_requested")) == 2


# ---------------------------------------------------------------------------
# 4. Identical candidate + identical findings twice -> sticky owner block
# ---------------------------------------------------------------------------


def test_identical_findings_twice_blocks_for_owner_decision_without_rerun(
    kanban_home,
):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        first = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert first["outcome"] == "handed_back"

        # Implementer claims again but makes no real change; candidate
        # digest stays "digest-1" (nothing changed) and is sent back up.
        implementer = kb.claim_task(conn, tid, claimer="worker:2")
        assert implementer is not None
        assert kb.request_review(
            conn, tid, summary="no-op resubmit", reviewer="reviewer",
            expected_run_id=implementer.current_run_id,
        )
        review2 = kb.claim_review_task(conn, tid, claimer="reviewer:2")
        assert review2 is not None

        second = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=review2.current_run_id,
        )
        assert second["outcome"] == "owner_decision_blocked"

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        # Sticky: a worker-initiated block, not the circuit breaker.
        assert kb._has_sticky_block(conn, tid) is True

        repeated = _events(conn, tid, kind="review_findings_repeated")
        assert len(repeated) == 1
        assert repeated[0][1]["candidate_digest"] == "digest-1"
        assert repeated[0][1]["fingerprints"] == first["fingerprints"]

        # No third handback document was attached, and the implementer was
        # never re-run.
        assert len(kb.list_attachments(conn, tid)) == 1
        assert len(_events(conn, tid, kind="changes_requested")) == 1

        # recompute_ready must not silently auto-promote this.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_repeated_findings_stop_is_all_or_none_against_a_takeover(
    kanban_home, monkeypatch,
):
    """The owner stop on repeated findings is ONE decision: the sticky block
    (compare-and-swapped on the review run that raised them, which it also
    closes) and the ``review_findings_repeated`` event commit together or
    not at all.

    Before this fix the event committed in a transaction of its own and the
    block followed in a second one, so a reclaim and re-claim landing between
    them left a successor review run active and unblocked, with the stale
    event on its task. The barrier sits on the seam a takeover really uses:
    the next write transaction opened once the repeated event is durable.
    On the old code that is the block's own transaction, so the takeover
    lands in the gap; on the fixed code no transaction opens between the
    event and the block, so the takeover can only try after the stop and
    finds a blocked task with no claim to take.
    """
    with kb.connect() as setup:
        tid, review = _hand_off_to_review(setup)
        first = kb.submit_review_findings(
            setup, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert first["outcome"] == "handed_back"
        implementer = kb.claim_task(setup, tid, claimer="worker:2")
        assert implementer is not None
        assert kb.request_review(
            setup, tid, summary="no-op resubmit", reviewer="reviewer",
            expected_run_id=implementer.current_run_id,
        )
        review2 = kb.claim_review_task(setup, tid, claimer="reviewer:2")
        assert review2 is not None
    stale_run = int(review2.current_run_id)

    takeover: dict = {}
    probing = {"on": False}  # the probe and the takeover open transactions of their own
    real_write_txn = kb.write_txn

    def _repeated_event_is_durable() -> bool:
        with kb.connect() as reader:
            return reader.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? "
                "AND kind = 'review_findings_repeated' LIMIT 1",
                (tid,),
            ).fetchone() is not None

    def takeover_before_the_next_write_txn(conn_, **kwargs):
        if not takeover and not probing["on"]:
            probing["on"] = True
            try:
                if _repeated_event_is_durable():
                    with kb.connect() as other:
                        takeover["reclaimed"] = kb.reclaim_task(
                            other, tid, reason="operator takeover",
                            signal_fn=lambda *_: None,
                        )
                        successor = (
                            kb.claim_review_task(other, tid, claimer="reviewer:3")
                            if takeover["reclaimed"] else None
                        )
                        takeover["successor_run"] = (
                            int(successor.current_run_id)
                            if successor is not None else None
                        )
            finally:
                probing["on"] = False
        return real_write_txn(conn_, **kwargs)

    monkeypatch.setattr(kb, "write_txn", takeover_before_the_next_write_txn)

    with kb.connect() as conn:
        verdict = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-1")],
            candidate_digest="digest-1",
            expected_run_id=stale_run,
        )
        # The next write transaction anyone opens after the verdict is the
        # earliest moment a takeover can try on the fixed code.
        with kb.write_txn(conn):
            pass

    assert takeover, (
        "the barrier never fired, so no takeover was attempted and this "
        "test proved nothing"
    )
    assert verdict["outcome"] == "owner_decision_blocked", verdict
    assert takeover.get("successor_run") is None, (
        "a successor review run started around the owner stop"
    )

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert kb._has_sticky_block(conn, tid) is True
        repeated_runs = [
            r["run_id"] for r in conn.execute(
                "SELECT run_id FROM task_events WHERE task_id = ? "
                "AND kind = 'review_findings_repeated' ORDER BY id",
                (tid,),
            ).fetchall()
        ]
        assert repeated_runs == [stale_run]
        runs = {r.id: r for r in kb.list_runs(conn, tid)}
        assert runs[stale_run].status == "blocked"
        assert max(runs) == stale_run, "a later run exists on the task"


def _stored_blobs(task_id) -> list[str]:
    directory = kb.task_attachments_dir(task_id)
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir())


def _take_route_authority(conn, task_id) -> None:
    """Make the card receipt-owned by the owner with no approved route lock.

    That is real migrated owner work, and it is the state in which
    ``role_transition_route`` refuses EVERY role change — including the
    rework handback ``request_changes`` performs. Arranged through the real
    write transaction, the same way the suite arranges any other durable
    column.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 1 WHERE id = ?", (task_id,),
        )


def _release_route_authority(conn, task_id) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 0 WHERE id = ?", (task_id,),
        )


def test_refused_handback_leaves_no_delivery_record_and_no_orphan_attachment(
    kanban_home,
):
    """A handback that did not happen must leave no trace that it did.

    The route authority refuses the ``request_changes`` role change. If the
    attachment and the ``review_findings_delivered`` event have already been
    committed by then, the audit trail claims findings were delivered while
    no ``changes_requested`` transition ever occurred — and the reviewer's
    retry is read as a REPEAT of that phantom handback and sticky-blocks the
    task with zero successful handbacks.
    """
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        _take_route_authority(conn, tid)

        refused = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert refused["outcome"] == "error", (
            f"expected a structured kernel refusal, got {refused!r}"
        )
        assert refused.get("reason")

        # No delivery record of any kind, and nothing left behind.
        assert _events(conn, tid, kind="review_findings_delivered") == []
        assert _events(conn, tid, kind="changes_requested") == []
        assert kb.list_attachments(conn, tid) == []
        assert _stored_blobs(tid) == []
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == review.current_run_id

        # The owner approves the rework; the reviewer retries the IDENTICAL
        # findings. With zero successful handbacks so far this is a FIRST
        # handback, not a repeat.
        _release_route_authority(conn, tid)
        retry = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert retry["outcome"] == "handed_back", (
            "the retry was treated as a repeat of a handback that never "
            f"happened; outcome={retry['outcome']!r}"
        )
        assert len(kb.list_attachments(conn, tid)) == 1
        assert len(_events(conn, tid, kind="review_findings_delivered")) == 1
        assert len(_events(conn, tid, kind="changes_requested")) == 1
        assert _events(conn, tid, kind="review_findings_repeated") == []
        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.status == "ready"


# ---------------------------------------------------------------------------
# 5. Clean pass approves the task from the review lane
# ---------------------------------------------------------------------------


def test_clean_pass_approves_task_from_review_lane(kanban_home):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn)
        result = kb.submit_review_findings(
            conn, tid,
            findings=[],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert result["outcome"] == "passed"
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
        assert _events(conn, tid, kind="completed")


# ---------------------------------------------------------------------------
# 6. Reviewer contract is kernel-supplied, only on review-lane runs
# ---------------------------------------------------------------------------


def test_reviewer_contract_only_on_review_lane_runs(kanban_home):
    with kb.connect() as conn:
        # Normal implementer run: no reviewer contract.
        tid = kb.create_task(conn, title="implement", assignee="worker")
        kb.claim_task(conn, tid)
        normal_context = kb.build_worker_context(conn, tid)
        assert "## Reviewer contract" not in normal_context

        # Review-lane run: contract present, bounded, names the severities.
        _tid, review = _hand_off_to_review(conn, title="review lane task")
        review_context = kb.build_worker_context(conn, _tid)
        assert "## Reviewer contract" in review_context
        for sev in kb.REVIEW_FINDING_SEVERITIES:
            assert sev in review_context
        section = review_context.split("## Reviewer contract", 1)[1]
        section = section.split("##", 1)[0]
        assert len(section) < 800


# ---------------------------------------------------------------------------
# 6b. The mutation fence sits on the MUTATOR, not on one of its helpers
# ---------------------------------------------------------------------------


def test_submit_review_findings_is_the_bounded_mutation_not_its_helpers(
    kanban_home,
):
    """``submit_review_findings`` is the public mutator, so the entry
    deadline is opened for IT.

    Its read-only helpers must not open one of their own — a decorator that
    slid onto a helper would fence the wrong function and leave the real
    entry point waiting on the ordinary busy timeout. Asserts the actual
    runtime mechanism ``@bounded_mutation`` installs (a call into
    ``_mutation_deadline``), not source text; the wall-clock proof that the
    fence really binds is the exclusive-lock matrix in
    ``test_kanban_removal_fence_mutation_deadline.py``.
    """
    opened_for: list[str] = []
    real_deadline = kb._mutation_deadline

    @contextlib.contextmanager
    def _spy(conn, what):
        opened_for.append(what)
        with real_deadline(conn, what) as deadline:
            yield deadline

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="fence placement")

        kb._mutation_deadline = _spy
        try:
            findings = kb.build_review_findings_document(
                [_finding()], candidate_digest="digest-1",
            )["findings"]
            assert kb._review_findings_history(conn, tid) == []
            assert kb._previously_resolved_review_findings(
                conn, tid, findings, candidate_digest="digest-1",
            ) == []
            assert opened_for == [], (
                "a read-only review-findings helper opened a mutation "
                f"deadline of its own: {opened_for!r}"
            )

            result = kb.submit_review_findings(
                conn, tid,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )
        finally:
            kb._mutation_deadline = real_deadline

    assert result["outcome"] == "handed_back"
    assert opened_for and opened_for[0] == "submit_review_findings", (
        "submit_review_findings must open the entry deadline itself; "
        f"instead observed: {opened_for!r}"
    )


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------


def test_cli_review_findings_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="cli reviewed task")
        run_id = review.current_run_id

    doc_path = tmp_path / "findings.json"
    doc_path.write_text(
        json.dumps({
            "candidate_digest": "digest-1",
            "findings": [_finding()],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    output = kc.run_slash(f"review-findings {tid} --file {doc_path}")
    assert "handed back" in output

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert kb.list_attachments(conn, tid)


# ---------------------------------------------------------------------------
# 8. Dashboard endpoint
# ---------------------------------------------------------------------------


def test_dashboard_review_findings_endpoint(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import importlib.util
    import sys

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_review_findings_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = fastapi.FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="dashboard reviewed task")
        run_id = review.current_run_id

    resp = client.post(
        f"/api/plugins/kanban/tasks/{tid}/review-findings",
        json={
            "candidate_digest": "digest-1",
            "findings": [_finding()],
            "expected_run_id": run_id,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["outcome"] == "handed_back"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert kb.list_attachments(conn, tid)

    # A malformed document is a 400, not a 500.
    with kb.connect() as conn:
        tid2, review2 = _hand_off_to_review(conn, title="dashboard malformed")
        run_id2 = review2.current_run_id
    bad = client.post(
        f"/api/plugins/kanban/tasks/{tid2}/review-findings",
        json={
            "candidate_digest": "digest-1",
            "findings": [{"severity": "major"}],
            "expected_run_id": run_id2,
        },
    )
    assert bad.status_code == 400

    # A kernel-level route refusal is the documented 409, never a 500 — and
    # it leaves no delivery record behind for a retry to trip over.
    with kb.connect() as conn:
        tid3, review3 = _hand_off_to_review(conn, title="dashboard refused")
        run_id3 = review3.current_run_id
        _take_route_authority(conn, tid3)
    refused = client.post(
        f"/api/plugins/kanban/tasks/{tid3}/review-findings",
        json={
            "candidate_digest": "digest-1",
            "findings": [_finding()],
            "expected_run_id": run_id3,
        },
    )
    assert refused.status_code == 409, (
        f"expected the documented 409 kernel refusal, got "
        f"{refused.status_code}: {refused.text}"
    )
    assert refused.json()["detail"]
    with kb.connect() as conn:
        assert kb.list_attachments(conn, tid3) == []
        assert _events(conn, tid3, kind="review_findings_delivered") == []
        assert _events(conn, tid3, kind="changes_requested") == []
        assert kb.get_task(conn, tid3).status == "running"
