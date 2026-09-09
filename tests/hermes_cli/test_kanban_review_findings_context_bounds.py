"""The inlined review-findings section is bounded in BYTES and fully redacted.

``build_worker_context`` inlines the reviewer's latest handback document into
the next implementer's prompt, under a declared
``_CTX_MAX_REVIEW_FINDINGS_BYTES`` cap. Three things have to be true for that
cap to mean anything, and none of them was:

* the cap has to cover the WHOLE rendered section — headers, fence lines and
  the omission note included, not just the finding blocks;
* it has to count UTF-8 BYTES, because a character count under-measures
  multibyte finding text by up to 4x while the constant is named ``_BYTES``;
  and
* every reviewer-supplied byte in the section has to go through
  ``redact_review_value`` — the candidate digest was inserted raw, before the
  budget was even computed, at whatever length the reviewer chose.

This file drives all of it through the real ``build_worker_context``, plus the
kernel-injected reviewer contract that tells the reviewer what a clean verdict
actually is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

CAP = kb._CTX_MAX_REVIEW_FINDINGS_BYTES
SECTION_HEADING = "## Review findings to fix"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


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


def _findings_section(context: str) -> str:
    """The rendered section exactly as the worker receives it.

    Everything from its heading up to the next markdown H2, which is the
    slice the cap is a claim about.
    """
    assert SECTION_HEADING in context, "no findings section in the context"
    tail = context.split(SECTION_HEADING, 1)[1]
    end = tail.find("\n## ")
    body = tail if end == -1 else tail[: end + 1]
    return SECTION_HEADING + body


def _section_bytes(context: str) -> int:
    return len(_findings_section(context).encode("utf-8"))


def _next_worker_context(conn, tid: str) -> str:
    """The context the NEXT implementer is really spawned with."""
    assert kb.claim_task(conn, tid, claimer="worker:2") is not None
    return kb.build_worker_context(conn, tid)


# ---------------------------------------------------------------------------
# 1. The candidate digest is bounded at the validation boundary
# ---------------------------------------------------------------------------


def test_an_unbounded_candidate_digest_is_refused_and_never_reaches_a_worker(
    kanban_home,
):
    """A 100k-character "digest" is not a digest.

    It used to be accepted verbatim and inlined raw ahead of the budget
    computation, producing a ~100 KB section against a declared 8 KB cap.
    """
    oversize = "A" * 100_000
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="oversize digest")
        with pytest.raises(kb.ReviewFindingsError, match="candidate_digest"):
            kb.submit_review_findings(
                conn, tid,
                findings=[_finding(candidate_digest=oversize)],
                candidate_digest=oversize,
                expected_run_id=review.current_run_id,
            )

        # Refused at the boundary, so nothing was stored and no worker can
        # ever be handed it.
        assert kb.list_attachments(conn, tid) == []
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"


@pytest.mark.parametrize(
    "digest",
    [
        "a" * 129,                       # one char past the bound
        "has spaces in it",
        "line\nbreak",
        "back`tick",
        "```fence```",
        "-leading-punctuation",
    ],
)
def test_out_of_shape_candidate_digests_are_refused(kanban_home, digest):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title=f"shape {len(digest)}")
        with pytest.raises(kb.ReviewFindingsError, match="candidate_digest"):
            kb.submit_review_findings(
                conn, tid,
                findings=[],
                candidate_digest=digest,
                expected_run_id=review.current_run_id,
            )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running", "an invalid digest approved the task"


def test_a_full_length_hex_digest_is_accepted(kanban_home):
    """Regression guard: real digests must still work.

    A sha-256 hex digest, a git sha and a dotted build token are all
    legitimate identities for a reviewed snapshot.
    """
    for digest in (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "9f4d1a2",
        "build-2026.09.08+rev.31",
    ):
        with kb.connect() as conn:
            tid, review = _hand_off_to_review(conn, title=f"ok {digest[:8]}")
            result = kb.submit_review_findings(
                conn, tid,
                findings=[_finding(candidate_digest=digest)],
                candidate_digest=digest,
                expected_run_id=review.current_run_id,
            )
            assert result["outcome"] == "handed_back", result
            context = _next_worker_context(conn, tid)
            assert digest in _findings_section(context)


# ---------------------------------------------------------------------------
# 2. A stored document from before the bound existed cannot blow the cap
# ---------------------------------------------------------------------------


def _plant_delivered_document(conn, tid: str, run_id: int, document: dict) -> int:
    """Put a handback document on the card the way a handback does.

    Older builds accepted documents this validator now refuses, and those
    documents are still sitting on real boards. The renderer therefore has to
    hold the cap on its own rather than trusting the writer that stored them,
    so the fixture is the durable state, written through the native store and
    the native event append.
    """
    attachment_id = kb.store_attachment_bytes(
        conn, tid, kb.REVIEW_FINDINGS_ATTACHMENT_FILENAME,
        json.dumps(document, ensure_ascii=False).encode("utf-8"),
        content_type="application/json", uploaded_by="reviewer",
        expected_run_id=run_id,
    )
    with kb.write_txn(conn):
        kb._append_event(
            conn, tid, "review_findings_delivered",
            {
                "attachment_id": attachment_id,
                "candidate_digest": document.get("candidate_digest"),
                "fingerprints": [],
                "count": len(document.get("findings") or []),
            },
            run_id=run_id,
        )
    return attachment_id


def test_a_legacy_document_with_a_giant_digest_stays_within_the_cap(kanban_home):
    oversize = "A" * 100_000
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="legacy giant digest")
        _plant_delivered_document(conn, tid, review.current_run_id, {
            "schema_version": kb.REVIEW_FINDINGS_SCHEMA_VERSION,
            "candidate_digest": oversize,
            "findings": [_finding(candidate_digest=oversize)],
        })
        assert kb.request_changes(
            conn, tid, reason="see the attached findings",
            expected_run_id=review.current_run_id,
        )[0]

        context = kb.build_worker_context(conn, tid)
        assert oversize not in context, (
            "a 100,000-character candidate digest reached the worker prompt "
            "verbatim"
        )
        if SECTION_HEADING in context:
            assert _section_bytes(context) <= CAP


# ---------------------------------------------------------------------------
# 3. The cap counts UTF-8 bytes, over the whole section
# ---------------------------------------------------------------------------


def test_multibyte_findings_are_capped_in_bytes_not_characters(kanban_home):
    """Every finding field is CJK, so bytes are ~3x the character count.

    A character-counted budget lets roughly three times the declared cap
    through, which is exactly the prompt bloat the constant exists to
    prevent.
    """
    chunk = "設計上の欠陥" * 400  # 2,400 chars / 7,200 UTF-8 bytes
    findings = [
        _finding(
            file=f"src/日本語/モジュール{i}.py",
            lines="10-20",
            problem=chunk,
            impact=chunk,
            smallest_fix=chunk,
        )
        for i in range(6)
    ]
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="multibyte findings")
        result = kb.submit_review_findings(
            conn, tid, findings=findings, candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert result["outcome"] == "handed_back", result

        context = _next_worker_context(conn, tid)
        section = _findings_section(context)
        size = len(section.encode("utf-8"))
        assert size <= CAP, (
            f"the inlined section is {size} UTF-8 bytes against a declared "
            f"{CAP}-byte cap ({len(section)} characters)"
        )
        # Nothing was cut mid-character, and the omission is stated.
        assert "�" not in section
        assert "omitted for size" in section
        # The document itself is still complete on the card.
        stored = json.loads(kb.read_attachment_bytes(
            kb.list_attachments(conn, tid)[0]
        ))
        assert len(stored["findings"]) == len(findings)


def test_the_cap_covers_the_headers_fences_and_omission_note(kanban_home):
    """Many oversized findings — the section as a WHOLE must fit.

    The budget used to be spent on finding blocks alone, so both headers,
    both fence lines, the trailing blank line and the omission note were all
    emitted on top of a cap that was already exhausted.
    """
    # Sized so the finding blocks alone land just under the cap: the section
    # overhead the old budget ignored is then the whole overrun.
    findings = [
        _finding(
            file=f"src/module_{i}.py",
            problem="P" * 1300,
            impact="I" * 1300,
            smallest_fix="F" * 1300,
        )
        for i in range(20)
    ]
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="section overhead")
        result = kb.submit_review_findings(
            conn, tid, findings=findings, candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert result["outcome"] == "handed_back", result

        context = _next_worker_context(conn, tid)
        size = _section_bytes(context)
        assert size <= CAP, (
            f"the whole rendered section is {size} bytes against a declared "
            f"{CAP}-byte cap"
        )
        section = _findings_section(context)
        assert "omitted for size" in section
        # Still useful: the section that DID fit carries whole findings.
        assert "P" * 1300 in section


def test_a_small_handback_is_inlined_whole(kanban_home):
    """Regression guard: bounding the section must not start dropping
    findings that comfortably fit."""
    findings = [_finding(file=f"src/m{i}.py") for i in range(3)]
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="small handback")
        assert kb.submit_review_findings(
            conn, tid, findings=findings, candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )["outcome"] == "handed_back"

        section = _findings_section(_next_worker_context(conn, tid))
        assert "candidate: digest-1" in section
        for i in range(3):
            assert f"src/m{i}.py" in section
        assert "off-by-one in the loop bound" in section
        assert "omitted for size" not in section
        assert len(section.encode("utf-8")) <= CAP


# ---------------------------------------------------------------------------
# 4. The digest is redacted like every other reviewer-supplied value
# ---------------------------------------------------------------------------


def test_a_secret_looking_candidate_digest_is_redacted(kanban_home):
    """The digest was the one reviewer-supplied value skipping redaction.

    A reviewer (or a script) that pastes a credential where the snapshot
    identity belongs must not have it copied verbatim into the next worker's
    durable prompt.
    """
    secret = "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKK"
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="secret digest")
        assert kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest=secret)],
            candidate_digest=secret,
            expected_run_id=review.current_run_id,
        )["outcome"] == "handed_back"

        context = _next_worker_context(conn, tid)
        assert SECTION_HEADING in context
        assert secret not in context, (
            "a secret-looking candidate digest reached the worker context "
            "verbatim"
        )
        # Redacted, not merely dropped: the section still identifies the
        # snapshot enough to be useful.
        section = _findings_section(context)
        assert "candidate:" in section


# ---------------------------------------------------------------------------
# 5. The kernel-injected reviewer contract states the real rule
# ---------------------------------------------------------------------------


def test_reviewer_contract_states_that_only_an_empty_verdict_approves(
    kanban_home,
):
    """The contract must describe the kernel that exists.

    It used to promise "if nothing survives after already-resolved findings
    are dropped, the task is approved automatically" — a rule the kernel no
    longer implements. Every finding a live review reports is outstanding, so
    a reviewer following that sentence would expect an approval the board will
    never give, and would treat re-raising a previously-reported defect as
    pointless.
    """
    with kb.connect() as conn:
        tid, _review = _hand_off_to_review(conn, title="reviewer contract")
        context = kb.build_worker_context(conn, tid)

    assert "## Reviewer contract" in context
    section = context.split("## Reviewer contract", 1)[1].split("\n## ", 1)[0]

    # The rule that no longer exists must be gone.
    assert "approved automatically" not in section
    assert "already-resolved" not in section
    assert "nothing survives" not in section

    # The real contract: every reported finding stays outstanding, and only
    # an explicitly empty verdict approves.
    lowered = section.lower()
    assert "outstanding" in lowered
    assert "empty" in lowered
    # Still bounded, and still names the closed severity vocabulary.
    assert len(section) < 900
    for severity in kb.REVIEW_FINDING_SEVERITIES:
        assert severity in section


def test_the_reviewer_contract_matches_what_the_kernel_does(kanban_home):
    """The contract's two promises, exercised against the real kernel.

    A contract is only worth injecting if it is true, so assert the
    behaviour it describes rather than only its wording: an explicitly empty
    verdict approves, and a re-reported finding is still handed back rather
    than dropped as already-resolved.
    """
    with kb.connect() as conn:
        # Promise 1: an explicitly empty verdict approves.
        clean_tid, clean_review = _hand_off_to_review(conn, title="clean")
        assert kb.submit_review_findings(
            conn, clean_tid, findings=[], candidate_digest="digest-1",
            expected_run_id=clean_review.current_run_id,
        )["outcome"] == "passed"
        clean = kb.get_task(conn, clean_tid)
        assert clean is not None and clean.status == "done"

        # Promise 2: a finding the implementer claimed to have fixed, and the
        # live review re-reports, is still handed back — not dropped, and
        # certainly not an approval.
        tid, review = _hand_off_to_review(conn, title="re-raised")
        assert kb.submit_review_findings(
            conn, tid, findings=[_finding()], candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )["outcome"] == "handed_back"

        # The implementer reworks and asks for another look; the reviewer
        # finds the SAME defect on the new candidate.
        rework = kb.claim_task(conn, tid, claimer="worker:2")
        assert rework is not None
        assert kb.request_review(
            conn, tid, summary="fixed it", reviewer="reviewer",
            expected_run_id=rework.current_run_id,
        )
        second = kb.claim_review_task(conn, tid, claimer="reviewer:2")
        assert second is not None
        again = kb.submit_review_findings(
            conn, tid,
            findings=[_finding(candidate_digest="digest-2")],
            candidate_digest="digest-2",
            expected_run_id=second.current_run_id,
        )
        assert again["outcome"] == "handed_back", (
            "a re-raised finding was dropped as already-resolved and the "
            f"task was approved with the defect still in it: {again!r}"
        )
        assert again["re_raised_fingerprints"]
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "ready"
