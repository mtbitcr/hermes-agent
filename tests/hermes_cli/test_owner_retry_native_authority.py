"""The owner retry is a NATIVE run authority, and an unbound call writes nothing.

``retry_task`` records the OWNER's own stated reason for trying stopped work
again — on the task, on the exact stopped attempt, and on the owner-facing run
receipt. So the one thing it must never accept is a caller that is not the
authenticated owner run: a reason chosen by a model, persisted under the
owner's name, is indistinguishable afterwards from a reason the owner gave.

The check therefore runs BEFORE the receipt read, the replay, the eligibility
read and every mutation, and this asserts that ordering the only way that
means anything — against the databases. Nothing is written: no receipt row, no
task event, no run, and the work stays exactly where the worker stopped it.
The confirmation is never even requested, because asking the owner to approve
an unauthorized retry is already the wrong question.

The authority these tests carry is minted the way the API server mints it —
SHA-256 over the closed retry payload, sorted keys, no whitespace, ASCII —
never through the kernel's own helper, so what is really proven is that both
sides canonicalize the same request into the same bytes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db

from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _bootstrap_board,
    _capability_stopped_task,
    _configured_provider,
    _temporarily_patch,
    _with_approver,
    ctx,
)


_REASON = "I have added the provider credentials to the account."


def _gateway_minted_authority(ctx, payload: dict) -> ow.OwnerContext:
    """One retry run's authority, minted exactly as the API server mints it."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return ow.OwnerContext(
        actor=ctx.actor,
        profile=ctx.profile,
        session=ctx.session,
        authority=ow.OwnerProposalAuthority(
            actor=ctx.actor,
            profile=ctx.profile,
            session=ctx.session,
            conversation="raphael-owner-" + "c" * 32,
            response_id="resp_" + "d" * 32,
            operation="owner_task_retry",
            idempotency_key=payload["idempotency_key"],
            payload_digest=hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        ),
    )


def _board_state(board: str, task_id: str) -> dict:
    """Everything a retry would change, read straight off the board."""
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task = kanban_db.get_task(conn, task_id)
        return {
            "status": task.status,
            "block_kind": task.block_kind,
            "current_run_id": task.current_run_id,
            "events": conn.execute(
                "SELECT COUNT(*) FROM task_events"
            ).fetchone()[0],
            "task_events": conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,),
            ).fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0],
            "tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
        }


def _receipt_count() -> int:
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        return pconn.execute(
            "SELECT COUNT(*) FROM owner_workspace_receipts"
        ).fetchone()[0]


@contextlib.contextmanager
def _recorded_confirmation(approved: bool):
    """Answer the owner's confirmation without a live approver thread.

    Returns the list of operations that actually reached the owner, so "the
    refusal happened before anyone was asked" is an assertion rather than an
    assumption — and an unauthorized call that slipped through cannot hang the
    test waiting for a decision nobody is going to make.
    """
    asked: list[str] = []

    def _record(ctx_, *, operation, digest, description):
        asked.append(operation)
        return {"approved": approved, "reason": None if approved else "timeout"}

    with _temporarily_patch(ow, "_confirm", _record):
        yield asked


def _stopped_work(ctx):
    setup = _bootstrap_board(ctx)
    task_id = _capability_stopped_task(
        setup["board"], setup["project_id"], "B03 — Connect the payment provider",
    )
    return setup, task_id


def test_retry_without_owner_run_authority_is_refused_and_writes_nothing(ctx):
    """No authority: refused up front, with zero durable effect anywhere."""
    setup, task_id = _stopped_work(ctx)
    args = {
        "idempotency_key": "retry-unauthorized",
        "project_id": setup["project_id"],
        "task_id": task_id,
        "reason": _REASON,
    }
    before = _board_state(setup["board"], task_id)
    receipts_before = _receipt_count()

    with _recorded_confirmation(approved=True) as asked:
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow.retry_task(ctx, **args)

    assert excinfo.value.code == "owner_run_authority_required"
    assert asked == [], "an unauthorized retry must not reach the owner at all"
    assert _receipt_count() == receipts_before
    assert _board_state(setup["board"], task_id) == before
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.committed_owner_retry_event(
            conn, task_id,
            actor=ctx.actor, profile=ctx.profile,
            idempotency_key=args["idempotency_key"],
        ) is None

    # And the same request, carried by the run authority the gateway mints
    # from that exact payload, still retries the work: what is refused above
    # is being unbound, not the operation.
    with _recorded_confirmation(approved=True) as asked:
        result = ow.retry_task(_gateway_minted_authority(ctx, args), **args)
    assert result["ok"] is True, result
    assert result["retry_reason"] == _REASON
    assert asked == ["owner_task_retry"]
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, task_id).status == "ready"
        retried = kanban_db.committed_owner_retry_event(
            conn, task_id,
            actor=ctx.actor, profile=ctx.profile,
            idempotency_key=args["idempotency_key"],
        )
        assert retried is not None
        assert (retried.payload or {})["reason"] == _REASON


def test_retry_authority_bound_to_a_different_reason_writes_nothing(ctx):
    """A substituted reason is a different request, not the owner's own.

    The authority the owner's run minted names one exact canonical payload. A
    caller that keeps the key and the task but swaps the REASON — the whole
    point of the attack, since the reason is what ends up on the board under
    the owner's name — is refused with the same zero-write outcome.
    """
    setup, task_id = _stopped_work(ctx)
    args = {
        "idempotency_key": "retry-swapped-reason",
        "project_id": setup["project_id"],
        "task_id": task_id,
        "reason": _REASON,
    }
    authorized = _gateway_minted_authority(ctx, args)
    before = _board_state(setup["board"], task_id)
    receipts_before = _receipt_count()

    with _recorded_confirmation(approved=True) as asked:
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow.retry_task(
                authorized,
                **{**args, "reason": "The model decided this deserves a retry."},
            )

    assert excinfo.value.code == "owner_run_authority_required"
    assert asked == []
    assert _receipt_count() == receipts_before
    assert _board_state(setup["board"], task_id) == before
