"""The gateway and the kernel canonicalize one retry into the SAME bytes.

An owner retry run is admitted by a digest: the API server validates the run's
closed retry payload, mints ``payload_digest`` over its canonical form, and
hands the frozen payload to the native handler; the kernel then refuses to
retry anything that is not bound to that exact authority.

Two canonical forms would make that binding meaningless in either direction —
too loose (the kernel accepts what no run authorized) or too tight (a legitimate
retry can never be applied at all). So this drives the REAL gateway validator
(``APIServerAdapter._validated_owner_retry_authority``) and the REAL kernel and
asserts they agree on the same request, and only on that request.

Deliberately free of aiohttp: the full HTTP path for this run lives in
tests/gateway/test_api_server_runs.py, and this file has to run wherever the
kernel does.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from plugins.dashboard_auth.raphael_workspace import model_policy


_SESSION = "run_owner_retry_authority"
_REASON = "I have added the provider credentials to the account."


@pytest.fixture(autouse=True)
def _resolved_owner_task_routes():
    """Resolve owner task routes without a real profile config on disk.

    Only the on-disk provider selection is faked; the admitted matrix, the
    tier resolution and the durable lock minting are all production code.
    """
    original = model_policy.configured_assignment_for
    model_policy.configured_assignment_for = (
        lambda profile: model_policy.assignment_for(profile, "anthropic")
    )
    try:
        yield
    finally:
        model_policy.configured_assignment_for = original


def _owner_context(authority=None) -> ow.OwnerContext:
    return ow.OwnerContext(
        actor="default", profile="default", session=_SESSION, authority=authority,
    )


def _graph_payload(idempotency_key: str) -> dict:
    return {
        "idempotency_key": idempotency_key,
        "mode": "new",
        "project_name": "Workshop pilot",
        "project_description": "A private workshop pilot.",
        "project_id": None,
        "request_title": "Prepare the workshop",
        "specification": "Create one owner-visible workshop milestone.",
        "current_milestone": "Prepare the workshop",
        "owner_visible_result": "A reviewed workshop plan.",
        "root_assignee": "default",
        "tasks": [{
            "title": "Draft the workshop plan",
            "body": "Prepare the private workshop plan.",
            "assignee": "default",
            "responsibility": "B03",
            "execution_tier": "routine",
            "parents": [],
        }],
        "later_milestones": [],
    }


def _committed_project(idempotency_key: str) -> dict:
    """One real receipt-backed Project, committed through the owner kernel."""
    payload = _graph_payload(idempotency_key)
    context = _owner_context(
        ow.OwnerProposalAuthority(
            actor="default",
            profile="default",
            session=_SESSION,
            conversation="raphael-owner-" + "7" * 32,
            response_id="resp_" + "6" * 32,
            operation="owner_task_graph_commit",
            idempotency_key=idempotency_key,
            payload_digest=ow._digest(payload),
        )
    )
    with (
        patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[SimpleNamespace(name="default")],
        ),
        patch(
            "hermes_cli.owner_workspace._confirm",
            return_value={"approved": True, "reason": None},
        ),
    ):
        return ow.commit_task_graph(context, **payload)


def _capability_stopped_task(board: str, project_id: str) -> str:
    """Drive real kernel transitions until a worker hits a hard wall."""
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task_id = kanban_db.create_task(
            conn, title="B03 — Connect the payment provider",
            assignee="default", project_id=project_id,
        )
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.block_task(
            conn, task_id,
            reason="the provider account has no API credentials",
            kind="capability",
            expected_run_id=kanban_db.get_task(conn, task_id).current_run_id,
        )
    return task_id


def _board_state(board: str, task_id: str) -> dict:
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task = kanban_db.get_task(conn, task_id)
        return {
            "status": task.status,
            "block_kind": task.block_kind,
            "events": conn.execute(
                "SELECT COUNT(*) FROM task_events"
            ).fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0],
        }


def _receipt_count() -> int:
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        return pconn.execute(
            "SELECT COUNT(*) FROM owner_workspace_receipts"
        ).fetchone()[0]


def test_the_gateway_minted_retry_authority_is_the_only_one_the_kernel_accepts():
    """One canonical form on both sides — and nothing else gets through."""
    created = _committed_project("graph-retry-authority")
    task_id = _capability_stopped_task(created["board"], created["project_id"])
    idempotency_key = "owner-retry-authority-1"
    # As submitted, semantic whitespace and all: the gateway is what decides
    # the canonical form the digest is minted over.
    submitted = {
        "idempotency_key": idempotency_key,
        "project_id": created["project_id"],
        "task_id": task_id,
        "reason": f"  {_REASON}  ",
    }

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    minted = adapter._validated_owner_retry_authority(
        {
            "operation": "owner_task_retry",
            "idempotency_key": idempotency_key,
            "payload": dict(submitted),
        },
        {
            "profile": "default",
            "mode": "existing",
            "project_slug": created["project_slug"],
        },
        "default",
    )
    assert minted["operation"] == "owner_task_retry"

    authority = ow.OwnerProposalAuthority(
        actor="default",
        profile="default",
        session=_SESSION,
        conversation="raphael-owner-" + "8" * 32,
        response_id="resp_" + "9" * 32,
        operation=minted["operation"],
        idempotency_key=minted["idempotency_key"],
        payload_digest=minted["payload_digest"],
    )

    # A substituted reason is a different request: the model's own words can
    # never ride in on the authority the owner's run minted, and the refusal
    # leaves the board, the runs and the receipts untouched.
    before = _board_state(created["board"], task_id)
    receipts_before = _receipt_count()
    with patch(
        "hermes_cli.owner_workspace._confirm",
        return_value={"approved": True, "reason": None},
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow.retry_task(
                _owner_context(authority),
                **{**minted["payload"], "reason": "The model wants this rerun."},
            )
    assert excinfo.value.code == "owner_run_authority_required"
    assert _board_state(created["board"], task_id) == before
    assert _receipt_count() == receipts_before

    # The payload the gateway froze and forwarded IS accepted — the two sides
    # canonicalized the same request into the same bytes.
    with patch(
        "hermes_cli.owner_workspace._confirm",
        return_value={"approved": True, "reason": None},
    ):
        result = ow.retry_task(_owner_context(authority), **minted["payload"])
    assert result["ok"] is True, result
    assert result["retry_reason"] == _REASON
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        assert kanban_db.get_task(conn, task_id).status == "ready"
        retried = kanban_db.committed_owner_retry_event(
            conn, task_id,
            actor="default", profile="default",
            idempotency_key=idempotency_key,
        )
        assert retried is not None
        assert (retried.payload or {})["reason"] == _REASON
