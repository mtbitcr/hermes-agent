"""Owner Workspace payload text through the one owner text policy.

The removal state's last error and the retry refusal sentence reach the owner
through :func:`hermes_cli.owner_text.check_owner_text`. The removal state is a
read: the stored removal operation keeps exactly what it recorded.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import owner_workspace as ow, projects_db
from hermes_cli.owner_text import check_owner_text
from hermes_cli.owner_workspace import _REMOVAL_SAFE_ERRORS, _project_removal_state

# Each kind's fixed sentence, read from the one policy: text that is not a
# string always gets it.
_STORED_FAILURE = check_owner_text(None, "stored_failure")
_RETRY_REASON = check_owner_text(None, "retry_reason")

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "driver.py", line 3, in advance\n'
    "OSError: [Errno 28] No space left on device"
)


def _failed_removal(project_id, last_error):
    """One stopped removal whose driver recorded ``last_error``."""
    with projects_db.connect_closing() as conn:
        assert projects_db.record_removal_operation(
            conn, project_id=project_id, idempotency_key="rk",
            action="start", phase="failed", mode="reversible",
        )
        projects_db.update_removal_operation(
            conn, project_id, "rk", last_error=last_error,
        )


def _read_removal_state(project_id):
    with projects_db.connect_closing() as conn:
        before = dict(projects_db.get_removal_operation(conn, project_id, "rk"))
        state = _project_removal_state(conn, project_id, None)
        # A read: the stored operation keeps exactly what it recorded.
        assert dict(projects_db.get_removal_operation(conn, project_id, "rk")) == before
    return state, before


def test_removal_last_error_is_checked():
    _failed_removal("pr-traceback", _TRACEBACK)

    state, stored = _read_removal_state("pr-traceback")

    assert state["last_error"] == _STORED_FAILURE
    assert state["phase"] == "failed"
    assert stored["last_error"] == _TRACEBACK


@pytest.mark.parametrize("last_error", [
    _REMOVAL_SAFE_ERRORS["driver_failed"],
    _REMOVAL_SAFE_ERRORS["permanent_not_confirmed"],
    None,
])
def test_a_plain_removal_last_error_passes_unchanged(last_error):
    _failed_removal("pr-plain", last_error)

    state, _stored = _read_removal_state("pr-plain")

    assert state["last_error"] == last_error


def test_a_retry_refusal_reason_is_checked():
    task = SimpleNamespace(status="blocked", block_kind="needs_input")

    refused = ow._retry_refused(
        "t_0000abcd", task, "This work waits on t_0000abcd, so nothing was retried.",
    )

    assert refused == {
        "ok": False,
        "error": "not_retryable",
        "task_id": "t_0000abcd",
        "current_status": "blocked",
        "reason": _RETRY_REASON,
    }


@pytest.mark.parametrize("task", [
    SimpleNamespace(status="blocked", block_kind="needs_input"),
    SimpleNamespace(status="review", block_kind=None),
    SimpleNamespace(status="archived", block_kind=None),
])
def test_a_plain_retry_refusal_reason_passes_unchanged(task):
    reason = ow._retry_refusal_reason(task)

    refused = ow._retry_refused("t_0000abcd", task, reason)

    assert refused["reason"] == reason
    assert "can be tried again" in refused["reason"]
