"""A suggestion whose label cannot be shown gets a plain title by kind (finding 83)."""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli import kanban_db, owner_workspace as ow
from hermes_cli.owner_text import check_owner_text
from tools import approval


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_titles")


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _suggestion_title(ctx, *, kind: str, label: str) -> str:
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    setup = ow.bootstrap(
        ctx, idempotency_key=f"titles-{time.monotonic()}", name="Suggestion Titles",
    )
    approver.join()
    with kanban_db.connect(board=setup["board"]) as conn:
        kanban_db.create_recommendation(
            conn,
            project_id=setup["project_id"],
            target_profile="raphael-planner",
            recommendation_kind=kind,
            recommendation_subject_id="workshop-research",
            recommendation_label=label,
            recommendation_rationale="The current milestone needs public-source research.",
            recommendation_evidence={
                "schema_version": 1,
                "need": "The workshop outline needs current public evidence.",
                "expected_benefit": "Keep the owner-facing advice current.",
                "requested_scope": {
                    flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS
                },
                "risks": "Low",
                "cost": "No added cost",
                "rollback": "Remove the staged configuration.",
            },
            provenance_authority="project-steward",
        )
    [suggestion] = [
        item for item in ow.list_owner_decisions(ctx)["data"]
        if item["kind"] == "capability"
    ]
    return suggestion["title"]


def test_a_suggestion_naming_a_team_role_gets_its_kind_title(ctx):
    assert _suggestion_title(
        ctx, kind="skill", label="Give raphael-planner web research",
    ) == "A suggested new skill"


def test_a_model_setting_suggestion_gets_its_kind_title(ctx):
    assert _suggestion_title(
        ctx, kind="provider_model_policy", label="Switch drafting to Claude Opus",
    ) == "A suggested model setting change"


def test_a_plain_suggestion_label_is_kept(ctx):
    assert _suggestion_title(
        ctx, kind="skill", label="Add workshop research support",
    ) == "Add workshop research support"


_POLICY_PLACEHOLDER = check_owner_text(None, "title")


@pytest.mark.parametrize(
    "label",
    [
        f"\x1b]0;Claude Opus\x07{_POLICY_PLACEHOLDER}",
        f"\x1b]0;raphael-planner\x07{_POLICY_PLACEHOLDER}",
    ],
    ids=["model-name", "team-role"],
)
def test_a_rejected_label_cleaning_to_the_placeholder_gets_its_kind_title(
    ctx, label,
):
    assert _suggestion_title(
        ctx, kind="provider_model_policy", label=label,
    ) == "A suggested model setting change"


@pytest.mark.parametrize(
    "label",
    [
        ow._UNTITLED_WORK_ITEM,
        f"  {ow._UNTITLED_WORK_ITEM}  ",
        "Untitled   work\titem",
        ow._UNTITLED_WORK_ITEM.upper(),
        ow._UNTITLED_WORK_ITEM.lower(),
        _POLICY_PLACEHOLDER,
        f"  {_POLICY_PLACEHOLDER}  ",
        "  ".join(_POLICY_PLACEHOLDER.split()),
        _POLICY_PLACEHOLDER.upper(),
        _POLICY_PLACEHOLDER.lower(),
    ],
    ids=[
        "untitled", "untitled-padded", "untitled-spaced", "untitled-upper",
        "untitled-lower", "placeholder", "placeholder-padded",
        "placeholder-spaced", "placeholder-upper", "placeholder-lower",
    ],
)
def test_a_reserved_placeholder_label_gets_its_kind_title(ctx, label):
    assert _suggestion_title(
        ctx, kind="skill", label=label,
    ) == "A suggested new skill"


def test_every_kind_title_passes_the_shared_checker():
    titles = getattr(ow, "_SUGGESTION_KIND_TITLES", {})
    fallback = getattr(ow, "_SUGGESTION_FALLBACK_TITLE", None)
    assert set(titles) == kanban_db.VALID_RECOMMENDATION_KINDS
    placeholders = {ow._UNTITLED_WORK_ITEM, check_owner_text(None, "title")}
    for title in (*titles.values(), fallback):
        assert check_owner_text(title, "title") == title
        assert ow.owner_title(title) == title
        assert title not in placeholders
