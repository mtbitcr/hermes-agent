"""Model update examples: each answer of the undo and the practice route.

Each answer of section 5.9 of the undo and practice plan is produced once by
the real handler, through the real dashboard app with a real Models token, over
the real home of tests/hermes_cli/test_web_router_model_updates.py, and saved
as the owner app reads it. Together the two examples take every value of both
``state`` vocabularies of tests/contracts/conftest.py.
"""

from __future__ import annotations

import pytest

from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.contracts import conftest as contract
from tests.hermes_cli.test_web_router_model_updates import (  # noqa: F401  (fixtures)
    _apply,
    _finish,
    _practice,
    _running_work,
    _undo,
    _write_route,
    dash,
    home,
    live_models,
    owner,
    roles,
    team,
)

FAMILY = "models"


def test_each_undo_and_practice_answer_is_saved_once_by_the_real_handler(team, owner_payload_example):
    undo = {"none": _undo(team).json()}
    _apply(team)
    running = _running_work(team)
    undo["busy"] = _undo(team).json()
    _finish(team, running)
    flip = model_policy.apply_model_update_record

    def moved_first(update):
        # Settings moves one role after the undo read the roles and revisions.
        _write_route("raphael-business", "openai-codex", "gpt-5.6-terra", "max")
        return flip(update)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_policy, "apply_model_update_record", moved_first)
        undo["not_applied"] = _undo(team).json()
    undo["undone"] = _undo(team).json()
    undo["already_undone"] = _undo(team).json()
    practice = {"started": _practice(team).json()}
    practice["already_started"] = _practice(team).json()

    for answers in (undo, practice):
        assert {key: answer["state"] for key, answer in answers.items()} == {key: key for key in answers}
    assert owner_payload_example(FAMILY, "model_update_undo", undo) == undo
    assert owner_payload_example(FAMILY, "model_update_practice", practice) == practice


def test_the_model_update_examples_take_every_state():
    """Read together, the saved Models examples take each value of each row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
    for record_kind, states in (
        ("model_update_undo", {"undone", "none", "already_undone", "busy", "not_applied"}),
        ("model_update_practice", {"started", "already_started"}),
    ):
        assert (f"{FAMILY}/{record_kind}", ("state",), frozenset(states)) in contract.CLOSED_VOCABULARIES
