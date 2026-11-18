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
    _SEAMS,
    _SWEEP,
    _answer_is_true,
    _apply,
    _configs,
    _counted_undo,
    _fault_free_count,
    _finish,
    _first_then_second,
    _practice,
    _record_bytes,
    _restore_home,
    _restore_route_latch,
    _running_work,
    _seam_faults,
    _stores,
    _sweep_roles,
    _sweep_start,
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


# ---------------------------------------------------------------------------
# Moved here unchanged from tests/hermes_cli/test_web_router_model_updates.py,
# so that each of the two files keeps the owner's time limit: the error at
# each seam of owner rule 1 and the error sweep of point 8 of owner rule 2
# (their real-exit twins stay there). They run on the real clock and real
# ids, as they did there.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seam", _SEAMS, ids=[seam[0] for seam in _SEAMS])
def test_an_error_at_each_seam_then_one_more_undo_ends_whole(team, monkeypatch, seam, real_clock_and_ids):
    suggestion_id = _apply(team)
    before = (_record_bytes(), _configs())
    earlier = _stores(team, suggestion_id)
    with monkeypatch.context() as patch:
        _seam_faults(patch, seam, "error")
        first = _undo(team)
    _answer_is_true(first, team, suggestion_id, before, earlier)

    earlier = _stores(team, suggestion_id)
    resp = _undo(team)

    assert resp.status_code == 200, resp.text
    _answer_is_true(resp, team, suggestion_id, before, earlier)


@pytest.mark.parametrize("scenario", sorted(_SWEEP))
def test_rule2_an_error_at_each_counted_call_then_one_more_undo_ends_whole(team, tmp_path, scenario, real_clock_and_ids):
    roles = _sweep_roles(scenario)
    suggestion_id, before = _sweep_start(team, scenario, tmp_path / "saved-home")
    n = len(_fault_free_count(team, tmp_path / "saved-home", suggestion_id, before, roles).calls)
    print(f"RULE2_SWEEP error {scenario}: N={n}")
    for k in range(1, n + 1):
        _restore_home(tmp_path / "saved-home")
        earlier = _stores(team, suggestion_id)
        first, calls = _counted_undo(team, fault=k)
        assert len(calls.calls) >= k, f"call {k} was not reached"
        _first_then_second(team, suggestion_id, before, earlier, first,
                           f"call {k} {calls.calls[k - 1]} {calls.paths[k - 1]}", roles)
