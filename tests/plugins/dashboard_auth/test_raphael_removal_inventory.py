"""What a removal owns remotely, read with nothing local to consult.

The only thing that makes a removed board's remote machines attributable
is the owner tag the kernel stamps at CREATE time, so an independent
reader — another host, after the board's store is gone, with no receipt
file — must be able to list them through the SDK's own metadata filter.
And an inventory READS: the fake control plane records every call it is
asked to make, so "it mutated nothing" is observed, not promised.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch as sd
from tests.plugins.dashboard_auth.test_raphael_kernel_resource_provenance import (  # noqa: F401,E501
    FakeManager, _foreign_info, sdk,
)
from tests.plugins.dashboard_auth.test_raphael_sandbox_dispatch import (  # noqa: F401,E501
    FakeSandbox, _provision, _reset_fake, host,
)

_REMOVAL = "rm-0001-0002-0003"


def _mutations() -> dict:
    """Every mutating call the fake control plane has been asked to make."""
    live = FakeSandbox.live.values()
    return {
        "created": len(FakeSandbox.created),
        "killed": sorted(box.id for box in live if box.killed),
        "connects": len(FakeSandbox.connect_calls),
        "commands": sorted(cmd for box in live for cmd in box.commands_log),
        "uploads": sorted(str(item) for box in live for item in box.uploads),
        "vault": sorted(str(call) for box in live for call in box.vault_calls),
    }


def test_a_removals_inventory_is_read_through_the_sdk_with_no_local_receipt(
    host, sdk
):
    """The owner tag plus the board is the whole query — and it mutates nothing."""
    receipt = _provision()
    FakeManager.extra = [
        _foreign_info("sbx-other-owner", {
            sd.KERNEL_OWNER_KEY: "somebody-else", "hermes_board": "default"}),
        _foreign_info("sbx-other-board", {
            sd.KERNEL_OWNER_KEY: sd.KERNEL_OWNER, "hermes_board": "elsewhere"}),
        _foreign_info("sbx-same-removal", {
            sd.KERNEL_OWNER_KEY: sd.KERNEL_OWNER, "hermes_board": "default"}),
    ]
    # An independent reader: the board's own store is gone, so nothing on
    # disk records what this removal owns.
    kb.kanban_db_path(board="default").unlink()
    before = _mutations()

    inventory = sd.list_removal_owned_sandboxes(
        board="default", removal_id=_REMOVAL, sdk=sdk)

    assert (inventory["removal_id"], inventory["board"], inventory["owner"]) == (
        _REMOVAL, "default", sd.KERNEL_OWNER)
    ids = {entry["sandbox_id"] for entry in inventory["resources"]}
    assert receipt["sandbox_id"] in ids
    assert "sbx-same-removal" in ids, "a later page's machine was lost"
    assert "sbx-other-owner" not in ids, "a machine this kernel does not own"
    assert "sbx-other-board" not in ids, "a machine on another board"
    # The machine this kernel created is attributable to its creation-time
    # intent, read out of its own metadata rather than a receipt.
    mine, = [e for e in inventory["resources"]
             if e["sandbox_id"] == receipt["sandbox_id"]]
    assert mine["intent_id"] and mine["kind"] == sd.KERNEL_SANDBOX_KIND
    assert mine["task_id"] == host.task_id

    # Every question went through the SDK's own filter model, carrying the
    # owner tag and the board…
    assert FakeManager.filters, "nothing was asked of the control plane"
    for sent in FakeManager.filters:
        assert isinstance(sent, sdk.sandbox_filter)
        assert sent.metadata == inventory["selector"]
        assert sent.metadata[sd.KERNEL_OWNER_KEY] == sd.KERNEL_OWNER
        assert sent.page_size and sent.page_size > 0
    # …and paging was honoured rather than assumed to be one page.
    assert [sent.page for sent in FakeManager.filters] == list(
        range(1, len(FakeManager.filters) + 1))
    assert len(FakeManager.filters) > 1
    assert FakeManager.closed == 1, "the fleet reader was not closed"
    # Nothing created, killed, connected to, run on, or uploaded to.
    assert _mutations() == before


def test_the_inventory_needs_the_removal_it_is_for(host, sdk):
    """An inventory with nothing to attribute it to is refused, not faked."""
    with pytest.raises(ValueError):
        sd.list_removal_owned_sandboxes(board="default", removal_id="")
    assert FakeManager.filters == [], "the control plane was asked anyway"


def test_an_empty_inventory_is_still_a_read(host, sdk):
    """A board that owns nothing answers empty — and still changes nothing."""
    before = _mutations()

    inventory = sd.list_removal_owned_sandboxes(
        board="no-such-board", removal_id=_REMOVAL, sdk=sdk)

    assert inventory["resources"] == [] and inventory["read_only"] is True
    assert _mutations() == before
    assert FakeManager.closed == 1
