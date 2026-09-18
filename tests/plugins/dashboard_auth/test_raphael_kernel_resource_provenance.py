"""Creation-time provenance for the machines the kernel creates remotely.

The intent is durable OUTSIDE the removable board BEFORE the machine
exists, the machine is stamped with the identity that intent minted, what
the kernel owns is read back THROUGH the SDK's metadata filter (never from
a local receipt), verification writes nothing and never upgrades, and a
crash between intent and machine resolves in BOTH directions.

The fake control plane answers ``list_sandbox_infos`` out of the registry
``FakeSandbox`` creates into, using REAL SDK response models, so a machine
appears in an enumeration only because a creation put it there.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch as sd
from tests.plugins.dashboard_auth.test_raphael_sandbox_dispatch import (  # noqa: F401
    FakeSandbox, _provision, _reset_fake, host,
)


class FakeManager:
    """The SDK's fleet read surface: it lists and closes, and can do no more."""

    filters: list = []
    extra: list = []
    closed = 0
    #: One info per page, so a caller that stops at the first page visibly
    #: loses machines the control plane really has.
    page_size = 1

    @classmethod
    def create(cls, connection_config=None):
        return cls()

    @classmethod
    def reset(cls) -> None:
        cls.filters, cls.extra, cls.closed = [], [], 0

    def list_sandbox_infos(self, filter):
        from opensandbox.models.sandboxes import PagedSandboxInfos, PaginationInfo

        type(self).filters.append(filter)
        wanted = dict(filter.metadata or {})
        known = [b.get_info() for b in FakeSandbox.live.values()] + list(type(self).extra)
        matching = [
            info for info in known
            if all((info.metadata or {}).get(k) == v for k, v in wanted.items())
        ]
        size, page = type(self).page_size, filter.page or 1
        pages = max(1, -(-len(matching) // size))
        return PagedSandboxInfos(
            sandbox_infos=matching[(page - 1) * size: page * size],
            pagination=PaginationInfo(
                page=page, page_size=size, total_items=len(matching),
                total_pages=pages, has_next_page=page < pages,
            ),
        )

    def close(self) -> None:
        type(self).closed += 1


@pytest.fixture
def sdk(monkeypatch):
    """Real SDK request/response models, fake machine class and fleet reader."""
    FakeManager.reset()
    fake = dataclasses.replace(sd._load_sdk(), sandbox=FakeSandbox, manager=FakeManager)
    monkeypatch.setattr(sd, "_load_sdk", lambda: fake)
    return fake


def _foreign_info(sandbox_id: str, metadata: dict):
    """A machine the control plane reports that this kernel did not record."""
    from opensandbox.models.sandboxes import SandboxInfo, SandboxStatus

    return SandboxInfo(
        id=sandbox_id, status=SandboxStatus(state="RUNNING"), entrypoint=[],
        created_at=datetime.now(timezone.utc), metadata=metadata,
    )


def _intents(**kwargs) -> list:
    return kb.kernel_resource_intents(kind=sd.KERNEL_SANDBOX_KIND, **kwargs)


def _register_state() -> tuple:
    """Every content byte of the register store, and the rows it serves.

    ``-shm`` is excluded and only that: SQLite's in-memory index of the
    write-ahead log, which any reader attaching to a WAL store touches.
    """
    rows = _intents()
    path = kb.register_db_path()
    return {
        f.name: f.read_bytes() for f in path.parent.glob(f"{path.name}*")
        if not f.name.endswith("-shm")
    }, rows


def test_kernel_owned_machines_are_enumerated_through_the_sdk_filter(host, sdk):
    """The fleet read is a filtered, paged SDK query — not a receipt read."""
    receipt = _provision()
    FakeManager.extra = [
        _foreign_info("sbx-other-owner", {"hermes_owner": "somebody-else"}),
        _foreign_info("sbx-same-owner", {
            sd.KERNEL_OWNER_KEY: sd.KERNEL_OWNER, "hermes_board": "default",
        }),
    ]
    # An independent reader has no local receipt at all: the board's own
    # store is gone and nothing on disk records what was created.
    kb.kanban_db_path(board="default").unlink()

    infos = sd.list_kernel_owned_sandboxes(metadata={"hermes_board": "default"}, sdk=sdk)

    ids = {str(info.id) for info in infos}
    assert receipt["sandbox_id"] in ids
    assert "sbx-same-owner" in ids, "a later page's machine was lost"
    assert "sbx-other-owner" not in ids, "a machine this kernel does not own"
    # Every query went through the SDK's own filter model, carrying the
    # ownership selector asked for…
    assert FakeManager.filters, "nothing was asked of the control plane"
    for sent in FakeManager.filters:
        assert isinstance(sent, sdk.sandbox_filter)
        assert sent.metadata[sd.KERNEL_OWNER_KEY] == sd.KERNEL_OWNER
        assert sent.metadata["hermes_board"] == "default"
        assert sent.page_size and sent.page_size > 0
    # …and paging was honoured rather than assumed to be one page.
    assert [s.page for s in FakeManager.filters] == list(
        range(1, len(FakeManager.filters) + 1)
    )
    assert len(FakeManager.filters) > 1
    assert FakeManager.closed == 1


def test_the_creation_intent_is_durable_outside_the_board_before_the_machine(
    host, sdk
):
    """Minted first, stamped into the machine, and it outlives the board."""
    observed: list = []
    # Inside create: the intent must ALREADY be durable, and the machine
    # must carry the identity that intent minted.
    FakeSandbox.behavior["after_create"] = lambda box: observed.append(
        (_intents(), dict(box.create_kwargs["metadata"]))
    )
    receipt = _provision()

    (rows, metadata), = observed
    row, = rows
    assert row["state"] == kb.KERNEL_INTENT_INTENDED
    assert row["subject_id"] is None, "an id was claimed before one existed"
    assert metadata[sd.KERNEL_INTENT_KEY] == row["record_id"]
    assert metadata[sd.KERNEL_OWNER_KEY] == sd.KERNEL_OWNER
    assert (metadata["hermes_task"], metadata["hermes_run"],
            metadata["hermes_board"]) == (host.task_id, str(host.run_id), "default")
    assert metadata["hermes_generation"]

    settled, = _intents()
    assert settled["subject_id"] == receipt["sandbox_id"]
    assert (settled["task_id"], settled["run_id"], settled["board_name"]) == (
        host.task_id, host.run_id, "default",
    )

    # The board's own store is destroyed; the intent and the exact resource
    # identity live elsewhere and are still readable.
    kb.kanban_db_path(board="default").unlink()
    survivor, = _intents()
    assert (survivor["subject_id"], survivor["record_id"]) == (
        receipt["sandbox_id"], settled["record_id"],
    )


def test_verification_writes_nothing_and_never_authorizes_destruction(host, sdk):
    """An unrecorded machine reads UNVERIFIED, for good, and grants nothing."""
    recorded = _provision()["sandbox_id"]
    FakeManager.extra = [_foreign_info("sbx-nobody-recorded", {
        sd.KERNEL_OWNER_KEY: sd.KERNEL_OWNER, "hermes_board": "default",
    })]
    before = _register_state()

    # An independent reader: enumerate through the SDK, verify each against
    # the creation-time ledger. Neither step may write.
    verdicts = {
        str(info.id): kb.verify_kernel_creation(
            str(info.id), kind=sd.KERNEL_SANDBOX_KIND,
        )
        for info in sd.list_kernel_owned_sandboxes(sdk=sdk)
    }

    assert verdicts[recorded].verdict == kb.RECEIPT_VERDICT_PASS
    unverified = verdicts["sbx-nobody-recorded"]
    assert unverified.verdict == kb.RECEIPT_VERDICT_UNVERIFIED
    # Not permission, and not mistakable for it.
    assert unverified.authorizes_destruction is False
    assert verdicts[recorded].authorizes_destruction is False
    with pytest.raises(TypeError):
        bool(unverified)
    # Observing it changes nothing: not the store, not the verdict.
    assert _register_state() == before
    assert kb.verify_kernel_creation("sbx-nobody-recorded").verdict == (
        kb.RECEIPT_VERDICT_UNVERIFIED
    )
    assert _register_state() == before
    # …and nothing was destroyed on the strength of a verdict.
    assert not FakeSandbox.live[recorded].killed


def test_a_crash_between_intent_and_creation_resolves_in_both_directions(host, sdk):
    """One direction leaves an orphan machine; the other leaves no machine."""
    # (a) The machine came into being, then this process died before it
    # could record the id — the machine is still out there.
    def _die_holding_the_new_machine(box):
        raise RuntimeError("the host died holding the new machine")

    FakeSandbox.behavior["after_create"] = _die_holding_the_new_machine
    assert "error" in _provision()
    orphan_intent, = _intents()
    assert orphan_intent["state"] == kb.KERNEL_INTENT_INTENDED

    resolved = sd.recover_creation_intents(sdk=sdk)

    orphan, = resolved["orphan_resources"]
    assert not resolved["never_created"]
    assert orphan["intent_id"] == orphan_intent["record_id"]
    row, = _intents(record_id=orphan_intent["record_id"])
    assert row["state"] == kb.KERNEL_INTENT_ORPHAN_RESOURCE
    assert row["subject_id"] == orphan["sandbox_id"]
    assert row["resolution"], "the resolution was not recorded"
    assert FakeSandbox.live[orphan["sandbox_id"]], "the machine was destroyed"

    # (b) The machine never came into being at all. Same crash window,
    # opposite direction, and it must not stay pending forever.
    FakeSandbox.behavior.pop("after_create")
    FakeSandbox.behavior["create_fails"] = True
    assert "error" in _provision()
    pending, = _intents(state=kb.KERNEL_INTENT_INTENDED)

    resolved = sd.recover_creation_intents(sdk=sdk)

    assert not resolved["orphan_resources"]
    never, = resolved["never_created"]
    assert never["intent_id"] == pending["record_id"]
    row, = _intents(record_id=pending["record_id"])
    assert row["state"] == kb.KERNEL_INTENT_NEVER_CREATED
    assert row["subject_id"] is None
    assert row["resolution"], "the resolution was not recorded"
    # Neither direction deleted history: both intents are still there,
    # each saying how it ended.
    assert {row["record_id"] for row in _intents()} == {
        orphan_intent["record_id"], pending["record_id"],
    }
