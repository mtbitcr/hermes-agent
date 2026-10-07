"""Shared setup for the platform half of the owner payload contract tests.

Each test saves one example of an owner payload, built by the real producer,
as ``owner_payloads/<family>/<record_kind>.json`` for the owner app's checks
to read. Tests assert relationships only: a saved example has the shape the
producer returns now, record by record, and every value of each closed
vocabulary appears in some example. Examples are rewritten only when
``--write-owner-payloads`` is given and are never compared byte for byte.

``CLOSED_VOCABULARIES`` is the one table of closed vocabularies: one row per
field the contract map names as closed, as ``(payload, path, values)``. The
payload is a family folder name, or ``family/record_kind`` for a row that
applies to one record kind only. The path is a tuple of steps: a plain step is
a key to descend into, and a step written ``key=value`` keeps the current
record only when its ``key`` equals ``value``, without descending; the last
step is the field. Lists met along the path are followed item by item, and a
field holding a list of scalars has each item checked. The values are a
frozenset, where an entry may be a compiled pattern for a sentence template
that must match the whole value. A discriminator a filter reads has its own
row; a field the table does not list is not vocabulary-checked.

``check_closed_vocabularies`` reads only that table. It tries each row's path
from every record anywhere in a payload, list items and strings holding a JSON
object or list included, fails on a value outside the row with the payload,
the concrete path and the value, and returns the values each row took. The
example fixture runs it on the live payload and on the saved example, which
must take the same values. ``assert_closed_vocabularies_covered`` runs it on
every saved example of a family and requires each value and template of each
of that family's rows to occur in some example. ``closed_values`` returns the
values a path takes, for a test's own relationship assertions.

The frozen clock and frozen ids only keep a rewritten example stable; no test
asserts their values. The clock reaches the ``datetime`` bindings of the
repository modules already imported when a test starts, so a test module
imports its producers at the top.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import json
import os
import re
import secrets
import sys
import time
import uuid
from pathlib import Path

import pytest

import hermes_time  # noqa: F401  (its ``datetime`` binding is the cron clock)

OWNER_PAYLOADS = Path(__file__).resolve().parent / "owner_payloads"
REPOSITORY = Path(__file__).resolve().parents[2]
WRITE_OPTION = "--write-owner-payloads"
FROZEN_TIME = 1_790_000_000.0
_REAL_DATETIME = datetime.datetime

# Value sets more than one row shares, each cited where its rows are.
_REVIEW_STATES = frozenset({"awaiting_review", "changes_requested", "approved", "none"})
_KERNEL_STOPPED_WORK = frozenset({"gave_up", "capability", "none"})
_KNOWN_OR_UNKNOWN = frozenset({"known", "unknown"})
_COST_STATES = frozenset({"estimated", "exact", "reported", "included", "unknown"})
_COST_SUMMARIES = frozenset({
    "Estimated model usage for this recorded route.",
    "Recorded model usage cost for this route.",
    "Provider-reported model usage cost for this route.",
    "Included in the connected provider plan.",
    "This record does not contain an authoritative cost.",
})
_EXTERNAL_EFFECT = frozenset({
    "This record does not confirm whether an external service changed.",
})
_RECEIPT_SUMMARIES = frozenset({
    "Work is still in progress.",
    "Work finished.",
    "Work finished and is awaiting review.",
    "Work is scheduled for later.",
    "Work stopped at the AI provider's usage limit and will start again by itself.",
    re.compile(
        r"Work stopped at the AI provider's usage limit and will start again "
        r"by itself after \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\."
    ),
    "Work stopped and needs attention.",
    "The final outcome could not be confirmed.",
})
_STREAM_EVENTS = frozenset({
    "response.created", "response.output_item.added", "response.output_text.delta",
    "response.output_item.done", "response.output_text.done", "response.completed",
    "response.failed",
})
_OUTPUT_ITEMS = frozenset({"message", "function_call", "function_call_output"})
_APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})
_DECISION_LIST = "object=hermes.owner_workspace.decision_list"

CLOSED_VOCABULARIES = (
    # owner_snapshot: the snapshot route, its steward and the run projection.
    # api_server.py:13086
    ("owner_snapshot", ("object",), frozenset({"hermes.owner_workspace.project_snapshot"})),
    # kanban_db.py:29419-29422
    ("owner_snapshot", ("review_state",), _REVIEW_STATES),
    # kanban_db.py:28848-28850; owner_workspace.py:2621
    ("owner_snapshot", ("stopped_work",), _KERNEL_STOPPED_WORK | {"provider_wait"}),
    # kanban_db.py:25137-25140
    ("owner_snapshot", ("retry_origin",),
     frozenset({"none", "automatic", "owner", "unattributed"})),
    # owner_workspace.py:2977-3000
    ("owner_snapshot", ("receipt", "outcome"),
     frozenset({"running", "completed", "attention", "waiting", "unknown"})),
    # owner_workspace.py:2977-3000, :2936-2941; :2622 and :2625 with the
    # provider wait; :3021 with the owner wait
    ("owner_snapshot", ("receipt", "summary"), _RECEIPT_SUMMARIES | {
        re.compile(
            r"Waiting for the AI provider\. Work resumes by itself after "
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\."
        ),
        "The AI provider declined this work as worded. It waits for you.",
        "The review asked for changes; the work went back for rework.",
    }),
    # owner_workspace.py:3006-3007
    ("owner_snapshot", ("receipt", "external_effect", "state"), frozenset({"unknown"})),
    ("owner_snapshot", ("receipt", "external_effect", "summary"), _EXTERNAL_EFFECT),
    # owner_workspace.py:2638, :2884
    ("owner_snapshot", ("receipt", "runtime", "state"), _KNOWN_OR_UNKNOWN),
    # owner_workspace.py:2817
    ("owner_snapshot", ("receipt", "runtime", "capability", "state"), _KNOWN_OR_UNKNOWN),
    # owner_workspace.py:2642, :2902
    ("owner_snapshot", ("receipt", "cost", "state"), _COST_STATES),
    # owner_workspace.py:2643, :2913-2916
    ("owner_snapshot", ("receipt", "cost", "summary"), _COST_SUMMARIES),
    # owner_workspace.py:3011
    ("owner_snapshot", ("receipt", "evidence", "state"), frozenset({"available"})),
    ("owner_snapshot", ("receipt", "evidence", "kind"), frozenset({"project_activity"})),
    # owner_workspace.py:3017
    ("owner_snapshot", ("receipt", "owner_retry", "state"), frozenset({"requested"})),
    # owner_workspace.py:5005-5033, :4699
    ("owner_snapshot", ("execution", "state"), frozenset({
        "working", "waiting_for_approval", "waiting_for_you", "paused",
        "needs_attention", "complete", "deleted",
    })),
    # owner_workspace.py:5006-5037, :4700-4702
    ("owner_snapshot", ("execution", "summary"), frozenset({
        "Raphael is waiting for an approved milestone before starting work.",
        "The approved work is complete.",
        "Raphael is finishing work already underway, then will stay paused.",
        "Raphael is paused and will not start new work.",
        "Raphael needs your answer before the plan can continue.",
        "Raphael found a problem and is preparing the safest next step.",
        "Raphael is coordinating the approved milestone.",
        "This Project was deleted. Its retained copy can be restored.",
        "This Project was deleted.",
    })),
    # owner_workspace.py:2628-2631, read at :3186-3207
    ("owner_snapshot", ("planning_context", "tasks", "status"), frozenset({
        "triage", "todo", "scheduled", "ready", "running", "blocked", "review",
        "done", "archived",
    })),
    # owner_workspace.py:4984 over :4901-4904, labels at :4374-4383
    ("owner_snapshot", ("active_work", "state"),
     frozenset({"Planned", "Scheduled", "Ready", "In progress", "Being checked"})),
    # owner_workspace.py:4960 over :4889-4895, labels at :4374-4383
    ("owner_snapshot", ("needs_attention", "state"), frozenset({"Needs attention", "Blocked"})),
    # owner_workspace.py:4970
    ("owner_snapshot", ("decisions_needed", "state"), frozenset({"Waiting for your answer"})),
    # api_server.py:13060, :13071, :13080, :13006
    ("owner_snapshot", ("error", "code"), frozenset({
        "project_not_found", "owner_workspace_unavailable", "owner_workspace_not_enabled",
    })),
    # api_server.py:5718
    ("owner_snapshot", ("error", "type"), frozenset({"invalid_request_error"})),

    # owner_conversation: /v1/responses and its conversation routes.
    # api_server.py:11291, :11272, :11326, :11371, :11489; :9632, :10850, :11532
    ("owner_conversation", ("object",), frozenset({
        "hermes.response.owner_history", "hermes.response.owner_sessions",
        "hermes.response.owner_proposal_consumption",
        "hermes.response.owner_recovery_acknowledgement",
        "hermes.response.owner_authority", "response",
    })),
    # api_server.py:2681, :1707-1717
    ("owner_conversation", ("kind",), frozenset({
        "question", "no_change", "proposal", "project_change_proposal", "failure",
    })),
    # api_server.py:1390-1401, used at :2737-2747
    ("owner_conversation", ("kind=failure", "message"), frozenset({
        "Raphael could not prepare a safe plan for this request. Nothing was "
        "changed. You can send it again.",
        "Raphael could not work on this request as it is worded. Nothing was "
        "changed. Rewording the request may help.",
        "The AI provider did not answer, so nothing was prepared and nothing was "
        "changed. You can send it again in a few minutes.",
    })),
    # api_server.py:3020-3162, answered at :11371-11375
    ("owner_conversation", ("outcome",), frozenset({"mismatch", "retired", "absent"})),
    # api_server.py:11396-11413
    ("owner_conversation", ("object=hermes.response.owner_authority", "action"), frozenset({
        "claim", "abandon", "attach", "complete", "release", "reconcile", "close",
    })),
    # api_server.py:9632, :10850, :11532, and the response each event carries
    ("owner_conversation", ("object=response", "status"),
     frozenset({"queued", "in_progress", "completed", "failed", "incomplete"})),
    # api_server.py:9737, :9760, :9769, :9850, :9996, :10078, :10107
    ("owner_conversation", ("event",), _STREAM_EVENTS),
    ("owner_conversation", ("data", "type"), _STREAM_EVENTS),
    # api_server.py:10363, :10414, :10654, :10208, :11286; None where no code
    # is given (:5718)
    ("owner_conversation", ("error", "code"), frozenset({
        "idempotency_conflict", "owner_conversation_stale", "owner_conversation_locked",
        "owner_response_incomplete", "owner_history_unavailable", None,
    })),
    # api_server.py:10207
    ("owner_conversation", ("error", "code=owner_response_incomplete", "message"),
     frozenset({"That request is still being prepared. Reload and try again."})),
    # api_server.py:5718 and its server_error callers
    ("owner_conversation", ("error", "type"),
     frozenset({"invalid_request_error", "server_error"})),
    # Response output items, stored and streamed (finding 280, not in the map).
    # api_server.py:9712, :9754, :9795, :9811, :9843, :9860, :9868, :10005,
    # :10049, :12208, :12222, :12234
    ("owner_conversation", ("object=response", "output", "type"), _OUTPUT_ITEMS),
    ("owner_conversation", ("item", "type"), _OUTPUT_ITEMS),
    # api_server.py:9755, :9796, :9844, :9863, :10006
    ("owner_conversation", ("item", "status"), frozenset({"in_progress", "completed"})),
    # api_server.py:9713, :9756, :10007, :10050, :12235
    ("owner_conversation", ("type=message", "role"), frozenset({"assistant"})),
    # api_server.py:9714, :10009, :10052, :12238
    ("owner_conversation", ("type=message", "content", "type"), frozenset({"output_text"})),
    # api_server.py:9857
    ("owner_conversation", ("type=function_call_output", "output", "type"),
     frozenset({"input_text"})),

    # runs: /v1/runs, its events, approval, steer and stop.
    # api_server.py:12835, :15545, :15609
    ("runs", ("object",),
     frozenset({"hermes.run", "hermes.run.approval_response", "hermes.run.steer"})),
    # the statuses _set_run_status (api_server.py:12830) is called with
    ("runs", ("object=hermes.run", "status"), frozenset({
        "queued", "running", "waiting_for_approval", "completed", "failed", "cancelled",
        "stopping",
    })),
    # api_server.py:12867, :12875, :12884, :12932, :14761, :15005, :15535,
    # :15604, :14895, :15200, :14944
    ("runs", ("event",), frozenset({
        "tool.started", "tool.completed", "reasoning.available", "subagent.start",
        "subagent.complete", "message.delta", "approval.request", "approval.responded",
        "run.steered", "run.completed", "run.failed", "run.cancelled",
    })),
    # api_server.py:15481, offered and answered alike
    ("runs", ("choices",), _APPROVAL_CHOICES),
    ("runs", ("choice",), _APPROVAL_CHOICES),
    # A finished child's status, relayed as given (api_server.py:12889-12909);
    # tools/delegate_tool_child_run.py:473, :477, :484, :707 and
    # tools/delegate_tool.py:353 (finding 280, not in the map)
    ("runs", ("event=subagent.complete", "status"), frozenset({
        "completed", "failed", "interrupted", "timeout", "error",
    })),
    # api_server.py:15378
    ("runs/run_started", ("body", "status"), frozenset({"started"})),
    # api_server.py:15641
    ("runs/stop", ("body", "status"), frozenset({"stopping"})),
    # api_server.py:5710, sentence at :1382-1385
    ("runs/status_failed_restart", ("error",), frozenset({
        "This stopped before it finished because Raphael restarted. Nothing was "
        "changed. You can ask for it again.",
    })),
    # api_server.py:15402
    ("runs", ("error", "code"), frozenset({"run_not_found"})),
    # api_server.py:5718
    ("runs", ("error", "type"), frozenset({"invalid_request_error"})),

    # owner_decisions (the next card): the decision list and decision routes.
    # api_server.py:13252, :13340
    ("owner_decisions", ("object",), frozenset({
        "hermes.owner_workspace.decision_list", "hermes.owner_workspace.decision",
    })),
    # owner_workspace.py:3906, :3914, :3919; api_server.py:13240
    ("owner_decisions", (_DECISION_LIST, "data", "kind"),
     frozenset({"capability", "owner_input", "run_approval"})),
    # owner_workspace.py:3905, :3913, :3918; api_server.py:13239
    ("owner_decisions", (_DECISION_LIST, "data", "authority"),
     frozenset({"recommendation", "task", "run"})),
    # api_server.py:13205-13213
    ("owner_decisions", (_DECISION_LIST, "data", "kind=run_approval", "title"), frozenset({
        "Approve the new Project", "Approve the first Project milestone",
        "Approve Project changes", "Approve a work-state change",
        "Approve an owner reply", "Approve trying stopped work again",
        "Approve the Project lifecycle change",
    })),
    # api_server.py:13244-13246
    ("owner_decisions", (_DECISION_LIST, "data", "kind=run_approval", "reason"),
     frozenset({"Raphael is waiting for your confirmation before changing this Project."})),
    # owner_workspace.py:3916, :3921
    ("owner_decisions", (_DECISION_LIST, "data", "kind=owner_input", "reason"),
     frozenset({"Raphael needs your answer before this work can continue."})),
    # api_server.py:13310, :13316, :13325, :13334, :13281; no code at :13291
    ("owner_decisions", ("error", "code"), frozenset({
        "decision_not_found", "invalid_argument", "owner_workspace_unavailable",
        "owner_workspace_not_enabled", None,
    })),

    # owner_projects_and_attachments (the next card).
    # api_server.py:12987
    ("owner_projects_and_attachments", ("object",),
     frozenset({"hermes.owner_workspace.project_list"})),
    # api_server.py:12956, :12981, :13105, :13124, :13134, :13143
    ("owner_projects_and_attachments", ("error", "code"), frozenset({
        "owner_workspace_not_enabled", "owner_workspace_unavailable", "attachment_not_found",
    })),

    # workspace_machine (the next card): the board and run receipt routes.
    # kanban_db.py:29419-29422, read at plugin_api.py:3964
    ("workspace_machine", ("review_state",), _REVIEW_STATES),
    # kanban_db.py:28848-28850, read at plugin_api.py:3968
    ("workspace_machine", ("stopped_work",), _KERNEL_STOPPED_WORK),
    # kanban_risk_tier.py RISK_TIERS; null where no tier is recorded
    ("workspace_machine", ("risk_tier",), frozenset({0, 1, 2, None})),
    # owner_workspace.py:2977-3000, built at plugin_api.py:4217 with no
    # provider or owner wait
    ("workspace_machine", ("receipt", "outcome"),
     frozenset({"running", "completed", "attention", "unknown"})),
    ("workspace_machine", ("receipt", "summary"), _RECEIPT_SUMMARIES),
    # owner_workspace.py:3006-3007
    ("workspace_machine", ("receipt", "external_effect", "state"), frozenset({"unknown"})),
    ("workspace_machine", ("receipt", "external_effect", "summary"), _EXTERNAL_EFFECT),
    # owner_workspace.py:2638, :2884, :2817
    ("workspace_machine", ("receipt", "runtime", "state"), _KNOWN_OR_UNKNOWN),
    ("workspace_machine", ("receipt", "runtime", "capability", "state"), _KNOWN_OR_UNKNOWN),
    # owner_workspace.py:2642-2643, :2902, :2913-2916
    ("workspace_machine", ("receipt", "cost", "state"), _COST_STATES),
    ("workspace_machine", ("receipt", "cost", "summary"), _COST_SUMMARIES),
    # owner_workspace.py:3011
    ("workspace_machine", ("receipt", "evidence", "state"), frozenset({"available"})),
    ("workspace_machine", ("receipt", "evidence", "kind"), frozenset({"project_activity"})),

    # automations (the next card): cron.py:271; api_server.py:11962, :11982
    ("automations/gateway_unavailable", ("detail", "code"),
     frozenset({"gateway_unavailable"})),

    # models (the next card): web_server.py:7071
    ("models/invalid_model_options", ("detail",), frozenset({"Invalid model options"})),
)
_OUTSIDE = object()


def _decoded(value, where: str):
    """A string holding a JSON object or list, decoded and marked in ``where``."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value), f"{where}<json>"
        except ValueError:
            pass
    return value, where


def _key(where: str, key) -> str:
    return f"{where}.{key}" if where else str(key)


def _records(value, where: str = ""):
    """``(where, record)`` for every dict anywhere in ``value``."""
    value, where = _decoded(value, where)
    if isinstance(value, dict):
        yield where, value
        for key, item in value.items():
            yield from _records(item, _key(where, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _records(item, f"{where}[{index}]")


def _follow(value, steps: tuple, where: str):
    """``(where, value)`` for each value the path ``steps`` reaches from ``value``."""
    if steps:
        value, where = _decoded(value, where)
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _follow(item, steps, f"{where}[{index}]")
    elif not steps:
        yield where, value
    elif isinstance(value, dict):
        key, is_filter, wanted = steps[0].partition("=")
        if is_filter:
            if value.get(key) == wanted:
                yield from _follow(value, steps[1:], where)
        elif key in value:
            yield from _follow(value[key], steps[1:], _key(where, key))


def _entry(value, values):
    """The entry of ``values`` that admits ``value``: the value itself, or the
    template it matches whole; ``_OUTSIDE`` when none does."""
    if not isinstance(value, (dict, list)) and value in values:
        return value
    if isinstance(value, str):
        for entry in values:
            if isinstance(entry, re.Pattern) and entry.fullmatch(value):
                return entry
    return _OUTSIDE


def check_closed_vocabularies(label: str, family: str, record_kind: str, payload) -> dict:
    """The values each row of the table that applies to ``family/record_kind``
    takes in ``payload``, keyed by ``(payload, path)``. A value outside its row
    fails, naming the payload (``label`` says live or saved), the concrete path
    and the value."""
    records = list(_records(payload))
    taken: dict = {}
    for scope, path, values in CLOSED_VOCABULARIES:
        if scope not in (family, f"{family}/{record_kind}"):
            continue
        seen = taken.setdefault((scope, path), set())
        for start, record in records:
            for where, value in _follow(record, path, start):
                entry = _entry(value, values)
                assert entry is not _OUTSIDE, (
                    f"{label} {family}/{record_kind}: {where} = {value!r} is outside "
                    f"the closed vocabulary {scope} {path}"
                )
                seen.add(entry)
    return taken


def assert_closed_vocabularies_covered(family: str) -> None:
    """Every saved example of ``family`` keeps to the table, and together they
    take each value and template of each of the family's rows."""
    taken: dict = {}
    for path in sorted((OWNER_PAYLOADS / family).glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        for row, values in check_closed_vocabularies("saved", family, path.stem, saved).items():
            taken.setdefault(row, set()).update(values)
    missing = {
        f"{scope} {path}": sorted(map(str, values - taken.get((scope, path), set())))
        for scope, path, values in CLOSED_VOCABULARIES
        if scope.split("/")[0] == family and values - taken.get((scope, path), set())
    }
    assert not missing, f"no saved {family} example takes {missing}"


def closed_values(payload, path: tuple) -> set:
    """Every value ``path`` reaches from any record of ``payload``."""
    return {
        value for start, record in _records(payload)
        for _where, value in _follow(record, path, start)
    }


def pytest_addoption(parser):
    parser.addoption(
        WRITE_OPTION,
        action="store_true",
        default=False,
        help="Rewrite tests/contracts/owner_payloads from the real producers.",
    )


def _shape(value):
    """The keys of a JSON value, record by record, without its values.

    Each record of a list keeps its own shape and is paired by position with
    the record in the same place among those of its event kind (records with
    no ``event`` pair among themselves), so a record that loses a key cannot
    hide behind another record's shape. The kind only pairs records; events of
    different kinds may interleave differently from run to run. A scalar in a
    list has no keys and is left out. A list's shape is marked as a list, so
    an empty list and an empty object differ. A string holding JSON, like a
    stored owner reply, is compared by its decoded shape.
    """
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        records: dict = {}
        for item in value:
            shape = _shape(item)
            if shape is None:
                continue
            kind = item.get("event") if isinstance(item, dict) else None
            records.setdefault(kind if isinstance(kind, str) else None, []).append(shape)
        return ("list", records)
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return _shape(json.loads(value))
        except ValueError:
            return None
    return None


@pytest.fixture
def owner_payload_example(request):
    """``example(family, record_kind, payload)`` returns the saved example.

    With the write option the live ``payload`` is written first. Either way
    the saved file is read back, and its shape, record by record, must equal
    the shape of the live payload the real producer just returned. Both keep
    to the closed vocabularies, and each row takes the same values in both.
    """
    write = request.config.getoption(WRITE_OPTION, default=False)

    def example(family: str, record_kind: str, payload):
        payload = json.loads(json.dumps(payload))
        live = check_closed_vocabularies("live", family, record_kind, payload)
        path = OWNER_PAYLOADS / family / f"{record_kind}.json"
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        assert path.is_file(), (
            f"no saved example {family}/{record_kind}.json; "
            f"write it with {WRITE_OPTION}"
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert _shape(saved) == _shape(payload), f"{family}/{record_kind}"
        assert check_closed_vocabularies("saved", family, record_kind, saved) == live, (
            f"{family}/{record_kind}: the saved example takes other closed values")
        return saved

    return example


class _FrozenClock(type):
    """Lets ``isinstance`` and ``issubclass`` treat the stand-in as ``datetime``."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, _REAL_DATETIME)

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, _REAL_DATETIME)


class _FrozenDatetime(_REAL_DATETIME, metaclass=_FrozenClock):
    """``datetime`` whose ``now`` reads ``time.time``; it builds real instances."""

    def __new__(cls, *args, **kwargs):
        return _REAL_DATETIME(*args, **kwargs)

    @classmethod
    def now(cls, tz=None):
        return _REAL_DATETIME.fromtimestamp(time.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.now(datetime.timezone.utc).replace(tzinfo=None)


def _repository_bindings(value):
    """``(module, name)`` for each global of an imported repository module bound to ``value``."""
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None) or {}
        path = str(namespace.get("__file__") or "")
        if namespace is globals() or "site-packages" in path or not path.startswith(
            f"{REPOSITORY}{os.sep}"
        ):
            continue
        for name, bound in list(namespace.items()):
            if bound is value:
                yield module, name


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Wall-clock reads return one instant; monotonic time still moves.

    ``time.time`` is frozen, and ``datetime.now`` wherever an imported
    repository module bound the ``datetime`` class, ``hermes_time.now``
    included. The ``datetime`` module itself is left alone.
    """
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)
    for module, name in list(_repository_bindings(_REAL_DATETIME)):
        monkeypatch.setattr(module, name, _FrozenDatetime)


@pytest.fixture(autouse=True)
def frozen_ids(monkeypatch):
    """``uuid4`` and ``token_hex`` hash a counter into every byte they return.

    Ids repeat from run to run, yet any prefix a producer keeps still differs
    between draws, and a ``uuid4`` keeps its version and variant bits.
    """
    counter = itertools.count(1)

    def _drawn(size: int) -> bytes:
        return hashlib.shake_256(str(next(counter)).encode()).digest(size)

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(bytes=_drawn(16), version=4))
    monkeypatch.setattr(
        secrets, "token_hex",
        lambda nbytes=None: _drawn(32 if nbytes is None else nbytes).hex(),
    )
