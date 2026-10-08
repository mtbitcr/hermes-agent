"""The owner's release decision (release card 8), served by the API server.

The owner reads the waiting batch and the current or last release in plain
words, and accepts, puts off or starts again without the operator. There is no
reject. The release record is the authority. It is read only through the
``release_ledger`` readers, on a read-only connection, so a read changes no
byte of the live home. Each answer reads first, then takes one reservation,
then takes one action. Accept and put off go through the decide step, bound to
the version and digest the owner was shown: a stale answer records nothing,
and a repeated one records no second answer. A recorded accept, and start
again, run ``hermes release start BATCH`` once as a child process of the
gateway, never the start function inside this process, because the start
removes variables from its own environment; the start's own checks refuse
while a release unit is at work. Once the child returns, one more read of the
release record and the release units is the only source of the answer and of
the decision returned with it; the child's exit code and every earlier read
decide nothing. Each title is shown as ``owner_title`` shows it, the one title
boundary of every owner surface; the version and digest keep their own fields.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import sys
from typing import Any, Optional

from aiohttp import web

from gateway.platforms.api_server import _openai_error, _owner_workspace_toolset_enabled
from hermes_cli import release_ledger, release_unit
from hermes_cli.kanban_risk_tier import RISK_NOT_RECORDED

logger = logging.getLogger(__name__)

# What the decide step records as the answer's origin: this page, never a secret.
_PAGE_REF = "owner-workspace:release"
_DECISIONS = {"accept": "accepted", "defer": "deferred"}
_STARTABLE = ("accepted", "releasing", "failed")
_BATCH_ID = re.compile(r"[1-9][0-9]{0,17}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_WAITING = {
    "open": "Waiting for your decision.",
    "deferred": "Put off by you. New changes still join it.",
}
_TIERS = {
    0: "Tier 0, low risk: the release goes forward once.",
    1: "Tier 1, medium risk: the release goes forward once.",
    2: "Tier 2, high risk: the release goes forward, rehearses the way back, then goes forward again.",
}
_STATES = {
    "accepted": "Accepted. Its release has not begun.",
    "releasing": "Releasing.",
    "failed": "Stopped midway.",
    "released": "Released.",
    "folded": "Ended without being released.",
}
_UNIT = {
    True: "Its release unit is running.",
    False: "Its release unit is not running.",
    None: "Whether its release unit is running could not be read.",
}
_OUTCOMES = {
    None: "No outcome yet.",
    "released": "It was released.",
    "failed": "It failed midway and is kept for recovery.",
    "restored": "It was rolled back. Its changes went back to the waiting batch.",
    "refused": "It was refused before anything changed. Its changes went back to the waiting batch.",
}
# What an answer says. After the start child the final read alone picks it, alike on the accept and
# the start route; a release that ended answers with its outcome sentence instead.
_ANSWERS = {
    "put_off": "Put off. The batch keeps waiting, and new changes join it.",
    "releasing": "The release is running.",
    "could_not_start": "The release could not start.",
    "another_release": "Another release is at work.",
}
_CHANGED = "The waiting batch changed after it was shown. Nothing was recorded; here is the new list."
# The batches an answer is under way for in this gateway: the one reservation before its action.
_RESERVED: set[int] = set()


def route(adapter: Any, action: str):
    """The handler of one release route, behind the owner authentication of the decision routes."""

    async def handle(request: web.Request) -> web.Response:
        auth_err = adapter._check_auth(request)
        if auth_err:
            return auth_err
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return _refusal(
                    404, "Owner workspace is not enabled for this profile", "owner_workspace_not_enabled"
                )
            if action == "view":
                return web.json_response(_view(*await asyncio.to_thread(_read)))
            body: dict[str, Any] = {}
            if action != "start":
                body, err = await adapter._read_json_body(request)
                if err:
                    return err
            return await _answer(request, action, body)
        except Exception:
            logger.exception("[api_server] owner release %s failed", action)
            return _refusal(
                503,
                "The release decision is unavailable right now. Please try again shortly.",
                "owner_workspace_unavailable",
            )

    return handle


async def _answer(request: web.Request, action: str, body: dict[str, Any]) -> web.Response:
    """Accept, put off or start again: every read first, then one reservation, then one action."""
    batch_id, version, digest = _batch_id(request), body.get("version"), body.get("digest")
    shown = set(body) == {"version", "digest"} and type(version) is int and version >= 0 and type(digest) is str
    if batch_id is None or (action != "start" and not (shown and _DIGEST.fullmatch(digest))):
        return _refusal(400, "Invalid release decision request", "invalid_argument")
    batches, busy = await asyncio.to_thread(_read)
    refused = _refused(action, batch_id, version, digest, batches, busy)
    if refused:
        return _refusal(*refused, _view(batches, busy))
    _RESERVED.add(batch_id)
    try:
        if action != "start":
            await asyncio.to_thread(_decide, batch_id, _DECISIONS[action], version, digest)
        # Accept and start again run the start once; its own checks refuse while a release unit is at work.
        if action != "defer":
            await _run_start(batch_id)
    except PermissionError:  # the decide step refuses a kanban worker
        return _refusal(403, "Only the owner decides a release. Nothing was recorded.", "release_decision_refused")
    except ValueError:  # the batch changed between the reads and the decide step
        return _refusal(409, _CHANGED, "release_changed", _view(*await asyncio.to_thread(_read)))
    finally:
        _RESERVED.discard(batch_id)
    # One readback, for the answer and the decision it returns, so the two cannot disagree.
    batches, busy = await asyncio.to_thread(_read)
    if action == "defer":
        answer, message = "put_off", _ANSWERS["put_off"]
    else:  # accept and start again alike, whatever the start child exited with
        answer, message = _after_start(batch_id, batches, busy)
    return web.json_response({
        "object": "hermes.owner_workspace.release_answer",
        "answer": answer,
        "message": message,
        "decision": _view(batches, busy),
    })


def _after_start(
    batch_id: int, batches: list[dict[str, Any]], busy: Optional[list[tuple[str, str]]]
) -> tuple[str, str]:
    """The answer after the start child, as (answer, message), from the final read alone.

    The child's exit code and every earlier read decide nothing. The words agree with the decision
    returned with them: ``_release`` reads this batch from the same snapshot, and tells which batch
    a busy unit belongs to by the unit's name.
    """
    release = _release(next(batch for batch in batches if batch["batch_id"] == batch_id), busy)
    if release["outcome"]:  # released, restored, refused or failed: never announced as running
        return release["outcome"], release["outcome_text"]
    if release["unit_running"]:  # its own unit is starting or at work, accepted or already releasing
        return "releasing", _ANSWERS["releasing"]
    if busy:  # only units of other batches are at work, and the decision offers no start again
        return "another_release", _ANSWERS["another_release"]
    if busy == [] and release["state"] == "accepted":  # no unit of it is loaded; start again is offered
        return "could_not_start", _ANSWERS["could_not_start"]
    # Any other snapshot, as a release that stopped midway (releasing, and no unit runs) or units that
    # could not be read: the words the decision gives it.
    words = f"{release['status']} {release['unit']} {release['outcome_text']}"
    return ("stopped" if busy == [] else "unreadable"), words


def _refused(action, batch_id, version, digest, batches, busy) -> Optional[tuple[int, str, str]]:
    """Why the reads refuse the answer, as (status, message, code); None when it may go on."""
    if batch_id in _RESERVED:
        return 409, "An answer for this release is under way. Nothing was changed.", "release_under_way"
    if action != "start":
        waiting = _waiting_batch(batches)
        shown = waiting and (waiting["batch_id"], waiting["version"], waiting["digest"])
        if shown != (batch_id, version, digest):
            return 409, _CHANGED, "release_changed"
        if action == "defer" and waiting["state"] == "deferred":
            return 409, "The batch is already put off. Nothing was recorded.", "release_already_put_off"
        return None
    if busy is None:
        return 503, "Whether a release unit runs could not be read. Nothing was started.", "release_units_unreadable"
    if busy:
        return 409, "A release unit is running. Nothing was started.", "release_unit_running"
    current = _current(batches)
    if current is None or current["batch_id"] != batch_id or current["state"] not in _STARTABLE:
        return 409, "This release cannot be started again. Nothing was started.", "release_not_startable"
    return None


def _read() -> tuple[list[dict[str, Any]], Optional[list[tuple[str, str]]]]:
    """Every read: the release record, read-only, and the release units at work (None: unreadable)."""
    path, batches = release_ledger.ledger_path(), []
    if path.is_file():  # without a store nothing was merged yet, and none is made here
        conn = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True, isolation_level=None, timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            batches = release_ledger.list_batches(conn)
        finally:
            conn.close()
    try:
        busy = release_unit.busy_units()
    except release_unit.UnitError as error:
        logger.debug("[api_server] the release units could not be read: %s", error)
        busy = None
    return batches, busy


def _decide(batch_id: int, decision: str, version: int, digest: str) -> None:
    """The merged decide step, bound to the version and digest the owner was shown."""
    conn = release_ledger.connect()
    try:
        release_ledger.decide_release(
            conn, batch_id, decision=decision, shown_digest=digest, expected_version=version, decision_ref=_PAGE_REF
        )
    finally:
        conn.close()


async def _run_start(batch_id: int) -> None:
    """``hermes release start BATCH`` as a child process.

    Its exit code and its output go to the gateway log, never into the answer.
    """
    try:
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hermes_cli.main", "release", "start", str(batch_id),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await child.communicate()
    except OSError:
        logger.exception("[api_server] the release start of batch %s could not run", batch_id)
        return
    logger.info(
        "[api_server] the release start of batch %s exited %s: %s",
        batch_id, child.returncode, output.decode(errors="replace").strip(),
    )


def _view(batches: list[dict[str, Any]], busy: Optional[list[tuple[str, str]]]) -> dict[str, Any]:
    """The decision in plain words: the waiting batch, and the current or last release."""
    waiting, current = _waiting_batch(batches), _current(batches)
    return {
        "object": "hermes.owner_workspace.release_decision",
        "waiting": None if waiting is None else _waiting(waiting),
        "release": None if current is None else _release(current, busy),
    }


def _waiting_batch(batches: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return next((batch for batch in batches if batch["state"] in _WAITING), None)


def _current(batches: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The release at work or kept for recovery, else the next accepted one, else the last that ended.

    Releases begin one at a time in batch order, so the last that ended is the last one listed.
    """
    for states, pick in ((("releasing", "failed"), 0), (("accepted",), 0), (("released", "folded"), -1)):
        found = [batch for batch in batches if batch["state"] in states]
        if found:
            return found[pick]
    return None


def _waiting(batch: dict[str, Any]) -> dict[str, Any]:
    from hermes_cli.owner_workspace import owner_title

    members = batch["members"]
    sentence = _TIERS.get(batch["tier"], _TIERS[2])
    if not all(member["tier_recorded"] for member in members):
        sentence = f"{RISK_NOT_RECORDED} {sentence}"
    return {
        "batch_id": batch["batch_id"],
        "status": _WAITING[batch["state"]],
        "titles": [owner_title(member["title"]) for member in members],
        "count": len(members),
        "tier": batch["tier"],
        "tier_sentence": sentence,
        "version": batch["version"],
        "digest": batch["digest"],
        "actions": ["accept"] if batch["state"] == "deferred" else ["accept", "defer"],
    }


def _release(batch: dict[str, Any], busy: Optional[list[tuple[str, str]]]) -> dict[str, Any]:
    running = None if busy is None else any(unit == release_unit.unit_name(batch["batch_id"]) for unit, _ in busy)
    return {
        "batch_id": batch["batch_id"],
        "state": batch["state"],
        "status": _STATES[batch["state"]],
        "unit_running": running,
        "unit": _UNIT[running],
        "outcome": batch["outcome"],
        "outcome_text": _OUTCOMES[batch["outcome"]],
        # Start again: an accepted release that could not start, or one that stopped midway.
        "actions": ["start"] if busy == [] and batch["state"] in _STARTABLE else [],
    }


def _batch_id(request: web.Request) -> Optional[int]:
    text = request.match_info.get("batch_id", "")
    return int(text) if _BATCH_ID.fullmatch(text) else None


def _refusal(status: int, message: str, code: str, view: Optional[dict[str, Any]] = None) -> web.Response:
    body = _openai_error(message, code=code)
    if view is not None:
        body["decision"] = view
    return web.json_response(body, status=status)
