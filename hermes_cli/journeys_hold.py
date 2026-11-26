"""Whether new kanban work waits for the owner journeys of a release.

After a release the release command starts the owner journeys, and one run
resets the test stack on the build server several times; a card that starts
beside it breaks the run. The gateway dispatcher asks :func:`state` after the
owner's pause (``_kanban_dispatch_allowed`` in ``gateway/kanban_watchers.py``)
and starts no new work while the hold is active.

The rule reads, in the root home also when a worker's environment pins a board
store or a profile home, whether ``journeys/settings`` is there, the file
``journeys/result`` (read as ``owner_write_watch._journeys_verdict`` reads it)
and the last ``release_finished`` event of the release record (opened
read-only, as ``release_cmd._record`` opens it). The first line that fits
gives the answer:

a. the settings file is not there: no hold;
b. the result reads ``started`` with no ``finished=`` line, and its start is
   less than ``HOLD_LIMIT`` before now: held, reason ``running``, at the
   latest until that start plus ``HOLD_LIMIT``;
c. the last release ended ``released``, not settled outside the release
   command and not finished by a recovery, less than ``START_GRACE`` before
   now, and the result shows no run started at or after that end: held, reason
   ``starting``, at the latest until that end plus ``START_GRACE``;
d. anything else: no hold.

A result file that is missing, unreadable or not understood, or that starts
later than now, shows no run. Every error gives no hold, so the hold can never
stop the dispatcher for good.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from hermes_cli import release_ledger
from hermes_constants import get_default_hermes_root

HOLD_LIMIT = 45 * 60  # seconds a journeys run at work holds new work, from its start
START_GRACE = 10 * 60  # seconds new work waits after a release end for its journeys to start


def state(now: float) -> dict[str, Any]:
    """The hold at ``now``, in seconds since the epoch: ``active``, its ``reason``
    (``running`` or ``starting``, else ``None``) and its ``latest_end`` in seconds
    since the epoch (``None`` with no hold). Writes nothing and never raises."""
    no_hold: dict[str, Any] = {"active": False, "reason": None, "latest_end": None}
    try:
        root = get_default_hermes_root()
        if not os.path.lexists(root / "journeys" / "settings"):
            return no_hold

        start = None  # the start of the run the result file shows
        at_work = False
        try:
            text = (root / "journeys" / "result").read_text(encoding="utf-8")
            fields = {}
            for line in text.splitlines():
                pair = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
                if pair is None:
                    raise ValueError("a line that is no NAME=value")
                fields[pair.group(1)] = pair.group(2)
            if fields.get("result") in ("passed", "failed", "started") and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fields.get("started", "")
            ):
                started = datetime.strptime(fields["started"], "%Y-%m-%dT%H:%M:%SZ")
                started = started.replace(tzinfo=timezone.utc).timestamp()
                if started <= now:
                    start = started
                    at_work = fields["result"] == "started" and "finished" not in fields
        except (OSError, ValueError):  # missing, unreadable or not understood: no run
            pass
        if at_work and now - start < HOLD_LIMIT:
            return {"active": True, "reason": "running", "latest_end": start + HOLD_LIMIT}

        conn = sqlite3.connect(
            f"{release_ledger.ledger_path().absolute().as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=1.0,
        )
        try:
            row = conn.execute(
                "SELECT payload, created_at, EXISTS (SELECT 1 FROM release_events AS recovery "
                "WHERE recovery.batch_id = finished.batch_id AND recovery.kind = 'recovery_began') "
                "FROM release_events AS finished WHERE kind = 'release_finished' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return no_hold
        payload, end, recovered = json.loads(row[0]), row[1], row[2]
        if (
            isinstance(payload, dict)
            and payload.get("outcome") == "released"
            and "outside" not in payload
            and not recovered
            and 0 <= now - end < START_GRACE
            and (start is None or start < end)
        ):
            return {"active": True, "reason": "starting", "latest_end": end + START_GRACE}
        return no_hold
    except Exception:
        return no_hold
