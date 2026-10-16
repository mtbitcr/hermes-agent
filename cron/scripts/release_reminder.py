#!/usr/bin/env python3
"""Script-only hourly cron job: remind the owner once per Vienna evening while a release waits.

Prints one line when the release record says a reminder is due, then records that Vienna day.
Prints nothing otherwise, and nothing (exit 0) when the record cannot be opened or read.
"""

from __future__ import annotations

import contextlib
import sys
from datetime import datetime, timezone
from typing import Optional

REMINDER_LINE = "A release is waiting for your decision on the Decisions page."


def main(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    try:
        from hermes_cli import release_ledger

        with contextlib.closing(release_ledger.connect()) as conn:
            if release_ledger.reminder_due(conn, now):
                sys.stdout.write(REMINDER_LINE + "\n")
                sys.stdout.flush()
                release_ledger.record_reminder(conn, now)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
