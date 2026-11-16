"""The one Decisions item before Raphael's service access ends.

Card A of the service access expiry plan. The owner app reaches Raphael with
one key per surface, and each key has an end date in the root registry of
Raphael's managed credentials. Once the earliest end among the surfaces'
newest keys is 14 days away or less, Decisions lists one standing item; it
stays, with the "has ended" words, until a newer key exists. The item is built
from the end dates and the surfaces alone: no other field of a key record is
ever copied.
"""

from __future__ import annotations

import hashlib
import logging
import time

from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)

# Old readers keep their closed schema: only a reader that names this gets the item.
SERVICE_ACCESS_CAPABILITY = "service_access_v1"
SERVICE_ACCESS_WARNING_DAYS = 14

# The surfaces the owner app holds a key for, in the order the owner reads them.
# Keys of any other surface (the recommendations family) raise no item.
SURFACE_WORDS = {
    "workspace": "Projects",
    "connections": "Connections",
    "automations": "Automations",
    "models": "Models",
}

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_PARTS = "Parts of the app that use them will stop working until the access is renewed."

# The titles of the run approvals the same Decisions handler lists. They moved
# here from the handler only to keep gateway/platforms/api_server.py within its
# line ceiling.
RUN_APPROVAL_TITLES = {
    "owner_workspace_bootstrap": "Approve the new Project",
    "owner_task_graph_commit": "Approve the first Project milestone",
    "owner_project_plan_commit": "Approve Project changes",
    "owner_task_move": "Approve a work-state change",
    "owner_task_comment": "Approve an owner reply",
    "owner_task_retry": "Approve trying stopped work again",
    "owner_project_lifecycle": "Approve the Project lifecycle change",
}


def service_access_item(owner_profile: str, now: float) -> dict | None:
    """The owner's service access item at ``now``, or None while every
    surface's newest key ends more than 14 days later.

    No state of the registry may break Decisions: when it cannot be read, the
    answer comes without the item."""
    try:
        return _item(owner_profile, now, _latest_ends())
    except Exception as exc:
        # The type only: the text of a store error can carry a path.
        logger.warning("Service access item left out: %s", type(exc).__name__)
        return None


def _latest_ends() -> dict[str, int]:
    """Each surface's latest end among its keys that were not revoked."""
    from plugins.dashboard_auth.raphael_workspace import token_store

    # A profile's gateway reads the root registry, where Raphael's keys live.
    pin = set_hermes_home_override(get_default_hermes_root())
    try:
        records = token_store.load_records()
    finally:
        reset_hermes_home_override(pin)
    ends: dict[str, int] = {}
    for record in records:
        surface = token_store.policy_for_token_id(record.token_id).surface
        if record.revoked_at is None and surface in SURFACE_WORDS:
            ends[surface] = max(ends.get(surface, record.expires_at), record.expires_at)
    return ends


def _item(owner_profile: str, now: float, ends: dict[str, int]) -> dict | None:
    window = SERVICE_ACCESS_WARNING_DAYS * 86400
    listed = [surface for surface in SURFACE_WORDS if surface in ends and ends[surface] - now <= window]
    if not listed:
        return None
    end = min(ends[surface] for surface in listed)
    day = time.gmtime(end)
    date = f"{day.tm_mday} {_MONTHS[day.tm_mon - 1]} {day.tm_year}"
    words = [SURFACE_WORDS[surface] for surface in listed]
    surfaces = f"{', '.join(words[:-1])} and {words[-1]}" if len(words) > 1 else words[0]
    projects = "workspace" in listed
    if end <= now:
        title = "Raphael's access has ended"
        reason = f"The app lost its access to Raphael for {surfaces} on {date}. " + (
            "Your Projects cannot show their status and Send cannot start new work "
            "until the person who runs Raphael's server renews the access."
            if projects else _PARTS
        )
    else:
        title = f"Raphael's access ends on {date}"
        reason = f"On {date} the app loses its access to Raphael for {surfaces}. " + (
            "After that, your Projects will not show their status and Send will not start new work."
            if projects else _PARTS
        ) + " The person who runs Raphael's server can renew the access before then."
    return {
        "decision_ref": "decision_" + hashlib.sha256(
            f"{owner_profile}\x00service_access".encode("utf-8")
        ).hexdigest()[:32],
        "authority": "service",
        "kind": "service_access",
        "project_slug": None,
        "project_name": "Raphael",
        "title": title,
        "reason": reason,
        # Waiting since the window opened, so the item does not move from day to day.
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end - window)),
    }
