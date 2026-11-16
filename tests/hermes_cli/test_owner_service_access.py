"""The one Decisions item before Raphael's service access ends.

Card A of the service access expiry plan. ``service_access_item`` reads only
the end dates in the root registry of Raphael's managed credentials and gives
one standing item once the earliest end among the four surfaces the owner app
holds a key for is 14 days away or less; it keeps the item, with the "has
ended" words, until a newer key exists. Each test writes the registry in the
store's own format with fixed end dates and reads the item at a fixed moment,
so every owner text is checked word for word.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from hermes_cli import owner_service_access as osa
from hermes_constants import get_default_hermes_root, get_hermes_home
from plugins.dashboard_auth.raphael_workspace import token_store

DAY = 86400
END = 1802595600  # 2027-02-14T09:00:00Z
LATER_END = 1803859200  # 2027-03-01T00:00:00Z
OWNER = "default"
REF = "decision_" + hashlib.sha256(f"{OWNER}\x00service_access".encode("utf-8")).hexdigest()[:32]

ENDS_ON = "Raphael's access ends on 14 February 2027"
HAS_ENDED = "Raphael's access has ended"
PROJECTS_END = (
    "On 14 February 2027 the app loses its access to Raphael for Projects. "
    "After that, your Projects will not show their status and Send will not start new work. "
    "The person who runs Raphael's server can renew the access before then."
)
PROJECTS_ENDED = (
    "The app lost its access to Raphael for Projects on 14 February 2027. "
    "Your Projects cannot show their status and Send cannot start new work "
    "until the person who runs Raphael's server renews the access."
)


def _record(surface: str, *, end: int, revoked: bool = False) -> dict:
    """One record of ``surface`` as the store writes it, ending at ``end``."""
    policy = token_store.policy_for_surface(surface)
    issued = end - min(30 * DAY, policy.max_ttl_seconds)
    return {
        "token_id": policy.token_prefix + secrets.token_urlsafe(12),
        "digest": hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        "principal": policy.principal,
        "scope": policy.scope,
        "project": policy.project,
        "board": policy.board,
        "grant": policy.grant,
        "issued_at": issued,
        "expires_at": end,
        "revoked_at": issued + 60 if revoked else None,
    }


def _write_registry(home: Path, records: list[dict]) -> Path:
    """The registry under ``home``, owner-only as the store keeps it."""
    folder = home / "dashboard_auth"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = folder / "raphael_workspace_tokens.json"
    path.write_text(json.dumps({"version": 1, "tokens": records}), encoding="utf-8")
    path.chmod(0o600)
    return path


def _item(title: str, reason: str, created_at: str = "2027-01-31T09:00:00Z") -> dict:
    return {
        "decision_ref": REF,
        "authority": "service",
        "kind": "service_access",
        "project_slug": None,
        "project_name": "Raphael",
        "title": title,
        "reason": reason,
        "created_at": created_at,
    }


def test_window_edges_at_15_14_and_0_days_and_after_expiry():
    _write_registry(get_default_hermes_root(), [_record("workspace", end=END)])

    assert osa.service_access_item(OWNER, END - 15 * DAY) is None
    assert osa.service_access_item(OWNER, END - 14 * DAY - 1) is None
    # 14 days and 0 days before the end: the key still works.
    assert osa.service_access_item(OWNER, END - 14 * DAY) == _item(ENDS_ON, PROJECTS_END)
    assert osa.service_access_item(OWNER, END - 1) == _item(ENDS_ON, PROJECTS_END)
    # From the end on, as long as no newer key exists.
    assert osa.service_access_item(OWNER, END) == _item(HAS_ENDED, PROJECTS_ENDED)
    assert osa.service_access_item(OWNER, END + 40 * DAY) == _item(HAS_ENDED, PROJECTS_ENDED)


def test_one_item_for_all_surfaces_and_stable_ref():
    _write_registry(get_default_hermes_root(), [
        _record("models", end=END + 3 * DAY),
        _record("workspace", end=END + 2 * DAY),
        _record("automations", end=END + DAY),
        _record("connections", end=END),
    ])
    expected = _item(
        ENDS_ON,
        "On 14 February 2027 the app loses its access to Raphael for "
        "Projects, Connections, Automations and Models. "
        "After that, your Projects will not show their status and Send will not start new work. "
        "The person who runs Raphael's server can renew the access before then.",
    )

    # Two reads on different days give the same item: same ref, same waiting time.
    assert osa.service_access_item(OWNER, END - 10 * DAY) == expected
    assert osa.service_access_item(OWNER, END - 9 * DAY) == expected
    other = osa.service_access_item("coder", END - 10 * DAY)
    assert other["decision_ref"] == "decision_" + hashlib.sha256(
        "coder\x00service_access".encode("utf-8")
    ).hexdigest()[:32]


def test_without_projects_the_middle_sentence_names_the_parts_of_the_app():
    _write_registry(get_default_hermes_root(), [
        _record("workspace", end=LATER_END + 60 * DAY),
        _record("models", end=LATER_END + DAY),
        _record("connections", end=LATER_END),
    ])

    assert osa.service_access_item(OWNER, LATER_END - 2 * DAY) == _item(
        "Raphael's access ends on 1 March 2027",
        "On 1 March 2027 the app loses its access to Raphael for Connections and Models. "
        "Parts of the app that use them will stop working until the access is renewed. "
        "The person who runs Raphael's server can renew the access before then.",
        "2027-02-15T00:00:00Z",
    )
    assert osa.service_access_item(OWNER, LATER_END + 2 * DAY) == _item(
        HAS_ENDED,
        "The app lost its access to Raphael for Connections and Models on 1 March 2027. "
        "Parts of the app that use them will stop working until the access is renewed.",
        "2027-02-15T00:00:00Z",
    )


def test_item_clears_after_newer_active_token():
    root = get_default_hermes_root()
    current = _record("workspace", end=END)
    _write_registry(root, [current])
    assert osa.service_access_item(OWNER, END - 5 * DAY) == _item(ENDS_ON, PROJECTS_END)
    assert osa.service_access_item(OWNER, END + DAY) == _item(HAS_ENDED, PROJECTS_ENDED)

    # A renewal adds a newer key that ends more than 14 days away; nothing is deleted.
    _write_registry(root, [current, _record("workspace", end=END + 60 * DAY)])

    assert osa.service_access_item(OWNER, END - 5 * DAY) is None
    assert osa.service_access_item(OWNER, END + DAY) is None


def test_revoked_records_are_ignored():
    root = get_default_hermes_root()
    # A revoked key that ends soon raises nothing, also when it is a surface's only key.
    _write_registry(root, [
        _record("workspace", end=END, revoked=True),
        _record("workspace", end=END + 60 * DAY),
        _record("models", end=END, revoked=True),
    ])
    assert osa.service_access_item(OWNER, END - 5 * DAY) is None
    assert osa.service_access_item(OWNER, END + DAY) is None

    # A revoked newer key does not hide the key that still ends soon.
    _write_registry(root, [
        _record("workspace", end=END),
        _record("workspace", end=END + 60 * DAY, revoked=True),
    ])
    assert osa.service_access_item(OWNER, END - 5 * DAY) == _item(ENDS_ON, PROJECTS_END)


def test_recommendations_records_are_ignored():
    root = get_default_hermes_root()
    _write_registry(root, [_record("recommendations", end=END)])
    assert osa.service_access_item(OWNER, END - DAY) is None
    assert osa.service_access_item(OWNER, END + DAY) is None

    _write_registry(root, [_record("recommendations", end=END - 3 * DAY), _record("workspace", end=END)])
    assert osa.service_access_item(OWNER, END - 5 * DAY) == _item(ENDS_ON, PROJECTS_END)


def test_profile_home_still_reads_the_root_registry(monkeypatch):
    """The gateway of a profile runs with HERMES_HOME at that profile's home
    under the root; the item is built from the root registry all the same."""
    root = get_default_hermes_root()
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert get_default_hermes_root() == root
    assert get_hermes_home() == profile
    written = [
        _record("connections", end=END),
        _record("workspace", end=END + 60 * DAY),
        _record("automations", end=END + 60 * DAY),
    ]
    root_registry = _write_registry(root, written)
    # A registry of the profile's own, which must not be the one read.
    _write_registry(profile, [_record("models", end=END - 5 * DAY), _record("workspace", end=END - 5 * DAY)])
    reads = []
    real_load_records = token_store.load_records

    def counted_load_records():
        records = real_load_records()
        reads.append((token_store.store_path(), len(records)))
        return records

    monkeypatch.setattr(token_store, "load_records", counted_load_records)

    item = osa.service_access_item(OWNER, END - 3 * DAY)

    # The proof, read against the count of the records this test wrote: one
    # read, of the root registry, which held exactly those records.
    assert reads == [(root_registry, len(written))]
    assert item == _item(
        ENDS_ON,
        "On 14 February 2027 the app loses its access to Raphael for Connections. "
        "Parts of the app that use them will stop working until the access is renewed. "
        "The person who runs Raphael's server can renew the access before then.",
    )
    # The pin to the root ends with the read.
    assert get_hermes_home() == profile
