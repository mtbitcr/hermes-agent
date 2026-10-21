"""Card A2: the owner's delivery list, beside the reviewed policy.

The reviewed policy (``kanban_delivery_policy.json``) stays read-only source. The repositories the owner connects
are kept in a second list, one data file under the shared kanban root that holds the boards, so the dispatcher,
every worker and every profile read the same entries. Only :func:`record_confirmed_entry` writes it, and only with
the digest of the exact entry the owner was shown; each confirmation is appended to a log beside the list before
the list is written. An entry adds a repository and nothing else: it never names a repository of the reviewed
policy, so the reviewed entry always wins; its owner is the delivery installation's account; and its one required
check is ``quality``. A list that cannot be read counts as empty, so the reviewed policy alone still applies.
:mod:`hermes_cli.kanban_delivery` reads its policy through :func:`combined_policy`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_delivery_fences import _REPO, Decision, Policy, load_policy

# The account of the transport's fixed installation (kanban_delivery_github.INSTALLATION_ID): its tokens reach only
# this account's repositories, so an entry of any other owner could never be delivered.
_INSTALLATION_ACCOUNT = "mtbitcr"


def owner_policy_path() -> Path:
    """The owner list, in the kanban root the boards live under (kanban_db.boards_root's parent)."""
    return kb.kanban_home() / "kanban" / "delivery" / "owner_policy.json"


def confirmations_path() -> Path:
    """The append-only log of the owner's confirmations, beside the list."""
    return owner_policy_path().with_name("owner_policy_confirmations.jsonl")


def entry_digest(repository: str, required_checks: list[str]) -> str:
    """SHA-256 of the entry's canonical JSON: what the owner is shown and confirms."""
    canonical = json.dumps(
        {"repository": repository, "required_checks": required_checks}, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_entry(repository: str, required_checks: list[str], code_policy: Policy) -> Decision:
    """May the owner list name this entry? Fails closed: an entry not shown to be valid is refused."""
    if not isinstance(repository, str) or not _REPO.fullmatch(repository):
        return Decision(False, "invalid_repository", f"{repository!r} is not an owner/name repository")
    # GitHub names are case-insensitive, so another spelling of a reviewed repository is that repository.
    if repository.lower() in {name.lower() for name in code_policy.required_checks}:
        return Decision(False, "platform_repository", f"{repository} is in the reviewed policy")
    if repository.split("/")[0] != _INSTALLATION_ACCOUNT:
        return Decision(False, "foreign_account", f"{repository} is not a repository of {_INSTALLATION_ACCOUNT}")
    # Version 1: exactly the one check a connected repository's minimal CI reports; any other set never goes green.
    if required_checks != ["quality"]:
        return Decision(False, "checks_not_allowed", f"{repository}: the required checks must be exactly quality")
    return Decision(True, "valid", f"{repository} may be listed")


def _code_policy() -> Policy:
    # Late import: kanban_delivery imports this module at load time.
    from hermes_cli.kanban_delivery import _POLICY_FILE

    return load_policy(_POLICY_FILE.read_text(encoding="utf-8"))


def _listed(code_policy: Policy) -> Policy:
    """The owner list's valid entries; none when the list cannot be read or parsed."""
    try:
        listed = load_policy(owner_policy_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):  # RecursionError: JSON nested too deep to parse
        return Policy({})
    return Policy({
        repository: checks for repository, checks in listed.required_checks.items()
        if validate_entry(repository, list(checks), code_policy).allowed
    })


def record_confirmed_entry(repository: str, required_checks: list[str], shown_digest: str) -> Decision:
    """List the entry the owner confirmed with ``shown_digest``: the confirmation is appended to the log first,
    then the list is replaced atomically. A digest of anything else, or a repeat, records nothing."""
    code_policy = _code_policy()
    decision = validate_entry(repository, required_checks, code_policy)
    if not decision.allowed:
        return decision
    digest = entry_digest(repository, required_checks)
    if shown_digest != digest:
        return Decision(False, "digest_mismatch", f"{repository}: the confirmed digest is not this entry's")
    listed = _listed(code_policy).required_checks
    if repository in listed:
        return Decision(True, "already_confirmed", f"{repository} is already listed")
    path = owner_policy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    confirmation = {
        "repository": repository, "required_checks": required_checks, "digest": digest,
        "confirmed_at": int(time.time()),
    }
    with confirmations_path().open("a", encoding="utf-8") as log:
        log.write(json.dumps(confirmation, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
    entries = {**listed, repository: required_checks}
    document = {name: {"required_checks": list(checks)} for name, checks in entries.items()}
    kb._atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return Decision(True, "confirmed", f"{repository} is listed")


def owner_policy() -> Policy:
    """The owner list's valid entries alone: none collides with the reviewed policy. Empty when either file
    cannot be read or parsed."""
    try:
        code_policy = _code_policy()
    except (OSError, ValueError):
        return Policy({})
    return _listed(code_policy)


def combined_policy(code_policy: Policy) -> Policy:
    """The reviewed policy united with the owner list's valid entries; the reviewed entry always wins."""
    required_checks = dict(code_policy.required_checks)
    for repository, checks in _listed(code_policy).required_checks.items():
        required_checks.setdefault(repository, checks)
    return Policy(required_checks)
