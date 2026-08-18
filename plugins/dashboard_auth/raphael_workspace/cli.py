"""CLI for Raphael's managed Workspace and recommendations credentials.

Wires ``hermes kanban-workspace-token <subcommand>``:
  issue    — mint a new bearer token, write it once to an explicit path
  list     — show non-secret metadata for every issued token
  revoke   — disable one token by its token_id

Only non-secret fields ever reach stdout/argv here. ``issue`` is the only
subcommand that touches the secret at all; ``token_store.issue()`` writes it
directly to the explicit output path, and it is never returned here, printed,
logged, or included in the confirmation message.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli.dashboard_auth.audit import AuditWriteError
from plugins.dashboard_auth.raphael_workspace import token_store


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes kanban-workspace-token`` argparse tree."""
    subs = subparser.add_subparsers(dest="workspace_token_command")

    issue_p = subs.add_parser(
        "issue",
        help="Mint a new bearer token for the raphael-workspace read credential",
    )
    issue_p.add_argument(
        "--surface",
        choices=(token_store.WORKSPACE_SURFACE, token_store.RECOMMENDATIONS_SURFACE),
        default=token_store.WORKSPACE_SURFACE,
        help=(
            "Fixed credential surface to grant. Recommendations credentials "
            "have a hard maximum lifetime of 8 hours."
        ),
    )
    issue_p.add_argument(
        "--out",
        required=True,
        help=(
            "Path to write the plaintext bearer token to, exactly once. "
            "The parent directory must already exist and (POSIX) be "
            "owner-only — this command never creates or loosens directories "
            "for a secret output file."
        ),
    )
    issue_p.add_argument(
        "--replaces",
        dest="replaces_token_id",
        help=(
            "Active token_id this issuance is rotating away from. The old "
            "credential remains active until an explicit revoke after cutover."
        ),
    )
    ttl_group = issue_p.add_mutually_exclusive_group()
    ttl_group.add_argument(
        "--ttl-hours",
        type=int,
        help=(
            "Expiry in hours from now. Use 1-8 for the recommendations "
            "surface; omitted values use that surface's fixed default."
        ),
    )
    ttl_group.add_argument(
        "--ttl-days",
        type=int,
        help=(
            f"Expiry in days from now for the workspace surface (max "
            f"{token_store.MAX_TTL_SECONDS // 86400}); omitted values use "
            "that surface's fixed default."
        ),
    )

    subs.add_parser("list", help="List issued tokens (non-secret metadata only)")

    revoke_p = subs.add_parser("revoke", help="Revoke a token by its token_id")
    revoke_p.add_argument("token_id")

    subparser.set_defaults(func=workspace_token_command)


def _cmd_issue(
    *,
    out: str,
    surface: str,
    ttl_days: int | None,
    ttl_hours: int | None,
    replaces_token_id: str | None = None,
) -> int:
    out_path = Path(out).expanduser()
    try:
        policy = token_store.policy_for_surface(surface)
        if ttl_hours is not None:
            ttl_seconds = ttl_hours * 3600
        elif ttl_days is not None:
            ttl_seconds = ttl_days * 86400
        else:
            ttl_seconds = policy.default_ttl_seconds
        record = token_store.issue(
            out_path=out_path,
            ttl_seconds=ttl_seconds,
            replaces_token_id=replaces_token_id,
            surface=surface,
        )
    except (
        ValueError,
        token_store.TokenStoreError,
        token_store.PlaintextOutputError,
        AuditWriteError,
    ) as exc:
        print(f"error: {exc}")
        return 2
    print(f"issued token_id={record.token_id}")
    print(f"  surface:    {surface}")
    print(f"  principal:  {record.principal}")
    print(f"  scope:      {record.scope}")
    print(f"  project:    {record.project}")
    print(f"  board:      {record.board}")
    print(f"  issued_at:  {record.issued_at}")
    print(f"  expires_at: {record.expires_at}")
    if replaces_token_id:
        print(f"  replaces:   {replaces_token_id}")
    print(f"  written to: {out_path}")
    print(
        "The bearer credential was written once, in cleartext, to the path "
        "above. It will not be shown again — move/secure it now."
    )
    return 0


def _cmd_list() -> int:
    try:
        records = token_store.load_records()
    except (token_store.TokenStoreError, AuditWriteError) as exc:
        print(f"error: token store is unreadable: {exc}")
        return 2
    if not records:
        print("no tokens issued yet")
        return 0
    for r in records:
        policy = token_store.policy_for_token_id(r.token_id)
        assert policy is not None
        print(
            f"{r.token_id}  surface={policy.surface}  status={r.status}  "
            f"issued_at={r.issued_at}  expires_at={r.expires_at}  "
            f"revoked_at={r.revoked_at}"
        )
    return 0


def _cmd_revoke(token_id: str) -> int:
    try:
        record = token_store.revoke(token_id)
    except token_store.UnknownTokenError:
        print(f"error: no such token_id: {token_id}")
        return 2
    except (token_store.TokenStoreError, AuditWriteError) as exc:
        print(f"error: token store is unreadable: {exc}")
        return 2
    print(f"token_id={token_id} is revoked (revoked_at={record.revoked_at})")
    return 0


def workspace_token_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "workspace_token_command", None)
    if sub == "issue":
        return _cmd_issue(
            out=args.out,
            surface=args.surface,
            ttl_days=args.ttl_days,
            ttl_hours=args.ttl_hours,
            replaces_token_id=args.replaces_token_id,
        )
    if sub == "list":
        return _cmd_list()
    if sub == "revoke":
        return _cmd_revoke(args.token_id)
    print("usage: hermes kanban-workspace-token {issue,list,revoke}")
    return 2
