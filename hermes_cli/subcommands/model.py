"""``hermes model`` subcommand parser."""

from __future__ import annotations

import sys
from typing import Callable


def build_model_parser(subparsers, *, cmd_model: Callable) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    model_parser = subparsers.add_parser(
        "model", help="Select default model and provider",
        description="Interactively select your inference provider and default model")
    model_parser.add_argument(
        "--refresh", action="store_true",
        help="Wipe the model picker disk cache and re-fetch every provider's live /v1/models list.")
    model_parser.add_argument(
        "--portal-url", help="Portal base URL for Nous login (default: production portal)")
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)")
    model_parser.add_argument(
        "--client-id", default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)")
    model_parser.add_argument("--scope", default=None, help="OAuth scope to request for Nous login")
    model_parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not attempt to open the browser automatically during Nous login")
    model_parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)")
    model_parser.add_argument(
        "--ca-bundle", help="Path to CA bundle PEM file for Nous TLS verification")
    model_parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS verification for Nous login (testing only)")
    model_parser.set_defaults(func=cmd_model)

    # Optional, so a bare ``hermes model`` keeps the picker above; the check
    # sets its own handler because main.py dispatches ``args.func(args)``.
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    update_check = model_subparsers.add_parser(
        "update-check",
        help="Look for a newer version of each model in use and file at most one suggestion",
        description=(
            "Look for a newer version of every model a role uses, check that every "
            "sign-in lists it, that it supports the role's effort and that it answers "
            "one probe, then file at most one suggestion in Decisions. Nothing changes "
            "until the owner clicks Update. Runs inside the monthly planning card."))
    update_check.add_argument(
        "--practice", metavar="ROLE", default=None,
        help="File a practice suggestion for ROLE's current model; clicking Update then changes nothing")
    update_check.set_defaults(func=cmd_model_update_check)


def cmd_model_update_check(args) -> int:
    """Run ``hermes model update-check`` and print the receipt's outcome."""
    from hermes_cli import model_update_check

    try:
        receipt = model_update_check.run_update_check(practice_role=args.practice)
    except model_update_check.UpdateCheckRefused as exc:
        print(f"Update check refused: {exc}", file=sys.stderr)
        return 1
    print(f"{receipt['result']}: {receipt['reason']}")
    filed = receipt.get("filed")
    if filed:
        print(f"Filed {filed['subject_id']} as {filed['task_id']} on board {filed['board']}.")
    return 0
