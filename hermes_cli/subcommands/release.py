"""``hermes release`` subcommand parser: ``prepare``, ``status``, ``run``, ``start`` and
``recover``."""

from __future__ import annotations


def build_release_parser(subparsers) -> None:
    """Attach the ``release`` subcommand, with ``prepare``, ``status``, ``run``, ``start`` and
    ``recover``, to ``subparsers``."""
    release_parser = subparsers.add_parser(
        "release",
        help="Check a release before it runs, and run it: prepare, status, run, start, recover",
        description="Check the accepted release batch before it runs, and run it. prepare works "
            "out PREV and NEW, asks the host every release guard and prints each result; it "
            "writes nothing. status shows the waiting batch and the last outcome. run releases "
            "the accepted batch between a pause and a resume. Run it from the root Hermes home. "
            "start runs the release of one batch as its own user service, hermes-release-BATCH, "
            "from the root home, so it keeps running when the gateway stops. recover brings the "
            "platform back from a release of one batch that stopped midway, to exactly PREV or "
            "exactly NEW; run does the same for a batch that is releasing or failed.")
    release_subparsers = release_parser.add_subparsers(dest="release_command")
    release_subparsers.add_parser(
        "prepare", help="Run every release guard for the accepted batch; writes nothing")
    release_subparsers.add_parser("status", help="Show the waiting batch and the last outcome")
    run_parser = release_subparsers.add_parser(
        "run", help="Release the accepted batch between a pause and a resume")
    run_parser.add_argument(
        "batch", nargs="?", type=int,
        help="Release only this batch; refused unless it is the accepted batch a run takes first,"
            " or recovered when it is releasing or failed")
    start_parser = release_subparsers.add_parser(
        "start", help="Run the release of one batch as its own user service")
    start_parser.add_argument("batch", type=int, help="The number of the batch to release")
    recover_parser = release_subparsers.add_parser(
        "recover", help="Bring the platform back from a release of one batch that stopped midway")
    recover_parser.add_argument(
        "batch", type=int, help="The number of the batch whose release stopped midway")

    def _dispatch_release(args):  # noqa: ANN001
        if getattr(args, "release_command", None) is None:
            release_parser.print_help()
            return 0
        # Lazy import: the release host modules load the gateway's own modules.
        from hermes_cli.release_cmd import cmd_release

        return cmd_release(args)

    release_parser.set_defaults(func=_dispatch_release)
