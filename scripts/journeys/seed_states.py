"""Seed the journeys test stack for one journey (card 2): python seed_states.py JOURNEY < seed.json.

Journey 3 gets a waiting owner question and a waiting suggestion; journey 5 gets a card stopped after repeated
failures. All checks run first and the writes after; a refused run prints one fixed line and writes nothing.
"""

import contextlib
import json
import os
import subprocess
import sys
import time

PREFIX = "Journey test"
BOARD = "journey-test"
PROJECT_SLUG = "journey-test-project"
ROLES = {"3": ("question", "suggestion"), "5": ("stopped",)}
ASK = "Journey test: which option should this journey take?"
GAVE_UP = "worker crashed repeatedly"


class Stop(Exception):
    """The run ends here with this one line."""


def read_names(seed_text, journey):
    """The one place that knows the shape of seed.json, which no file in this repository describes. Assumed:
    {"prefix": "Journey test", "project": NAME, "journeys": {"3": {"question": NAME, "suggestion": NAME},
    "5": {"stopped": NAME}}} (journey 2 may be listed; another card seeds it). Returns {"project": NAME, role: NAME}."""
    if journey == "2":
        raise Stop("journey 2 is seeded by another card; nothing was written")
    try:
        seed = json.loads(seed_text)
        names = {"project": seed["project"], **seed["journeys"][journey]}
        prefix = seed["prefix"]
        shaped = set(names) == {"project", *ROLES[journey]}
    except (ValueError, KeyError, TypeError):
        shaped = False
    if not shaped:
        raise Stop("seed.json holds no seed for this journey in the expected shape; nothing was written")
    if prefix != PREFIX or not all(isinstance(name, str) and name.startswith(PREFIX) for name in names.values()):
        raise Stop(f'a name in seed.json does not start with "{PREFIX}"; nothing was written')
    return names


@contextlib.contextmanager
def at_root(root):
    """The root home as an operator's shell has it: without the pins a worker's environment carries."""
    saved = dict(os.environ)
    for name in [n for n in saved if n.startswith("HERMES_KANBAN_") and n != "HERMES_KANBAN_HOME"]:
        del os.environ[name]
    os.environ.pop("HERMES_PROFILE", None)
    os.environ["HERMES_HOME"] = str(root)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def hermes(*args):
    done = subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        raise Stop(f"hermes {' '.join(args[:2])} failed: {lines[-1] if lines else 'no message'}")
    return done.stdout


def make_card(title, *options):
    out = hermes("kanban", "--board", BOARD, "create", title, "--project", PROJECT_SLUG, *options, "--json")
    try:
        card = json.loads(out)
        return card["id"], card["project_id"]
    except (ValueError, KeyError, TypeError):
        raise Stop("hermes kanban create did not answer with a card") from None


def say(label, value):
    print(f"seed_states: {label}: {value}")


def seed_home(journey, names):
    from hermes_cli import kanban_db, projects_db, release_ledger
    from hermes_cli.config import read_raw_config

    if kanban_db.board_exists(BOARD):
        with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
            if kanban_db.list_tasks(conn, include_archived=True):
                raise Stop("the journey board already holds a card; nothing was written")
    # The copied file as it is: load_config() would lay out the home folder, which a refusal must not do.
    if (read_raw_config().get("kanban") or {}).get("dispatch_in_gateway") is not False:
        raise Stop("the work dispatcher setting does not read off; nothing was written")
    say("dispatcher setting", "off")

    hermes("kanban", "boards", "create", BOARD)
    hermes("project", "create", names["project"], "--slug", PROJECT_SLUG, "--board", BOARD)
    say("project", PROJECT_SLUG)

    if journey == "3":
        question, project_id = make_card(names["question"])
        hermes("kanban", "--board", BOARD, "block", question, ASK, "--kind", "needs_input")
        say("question", question)
        evidence = {
            "schema_version": 1,
            "need": "The journey test needs a suggestion that waits for the owner.",
            "expected_benefit": "The owner page lists one suggestion to accept or reject.",
            "requested_scope": dict.fromkeys(kanban_db.RECOMMENDATION_SCOPE_FLAGS, False),
            "risks": "None; accepting or rejecting it changes nothing outside the test stack.",
            "cost": "None.",
            "rollback": "Reject the suggestion.",
        }
        with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
            suggestion = kanban_db.create_recommendation(
                conn,
                project_id=project_id,
                target_profile="default",
                recommendation_kind="skill",
                recommendation_subject_id="journey-test-suggestion",
                recommendation_label=names["suggestion"],
                recommendation_rationale="Waiting for the owner to accept or reject it.",
                recommendation_evidence=evidence,
                provenance_authority="hermes-profile:default",
                provenance_ref=f"kanban-task:{question}",
                provenance_observed_at=int(time.time()),
            )
        say("suggestion", suggestion)
    else:
        stopped, _ = make_card(names["stopped"], "--max-retries", "1")
        with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
            if kanban_db.claim_task(conn, stopped) is None:
                raise Stop("the stopped card could not be claimed")
            if not kanban_db._record_task_failure(
                conn, stopped, GAVE_UP, outcome="spawn_failed", failure_limit=1, release_claim=True, end_run=True
            ):
                raise Stop("the stopped card did not stop")
        say("stopped card", stopped)

    with contextlib.closing(kanban_db.connect(board=BOARD)) as conn:
        say("work cards on the journey board", len(kanban_db.list_tasks(conn, include_archived=True)))
    with projects_db.connect_closing() as conn:
        say("projects in the root registry", len(projects_db.list_projects(conn, include_archived=True)))
    say("release record", "present" if release_ledger.ledger_path().exists() else "absent")


def main(argv, seed_text):
    try:
        if len(argv) != 1:
            raise Stop("usage: seed_states.py JOURNEY < seed.json")
        names = read_names(seed_text, argv[0])
        from hermes_cli import kanban_db

        root = kanban_db.kanban_home()
        with at_root(root):
            seed_home(argv[0], names)
    except Stop as stop:
        print(f"seed_states: {stop}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], sys.stdin.read()))
