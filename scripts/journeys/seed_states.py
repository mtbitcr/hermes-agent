"""Seed the journeys test stack for one journey (card 2): python seed_states.py JOURNEY < seed.json.

Journey 3 gets a waiting owner question and a waiting suggestion; journey 5 gets a card stopped after repeated
failures. All checks run first and the writes after; a refused run prints one fixed line and writes nothing.
"""

import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time

PREFIX = "Journey test"
BOARD = "journey-test"
PROJECT_SLUG = "journey-test-project"
PARTS = {"3": (("question", "questions"), ("suggestion", "suggestions")), "5": (("stopped", "cards"),)}
ASK = "Journey test: which option should this journey take?"
GAVE_UP = "worker crashed repeatedly"
RECEIPT_KEY = "journey-test-project-seed"
RECEIPT_OPERATION = "owner_task_graph_commit"
RECEIPT_DIGEST = hashlib.sha256(RECEIPT_KEY.encode("ascii")).hexdigest()
NO_GRAPH_ROOT = "journey-test-no-graph"


class Stop(Exception):
    """The run ends here with this one line."""


def lacks(what):
    return Stop(f"{what}; nothing was written")


def entry_for(seed, journey):
    """The one entry of seed.json's journeys for this journey: an object keyed by the number, or a list of
    objects that carry the number as journey, number or id."""
    journeys = seed.get("journeys")
    if isinstance(journeys, dict):
        entries = [entry for number, entry in journeys.items() if str(number) == journey]
    elif isinstance(journeys, list):
        keys = ("journey", "number", "id")
        entries = [entry for entry in journeys if isinstance(entry, dict) and any(str(entry.get(k)) == journey for k in keys)]
    else:
        raise lacks("seed.json has no journeys")
    if len(entries) > 1:
        raise lacks("seed.json has more than one entry for this journey")
    if not entries:
        raise lacks("seed.json has no entry for this journey")
    return entries[0]


def list_named(project, key):
    """The list called key: a direct child of the project, else the only one anywhere below it."""
    if isinstance(project.get(key), list):
        return project[key]
    found, pending = [], list(project.values())
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if isinstance(node.get(key), list):
                found.append(node[key])
                pending.extend(value for name, value in node.items() if name != key)
            else:
                pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    if len(found) > 1:
        raise lacks(f"seed.json has more than one {key} list for this project")
    if not found:
        raise lacks(f"seed.json has no {key} list for this project")
    return found[0]


def item_name(item, label):
    """A list item is a plain string, or an object named by name (or title)."""
    if isinstance(item, dict):
        item = next((item[key] for key in ("name", "title") if isinstance(item.get(key), str)), None)
    if not isinstance(item, str):
        raise lacks(f"a {label} in seed.json has no name")
    return item


def read_names(seed_text, journey):
    """The one place that knows seed.json's shape: journeys[n].projects[0] with a name and lists of questions, suggestions
    and cards. Returns {"project": NAME, role: NAME}; what the journey does not need is ignored, no value is printed."""
    if journey == "2":
        raise Stop("journey 2 is seeded by another card; nothing was written")
    if journey not in PARTS:
        raise lacks("this seed handles journeys 3 and 5 only")
    try:
        seed = json.loads(seed_text)
    except (ValueError, RecursionError):
        seed = None
    if not isinstance(seed, dict):
        raise lacks("seed.json is not a JSON object")
    entry = entry_for(seed, journey)
    projects = entry.get("projects") if isinstance(entry, dict) else None
    if not isinstance(projects, list):
        raise lacks("seed.json has no projects list for this journey")
    if len(projects) != 1:
        raise lacks("seed.json does not have exactly one project for this journey")
    project = projects[0]
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise lacks("the project in seed.json has no name")
    names = {"project": project["name"]}
    for role, key in PARTS[journey]:
        items = list_named(project, key)
        label = key[:-1]
        if len(items) != 1:
            raise lacks(f"seed.json does not have exactly one {label} for this journey")
        names[role] = item_name(items[0], label)
    if seed.get("prefix", PREFIX) != PREFIX or not all(name.startswith(PREFIX) for name in names.values()):
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


def board_holds_a_card(db_path):
    """True if the tasks table holds a row of any kind or status; the board is only read, never written."""
    uri = db_path.resolve().as_uri() + "?mode=ro"
    # Without -wal/-shm an immutable open adds none and sees every row; with them only a plain one sees the log.
    if not any(db_path.with_name(db_path.name + suffix).exists() for suffix in ("-wal", "-shm")):
        uri += "&immutable=1"
    try:
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as conn:
            return conn.execute("SELECT EXISTS (SELECT 1 FROM tasks)").fetchone()[0] == 1
    except sqlite3.Error:
        raise Stop("the journey board could not be read; nothing was written") from None


def owner_receipt_steps():
    """The platform steps that commit the owner receipt, found before the first write."""
    from hermes_cli import kanban_db, owner_workspace, projects_db

    names = ("resolve_owner_context", "_ensure_schema", "_terminal_replay", "_acquire_or_replay", "_update_progress", "_finalize_receipt")
    if not all(callable(getattr(owner_workspace, name, None)) for name in names) or not callable(getattr(kanban_db, "write_board_metadata", None)):
        raise Stop("the platform lacks a step that commits the owner receipt; nothing was written")
    return kanban_db, owner_workspace, projects_db


def commit_owner_receipt():
    """Commit the owner receipt and stamp the journey board, so that the owner workspace lists the Project; a repeat replays."""
    # No public function does this without creating cards, asking approval and activating, so these are bootstrap's own steps.
    kanban_db, ow, projects_db = owner_receipt_steps()
    ctx = ow.resolve_owner_context()
    try:
        with contextlib.closing(projects_db.connect()) as conn:
            project = projects_db.get_project(conn, PROJECT_SLUG)
            if project is None or not project.board_slug:
                raise Stop("the journey project is not in the root registry")
            ow._ensure_schema(conn)
            if ow._terminal_replay(conn, ctx, RECEIPT_KEY, RECEIPT_OPERATION, RECEIPT_DIGEST) is None:
                state, _, token = ow._acquire_or_replay(conn, ctx, RECEIPT_KEY, RECEIPT_OPERATION, RECEIPT_DIGEST)
                if state == "own":
                    ow._update_progress(conn, ctx, RECEIPT_KEY, token, project_id=project.id, board_slug=project.board_slug)
                    kanban_db.write_board_metadata(project.board_slug, project_id=project.id)
                    result = {"ok": True, "project_id": project.id, "root_task_id": NO_GRAPH_ROOT, "task_ids": []}
                    ow._finalize_receipt(conn, ctx, RECEIPT_KEY, token, status="committed", result=result)
    except (ow.OwnerWorkspaceError, sqlite3.Error, ValueError, OSError):
        raise Stop("the owner receipt could not be committed") from None


def seed_home(journey, names):
    from hermes_cli import kanban_db, projects_db, release_ledger
    from hermes_cli.config import read_raw_config

    db = kanban_db.kanban_db_path(BOARD)
    if db.exists() and board_holds_a_card(db):
        raise Stop("the journey board already holds a card; nothing was written")
    owner_receipt_steps()
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

    commit_owner_receipt()

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
