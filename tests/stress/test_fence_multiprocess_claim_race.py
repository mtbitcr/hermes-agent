"""Multi-PROCESS stress test for the board removal fence.

Companion to ``test_concurrency.py``. Several worker processes hammer a
shared board with real claims while a separate process commits the
fence-closing point underneath them. The invariants asserted afterwards
are the ones the fence exists to provide:

  - Exactly one process commits the open→closing transition; every other
    close attempt reports idempotent success WITHOUT writing.
  - No claim is ever granted after the closing gate is committed: every
    claim a worker recorded as granted is on a task that is running, and
    every task that no worker recorded as granted is still ready.
  - The same holds for the paths the first repair left ungated: a content
    write (``add_comment``) and a removal (``archive_task``) racing the
    close either land before it or refuse, never half-land after it.
  - Every observation of the (gate, epoch_mirror) pair is either fully
    open at the old epoch or fully closed at the new one. A half-closed
    pair is never visible to anyone.
  - The refusal carries the structured closed outcome, not a generic error.

Like the rest of ``tests/stress``, this is a ``__main__``-executable
script rather than a collected pytest module (see ``tests/stress/conftest.py``
— the directory sets ``collect_ignore_glob = ["*.py"]``). Run it with:

    python tests/stress/test_fence_multiprocess_claim_race.py
"""

import json
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

NUM_WORKERS = 4
NUM_MUTATORS = 2
NUM_CLOSERS = 3
NUM_OBSERVERS = 3
NUM_TASKS = 40
PROC_TIMEOUT_S = 120
WT = str(Path(__file__).resolve().parents[2])


def _pin_home(hermes_home: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HERMES_KANBAN_HOME"] = hermes_home
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        os.environ.pop(var, None)
    if WT not in sys.path:
        sys.path.insert(0, WT)


def claimer_proc(worker_id: int, hermes_home: str, board: str,
                 task_ids: list, result_file: str,
                 ready, enough_granted, closed) -> None:
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    granted, refused, other = [], [], []
    conn = kb.connect(board=board)
    ready.wait(60)  # every process has imported and connected
    try:
        for task_id in task_ids:
            try:
                claimed = kb.claim_task(conn, task_id, claimer=f"w{worker_id}")
            except kb.BoardFenceClosedError as exc:
                refused.append([task_id, exc.refusal.outcome.value])
                continue
            except Exception as exc:  # pragma: no cover - surfaced by the run
                other.append([task_id, f"{type(exc).__name__}: {exc}"])
                continue
            if claimed is not None:
                granted.append(task_id)
                if len(granted) >= 3:
                    enough_granted.set()
            time.sleep(0.001)
        # Event-based, not wall-clock: exactly one attempt that is
        # unambiguously AFTER the committed close.
        closed.wait(60)
        try:
            late = kb.claim_task(conn, task_ids[-1], claimer=f"late-w{worker_id}")
        except kb.BoardFenceClosedError as exc:
            refused.append(["<post-close>", exc.refusal.outcome.value])
        else:
            other.append([
                "<post-close>",
                f"claim after close returned {late!r} instead of refusing",
            ])
    finally:
        conn.close()

    Path(result_file).write_text(
        json.dumps({"granted": granted, "refused": refused, "other": other}),
        encoding="utf-8",
    )


def mutator_proc(worker_id: int, hermes_home: str, board: str,
                 task_ids: list, result_file: str,
                 ready, enough_granted, closed) -> None:
    """Hammer the CONTENT-WRITE and REMOVAL paths across the same close.

    These are the two classes the first repair left without an
    in-statement gate: only the two claims and the heartbeat folded Gate B
    into their own predicate, so a comment or an archive could land after
    the fence closed.
    """
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    committed, refused, other = [], [], []
    conn = kb.connect(board=board)
    ready.wait(60)
    try:
        for task_id in task_ids[worker_id::NUM_MUTATORS]:
            for kind, call in (
                ("comment", lambda t=task_id: kb.add_comment(
                    conn, t, f"m{worker_id}", "racing the fence")),
                ("archive", lambda t=task_id: kb.archive_task(conn, t)),
            ):
                try:
                    call()
                except kb.BoardFenceClosedError as exc:
                    refused.append([kind, exc.refusal.outcome.value])
                except Exception as exc:  # pragma: no cover - surfaced by the run
                    other.append([kind, f"{type(exc).__name__}: {exc}"])
                else:
                    committed.append([kind, task_id])
            time.sleep(0.001)
        closed.wait(60)
        # One attempt of each class unambiguously AFTER the commit.
        for kind, call in (
            ("comment", lambda: kb.add_comment(
                conn, task_ids[0], f"late-m{worker_id}", "after the close")),
            ("archive", lambda: kb.archive_task(conn, task_ids[0])),
        ):
            try:
                call()
            except kb.BoardFenceClosedError as exc:
                refused.append([f"<post-close>{kind}", exc.refusal.outcome.value])
            else:
                other.append([kind, "landed after the close instead of refusing"])
    finally:
        conn.close()

    Path(result_file).write_text(
        json.dumps({"committed": committed, "refused": refused, "other": other}),
        encoding="utf-8",
    )


def closer_proc(hermes_home: str, board: str, result_file: str,
                ready, enough_granted, closed) -> None:
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    ready.wait(60)
    enough_granted.wait(60)
    result = kb.commit_fence_closing_point(board)
    Path(result_file).write_text(
        json.dumps({
            "success": result.success,
            "transitioned": result.transitioned,
            "new_epoch": result.new_epoch,
            "message": result.message,
        }),
        encoding="utf-8",
    )
    closed.set()


def observer_proc(hermes_home: str, board: str, result_file: str,
                  ready, closed) -> None:
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    db_path = str(kb.kanban_db_path(board=board))
    seen = set()
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    ready.wait(60)

    def sample() -> None:
        row = conn.execute(
            "SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1"
        ).fetchone()
        if row is not None:
            seen.add((row["gate"], int(row["epoch_mirror"])))

    try:
        while not closed.is_set():
            sample()
        # Keep sampling briefly past the commit so the closed state is
        # observed too, not just the open one.
        for _ in range(200):
            sample()
    finally:
        conn.close()
    Path(result_file).write_text(
        json.dumps({"pairs": sorted(list(p) for p in seen)}), encoding="utf-8"
    )


def run() -> int:
    ctx = mp.get_context("spawn")
    tmp = tempfile.mkdtemp(prefix="fence-stress-")
    hermes_home = str(Path(tmp) / "hermes_home")
    Path(hermes_home).mkdir(parents=True, exist_ok=True)
    board = "stress-fence"

    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    # The SHIPPED creation API — the one the CLI and the dashboard call.
    meta = kb.create_board(board)
    assert Path(meta["db_path"]).exists(), "create_board produced no store"

    # Arm the fence via the recorded migration — create_board no longer
    # publishes a register entry on its own.
    backfill_result = kb.backfill_register_entry(board)
    assert backfill_result.success, f"backfill failed: {backfill_result.message}"

    conn = kb.connect(board=board)
    task_ids = []
    try:
        for i in range(NUM_TASKS):
            task_id = kb.create_task(conn, title=f"stress-{i}", assignee="worker")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,)
                )
            task_ids.append(task_id)
    finally:
        conn.close()

    # Commit the removal intent so the close has an authoritative epoch.
    entry = kb.get_register_entry(board)
    assert entry is not None, "backfill left no register authority"
    old_epoch, new_epoch = entry.epoch, entry.epoch + 1
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name=board,
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=new_epoch,
            epoch_before=old_epoch,
            gate_move=kb.GateMove.PENDING,
            epoch_lineage=(entry.epoch_lineage or []) + [new_epoch],
            created_at=entry.created_at,
            updated_at=int(time.time()),
        )
    )

    total = NUM_WORKERS + NUM_MUTATORS + NUM_CLOSERS + NUM_OBSERVERS
    ready = ctx.Barrier(total)
    enough_granted = ctx.Event()
    closed = ctx.Event()

    # Disjoint slices: a mutator archiving a task a claimer is about to
    # claim would starve the claim race instead of stressing it.
    claim_ids = task_ids[: NUM_TASKS // 2]
    mutate_ids = task_ids[NUM_TASKS // 2 :]

    procs, files = [], {}
    for i in range(NUM_WORKERS):
        path = str(Path(tmp) / f"claimer-{i}.json")
        files[f"claimer-{i}"] = path
        procs.append(ctx.Process(
            target=claimer_proc,
            args=(i, hermes_home, board, claim_ids, path,
                  ready, enough_granted, closed),
        ))
    for i in range(NUM_MUTATORS):
        path = str(Path(tmp) / f"mutator-{i}.json")
        files[f"mutator-{i}"] = path
        procs.append(ctx.Process(
            target=mutator_proc,
            args=(i, hermes_home, board, mutate_ids, path,
                  ready, enough_granted, closed),
        ))
    for i in range(NUM_CLOSERS):
        path = str(Path(tmp) / f"closer-{i}.json")
        files[f"closer-{i}"] = path
        procs.append(ctx.Process(
            target=closer_proc,
            args=(hermes_home, board, path, ready, enough_granted, closed),
        ))
    for i in range(NUM_OBSERVERS):
        path = str(Path(tmp) / f"observer-{i}.json")
        files[f"observer-{i}"] = path
        procs.append(ctx.Process(
            target=observer_proc,
            args=(hermes_home, board, path, ready, closed),
        ))

    for p in procs:
        p.start()
    for p in procs:
        p.join(PROC_TIMEOUT_S)
        assert p.exitcode == 0, f"process exited with {p.exitcode}"

    results = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in files.items()
    }

    failures = []

    # --- exactly one closer transitions ---------------------------------
    closers = [v for k, v in results.items() if k.startswith("closer-")]
    if not all(c["success"] for c in closers):
        failures.append(f"a close attempt failed: {closers}")
    transitions = sum(1 for c in closers if c["transitioned"])
    if transitions != 1:
        failures.append(f"expected exactly 1 transition, got {transitions}")

    # --- the fence really closed, at the authoritative epoch -------------
    db_path = str(kb.kanban_db_path(board=board))
    probe = sqlite3.connect(db_path)
    probe.row_factory = sqlite3.Row
    final = probe.execute(
        "SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1"
    ).fetchone()
    statuses = {
        r["id"]: r["status"]
        for r in probe.execute("SELECT id, status FROM tasks").fetchall()
    }
    probe.close()
    if final["gate"] != "closing" or int(final["epoch_mirror"]) != new_epoch:
        failures.append(f"final fence state wrong: {dict(final)}")

    # --- no half-closed pair was ever observable ------------------------
    legal = {(("open"), old_epoch), (("closing"), new_epoch)}
    for name, payload in results.items():
        if not name.startswith("observer-"):
            continue
        for gate, mirror in payload["pairs"]:
            if (gate, mirror) not in legal:
                failures.append(f"{name} saw half-closed pair {(gate, mirror)}")

    # --- grants and refusals agree with the stored rows ------------------
    granted, refused = set(), set()
    for name, payload in results.items():
        if not name.startswith("claimer-"):
            continue
        if payload["other"]:
            failures.append(f"{name} hit unexpected errors: {payload['other']}")
        granted.update(payload["granted"])
        refused.update(
            t for t, _ in payload["refused"] if t != "<post-close>"
        )
        post_close = [o for t, o in payload["refused"] if t == "<post-close>"]
        if post_close != ["refused-closed"]:
            failures.append(
                f"{name} post-close claim outcome was {post_close!r}"
            )
        for _task, outcome in payload["refused"]:
            if outcome != "refused-closed":
                failures.append(f"{name} refusal outcome was {outcome!r}")
    # The deterministic post-close attempts (checked above) prove the fence
    # is enforcing; the uncontrolled-race refusals are schedule-dependent.
    if not granted:
        failures.append("no claim ever succeeded: the race proved nothing")
    archived_ok = {
        task_id
        for name, payload in results.items()
        if name.startswith("mutator-")
        for kind, task_id in payload["committed"]
        if kind == "archive"
    }
    for task_id in task_ids:
        if task_id in archived_ok:
            expected = "archived"
        else:
            expected = "running" if task_id in granted else "ready"
        if statuses.get(task_id) != expected:
            failures.append(
                f"{task_id}: expected {expected}, found {statuses.get(task_id)}"
            )

    # --- content writes and removals obey the same gate -----------------
    committed_mutations, refused_mutations = 0, 0
    for name, payload in results.items():
        if not name.startswith("mutator-"):
            continue
        if payload["other"]:
            failures.append(f"{name} hit unexpected outcomes: {payload['other']}")
        committed_mutations += len(payload["committed"])
        refused_mutations += len(payload["refused"])
        post_close = sorted(
            kind for kind, _ in payload["refused"] if kind.startswith("<post-close>")
        )
        if post_close != ["<post-close>archive", "<post-close>comment"]:
            failures.append(f"{name} post-close mutations were {post_close!r}")
        for kind, outcome in payload["refused"]:
            if outcome != "refused-closed":
                failures.append(f"{name} {kind} refusal outcome was {outcome!r}")
    # The deterministic post-close attempts (checked above) prove the fence
    # is enforcing; the uncontrolled-race refusals are schedule-dependent.

    # Nothing a mutator was refused may be visible on disk: every archived
    # task must be one it recorded as committed.
    archived_now = {t for t, s in statuses.items() if s == "archived"}
    if not archived_now <= archived_ok:
        failures.append(
            f"archived tasks with no committing mutator: {sorted(archived_now - archived_ok)}"
        )

    print(f"tasks={NUM_TASKS} granted={len(granted)} refused={len(refused)} "
          f"mutations_committed={committed_mutations} "
          f"mutations_refused={refused_mutations} "
          f"closers={len(closers)} transitions={transitions}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: fence multiprocess claim race")
    return 0


if __name__ == "__main__":
    sys.exit(run())
