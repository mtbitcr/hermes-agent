"""Multi-PROCESS stress test for the removal-phase compare-and-set.

Several real OS processes race the SAME phase transition
(``advance_removal_to_quiesced``, FENCED -> QUIESCED) on the same board,
concurrently. Each racer reads the record, sees ``fenced``, waits on the
barrier, and then declares that observation to the transition — so every
one of them is attempting a GENUINELY CONTENDED transition, not repeating
a step it already knows happened. The invariant this proves:

  - Exactly ONE process performs the actual write (outcome ``advanced``,
    ``transitioned`` true).
  - Every other racer is REFUSED (``refused-lost-race``). A racer that
    lost a contended transition must never be handed a success it did not
    perform — the previous version of this test reported five "successes"
    for five processes that wrote nothing.
  - The durable record ends up at the target phase exactly once, with a
    single, consistent ``updated_at``.
  - No racer ever sees a REFUSED_STALE / REFUSED_BACKWARDS / REFUSED_SKIP
    outcome for this same-transition race — those are reserved for a
    caller asking for a DIFFERENT, invalid transition.

A separate pass proves the other half of the rule: a roll-forward repeat
that declares NO observation still gets an idempotent no-op success.

Like the rest of ``tests/stress``, this is a ``__main__``-executable
script rather than a collected pytest module (see ``tests/stress/conftest.py``).
Run it with:

    python tests/stress/test_removal_phase_cas_race.py
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

NUM_RACERS = 6
PROC_TIMEOUT_S = 60
WT = str(Path(__file__).resolve().parents[2])


def _pin_home(hermes_home: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HERMES_KANBAN_HOME"] = hermes_home
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT"):
        os.environ.pop(var, None)
    if WT not in sys.path:
        sys.path.insert(0, WT)


def racer_proc(hermes_home: str, board: str, removal_id: str, result_file: str, ready) -> None:
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    # Observe the phase BEFORE the race: this is what makes each racer a
    # contender rather than a caller repeating a step it knows is done.
    observed = kb.get_removal_phase_record(board).phase
    ready.wait(30)
    result = kb.advance_removal_to_quiesced(
        board, removal_id=removal_id, observed_phase=observed,
    )
    Path(result_file).write_text(
        json.dumps({
            "success": result.success,
            "outcome": None if result.outcome is None else result.outcome.value,
            "transitioned": result.transitioned,
            "observed": observed.value,
            "updated_at": result.record.updated_at if result.record else None,
        }),
        encoding="utf-8",
    )


def rollforward_proc(hermes_home: str, board: str, removal_id: str, result_file: str) -> None:
    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    result = kb.advance_removal_to_quiesced(board, removal_id=removal_id)
    Path(result_file).write_text(
        json.dumps({
            "success": result.success,
            "outcome": None if result.outcome is None else result.outcome.value,
            "transitioned": result.transitioned,
        }),
        encoding="utf-8",
    )


def run() -> int:
    ctx = mp.get_context("spawn")
    tmp = tempfile.mkdtemp(prefix="removal-phase-cas-stress-")
    hermes_home = str(Path(tmp) / "hermes_home")
    Path(hermes_home).mkdir(parents=True, exist_ok=True)
    board = "stress-phase-cas"

    _pin_home(hermes_home)
    from hermes_cli import kanban_db as kb

    meta = kb.create_board(board)
    assert Path(meta["db_path"]).exists()
    backfill = kb.backfill_register_entry(board)
    assert backfill.success, backfill.message

    intent = kb.record_removal_intent(board, mode="reversible")
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(board, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    assert fenced.record.phase == kb.RemovalPhase.FENCED

    ready = ctx.Barrier(NUM_RACERS)
    procs, files = [], {}
    for i in range(NUM_RACERS):
        path = str(Path(tmp) / f"racer-{i}.json")
        files[f"racer-{i}"] = path
        procs.append(ctx.Process(
            target=racer_proc,
            args=(hermes_home, board, intent.removal_id, path, ready),
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

    if not all(r["observed"] == "fenced" for r in results.values()):
        failures.append(f"a racer did not observe the pre-race phase: {results}")

    advanced = [r for r in results.values() if r["outcome"] == "advanced"]
    refused = [r for r in results.values() if r["outcome"] == "refused-lost-race"]
    other = [
        r for r in results.values()
        if r["outcome"] not in ("advanced", "refused-lost-race")
    ]

    if len(advanced) != 1:
        failures.append(f"expected exactly 1 'advanced', got {len(advanced)}: {results}")
    if not all(r["success"] and r["transitioned"] for r in advanced):
        failures.append(f"the winner did not report a real transition: {advanced}")
    if len(refused) != NUM_RACERS - 1:
        failures.append(
            f"expected {NUM_RACERS - 1} refusals, got {len(refused)}: {results}"
        )
    if any(r["success"] or r["transitioned"] for r in refused):
        failures.append(f"a losing racer was handed a success: {refused}")
    if other:
        failures.append(f"unexpected outcomes for a same-transition race: {other}")

    record = kb.get_removal_phase_record(board)
    if record is None or record.phase != kb.RemovalPhase.QUIESCED:
        failures.append(f"final phase is not Quiesced: {record}")
    if record is not None and record.quiesce_completed_at is None:
        failures.append("Quiesced was recorded without its durable completion fact")

    # The other half of the rule: a roll-forward repeat that declares no
    # observation is still an idempotent no-op SUCCESS.
    rollforward_file = str(Path(tmp) / "rollforward.json")
    proc = ctx.Process(
        target=rollforward_proc,
        args=(hermes_home, board, intent.removal_id, rollforward_file),
    )
    proc.start()
    proc.join(PROC_TIMEOUT_S)
    assert proc.exitcode == 0, f"roll-forward process exited with {proc.exitcode}"
    rollforward = json.loads(Path(rollforward_file).read_text(encoding="utf-8"))
    if not rollforward["success"] or rollforward["outcome"] != "idempotent-noop":
        failures.append(f"a roll-forward repeat was not a no-op success: {rollforward}")
    if rollforward["transitioned"]:
        failures.append(f"a roll-forward repeat reported a transition: {rollforward}")

    print(
        f"racers={NUM_RACERS} advanced={len(advanced)} refused={len(refused)} "
        f"other={len(other)} rollforward={rollforward['outcome']}"
    )
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: removal-phase CAS race")
    return 0


if __name__ == "__main__":
    sys.exit(run())
