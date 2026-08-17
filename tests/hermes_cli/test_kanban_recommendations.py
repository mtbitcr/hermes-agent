"""ITEM31BG Stage A: additive recommendation-card ``tasks`` schema migration and the
``create_recommendation`` creation seam. Read API / ordinary-task isolation live elsewhere."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
# Full pre-this-change ``tasks`` column set (from ``kb.SCHEMA_SQL``),
# missing only the fourteen new columns.
_LEGACY_TASKS_SQL = """CREATE TABLE tasks (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT, status TEXT NOT NULL, priority INTEGER DEFAULT 0,
    created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT, branch_name TEXT, project_id TEXT,
    claim_lock TEXT, claim_expires INTEGER, tenant TEXT, result TEXT, idempotency_key TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0, worker_pid INTEGER, last_failure_error TEXT,
    max_runtime_seconds INTEGER, last_heartbeat_at INTEGER, current_run_id INTEGER, workflow_template_id TEXT,
    current_step_key TEXT, skills TEXT, model_override TEXT, provider_override TEXT, reasoning_effort TEXT,
    max_retries INTEGER, goal_mode INTEGER NOT NULL DEFAULT 0, goal_max_turns INTEGER, session_id TEXT,
    block_kind TEXT, block_recurrences INTEGER NOT NULL DEFAULT 0
); """
_NEW_TASK_COLUMNS = (
    "task_kind", "recommendation_kind", "recommendation_subject_id", "recommendation_label", "recommendation_rationale",
    "target_profile", "review_policy", "provenance_authority", "provenance_ref", "provenance_observed_at",
    "recommendation_evidence", "recommendation_decision", "recommendation_effective_state", "recommendation_lifecycle_version",
)
_PROVENANCE = dict(provenance_authority="static-analyzer", provenance_ref="finding-123", provenance_observed_at=1_700_000_000)
_EVIDENCE = dict(schema_version=1, need="Repeated capability gap", expected_benefit="Complete work safely",
    requested_scope={flag: False for flag in kb.RECOMMENDATION_SCOPE_FLAGS}, risks="Low", cost="No added cost",
    rollback="Remove the staged configuration")
_VALID_KWARGS = dict(project_id="proj-1", target_profile="worker", recommendation_kind="skill",
    recommendation_subject_id="x", recommendation_label="x", recommendation_evidence=_EVIDENCE, **_PROVENANCE)
def _hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home
@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = _hermes_home(tmp_path, monkeypatch)
    kb.init_db()
    return home
def _make_legacy_db(db_path: Path) -> None:
    # Modern non-tasks tables plus a tasks table missing the fourteen new
    # columns, with one live running task, event, and run.
    conn = sqlite3.connect(str(db_path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript("DROP TABLE tasks;")
    conn.executescript(_LEGACY_TASKS_SQL)
    conn.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_at, started_at, workspace_kind, claim_lock, claim_expires, current_run_id) "
        "VALUES ('t-running', 'legacy running task', 'worker', 'running', 1000, 1000, 'scratch', 'host:1', 9999, 1)"
    )
    conn.execute("INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES ('t-running', 1, 'created', NULL, 1000)")
    conn.execute(
        "INSERT INTO task_runs (id, task_id, status, claim_lock, claim_expires, started_at) VALUES (1, 't-running', 'running', 'host:1', 9999, 1000)"
    )
    conn.commit()
    conn.close()
@pytest.fixture
def legacy_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Like kanban_home but seeds a legacy pre-migration DB; callers run their own init_db() to migrate.
    home = _hermes_home(tmp_path, monkeypatch)
    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_legacy_db(db_path)
    return home
def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
def _indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA index_list({table})") if not r["name"].startswith("sqlite_")}
# --- Migration ---
def test_fresh_db_columns_default_and_ordinary_create_task(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        cols = {r["name"]: r for r in conn.execute("PRAGMA table_info(tasks)")}
        for name in _NEW_TASK_COLUMNS:
            assert name in cols, f"missing column {name}"
        assert cols["task_kind"]["notnull"] == 1 and cols["task_kind"]["dflt_value"] == "'work'"
        assert "idx_tasks_recommendation_scope" in _indexes(conn, "tasks")
        tid = kb.create_task(conn, title="ordinary")
        row = conn.execute("SELECT task_kind, recommendation_kind, target_profile, review_policy FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["task_kind"] == "work"
        assert (row["recommendation_kind"], row["target_profile"], row["review_policy"]) == (None, None, None)
def test_legacy_migration_adds_columns_preserves_data_and_is_idempotent(legacy_home: Path) -> None:
    kb.init_db()
    with kb.connect_closing() as conn:
        cols = _table_cols(conn, "tasks")
        for name in _NEW_TASK_COLUMNS:
            assert name in cols
        assert "idx_tasks_recommendation_scope" in _indexes(conn, "tasks")
        row = conn.execute("SELECT task_kind, recommendation_kind, target_profile FROM tasks WHERE id = 't-running'").fetchone()
        assert (row["task_kind"], row["recommendation_kind"], row["target_profile"]) == ("work", None, None)  # backfilled to safe default
        task = kb.get_task(conn, "t-running")
        assert task is not None
        assert (task.status, task.assignee, task.claim_lock, task.claim_expires, task.current_run_id) == ("running", "worker", "host:1", 9999, 1)
        events = kb.list_events(conn, "t-running")
        assert len(events) == 1 and events[0].kind == "created" and events[0].run_id == 1
        run = conn.execute("SELECT * FROM task_runs WHERE task_id = 't-running'").fetchone()
        assert (run["status"], run["claim_lock"], run["claim_expires"], run["started_at"]) == ("running", "host:1", 9999, 1000)
    kb.init_db()  # second pass must be a clean no-op
    with kb.connect_closing() as conn:
        counts = (len(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in ("tasks", "task_events", "task_runs"))
        assert tuple(counts) == (1, 1, 1)
# --- create_recommendation: happy path ---
@pytest.mark.parametrize("kind", ["skill", "permission", "connection", "pipeline", "provider_model_policy", "profile_setting"])
def test_create_recommendation_accepts_all_valid_kinds(kanban_home: Path, kind: str) -> None:
    evidence = _EVIDENCE
    if kind == "permission":
        evidence = _evidence_with_scope(permission_widening=True)
    elif kind == "connection":
        evidence = _evidence_with_scope(connector_access=True)
    with kb.connect_closing() as conn:
        tid = kb.create_recommendation(
            conn,
            **{
                **_VALID_KWARGS,
                "recommendation_kind": kind,
                "recommendation_evidence": evidence,
            },
        )
        assert conn.execute("SELECT recommendation_kind FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == kind
# Merges: exact persisted field values, no run/workspace/branch/assignee, exactly one typed event, and never
# delegating to create_task — all against one recommendation, so the row is only built/fetched once.
def test_create_recommendation_persists_exact_fields_isolated_shape_and_skips_create_task(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_create_task = kb.create_task
    monkeypatch.setattr(kb, "create_task", lambda *a, **k: calls.append("create_task") or real_create_task(*a, **k))
    with kb.connect_closing() as conn:
        tid = kb.create_recommendation(
            conn, project_id="proj-1", target_profile="Worker", recommendation_kind="skill",
            recommendation_subject_id="translation", recommendation_label="Load the translation skill",
            recommendation_rationale="Repeated translation requests observed", **_PROVENANCE,
            recommendation_evidence=_EVIDENCE,
        )
        assert calls == []
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert (row["task_kind"], row["status"], row["review_policy"], row["project_id"]) == ("recommendation", "review", "owner", "proj-1")
        assert (row["target_profile"], row["recommendation_subject_id"], row["recommendation_label"], row["recommendation_rationale"]) == (
            "worker", "translation", "Load the translation skill", "Repeated translation requests observed",
        )  # target_profile canonicalized like assignee
        assert (row["provenance_authority"], row["provenance_ref"], row["provenance_observed_at"]) == ("static-analyzer", "finding-123", 1_700_000_000)
        assert row["title"] and row["body"] is None and row["assignee"] is None and row["current_run_id"] is None and row["workspace_path"] is None and row["branch_name"] is None  # title is a fixed non-semantic constant
        assert conn.execute("SELECT COUNT(*) AS c FROM task_runs WHERE task_id = ?", (tid,)).fetchone()["c"] == 0
        events = conn.execute("SELECT kind, run_id, payload FROM task_events WHERE task_id = ?", (tid,)).fetchall()
        assert len(events) == 1 and events[0]["kind"] == "recommendation_created" and events[0]["run_id"] is None
        payload = json.loads(events[0]["payload"])
        assert payload["recommendation_kind"] == "skill"
        assert payload["project_id"] == "proj-1" and payload["target_profile"] == "worker"
# --- create_recommendation: invalid input ---
@pytest.mark.parametrize(
    "overrides",
    [
        dict(recommendation_kind="not-a-kind"), dict(project_id="   "), dict(target_profile="   "), dict(recommendation_subject_id="  "),
        dict(recommendation_label=""), dict(provenance_authority=""), dict(recommendation_label="x" * 201), dict(recommendation_subject_id="x" * 201),
        dict(recommendation_rationale="x" * 4001), dict(provenance_observed_at="not-an-int"),
    ],
    ids=[
        "unknown-kind", "blank-project-id", "blank-target-profile", "blank-subject-id", "blank-label", "blank-provenance-authority",
        "oversized-label", "oversized-subject-id", "oversized-rationale", "non-int-observed-at",
    ],
)
def test_create_recommendation_rejects_invalid_input(kanban_home: Path, overrides: dict) -> None:
    with kb.connect_closing() as conn, pytest.raises(ValueError):
        kb.create_recommendation(conn, **{**_VALID_KWARGS, **overrides})
# --- ITEM31BH: atomic identity-scoped idempotency ---
def _rec_rows(conn) -> list:
    return conn.execute("SELECT * FROM tasks WHERE task_kind = 'recommendation' ORDER BY created_at, id").fetchall()
def _rec_events(conn) -> list:
    return conn.execute(
        "SELECT e.* FROM task_events e JOIN tasks t ON t.id = e.task_id WHERE t.task_kind = 'recommendation' ORDER BY e.id"
    ).fetchall()
def test_repeat_publish_returns_existing_id_with_no_second_row_or_event(kanban_home: Path) -> None:
    # Rewording the advice, observing it later, and noticing it under a different task must not re-nag the owner.
    with kb.connect_closing() as conn:
        first = kb.create_recommendation(conn, **_VALID_KWARGS)
        again = kb.create_recommendation(
            conn, **{**_VALID_KWARGS, "recommendation_label": "totally different wording",
                     "recommendation_rationale": "new evidence", "provenance_ref": "kanban-task:t-later",
                     "provenance_observed_at": 1_800_000_000},
        )
        assert again == first
        rows = _rec_rows(conn)
        assert len(rows) == 1 and [r["kind"] for r in _rec_events(conn)] == ["recommendation_created"]
        assert rows[0]["recommendation_label"] == "x"  # the first card wins; the later wording never overwrites it
def test_repeat_publish_backfills_only_legacy_missing_evidence(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        conn.execute(
            "UPDATE tasks SET recommendation_evidence = NULL, "
            "recommendation_decision = NULL, "
            "recommendation_effective_state = NULL, "
            "recommendation_lifecycle_version = NULL WHERE id = ?",
            (rec_id,),
        )
        conn.commit()
        assert kb.create_recommendation(conn, **_VALID_KWARGS) == rec_id
        row = conn.execute(
            "SELECT recommendation_evidence, recommendation_decision, "
            "recommendation_effective_state, recommendation_lifecycle_version "
            "FROM tasks WHERE id = ?",
            (rec_id,),
        ).fetchone()
        assert json.loads(row["recommendation_evidence"]) == _EVIDENCE
        assert tuple(row)[1:] == ("pending", "none", 0)
        assert [event["kind"] for event in _rec_events(conn)] == [
            "recommendation_created",
            "recommendation_evidence_added",
        ]
        assert kb.create_recommendation(conn, **_VALID_KWARGS) == rec_id
        assert len(_rec_rows(conn)) == 1
        assert len(_rec_events(conn)) == 2


@pytest.mark.parametrize(
    "overrides",
    [dict(project_id="proj-2"), dict(target_profile="other-worker"), dict(provenance_authority="hermes-profile:other"),
     dict(recommendation_kind="pipeline"), dict(recommendation_subject_id="different-subject")],
    ids=["project", "profile", "authority", "kind", "subject"],
)
def test_each_identity_dimension_creates_a_distinct_recommendation(kanban_home: Path, overrides: dict) -> None:
    with kb.connect_closing() as conn:
        base = kb.create_recommendation(conn, **_VALID_KWARGS)
        other = kb.create_recommendation(conn, **{**_VALID_KWARGS, **overrides})
        assert other != base and len(_rec_rows(conn)) == 2
def test_resolved_or_archived_recommendation_stays_the_dedup_authority(kanban_home: Path) -> None:
    # The owner already decided. A worker must not be able to re-raise the same advice by outliving the card.
    with kb.connect_closing() as conn:
        first = kb.create_recommendation(conn, **_VALID_KWARGS)
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (first,))
        conn.commit()
        assert kb.create_recommendation(conn, **_VALID_KWARGS) == first
        assert len(_rec_rows(conn)) == 1
def test_concurrent_publishers_on_separate_connections_collapse_to_one_row(kanban_home: Path) -> None:
    # Two workers racing the same advice: BEGIN IMMEDIATE serializes them, so the loser reads the winner's row.
    from concurrent.futures import ThreadPoolExecutor
    def _publish() -> str:
        with kb.connect_closing() as conn:
            return kb.create_recommendation(conn, **_VALID_KWARGS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: _publish(), range(4)))
    assert len(set(ids)) == 1
    with kb.connect_closing() as conn:
        assert len(_rec_rows(conn)) == 1 and len(_rec_events(conn)) == 1
def test_secret_bearing_fields_are_redacted_in_row_event_and_identity(kanban_home: Path) -> None:
    # Two publishes whose subject differs only inside the secret: redaction happens before both durability and
    # the identity digest, so the secret neither persists nor forks the identity into a second nag.
    with kb.connect_closing() as conn:
        first = kb.create_recommendation(
            conn, **{**_VALID_KWARGS, "recommendation_subject_id": "api_key=aaa111bbb222",
                     "recommendation_label": "password: hunter2secret", "recommendation_rationale": "token=aaa111bbb222"},
        )
        again = kb.create_recommendation(
            conn, **{**_VALID_KWARGS, "recommendation_subject_id": "api_key=ccc333ddd444",
                     "recommendation_label": "password: someothersecret", "recommendation_rationale": "token=ccc333ddd444"},
        )
        assert again == first
        rows = _rec_rows(conn)
        assert len(rows) == 1
        assert (rows[0]["recommendation_subject_id"], rows[0]["recommendation_label"], rows[0]["recommendation_rationale"]) == (
            "api_key=***", "password: ***", "token=***",
        )
        blob = json.dumps([dict(r) for r in rows]) + json.dumps([dict(e) for e in _rec_events(conn)])
        for secret in ("aaa111bbb222", "ccc333ddd444", "hunter2secret", "someothersecret"):
            assert secret not in blob
def test_identity_digest_is_opaque_and_never_stores_its_input(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_recommendation(conn, **_VALID_KWARGS)
        key = conn.execute("SELECT idempotency_key FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    assert key.startswith("rec1:") and len(key) == len("rec1:") + 64
    for raw in ("proj-1", "worker", "skill", "static-analyzer"):
        assert raw not in key
def test_ordinary_work_task_idempotency_is_unchanged_by_recommendation_keys(kanban_home: Path) -> None:
    # Neither namespace may see the other's key: work lookups filter task_kind='work', recommendations 'recommendation'.
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        shared = conn.execute("SELECT idempotency_key FROM tasks WHERE id = ?", (rec_id,)).fetchone()[0]
        work_a = kb.create_task(conn, title="ordinary", idempotency_key=shared)
        assert work_a != rec_id
        assert kb.create_task(conn, title="ordinary", idempotency_key=shared) == work_a
        assert kb.create_recommendation(conn, **_VALID_KWARGS) == rec_id
        assert len(_rec_rows(conn)) == 1


# --- ITEM32E: governed decision and effective-state lifecycle ---

def _evidence_with_scope(**scope_overrides) -> dict:
    evidence = json.loads(json.dumps(_EVIDENCE))
    evidence["requested_scope"].update(scope_overrides)
    return evidence


def _work_run(
    conn: sqlite3.Connection,
    label: str,
    *,
    stamp: int,
    project_id: str = "proj-1",
    active: bool = False,
) -> tuple[str, int]:
    task_id = kb.create_task(conn, title=label, initial_status="running")
    status = "running" if active else "done"
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            task_id,
            status,
            stamp,
            None if active else stamp,
            None if active else "completed",
        ),
    )
    run_id = int(cur.lastrowid)
    conn.execute(
        "UPDATE tasks SET status = ?, project_id = ?, current_run_id = ?, "
        "started_at = ?, completed_at = ? WHERE id = ?",
        (
            status,
            project_id,
            run_id,
            stamp,
            None if active else stamp,
            task_id,
        ),
    )
    conn.commit()
    return task_id, run_id


def _finish_run(
    conn: sqlite3.Connection, task_id: str, run_id: int, *, stamp: int
) -> None:
    conn.execute(
        "UPDATE task_runs SET status = 'done', ended_at = ?, outcome = 'completed' "
        "WHERE id = ? AND task_id = ?",
        (stamp, run_id, task_id),
    )
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
        (stamp, task_id),
    )
    conn.commit()


def _snapshot(conn: sqlite3.Connection, task_id: str) -> dict:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return kb.recommendation_lifecycle_snapshot(row)


def _decide(
    conn: sqlite3.Connection,
    task_id: str,
    governance: tuple[str, int],
    *,
    decision: str = "accepted",
    authority: str = "owner_approved",
    gate_ref: str | None = "owner-gate:item32e",
    version: int = 0,
) -> dict:
    return kb.decide_recommendation(
        conn,
        task_id,
        decision=decision,
        authority=authority,
        gate_ref=gate_ref,
        reason="Governance review completed",
        actor="raphael-owner",
        governance_task_id=governance[0],
        governance_run_id=governance[1],
        expected_lifecycle_version=version,
    )


def _transition(
    conn: sqlite3.Connection,
    task_id: str,
    governance: tuple[str, int],
    *,
    state: str,
    version: int,
    **evidence,
) -> dict:
    return kb.transition_recommendation(
        conn,
        task_id,
        effective_state=state,
        reason=f"Record {state} evidence",
        actor="raphael-owner",
        governance_task_id=governance[0],
        governance_run_id=governance[1],
        expected_lifecycle_version=version,
        **evidence,
    )


@pytest.mark.parametrize(
    "bad_evidence",
    [
        {**_EVIDENCE, "extra": True},
        {**_EVIDENCE, "schema_version": 2},
        {**_EVIDENCE, "requested_scope": {}},
        {
            **_EVIDENCE,
            "requested_scope": {
                **_EVIDENCE["requested_scope"],
                "network_access": "yes",
            },
        },
    ],
    ids=("extra-field", "schema-version", "missing-scope-flags", "non-boolean-scope"),
)
def test_recommendation_evidence_is_closed_and_fail_closed(
    kanban_home: Path, bad_evidence: dict
) -> None:
    with kb.connect_closing() as conn, pytest.raises(ValueError):
        kb.create_recommendation(
            conn, **{**_VALID_KWARGS, "recommendation_evidence": bad_evidence}
        )


def test_permission_and_connection_require_explicit_scope_flags(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        for kind, flag in (
            ("permission", "permission_widening"),
            ("connection", "connector_access"),
        ):
            with pytest.raises(ValueError, match=flag):
                kb.create_recommendation(
                    conn,
                    **{
                        **_VALID_KWARGS,
                        "recommendation_kind": kind,
                        "recommendation_subject_id": f"missing-{flag}",
                    },
                )


def test_decision_lifecycle_requires_fresh_governance_and_owner_for_widening(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        first_governance = _work_run(conn, "governance-defer", stamp=created_at + 1)
        assert _decide(
            conn,
            rec_id,
            first_governance,
            decision="deferred",
            authority="preauthorized_non_widening",
            gate_ref=None,
        )["lifecycle_version"] == 1

        with pytest.raises(ValueError, match="distinct"):
            _decide(
                conn,
                rec_id,
                first_governance,
                authority="preauthorized_non_widening",
                version=1,
            )
        second_governance = _work_run(
            conn, "governance-accept", stamp=created_at + 2
        )
        with pytest.raises(ValueError, match="gate_ref"):
            _decide(
                conn,
                rec_id,
                second_governance,
                authority="preauthorized_non_widening",
                gate_ref=None,
                version=1,
            )
        accepted = _decide(
            conn,
            rec_id,
            second_governance,
            authority="preauthorized_non_widening",
            version=1,
        )
        assert accepted == {
            "recommendation_id": rec_id,
            "decision": "accepted",
            "effective_state": "none",
            "lifecycle_version": 2,
        }
        with pytest.raises(ValueError, match="version mismatch"):
            _decide(conn, rec_id, second_governance, version=1)
        immutable_governance = _work_run(
            conn, "governance-after-accept", stamp=created_at + 3
        )
        with pytest.raises(ValueError, match="illegal recommendation decision"):
            _decide(
                conn,
                rec_id,
                immutable_governance,
                decision="rejected",
                version=2,
            )
        assert _snapshot(conn, rec_id)["decision"] == "accepted"

        widened_id = kb.create_recommendation(
            conn,
            **{
                **_VALID_KWARGS,
                "recommendation_kind": "permission",
                "recommendation_subject_id": "filesystem-write",
                "recommendation_evidence": _evidence_with_scope(
                    permission_widening=True
                ),
            },
        )
        widening_governance = _work_run(
            conn, "governance-widening", stamp=created_at + 4
        )
        with pytest.raises(ValueError, match="owner_approved"):
            _decide(
                conn,
                widened_id,
                widening_governance,
                authority="preauthorized_non_widening",
            )
        assert _decide(conn, widened_id, widening_governance)["decision"] == "accepted"


def test_lifecycle_db_mutators_refuse_worker_context_before_writes(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        governance = _work_run(
            conn, "governance-decision", stamp=created_at + 1
        )
        before_events = len(kb.list_events(conn, rec_id))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
        with pytest.raises(PermissionError, match="operator-only"):
            _decide(conn, rec_id, governance)
        assert _snapshot(conn, rec_id)["decision"] == "pending"
        assert len(kb.list_events(conn, rec_id)) == before_events


def test_lifecycle_requires_the_latest_completed_governance_run(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        governance_task, old_run = _work_run(
            conn, "governance-decision", stamp=created_at + 1
        )
        cur = conn.execute(
            "INSERT INTO task_runs "
            "(task_id, status, started_at, ended_at, outcome) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                governance_task,
                "done",
                created_at + 2,
                created_at + 2,
                "completed",
            ),
        )
        latest_run = int(cur.lastrowid)
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (latest_run, governance_task),
        )
        conn.commit()
        with pytest.raises(ValueError, match="latest run"):
            _decide(conn, rec_id, (governance_task, old_run))
        assert _decide(conn, rec_id, (governance_task, latest_run))[
            "decision"
        ] == "accepted"


def test_effective_lifecycle_requires_readback_and_supports_revocation(
    kanban_home: Path,
) -> None:
    config_identity = "a" * 64
    rollback_identity = "b" * 64
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(
            conn,
            **{
                **_VALID_KWARGS,
                "recommendation_kind": "profile_setting",
                "recommendation_subject_id": "agent.max_turns",
            },
        )
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        assert _decide(
            conn,
            rec_id,
            _work_run(conn, "governance-decision", stamp=created_at + 1),
        )["lifecycle_version"] == 1
        staged = _transition(
            conn,
            rec_id,
            _work_run(conn, "governance-stage", stamp=created_at + 2),
            state="staged",
            version=1,
            native_surface="hermes.profile.agent.max_turns",
            config_identity=config_identity,
            rollback_identity=rollback_identity,
        )
        assert staged["effective_state"] == "staged"

        canary = _work_run(
            conn, "new-session-canary", stamp=created_at + 4, active=True
        )
        assert _transition(
            conn,
            rec_id,
            _work_run(conn, "governance-canary", stamp=created_at + 3),
            state="canary_running",
            version=2,
            canary_task_id=canary[0],
            canary_run_id=canary[1],
        )["effective_state"] == "canary_running"
        _finish_run(conn, canary[0], canary[1], stamp=created_at + 5)

        stale_governance = _work_run(
            conn, "stale-governance-verify", stamp=created_at + 4
        )
        verifier = _work_run(
            conn, "independent-verifier", stamp=created_at + 7
        )
        with pytest.raises(ValueError, match="after the canary run"):
            _transition(
                conn,
                rec_id,
                stale_governance,
                state="verified",
                version=3,
                canary_task_id=canary[0],
                canary_run_id=canary[1],
                verifier_task_id=verifier[0],
                verifier_run_id=verifier[1],
                readback_identity=config_identity,
            )
        verify_governance = _work_run(
            conn, "governance-verify", stamp=created_at + 6
        )
        with pytest.raises(ValueError, match="equal config_identity"):
            _transition(
                conn,
                rec_id,
                verify_governance,
                state="verified",
                version=3,
                canary_task_id=canary[0],
                canary_run_id=canary[1],
                verifier_task_id=verifier[0],
                verifier_run_id=verifier[1],
                readback_identity="c" * 64,
            )
        assert _snapshot(conn, rec_id)["effective_state"] == "canary_running"
        assert _transition(
            conn,
            rec_id,
            verify_governance,
            state="verified",
            version=3,
            canary_task_id=canary[0],
            canary_run_id=canary[1],
            verifier_task_id=verifier[0],
            verifier_run_id=verifier[1],
            readback_identity=config_identity,
        )["effective_state"] == "verified"
        assert _transition(
            conn,
            rec_id,
            _work_run(conn, "governance-promote", stamp=created_at + 8),
            state="promoted",
            version=4,
        )["effective_state"] == "promoted"
        revoke_governance = _work_run(
            conn, "governance-revoke", stamp=created_at + 9
        )
        revoke_verifier = _work_run(
            conn, "independent-revoke-verifier", stamp=created_at + 10
        )
        final = _transition(
            conn,
            rec_id,
            revoke_governance,
            state="revoked",
            version=5,
            readback_identity=rollback_identity,
            verifier_task_id=revoke_verifier[0],
            verifier_run_id=revoke_verifier[1],
        )
        assert final == {
            "recommendation_id": rec_id,
            "decision": "accepted",
            "effective_state": "revoked",
            "lifecycle_version": 6,
        }
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (rec_id,),
        ).fetchall()
        assert [event["kind"] for event in events] == [
            "recommendation_created",
            "recommendation_decided",
            "recommendation_transitioned",
            "recommendation_transitioned",
            "recommendation_transitioned",
            "recommendation_transitioned",
            "recommendation_transitioned",
        ]
        versions = [
            json.loads(event["payload"]).get("lifecycle_version", 0)
            for event in events
        ]
        assert versions == list(range(7))


def test_rollback_and_invalid_transition_leave_snapshot_atomic(
    kanban_home: Path,
) -> None:
    config_identity = "d" * 64
    rollback_identity = "e" * 64
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        _decide(
            conn,
            rec_id,
            _work_run(conn, "governance-decision", stamp=created_at + 1),
        )
        with pytest.raises(ValueError, match="illegal recommendation effective"):
            _transition(
                conn,
                rec_id,
                _work_run(conn, "governance-skip", stamp=created_at + 2),
                state="promoted",
                version=1,
            )
        assert _snapshot(conn, rec_id)["effective_state"] == "none"
        stage_governance = _work_run(
            conn, "governance-stage", stamp=created_at + 3
        )
        with pytest.raises(ValueError, match="must be distinct"):
            _transition(
                conn,
                rec_id,
                stage_governance,
                state="staged",
                version=1,
                native_surface="hermes.skills.catalog",
                config_identity=config_identity,
                rollback_identity=config_identity,
            )
        _transition(
            conn,
            rec_id,
            stage_governance,
            state="staged",
            version=1,
            native_surface="hermes.skills.catalog",
            config_identity=config_identity,
            rollback_identity=rollback_identity,
        )
        rollback_governance = _work_run(
            conn, "governance-rollback", stamp=created_at + 4
        )
        rollback_verifier = _work_run(
            conn, "independent-rollback-verifier", stamp=created_at + 5
        )
        with pytest.raises(ValueError, match="equal rollback_identity"):
            _transition(
                conn,
                rec_id,
                rollback_governance,
                state="rolled_back",
                version=2,
                readback_identity=config_identity,
                verifier_task_id=rollback_verifier[0],
                verifier_run_id=rollback_verifier[1],
            )
        assert _snapshot(conn, rec_id) == {
            "recommendation_id": rec_id,
            "decision": "accepted",
            "effective_state": "staged",
            "lifecycle_version": 2,
        }
        assert _transition(
            conn,
            rec_id,
            rollback_governance,
            state="rolled_back",
            version=2,
            readback_identity=rollback_identity,
            verifier_task_id=rollback_verifier[0],
            verifier_run_id=rollback_verifier[1],
        )["effective_state"] == "rolled_back"


def test_active_canary_cannot_be_rolled_back_and_rollback_needs_verifier(
    kanban_home: Path,
) -> None:
    config_identity = "1" * 64
    rollback_identity = "2" * 64
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        _decide(
            conn,
            rec_id,
            _work_run(conn, "governance-decision", stamp=created_at + 1),
        )
        _transition(
            conn,
            rec_id,
            _work_run(conn, "governance-stage", stamp=created_at + 2),
            state="staged",
            version=1,
            native_surface="hermes.skills.catalog",
            config_identity=config_identity,
            rollback_identity=rollback_identity,
        )
        canary = _work_run(
            conn, "active-canary", stamp=created_at + 4, active=True
        )
        _transition(
            conn,
            rec_id,
            _work_run(conn, "governance-canary", stamp=created_at + 3),
            state="canary_running",
            version=2,
            canary_task_id=canary[0],
            canary_run_id=canary[1],
        )
        rollback_governance = _work_run(
            conn, "governance-rollback", stamp=created_at + 5
        )
        rollback_verifier = _work_run(
            conn, "independent-rollback-verifier", stamp=created_at + 6
        )
        with pytest.raises(ValueError, match="canary Task/run must be completed"):
            _transition(
                conn,
                rec_id,
                rollback_governance,
                state="rolled_back",
                version=3,
                readback_identity=rollback_identity,
                verifier_task_id=rollback_verifier[0],
                verifier_run_id=rollback_verifier[1],
            )
        assert _snapshot(conn, rec_id)["effective_state"] == "canary_running"

        _finish_run(conn, canary[0], canary[1], stamp=created_at + 5)
        with pytest.raises(ValueError, match="verifier_task_id"):
            _transition(
                conn,
                rec_id,
                rollback_governance,
                state="rolled_back",
                version=3,
                readback_identity=rollback_identity,
            )
        assert _transition(
            conn,
            rec_id,
            rollback_governance,
            state="rolled_back",
            version=3,
            readback_identity=rollback_identity,
            verifier_task_id=rollback_verifier[0],
            verifier_run_id=rollback_verifier[1],
        )["effective_state"] == "rolled_back"


def test_decision_rejects_incomplete_and_cross_project_governance(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        rec_id = kb.create_recommendation(conn, **_VALID_KWARGS)
        created_at = conn.execute(
            "SELECT created_at FROM tasks WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        active = _work_run(
            conn, "active-governance", stamp=created_at + 1, active=True
        )
        with pytest.raises(ValueError, match="must be completed"):
            _decide(conn, rec_id, active)
        other_project = _work_run(
            conn,
            "wrong-project-governance",
            stamp=created_at + 2,
            project_id="proj-2",
        )
        with pytest.raises(ValueError, match="recommendation project"):
            _decide(conn, rec_id, other_project)
        assert _snapshot(conn, rec_id)["decision"] == "pending"
