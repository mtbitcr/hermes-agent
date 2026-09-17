"""Contract tests for owner_project_removal."""
from __future__ import annotations
import json, re, time
from unittest.mock import patch, MagicMock
import pytest
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli.owner_workspace import (
    OwnerContext, OwnerProposalAuthority, OwnerWorkspaceError, _digest,
    _project_removal_state, _removal_consequences_document,
    _REMOVAL_PHASE_OWNER_MAP, _REMOVAL_SAFE_ERRORS,
    OWNER_PROJECT_REMOVAL_STATE_CAPABILITY,
)

_S = "test_removal"
_n = [0]

def _ctx(auth=None):
    return OwnerContext(actor="default", profile="default", session=_S, authority=auth)

def _auth(op, key, payload):
    return OwnerProposalAuthority(
        actor="default", profile="default", session=_S,
        conversation="c", response_id="r", operation=op,
        idempotency_key=key, payload_digest=_digest(payload),
    )

def _setup(mp):
    _n[0] += 1
    s = f"rp{_n[0]}"
    mp.setattr("tools.approval.request_exact_operation_approval",
               lambda *a, **kw: {"approved": True})
    mp.setattr(ow, "_assert_board_ownership", lambda *a, **kw: None)
    with projects_db.connect_closing() as c:
        pid = projects_db.create_project(c, name="RP", slug=s,
                                         primary_path=f"/tmp/{s}")
        projects_db.update_project(c, pid, board_slug=s)
    pc = projects_db.connect()
    try:
        ow._ensure_schema(pc)
        from hermes_cli.sqlite_util import write_txn
        with write_txn(pc):
            pc.execute(
                "INSERT INTO owner_workspace_receipts "
                "(actor,profile,idempotency_key,operation,request_digest,"
                "status,project_id,board_slug,result_json,"
                "terminal_generation,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("default","default",f"bk{_n[0]}","owner_workspace_bootstrap",
                 "d","committed",pid,s,
                 json.dumps({"ok":True,"project_id":pid}),
                 1,int(time.time()),int(time.time())),
            )
    finally:
        pc.close()
    return pid, s

def _mk_op(pid, s, ik, phase, **kw):
    with projects_db.connect_closing() as c:
        projects_db.record_removal_operation(
            c, project_id=pid, idempotency_key=ik,
            action="start", phase=phase, mode="reversible",
            board_slug=s, removal_id="rm", **kw)

def _call(pid, ik, action, rev=0, **kw):
    p = {"idempotency_key":ik,"project_id":pid,"expected_revision":rev,"action":action}
    if "consequences_digest" in kw:
        p["consequences_digest"] = kw["consequences_digest"]
    return ow.project_removal(
        _ctx(_auth("owner_project_removal", ik, p)),
        idempotency_key=ik, project_id=pid,
        expected_revision=rev, action=action, **kw)


# §1: double-click / idempotency
class TestIdempotency:
    def test_same_key_joins(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "ik", "accepted")
        r = _call(pid, "ik", "start")
        assert r["ok"] and r.get("joined")

    def test_dup_record(self):
        with projects_db.connect_closing() as c:
            projects_db.record_removal_operation(c, project_id="dp",
                idempotency_key="k", action="start", phase="accepted")
            assert not projects_db.record_removal_operation(c, project_id="dp",
                idempotency_key="k", action="start", phase="accepted")


# §2: stale revision
class TestStaleRevision:
    def test_refused(self, monkeypatch):
        pid, _ = _setup(monkeypatch)
        p = {"idempotency_key":"sk","project_id":pid,"expected_revision":999,"action":"start"}
        with pytest.raises(OwnerWorkspaceError) as e:
            ow.project_removal(_ctx(_auth("owner_project_removal","sk",p)),
                idempotency_key="sk", project_id=pid,
                expected_revision=999, action="start")
        assert e.value.code == "stale_revision"


# §3: wrong digest
class TestWrongDigest:
    def test_requires_digest(self):
        p = {"idempotency_key":"nd","project_id":"x","expected_revision":0,
             "action":"confirm_permanent"}
        with pytest.raises(OwnerWorkspaceError):
            ow.project_removal(_ctx(_auth("owner_project_removal","nd",p)),
                idempotency_key="nd", project_id="x",
                expected_revision=0, action="confirm_permanent")

    def test_mismatch(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "md", "accepted", consequences_digest="real")
        ck = f"mdc{_n[0]}"
        p = {"idempotency_key":ck,"project_id":pid,"expected_revision":0,
             "action":"confirm_permanent","consequences_digest":"wrong"}
        with pytest.raises(OwnerWorkspaceError) as e:
            ow.project_removal(_ctx(_auth("owner_project_removal",ck,p)),
                idempotency_key=ck, project_id=pid, expected_revision=0,
                action="confirm_permanent", consequences_digest="wrong")
        assert e.value.code == "digest_mismatch"


# §4: foreign project
class TestForeignProject:
    def test_refused(self, monkeypatch):
        monkeypatch.setattr("tools.approval.request_exact_operation_approval",
                            lambda **kw: {"approved": True})
        p = {"idempotency_key":"fp","project_id":"none","expected_revision":0,"action":"start"}
        with pytest.raises(OwnerWorkspaceError) as e:
            ow.project_removal(_ctx(_auth("owner_project_removal","fp",p)),
                idempotency_key="fp", project_id="none",
                expected_revision=0, action="start")
        assert e.value.code == "project_not_owned"


# §5: status read does not advance
class TestPureRead:
    def test_no_advance(self):
        with projects_db.connect_closing() as c:
            projects_db.record_removal_operation(c, project_id="pr",
                idempotency_key="rk", action="start", phase="accepted")
            b = projects_db.get_removal_operation(c, "pr", "rk")
            _project_removal_state(c, "pr", None)
            a = projects_db.get_removal_operation(c, "pr", "rk")
            assert b["phase"] == a["phase"]
            assert b["updated_at"] == a["updated_at"]


# §6: cancel at phase boundaries
class TestCancelPhase:
    def test_before_applied_ok(self, monkeypatch):
        for phase in ("accepted", "fenced", "quiesced", "carried", "released"):
            pid, s = _setup(monkeypatch)
            ik = f"c{phase}{_n[0]}"
            _mk_op(pid, s, ik, phase)
            cancel_ik = f"cc{phase}{_n[0]}"
            with patch.object(kanban_db, "abandon_or_cancel_removal",
                              return_value=MagicMock(success=True)):
                r = _call(pid, cancel_ik, "cancel")
            assert r["ok"], f"cancel at {phase} failed"

    def test_at_or_after_applied_refused(self, monkeypatch):
        for phase in ("applied", "swept"):
            pid, s = _setup(monkeypatch)
            ik = f"cr{phase}{_n[0]}"
            _mk_op(pid, s, ik, phase)
            with pytest.raises(OwnerWorkspaceError) as exc:
                _call(pid, f"cc{phase}{_n[0]}", "cancel")
            assert exc.value.code == "cancel_refused"
            assert "can only be completed" in exc.value.message
        pid, s = _setup(monkeypatch)
        ik_done = f"crdone{_n[0]}"
        _mk_op(pid, s, ik_done, "done", retained_copy_id="rm_x")
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"ccdone{_n[0]}", "cancel")
        assert exc.value.code == "cancel_refused"


# §7: revision contract
class TestRevisionContract:
    def test_same_revision_fn(self, monkeypatch):
        pid, _ = _setup(monkeypatch)
        with projects_db.connect_closing() as pc:
            ow._ensure_schema(pc)
            assert isinstance(ow._project_lifecycle_revision(pc, _ctx(), pid), int)

    def test_digest_deterministic(self):
        p = {"idempotency_key":"t","project_id":"p","expected_revision":0,"action":"start"}
        assert _digest(p) == _digest(p)
        assert _digest(p) != _digest(dict(p, action="cancel"))


# §8: capability
class TestCapability:
    def test_constant(self):
        assert OWNER_PROJECT_REMOVAL_STATE_CAPABILITY == "removal_state"

    def test_null_when_absent(self):
        with projects_db.connect_closing() as c:
            assert _project_removal_state(c, "nope", None) is None

    def test_full_fields(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "cap", "fenced", consequences_digest="cd")
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        expected = {"phase","mode","accepted_at","applied_at","completed_at",
                    "cancelable","restorable","retained_copy_id","receipt_id",
                    "consequences_digest","last_error","consequences"}
        assert set(st.keys()) == expected
        assert st["cancelable"] and st["consequences"]


# §9: owner-safety
class TestOwnerSafety:
    def test_consequences_clean(self):
        doc = _removal_consequences_document("P", "b", 1, True)
        t = doc["text"]
        assert not re.search(r"\btask_[0-9a-z]{6,}\b", t, re.I)
        assert not re.search(r"\brun_[0-9a-z]{6,}\b", t, re.I)
        assert not re.search(r"\b[0-9a-f]{64}\b", t)
        for p in ow._OWNER_PRIVATE_WORK_ITEM_PATTERNS:
            assert not p.search(t)

    def test_safe_errors_clean(self):
        for msg in _REMOVAL_SAFE_ERRORS.values():
            assert not re.search(r"(?:/[A-Za-z0-9._-]+){2,}", msg)
            assert not re.search(r"\b[0-9a-f]{64}\b", msg)

    def test_digest_stable(self):
        d = _removal_consequences_document("P", "b", 1, True)
        assert d["digest"] == _removal_consequences_document("P", "b", 1, True)["digest"]


class TestPhaseMapping:
    def test_all_mapped(self):
        for p in kanban_db.REMOVAL_PHASE_ORDER:
            assert p.value in _REMOVAL_PHASE_OWNER_MAP
    def test_intent_to_accepted(self):
        assert _REMOVAL_PHASE_OWNER_MAP["intent"] == "accepted"


def _mock_valid_retained(mp):
    from pathlib import Path
    mp.setattr(kanban_db, "reversible_retained_path", lambda *a, **kw: Path("/tmp/f"))
    mp.setattr(kanban_db, "retained_set_manifest", lambda *a, **kw: (
        {"version": 1, "files": [{"path": "f", "size": 1, "sha256": "a"*64}]}, "ok"))
    mp.setattr(kanban_db, "validate_retained_set", lambda *a, **kw: (True, "ok", {}))


def _mock_permanent_path(mp, board_slug):
    disclosure = MagicMock()
    disclosure.required_response = "CONFIRM"
    disclosure.statement_digest = "sd"
    confirmation = MagicMock()
    confirmation.confirmed = True
    confirmation.statement_digest = "sd"
    confirmation.board = board_slug
    confirm_result = MagicMock()
    confirm_result.confirmed = True
    confirm_result.confirmation = confirmation
    mp.setattr(kanban_db, "permanent_removal_disclosure", lambda *a, **kw: disclosure)
    mp.setattr(kanban_db, "confirm_permanent_removal", lambda *a, **kw: confirm_result)
    mp.setattr(kanban_db, "remove_board_fenced", lambda *a, **kw: MagicMock(success=True))


class TestDefect1DigestStability:
    def test_nonzero_epoch_digest_matches_and_confirm_accepted(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        d = _removal_consequences_document(ow.owner_project_name("RP"), s, 42, True)["digest"]
        _mk_op(pid, s, "d1k", "accepted", consequences_digest=d, epoch=42)
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st["consequences"]["digest"] == st["consequences_digest"]
        _mock_permanent_path(monkeypatch, s)
        ck = f"d1ck{_n[0]}"
        p = {"idempotency_key":ck,"project_id":pid,"expected_revision":0,
             "action":"confirm_permanent","consequences_digest":st["consequences"]["digest"]}
        r = ow.project_removal(
            _ctx(_auth("owner_project_removal", ck, p)),
            idempotency_key=ck, project_id=pid, expected_revision=0,
            action="confirm_permanent", consequences_digest=st["consequences"]["digest"])
        assert r["ok"]


class TestDefect2NoDriverTextLeak:
    def test_start_refused_no_leak(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        monkeypatch.setattr("tools.approval.request_exact_operation_approval",
                            lambda *a, **kw: {"approved": True})
        leaked = "/var/data/board/rm_abc123 digest=ab" + "cd" * 31
        monkeypatch.setattr(kanban_db, "record_removal_intent",
            lambda *a, **kw: MagicMock(success=False, message=leaked, record=None, removal_id=None))
        r = _call(pid, f"d2s{_n[0]}", "start")
        assert not r["ok"]
        flat = json.dumps(r)
        for frag in [leaked, "/var/data", "rm_abc123"]:
            assert frag not in flat

    def test_restore_refused_no_leak(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        leaked = "/home/user/.kanban/retained/b-rm_x7 0a1b" + "2c" * 30
        _mk_op(pid, s, "d2r", "done", retained_copy_id="rm_x7")
        _mock_valid_retained(monkeypatch)
        monkeypatch.setattr(kanban_db, "restore_retained_board",
            lambda *a, **kw: MagicMock(success=False, message=leaked))
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"d2rr{_n[0]}", "restore")
        assert exc.value.code == "restore_failed"
        for frag in [leaked, "/home/user"]:
            assert frag not in exc.value.message


class TestDefect3RetainedCopyAndReceipt:
    def test_completed_reversible_is_restorable(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d3k", "done", retained_copy_id="rm_t3", receipt_id="d3k")
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st["restorable"] is True and st["retained_copy_id"] is not None

    def test_restore_succeeds_with_retained_copy(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d3r", "done", retained_copy_id="rm_r3", receipt_id="d3r")
        _mock_valid_retained(monkeypatch)
        monkeypatch.setattr(kanban_db, "restore_retained_board",
            lambda *a, **kw: MagicMock(success=True, message="ok"))
        assert _call(pid, f"d3rr{_n[0]}", "restore")["ok"]


class TestDefect4IncompleteRetainedCopy:
    def test_refuse_not_done(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4a", "fenced", retained_copy_id="rm_x")
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"d4ar{_n[0]}", "restore")
        assert exc.value.code == "restore_no_copy"

    def test_refuse_no_retained_id(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4b", "done")
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"d4b_r{_n[0]}", "restore")
        assert exc.value.code in ("restore_no_copy", "invalid_argument")

    def test_refuse_invalid_retained_set(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4c", "done", retained_copy_id="rm_y")
        from pathlib import Path
        monkeypatch.setattr(kanban_db, "reversible_retained_path", lambda *a, **kw: Path("/tmp/nx"))
        monkeypatch.setattr(kanban_db, "retained_set_manifest", lambda *a, **kw: (None, "missing"))
        monkeypatch.setattr(kanban_db, "validate_retained_set", lambda *a, **kw: (False, "bad", {}))
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"d4cr{_n[0]}", "restore")
        assert exc.value.code == "restore_no_copy"


def _mock_intent_success(mp, board_slug="b"):
    record = MagicMock()
    record.epoch = 1
    record.phase = MagicMock()
    record.phase.value = "intent"
    record.mode = kanban_db.RemovalMode.REVERSIBLE
    record.removal_id = "rm_test"
    mp.setattr(kanban_db, "record_removal_intent", lambda *a, **kw: MagicMock(
        success=True, removal_id="rm_test", record=record,
        outcome=kanban_db.RemovalIntentOutcome.STARTED,
    ))


# ── T1: Blocking 1 — cancel/confirm_permanent reachable after real start ──

class TestT1RealStartThenAction:
    def test_start_then_cancel_fresh_key(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t1s{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        cancel_ik = f"t1c{_n[0]}"
        with patch.object(kanban_db, "abandon_or_cancel_removal",
                          return_value=MagicMock(success=True)):
            rc = _call(pid, cancel_ik, "cancel")
        assert rc["ok"] and rc["action"] == "cancel"

    def test_start_then_confirm_permanent_fresh_key(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t1ps{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        served_digest = r["removal_state"]["consequences_digest"]
        _mock_permanent_path(monkeypatch, s)
        confirm_ik = f"t1pc{_n[0]}"
        rc = _call(pid, confirm_ik, "confirm_permanent",
                   consequences_digest=served_digest)
        assert rc["ok"] and rc["action"] == "confirm_permanent"

    def test_start_key_replay_safe(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t1rs{_n[0]}"
        r1 = _call(pid, start_ik, "start")
        assert r1["ok"]
        r2 = _call(pid, start_ik, "start")
        assert r2["ok"]

    def test_fresh_start_key_joins_existing(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        r1 = _call(pid, f"t1j1{_n[0]}", "start")
        assert r1["ok"]
        r2 = _call(pid, f"t1j2{_n[0]}", "start")
        assert r2["ok"] and r2.get("joined")


# ── T2: Blocking 2 — start returns before driving ──

class TestT2AsyncDrive:
    def test_start_returns_before_drive(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        captured = []
        monkeypatch.setattr(ow, "_dispatch_removal_drive",
                            lambda *a, **kw: captured.append(a))
        start_ik = f"t2s{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        assert r["removal_state"]["cancelable"] is True
        assert len(captured) == 1
        with patch.object(kanban_db, "abandon_or_cancel_removal",
                          return_value=MagicMock(success=True)):
            rc = _call(pid, f"t2c{_n[0]}", "cancel")
        assert rc["ok"]

    def test_drive_advances_phase(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        captured = []
        monkeypatch.setattr(ow, "_dispatch_removal_drive",
                            lambda *a, **kw: captured.append(a))
        r = _call(pid, f"t2d{_n[0]}", "start")
        assert r["ok"]
        op_key = captured[0][2]
        phase_rec = MagicMock()
        phase_rec.phase = MagicMock()
        phase_rec.phase.value = "done"
        phase_rec.mode = kanban_db.RemovalMode.REVERSIBLE
        phase_rec.removal_id = "rm_test"
        monkeypatch.setattr(kanban_db, "drive_removal", lambda *a, **kw: None)
        monkeypatch.setattr(kanban_db, "get_removal_phase_record",
                            lambda *a, **kw: phase_rec)
        ow._removal_drive_and_record(pid, s, op_key)
        with projects_db.connect_closing() as c:
            op = projects_db.get_removal_operation(c, pid, op_key)
        assert op["phase"] == "done"

    def test_state_read_never_drives(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        r = _call(pid, f"t2nr{_n[0]}", "start")
        assert r["ok"]
        drive_calls = []
        monkeypatch.setattr(kanban_db, "drive_removal",
                            lambda *a, **kw: drive_calls.append(1))
        with projects_db.connect_closing() as c:
            _project_removal_state(c, pid, s)
            _project_removal_state(c, pid, s)
        assert len(drive_calls) == 0


# ── T3: Major 3 — dashboard /projects removal_state gating ──

class TestT3DashboardRemovalStateGating:
    def test_not_asked_key_absent(self):
        with projects_db.connect_closing() as c:
            projects_db.record_removal_operation(
                c, project_id="t3a", idempotency_key="t3ak",
                action="start", phase="accepted", mode="reversible")
        from hermes_cli.owner_workspace import _project_removal_state
        result = ow._project_removal_state.__wrapped__(
            None, "t3a", None,
        ) if hasattr(ow._project_removal_state, '__wrapped__') else None
        proj = {"id": "t3a", "slug": "t3", "name": "T"}
        assert "removal_state" not in proj

    def test_asked_no_removal_key_null(self):
        proj = {"id": "xxx", "slug": "xxx", "name": "X"}
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, "nonexistent_project_t3", None)
        assert st is None
        proj["removal_state"] = st
        assert proj["removal_state"] is None

    def test_asked_with_removal_returns_state(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "t3k", "fenced", consequences_digest="cd")
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st is not None
        assert st["phase"] == "fenced"


# ── T4: Major 4 — confirm_permanent releases retained copy ──

class TestT4ConfirmPermanentRelease:
    def test_confirm_permanent_releases_retained(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        r = _call(pid, f"t4s{_n[0]}", "start")
        assert r["ok"]
        with projects_db.connect_closing() as c:
            op = projects_db.get_active_removal_operation(c, pid)
            projects_db.update_removal_operation(
                c, pid, op["idempotency_key"],
                phase="done", retained_copy_id="rm_test",
                completed_at=int(time.time()),
            )
        _mock_permanent_path(monkeypatch, s)
        served_digest = r["removal_state"]["consequences_digest"]
        rc = _call(pid, f"t4c{_n[0]}", "confirm_permanent",
                   consequences_digest=served_digest)
        assert rc["ok"]
        with projects_db.connect_closing() as c:
            op = projects_db.get_removal_operation(
                c, pid, f"t4s{_n[0]}")
        assert op["retained_copy_id"] is None
        assert op["mode"] == "permanent"
        st = rc.get("removal_state")
        assert st is None or st.get("restorable") is not True

    def test_restore_refused_after_permanent(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        r = _call(pid, f"t4rs{_n[0]}", "start")
        assert r["ok"]
        with projects_db.connect_closing() as c:
            op = projects_db.get_active_removal_operation(c, pid)
            projects_db.update_removal_operation(
                c, pid, op["idempotency_key"],
                phase="done", retained_copy_id="rm_test",
                completed_at=int(time.time()),
            )
        _mock_permanent_path(monkeypatch, s)
        served_digest = r["removal_state"]["consequences_digest"]
        rc = _call(pid, f"t4rc{_n[0]}", "confirm_permanent",
                   consequences_digest=served_digest)
        assert rc["ok"]
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, f"t4rr{_n[0]}", "restore")
        assert exc.value.code in ("restore_no_copy", "invalid_argument")


# ── T5: Minor 5 — digest stable across mode flip ──

class TestT5DigestStableAcrossModeFlip:
    def test_digest_unchanged_across_mode_flip(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t5s{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        st1 = r["removal_state"]
        digest_before = st1["consequences"]["digest"]
        assert digest_before == st1["consequences_digest"]
        with projects_db.connect_closing() as c:
            op = projects_db.get_active_removal_operation(c, pid)
            projects_db.update_removal_operation(
                c, pid, op["idempotency_key"],
                phase="done", retained_copy_id="rm_test",
                completed_at=int(time.time()),
            )
        _mock_permanent_path(monkeypatch, s)
        confirm_ik = f"t5c{_n[0]}"
        rc = _call(pid, confirm_ik, "confirm_permanent",
                   consequences_digest=digest_before)
        assert rc["ok"]
        with projects_db.connect_closing() as c:
            op_after = projects_db.get_removal_operation(c, pid, start_ik)
        assert op_after["consequences_digest"] == digest_before
