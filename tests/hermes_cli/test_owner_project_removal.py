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

def _drove_ok(*a, **kw):
    """What the real driver returns when it reaches Done."""
    return kanban_db.FencedRemovalResult(True, "removal complete")


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

    def test_confirm_permanent_denied_leaves_phase_and_mode_unchanged(
        self, monkeypatch,
    ):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t1pd_s{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        served_digest = r["removal_state"]["consequences_digest"]

        continue_mock = MagicMock()
        disclosure_mock = MagicMock()
        confirm_mock = MagicMock()
        fence_mock = MagicMock()
        update_mock = MagicMock()
        monkeypatch.setattr(
            kanban_db, "continue_archived_removal_as_permanent", continue_mock)
        monkeypatch.setattr(
            kanban_db, "permanent_removal_disclosure", disclosure_mock)
        monkeypatch.setattr(kanban_db, "confirm_permanent_removal", confirm_mock)
        monkeypatch.setattr(kanban_db, "remove_board_fenced", fence_mock)
        monkeypatch.setattr(projects_db, "update_removal_operation", update_mock)
        monkeypatch.setattr(
            "tools.approval.request_exact_operation_approval",
            lambda *a, **kw: {"approved": False, "reason": "owner declined"},
        )

        confirm_ik = f"t1pd_c{_n[0]}"
        rc = _call(
            pid, confirm_ik, "confirm_permanent", consequences_digest=served_digest,
        )

        assert rc["ok"] is False
        assert rc["error"] == "confirmation_denied"
        assert rc["reason"] == "owner declined"
        continue_mock.assert_not_called()
        disclosure_mock.assert_not_called()
        confirm_mock.assert_not_called()
        fence_mock.assert_not_called()
        update_mock.assert_not_called()

        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st["phase"] == "accepted"
        assert st["mode"] == "recoverable"

    def test_confirm_permanent_approved_continues_as_before(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = f"t1pa_s{_n[0]}"
        r = _call(pid, start_ik, "start")
        assert r["ok"]
        served_digest = r["removal_state"]["consequences_digest"]

        _mock_permanent_path(monkeypatch, s)
        confirm_ik = f"t1pa_c{_n[0]}"
        rc = _call(
            pid, confirm_ik, "confirm_permanent", consequences_digest=served_digest,
        )

        assert rc["ok"] is True
        assert rc["action"] == "confirm_permanent"
        assert rc["removal_state"]["mode"] == "permanent"

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
        monkeypatch.setattr(kanban_db, "drive_removal", _drove_ok)
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
    def test_not_asked_key_absent(self, monkeypatch):
        """Builder without include_removal_state must NOT include the key."""
        from plugins.kanban.dashboard import plugin_api as papi
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "t3ak2", "accepted", consequences_digest="cd")
        monkeypatch.setattr(
            papi, "_workspace_scope_snapshot",
            lambda: (MagicMock(id=pid, slug=s, name="T", board_slug=s), None),
        )
        result = papi._workspace_projects_response(include_removal_state=False)
        assert result is not None
        proj = result["projects"][0]
        assert "removal_state" not in proj

    def test_asked_includes_key(self, monkeypatch):
        """Builder with include_removal_state=True must include the key."""
        from plugins.kanban.dashboard import plugin_api as papi
        pid, s = _setup(monkeypatch)
        monkeypatch.setattr(
            papi, "_workspace_scope_snapshot",
            lambda: (MagicMock(id=pid, slug=s, name="T", board_slug=s), None),
        )
        result = papi._workspace_projects_response(include_removal_state=True)
        assert result is not None
        proj = result["projects"][0]
        assert "removal_state" in proj
        assert proj["removal_state"] is None

    def test_asked_with_removal_returns_state(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "t3k", "fenced", consequences_digest="cd")
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st is not None
        assert st["phase"] == "fenced"

    def test_route_no_query_no_removal_state_key(self, monkeypatch, tmp_path):
        """GET /projects with no query string must omit removal_state."""
        from pathlib import Path
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from hermes_cli import kanban_db as kb
        from hermes_cli.dashboard_auth import clear_providers, register_provider
        from hermes_cli.dashboard_auth import token_auth
        from plugins.dashboard_auth.raphael_workspace import (
            BOARD, PROJECT, WorkspaceReadTokenProvider, token_store,
        )
        from plugins.kanban.dashboard import plugin_api as papi

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        kb._INITIALIZED_PATHS.clear()

        repo = tmp_path / "workspace-repo"
        repo.mkdir()
        with projects_db.connect_closing() as conn:
            project_id = projects_db.create_project(
                conn, name="Raphael Workspace", primary_path=str(repo))
        kb.create_board(BOARD, name="Raphael Workspace", project_id=project_id)
        kb.init_db(board=BOARD)

        token_dir = home / "workspace-token"
        token_dir.mkdir(mode=0o700)
        token_path = token_dir / "bearer"
        token_store.issue(out_path=token_path)
        bearer = token_path.read_text(encoding="utf-8").strip()

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(WorkspaceReadTokenProvider())
        papi._register_workspace_machine_routes()

        app = FastAPI()
        app.include_router(papi.router, prefix="/api/plugins/kanban")

        @app.middleware("http")
        async def machine_auth(request, call_next):
            return await token_auth.token_auth_middleware(request, call_next)

        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {bearer}",
                       "X-Forwarded-For": "203.0.113.10"}
            resp = client.get("/api/plugins/kanban/projects", headers=headers)
            assert resp.status_code == 200, f"expected 200 got {resp.status_code}"
            proj = resp.json()["projects"][0]
            assert "removal_state" not in proj

        clear_providers()
        token_auth.clear_token_routes()
        kb._INITIALIZED_PATHS.clear()


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


# ── T6: Blocking 2 — resume and background seam ──

_REAL_DISPATCH = ow._dispatch_removal_drive


class TestT6ResumeAndBackgroundSeam:
    def test_resume_drives_interrupted_operation(self, monkeypatch):
        """Reviewer reproduction: a real start whose drive is abandoned
        mid-phase is resumed, after a simulated process restart, strictly
        from the durable phase record -- reaching a terminal phase with no
        new owner request."""
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)

        # -- process 1: a real start whose background drive never completes.
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        start_ik = "t6s%d" % _n[0]
        r = _call(pid, start_ik, "start")
        assert r["ok"]

        # The drive reached an intermediate phase and then died there.
        with projects_db.connect_closing() as c:
            projects_db.update_removal_operation(c, pid, start_ik, phase="carried")
            mid = projects_db.get_removal_operation(c, pid, start_ik)
        assert mid["phase"] == "carried"

        # -- process 2 (restart): only the resume entry point runs. No owner call.
        driven = []

        def _fake_drive(board, *a, **kw):
            driven.append(board)
            return _drove_ok(board)

        phase_rec = MagicMock()
        phase_rec.phase = MagicMock()
        phase_rec.phase.value = "done"
        phase_rec.mode = kanban_db.RemovalMode.REVERSIBLE
        phase_rec.removal_id = "rm_test"
        monkeypatch.setattr(kanban_db, "drive_removal", _fake_drive)
        monkeypatch.setattr(kanban_db, "get_removal_phase_record",
                            lambda *a, **kw: phase_rec)
        # the production dispatch seam, restored for the restart half
        monkeypatch.setattr(ow, "_dispatch_removal_drive", _REAL_DISPATCH)

        ow.resume_removal_operations()
        for t in list(ow._removal_background_threads):
            t.join(timeout=20)

        # The resumed drive ran against the board named by the durable record ...
        assert driven == [s]
        # ... and the durable operation advanced to terminal on its own.
        with projects_db.connect_closing() as c:
            op = projects_db.get_removal_operation(c, pid, start_ik)
        assert op["phase"] == "done"

    def test_resume_skips_terminal_operations(self, monkeypatch):
        """Operations at terminal phases must NOT be re-dispatched."""
        pid, s = _setup(monkeypatch)
        for phase in ("done", "cancelled", "failed", "restored"):
            _mk_op(pid, s, f"t6t_{phase}_{_n[0]}", phase)

        dispatched = []
        monkeypatch.setattr(ow, "_dispatch_removal_drive",
                            lambda bs, pid_, ok: dispatched.append(pid_))
        ow.resume_removal_operations()
        assert pid not in dispatched

    def test_status_read_does_not_resume(self, monkeypatch):
        """Reading removal_state must never dispatch a drive."""
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "t6nr", "fenced")
        drive_calls = []
        monkeypatch.setattr(kanban_db, "drive_removal",
                            lambda *a, **kw: drive_calls.append(1))
        with projects_db.connect_closing() as c:
            _project_removal_state(c, pid, s)
        assert len(drive_calls) == 0

    def test_drive_uses_tracked_background(self, monkeypatch):
        """The dispatch mechanism must track tasks/threads, not fire-and-forget."""
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "t6bg", "accepted")
        monkeypatch.setattr(kanban_db, "drive_removal", _drove_ok)
        phase_rec = MagicMock()
        phase_rec.phase = MagicMock()
        phase_rec.phase.value = "done"
        phase_rec.mode = kanban_db.RemovalMode.REVERSIBLE
        phase_rec.removal_id = "rm_test"
        monkeypatch.setattr(kanban_db, "get_removal_phase_record",
                            lambda *a, **kw: phase_rec)
        ow._dispatch_removal_drive(s, pid, "t6bg")
        for t in list(ow._removal_background_threads):
            t.join(timeout=10)
        with projects_db.connect_closing() as c:
            op = projects_db.get_removal_operation(c, pid, "t6bg")
        assert op is not None
        assert op["phase"] == "done"


def _real_board(monkeypatch):
    """A Project whose board really exists and is really fenced."""
    from tests.hermes_cli._kanban_fence_support import create_fenced_board
    pid, s = _setup(monkeypatch)
    create_fenced_board(s)
    return pid, s


def _start_and_join(pid, tag):
    """The shipped owner start, driven to a standstill on its own thread."""
    assert _call(pid, f"{tag}{_n[0]}", "start")["ok"]
    for t in list(ow._removal_background_threads):
        t.join(timeout=60)


class TestFinding1OwnerModeVocabulary:
    def test_owner_surface_says_recoverable_then_permanent(self, monkeypatch):
        """Owner-facing mode is 'recoverable' or 'permanent'; the kernel's
        own word stays 'reversible' underneath."""
        pid, s = _setup(monkeypatch)
        _mock_intent_success(monkeypatch, s)
        monkeypatch.setattr(ow, "_dispatch_removal_drive", lambda *a, **kw: None)
        started = _call(pid, f"f1s{_n[0]}", "start")
        assert started["removal_state"]["mode"] == "recoverable"
        with projects_db.connect_closing() as c:
            assert _project_removal_state(c, pid, s)["mode"] == "recoverable"
            assert projects_db.get_active_removal_operation(c, pid)["mode"] == (
                "reversible"
            )
        _mock_permanent_path(monkeypatch, s)
        confirmed = _call(
            pid, f"f1c{_n[0]}", "confirm_permanent",
            consequences_digest=started["removal_state"]["consequences_digest"],
        )
        assert confirmed["removal_state"]["mode"] == "permanent"
        with projects_db.connect_closing() as c:
            assert _project_removal_state(c, pid, s)["mode"] == "permanent"


class TestMinorAppliedAtStamped:
    def test_completed_removal_reports_when_content_was_applied(self, monkeypatch):
        """A real drive passes Applied on its way to Done; the owner is
        still owed the instant content was applied."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "mas")
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st["phase"] == "done"
        assert st["completed_at"] is not None
        assert st["applied_at"] is not None
        assert st["applied_at"] <= st["completed_at"]


class TestFinding4DeletedSnapshotPresentation:
    def test_completed_removal_presents_deleted_with_restore(self, monkeypatch):
        """The board is gone by design: the snapshot comes from the
        projects-list record and the retained-copy facts."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f4s")
        assert not kanban_db.board_exists(s)
        snapshot = ow.read_project_snapshot(_ctx(), s)
        assert snapshot["project"]["slug"] == s
        assert snapshot["steward"]["execution"]["state"] == "deleted"
        assert snapshot["removal_state"]["restorable"] is True
        assert snapshot["removal_state"]["mode"] == "recoverable"
        assert ow.project_steward_snapshot(project_id=pid)["execution"]["state"] == (
            "deleted"
        )


class TestFinding5RestoreClearsArchivedMarker:
    def test_restored_project_is_visible_and_paused(self, monkeypatch):
        """The archived marker rides inside the retained copy; a restore
        that leaves it there hides the Project forever."""
        from tests.hermes_cli._kanban_fence_support import ready_task

        pid, s = _real_board(monkeypatch)
        conn = kanban_db.connect(board=s)
        try:
            task_id = ready_task(conn, title="work")
            conn.execute(
                "UPDATE tasks SET project_id = ? WHERE id = ?", (pid, task_id),
            )
            conn.commit()
        finally:
            conn.close()
        _start_and_join(pid, "f5s")
        assert _call(pid, f"f5r{_n[0]}", "restore")["ok"]
        metadata = kanban_db.read_board_metadata(s)
        assert metadata["archived"] is False
        assert kanban_db.RETAINED_ARCHIVED_MARKER_KEY not in metadata
        snapshot = ow.read_project_snapshot(_ctx(), s)
        assert snapshot["project"]["slug"] == s
        assert snapshot["steward"]["execution"]["state"] == "paused"
        assert snapshot["steward"]["execution"]["paused"] is True


class TestFinding6PermanentContinuesFromArchived:
    def test_confirm_permanent_after_a_completed_recoverable_removal(
        self, monkeypatch,
    ):
        """The contract offers confirm_permanent on a completed recoverable
        operation with the served digest; the kernel must agree too."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f6s")
        with projects_db.connect_closing() as c:
            served = _project_removal_state(c, pid, s)
        assert served["phase"] == "done" and served["restorable"] is True
        confirmed = _call(
            pid, f"f6c{_n[0]}", "confirm_permanent",
            consequences_digest=served["consequences_digest"],
        )
        assert confirmed["ok"], confirmed
        assert confirmed["removal_state"]["mode"] == "permanent"
        assert confirmed["removal_state"]["restorable"] is False
        # The kernel agrees: hard-removed, no retained copy left.
        assert kanban_db.get_register_entry(s).lifecycle is (
            kanban_db.BoardLifecycle.HARD_REMOVED
        )
        assert not kanban_db.reversible_retained_path(
            s, served["retained_copy_id"],
        ).exists()

    def test_a_refused_permanent_withdraws_the_restore_it_can_no_longer_honour(
        self, monkeypatch,
    ):
        """continue_archived_removal_as_permanent reinstalls and discards
        the retained copy before remove_board_fenced ever runs; a refusal
        there must not leave the owner offered a Restore that no longer
        has anything to restore."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f6rs")
        with projects_db.connect_closing() as c:
            served = _project_removal_state(c, pid, s)
        assert served["phase"] == "done" and served["restorable"] is True
        retained_copy_id = served["retained_copy_id"]
        assert retained_copy_id
        monkeypatch.setattr(
            kanban_db, "remove_board_fenced",
            lambda *a, **kw: kanban_db.FencedRemovalResult(
                False, "refused for the test",
            ),
        )
        confirmed = _call(
            pid, f"f6rc{_n[0]}", "confirm_permanent",
            consequences_digest=served["consequences_digest"],
        )
        assert confirmed["ok"] is False
        with projects_db.connect_closing() as c:
            after = _project_removal_state(c, pid, s)
        assert after["restorable"] is False
        assert after["retained_copy_id"] is None
        assert not kanban_db.reversible_retained_path(s, retained_copy_id).exists()
        assert kanban_db.get_register_entry(s).lifecycle is (
            kanban_db.BoardLifecycle.LIVE
        )

    def test_confirmation_refused_withdraws_retained_copy_when_discarded(
        self, monkeypatch,
    ):
        """When confirm_permanent_removal refuses confirmation, the retained
        copy has already been discarded by continue_archived_removal_as_permanent;
        the handle must be withdrawn to avoid a false Restore offer."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f6conf")
        with projects_db.connect_closing() as c:
            served = _project_removal_state(c, pid, s)
        assert served["phase"] == "done" and served["restorable"] is True
        retained_copy_id = served["retained_copy_id"]
        assert retained_copy_id
        # Force confirmation refusal after continue_archived has already discarded.
        conf_result = MagicMock()
        conf_result.confirmed = False
        monkeypatch.setattr(
            kanban_db, "confirm_permanent_removal", lambda *a, **kw: conf_result,
        )
        confirmed = _call(
            pid, f"f6confr{_n[0]}", "confirm_permanent",
            consequences_digest=served["consequences_digest"],
        )
        assert confirmed["ok"] is False
        with projects_db.connect_closing() as c:
            after = _project_removal_state(c, pid, s)
        # Handle withdrawn because the copy is gone.
        assert after["restorable"] is False
        assert after["retained_copy_id"] is None
        assert not kanban_db.reversible_retained_path(s, retained_copy_id).exists()

    def test_exception_after_discard_withdraws_retained_copy(self, monkeypatch):
        """If an exception is raised AFTER continue_archived_removal_as_permanent
        has discarded the retained copy, the generic exception handler must
        withdraw the handle."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f6exc")
        with projects_db.connect_closing() as c:
            served = _project_removal_state(c, pid, s)
        assert served["phase"] == "done" and served["restorable"] is True
        retained_copy_id = served["retained_copy_id"]
        assert retained_copy_id
        # Let continue_archived succeed (discarding the copy), then raise.
        orig_confirm = kanban_db.confirm_permanent_removal
        def _raise_after_continue(*a, **kw):
            raise RuntimeError("forced exception after discard")
        monkeypatch.setattr(
            kanban_db, "confirm_permanent_removal", _raise_after_continue,
        )
        confirmed = _call(
            pid, f"f6excr{_n[0]}", "confirm_permanent",
            consequences_digest=served["consequences_digest"],
        )
        assert confirmed["ok"] is False
        with projects_db.connect_closing() as c:
            after = _project_removal_state(c, pid, s)
        # Handle withdrawn because the copy is gone.
        assert after["restorable"] is False
        assert after["retained_copy_id"] is None
        assert not kanban_db.reversible_retained_path(s, retained_copy_id).exists()

    def test_pre_discard_refusal_keeps_restorable_offer(self, monkeypatch):
        """If continue_archived_removal_as_permanent refuses WITHOUT discarding
        the retained copy, the exception handler must KEEP the handle and the
        Restore offer — this is a pre-discard refusal."""
        pid, s = _real_board(monkeypatch)
        _start_and_join(pid, "f6pre")
        with projects_db.connect_closing() as c:
            served = _project_removal_state(c, pid, s)
        assert served["phase"] == "done" and served["restorable"] is True
        retained_copy_id = served["retained_copy_id"]
        assert retained_copy_id
        retained_path = kanban_db.reversible_retained_path(s, retained_copy_id)
        assert retained_path.exists()
        # Force continue_archived to fail WITHOUT discarding the copy.
        monkeypatch.setattr(
            kanban_db, "continue_archived_removal_as_permanent",
            lambda *a, **kw: kanban_db.RestoreResult(
                False, "pre-discard refusal for test",
            ),
        )
        confirmed = _call(
            pid, f"f6prer{_n[0]}", "confirm_permanent",
            consequences_digest=served["consequences_digest"],
        )
        assert confirmed["ok"] is False
        with projects_db.connect_closing() as c:
            after = _project_removal_state(c, pid, s)
        # Handle KEPT because the copy is still present.
        assert after["restorable"] is True
        assert after["retained_copy_id"] == retained_copy_id
        assert retained_path.exists()
