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
               lambda **kw: {"approved": True})
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
        p = {"idempotency_key":"md","project_id":pid,"expected_revision":0,
             "action":"confirm_permanent","consequences_digest":"wrong"}
        with pytest.raises(OwnerWorkspaceError) as e:
            ow.project_removal(_ctx(_auth("owner_project_removal","md",p)),
                idempotency_key="md", project_id=pid, expected_revision=0,
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
            with patch.object(kanban_db, "abandon_or_cancel_removal",
                              return_value=MagicMock(success=True)):
                r = _call(pid, ik, "cancel")
            assert r["ok"], f"cancel at {phase} failed"

    def test_at_or_after_applied_refused(self, monkeypatch):
        for phase in ("applied", "swept", "done"):
            pid, s = _setup(monkeypatch)
            ik = f"cr{phase}{_n[0]}"
            _mk_op(pid, s, ik, phase)
            with pytest.raises(OwnerWorkspaceError) as exc:
                _call(pid, ik, "cancel")
            assert exc.value.code == "cancel_refused"
            assert "can only be completed" in exc.value.message


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


class TestDefect1DigestStability:
    def test_nonzero_epoch_digest_matches_and_confirm_accepted(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        d = _removal_consequences_document(ow.owner_project_name("RP"), s, 42, True)["digest"]
        _mk_op(pid, s, "d1k", "accepted", consequences_digest=d, epoch=42)
        with projects_db.connect_closing() as c:
            st = _project_removal_state(c, pid, s)
        assert st["consequences"]["digest"] == st["consequences_digest"]
        p = {"idempotency_key":"d1k","project_id":pid,"expected_revision":0,
             "action":"confirm_permanent","consequences_digest":st["consequences"]["digest"]}
        r = ow.project_removal(
            _ctx(_auth("owner_project_removal", "d1k", p)),
            idempotency_key="d1k", project_id=pid, expected_revision=0,
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
            _call(pid, "d2r", "restore")
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
        assert _call(pid, "d3r", "restore")["ok"]


class TestDefect4IncompleteRetainedCopy:
    def test_refuse_not_done(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4a", "fenced", retained_copy_id="rm_x")
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, "d4a", "restore")
        assert exc.value.code == "restore_no_copy"

    def test_refuse_no_retained_id(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4b", "done")
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, "d4b", "restore")
        assert exc.value.code == "restore_no_copy"

    def test_refuse_invalid_retained_set(self, monkeypatch):
        pid, s = _setup(monkeypatch)
        _mk_op(pid, s, "d4c", "done", retained_copy_id="rm_y")
        from pathlib import Path
        monkeypatch.setattr(kanban_db, "reversible_retained_path", lambda *a, **kw: Path("/tmp/nx"))
        monkeypatch.setattr(kanban_db, "retained_set_manifest", lambda *a, **kw: (None, "missing"))
        monkeypatch.setattr(kanban_db, "validate_retained_set", lambda *a, **kw: (False, "bad", {}))
        with pytest.raises(OwnerWorkspaceError) as exc:
            _call(pid, "d4c", "restore")
        assert exc.value.code == "restore_no_copy"
