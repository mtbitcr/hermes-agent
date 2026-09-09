"""Only an explicitly empty findings array is a clean verdict.

``submit_review_findings`` approves a task from the review lane when the
reviewer reports zero findings. That makes ``findings`` the single most
consequential field on the board's whole review surface: anything that
silently becomes "no findings" is an automatic APPROVAL of unreviewed work.

Two coercions used to do exactly that — an ``Iterable`` check that a ``dict``
or ``set`` satisfies, and ``raw.get("findings") or []`` in the document
parser, which turns ``{}``, ``0``, ``""`` and ``None`` into a clean pass. A
reviewer (or a model filling in a JSON template) that emits ``"findings": {}``
must be REFUSED, not thanked.

Driven through the real ``hermes kanban review-findings`` dispatch, the real
dashboard endpoint and the real document parser.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _finding(**overrides) -> dict:
    base = {
        "severity": "blocking",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "unchecked index",
        "impact": "IndexError on an empty batch",
        "smallest_fix": "guard the empty case",
        "candidate_digest": "digest-1",
    }
    base.update(overrides)
    return base


def _hand_off_to_review(conn, title="reviewed task", *, reviewer="reviewer"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    implementation = kb.claim_task(conn, tid, claimer="worker:1")
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", reviewer=reviewer,
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, tid, claimer=f"{reviewer}:1")
    assert review is not None
    return tid, review


def _cli(argv: list[str]) -> tuple[int, str]:
    """Run one ``hermes kanban …`` command through the real dispatch.

    Same argparse tree and same ``kanban_command`` the ``hermes`` entry point
    uses, so the exit code is the one a shell would see.
    """
    wrap = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = wrap.add_subparsers(dest="cmd")
    kc.build_parser(sub)
    args = wrap.parse_args(["kanban", *argv])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = kc.kanban_command(args)
    return code, (out.getvalue() + err.getvalue()).strip()


def _submit_via_cli(tmp_path: Path, tid: str, document: dict) -> tuple[int, str]:
    doc_path = tmp_path / f"findings-{tid}.json"
    doc_path.write_text(json.dumps(document), encoding="utf-8")
    return _cli(["review-findings", tid, "--file", str(doc_path)])


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_empty_findings_array_is_the_only_clean_verdict(
    kanban_home, tmp_path, monkeypatch,
):
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="clean pass")
        run_id = review.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    code, output = _submit_via_cli(
        tmp_path, tid, {"candidate_digest": "digest-1", "findings": []},
    )
    assert code == 0, output
    assert "review passed" in output
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"


@pytest.mark.parametrize(
    "label,document",
    [
        ("empty_object", {"candidate_digest": "digest-1", "findings": {}}),
        ("zero", {"candidate_digest": "digest-1", "findings": 0}),
        ("empty_string", {"candidate_digest": "digest-1", "findings": ""}),
        ("null", {"candidate_digest": "digest-1", "findings": None}),
        ("missing_key", {"candidate_digest": "digest-1"}),
        ("false", {"candidate_digest": "digest-1", "findings": False}),
        ("object_of_findings", {
            "candidate_digest": "digest-1",
            "findings": {"one": _finding()},
        }),
    ],
)
def test_cli_refuses_a_findings_value_that_is_not_a_list(
    kanban_home, tmp_path, monkeypatch, label, document,
):
    """None of these is a verdict, so none of them may approve the task."""
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title=f"refuse {label}")
        run_id = review.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    code, output = _submit_via_cli(tmp_path, tid, document)
    assert code != 0, (
        f"the CLI accepted findings={label} and reported: {output}"
    )
    assert "findings" in output.lower()

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status != "done", (
            f"findings={label} was coerced into a clean verdict and APPROVED "
            "the task"
        )
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert kb.list_attachments(conn, tid) == []
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND kind IN ('completed', 'review_findings_delivered')",
            (tid,),
        ).fetchone()["n"] == 0


def test_cli_still_hands_back_a_real_findings_list(
    kanban_home, tmp_path, monkeypatch,
):
    """Regression guard: the ordinary handback is untouched."""
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="real handback")
        run_id = review.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    code, output = _submit_via_cli(
        tmp_path, tid,
        {"candidate_digest": "digest-1", "findings": [_finding()]},
    )
    assert code == 0, output
    assert "handed back" in output
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert len(kb.list_attachments(conn, tid)) == 1


# ---------------------------------------------------------------------------
# The kernel validator and the stored-document parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [{}, 0, "", None, False, {"one": "two"}, set(), ("a",), 3],
)
def test_validator_requires_a_concrete_list(value):
    with pytest.raises(kb.ReviewFindingsError, match="list"):
        kb.build_review_findings_document(value, candidate_digest="digest-1")


@pytest.mark.parametrize("value", [{}, 0, "", None, False])
def test_parser_refuses_a_falsy_findings_value(value):
    """A stored document whose ``findings`` is falsy is malformed, not clean.

    The parser is what reads a handback document back off the attachment, so
    coercing a falsy value to ``[]`` there presented a corrupt document as a
    reviewer's clean verdict.
    """
    document = {
        "schema_version": kb.REVIEW_FINDINGS_SCHEMA_VERSION,
        "candidate_digest": "digest-1",
        "findings": value,
    }
    with pytest.raises(kb.ReviewFindingsError, match="list"):
        kb.parse_review_findings_document(document)


def test_parser_refuses_a_document_with_no_findings_key():
    with pytest.raises(kb.ReviewFindingsError, match="list"):
        kb.parse_review_findings_document({
            "schema_version": kb.REVIEW_FINDINGS_SCHEMA_VERSION,
            "candidate_digest": "digest-1",
        })


def test_parser_round_trips_a_valid_document():
    document = kb.build_review_findings_document(
        [_finding()], candidate_digest="digest-1",
    )
    assert kb.parse_review_findings_document(
        json.loads(json.dumps(document))
    ) == document
    empty = kb.build_review_findings_document([], candidate_digest="digest-1")
    assert kb.parse_review_findings_document(
        json.loads(json.dumps(empty))
    )["findings"] == []


# ---------------------------------------------------------------------------
# The dashboard endpoint is the same verdict boundary
# ---------------------------------------------------------------------------


def _dashboard_client(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_schema_boundary_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = fastapi.FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def test_dashboard_requires_an_explicit_findings_array(kanban_home, tmp_path):
    client = _dashboard_client(tmp_path)
    with kb.connect() as conn:
        tid, review = _hand_off_to_review(conn, title="dashboard omitted")
        run_id = review.current_run_id

    omitted = client.post(
        f"/api/plugins/kanban/tasks/{tid}/review-findings",
        json={"candidate_digest": "digest-1", "expected_run_id": run_id},
    )
    assert omitted.status_code == 422, (
        "an omitted findings array was accepted as a clean verdict: "
        f"{omitted.status_code} {omitted.text}"
    )
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"

    approved = client.post(
        f"/api/plugins/kanban/tasks/{tid}/review-findings",
        json={
            "candidate_digest": "digest-1",
            "findings": [],
            "expected_run_id": run_id,
        },
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["outcome"] == "passed"
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
