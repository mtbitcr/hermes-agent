"""An Automation can carry one optional link to a Project, held as its slug."""

import json

import pytest
from fastapi import HTTPException

from cron import jobs as cron_jobs
from hermes_cli import owner_workspace as ow
from hermes_cli import projects_db, web_server
from hermes_cli.web_models import CronJobCreate
from tools import approval

SESSION = "run_cron_project"


@pytest.fixture(autouse=True)
def _scheduler_registration(monkeypatch):
    class _Provider:
        def register_job(self, job):
            return None

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", lambda: _Provider()
    )


def _make_project(name: str = "Cron Link Project", key: str = "cron-link-1") -> dict:
    ctx = ow.OwnerContext(actor="default", profile="default", session=SESSION)
    approval.register_gateway_notify(SESSION, lambda data: None)
    import threading
    import time

    def approve():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with approval._lock:
                queue = approval._gateway_queues.get(SESSION)
                aid = queue[0].approval_id if queue else None
            if aid:
                approval.resolve_gateway_approval(SESSION, "once", approval_id=aid)
                return
            time.sleep(0.005)

    thread = threading.Thread(target=approve)
    thread.start()
    try:
        result = ow.bootstrap(ctx, idempotency_key=key, name=name, description="d")
    finally:
        thread.join()
        approval.unregister_gateway_notify(SESSION)
    assert result["ok"] is True
    with projects_db.connect_closing() as conn:
        project = projects_db.get_project(conn, result["project_id"])
    return {"id": result["project_id"], "slug": project.slug}


def _stored_jobs() -> list:
    path = cron_jobs._current_cron_store().jobs_file
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["jobs"] if isinstance(data, dict) else data


def _create(**fields):
    return web_server._create_cron_job_sync(
        CronJobCreate(prompt="say hi", schedule="every 1h", name="linked", **fields)
    )


def test_valid_slug_is_listed_only_when_capability_requested():
    project = _make_project()

    created = _create(project=project["slug"])
    assert created["project"] == project["slug"]
    assert _stored_jobs()[0]["project"] == project["slug"]

    with_cap = web_server._list_cron_jobs_sync("default", include_project=True)
    assert [job["project"] for job in with_cap] == [project["slug"]]

    without_cap = web_server._list_cron_jobs_sync("default")
    assert len(without_cap) == 1
    assert "project" not in without_cap[0]


def test_unknown_slug_is_refused_plainly_and_nothing_is_stored():
    _make_project()

    with pytest.raises(HTTPException) as excinfo:
        _create(project="no-such-project")

    assert 400 <= excinfo.value.status_code < 500
    detail = str(excinfo.value.detail)
    assert "project" in detail.lower()
    for leak in ("Traceback", "/", "Error(", "sqlite", ".db"):
        assert leak not in detail
    assert _stored_jobs() == []


def test_unknown_slug_with_no_projects_at_all_is_refused():
    with pytest.raises(HTTPException) as excinfo:
        _create(project="anything")
    assert 400 <= excinfo.value.status_code < 500
    assert _stored_jobs() == []


def test_archived_project_slug_is_refused():
    project = _make_project()
    with projects_db.connect_closing() as conn:
        assert projects_db.archive_project(conn, project["id"]) is True

    with pytest.raises(HTTPException) as excinfo:
        _create(project=project["slug"])

    assert 400 <= excinfo.value.status_code < 500
    assert _stored_jobs() == []


def test_job_created_without_project_is_unchanged():
    created = _create()
    assert "project" not in created

    stored = _stored_jobs()
    assert len(stored) == 1
    assert "project" not in stored[0]

    from starlette.testclient import TestClient

    listed = web_server._list_cron_jobs_sync("default")
    assert len(listed) == 1
    assert "project" not in listed[0]

    client = TestClient(web_server.app)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    try:
        routed = client.get(
            "/api/cron/jobs?profile=default&capabilities=automation_project",
            headers=headers,
        )
    finally:
        client.close()
    assert routed.status_code == 200
    assert [job["id"] for job in routed.json()] == [listed[0]["id"]]
    assert all("project" not in job for job in routed.json())


def test_list_route_reads_the_capabilities_query_parameter():
    from starlette.testclient import TestClient

    project = _make_project()
    _create(project=project["slug"])
    client = TestClient(web_server.app)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    try:
        plain = client.get("/api/cron/jobs?profile=default", headers=headers)
        asked = client.get(
            "/api/cron/jobs?profile=default&capabilities=other,automation_project",
            headers=headers,
        )
    finally:
        client.close()

    assert plain.status_code == 200
    assert all("project" not in job for job in plain.json())
    assert [job["project"] for job in asked.json()] == [project["slug"]]
