"""GET /api/cron/schedule-preview: when a schedule would first run, before any job exists.

The platform clock is pinned where cron.jobs reads it (its ``_hermes_now``): one
instant, expressed in the zone the real clock picks for the active profile. The
preview, compute_next_run and job creation therefore all see the same time.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cron.jobs import ONESHOT_GRACE_SECONDS

PINNED = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
WEEKLY = "0 9 * * 1"
# A one-time start two days after the pinned clock, as wall-clock time with no offset.
ONE_TIME_WALL_CLOCK = (PINNED + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
ZONES = {"default": "America/New_York", "worker_alpha": "Asia/Tokyo"}
JOB = {"prompt": "Send the weekly owner summary", "name": "Weekly owner summary"}


@pytest.fixture()
def profile_homes(tmp_path, monkeypatch):
    """An isolated default home and one named profile, each in its own time zone."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    homes = {"default": default_home, "worker_alpha": profiles_root / "worker_alpha"}
    for name, home in homes.items():
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            f"model: test-model\ntimezone: {ZONES[name]}\n", encoding="utf-8"
        )
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return homes


@pytest.fixture()
def pinned_clock(monkeypatch):
    import hermes_time
    from cron import jobs as cron_jobs

    monkeypatch.setattr(
        cron_jobs, "_hermes_now", lambda: PINNED.astimezone(hermes_time.now().tzinfo)
    )
    return PINNED


@pytest.fixture()
def client(profile_homes, pinned_clock):
    from hermes_cli import web_server  # noqa: F401  (the router reads its helpers here)
    from hermes_cli.web_routers import cron as cron_routes

    app = FastAPI()
    app.include_router(cron_routes.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def machine_app(profile_homes, pinned_clock, tmp_path):
    """The cron router behind the real machine-token check, with one job already stored.

    Real Automations and Connections tokens are issued, and their providers are
    registered for this test only.
    """
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import audit as audit_mod
    from hermes_cli.dashboard_auth import register_provider, token_auth
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from hermes_cli.web_routers import cron as cron_routes
    from plugins.dashboard_auth.raphael_workspace import (
        AutomationsManageTokenProvider,
        ConnectionsManageTokenProvider,
        token_store,
    )

    cron_routes._register_automations_machine_routes()
    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)
    automations_path = token_dir / "automations.token"
    connections_path = token_dir / "connections.token"
    automations_record = token_store.issue(
        out_path=automations_path, surface=token_store.AUTOMATIONS_SURFACE
    )
    token_store.issue(out_path=connections_path, surface=token_store.CONNECTIONS_SURFACE)
    automations = {"Authorization": "Bearer " + automations_path.read_text(encoding="utf-8").strip()}
    connections = {"Authorization": "Bearer " + connections_path.read_text(encoding="utf-8").strip()}

    providers = [AutomationsManageTokenProvider(), ConnectionsManageTokenProvider()]
    previous = {provider.name: snapshot_registration(provider.name) for provider in providers}
    try:
        for provider in providers:
            if previous[provider.name] is None:
                register_provider(provider)

        web_server._call_cron_for_profile(
            "default", "create_job", prompt="Summarize owner progress", schedule="every 1h", name="Owner summary"
        )

        app = FastAPI()
        app.include_router(cron_routes.router)

        @app.middleware("http")
        async def machine_auth(request, call_next):
            return await token_auth.token_auth_middleware(request, call_next)

        yield {
            "app": app,
            "automations": automations,
            "connections": connections,
            "automations_credential_id": automations_record.token_id,
            "store": profile_homes["default"] / "cron" / "jobs.json",
            # Owner rule of 2026-10-15: the line the shared machine-token sign-in check writes to this
            # log for each call is the platform's security record, not a write by the preview.
            "audit_log": audit_mod._resolve_log_path(),
        }
    finally:
        for provider in providers:
            if previous[provider.name] is None:
                restore_registration(provider.name, provider, None)


def _preview(client, schedule, profile="default", headers=None):
    return client.get(
        "/api/cron/schedule-preview",
        params={"schedule": schedule, "profile": profile},
        headers=headers,
    )


def _create(client, schedule, profile="default"):
    response = client.post(f"/api/cron/jobs?profile={profile}", json={**JOB, "schedule": schedule})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _stored_next_run_at(home, job_id):
    jobs = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    [job] = [job for job in jobs if job["id"] == job_id]
    return job["next_run_at"]


def _files(root):
    """Every file under ``root`` with its bytes, so a write anywhere shows up."""
    return {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_weekly_preview_is_the_next_run_creation_stores(client, profile_homes):
    preview = _preview(client, WEEKLY)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert datetime.fromisoformat(body["now"]) == PINNED
    assert datetime.fromisoformat(body["next_run_at"]) > PINNED

    job_id = _create(client, WEEKLY)
    assert _stored_next_run_at(profile_homes["default"], job_id) == body["next_run_at"]


@pytest.mark.parametrize(
    "before",
    [timedelta(hours=1), timedelta(seconds=ONESHOT_GRACE_SECONDS / 2)],
    ids=["one_hour_before", "inside_the_one_shot_grace_window"],
)
def test_one_time_start_before_now_answers_409_and_writes_nothing(client, tmp_path, before):
    _create(client, WEEKLY)
    files_before = _files(tmp_path)

    response = _preview(client, (PINNED - before).isoformat())

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "schedule_in_past"
    assert datetime.fromisoformat(detail["now"]) == PINNED
    assert _files(tmp_path) == files_before


@pytest.mark.parametrize("schedule", [WEEKLY, ONE_TIME_WALL_CLOCK], ids=["weekly", "one_time"])
def test_profile_time_zone_shifts_the_preview_as_it_shifts_creation(client, profile_homes, schedule):
    answers = {}
    for profile, home in profile_homes.items():
        preview = _preview(client, schedule, profile)
        assert preview.status_code == 200, preview.text
        answers[profile] = preview.json()
        job_id = _create(client, schedule, profile)
        assert _stored_next_run_at(home, job_id) == answers[profile]["next_run_at"]

    default, worker = answers["default"], answers["worker_alpha"]
    assert datetime.fromisoformat(default["now"]) == datetime.fromisoformat(worker["now"]) == PINNED
    assert default["next_run_at"] != worker["next_run_at"]


def test_only_the_automations_token_reaches_the_preview_and_the_store_is_untouched(
    machine_app, tmp_path
):
    automations, connections = machine_app["automations"], machine_app["connections"]
    store, audit_log = machine_app["store"], machine_app["audit_log"]
    store_before = store.read_bytes()
    files_before = _files(tmp_path)

    with TestClient(machine_app["app"]) as client:
        allowed = _preview(client, WEEKLY, headers=automations)
        assert allowed.status_code == 200, allowed.text
        assert datetime.fromisoformat(allowed.json()["now"]) == PINNED
        in_past = _preview(client, (PINNED - timedelta(hours=1)).isoformat(), headers=automations)
        assert in_past.status_code == 409, in_past.text
        assert in_past.json()["detail"]["code"] == "schedule_in_past"
        unparseable = _preview(client, "not a schedule", headers=automations)
        assert unparseable.status_code == 400, unparseable.text
        unknown_profile = _preview(client, WEEKLY, "missing_profile", headers=automations)
        assert unknown_profile.status_code == 404, unknown_profile.text
        invalid_profile = _preview(client, WEEKLY, "../bad", headers=automations)
        assert invalid_profile.status_code == 400, invalid_profile.text
        no_schedule = client.get(
            "/api/cron/schedule-preview", params={"profile": "default"}, headers=automations
        )
        assert no_schedule.status_code == 422, no_schedule.text
        refused = _preview(client, WEEKLY, headers=connections)
        assert refused.status_code == 403, refused.text

    assert store.read_bytes() == store_before
    files_after = _files(tmp_path)
    changed = {
        path
        for path in files_before.keys() | files_after.keys()
        if files_before.get(path) != files_after.get(path)
    }
    assert changed == {audit_log}
    # The audit log only grows: what it held before is an unchanged prefix of what it holds now.
    assert files_after[audit_log].startswith(files_before.get(audit_log, b""))


@pytest.mark.parametrize(
    "schedule",
    ["0 9 31 2 *", "999999999999999999999999999999h", "every 999999999999999999999999h"],
    ids=["cron_on_february_31", "one_time_duration_overflows", "interval_overflows"],
)
def test_schedule_errors_answer_400_with_one_audit_record_and_write_nothing(
    machine_app, tmp_path, schedule
):
    store, audit_log = machine_app["store"], machine_app["audit_log"]
    store_before = store.read_bytes()
    files_before = _files(tmp_path)

    with TestClient(machine_app["app"]) as client:
        response = _preview(client, schedule, headers=machine_app["automations"])

    assert response.status_code == 400, response.text
    assert store.read_bytes() == store_before
    files_after = _files(tmp_path)
    audit_before, audit_after = files_before.pop(audit_log, b""), files_after.pop(audit_log, b"")
    assert files_after == files_before
    # The audit log gained exactly one record, appended after its unchanged earlier bytes.
    assert audit_after.startswith(audit_before)
    appended = [json.loads(line) for line in audit_after[len(audit_before):].splitlines()]
    assert len(appended) == 1, appended
    expected = {
        "event": "token_auth_failure",
        "reason": "downstream_rejected",
        "status": 400,
        "decision": "deny",
        "route_template": "/api/cron/schedule-preview",
        "credential_id": machine_app["automations_credential_id"],
    }
    assert {key: appended[0].get(key) for key in expected} == expected
