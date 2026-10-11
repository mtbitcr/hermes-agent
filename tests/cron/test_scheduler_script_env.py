"""A cron job's script learns its own job id from HERMES_CRON_JOB_ID.

The id goes only into the environment built for the script process: a script
run without a job gets no such variable, and the scheduler's own environment
never changes.

The folder of sys.executable is first on the script's PATH, so a script started
by hand finds the commands a scheduled run finds.
"""

import json
import os
import sys
import threading
from unittest.mock import MagicMock

import pytest

import cron.scheduler as scheduler

PROBE = "job_id_probe.py"

RECURRING_JOB = {
    "id": "hourly-prod-check",
    "script": PROBE,
    "no_agent": True,
    "schedule": {"kind": "interval", "minutes": 60},
}

ONE_SHOT_JOB = {
    "id": "one-shot-check",
    "script": PROBE,
    "no_agent": True,
    "schedule": {"kind": "once", "run_at": "2026-10-05T12:00:00+00:00"},
    "run_claim": {"at": "2026-10-05T12:00:00+00:00", "by": "dispatch-owner"},
}


@pytest.fixture
def probe_script(tmp_path, monkeypatch):
    """A script in a temporary HERMES_HOME that prints the job id it sees."""
    scripts_dir = tmp_path / "hermes_home" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / PROBE).write_text(
        "import json, os\n"
        "print(json.dumps(os.environ.get('HERMES_CRON_JOB_ID')))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(scripts_dir.parent))
    monkeypatch.delenv("HERMES_CRON_JOB_ID", raising=False)
    return PROBE


def _job_id_seen_by(result):
    """The HERMES_CRON_JOB_ID value the probe printed (None when unset)."""
    ok, output = result
    assert ok, output
    return json.loads(output)


# The three ways the claim-heartbeat wrapper runs a job's script.
WRAPPER_PATHS = [
    pytest.param(RECURRING_JOB, False, id="job-without-one-shot-claim"),
    pytest.param(ONE_SHOT_JOB, False, id="one-shot-heartbeat-starts"),
    pytest.param(ONE_SHOT_JOB, True, id="one-shot-heartbeat-fails-to-start"),
]


def _fail_heartbeat_start(monkeypatch):
    start = MagicMock(side_effect=RuntimeError("can't start new thread"))
    monkeypatch.setattr(threading.Thread, "start", start)
    return start


@pytest.mark.parametrize(("job", "heartbeat_start_fails"), WRAPPER_PATHS)
def test_script_run_for_a_job_sees_its_job_id(
    probe_script, monkeypatch, job, heartbeat_start_fails
):
    """Each way the claim-heartbeat wrapper runs a job's script passes the id."""
    if heartbeat_start_fails:
        start = _fail_heartbeat_start(monkeypatch)

    result = scheduler._run_job_script_with_claim_heartbeat(job, probe_script)

    assert _job_id_seen_by(result) == job["id"]
    if heartbeat_start_fails:
        start.assert_called_once()


@pytest.mark.parametrize(
    "agent_job",
    [
        pytest.param(lambda job: {**job, "no_agent": False}, id="no-agent-false"),
        pytest.param(
            lambda job: {key: value for key, value in job.items() if key != "no_agent"},
            id="no-agent-missing",
        ),
    ],
)
@pytest.mark.parametrize(("job", "heartbeat_start_fails"), WRAPPER_PATHS)
def test_pre_check_script_of_an_agent_job_sees_no_job_id(
    probe_script, monkeypatch, job, heartbeat_start_fails, agent_job
):
    """Only a script-only job tells its script its id; an agent job's pre-check script gets none."""
    if heartbeat_start_fails:
        _fail_heartbeat_start(monkeypatch)

    result = scheduler._run_job_script_with_claim_heartbeat(agent_job(job), probe_script)

    assert _job_id_seen_by(result) is None


def test_script_run_without_a_job_sees_no_job_id(probe_script):
    """The same script sees the id when run for a job and nothing without one."""
    for_job = scheduler._run_job_script_with_claim_heartbeat(RECURRING_JOB, probe_script)
    assert _job_id_seen_by(for_job) == RECURRING_JOB["id"]

    without_job = scheduler._run_job_script(probe_script)
    assert _job_id_seen_by(without_job) is None


def test_job_id_never_reaches_the_scheduler_environment(probe_script):
    """The id goes to the script's copy of the environment, never os.environ."""
    before = dict(os.environ)

    result = scheduler._run_job_script_with_claim_heartbeat(RECURRING_JOB, probe_script)

    assert _job_id_seen_by(result) == RECURRING_JOB["id"]
    assert dict(os.environ) == before
    assert "HERMES_CRON_JOB_ID" not in os.environ


VENV_COMMAND = "venv-bin-only-command"
PRINT_PATH = 'printf %s "$PATH"\n'


@pytest.fixture
def venv_bin(tmp_path, monkeypatch):
    """A venv-style bin folder holding one command, standing in for sys.executable's."""
    folder = tmp_path / "venv" / "bin"
    folder.mkdir(parents=True)
    command = folder / VENV_COMMAND
    command.write_text("#!/bin/sh\necho found-in-venv-bin\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(folder / "python"))
    return folder


def _run_shell_script(tmp_path, monkeypatch, body):
    """Run `body` as a job script from a temporary HERMES_HOME; returns (ok, output)."""
    scripts_dir = tmp_path / "hermes_home" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "path_probe.sh").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(scripts_dir.parent))
    return scheduler._run_job_script("path_probe.sh")


def test_script_finds_a_command_in_the_interpreter_folder(
    venv_bin, tmp_path, monkeypatch
):
    """A script whose PATH lacks the interpreter's folder still finds a command placed there."""
    assert str(venv_bin) not in os.environ["PATH"].split(os.pathsep)

    ok, output = _run_shell_script(tmp_path, monkeypatch, VENV_COMMAND + "\n")

    assert ok, output
    assert output == "found-in-venv-bin"


def test_interpreter_folder_goes_first_on_the_script_path(
    venv_bin, tmp_path, monkeypatch
):
    """The script's PATH is the interpreter's folder, then the PATH the scheduler had."""
    before = os.environ["PATH"]

    ok, output = _run_shell_script(tmp_path, monkeypatch, PRINT_PATH)

    assert ok, output
    assert output == str(venv_bin) + os.pathsep + before
    assert os.environ["PATH"] == before


def test_path_already_led_by_the_interpreter_folder_is_unchanged(
    venv_bin, tmp_path, monkeypatch
):
    """A PATH that already starts with the interpreter's folder reaches the script as it is.

    This is a preservation control that passes before and after the change.
    """
    led = str(venv_bin) + os.pathsep + os.environ["PATH"]
    monkeypatch.setenv("PATH", led)

    ok, output = _run_shell_script(tmp_path, monkeypatch, PRINT_PATH)

    assert ok, output
    assert output == led


def test_script_without_a_path_gets_only_the_interpreter_folder(
    venv_bin, tmp_path, monkeypatch
):
    """With no PATH at all the script gets the folder alone, with no empty entry after it."""
    monkeypatch.delenv("PATH", raising=False)

    ok, output = _run_shell_script(tmp_path, monkeypatch, PRINT_PATH)

    assert ok, output
    assert output == str(venv_bin)
