"""Card A5: the owner connects their own GitHub repository to a project, served by the API server.

Each test runs on its own Hermes root that serves the owner workspace, with one project "app". GitHub is never
reached: the check's read of main gets a fake answer, and connect_repository is a fake that makes the clone, a
local git repository, and binds its folder as the real one does. The projects store, the owner delivery list and
the clone the routes read and write are real.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_repository as routes
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import owner_write_watch as watch
from hermes_cli import projects_db as pdb
from hermes_cli import repo_connect
from hermes_cli.kanban_delivery_fences import Decision
from hermes_cli.kanban_delivery_github import GitHubTransport, GitHubTransportError
from hermes_cli.kanban_delivery_owner_policy import _code_policy, confirmations_path, entry_digest, owner_policy_path
from hermes_cli.repo_bootstrap import _EMAIL, _MESSAGE, _NAME

KEY = "sk-owner-secret"
OWNER = {"Authorization": f"Bearer {KEY}"}
TEMPLATE = "/v1/owner-workspace/projects/{slug}/repository"
VIEW = "/v1/owner-workspace/projects/app/repository"
CHECK = f"{VIEW}/check"
CONNECT = f"{VIEW}/connect"
REPOSITORY = "mtbitcr/example-app"
DIGEST = entry_digest(REPOSITORY, ["quality"])
COMMIT = "c" * 40
CONNECT_CODES = (
    "foreign_account", "unknown_project", "project_has_primary_folder", "repository_has_folder",
    "repository_has_project", "main_unreadable", "token_request_failed", "push_lease_mismatch",
)
_GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"}


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hermes root at ``<tmp>/.hermes`` that serves the owner workspace and holds the project "app"."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _owner_workspace(home, monkeypatch, enabled=True)
    with pdb.connect_closing() as conn:
        pdb.create_project(conn, name="App", slug="app")
    return home


@pytest.fixture(autouse=True)
def github(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """GitHub's answer to the read of main (409: an empty repository), and every request the routes sent."""
    fake = SimpleNamespace(status=409, sha=None, requests=[])

    def request(transport, method, path, **_options):
        fake.requests.append((transport.step, method, path))
        return {"status": fake.status, "data": {"object": {"sha": fake.sha}} if fake.status == 200 else None}

    monkeypatch.setattr(GitHubTransport, "request", request)
    return fake


@pytest.fixture(autouse=True)
def connects(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """connect_repository as a fake that makes the clone and binds its folder as the project's primary folder, as
    the real one does, or raises ``refusal``. ``bootstrapped`` says whether the repository was empty, so its first
    commit is written here. Each call is kept."""
    fake = SimpleNamespace(calls=[], refusal=None, bootstrapped=True)

    def connect(project_slug, repository):
        fake.calls.append((project_slug, repository))
        if fake.refusal is not None:
            raise fake.refusal
        folder = repo_connect._folder(repository)
        folder.mkdir(parents=True)
        _clone(folder, bootstrapped=fake.bootstrapped)
        with pdb.connect_closing() as conn:
            path = pdb.add_folder(conn, pdb.get_project(conn, project_slug).id, str(folder))
        return {"repository": repository, "folder": path, "bootstrapped": fake.bootstrapped, "commit": COMMIT}

    monkeypatch.setattr(repo_connect, "connect_repository", connect)
    return fake


def _owner_workspace(home: Path, monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    monkeypatch.setattr("gateway.run._hermes_home", home)
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": enabled}}}}
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _clone(folder: Path, *, bootstrapped: bool) -> None:
    """Make ``folder`` a git repository whose main holds one root commit: the first commit
    hermes_cli/repo_bootstrap.py writes when the repository was empty, else the owner's own."""
    name, email, message = (_NAME, _EMAIL, _MESSAGE) if bootstrapped else ("Owner", "owner@example.invalid", "Start")
    env = {**os.environ, **_GIT_ENV, "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
           "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
    (folder / "README.md").write_text("# Example app\n", encoding="utf-8")
    for argv in (["init", "--quiet", "--initial-branch=main"], ["add", "README.md"],
                 ["commit", "--quiet", "-m", message]):
        subprocess.run(["git", "-C", str(folder), *argv], env=env, check=True, capture_output=True)


def _client(key: str = KEY) -> TestClient:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith(TEMPLATE):
            app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


def _tree(home: Path) -> dict:
    """Every folder and file under the root, each file byte for byte; SQLite's own side files, which a read
    alone may touch, left out."""
    return {
        path.relative_to(home).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(home.rglob("*"))
        if not path.name.endswith(("-wal", "-shm", "-journal"))
    }


def _confirmations() -> list:
    path = confirmations_path()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


async def _refused(response, status: int, code: str) -> str:
    assert response.status == status
    error = (await response.json())["error"]
    assert error["code"] == code
    return error["message"]


@pytest.mark.asyncio
async def test_routes_refuse_a_request_without_owner_authentication(root, github, connects):
    routes = [("GET", VIEW, None), ("POST", CHECK, {"repository": REPOSITORY}),
              ("POST", CONNECT, {"repository": REPOSITORY, "digest": DIGEST})]
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        for method, path, body in routes:
            for headers in ({}, {"Authorization": "Bearer sk-not-the-owner"}):
                response = await client.request(method, path, json=body, headers=headers)
                assert response.status == 401, (method, path)
        assert (await client.get(VIEW, headers=OWNER)).status == 200

    assert _tree(root) == before
    assert github.requests == connects.calls == []


@pytest.mark.asyncio
async def test_routes_refuse_a_profile_without_the_owner_workspace(root, github, connects, monkeypatch):
    _owner_workspace(root, monkeypatch, enabled=False)
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        for method, path, body in [("GET", VIEW, None), ("POST", CHECK, {"repository": REPOSITORY}),
                                   ("POST", CONNECT, {"repository": REPOSITORY, "digest": DIGEST})]:
            response = await client.request(method, path, json=body, headers=OWNER)
            await _refused(response, 404, "owner_workspace_not_enabled")

    assert _tree(root) == before
    assert github.requests == connects.calls == []


@pytest.mark.asyncio
async def test_check_writes_nothing_and_answers_the_digest(root, github, connects):
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        empty = await client.post(CHECK, json={"repository": REPOSITORY}, headers=OWNER)
        github.status, github.sha = 200, COMMIT
        held = await client.post(CHECK, json={"repository": REPOSITORY}, headers=OWNER)
        github.status = 500
        unreadable = await client.post(CHECK, json={"repository": REPOSITORY}, headers=OWNER)
        assert (empty.status, held.status) == (200, 200)
        shown = {"object": "hermes.owner_workspace.repository_check", "repository": REPOSITORY,
                 "required_checks": ["quality"], "digest": DIGEST}
        assert await empty.json() == {**shown, "main_empty": True}
        assert await held.json() == {**shown, "main_empty": False}
        message = await _refused(unreadable, 409, "repository_refused")
        assert "main_unreadable" not in message

    assert _tree(root) == before
    assert github.requests == [("read_checks", "GET", f"/repos/{REPOSITORY}/git/ref/heads/main")] * 3
    assert connects.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [
    entry_digest(REPOSITORY, ["quality", "lint"]),
    entry_digest("mtbitcr/other-app", ["quality"]),
    entry_digest(REPOSITORY.upper(), ["quality"]),
    "0" * 64,
])
async def test_connect_with_a_stale_digest_answers_409_and_writes_nothing(root, github, connects, digest):
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        response = await client.post(CONNECT, json={"repository": REPOSITORY, "digest": digest}, headers=OWNER)
        await _refused(response, 409, "repository_changed")
        view = await (await client.get(VIEW, headers=OWNER)).json()

    assert view["state"] == "none"
    assert _tree(root) == before
    assert connects.calls == []
    assert not confirmations_path().exists() and not owner_policy_path().exists()


@pytest.mark.asyncio
async def test_connect_with_the_right_digest_connects_once_and_records_one_confirmation(root, github, connects):
    async with _client() as client:
        view = await (await client.get(VIEW, headers=OWNER)).json()
        shown = await (await client.post(CHECK, json={"repository": REPOSITORY}, headers=OWNER)).json()
        response = await client.post(
            CONNECT, json={"repository": REPOSITORY, "digest": shown["digest"]}, headers=OWNER)
        text = await response.text()
        connected = await (await client.get(VIEW, headers=OWNER)).json()

    assert view == {"object": "hermes.owner_workspace.repository", "state": "none", "repository": None}
    assert response.status == 200
    assert json.loads(text) == {
        "object": "hermes.owner_workspace.repository_receipt", "repository": REPOSITORY, "bootstrapped": True}
    # The answer never holds the clone's folder, nor the commit.
    folder = repo_connect._folder(REPOSITORY)
    assert str(folder) not in text and "repositories" not in text and COMMIT not in text
    assert connected == {"object": "hermes.owner_workspace.repository", "state": "connected",
                         "repository": REPOSITORY}
    assert connects.calls == [("app", REPOSITORY)]
    [line] = _confirmations()
    assert (line["repository"], line["required_checks"], line["digest"]) == (REPOSITORY, ["quality"], DIGEST)
    listed = json.loads(owner_policy_path().read_text(encoding="utf-8"))
    assert listed == {REPOSITORY: {"required_checks": ["quality"]}}


@pytest.mark.asyncio
async def test_a_repeat_connect_writes_nothing(root, github, connects):
    connects.bootstrapped = False
    body = {"repository": REPOSITORY, "digest": DIGEST}
    async with _client() as client:
        first = await client.post(CONNECT, json=body, headers=OWNER)
        assert first.status == 200
        receipt = await first.json()
        before = _tree(root)
        repeats = [await client.post(CONNECT, json=body, headers=OWNER) for _ in range(2)]
        answers = [await response.json() for response in repeats]

    assert [response.status for response in repeats] == [200, 200]
    assert answers == [receipt] * 2
    assert _tree(root) == before
    assert connects.calls == [("app", REPOSITORY)]
    assert len(_confirmations()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bootstrapped", [True, False], ids=["empty", "non-empty"])
async def test_a_repeat_answers_the_first_receipt_also_from_a_fresh_module_state(
        root, github, connects, bootstrapped):
    """The receipt comes from the repository on every success: bootstrapped says whether main in the clone has one
    root commit, written as hermes_cli/repo_bootstrap.py writes it. So a repeat, also from a fresh module state as
    after a gateway restart, writes nothing and answers the first receipt, for an empty and a non-empty repository."""
    connects.bootstrapped = bootstrapped
    body = {"repository": REPOSITORY, "digest": DIGEST}
    async with _client() as client:
        first = await client.post(CONNECT, json=body, headers=OWNER)
        assert first.status == 200
        receipt = await first.json()
    importlib.reload(routes)  # a fresh module state, as after a gateway restart
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        repeat = await client.post(CONNECT, json=body, headers=OWNER)
        assert repeat.status == 200
        answer = await repeat.json()

    assert receipt == {
        "object": "hermes.owner_workspace.repository_receipt", "repository": REPOSITORY, "bootstrapped": bootstrapped}
    assert answer == receipt
    assert _tree(root) == before
    assert connects.calls == [("app", REPOSITORY)]
    assert len(_confirmations()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", sorted(_code_policy().required_checks))
async def test_a_repository_of_the_reviewed_policy_is_refused_before_any_write(root, github, connects, repository):
    """The entry is validated before connect_repository: a repository of the reviewed policy, a platform
    repository, answers one plain 409 sentence, also on a retry, and nothing is cloned, bound or recorded."""
    body = {"repository": repository, "digest": entry_digest(repository, ["quality"])}
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        for _ in range(2):
            response = await client.post(CONNECT, json=body, headers=OWNER)
            text = await response.text()
            message = await _refused(response, 409, "repository_not_listed")
            assert "platform_repository" not in text
            assert message.endswith(".") and message.count(".") == 1 and message[0].isupper()
        view = await (await client.get(VIEW, headers=OWNER)).json()

    assert view["state"] == "none"
    assert connects.calls == []
    assert _tree(root) == before
    assert not confirmations_path().exists() and not owner_policy_path().exists()


@pytest.mark.asyncio
async def test_a_refused_confirmation_stays_refused_on_a_retry_and_records_nothing(
        root, github, connects, monkeypatch):
    """A connect whose confirmation record_confirmed_entry refuses answers the plain 409 sentence. The retry finds
    the clone bound as the project's primary folder, so it skips connect_repository, but a bound clone alone is no
    finished connect: the confirmation is asked again, refused again, and nothing is recorded."""
    asked = []

    def refuse(repository, required_checks, shown_digest):
        asked.append(repository)
        return Decision(False, "platform_repository", f"{repository} is in the reviewed policy")

    monkeypatch.setattr(routes, "record_confirmed_entry", refuse)
    body = {"repository": REPOSITORY, "digest": DIGEST}
    async with _client() as client:
        message = await _refused(await client.post(CONNECT, json=body, headers=OWNER), 409, "repository_not_listed")
        before = _tree(root)
        retry = await client.post(CONNECT, json=body, headers=OWNER)
        assert await _refused(retry, 409, "repository_not_listed") == message

    assert connects.calls == [("app", REPOSITORY)]
    assert asked == [REPOSITORY] * 2
    assert _tree(root) == before
    assert _confirmations() == [] and not owner_policy_path().exists()


@pytest.mark.asyncio
async def test_a_confirmation_write_that_raises_answers_503_and_the_retry_records_one_line_then_succeeds(
        root, github, connects):
    """The confirmation cannot be appended to its log, so the connect answers the existing 503 sentence with the
    clone bound and nothing recorded. The retry skips connect_repository, records the one confirmation line, lists
    the entry and answers the receipt."""
    confirmations_path().mkdir(parents=True)  # the log is a folder: appending the confirmation raises
    body = {"repository": REPOSITORY, "digest": DIGEST}
    async with _client() as client:
        await _refused(await client.post(CONNECT, json=body, headers=OWNER), 503, "owner_workspace_unavailable")
        assert not owner_policy_path().exists()
        confirmations_path().rmdir()
        retry = await client.post(CONNECT, json=body, headers=OWNER)
        text = await retry.text()

    assert retry.status == 200, text
    assert json.loads(text) == {
        "object": "hermes.owner_workspace.repository_receipt", "repository": REPOSITORY, "bootstrapped": True}
    assert connects.calls == [("app", REPOSITORY)]
    [line] = _confirmations()
    assert (line["repository"], line["required_checks"], line["digest"]) == (REPOSITORY, ["quality"], DIGEST)
    listed = json.loads(owner_policy_path().read_text(encoding="utf-8"))
    assert listed == {REPOSITORY: {"required_checks": ["quality"]}}


@pytest.mark.asyncio
async def test_a_true_repeat_appends_no_second_confirmation_line(root, github, connects, monkeypatch):
    """A repeat of a finished connect still runs every step: connect_repository is skipped, as the project's
    primary folder holds the repository, and record_confirmed_entry answers already_confirmed, so the repeat
    appends no second confirmation line."""
    record, decisions = routes.record_confirmed_entry, []

    def recorded(*args):
        decision = record(*args)
        decisions.append(decision.code)
        return decision

    monkeypatch.setattr(routes, "record_confirmed_entry", recorded)
    body = {"repository": REPOSITORY, "digest": DIGEST}
    async with _client() as client:
        answers = [await client.post(CONNECT, json=body, headers=OWNER) for _ in range(2)]
        receipts = [await answer.json() for answer in answers]

    assert [answer.status for answer in answers] == [200, 200]
    assert receipts[1] == receipts[0]
    assert decisions == ["confirmed", "already_confirmed"]
    assert connects.calls == [("app", REPOSITORY)]
    assert len(_confirmations()) == 1


WRONG_BODIES = [
    (CHECK, {}),
    (CHECK, []),
    (CHECK, "mtbitcr/example-app"),
    (CHECK, {"repository": None}),
    (CHECK, {"repository": ["mtbitcr/example-app"]}),
    (CHECK, {"repository": REPOSITORY, "digest": DIGEST}),
    (CHECK, {"repository": REPOSITORY, "project": "app"}),
    (CHECK, {"repo": REPOSITORY}),
    (CHECK, {"repository": "example-app"}),  # names the transport refuses
    (CHECK, {"repository": "mtbitcr/.."}),
    (CHECK, {"repository": "mt_bitcr/example-app"}),
    (CHECK, {"repository": "mtbitcr/example-app/main"}),
    (CONNECT, {}),
    (CONNECT, {"repository": REPOSITORY}),
    (CONNECT, {"digest": DIGEST}),
    (CONNECT, {"repository": REPOSITORY, "digest": DIGEST, "required_checks": ["quality"]}),
    (CONNECT, {"repository": REPOSITORY, "digest": DIGEST.upper()}),
    (CONNECT, {"repository": REPOSITORY, "digest": DIGEST[:-1]}),
    (CONNECT, {"repository": REPOSITORY, "digest": None}),
    (CONNECT, {"repository": 7, "digest": DIGEST}),
    (CONNECT, [REPOSITORY, DIGEST]),
    (CONNECT, {"repository": "mtbitcr/..", "digest": entry_digest("mtbitcr/..", ["quality"])}),
    (CONNECT, {"repository": "example-app", "digest": entry_digest("example-app", ["quality"])}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body", WRONG_BODIES)
async def test_each_wrong_body_answers_400(root, github, connects, path, body):
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        response = await client.post(path, json=body, headers=OWNER)
        await _refused(response, 400, "invalid_argument")
        for raw in (b"", b"{", b"null"):
            response = await client.post(path, data=raw, headers={**OWNER, "Content-Type": "application/json"})
            await _refused(response, 400, "invalid_argument")

    assert _tree(root) == before
    assert github.requests == connects.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("code", CONNECT_CODES)
async def test_each_connect_refusal_answers_one_plain_sentence_with_409(root, github, connects, code):
    error = repo_connect.RepositoryConnectError if code in CONNECT_CODES[:6] else GitHubTransportError
    connects.refusal = error(code)
    async with _client() as client:
        response = await client.post(CONNECT, json={"repository": REPOSITORY, "digest": DIGEST}, headers=OWNER)
        text = await response.text()
        message = await _refused(response, 409, "repository_refused")

    assert code not in text
    assert message.endswith(".") and message.count(".") == 1 and message[0].isupper()
    assert connects.calls == [("app", REPOSITORY)]
    assert _confirmations() == [] and not owner_policy_path().exists()


@pytest.mark.asyncio
async def test_each_connect_refusal_has_its_own_sentence(root, github, connects):
    messages = {}
    async with _client() as client:
        for code in CONNECT_CODES[:6]:
            connects.refusal = repo_connect.RepositoryConnectError(code)
            response = await client.post(
                CONNECT, json={"repository": REPOSITORY, "digest": DIGEST}, headers=OWNER)
            messages[code] = await _refused(response, 409, "repository_refused")

    assert len(set(messages.values())) == len(messages)


@pytest.mark.asyncio
async def test_a_repository_name_the_transport_refuses_in_connect_answers_400(root, github, connects):
    connects.refusal = GitHubTransportError("bad_repository")
    async with _client() as client:
        response = await client.post(CONNECT, json={"repository": REPOSITORY, "digest": DIGEST}, headers=OWNER)
        text = await response.text()
        await _refused(response, 400, "invalid_argument")

    assert "bad_repository" not in text
    assert _confirmations() == []


@pytest.mark.asyncio
async def test_a_project_holding_another_folder_shows_none_and_its_connect_is_refused(root, github, connects):
    local = root / "work" / "app"
    local.mkdir(parents=True)
    with pdb.connect_closing() as conn:
        pdb.add_folder(conn, pdb.get_project(conn, "app").id, str(local))
    connects.refusal = repo_connect.RepositoryConnectError("project_has_primary_folder")
    async with _client() as client:
        view = await (await client.get(VIEW, headers=OWNER)).json()
        response = await client.post(CONNECT, json={"repository": REPOSITORY, "digest": DIGEST}, headers=OWNER)
        await _refused(response, 409, "repository_refused")
        missing = await client.get("/v1/owner-workspace/projects/no-such-project/repository", headers=OWNER)
        await _refused(missing, 404, "project_not_found")

    assert view["state"] == "none"
    assert _confirmations() == []


@pytest.mark.asyncio
async def test_the_two_repository_probes_get_their_healthy_answer_from_the_real_routes(root, github, connects):
    probes = [probe for probe in watch.PROBES["default"] if "/repository/" in probe.path]
    assert len(probes) == 2
    client = _client()  # the adapter makes its own response store first
    before = _tree(root)
    async with client:
        for probe in probes:
            response = await client.post(probe.path.removeprefix("/p/default"), json=probe.body, headers=OWNER)
            assert watch.classify(probe, response.status, await response.json()) == watch.PASS, probe.name

    assert _tree(root) == before
    assert github.requests == connects.calls == []
