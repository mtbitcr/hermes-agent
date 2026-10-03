"""GitHub App transport for delivery: allowlist, narrowed tokens, leased push, no token egress.

Tests 15 to 17 of the delivery plan's section 8 plus the card's socket and token-request
tests. A local stand-in plays GitHub (the pattern of test_kanban_pr_acceptance.py) and every
key is a throwaway generated here.
"""
import base64
import http.client
import importlib
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

REPO = "mtbitcr/hermes-agent"
SIBLING = "mtbitcr/raphael-workspace"
APP_ID = "4485539"
INSTALLATION = 151217494
TOKEN = "ghs_" + "Zq81Lw0Xp3Tn" * 3  # what the stand-in mints, then echoes in bodies, headers and errors
STRAY = "ghs_" + "Vb27Kc5Hd9Rm" * 3  # an unrelated token-shaped value inside a projected field
HEAD = "a" * 40
BASE = "b" * 40
SAMPLE = {"repo": REPO, "number": "7", "sha": HEAD, "id": "9", "head_sha": HEAD, "installation": str(INSTALLATION)}
SECTION_4_PERMISSIONS = {"contents", "pull_requests", "checks", "actions"}
ADMIN_CALLS = [
    ("GET", f"/repos/{REPO}/branches/main/protection"),
    ("PUT", f"/repos/{REPO}/branches/main/protection"),
    ("DELETE", f"/repos/{REPO}/branches/main/protection"),
    ("POST", f"/repos/{REPO}/branches/main/protection/enforce_admins"),
    ("GET", f"/repos/{REPO}/rules/branches/main"),
    ("POST", f"/repos/{REPO}/rulesets"),
    ("PATCH", f"/repos/{REPO}"),
    ("DELETE", f"/repos/{REPO}"),
    ("POST", f"/repos/{REPO}/hooks"),
    ("PUT", f"/repos/{REPO}/collaborators/someone"),
    ("POST", f"/repos/{REPO}/keys"),
    ("PUT", f"/repos/{REPO}/actions/permissions"),
    ("DELETE", f"/app/installations/{INSTALLATION}"),
    ("POST", f"/app/installations/{INSTALLATION}/access_tokens"),
    ("PUT", f"/user/installations/{INSTALLATION}/repositories/1"),
    ("POST", "/graphql"),
]
_RUN = subprocess.run


def _module():
    """Import inside each test, so a missing module fails that test rather than the whole file."""
    return importlib.import_module("hermes_cli.kanban_delivery_github")


def _transport_module(monkeypatch, github):
    """The module with its one connection seam pointed at the local stand-in."""
    mod = _module()
    port = github["port"]
    monkeypatch.setattr(mod, "_connect", lambda: http.client.HTTPConnection("127.0.0.1", port, timeout=10))
    return mod


def _new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key):
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _git(*args, cwd=None):
    return _RUN(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repositories(tmp_path):
    """A card repository with three chained commits, and an empty bare remote laid out like GitHub."""
    work = tmp_path / "work"
    _git("init", "-q", "-b", "main", str(work))
    commits = []
    for n in range(3):
        _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
             "commit", "-q", "--allow-empty", "-m", f"c{n}", cwd=work)
        commits.append(_git("rev-parse", "HEAD", cwd=work))
    remotes = tmp_path / "remotes"
    bare = remotes / "mtbitcr" / "hermes-agent.git"
    _git("init", "-q", "--bare", str(bare))
    return work, bare, remotes.as_uri(), commits


def _remote_head(bare, branch):
    return _RUN(["git", "--git-dir", str(bare), "rev-parse", "--verify", "-q", f"refs/heads/{branch}"],
                capture_output=True, text=True).stdout.strip()


def _record_git(monkeypatch):
    """Every subprocess the module starts, with its argv and environment, still really run."""
    calls = []

    def run(argv, *args, **kwargs):
        calls.append((list(argv), dict(kwargs.get("env") or os.environ)))
        return _RUN(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _capture_all_logs(caplog):
    caplog.set_level(logging.DEBUG)
    for name in list(logging.root.manager.loggerDict):
        caplog.set_level(logging.DEBUG, logger=name)


def _log_text(caplog):
    return "\n".join(f"{r.name} {r.getMessage()} {r.args!r} {r.exc_text or ''}" for r in caplog.records)


def _error_text(exc):
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts += [repr(exc), str(exc), repr(exc.args), repr(getattr(exc, "__dict__", {}))]
        parts += traceback.format_exception(type(exc), exc, exc.__traceback__)
        exc = exc.__cause__ or exc.__context__
    return "\n".join(parts)


def _files_text(root):
    return {p: p.read_bytes().decode("utf-8", "replace") for p in root.rglob("*") if p.is_file()}


def _template_regex(template):
    pattern = re.escape(template).replace(re.escape("{repo}"), re.escape(REPO))
    return re.compile(re.sub(r"\\\{\w+\\\}", "[^/?]+", pattern))


@pytest.fixture
def github(tmp_path):
    key = _new_key()
    key_file = tmp_path / "keys" / "app.pem"
    key_file.parent.mkdir()
    key_file.write_bytes(_pem(key))
    state = {"public": key.public_key(), "key_file": key_file, "jwts": [], "rejected_jwts": 0,
             "token_requests": [], "calls": [], "token_reply": "minted"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._route("PUT")

        def _reply(self, status, value=None, extra=(), raw=None, cut=0):
            data = raw if raw is not None else b"" if value is None else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("X-Echo", TOKEN)
            for name, val in extra:
                self.send_header(name, val)
            self.send_header("Content-Length", str(len(data) + cut))  # cut: the connection drops mid-body
            self.end_headers()
            self.wfile.write(data)

        def _mint(self, auth, body):
            app_jwt = auth.removeprefix("Bearer ")
            try:
                claims = jwt.decode(app_jwt, state["public"], algorithms=["RS256"], issuer=APP_ID,
                                    options={"require": ["exp", "iat", "iss"]})
            except jwt.PyJWTError:
                state["rejected_jwts"] += 1
                return self._reply(401, {"message": f"A JSON web token could not be decoded {TOKEN}"})
            if claims["exp"] - time.time() > 600:
                state["rejected_jwts"] += 1
                return self._reply(401, {"message": "Expiration time too far in the future"})
            state["jwts"].append(app_jwt)
            state["token_requests"].append(body)
            if state["token_reply"] == "refused":
                return self._reply(422, {"message": f"Permission denied for {TOKEN}", "token": TOKEN})
            return self._reply(201, {
                "token": TOKEN, "expires_at": "2026-10-03T12:00:00Z", "permissions": body["permissions"],
                "repository_selection": "selected",
                "repositories": [{"name": n, "full_name": f"mtbitcr/{n}"} for n in body["repositories"]]},
                cut=64 if state["token_reply"] == "truncated" else 0)

        def _route(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            auth = self.headers.get("Authorization", "")
            if method == "POST" and self.path == f"/app/installations/{INSTALLATION}/access_tokens":
                return self._mint(auth, body)
            state["calls"].append((method, self.path))
            path = self.path.split("?")[0]
            if auth != f"Bearer {TOKEN}":
                return self._reply(401, {"message": f"Bad credentials {TOKEN}"})
            if method == "GET" and path.endswith("/pulls/7"):
                return self._reply(200, {
                    "number": 7, "state": "open", "title": TOKEN, "body": TOKEN, "token": TOKEN,
                    "head": {"sha": HEAD, "ref": TOKEN, "label": TOKEN},
                    "base": {"sha": BASE, "ref": STRAY}, "mergeable_state": "clean"})
            if method == "GET" and path.endswith("/pulls/8"):
                return self._reply(200, raw=f"<html>Bad gateway {TOKEN}</html>".encode())
            if method == "PUT" and path.endswith("/pulls/7/merge"):
                return self._reply(405, {"message": f"Pull Request is not mergeable {TOKEN}",
                                         "documentation_url": f"https://docs.example/{TOKEN}"})
            if path.endswith("/logs"):
                return self._reply(302, None, [("Location", f"https://storage.example/log?sig={TOKEN}")])
            return self._reply(200, {})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["port"] = server.server_port
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_no_administration_call_is_possible(github, monkeypatch):
    mod = _transport_module(monkeypatch, github)
    forbidden = ("protection", "rules", "settings", "hooks", "collaborators", "keys", "admin",
                 "permissions", "branches", "installation")
    for entry in mod.REST_ALLOWLIST:
        assert entry.method in {"GET", "POST", "PUT"}
        if (entry.method, entry.template) == ("POST", "/app/installations/{installation}/access_tokens"):
            continue  # the App's own call that mints a narrowed token
        segments = entry.template.strip("/").split("/")
        assert segments[:2] == ["repos", "{repo}"] and len(segments) > 2
        assert not [s for s in segments for word in forbidden if word in s.lower()]

    for step in mod.STEP_PERMISSIONS:
        transport = mod.GitHubTransport(step, REPO, key_path=github["key_file"])
        for method, path in ADMIN_CALLS:
            with pytest.raises(mod.GitHubTransportError):
                transport.request(method, path)
    assert github["calls"] == [] and github["token_requests"] == []  # refused before the network

    for step in mod.STEP_PERMISSIONS:
        transport = mod.GitHubTransport(step, REPO, key_path=github["key_file"])
        for entry in mod.REST_ALLOWLIST:
            try:
                transport.request(entry.method, entry.template.format(**SAMPLE),
                                  query={k: SAMPLE[k] for k in entry.required_query})
            except mod.GitHubTransportError as exc:
                assert exc.reason in {"step_not_permitted", "endpoint_not_allowed"}
    allowed = [(e.method, _template_regex(e.template)) for e in mod.REST_ALLOWLIST]
    assert github["calls"]
    for method, path in github["calls"]:  # everything that reached the network was allowlisted
        assert any(m == method and rx.fullmatch(path.split("?")[0]) for m, rx in allowed), (method, path)
    assert len(github["token_requests"]) == len(mod.STEP_PERMISSIONS)
    for body in github["token_requests"]:
        assert "administration" not in json.dumps(body).lower()
        assert body["repositories"] == ["hermes-agent"]


def test_token_never_reaches_model_visible_output(github, monkeypatch, caplog, capfd, tmp_path):
    _capture_all_logs(caplog)
    home = tmp_path / "home"  # any git global or credential file would land here
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, commits = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    git_calls = _record_git(monkeypatch)
    outputs, errors = [], []

    merge = mod.GitHubTransport("merge", REPO, key_path=github["key_file"])
    pull = merge.request("GET", f"/repos/{REPO}/pulls/7")
    assert pull["status"] == 200 and pull["data"]["number"] == 7 and pull["data"]["head"]["sha"] == HEAD
    refused = merge.request("PUT", f"/repos/{REPO}/pulls/7/merge", body={"sha": HEAD})
    assert refused["status"] == 405
    logs = mod.GitHubTransport("read_checks", REPO, key_path=github["key_file"]).request(
        "GET", f"/repos/{REPO}/actions/jobs/9/logs")
    assert logs == {"status": 302, "data": None}  # the redirect to signed storage is not followed
    pushed = mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).push(
        work, "delivery/card-1", commits[0], expected="")
    assert pushed["head"] == commits[0] and _remote_head(bare, "delivery/card-1") == commits[0]
    outputs += [pull, refused, logs, pushed, repr(merge)]
    with pytest.raises(mod.GitHubTransportError) as garbled:
        merge.request("GET", f"/repos/{REPO}/pulls/8")
    errors.append(garbled.value)
    for reply in ("refused", "truncated"):
        github["token_reply"] = reply
        with pytest.raises(mod.GitHubTransportError) as failed:
            mod.GitHubTransport("handoff", REPO, key_path=github["key_file"]).request("GET", f"/repos/{REPO}/pulls/7")
        errors.append(failed.value)
    monkeypatch.setattr(mod, "_connect", lambda: http.client.HTTPConnection("127.0.0.1", 9, timeout=2))
    with pytest.raises(mod.GitHubTransportError) as unreachable:
        mod.GitHubTransport("handoff", REPO, key_path=github["key_file"]).request("GET", f"/repos/{REPO}/pulls/7")
    errors.append(unreachable.value)

    jwts = list(github["jwts"])
    assert len(jwts) == 5 and github["rejected_jwts"] == 0
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    key_line = github["key_file"].read_text().splitlines()[1]
    out, err = capfd.readouterr()
    visible = "\n".join([repr(outputs), *map(_error_text, errors), _log_text(caplog), caplog.text, out, err,
                         *(" ".join(argv) for argv, _ in git_calls)])
    for secret in (TOKEN, STRAY, basic, key_line, *jwts):
        assert secret not in visible
    for argv, _ in git_calls:
        assert not [a for a in argv if "@" in a and "://" in a]  # no credential in the remote address
    for path, text in _files_text(tmp_path).items():  # git config, credential files and anything else
        for secret in (TOKEN, basic, *jwts):
            assert secret not in text, path


def test_store_path_under_dispatcher_environment(github, monkeypatch, tmp_path):
    root = tmp_path / "root"  # a root outside the platform default, as in Docker or custom installs
    profile = root / "profiles" / "raphael-claude-worker"
    board = root / "kanban" / "boards" / "delivery"
    workspace = board / "workspaces" / "t_0001"
    for path in (profile, workspace):
        path.mkdir(parents=True)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    for name, value in (("HERMES_HOME", profile), ("HERMES_KANBAN_DB", board / "kanban.db"),
                        ("HERMES_KANBAN_BOARD", "delivery"), ("HERMES_KANBAN_TASK", "t_0001"),
                        ("HERMES_KANBAN_WORKSPACES_ROOT", board / "workspaces"),
                        ("HERMES_KANBAN_WORKSPACE", workspace), ("TERMINAL_CWD", workspace),
                        ("HERMES_PROFILE", "raphael-claude-worker")):
        monkeypatch.setenv(name, str(value))
    monkeypatch.chdir(workspace)  # the dispatcher starts the worker inside the task workspace
    root_key, relative = _new_key(), Path("secrets") / "github-app" / "raphael-agent-factory.pem"
    for base, key in ((root, root_key), (profile, _new_key()), (board, _new_key()), (workspace, _new_key())):
        (base / relative).parent.mkdir(parents=True)
        (base / relative).write_bytes(_pem(key))
    github["public"] = root_key.public_key()
    mod = _transport_module(monkeypatch, github)

    result = mod.GitHubTransport("handoff", REPO).request("GET", f"/repos/{REPO}/pulls/7")

    assert result["status"] == 200 and result["data"]["number"] == 7
    assert len(github["jwts"]) == 1 and github["rejected_jwts"] == 0  # the stand-in's own count
    assert len(github["token_requests"]) == 1


def test_path_outside_the_allowlist_is_refused_before_any_socket(monkeypatch, tmp_path):
    mod = _module()
    opened = []

    def no_socket(*args, **kwargs):
        opened.append(args)
        pytest.fail("a socket was created for a refused request")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    missing_key = tmp_path / "no-such-key.pem"  # a refused call never mints, so never reads the key
    outside = [
        *ADMIN_CALLS,
        ("GET", f"/repos/{SIBLING}/pulls/7"),
        ("GET", f"/repos/{REPO}/pulls/7/files"),
        ("GET", f"/repos/{REPO}/pulls/07"),
        ("GET", f"/repos/{REPO}/pulls/7/"),
        ("GET", f"/repos/{REPO}//pulls/7"),
        ("GET", f"/repos/{REPO}/pulls/7/../../hooks"),
        ("GET", f"/repos/{REPO}/pulls/%37"),
        ("GET", f"/repos/{REPO}%2Fhooks"),
        ("GET", f"/repos/{REPO}/pulls/7?per_page=100"),
        ("GET", f"https://evil.example/repos/{REPO}/pulls/7"),
        ("get", f"/repos/{REPO}/pulls/7"),
        ("DELETE", f"/repos/{REPO}/pulls/7"),
        ("GET", f"/repos/{REPO}/commits/{HEAD.upper()}/check-runs"),
        ("GET", f"/repos/{REPO}/actions/runs"),
    ]
    for step in mod.STEP_PERMISSIONS:
        transport = mod.GitHubTransport(step, REPO, key_path=missing_key)
        for method, path in outside:
            with pytest.raises(mod.GitHubTransportError) as refused:
                transport.request(method, path)
            assert refused.value.reason in {"endpoint_not_allowed", "query_not_allowed"}, (step, method, path)
        with pytest.raises(mod.GitHubTransportError) as refused:
            transport.request("GET", f"/repos/{REPO}/pulls", query={"per_page": "100", "q": "is:open"})
        assert refused.value.reason == "query_not_allowed"
    assert opened == []


def test_no_token_request_names_administration(github, monkeypatch):
    mod = _transport_module(monkeypatch, github)
    for step in ({"administration": "write"}, "administration", "publish ", None):
        with pytest.raises(mod.GitHubTransportError):
            mod.GitHubTransport(step, REPO, key_path=github["key_file"])
    with pytest.raises(TypeError):
        mod.STEP_PERMISSIONS["publish"]["administration"] = "write"
    with pytest.raises(TypeError):
        mod.STEP_PERMISSIONS["admin"] = {"administration": "write"}

    transports = 0
    for repository in (REPO, SIBLING):
        for step in mod.STEP_PERMISSIONS:
            transport = mod.GitHubTransport(step, repository, key_path=github["key_file"])
            transports += 1
            for entry in mod.REST_ALLOWLIST:  # every call the step may make shares its one token
                try:
                    transport.request(entry.method, entry.template.format(**{**SAMPLE, "repo": repository}),
                                      query={k: SAMPLE[k] for k in entry.required_query})
                except mod.GitHubTransportError as exc:
                    assert exc.reason in {"step_not_permitted", "endpoint_not_allowed"}

    requests = github["token_requests"]
    assert len(requests) == transports  # the stand-in's own count: one per step and repository
    for body in requests:
        assert "administration" not in json.dumps(body).lower()
        assert set(body) == {"repositories", "permissions"} and len(body["repositories"]) == 1
        assert body["permissions"] and set(body["permissions"]) <= SECTION_4_PERMISSIONS
        assert set(body["permissions"].values()) <= {"read", "write"}
    assert {body["repositories"][0] for body in requests} == {"hermes-agent", "raphael-workspace"}


def test_push_carries_an_explicit_lease_and_keeps_the_credential_in_the_environment(github, monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, second, third) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    calls = _record_git(monkeypatch)
    branch = "delivery/card-1"
    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])

    assert publish.push(work, branch, first, expected="")["head"] == first  # the branch had to be absent
    assert _remote_head(bare, branch) == first
    for head, expected in ((second, ""), (second, third), (second, "c" * 40)):
        with pytest.raises(mod.GitHubTransportError) as stale:
            publish.push(work, branch, head, expected=expected)
        assert stale.value.reason == "push_lease_mismatch"
        assert _remote_head(bare, branch) == first
    assert publish.push(work, branch, second, expected=first)["head"] == second
    assert _remote_head(bare, branch) == second

    pushes = len(calls)
    for name, head, expected in ((branch, third, None), (branch, third, "main"), (branch, third, second.upper()),
                                 (branch, third, second[:39]), (branch, "HEAD", second), ("+" + branch, third, second),
                                 ("../main", third, second), ("delivery/a..b", third, second), ("", third, second)):
        with pytest.raises(mod.GitHubTransportError) as bad:
            publish.push(work, name, head, expected=expected)
        assert bad.value.reason == "bad_push_target", (name, head, expected)
    with pytest.raises(mod.GitHubTransportError) as denied:
        mod.GitHubTransport("read_checks", REPO, key_path=github["key_file"]).push(work, branch, third, expected=second)
    assert denied.value.reason == "step_not_permitted"
    assert len(calls) == pushes and len(github["token_requests"]) == 1  # refused before git or a token

    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    lease = re.compile(rf"--force-with-lease=refs/heads/{re.escape(branch)}:([0-9a-f]{{40}})?")
    for argv, env in calls:
        leases = [a for a in argv if a.startswith("--force-with-lease")]
        assert len(leases) == 1 and lease.fullmatch(leases[0])
        assert not {"--force", "-f", "--mirror", "--all", "--tags", "--delete", "-d", "--prune"} & set(argv)
        assert not [a for a in argv if a.startswith("+")]
        assert f"{origin}/{REPO}.git" in argv
        assert not [a for a in argv if TOKEN in a or basic in a]
        config = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
        assert config[f"http.{origin}/.extraheader"] == f"Authorization: basic {basic}"
        assert config["credential.helper"] == ""
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        holders = [k for k, v in env.items() if TOKEN in v or basic in v]
        assert len(holders) == 1 and holders[0].startswith("GIT_CONFIG_VALUE_")
    assert not [k for k, v in os.environ.items() if TOKEN in v or basic in v]
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and basic not in text, path
