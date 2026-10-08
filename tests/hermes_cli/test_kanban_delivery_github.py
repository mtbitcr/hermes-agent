"""GitHub App transport for delivery: allowlist, narrowed tokens, leased push, no token egress.

Tests 15 to 17 of the delivery plan's section 8, the card's socket and token-request tests, and
the review's regressions. A local stand-in plays GitHub (the pattern of
test_kanban_pr_acceptance.py) and every key is a throwaway generated here.
"""
import base64
import http.client
import importlib
import io
import json
import logging
import os
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
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
BASIC = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()  # the token as git's Basic header carries it
DUMMY = "ghs_" + "Nf64Jt8Yw2Qs" * 3  # a credential of the controls' own, so a control never plants TOKEN anywhere
DUMMY_BASIC = base64.b64encode(f"x-access-token:{DUMMY}".encode()).decode()
HEAD = "a" * 40
BASE = "b" * 40
SAMPLE = {"repo": REPO, "number": "7", "sha": HEAD, "id": "9", "head_sha": HEAD, "installation": str(INSTALLATION),
          "branch": "card-1"}
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
_MKDTEMP = tempfile.mkdtemp


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


def _remote_refs(bare):
    """Every ref the bare remote holds, read back from the remote itself rather than from output."""
    listed = _RUN(["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname)"],
                  capture_output=True, text=True).stdout.split()
    return sorted(listed)


def _apply_git_config(monkeypatch, scope, work, tmp_path, settings):
    """Put *settings* where the named git configuration scope reads them: the card repository's own
    config, the global file, or the system file. git writes each file, so the section quoting is
    exactly what git reads back."""
    if scope == "repository":
        for key, value in settings:
            _git("config", key, value, cwd=work)
        return
    path = tmp_path / f"{scope}.gitconfig"
    for key, value in settings:
        _git("config", "--file", str(path), key, value)
    if scope == "system":
        monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(path))
    else:
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(path))


def _record_git(monkeypatch):
    """Every subprocess the module starts, with its argv and environment, still really run."""
    calls = []

    def run(argv, *args, **kwargs):
        calls.append((list(argv), dict(kwargs.get("env") or os.environ)))
        return _RUN(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _record_mkdtemp(monkeypatch):
    """Every directory tempfile.mkdtemp makes from here on, the push's private one included."""
    made = []

    def mkdtemp(*args, **kwargs):
        made.append(Path(_MKDTEMP(*args, **kwargs)))
        return str(made[-1])

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    return made


def _answer_pushes(monkeypatch, github, answer, inspect):
    """Every subprocess the module starts, with its argv, its environment and the number of token
    requests the stand-in GitHub had seen when it started. Local git runs for real. A push is never
    run: *inspect* sees it as it would start, and *answer* comes back as its porcelain output with
    status 0, so nothing reaches the network."""
    calls = []

    def run(argv, *args, **kwargs):
        env = dict(kwargs.get("env") or os.environ)
        calls.append((list(argv), env, len(github["token_requests"])))
        if "push" not in argv and not [a for a in argv if "github.com" in a]:
            return _RUN(argv, *args, **kwargs)
        inspect(list(argv), env)
        return subprocess.CompletedProcess(argv, 0, stdout=answer, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _subcommand(argv):
    """The git command *argv* runs: its first word after git's own options and their values."""
    words = iter(argv[1:])
    for word in words:
        if word in ("-c", "-C", "--git-dir"):
            next(words, None)
        elif not word.startswith("-"):
            return word
    return None


def _carries(argv, env):
    """Whether a git call holds the credential anywhere, in its argv or its environment."""
    return any(secret in text for secret in (TOKEN, BASIC) for text in (*argv, *env.values()))


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


def _block_sockets(monkeypatch):
    opened = []

    def no_socket(*args, **kwargs):
        opened.append(args)
        pytest.fail("a socket was created")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    return opened


class _MemorySocket:
    """Takes a token request and answers it from memory, granting what was asked for."""

    def __init__(self):
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def makefile(self, mode):
        asked = json.loads(self.sent.partition(b"\r\n\r\n")[2])
        body = json.dumps({"token": TOKEN, "permissions": asked["permissions"],
                           "repositories": [{"full_name": f"mtbitcr/{n}"} for n in asked["repositories"]]}).encode()
        return io.BytesIO(b"HTTP/1.1 201 Created\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))

    def close(self):
        pass


class _MemoryConnection(http.client.HTTPConnection):
    """A connection that opens no socket."""

    def connect(self):
        self.sock = _MemorySocket()


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

        def do_PATCH(self):
            self._route("PATCH")

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
                    "base": {"sha": BASE, "ref": STRAY}, "mergeable_state": "clean", "merge_commit_sha": BASIC})
            if method == "GET" and path.endswith("/pulls/8"):
                return self._reply(200, raw=f"<html>Bad gateway {TOKEN}</html>".encode())
            if method == "PATCH" and path.endswith("/pulls/7"):
                return self._reply(405, {"message": f"Pull Request is not mergeable {TOKEN}",
                                         "documentation_url": f"https://docs.example/{TOKEN}"})
            if method == "GET" and path.endswith("/annotations"):
                return self._reply(200, [
                    {"path": ".github", "annotation_level": "failure", "title": "Failed test",
                     "message": "tests/a.py::test_one", "raw_details": TOKEN, "blob_href": f"https://x.example/{TOKEN}"},
                    {"path": ".github", "annotation_level": "notice", "title": f"Failed test {STRAY}",
                     "message": f"count 1 {TOKEN}", "start_line": 1}])
            if path.endswith("/logs"):
                return self._reply(302, None, [("Location", f"https://storage.example/log?sig={TOKEN}")])
            if method == "GET" and path.endswith("/reviews"):  # Link: a next page (7), the last (8), none (9), cut (10)
                url = f"<https://api.github.com/repositories/1/pulls/7/reviews?page=3&t={TOKEN}>"
                link = {"7": f'{url}; rel="next", {url}; rel="last"', "8": f'{url}; rel="prev", {url}; rel="first"',
                        "10": f'{url[:-1]}; rel="next"'}.get(path.split("/")[-2])
                return self._reply(200, [{"id": 1, "state": "APPROVED", "commit_id": HEAD, "user": {"login": "b"}}],
                                   [("Link", link)] if link else ())
            if method == "GET" and "/git/ref/heads/" in path:  # one branch exists; any other is absent
                if path == f"/repos/{REPO}/git/ref/heads/delivery/card-1":
                    return self._reply(200, {"ref": "refs/heads/delivery/card-1", "node_id": TOKEN, "url": TOKEN,
                                             "object": {"sha": HEAD, "type": "commit", "url": TOKEN}})
                return self._reply(404, {"message": f"Not Found {TOKEN}"})
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
        assert entry.method in {"GET", "POST", "PUT", "PATCH"}
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
                assert exc.reason in {"step_not_permitted", "endpoint_not_allowed", "body_not_allowed"}
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
    refused = merge.request("PATCH", f"/repos/{REPO}/pulls/7", body={"state": "closed"})
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


@pytest.mark.parametrize("layout", ["production", "custom"])
def test_store_path_under_dispatcher_environment(github, monkeypatch, tmp_path, layout):
    """Plan test 17 as the owner scoped it for this slice: it covers what the transport resolves,
    the App key path. Under the exact environment the dispatcher gives a worker, that path resolves
    to the root's secrets directory, not to a profile home or a board. The transport reads no
    registry and no store, so no registry count applies; the registry part starts with slice 3."""
    mod = _transport_module(monkeypatch, github)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)  # the platform default root, as test_profiles.py moves it
    # production: the platform default root; custom: a root elsewhere, as in Docker or custom installs
    root = home / ".hermes" if layout == "production" else tmp_path / "custom-root"
    profile = root / "profiles" / "raphael-claude-worker"
    board = root / "kanban" / "boards" / "delivery"
    workspace = board / "workspaces" / "t_0001"
    for path in (profile, workspace):
        path.mkdir(parents=True)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    worker_env = {  # every variable _default_spawn sets, with the worker's own board pinned
        "HERMES_HOME": profile, "HERMES_TENANT": "raphael", "HERMES_KANBAN_TASK": "t_0001",
        "HERMES_KANBAN_WORKSPACE": workspace, "HERMES_SESSION_SOURCE": "kanban", "TERMINAL_CWD": workspace,
        "HERMES_KANBAN_BRANCH": "delivery/t_0001", "HERMES_KANBAN_RUN_ID": "1",
        "HERMES_KANBAN_CLAIM_LOCK": "dispatcher:1", "HERMES_KANBAN_GOAL_MODE": "1",
        "HERMES_KANBAN_GOAL_MAX_TURNS": "20", "TERMINAL_TIMEOUT": "3600", "TERMINAL_MAX_FOREGROUND_TIMEOUT": "3600",
        "HERMES_KANBAN_DB": board / "kanban.db", "HERMES_KANBAN_WORKSPACES_ROOT": board / "workspaces",
        "HERMES_KANBAN_BOARD": "delivery", "HERMES_PROFILE": "raphael-claude-worker",
        "XDG_RUNTIME_DIR": tmp_path / "run", "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(tmp_path / "run" / "bus"),
    }
    for name, value in worker_env.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.chdir(workspace)  # the dispatcher starts the worker inside the task workspace
    root_key, relative = _new_key(), Path("secrets") / "github-app" / "raphael-agent-factory.pem"
    for base, key in ((root, root_key), (profile, _new_key()), (board, _new_key()), (workspace, _new_key())):
        (base / relative).parent.mkdir(parents=True)
        (base / relative).write_bytes(_pem(key))
    github["public"] = root_key.public_key()  # the stand-in accepts a JWT signed with the root key only

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
                    assert exc.reason in {"step_not_permitted", "endpoint_not_allowed", "body_not_allowed"}

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
    carrying = 0
    for argv, env in calls:
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert not [a for a in argv if TOKEN in a or basic in a]
        holders = [k for k, v in env.items() if TOKEN in v or basic in v]
        if not holders:  # the init and the copy, run before any credential exists: nothing to leak
            assert "push" not in argv and not [a for a in argv if a.startswith("--force-with-lease")]
            continue
        carrying += 1
        leases = [a for a in argv if a.startswith("--force-with-lease")]
        assert len(leases) == 1 and lease.fullmatch(leases[0])
        assert not {"--force", "-f", "--mirror", "--all", "--tags", "--delete", "-d", "--prune"} & set(argv)
        assert not [a for a in argv if a.startswith("+")]
        assert f"{origin}/{REPO}.git" in argv
        config = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
        assert config[f"http.{origin}/.extraheader"] == f"Authorization: basic {basic}"
        assert config["credential.helper"] == ""
        assert len(holders) == 1 and holders[0].startswith("GIT_CONFIG_VALUE_")
    assert carrying == 5  # the five pushes above that reached git, each carrying it once
    assert not [k for k, v in os.environ.items() if TOKEN in v or basic in v]
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and basic not in text, path


def test_http_debugging_prints_neither_the_token_nor_the_app_jwt(github, monkeypatch, capfd):
    """Wire debugging switched on for every connection, as a host process may do: neither the App
    JWT the token request carries nor the installation token is printed."""
    mod = _transport_module(monkeypatch, github)
    monkeypatch.setattr(http.client.HTTPConnection, "debuglevel", 1)  # inherited by every connection

    result = mod.GitHubTransport("handoff", REPO, key_path=github["key_file"]).request("GET", f"/repos/{REPO}/pulls/7")

    assert result["status"] == 200 and result["data"]["number"] == 7
    assert len(github["jwts"]) == 1 and github["rejected_jwts"] == 0
    out, err = capfd.readouterr()
    printed = [name for name, secret in (("installation token", TOKEN), ("App JWT", github["jwts"][0]))
               if secret in out + err]
    assert printed == []


def test_push_forwards_no_git_tracing(github, monkeypatch, tmp_path):
    """Tracing switched on in the environment, in global git config or in system git config never
    records the push's credential. Each set of sinks is first shown to catch a plain git push that
    carries the same credential."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, *_) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    traces, global_config, system_config = tmp_path / "traces", tmp_path / "global.gitconfig", tmp_path / "system.gitconfig"
    traces.mkdir()
    monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_config))
    listed = ",".join(["GIT_CONFIG_PARAMETERS", *(f"GIT_CONFIG_VALUE_{n}" for n in range(10))])
    env_sinks = {name: str(traces / name) for name in (
        "GIT_TRACE", "GIT_TRACE_SETUP", "GIT_TRACE_PACKET", "GIT_TRACE_CURL", "GIT_TRACE_PERFORMANCE",
        "GIT_TRACE2", "GIT_TRACE2_PERF", "GIT_TRACE2_EVENT")}
    env_sinks.update(GIT_TRACE_REDACT="0", GIT_CURL_VERBOSE="1", GIT_TRACE2_ENV_VARS=listed,
                     GIT_TRACE2_CONFIG_PARAMS="*")
    targets = "".join(f"\t{key} = {traces / key}\n" for key in ("normalTarget", "perfTarget", "eventTarget"))
    config_sinks = f"[trace2]\n{targets}\tenvVars = {listed}\n\tconfigParams = *\n"
    plain = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"http.{origin}/.extraheader",
             "GIT_CONFIG_VALUE_0": f"Authorization: basic {BASIC}"}
    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])
    phases = ("environment", "global", "system")

    for phase in phases:
        for name, value in env_sinks.items():
            if phase == "environment":
                monkeypatch.setenv(name, value)
            else:
                monkeypatch.delenv(name, raising=False)
        global_config.write_text(config_sinks if phase == "global" else "")
        system_config.write_text(config_sinks if phase == "system" else "")
        _RUN(["git", "-C", str(work), "push", "-q", f"{origin}/{REPO}.git", f"{first}:refs/heads/plain/{phase}"],
             env={**os.environ, **plain}, check=True, capture_output=True)
        assert [p for p, text in _files_text(traces).items() if BASIC in text], phase  # these sinks would catch it
        for path in traces.iterdir():
            path.unlink()

        assert publish.push(work, f"delivery/{phase}", first, expected="")["state"] == "pushed"

        assert [p.name for p, text in _files_text(traces).items() if TOKEN in text or BASIC in text] == [], phase
        assert list(traces.iterdir()) == [], phase  # nothing traced at all, so no https trace either
    for phase in phases:
        assert _remote_head(bare, f"delivery/{phase}") == first


def test_push_runs_no_git_hook(github, monkeypatch, tmp_path):
    """A pre-push hook that would copy the credential out never runs, whether it sits in the card
    repository or in a hooks path that repository or global config names; the leased push still
    goes through."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, second, third) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    stolen = tmp_path / "stolen"  # every hook copies its environment here, credential included
    repository_hooks, global_hooks = tmp_path / "repository-hooks", tmp_path / "global-hooks"
    for hooks in (work / ".git" / "hooks", repository_hooks, global_hooks):
        hooks.mkdir(exist_ok=True)
        (hooks / "pre-push").write_text(f"#!/bin/sh\nenv >> '{stolen}'\n")
        (hooks / "pre-push").chmod(0o755)
    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])
    branch = "delivery/card-1"

    assert publish.push(work, branch, first, expected="")["head"] == first
    assert not stolen.exists(), "the card repository's own hook ran"
    _git("config", "core.hooksPath", str(repository_hooks), cwd=work)
    assert publish.push(work, branch, second, expected=first)["head"] == second
    assert not stolen.exists(), "the hook in the repository's hooks path ran"
    _git("config", "--unset", "core.hooksPath", cwd=work)
    global_config.write_text(f"[core]\n\thooksPath = {global_hooks}\n")
    assert publish.push(work, branch, third, expected=second)["head"] == third
    assert not stolen.exists(), "the hook in the global hooks path ran"

    assert _remote_head(bare, branch) == third
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and BASIC not in text, path


@pytest.mark.parametrize("transport", ["ext", "ssh"])
@pytest.mark.parametrize("scope", ["repository", "global", "system"])
def test_push_ignores_configured_url_rewrites_and_transport_commands(github, monkeypatch, tmp_path,
                                                                    scope, transport):
    """A rewrite that sends the push through a helper program — named by the card repository's own
    configuration, by global configuration or by system configuration: the helper never runs, so it
    never sees the credential, and the leased push still reaches the intended remote. Each
    configuration is first shown to run the helper for a plain push that carries a credential of the
    control's own, so the regression cannot pass vacuously."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, second, *_) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    stolen = tmp_path / "stolen"  # the helper keeps every environment it is started with
    helper = tmp_path / "helper"
    helper.write_text(f"#!/bin/sh\nenv >> '{stolen}'\nexit 1\n")  # and makes no request of its own
    helper.chmod(0o755)
    rewrites = {"ext": [("protocol.ext.allow", "always"), (f"url.ext::{helper} .insteadOf", f"{origin}/")],
                "ssh": [("core.sshCommand", str(helper)),
                        ("url.ssh://example.invalid/.insteadOf", f"{origin}/")]}
    _apply_git_config(monkeypatch, scope, work, tmp_path, rewrites[transport])
    branch = "delivery/card-1"

    plain = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"http.{origin}/.extraheader",
             "GIT_CONFIG_VALUE_0": f"Authorization: basic {DUMMY_BASIC}"}
    _RUN(["git", "-C", str(work), "push", "--porcelain", f"{origin}/{REPO}.git", f"{first}:refs/heads/plain"],
         env={**os.environ, **plain}, capture_output=True)
    assert stolen.exists() and DUMMY_BASIC in stolen.read_text()  # this helper would catch a credential
    stolen.unlink()

    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])
    try:
        outcome = publish.push(work, branch, second, expected="")
    except mod.GitHubTransportError as exc:  # whatever escapes is reported beside the leak itself
        outcome = exc.reason

    held = sorted(path.name for path, text in _files_text(tmp_path).items() if TOKEN in text or BASIC in text)
    assert (held, stolen.exists()) == ([], False)  # no configured helper ran, and no file holds it
    assert outcome == {"state": "pushed", "branch": branch, "head": second}
    assert _remote_refs(bare) == [f"refs/heads/{branch}"]
    assert list(Path(tempfile.gettempdir()).glob("hermes-delivery-push-*")) == []  # nothing left behind


def test_push_cannot_be_redirected_to_another_remote_of_the_same_scheme(github, monkeypatch, tmp_path):
    """Rewrites that keep GIT_ORIGIN's own scheme, so restricting the transport cannot refuse them:
    all three of url.insteadOf, url.pushInsteadOf and remote.<the push URL>.pushurl name a second
    bare remote. The decoy receives nothing and the intended remote receives the branch."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, *_) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    decoy_root = tmp_path / "decoy"
    decoy = decoy_root / "mtbitcr" / "hermes-agent.git"
    _git("init", "-q", "--bare", str(decoy))
    _apply_git_config(monkeypatch, "repository", work, tmp_path, [
        (f"url.{decoy_root.as_uri()}/.insteadOf", f"{origin}/"),
        (f"url.{decoy_root.as_uri()}/.pushInsteadOf", f"{origin}/"),
        (f"remote.{origin}/{REPO}.git.pushurl", decoy.as_uri())])
    branch = "delivery/card-1"

    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])
    try:
        outcome = publish.push(work, branch, first, expected="")
    except mod.GitHubTransportError as exc:
        outcome = exc.reason

    assert _remote_refs(decoy) == []  # the decoy never became the push's destination
    assert outcome == {"state": "pushed", "branch": branch, "head": first}
    assert _remote_refs(bare) == [f"refs/heads/{branch}"]
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and BASIC not in text, path


@pytest.mark.parametrize("scope", ["repository", "global"])
def test_push_publishes_no_ref_beside_the_leased_branch(github, monkeypatch, tmp_path, scope):
    """push.followTags in the card repository's own configuration or in global configuration: the
    push writes exactly one remote ref, the leased branch, and the annotated tag sitting on the very
    commit being pushed stays unpublished."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, *_) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "tag", "-a", "-m", "t", "not-leased",
         first, cwd=work)
    _apply_git_config(monkeypatch, scope, work, tmp_path, [("push.followTags", "true")])
    branch = "delivery/probe"

    pushed = mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).push(
        work, branch, first, expected="")

    assert pushed == {"state": "pushed", "branch": branch, "head": first}
    assert _remote_refs(bare) == [f"refs/heads/{branch}"]  # read back from the remote, not from output


def test_push_follows_the_owner_rule(github, monkeypatch, tmp_path):
    """The owner's rule for the credential-bearing push, as the next review judges it, with the
    production GIT_ORIGIN and host variables planted that would redirect or widen the push if
    inherited. The init and the copy run for real; the push is answered here and never run. The
    commit reaches a fresh bare repository in a private temporary directory before any token exists.
    Then exactly one git holds the credential, the push: it reads no git configuration the transport
    does not set, may speak only https, and names one URL, one refspec and its lease. The directory
    is gone afterwards."""
    mod = _transport_module(monkeypatch, github)
    assert mod.GIT_ORIGIN == "https://github.com"  # the production origin, so the push below is never run
    work, _, _, (first, second, _) = _repositories(tmp_path)
    helper = tmp_path / "helper"
    helper.write_text(f"#!/bin/sh\nenv >> '{tmp_path / 'stolen'}'\nexit 1\n")
    helper.chmod(0o755)
    hostile = tmp_path / "hostile.gitconfig"  # a rewrite through the helper, and tags that follow the branch
    _git("config", "--file", str(hostile), f"url.ext::{helper} .insteadOf", "https://github.com/")
    _git("config", "--file", str(hostile), "push.followTags", "true")
    planted = {
        "GIT_ALLOW_PROTOCOL": "ext:https", "GIT_SSH_COMMAND": str(helper), "GIT_SSH": str(helper),
        "GIT_PROXY_COMMAND": str(helper), "GIT_ASKPASS": str(helper), "SSH_ASKPASS": str(helper),
        "GIT_CONFIG_PARAMETERS": f"'url.ext::{helper} .insteadof'='https://github.com/' 'push.followtags'='true'",
        "GIT_CONFIG_SYSTEM": str(hostile), "GIT_DIR": str(work / ".git"), "HTTPS_PROXY": "http://127.0.0.1:9",
        "GIT_SSL_NO_VERIFY": "1", "GIT_TRACE": str(tmp_path / "trace"), "GIT_TRACE_CURL": str(tmp_path / "curl"),
        "GIT_CURL_VERBOSE": "1", "SSLKEYLOGFILE": str(tmp_path / "tls-keys.log")}
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))  # to be replaced by the push's own empty file
    made = _record_mkdtemp(monkeypatch)
    branch = "delivery/card-1"
    ref = f"refs/heads/{branch}"
    at_push = {}

    def inspect(argv, env):  # the moment the push would start
        temp = made[0] if len(made) == 1 else None
        git_dir = argv[argv.index("--git-dir") + 1] if "--git-dir" in argv else str(work / ".git")
        clean = {"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

        def read(*args):
            return _RUN(["git", "--git-dir", git_dir, *args], env=clean, capture_output=True, text=True).stdout.split()

        global_file = Path(env.get("GIT_CONFIG_GLOBAL") or os.devnull)
        at_push.update(
            bare=read("rev-parse", "--is-bare-repository"), history=read("rev-list", second),
            config=global_file.read_bytes() if global_file.is_file() else None,
            executables=None if temp is None else [
                p.name for p in temp.rglob("*") if p.is_file() and os.access(p, os.X_OK)],
            holders=None if temp is None else [
                p.name for p, text in _files_text(temp).items() if TOKEN in text or BASIC in text])

    answer = f"To https://github.com/{REPO}.git\n*\t{second}:{ref}\t[new branch]\nDone\n".encode()
    calls = _answer_pushes(monkeypatch, github, answer, inspect)

    pushed = mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).push(work, branch, second, expected="")

    carrying = [(argv, env) for argv, env, _ in calls if _carries(argv, env)]
    assert [_subcommand(argv) for argv, _ in carrying] == ["push"]  # exactly one git holds the credential
    ((argv, env),) = carrying
    git_dir = argv[argv.index("--git-dir") + 1] if "--git-dir" in argv else None
    # Finding 1 of the security review: the program is an absolute git, never a bare name.
    assert os.path.isabs(argv[0]) and os.path.basename(argv[0]) == "git"
    assert argv[1:argv.index("push")] == ["-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
                                          "--git-dir", git_dir]
    assert len(made) == 1 and made[0] in Path(git_dir).parents  # a fresh repository in one private directory
    temp = made[0]
    options = [a for a in argv[argv.index("push") + 1:] if a.startswith("-")]
    assert sorted(options) == sorted(["--porcelain", "--no-follow-tags", f"--force-with-lease={ref}:"])
    assert [a for a in argv[argv.index("push") + 1:] if not a.startswith("-")] == [
        f"https://github.com/{REPO}.git", f"{second}:{ref}"]  # one URL and exactly one refspec
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert temp in Path(env["GIT_CONFIG_GLOBAL"]).parents and at_push["config"] == b""  # an empty file of its own
    assert env["HOME"] == env["XDG_CONFIG_HOME"] == str(temp)
    assert [name for name in planted if name in env] == []
    holders = [name for name, value in env.items() if TOKEN in value or BASIC in value]
    assert len(holders) == 1 and holders[0].startswith("GIT_CONFIG_VALUE_")
    config = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
    assert config["http.https://github.com/.extraheader"] == f"Authorization: basic {BASIC}"

    assert at_push["bare"] == ["true"] and at_push["history"] == [second, first]  # the commit, with its history
    assert at_push["executables"] == [] and at_push["holders"] == []  # no hook to run, no file with the credential

    assert [_subcommand(argv) for argv, _, _ in calls] == ["init", "fetch", "push"]
    (init, init_env, _), (copy, copy_env, requests_at_copy), (_, _, requests_at_push) = calls
    assert (requests_at_copy, requests_at_push, len(github["token_requests"])) == (0, 1, 1)  # minted after the copy
    assert not _carries(init, init_env) and not _carries(copy, copy_env)
    assert {"--bare", "--quiet", "--template=", git_dir} <= set(init)
    assert {("-c", "protocol.allow=never"), ("-c", "protocol.file.allow=always"), ("--git-dir", git_dir)} <= set(
        zip(copy, copy[1:]))
    assert {"--no-tags", "--no-write-fetch-head", str(work)} <= set(copy)
    assert copy[-1].split(":")[0] == second  # exactly the commit, from the worktree's absolute path
    for other in (init_env, copy_env):  # every git the push starts reads only what the transport sets
        assert [other.get(name) for name in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "HOME", "XDG_CONFIG_HOME")] == [
            env[name] for name in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "HOME", "XDG_CONFIG_HOME")]
        assert [name for name in planted if name in other] == []

    assert pushed == {"state": "pushed", "branch": branch, "head": second}
    assert not temp.exists()  # the repository, its copy of the commit and the empty config are gone


def test_push_of_a_commit_missing_from_the_worktree_requests_no_token(github, monkeypatch, tmp_path):
    """The commit is not in the worktree, so the copy fails: push_failed before any token request,
    with no push started and the temporary directory gone."""
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, _ = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    made = _record_mkdtemp(monkeypatch)
    calls = _record_git(monkeypatch)

    with pytest.raises(mod.GitHubTransportError) as failed:
        mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).push(
            work, "delivery/card-1", "c" * 40, expected="")

    assert failed.value.reason == "push_failed"
    assert github["token_requests"] == []  # the copy failed before any credential existed
    assert [argv for argv, _ in calls if _subcommand(argv) == "push"] == []
    assert len(made) == 1 and not made[0].exists()
    assert _remote_refs(bare) == []


def test_push_with_undecodable_git_output_yields_only_fixed_results(github, monkeypatch, tmp_path):
    """git output that is not UTF-8 and echoes the credential: a push refused on its lease and an
    accepted one still end in a fixed reason or a fixed result that carry none of that output."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, second, third) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    branch = "delivery/card-1"
    _git("push", "-q", f"{origin}/{REPO}.git", f"{first}:refs/heads/{branch}", cwd=work)
    real_git, stand_in = shutil.which("git"), tmp_path / "bin" / "git"  # the real git, wrapped in malformed output
    echo = f"printf '\\377\\376 {TOKEN} Authorization: basic {BASIC}\\n'"
    stand_in.parent.mkdir()
    stand_in.write_text(f"#!/bin/sh\n{echo}\n{echo} >&2\n{real_git} \"$@\"\nstatus=$?\n{echo}\n{echo} >&2\nexit $status\n")
    stand_in.chmod(0o755)
    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])
    outcomes = []

    with monkeypatch.context() as patch:
        patch.setenv("PATH", os.pathsep.join([str(stand_in.parent), os.environ["PATH"]]))  # the stand-in first
        for head, expected in ((second, third), (second, first)):  # refused on its lease, then accepted
            try:
                outcomes.append(publish.push(work, branch, head, expected=expected))
            except Exception as exc:  # whatever escapes is examined below
                outcomes.append(exc)

    for outcome in outcomes:
        text = _error_text(outcome) if isinstance(outcome, BaseException) else repr(outcome)
        assert TOKEN not in text and BASIC not in text, type(outcome).__name__
    refused, accepted = outcomes
    assert isinstance(refused, mod.GitHubTransportError) and refused.reason == "push_lease_mismatch"
    assert accepted == {"state": "pushed", "branch": branch, "head": second}
    assert _remote_head(bare, branch) == second


def test_tls_and_git_write_no_key_log(monkeypatch, tmp_path):
    """SSLKEYLOGFILE set: the module's own TLS context logs no keys and the push's git does not
    inherit the setting. Constructing an HTTPS connection does not connect, so no socket opens."""
    mod = _module()
    opened = _block_sockets(monkeypatch)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "tls-keys.log"))
    context = mod._connect()._context  # the context the module's own connection would handshake with
    key_file = tmp_path / "app.pem"
    key_file.write_bytes(_pem(_new_key()))
    monkeypatch.setattr(mod, "_connect", lambda: _MemoryConnection("github.invalid"))  # mints without a socket
    work, bare, origin, (first, *_) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    calls = _record_git(monkeypatch)

    pushed = mod.GitHubTransport("publish", REPO, key_path=key_file).push(work, "delivery/card-1", first, expected="")

    assert pushed["head"] == first and _remote_head(bare, "delivery/card-1") == first
    assert calls
    inherited = [env["SSLKEYLOGFILE"] for _, env in calls if "SSLKEYLOGFILE" in env]
    assert (context.keylog_filename, inherited) == (None, [])
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname  # still a verifying context
    assert opened == []


def test_commit_statuses_are_refused_before_any_socket_or_token(monkeypatch, tmp_path):
    """The owner's call: no step reads commit statuses. The read is refused as outside the
    allowlist, before any socket or token, and no step's token names the permission."""
    mod = _module()
    opened = _block_sockets(monkeypatch)
    missing_key = tmp_path / "no-such-key.pem"  # minting would read it first, so a refusal here minted nothing
    paths = (f"/repos/{REPO}/commits/{HEAD}/statuses", f"/repos/{REPO}/commits/{HEAD}/status",
             f"/repos/{REPO}/statuses/{HEAD}")
    for step, permissions in mod.STEP_PERMISSIONS.items():
        assert "statuses" not in permissions, step
        transport = mod.GitHubTransport(step, REPO, key_path=missing_key)
        for path in paths:
            for query in (None, {"per_page": "100"}):
                with pytest.raises(mod.GitHubTransportError) as refused:
                    transport.request("GET", path, query=query)
                assert refused.value.reason == "endpoint_not_allowed", (step, path, query)
    assert opened == []


def test_the_annotations_read_is_the_one_new_call_and_the_unused_merge_and_review_posts_are_gone(github, monkeypatch):
    """The red check card adds one read: the annotations of one check run in the bound repository, with
    checks read, each annotation's title and message only. The owner decisions of 2026-10-04 removed the
    unused REST merge and review post, so no step reaches them, and every other path and method near the
    new read stays refused before any token or socket."""
    mod = _transport_module(monkeypatch, github)
    entries = {(e.method, e.template) for e in mod.REST_ALLOWLIST}
    assert ("GET", "/repos/{repo}/check-runs/{id}/annotations") in entries
    assert not entries & {("PUT", "/repos/{repo}/pulls/{number}/merge"), ("POST", "/repos/{repo}/pulls/{number}/reviews")}
    path = f"/repos/{REPO}/check-runs/9/annotations"
    outside = [("PUT", f"/repos/{REPO}/pulls/7/merge"), ("POST", f"/repos/{REPO}/pulls/7/reviews"),
               ("GET", f"/repos/{SIBLING}/check-runs/9/annotations"), ("GET", f"/repos/{REPO}/check-runs/09/annotations"),
               ("GET", f"/repos/{REPO}/check-runs/x/annotations"), ("GET", f"{path}/1"), ("GET", f"{path}/"),
               ("GET", f"/repos/{REPO}/check-runs/9"), ("GET", f"/repos/{REPO}/check-suites/9/annotations"),
               *((method, path) for method in ("POST", "PUT", "PATCH", "DELETE", "get"))]
    for step, permissions in mod.STEP_PERMISSIONS.items():
        transport = mod.GitHubTransport(step, REPO, key_path=github["key_file"])
        for method, target in outside:
            with pytest.raises(mod.GitHubTransportError) as refused:
                transport.request(method, target, body={})
            assert refused.value.reason == "endpoint_not_allowed", (step, method, target)
        for query in ({"per_page": "101"}, {"filter": "latest"}, {"page": "0"}):
            with pytest.raises(mod.GitHubTransportError) as refused:
                transport.request("GET", path, query=query)
            assert refused.value.reason == "query_not_allowed", (step, query)
        if "checks" not in permissions:
            with pytest.raises(mod.GitHubTransportError) as refused:
                transport.request("GET", path)
            assert refused.value.reason == "step_not_permitted", step
    assert github["calls"] == [] and github["token_requests"] == []

    read = mod.GitHubTransport("read_checks", REPO, key_path=github["key_file"]).request(
        "GET", path, query={"per_page": 100, "page": 1})

    assert read["status"] == 200 and read["data"][0] == {"title": "Failed test", "message": "tests/a.py::test_one"}
    assert [sorted(note) for note in read["data"]] == [["message", "title"]] * 2
    assert TOKEN not in repr(read) and STRAY not in repr(read)
    assert github["calls"] == [("GET", f"{path}?per_page=100&page=1")]


def test_push_starts_no_git_from_the_working_directory(github, monkeypatch, tmp_path):
    """Security review of PR 141, finding 1: a git program in the caller's working directory,
    reached through an empty or relative PATH entry, never runs, and the leased push still works."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, (first, _second, _third) = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    card = tmp_path / "card"
    card.mkdir()
    stolen = tmp_path / "stolen"  # the planted git copies its environment here, credential included
    (card / "git").write_text(f"#!/bin/sh\nenv >> '{stolen}'\nexit 1\n")
    (card / "git").chmod(0o755)
    path = os.environ["PATH"]
    monkeypatch.chdir(card)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", path]))
    publish = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])

    assert publish.push(work, "delivery/card-1", first, expected="")["head"] == first
    assert not stolen.exists(), "a git from the working directory ran"
    # The readback below starts the test's own git, so it leaves the planted directory first.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", path)
    assert _remote_head(bare, "delivery/card-1") == first


def test_the_api_connection_opens_no_tls_key_log(monkeypatch, tmp_path):
    """Security review of PR 141, finding 2: SSLKEYLOGFILE names a key log; building the API
    connection neither opens nor creates it, and the connection still verifies the server."""
    keylog = tmp_path / "tls-keys.log"
    monkeypatch.setenv("SSLKEYLOGFILE", str(keylog))
    connection = _module()._connect()
    try:
        assert not keylog.exists()
        assert connection._context.verify_mode == ssl.CERT_REQUIRED
        assert connection._context.check_hostname is True
    finally:
        connection.close()


def test_branch_head_reads_the_commit_the_branch_holds(github, monkeypatch):
    """Finding 1 of the slice 4 review: the one branch read goes through the allowlist and the
    step's one narrowed token, and returns the head SHA of the branch."""
    mod = _transport_module(monkeypatch, github)

    head = mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).branch_head("delivery/card-1")

    assert head == HEAD
    assert github["calls"] == [("GET", f"/repos/{REPO}/git/ref/heads/delivery/card-1")]
    assert github["token_requests"] == [
        {"repositories": ["hermes-agent"], "permissions": {"contents": "write", "pull_requests": "write"}}]


def test_branch_head_is_none_when_github_answers_404(github, monkeypatch):
    """The same read for a branch GitHub does not have: GitHub answers 404 and the read returns None,
    the branch's absence."""
    mod = _transport_module(monkeypatch, github)

    head = mod.GitHubTransport("publish", REPO, key_path=github["key_file"]).branch_head("delivery/card-2")

    assert head is None
    assert github["calls"] == [("GET", f"/repos/{REPO}/git/ref/heads/delivery/card-2")]


@pytest.mark.parametrize("read, sends", [({"state": "open", "head": {"sha": HEAD}}, True), (None, False),
                                         ({"state": "closed", "head": {"sha": HEAD}}, False),
                                         ({"state": "open", "head": {"sha": "f" * 40}}, False)])
def test_the_arm_reads_nothing_and_sends_only_on_the_transports_own_read_of_the_open_head(monkeypatch, read, sends):
    """Owner rule 2 of round 2: the arm makes no read and mints no token. It sends the one mutation, with the
    token the transport's earlier reads minted, only when its own read of the pull request showed it open at
    the head. A reviews read keeps each review's author login, state, commit and body's first line only."""
    mod = _module()
    sent, mints = [], []
    answers = {f"/repos/{REPO}/pulls/41": dict(read or {}, number=41, node_id="PR_kw"),
               f"/repos/{REPO}/pulls/41/reviews": [{"id": 3, "state": "APPROVED", "commit_id": HEAD, "html_url": STRAY,
                                                    "body": "Review card t_1 \r\nfine", "user": {"login": "b", "id": 9}}]}
    monkeypatch.setattr(mod, "_exchange", lambda method, target, authorization, payload=None: (
        sent.append((method, target.split("?")[0], authorization)) or (200, answers.get(target.split("?")[0], {}))))
    monkeypatch.setattr(mod.GitHubTransport, "_installation_token", lambda self: (
        mints.append(self) or setattr(self, "_token", DUMMY) or DUMMY))
    github = mod.GitHubTransport("publish", REPO)
    assert github.request("GET", f"/repos/{REPO}/pulls/41/reviews", query={"per_page": 100})["data"] == [
        {"user": {"login": "b"}, "state": "APPROVED", "commit_id": HEAD, "body": "Review card t_1"}]
    if read is not None:
        github.request("GET", f"/repos/{REPO}/pulls/41")
    reads = len(sent)

    if sends:
        github.arm_auto_merge(41, HEAD)
    for number, head in ((41, BASE), (42, HEAD)) + (() if sends else ((41, HEAD),)):
        with pytest.raises(mod.GitHubTransportError):
            github.arm_auto_merge(number, head)

    assert sent[reads:] == ([("POST", "/graphql", f"Bearer {DUMMY}")] if sends else []) and len(mints) == reads


def test_the_reviews_read_says_only_whether_github_names_a_next_page(github, monkeypatch):
    """The fenced reviews read keeps one fact of the answer's Link header, whether it names a next page
    (rel="next"): True when it does, False when it names none or sends no Link header, None when it cannot be read.
    The header itself, its URLs and the token they carry included, never comes back."""
    mod = _transport_module(monkeypatch, github)
    transport = mod.GitHubTransport("publish", REPO, key_path=github["key_file"])

    answers = [transport.request("GET", f"/repos/{REPO}/pulls/{number}/reviews", query={"per_page": 100, "page": 2})
               for number in (7, 8, 9, 10)]

    assert answers == [{"status": 200, "data": [{"user": {"login": "b"}, "state": "APPROVED", "commit_id": HEAD,
                                                 "body": ""}], "next_page": more} for more in (True, False, False, None)]
