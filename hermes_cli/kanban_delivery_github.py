"""GitHub App access for delivery: narrowed installation tokens, a fixed REST allowlist and leased
pushes, with no token in any result, error, log or file.

Nothing calls this yet. A transport serves one step on one repository. It signs the App JWT
in-process from the key file and mints one installation token naming only that step's permissions
and that repository. Every call is checked against REST_ALLOWLIST before any token or connection
exists, and results carry fixed fields only.

The push that carries the credential runs from a fresh bare repository in a private temporary
directory, made for that one push and removed after it. The commit is copied in from the card worktree
before any credential exists. The push then reads no git configuration the transport does not set,
inherits no variable from the host beyond what starting a process needs, may speak only GIT_ORIGIN's
own protocol, and names one URL, one refspec and its lease. So no hook, trace, URL rewrite, custom
transport or push setting of the worktree can reach it.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import time
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from urllib.parse import urlencode, urlsplit

import jwt

from agent.redact import redact_sensitive_text
from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_constants import get_default_hermes_root

APP_ID = "4485539"  # a string: GitHub requires the JWT issuer to be one
INSTALLATION_ID = 151217494
API_HOST = "api.github.com"
GIT_ORIGIN = "https://github.com"

# The only schemes a push may speak, and the only ones its git is allowed to use: https in
# production, file where a test points GIT_ORIGIN at a local root. ext::, ssh, git and plain http are
# refused, so no configured rewrite or helper program can stand in for the transport.
_PUSH_SCHEMES = ("https", "file")
# The whole environment of every git this module starts is built from these names: locations the
# operating system needs to start a process at all. Nothing git reads as policy is carried over: no
# configuration, transport, proxy, trust-store, askpass or trace variable can arrive from the host.
_GIT_ENV_NAMES = ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "TMPDIR")

# Callers name a step, never a permission map, so these are the only token requests there are.
# On GitHub write includes read, and every token gets metadata read without asking.
STEP_PERMISSIONS = MappingProxyType({
    "publish": MappingProxyType({"contents": "write", "pull_requests": "write"}),  # T1
    "read_checks": MappingProxyType({"checks": "read", "actions": "read", "pull_requests": "read"}),  # T2
    "rerun_flaky": MappingProxyType({"actions": "write"}),  # T3
    "merge": MappingProxyType({"contents": "write", "pull_requests": "write"}),  # T6
    "handoff": MappingProxyType({"pull_requests": "read"}),  # T7
})

_COMPONENT = r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
_BRANCH = re.compile(rf"(?!.*\.\.)(?!.*\.lock(?:/|$)){_COMPONENT}(?:/{_COMPONENT})*")
_SHA = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9-]+/(?!\.\.?$)[A-Za-z0-9_.-]+")
_TOKEN = re.compile(r"[A-Za-z0-9_.-]+")  # it travels in a header, so nothing that could split one
_SEGMENTS = {"number": r"[1-9][0-9]{0,9}", "id": r"[1-9][0-9]{0,19}", "sha": _SHA.pattern, "branch": _BRANCH.pattern}
_PAGE = {"per_page": r"[1-9][0-9]?|100", "page": r"[1-9][0-9]{0,3}"}


class Endpoint(NamedTuple):
    method: str
    template: str
    permission: tuple[str, str] | None  # what the step must hold; None for the App's own token call
    query: MappingProxyType = MappingProxyType({})  # allowed parameters, each with its value pattern
    required_query: tuple[str, ...] = ()
    body: object = None  # when set, the one test a body must pass: the call sends nothing else


_TOKEN_ENDPOINT = Endpoint("POST", "/app/installations/{installation}/access_tokens", None)

# GitHub's REST interface has no call that arms auto-merge, so this fixed mutation is the one GraphQL
# document there is: a merge commit, only while the pull request still holds the head the caller names.
# Only GitHubTransport.arm_auto_merge sends it, with variables it builds: request() never reaches it.
AUTO_MERGE_MUTATION = (
    "mutation($pullRequestId: ID!, $expectedHeadOid: GitObjectID!) { enablePullRequestAutoMerge(input: "
    "{pullRequestId: $pullRequestId, expectedHeadOid: $expectedHeadOid, mergeMethod: MERGE}) { clientMutationId } }")
_NODE_ID = re.compile(r"[A-Za-z0-9_=-]{1,100}")


def _arm_body(body) -> bool:
    variables = body.get("variables") if isinstance(body, dict) else None
    return (isinstance(body, dict) and body.keys() == {"query", "variables"} and body["query"] == AUTO_MERGE_MUTATION
            and isinstance(variables, dict) and variables.keys() == {"pullRequestId", "expectedHeadOid"}
            and isinstance(variables["pullRequestId"], str) and bool(_NODE_ID.fullmatch(variables["pullRequestId"]))
            and _is_sha(variables["expectedHeadOid"]))


def _close_body(body) -> bool:
    return isinstance(body, dict) and body == {"state": "closed"}


GRAPHQL_ALLOWLIST = (Endpoint("POST", "/graphql", ("pull_requests", "write"), body=_arm_body),)

# The plan's endpoint list and nothing else: no protection, rules, settings, hooks, collaborators,
# keys or installation management, and no other GraphQL. Commit statuses are left out by the owner's call:
# the required checks of both repositories are check runs and the App holds no commit-statuses
# permission, so the transport offers no statuses read and a later merge fence takes an empty
# statuses list.
REST_ALLOWLIST = (
    Endpoint("GET", "/repos/{repo}/pulls", ("pull_requests", "read"), MappingProxyType(
        {"state": "open|closed|all", "head": rf"[A-Za-z0-9-]+:{_BRANCH.pattern}", "base": _BRANCH.pattern, **_PAGE})),
    Endpoint("POST", "/repos/{repo}/pulls", ("pull_requests", "write")),
    Endpoint("GET", "/repos/{repo}/pulls/{number}", ("pull_requests", "read")),
    # The close of a pull request a rework card replaced: its state, and no other field.
    Endpoint("PATCH", "/repos/{repo}/pulls/{number}", ("pull_requests", "write"), body=_close_body),
    # The reviews auto-merge is armed on, read through this same fence.
    Endpoint("GET", "/repos/{repo}/pulls/{number}/reviews", ("pull_requests", "read"), MappingProxyType(_PAGE)),
    # T1's one branch read: the commit a head branch holds, and 404 when the branch does not exist.
    Endpoint("GET", "/repos/{repo}/git/ref/heads/{branch}", ("contents", "read")),
    Endpoint("GET", "/repos/{repo}/commits/{sha}/check-runs", ("checks", "read"),
             MappingProxyType({"filter": "latest|all", **_PAGE})),
    # The failed tests a red slice job names on its own check run (an Actions job's id is its check run's).
    Endpoint("GET", "/repos/{repo}/check-runs/{id}/annotations", ("checks", "read"), MappingProxyType(_PAGE)),
    Endpoint("GET", "/repos/{repo}/actions/runs", ("actions", "read"),
             MappingProxyType({"head_sha": _SHA.pattern, **_PAGE}), ("head_sha",)),
    Endpoint("GET", "/repos/{repo}/actions/runs/{id}/jobs", ("actions", "read"),
             MappingProxyType({"filter": "latest|all", **_PAGE})),
    # GitHub answers with a redirect to signed log storage, which is never followed.
    Endpoint("GET", "/repos/{repo}/actions/jobs/{id}/logs", ("actions", "read")),
    Endpoint("POST", "/repos/{repo}/actions/jobs/{id}/rerun", ("actions", "write")),
    _TOKEN_ENDPOINT,
)

# Fixed fields only: ids, SHAs, refs, states and counts, and of a user only its login (a review's
# author). Titles, bodies (but a listed review's first line), messages (but a check run annotation's
# title and message) and URLs never pass, nor headers.
_FIELDS = frozenset({
    "id", "number", "name", "state", "status", "conclusion", "merged", "mergeable", "mergeable_state",
    "sha", "head_sha", "merge_commit_sha", "commit_id", "ref", "head", "base", "app", "object",
    "total_count", "check_runs", "workflow_runs", "jobs", "steps", "run_id", "run_attempt",
    "node_id", "data", "errors", "enablePullRequestAutoMerge", "clientMutationId", "user", "login",
})
_MAX_TEXT = 256


class GitHubTransportError(Exception):
    """A refusal or failure named by a fixed reason code, never by a request, response or token."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _connect() -> http.client.HTTPConnection:
    # http.client follows no redirect and reads no proxy or netrc setting; httpcore's DEBUG trace
    # would log response headers. Its wire debugging is switched off in _exchange.
    # Built directly: create_default_context opens the file that SSLKEYLOGFILE names. This context
    # checks the hostname and requires a verified certificate, as create_default_context does.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    return http.client.HTTPSConnection(API_HOST, timeout=30, context=context)


def _names_next_page(link: str | None) -> bool | None:
    """Whether GitHub's Link header names a next page (rel="next"): False when it names none or
    there is none, None when it cannot be read. Only this fact leaves the transport, never the header."""
    if link is None:
        return False
    rels = [re.fullmatch(r'\s*<[^<>]*>\s*;\s*rel="([\w ]+)"\s*', value) for value in link.split(",")]
    if not all(rels):
        return None
    return any("next" in rel[1].lower().split() for rel in rels)


def _exchange(method: str, target: str, authorization: str, payload=None) -> tuple[int, object, bool | None]:
    """One request to GitHub: the status, the decoded body of a 2xx answer (else None), and whether
    its Link header names a next page (:func:`_names_next_page`). No header is passed on."""
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "hermes-kanban-delivery", "Authorization": authorization}
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection = _connect()
    # Any level above 0, which a host process may set as the class default, prints every header,
    # Authorization included; the response takes its level from the connection.
    connection.set_debuglevel(0)
    try:
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        status, raw, more = response.status, response.read(), _names_next_page(response.getheader("Link"))
    except (OSError, http.client.HTTPException):
        status = raw = more = None
    finally:
        connection.close()
    # Raised after the handler, not inside it: a chained IncompleteRead or JSONDecodeError would
    # carry the body, which for the token call holds the token.
    if status is None:
        raise GitHubTransportError("network_error")
    if not 200 <= status < 300 or not raw:
        return status, None, more  # GitHub's error text and redirect targets are never passed on
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if not isinstance(value, (dict, list)):
        raise GitHubTransportError("bad_response")
    return status, value, more


def _app_jwt(key_path: Path) -> str:
    """The App's JWT, signed here from the key file: no helper program and no copy of the key."""
    now = int(time.time())
    try:
        key = key_path.read_bytes()
    except OSError:
        raise GitHubTransportError("app_key_unavailable") from None
    try:
        # Backdated for clock drift; GitHub refuses a JWT that lives longer than ten minutes.
        return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": APP_ID}, key, algorithm="RS256")
    except (ValueError, TypeError, jwt.PyJWTError):
        raise GitHubTransportError("app_key_invalid") from None


def _path_pattern(template: str, repository: str) -> re.Pattern[str]:
    """An anchored pattern for one template, where {repo} matches only the bound repository."""
    pattern = re.escape(template).replace(re.escape("{repo}"), re.escape(repository))
    for name, segment in _SEGMENTS.items():
        pattern = pattern.replace(re.escape(f"{{{name}}}"), segment)
    return re.compile(pattern)


def _holds(permissions, needed: tuple[str, str]) -> bool:
    name, level = needed
    return permissions.get(name) in ("write", level)


def _is_sha(value) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _basic(token: str) -> str:
    """The token as git's Basic authorization header carries it."""
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


def _git_base_env(home: str) -> dict[str, str]:
    """The environment of every git a push starts, built name by name rather than as a copy of
    os.environ with entries removed. *home* is the push's private directory, and the only git
    configuration file read there is the empty one the transport made."""
    env = noninteractive_git_env({name: os.environ[name] for name in _GIT_ENV_NAMES if name in os.environ})
    env["GIT_CONFIG_NOSYSTEM"] = "1"  # the system file is not read, whatever GIT_CONFIG_SYSTEM says
    env["GIT_CONFIG_GLOBAL"] = os.path.join(home, ".gitconfig")  # nor ~/.gitconfig: this empty file instead
    env["HOME"] = env["XDG_CONFIG_HOME"] = home  # nor the XDG file, nor the ~/.netrc that curl reads
    return env


def _run_git(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess | None:
    """One git with nothing to read on stdin; None when it could not start or did not finish.

    The program is the absolute git found on the absolute PATH entries only, and it runs in the
    push's private directory (*env*'s HOME), so no git in the caller's working directory or in a
    relative PATH entry can start."""
    search = os.pathsep.join(entry for entry in env.get("PATH", "").split(os.pathsep) if os.path.isabs(entry))
    program = shutil.which("git", path=search)
    if not program or not os.path.isabs(program):
        return None
    try:
        return subprocess.run([program, *argv[1:]], env=env, cwd=env["HOME"], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None


def _copy_commit(home: str, worktree: str | Path, sha: str, ref: str) -> str:
    """A fresh bare repository in *home* holding commit *sha* and its history under *ref*, copied
    from the worktree before any credential exists. Of the gits a push starts, only the one that
    serves this copy reads the worktree's own configuration."""
    env = _git_base_env(home)
    repository = os.path.join(home, "push.git")
    try:
        Path(env["GIT_CONFIG_GLOBAL"]).touch(exist_ok=False)  # the global configuration: empty
    except OSError:
        raise GitHubTransportError("push_failed") from None
    # An empty --template= copies no sample hook or other file: the repository holds what init writes.
    init = ["git", "init", "--bare", "--quiet", "--template=", repository]
    # Only the file protocol, from the worktree's absolute path, for exactly this commit and no tag.
    fetch = ["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "--git-dir", repository,
             "fetch", "--no-tags", "--no-write-fetch-head", os.path.abspath(worktree), f"{sha}:{ref}"]
    for argv in (init, fetch):
        result = _run_git(argv, env)
        if result is None or result.returncode != 0:
            raise GitHubTransportError("push_failed")
    return repository


class GitHubTransport:
    """One step's GitHub access to one repository, through one narrowed installation token."""

    def __init__(self, step: str, repository: str, *, key_path: str | Path | None = None):
        if not isinstance(step, str) or step not in STEP_PERMISSIONS:
            raise GitHubTransportError("unknown_step")
        if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
            raise GitHubTransportError("bad_repository")
        self.step = step
        self.repository = repository
        self._permissions = STEP_PERMISSIONS[step]
        # One App key serves the whole machine, so it lives under the Hermes root, never under the
        # profile HERMES_HOME a worker runs with.
        self._key_path = Path(key_path) if key_path is not None else (
            get_default_hermes_root() / "secrets" / "github-app" / "raphael-agent-factory.pem")
        self._routes = [(e, _path_pattern(e.template, repository)) for e in REST_ALLOWLIST if e.permission]
        self._token: str | None = None
        self._pull: tuple | None = None  # number, state, head and node id, as this transport last read them

    def request(self, method: str, path: str, *, query=None, body=None) -> dict:
        """Call one allowlisted endpoint. Returns {"status", "data"}, where data holds the fixed
        fields of a 2xx JSON answer and is None for anything else. A listed page of reviews also
        says whether GitHub names a next page: "next_page" is True, False, or None when unknown."""
        endpoint = next((e for e, pattern in self._routes if e.method == method
                         and isinstance(path, str) and pattern.fullmatch(path)), None)
        if endpoint is None:
            raise GitHubTransportError("endpoint_not_allowed")
        values = {key: str(value) for key, value in (query or {}).items()}
        if not set(endpoint.required_query) <= values.keys() or not all(
                key in endpoint.query and re.fullmatch(endpoint.query[key], value) for key, value in values.items()):
            raise GitHubTransportError("query_not_allowed")
        if not _holds(self._permissions, endpoint.permission):
            raise GitHubTransportError("step_not_permitted")
        if endpoint.body is not None and not endpoint.body(body):
            raise GitHubTransportError("body_not_allowed")
        target = f"{path}?{urlencode(values)}" if values else path
        pull = (method, endpoint.template) == ("GET", "/repos/{repo}/pulls/{number}")
        self._pull = None if pull else self._pull
        status, value, *paging = _exchange(method, target, f"Bearer {self._installation_token()}", body)
        if endpoint.template.endswith("/reviews") and method == "GET" and isinstance(value, list):
            return {"status": status, "data": [self._review(review) for review in value],
                    "next_page": paging[0] if paging and isinstance(paging[0], bool) else None}
        if endpoint.template.endswith("/annotations") and isinstance(value, list):
            return {"status": status, "data": [self._annotation(annotation) for annotation in value]}
        data = None if value is None else self._project(value)
        if pull and status == 200 and isinstance(data, dict) and isinstance(data.get("head"), dict):
            self._pull = (data.get("number"), data.get("state"), data["head"].get("sha"), data.get("node_id"))
        return {"status": status, "data": data}

    def arm_auto_merge(self, number: int, head: str) -> dict:
        """The one GraphQL call: arm auto-merge, a merge commit, on pull request ``number`` of this
        repository at ``head``, both from its delivery record. The document and its variables are
        built here, never passed in. Nothing is read or minted here: the mutation goes, with the
        token this transport's earlier reads minted, only when its latest read of that pull request
        found it open at ``head``. Returns {"status", "data"} as :meth:`request` does."""
        endpoint = GRAPHQL_ALLOWLIST[0]
        if type(number) is not int or number < 1 or not _is_sha(head):
            raise GitHubTransportError("bad_arm_target")
        if not _holds(self._permissions, endpoint.permission):
            raise GitHubTransportError("step_not_permitted")
        held, state, pulled, node = self._pull or (None, None, None, None)
        body = {"query": AUTO_MERGE_MUTATION, "variables": {"pullRequestId": node, "expectedHeadOid": head}}
        if (held, state, pulled) != (number, "open", head) or not endpoint.body(body) or self._token is None:
            raise GitHubTransportError("pull_request_not_at_head")
        status, value, *_ = _exchange(endpoint.method, endpoint.template, f"Bearer {self._token}", body)
        return {"status": status, "data": None if value is None else self._project(value)}

    def branch_head(self, branch: str) -> str | None:
        """The commit `branch` holds now, or None when GitHub answers 404: the branch does not exist."""
        answer = self.request("GET", f"/repos/{self.repository}/git/ref/heads/{branch}")
        if answer["status"] == 404:
            return None
        target = answer["data"].get("object") if isinstance(answer["data"], dict) else None
        sha = target.get("sha") if isinstance(target, dict) else None
        if answer["status"] != 200 or not _is_sha(sha):
            raise GitHubTransportError("bad_response")
        return sha

    def push(self, worktree: str | Path, branch: str, sha: str, *, expected: str) -> dict:
        """Push commit `sha` to `branch`, leased on what the branch holds now: the commit `expected`,
        or "" when the branch must not exist yet."""
        if not (isinstance(branch, str) and _BRANCH.fullmatch(branch) and _is_sha(sha)
                and (expected == "" or _is_sha(expected))):
            raise GitHubTransportError("bad_push_target")
        if not _holds(self._permissions, ("contents", "write")):
            raise GitHubTransportError("step_not_permitted")
        ref = f"refs/heads/{branch}"
        url, scheme = self._push_url()  # the destination, fixed before any credential exists
        try:
            home = tempfile.mkdtemp(prefix="hermes-delivery-push-")  # mode 0700, for this one push
        except OSError:
            raise GitHubTransportError("push_failed") from None
        try:
            repository = _copy_commit(home, worktree, sha, ref)
            # Only GIT_ORIGIN's own protocol, so no rewrite to ext:: or ssh:// can be carried out. One
            # URL, one refspec and its lease, and --no-follow-tags, so nothing adds a ref beside them.
            argv = ["git", "-c", "protocol.allow=never", "-c", f"protocol.{scheme}.allow=always",
                    "--git-dir", repository, "push", "--porcelain", "--no-follow-tags",
                    f"--force-with-lease={ref}:{expected}", url, f"{sha}:{ref}"]
            result = _run_git(argv, self._git_env(_git_base_env(home)))  # the token is minted here
        finally:
            shutil.rmtree(home, ignore_errors=True)  # the repository, its copy of the commit and the config
        if result is None:
            raise GitHubTransportError("push_failed")
        # Bytes, decoded here with replacement: a strict decode fails with an error that carries git's
        # output, and with it whatever git echoed of the credential.
        lines = result.stdout.decode("utf-8", "replace").splitlines()
        line = next((fields for fields in (text.split("\t") for text in lines)
                     if len(fields) == 3 and fields[1].endswith(f":{ref}")), None)
        if line and line[0] == "!" and line[2].endswith("(stale info)"):
            raise GitHubTransportError("push_lease_mismatch")
        if result.returncode != 0 or not line or line[0] not in {"*", " ", "+", "="}:
            raise GitHubTransportError("push_failed")
        return {"state": "up_to_date" if line[0] == "=" else "pushed", "branch": branch, "head": sha}

    def _installation_token(self) -> str:
        if self._token is None:
            self._token = self._mint()
        return self._token

    def _mint(self) -> str:
        """One installation token naming only this step's permissions and this one repository."""
        request = {"repositories": [self.repository.split("/")[1]], "permissions": dict(self._permissions)}
        target = _TOKEN_ENDPOINT.template.format(installation=INSTALLATION_ID)
        status, value, *_ = _exchange("POST", target, f"Bearer {_app_jwt(self._key_path)}", request)
        token = value.get("token") if status == 201 and isinstance(value, dict) else None
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise GitHubTransportError("token_request_failed")
        granted = value.get("permissions")
        names = [r.get("full_name") for r in value.get("repositories") or () if isinstance(r, dict)]
        if names != [self.repository] or not isinstance(granted, dict) or {
                key: level for key, level in granted.items() if key != "metadata"} != request["permissions"]:
            raise GitHubTransportError("token_scope_mismatch")
        return token

    def _push_url(self) -> tuple[str, str]:
        """The one address this push may reach, built from GIT_ORIGIN and the bound repository, and
        the one scheme its git may speak. Checked here, before a credential is minted."""
        origin = urlsplit(GIT_ORIGIN)
        if origin.scheme not in _PUSH_SCHEMES or origin.query or origin.fragment or "@" in origin.netloc:
            raise GitHubTransportError("bad_push_target")
        return f"{GIT_ORIGIN}/{self.repository}.git", origin.scheme

    def _git_env(self, env: dict[str, str]) -> dict[str, str]:
        """The push's environment: *env* and the credential, which lives only here, as with_git_auth
        does it."""
        # Command scope outranks every file git could still read, though the push reads only the empty
        # global file and the config its fresh repository was made with: no hooks path, helper or
        # redirect is in force.
        config = {"credential.helper": "", "core.askPass": "", "core.hooksPath": os.devnull,
                  "http.followRedirects": "false",
                  f"http.{GIT_ORIGIN}/.extraheader": f"Authorization: basic {_basic(self._installation_token())}"}
        for index, (key, value) in enumerate(config.items()):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        env["GIT_CONFIG_COUNT"] = str(len(config))
        return env

    def _review(self, review) -> dict:
        """A listed review: its author's login, state, commit and its body's first line only."""
        review = review if isinstance(review, dict) else {}
        user, body = review.get("user"), review.get("body")
        line = body.split("\n", 1)[0].rstrip() if isinstance(body, str) else ""
        return {"user": {"login": self._project((user if isinstance(user, dict) else {}).get("login"))},
                "state": self._project(review.get("state")), "commit_id": self._project(review.get("commit_id")),
                "body": self._project(line)}

    def _annotation(self, annotation) -> dict:
        """A check run's annotation: its title and message only."""
        annotation = annotation if isinstance(annotation, dict) else {}
        return {"title": self._project(annotation.get("title")), "message": self._project(annotation.get("message"))}

    def _project(self, value):
        if isinstance(value, dict):
            return {key: self._project(item) for key, item in value.items() if key in _FIELDS}
        if isinstance(value, list):
            return [self._project(item) for item in value]
        if isinstance(value, str):
            for secret in (self._token, _basic(self._token)):  # the token, and git's Basic form of it
                value = value.replace(secret, "[redacted]")
            return redact_sensitive_text(value, force=True)[:_MAX_TEXT]
        return value
