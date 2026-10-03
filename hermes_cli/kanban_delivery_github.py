"""GitHub App access for delivery: narrowed installation tokens, a fixed REST allowlist and leased
pushes, with no token in any result, error, log or file.

Nothing calls this yet. A transport serves one step on one repository. It signs the App JWT
in-process from the key file and mints one installation token naming only that step's permissions
and that repository. Every call is checked against REST_ALLOWLIST before any token or connection
exists, and results carry fixed fields only.
"""
from __future__ import annotations

import base64
import http.client
import json
import re
import ssl
import subprocess
import time
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from urllib.parse import urlencode

import jwt

from agent.redact import redact_sensitive_text
from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_constants import get_default_hermes_root

APP_ID = "4485539"  # a string: GitHub requires the JWT issuer to be one
INSTALLATION_ID = 151217494
API_HOST = "api.github.com"
GIT_ORIGIN = "https://github.com"

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
_SEGMENTS = {"number": r"[1-9][0-9]{0,9}", "id": r"[1-9][0-9]{0,19}", "sha": _SHA.pattern}
_PAGE = {"per_page": r"[1-9][0-9]?|100", "page": r"[1-9][0-9]{0,3}"}


class Endpoint(NamedTuple):
    method: str
    template: str
    permission: tuple[str, str] | None  # what the step must hold; None for the App's own token call
    query: MappingProxyType = MappingProxyType({})  # allowed parameters, each with its value pattern
    required_query: tuple[str, ...] = ()


_TOKEN_ENDPOINT = Endpoint("POST", "/app/installations/{installation}/access_tokens", None)

# The plan's endpoint list and nothing else: no protection, rules, settings, hooks, collaborators,
# keys or installation management, and no GraphQL.
REST_ALLOWLIST = (
    Endpoint("GET", "/repos/{repo}/pulls", ("pull_requests", "read"), MappingProxyType(
        {"state": "open|closed|all", "head": rf"[A-Za-z0-9-]+:{_BRANCH.pattern}", "base": _BRANCH.pattern, **_PAGE})),
    Endpoint("POST", "/repos/{repo}/pulls", ("pull_requests", "write")),
    Endpoint("GET", "/repos/{repo}/pulls/{number}", ("pull_requests", "read")),
    Endpoint("PUT", "/repos/{repo}/pulls/{number}/merge", ("contents", "write")),
    Endpoint("POST", "/repos/{repo}/pulls/{number}/reviews", ("pull_requests", "write")),
    Endpoint("GET", "/repos/{repo}/commits/{sha}/check-runs", ("checks", "read"),
             MappingProxyType({"filter": "latest|all", **_PAGE})),
    Endpoint("GET", "/repos/{repo}/commits/{sha}/statuses", ("statuses", "read"), MappingProxyType(_PAGE)),
    Endpoint("GET", "/repos/{repo}/actions/runs", ("actions", "read"),
             MappingProxyType({"head_sha": _SHA.pattern, **_PAGE}), ("head_sha",)),
    Endpoint("GET", "/repos/{repo}/actions/runs/{id}/jobs", ("actions", "read"),
             MappingProxyType({"filter": "latest|all", **_PAGE})),
    # GitHub answers with a redirect to signed log storage, which is never followed.
    Endpoint("GET", "/repos/{repo}/actions/jobs/{id}/logs", ("actions", "read")),
    Endpoint("POST", "/repos/{repo}/actions/jobs/{id}/rerun", ("actions", "write")),
    _TOKEN_ENDPOINT,
)

# Fixed fields only: ids, SHAs, refs, states and counts. Titles, bodies, messages, URLs and users
# never pass, and neither do response headers.
_FIELDS = frozenset({
    "id", "number", "name", "state", "status", "conclusion", "merged", "mergeable", "mergeable_state",
    "sha", "head_sha", "merge_commit_sha", "commit_id", "ref", "head", "base", "app", "context",
    "total_count", "check_runs", "workflow_runs", "jobs", "steps", "run_id", "run_attempt",
})
_MAX_TEXT = 256


class GitHubTransportError(Exception):
    """A refusal or failure named by a fixed reason code, never by a request, response or token."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _connect() -> http.client.HTTPConnection:
    # http.client follows no redirect, reads no proxy or netrc setting and logs nothing; httpcore's
    # DEBUG trace would log response headers.
    return http.client.HTTPSConnection(API_HOST, timeout=30, context=ssl.create_default_context())


def _exchange(method: str, target: str, authorization: str, payload=None) -> tuple[int, object]:
    """One request to GitHub: the status, and the decoded body of a 2xx answer (else None)."""
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "hermes-kanban-delivery", "Authorization": authorization}
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection = _connect()
    try:
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        status, raw = response.status, response.read()
    except (OSError, http.client.HTTPException):
        status = raw = None
    finally:
        connection.close()
    # Raised after the handler, not inside it: a chained IncompleteRead or JSONDecodeError would
    # carry the body, which for the token call holds the token.
    if status is None:
        raise GitHubTransportError("network_error")
    if not 200 <= status < 300 or not raw:
        return status, None  # GitHub's error text and redirect targets are never passed on
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if not isinstance(value, (dict, list)):
        raise GitHubTransportError("bad_response")
    return status, value


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

    def request(self, method: str, path: str, *, query=None, body=None) -> dict:
        """Call one allowlisted endpoint. Returns {"status", "data"}, where data holds the fixed
        fields of a 2xx JSON answer and is None for anything else."""
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
        target = f"{path}?{urlencode(values)}" if values else path
        status, value = _exchange(method, target, f"Bearer {self._installation_token()}", body)
        return {"status": status, "data": None if value is None else self._project(value)}

    def push(self, worktree: str | Path, branch: str, sha: str, *, expected: str) -> dict:
        """Push commit `sha` to `branch`, leased on what the branch holds now: the commit `expected`,
        or "" when the branch must not exist yet."""
        if not (isinstance(branch, str) and _BRANCH.fullmatch(branch) and _is_sha(sha)
                and (expected == "" or _is_sha(expected))):
            raise GitHubTransportError("bad_push_target")
        if not _holds(self._permissions, ("contents", "write")):
            raise GitHubTransportError("step_not_permitted")
        ref = f"refs/heads/{branch}"
        argv = ["git", "-C", str(worktree), "push", "--porcelain", f"--force-with-lease={ref}:{expected}",
                f"{GIT_ORIGIN}/{self.repository}.git", f"{sha}:{ref}"]
        env = self._git_env()
        try:
            result = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is None:
            raise GitHubTransportError("push_failed")
        line = next((fields for fields in (text.split("\t") for text in result.stdout.splitlines())
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
        status, value = _exchange("POST", target, f"Bearer {_app_jwt(self._key_path)}", request)
        token = value.get("token") if status == 201 and isinstance(value, dict) else None
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise GitHubTransportError("token_request_failed")
        granted = value.get("permissions")
        names = [r.get("full_name") for r in value.get("repositories") or () if isinstance(r, dict)]
        if names != [self.repository] or not isinstance(granted, dict) or {
                key: level for key, level in granted.items() if key != "metadata"} != request["permissions"]:
            raise GitHubTransportError("token_scope_mismatch")
        return token

    def _git_env(self) -> dict[str, str]:
        """The push's environment: the only place its credential lives, as with_git_auth does it."""
        # Inherited command-scope git config is dropped: read after ours, it could bring a credential
        # helper back.
        env = {key: value for key, value in noninteractive_git_env().items()
               if not re.fullmatch(r"GIT_CONFIG_(?:COUNT|KEY_\d+|VALUE_\d+|PARAMETERS)", key)}
        env.pop("GIT_ASKPASS", None)
        env.pop("SSH_ASKPASS", None)
        basic = base64.b64encode(f"x-access-token:{self._installation_token()}".encode()).decode()
        config = {"credential.helper": "", "core.askPass": "", "http.followRedirects": "false",
                  f"http.{GIT_ORIGIN}/.extraheader": f"Authorization: basic {basic}"}
        for index, (key, value) in enumerate(config.items()):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        env["GIT_CONFIG_COUNT"] = str(len(config))
        return env

    def _project(self, value):
        if isinstance(value, dict):
            return {key: self._project(item) for key, item in value.items() if key in _FIELDS}
        if isinstance(value, list):
            return [self._project(item) for item in value]
        if isinstance(value, str):
            return redact_sensitive_text(value.replace(self._token, "[redacted]"), force=True)[:_MAX_TEXT]
        return value
