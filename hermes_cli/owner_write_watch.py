"""Hourly check of the owner write routes, run by one script-only cron job per serving profile.

``python -m hermes_cli.owner_write_watch <page check path>``

Each run sends this profile's fixed table of harmless requests to the local API server listener,
under the same ``/p/<profile>`` prefix the owner app uses and with this profile's own
API_SERVER_KEY. Every request names something that does not exist, so a healthy route answers
"not found" or "not valid" and nothing changes; only the exact healthy status and code passes, and
a refusal (401, 403, 404 owner_workspace_not_enabled, 503) fails. The default profile's job also
runs the page check and asks the deployed route policy whether each Automations, Connections and
Models route the owner app calls is still registered for its key (indirect signals: those keys stay
with the owner app, and no request is sent to those routes).

Printed output is the job's message: nothing while the state stays the same, one alert when checks
start to fail, one update when the failing set changes, one recovery message. A message whose
delivery failed prints again hourly when HERMES_CRON_JOB_ID gives the job's own id; without it
nothing prints twice, and the first alert says that the hourly repeat is off. Exit 0 whenever the
checks ran; non-zero only when they could not run (no key, unreadable state file, page check
missing, listener address unknown). The key is only ever the request's bearer: never printed,
logged or written.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
OK, FAILING = "ok", "failing"
STATE_FILE = "owner_write_watch_state.json"
REPORT_FILE = "owner_write_watch_report.json"
GAP = timedelta(hours=2)
# A failed announcement prints again at most once an hour; the margin absorbs scheduler jitter.
REPEAT_AFTER = timedelta(minutes=50)
# Added to the first alert when the job runs without its own job id.
REPEAT_OFF = "The hourly repeat is off: if this alert is not delivered, it is not sent again."
REQUEST_TIMEOUT_SECONDS = 30.0
PAGE_CHECK_TIMEOUT_SECONDS = 600
# The dashboard serving Connections and Models runs as the default profile; one job runs the
# indirect signals so a failure there alerts once, not once per profile.
INDIRECT_PROFILE = "default"
# The API server's DEFAULT_PORT and DEFAULT_HOST (gateway/platforms/api_server.py; a test keeps them
# equal): the port when the root listener section gives none, and the address every request goes to.
DEFAULT_PORT = 8642
DEFAULT_HOST = "127.0.0.1"


class Inconclusive(Exception):
    """No verdict from this answer (a timeout); the check keeps its previous state."""


class StateUnreadable(Exception):
    pass


class ListenerUnknown(Exception):
    """The listener is not in the one shape this check supports; the message names the shape."""


@dataclass(frozen=True)
class Probe:
    name: str
    label: str
    method: str
    path: str
    body: Dict[str, Any]
    status: int
    code: Optional[str] = None
    # The platform sends no code with these 400s, so the exact message is compared instead.
    message: Optional[str] = None
    busy_is_inconclusive: bool = False


@dataclass(frozen=True)
class Signal:
    group: str
    method: str
    path: str
    scope: str


@dataclass(frozen=True)
class Result:
    name: str
    label: str
    outcome: str
    detail: str = ""
    method: str = ""
    path: str = ""
    status: Optional[int] = None
    code: Optional[str] = None
    at: str = ""


@dataclass(frozen=True)
class Outcome:
    code: int
    kind: Optional[str] = None
    text: str = ""


_ZEROS = "0" * 32
_DECISION = f"decision_{_ZEROS}"
_CONVERSATION = f"/v1/responses/conversations/raphael-owner-{_ZEROS}"
_PLANNER = "/p/raphael-planner"

# Paths and prefixes as the owner app's conversation client builds them (main 7419773d).
PROBES: Dict[str, tuple] = {
    "default": (
        Probe("decisions", "Decisions accept, reject and defer (route 1)", "POST",
              f"/p/default/v1/owner-workspace/decisions/{_DECISION}/defer", {"reason": "hourly check"},
              404, "decision_not_found"),
        Probe("runs", "Starting owner runs: commit, lifecycle, removal, retry (routes 4, 6, 7, 8)", "POST",
              "/p/default/v1/runs", {"owner_lifecycle_authority": "hourly-check"},
              400, message="Invalid owner-workspace authority"),
        Probe("run_approval", "Run approvals (routes 5 to 8)", "POST",
              "/p/default/v1/runs/run_hourlycheck0000/approval", {}, 404, "run_not_found"),
    ),
    "raphael-planner": (
        Probe("planner_turn", "Conversation and automation planning (routes 2, 3)", "POST",
              f"{_PLANNER}/v1/responses", {}, 400, message="Missing 'input' field",
              busy_is_inconclusive=True),
        Probe("conversation_close", "Closing a conversation (route 9)", "POST",
              f"{_PLANNER}{_CONVERSATION}/authority", {}, 400,
              message="Invalid owner proposal authority request"),
        # Start over is the authority call with the action close (applyConversationAuthority, from
        # closeConversation), not a DELETE: without its response id it is refused before any lookup.
        Probe("conversation_history", "Clearing conversation history: Start over (route 12)", "POST",
              f"{_PLANNER}{_CONVERSATION}/authority", {"action": "close"}, 400,
              message="Invalid owner proposal authority request"),
        Probe("conversation_recovery", "Conversation recovery (route 10)", "POST",
              f"{_PLANNER}{_CONVERSATION}/recovery", {}, 400,
              message="Invalid owner recovery acknowledgement"),
        Probe("conversation_receipt", "Marking a proposal used (route 11)", "POST",
              f"{_PLANNER}{_CONVERSATION}/consume", {}, 400,
              message="Invalid owner proposal consumption request"),
    ),
}

_AUTOMATIONS_SCOPE = "cron.automations.manage"
_CONNECTIONS_SCOPE = "mcp.connections.manage"
_MODELS_SCOPE = "models.manage"
_JOB = "0" * 12
_ROLES = ("default", "raphael-planner", "raphael-business", "raphael-designer",
          "raphael-claude-worker", "raphael-builder", "raphael-verifier")

# Each method and path the owner app's Automations, Connections and Models clients send (main 7419773d).
SIGNALS = tuple(
    [Signal("automations", method, path, _AUTOMATIONS_SCOPE) for method, path in (
        ("GET", "/api/cron/jobs"),
        ("GET", "/api/cron/executions"),
        ("POST", "/api/cron/jobs"),
        ("POST", f"/api/cron/jobs/{_JOB}/pause"),
        ("POST", f"/api/cron/jobs/{_JOB}/resume"),
        ("POST", f"/api/cron/executions/{_ZEROS}/resend"),
    )]
    + [Signal("connections", method, path, _CONNECTIONS_SCOPE) for method, path in (
        ("GET", "/api/mcp/catalog"),
        ("GET", "/api/mcp/servers"),
        ("POST", "/api/mcp/catalog/install"),
        ("PUT", "/api/mcp/servers/claude-design/enabled"),
        ("POST", "/api/mcp/servers/claude-design/auth"),
        ("DELETE", "/api/mcp/servers/claude-design"),
        ("POST", f"/api/mcp/oauth/flows/{_ZEROS}/submit"),
        ("GET", f"/api/mcp/oauth/flows/{_ZEROS}"),
        ("GET", "/api/mcp/oauth/callback/claude-design"),
    )]
    + [Signal("models", method, path, _MODELS_SCOPE) for method, path in (
        ("GET", "/api/providers/oauth"),
        ("GET", "/api/model/options"),
        ("GET", "/api/model/info"),
        ("POST", "/api/providers/oauth/anthropic/start"),
        ("POST", "/api/providers/oauth/openai-codex/start"),
        ("POST", "/api/providers/oauth/anthropic/submit"),
        ("GET", f"/api/providers/oauth/openai-codex/poll/{_ZEROS}"),
        ("DELETE", "/api/providers/oauth/anthropic"),
        ("DELETE", "/api/providers/oauth/openai-codex"),
        ("POST", "/api/profiles/model-batch"),
        *(("PUT", f"/api/profiles/{role}/model") for role in _ROLES),
    )]
)
_WIRING = (
    ("automations", "Automations permission wiring (routes 13 to 15)"),
    ("connections", "Connections permission wiring (route 16)"),
    ("models", "Models permission wiring (route 17)"),
)
_PAGE_CHECK = "page_check"
_PAGE_CHECK_LABEL = "Page check of the Automations, Connections and Models pages (routes 13 to 17)"


def state_path() -> Path:
    return get_hermes_home() / STATE_FILE


def report_path() -> Path:
    return get_hermes_home() / REPORT_FILE


def format_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_duration(length: timedelta) -> str:
    days, minutes = divmod(max(0, int(length.total_seconds() // 60)), 1440)
    hours, minutes = divmod(minutes, 60)
    parts = [f"{days} d"] if days else []
    if hours:
        parts.append(f"{hours} h")
    if minutes or not parts:
        parts.append(f"{minutes} min")
    return " ".join(parts)


def classify(probe: Probe, status: Optional[int], payload: Any) -> str:
    """PASS only for the probe's exact healthy status and code (and message where it has no code)."""
    if status == 429 and probe.busy_is_inconclusive:
        return UNKNOWN
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or status != probe.status or error.get("code") != probe.code:
        return FAIL
    if probe.message is not None and error.get("message") != probe.message:
        return FAIL
    return PASS


def run(
    page_check: str,
    *,
    profile: Optional[str] = None,
    base: Optional[str] = None,
    send: Optional[Callable[..., Any]] = None,
    run_page_check: Optional[Callable[[str], str]] = None,
    read_key: Optional[Callable[[], Optional[str]]] = None,
    delivery_state: Optional[Callable[[Dict[str, Any]], str]] = None,
    route_policy: Optional[Callable[[str, str], Any]] = None,
    now: Optional[Callable[[], datetime]] = None,
    job_id: Optional[str] = None,
) -> Outcome:
    """One hourly run. Every argument after ``page_check`` defaults to the production one.
    ``job_id`` is the watch job's own cron job id, which main() reads from HERMES_CRON_JOB_ID:
    without it no delivery can be read, so every announcement reads unknown and never prints again."""
    clock = now or (lambda: datetime.now(timezone.utc))
    if profile is None:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name()
    probes = PROBES.get(profile)
    if probes is None:
        return _cannot_run(f"no owner write checks are defined for profile {profile!r}")
    indirect = profile == INDIRECT_PROFILE
    if indirect and not Path(page_check).is_file():
        return _cannot_run(f"page check missing: {page_check}")
    try:
        previous = _read_state(state_path())
    except StateUnreadable as exc:
        return _cannot_run(f"state file unreadable ({exc}): {state_path()}")
    key = (read_key or _own_profile_key)()
    if not key:
        return _cannot_run("no usable API_SERVER_KEY in this profile's own .env; nothing was sent")
    try:
        base = base or _listener_base()
    except ListenerUnknown as exc:
        return _cannot_run(f"listener address unknown: {exc}")
    except Exception as exc:
        return _cannot_run(f"listener address unknown: root configuration unreadable ({type(exc).__name__})")

    moment = clock()
    results = [_probe(probe, base, key, send or _http_send, clock) for probe in probes]
    if indirect:
        results.append(_page_check(page_check, run_page_check or _run_page_check, clock))
        results.extend(_wiring(route_policy or _deployed_route_policy, clock))
    reader = delivery_state or (lambda announced: _previous_delivery(announced, job_id))
    state, kind, text = _decide(previous, results, moment, profile, reader)
    try:
        _write(report_path(), {"profile": profile, "run_at": moment.isoformat(), "state": state["state"],
                               "checks": [asdict(result) for result in results]})
        _write(state_path(), state)
    except OSError as exc:
        print(f"owner write watch: state not saved: {type(exc).__name__}", file=sys.stderr)
        return Outcome(1, kind, text)
    return Outcome(0, kind, text)


def _cannot_run(reason: str) -> Outcome:
    print(f"owner write watch: checks not run: {reason}", file=sys.stderr)
    return Outcome(1)


def _probe(probe: Probe, base: str, key: str, send: Callable[..., Any], clock) -> Result:
    fields = dict(name=probe.name, label=probe.label, method=probe.method, path=probe.path)
    try:
        status, payload = send(probe.method, base + probe.path, probe.body, key)
    except Inconclusive:
        return Result(outcome=UNKNOWN, detail="no answer in time", at=clock().isoformat(), **fields)
    except Exception:
        return Result(outcome=FAIL, detail="no answer", at=clock().isoformat(), **fields)
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
    outcome = classify(probe, status, payload)
    detail = f"answered {status} {code}" if code else f"answered {status}"
    return Result(outcome=outcome, detail=detail, status=status, code=code, at=clock().isoformat(), **fields)


def _page_check(path: str, run_page_check: Callable[[str], str], clock) -> Result:
    outcome = run_page_check(path)
    return Result(_PAGE_CHECK, _PAGE_CHECK_LABEL, outcome if outcome in (PASS, FAIL, UNKNOWN) else FAIL,
                  detail="did not pass", method="run", path=path, at=clock().isoformat())


def _wiring(route_policy: Callable[[str, str], Any], clock) -> List[Result]:
    results = []
    for group, label in _WIRING:
        signals = [signal for signal in SIGNALS if signal.group == group]
        try:
            missing = [f"{s.method} {s.path}" for s in signals
                       if getattr(route_policy(s.method, s.path), "required_scope", None) != s.scope]
            detail = "not registered for the owner app's key: " + ", ".join(missing)
        except Exception as exc:
            missing, detail = [group], f"route policy unreadable: {type(exc).__name__}"
        results.append(Result(group, label, FAIL if missing else PASS, detail=detail if missing else "",
                              method="policy", at=clock().isoformat()))
    return results


def _decide(previous, results, moment, profile, delivery_state):
    """Compare with the stored state: ``(new state, message kind, message text)``."""
    labels = {result.name: result.label for result in results}
    inconclusive = {result.name for result in results if result.outcome == UNKNOWN}
    before = previous["state"] if previous else UNKNOWN
    failed_before = set(previous["failing"]) if previous else set()
    # A check without a verdict keeps its previous state: no update or recovery on a guess.
    failing = {r.name for r in results if r.outcome == FAIL} | (failed_before & inconclusive)
    if failing:
        state = FAILING
    elif inconclusive and before == UNKNOWN:
        state = UNKNOWN
    else:
        state = OK
    since = _moment(previous.get("since")) if previous else None
    announced = previous.get("announced") if previous else None
    kind, text = None, ""
    if state == FAILING and before != FAILING:
        kind, since = "alert", moment
        text = f"Owner actions check: {len(failing)} failing since {format_time(moment)}.\n"
        text += _failing_lines(results, failing)
    elif state == FAILING and failing != failed_before:
        kind = "update"
        text = f"Owner actions check update: {len(failing)} failing (since {format_time(since or moment)}).\n"
        text += _failing_lines(results, failing)
        fixed = sorted(failed_before - failing)
        if fixed:
            text += "\nFixed: " + "; ".join(labels.get(name, name) for name in fixed)
    elif state == OK and before == FAILING:
        kind = "recovery"
        began = since or moment
        text = (f"Owner actions check: all checks pass again. The failure lasted "
                f"{format_duration(moment - began)} ({format_time(began)} to {format_time(moment)}).")
    elif state == OK and before == UNKNOWN:
        kind = "active"
        text = (f"Owner actions hourly check is now active for profile {profile}: all {len(results)} checks "
                "pass. One message follows when a check fails and one when it is fixed.")
    if state != FAILING:
        since = None
    if kind:
        announced = {"text": text, "at": moment.isoformat()}
    elif announced and not announced.get("settled") and state == before and _repeat_due(announced, moment):
        # Owner decision 2: a failed announcement prints again, at most hourly while it stays
        # failed; once delivered or unknown it is settled and never prints again.
        if delivery_state(announced) == "failed":
            kind, text = "repeat", announced["text"]
            announced = {"text": text, "at": moment.isoformat()}
        else:
            announced = dict(announced, settled=True)
    last_run = _moment(previous.get("last_run")) if previous else None
    if last_run is not None and moment - last_run > GAP:
        gap = f"Checks stopped after {format_time(last_run)} and resumed at {format_time(moment)}."
        text = f"{text}\n{gap}" if text else gap
    return {
        "version": 1,
        "profile": profile,
        "state": state,
        "failing": sorted(failing),
        "since": since.isoformat() if since else None,
        "last_run": moment.isoformat(),
        "announced": announced,
    }, kind, text


def _failing_lines(results: List[Result], failing: set) -> str:
    lines = []
    for result in results:
        if result.name in failing:
            detail = result.detail if result.outcome == FAIL else "no clear answer this run"
            lines.append(f"- {result.label}: {detail}")
    return "\n".join(lines)


def _repeat_due(announced: Dict[str, Any], moment: datetime) -> bool:
    at = _moment(announced.get("at"))
    return isinstance(announced.get("text"), str) and at is not None and moment - at >= REPEAT_AFTER


def _moment(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _read_state(path: Path) -> Optional[Dict[str, Any]]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise StateUnreadable(type(exc).__name__) from None
    try:
        failing = state["failing"]
        if state["state"] not in (OK, FAILING, UNKNOWN) or not isinstance(failing, list) \
                or not all(isinstance(name, str) for name in failing):
            raise ValueError("unexpected shape")
        for field in ("since", "last_run"):
            _moment(state.get(field))
        announced = state.get("announced")
        if announced is not None and not isinstance(announced, dict):
            raise ValueError("unexpected shape")
    except (KeyError, TypeError, ValueError) as exc:
        raise StateUnreadable(type(exc).__name__) from None
    return state


def _write(path: Path, data: Dict[str, Any]) -> None:
    from utils import atomic_json_write
    atomic_json_write(path, data)


def _own_profile_key() -> Optional[str]:
    """This profile's API_SERVER_KEY from its own .env, read at run time; no other source."""
    from agent.secret_scope import load_env_file
    from hermes_cli.auth import has_usable_secret

    key = load_env_file(get_hermes_home() / ".env").get("API_SERVER_KEY")
    return key.strip() if has_usable_secret(key, min_length=16) else None


def _listener_base() -> str:
    """The one local listener serving every profile under ``/p/<profile>``, in the one shape this
    check supports: the root profile's own listener section (config.yaml over gateway.json, merged as
    the gateway merges them) gives the port, or none and the port is DEFAULT_PORT, and gives no host
    or 127.0.0.1, localhost or 0.0.0.0; the root config.yaml has no top-level api_server section; and
    this job's environment has neither API_SERVER_HOST nor API_SERVER_PORT, which the gateway applies
    over its configuration. Any other shape raises ListenerUnknown naming it. Nothing else of the root
    profile is read: never its .env or its key, and this process's environment is left as it was."""
    import os
    from gateway import config_loader
    from hermes_constants import get_default_hermes_root

    for name in ("API_SERVER_HOST", "API_SERVER_PORT"):
        if name in os.environ:
            raise ListenerUnknown(f"{name} in this job's environment")
    root = get_default_hermes_root()
    data = config_loader.load_legacy_gateway_json(root)
    layers = config_loader.read_yaml_layers(root)
    listener = config_loader.merge_platform_sections(layers, layers.get("gateway"), data).get("api_server")
    if "api_server" in layers:
        raise ListenerUnknown("a top-level api_server section in the root config.yaml")
    extra = listener.get("extra") if isinstance(listener, dict) else None
    extra = extra if isinstance(extra, dict) else {}
    if extra.get("host") not in (None, DEFAULT_HOST, "localhost", "0.0.0.0"):
        raise ListenerUnknown(f"the host {extra['host']!r}")
    raw = extra.get("port")
    port = DEFAULT_PORT if raw is None else int(raw) if isinstance(raw, str) and raw.isdecimal() else raw
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ListenerUnknown(f"the port {raw!r}, not an integer from 1 to 65535")
    return f"http://{DEFAULT_HOST}:{port}"


def _http_send(method: str, url: str, body: Dict[str, Any], key: str):
    import httpx

    try:
        # trust_env=False: never through a proxy; the bearer goes to the local listener only.
        with httpx.Client(trust_env=False, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = client.request(method, url, json=body, headers={"Authorization": f"Bearer {key}"})
    except httpx.TimeoutException:
        raise Inconclusive() from None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _run_page_check(path: str) -> str:
    try:
        completed = subprocess.run([path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=PAGE_CHECK_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return UNKNOWN
    except OSError:
        return FAIL
    return PASS if completed.returncode == 0 else FAIL


_ROUTE_POLICIES_LOADED = False


def _deployed_route_policy(method: str, path: str):
    """The token route policy the deployed dashboard code registers, loaded in this process."""
    global _ROUTE_POLICIES_LOADED
    from hermes_cli.dashboard_auth.token_auth import get_token_route_policy

    if not _ROUTE_POLICIES_LOADED:
        import hermes_cli.web_routers.cron  # noqa: F401  registers the Automations routes on import
        import hermes_cli.web_routers.mcp  # noqa: F401  registers the Connections routes on import
        from plugins.dashboard_auth.raphael_workspace.model_policy import register_models_machine_routes

        register_models_machine_routes()
        _ROUTE_POLICIES_LOADED = True
    return get_token_route_policy(method, path)


def _previous_delivery(announced: Dict[str, Any], job_id: Optional[str]) -> str:
    """Delivery state of the announcement, read only from the watch job's own execution that printed
    it: of every execution of ``job_id``, all read at once with no page and no fixed window, the one
    whose claimed_at is the newest instant at or before the announcement, compared in UTC and never
    as text (a daylight saving or time zone change puts the scheduler's local text out of order).
    Other jobs' executions are never read. ``unknown`` without the job id; when any own claimed_at is
    not ISO text with a UTC offset, or two own executions share that newest instant; or when that
    execution's record cannot be read or does not hold the announced text. Both stores are only
    read: a missing store reads unknown, and no store is created, set up, migrated or written."""
    at = _moment(announced.get("at"))
    if not job_id or at is None or not isinstance(announced.get("text"), str):
        return UNKNOWN
    try:
        newest, tied = None, False
        for execution_id, claimed_at in _own_executions(job_id):
            claimed = _instant(claimed_at)
            if claimed is None:
                return UNKNOWN
            if claimed > at:
                continue
            if newest is None or claimed > newest[0]:
                newest, tied = (claimed, execution_id), False
            elif claimed == newest[0]:
                tied = True
        if newest is None or tied:
            return UNKNOWN
        return _delivery_outcome(_delivery_record(newest[1]), announced["text"])
    except Exception:
        return UNKNOWN


def _own_executions(job_id: str) -> List[tuple]:
    """``(id, claimed_at)`` of every execution of ``job_id``, newest first in the store's order, read in
    one query over one snapshot. No page and no cursor: list_executions pages by ``claimed_at < cursor``,
    which drops the rest of the executions sharing the claimed_at a full page ends on; here all of them
    are read, however many there are, and the read always ends."""
    from cron import executions

    path = Path(executions.EXECUTIONS_FILE or get_hermes_home().resolve() / "cron" / "executions.db")
    with closing(_read_only(path)) as conn:
        return conn.execute(
            "SELECT id, claimed_at FROM executions WHERE job_id=? ORDER BY claimed_at DESC, id DESC",
            (job_id,),
        ).fetchall()


def _delivery_record(execution_id: str) -> Optional[Dict[str, Any]]:
    """One execution's delivery record with its re-send attempts, read by delivery_record's own rules,
    as load_many reads it, but over a read-only connection and in one snapshot: load_many also sets up
    the store it reads."""
    from cron import delivery_record

    delivery_record._platform_keys()  # imported before the store is read, as load_many does
    with closing(_read_only(delivery_record._path())) as conn:
        conn.row_factory = sqlite3.Row
        conn.text_factory = delivery_record._decoded
        conn.execute("BEGIN")
        return delivery_record._load_unlocked(conn, [execution_id], True).get(execution_id)


def _read_only(path: Path) -> sqlite3.Connection:
    """A connection that can only read the SQLite store at ``path``: a missing store raises instead of
    being created, and nothing in the store is set up, migrated or written."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def _instant(value: Any) -> Optional[datetime]:
    """An own execution's claimed_at as its instant in UTC; None unless it is ISO text with a UTC
    offset. Only the reader above reads a time this way."""
    try:
        moment = datetime.fromisoformat(value)
        return moment.astimezone(timezone.utc) if moment.utcoffset() is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _delivery_outcome(record: Optional[Dict[str, Any]], text: str) -> str:
    """How one execution's report ended for its chats, re-sends included: ``failed`` only when every
    chat ended failed (a re-send whose every chat was refused keeps it failed), ``delivered`` when
    every chat was delivered; a re-send in progress or unreadable, or any unknown outcome, reads
    ``unknown``."""
    from cron import delivery_record

    if not record or not isinstance(record.get("text"), str) or text not in record["text"]:
        return UNKNOWN
    if any(attempt["chats"] is None or attempt["state"] == "in_progress" for attempt in record["attempts"]):
        return UNKNOWN
    # The targets as the re-send attempts left them, by the same rule the re-send view applies.
    states = {target["state"] for target in delivery_record._latest_targets(record)}
    return states.pop() if len(states) == 1 and states <= {"failed", "delivered"} else UNKNOWN


def main(argv: Optional[List[str]] = None) -> int:
    import os

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m hermes_cli.owner_write_watch <page check path>", file=sys.stderr)
        return 2
    # The job's own id, which the scheduler gives a script-only job; unset or empty, it is absent.
    job_id = os.environ.get("HERMES_CRON_JOB_ID") or None
    outcome = run(args[0], job_id=job_id)
    text = outcome.text
    if outcome.kind == "alert" and job_id is None:
        text += "\n" + REPEAT_OFF
    if text:
        print(text)
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())
