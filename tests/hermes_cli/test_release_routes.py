"""Card 3 of the release route allowlist plan: the route plan and the rewrite of one route list.

plan_routes decides what a release changes in the route list of each role the gateway serves on this
host (plan section 2.2), and apply_routes rewrites one config.yaml text by that plan, in the one
written form it supports: the plain form of the owner facts for this card (points 9 to 17). There
is one test per rule of 2.2, then the rewrite's tests in the plain form, the texts outside it, the
refusals, the proof that the module imports nothing inside a function (K6), and the proof that both
functions only compute: no file, network, process or environment access. Where a test needs the
gateway's own reading of a text, it writes the text where the gateway reads config.yaml and asks
the gateway's resolver.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

import pytest
import yaml

from gateway.platforms.api_server import (
    _ROUTES_ALLOWLIST,
    _ROUTES_DENY_ALL,
    _ROUTES_UNRESTRICTED,
    _resolve_api_server_allowed_routes,
)
from hermes_cli import release_routes
from hermes_cli.config import get_config_path
from hermes_cli.release_routes import (
    RouteChange,
    RouteNote,
    RoutePlan,
    RoutePlanError,
    apply_routes,
    plan_routes,
)
from utils import IndentDumper

REPO = Path(__file__).resolve().parents[2]

BUILDER = "raphael-builder"
REVIEWER = "raphael-reviewer"
ACCEPT = "POST /v1/owner-workspace/decisions/{decision_ref}/accept"
REJECT = "POST /v1/owner-workspace/decisions/{decision_ref}/reject"
MODELS = "GET /v1/models"
RUNS = "POST /v1/runs"

_NO_KEY = object()


def route_file(routes: dict[str, list[str]]) -> str:
    """The text of gateway/platforms/api_server_route_profiles.json naming `routes`."""
    return json.dumps(routes, indent=2, sort_keys=True) + "\n"


def config(routes: object = _NO_KEY) -> bytes:
    """A role's config.yaml as hermes's own writer (atomic_yaml_write) writes it, with `routes` as
    gateway.api_server.allowed_routes; the key is left out when `routes` is not given."""
    api_server: dict[str, object] = {"enabled": True, "port": 8642}
    if routes is not _NO_KEY:
        api_server["allowed_routes"] = routes
    data = {
        "model": {"default": "anthropic/claude-sonnet-5"},
        "gateway": {"api_server": api_server},
    }
    return _hermes_yaml(data)


def _hermes_yaml(data: object) -> bytes:
    return yaml.dump(
        data, Dumper=IndentDumper, default_flow_style=False, sort_keys=False, allow_unicode=True
    ).encode()


def gateway_view(text: bytes | None) -> tuple[str, list[str]]:
    """The route list the gateway reads from a config.yaml holding `text` (None: no config.yaml)."""
    path = get_config_path()
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text)
    return _resolve_api_server_allowed_routes()


# --- plan section 2.2, one test per rule ---------------------------------------------------------


def test_no_prev_file_adds_and_removes_nothing():
    # The first release that carries the route file: nothing is added and nothing is removed.
    configs = {BUILDER: config([MODELS]), REVIEWER: config([REJECT, RUNS])}

    plan = plan_routes(None, route_file({ACCEPT: [BUILDER, REVIEWER]}), configs)

    assert plan == RoutePlan(changes=(), notes=())
    assert plan.added() == ()


def test_an_added_pair_in_allowlist_mode_adds_the_exact_entry():
    prev = route_file({MODELS: [BUILDER]})
    new = route_file({ACCEPT: [BUILDER], MODELS: [BUILDER, REVIEWER]})
    configs = {BUILDER: config([MODELS]), REVIEWER: config([RUNS])}

    plan = plan_routes(prev, new, configs)

    assert plan == RoutePlan(
        changes=(
            RouteChange(BUILDER, add=(ACCEPT,), remove=()),
            RouteChange(REVIEWER, add=(MODELS,), remove=()),
        ),
        notes=(),
    )
    assert plan.added() == ((BUILDER, ACCEPT), (REVIEWER, MODELS))
    assert gateway_view(apply_routes(configs[BUILDER], plan.changes[0])) == (
        _ROUTES_ALLOWLIST,
        [MODELS, ACCEPT],
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(None, id="no-config-file"),
        pytest.param(b"", id="empty-file"),
        pytest.param(b"# only a comment\n\n", id="comments-only"),
        pytest.param(config(), id="no-key"),
        pytest.param(config(None), id="null"),
        pytest.param(config([]), id="empty-list"),
        pytest.param(b"gateway:\n", id="null-gateway"),
        pytest.param(b"gateway:\n  api_server:\n", id="null-api-server"),
    ],
)
def test_an_unrestricted_list_is_skipped(text):
    assert gateway_view(text) == (_ROUTES_UNRESTRICTED, [])

    plan = plan_routes(route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: text})

    assert plan == RoutePlan(
        changes=(), notes=(RouteNote(BUILDER, ACCEPT, "this role's list does not limit routes"),)
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(config(True), id="bool"),
        pytest.param(config(""), id="empty-string"),
        pytest.param(config([MODELS, 42]), id="entry-not-a-string"),
        pytest.param(config(["models please"]), id="entry-not-a-route"),
        pytest.param(config({"GET": "/v1/models"}), id="mapping"),
        pytest.param(b"gateway: [\n", id="unparseable"),
        pytest.param(b"\xff\xfe\n", id="not-utf-8"),
        pytest.param(b"---\n", id="no-mapping"),
        pytest.param(b"- a list\n", id="list-root"),
        pytest.param(b"gateway: 5\n", id="gateway-not-a-mapping"),
    ],
)
def test_a_deny_all_list_refuses(text):
    assert gateway_view(text) == (_ROUTES_DENY_ALL, [])

    with pytest.raises(RoutePlanError) as refused:
        plan_routes(route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: text})

    assert refused.value.reason == f"role {BUILDER} has a route list that is not valid"
    # With no added pair for the role there is nothing to change in its list, so nothing refuses.
    prev = route_file({ACCEPT: [BUILDER], REJECT: [BUILDER]})
    assert plan_routes(prev, route_file({ACCEPT: [BUILDER]}), {BUILDER: text}) == RoutePlan((), ())


@pytest.mark.parametrize(
    "entry", [ACCEPT, "post /v1/owner-workspace/decisions/{decision_ref}/accept/"]
)
def test_an_exact_entry_already_present_is_skipped(entry):
    plan = plan_routes(route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: config([entry])})

    assert plan == RoutePlan(changes=(), notes=(RouteNote(BUILDER, ACCEPT, "already admitted"),))


def test_an_old_path_prefix_covering_it_is_skipped():
    configs = {
        BUILDER: config(["/v1/owner-workspace/decisions"]),
        # The same leading characters, but not a path prefix of the route: it does not cover it.
        REVIEWER: config(["/v1/owner"]),
    }

    plan = plan_routes(route_file({}), route_file({ACCEPT: [BUILDER, REVIEWER]}), configs)

    assert plan == RoutePlan(
        changes=(RouteChange(REVIEWER, add=(ACCEPT,), remove=()),),
        notes=(RouteNote(BUILDER, ACCEPT, "already admitted"),),
    )


def test_a_role_not_served_here_is_skipped():
    new = route_file({ACCEPT: [BUILDER, "raphael-elsewhere"]})

    plan = plan_routes(route_file({}), new, {BUILDER: config([MODELS])})

    assert plan == RoutePlan(
        changes=(RouteChange(BUILDER, add=(ACCEPT,), remove=()),),
        notes=(RouteNote("raphael-elsewhere", ACCEPT, "not served here"),),
    )


def test_a_removed_route_is_removed():
    # REJECT is in PREV's file and not in NEW's: its exact entry leaves every served list in
    # allowlist mode that holds it, whether PREV named the role for it or not.
    prev = route_file({ACCEPT: [BUILDER], REJECT: [BUILDER]})
    new = route_file({ACCEPT: [BUILDER]})
    configs = {
        "default": config(),
        BUILDER: config([ACCEPT, REJECT]),
        REVIEWER: config([MODELS, "post /v1/owner-workspace/decisions/{decision_ref}/reject/"]),
    }

    plan = plan_routes(prev, new, configs)

    assert plan == RoutePlan(
        changes=(
            RouteChange(BUILDER, add=(), remove=(REJECT,)),
            RouteChange(REVIEWER, add=(), remove=(REJECT,)),
        ),
        notes=(),
    )
    assert gateway_view(apply_routes(configs[REVIEWER], plan.changes[1])) == (
        _ROUTES_ALLOWLIST,
        [MODELS],
    )


def test_the_last_entry_is_kept():
    # An empty list means "not limited": removing a list's last entry would open every route.
    prev = route_file({ACCEPT: [BUILDER], REJECT: [BUILDER]})
    configs = {BUILDER: config([REJECT]), REVIEWER: config([ACCEPT, REJECT])}

    plan = plan_routes(prev, route_file({}), configs)

    assert plan == RoutePlan(
        changes=(RouteChange(REVIEWER, add=(), remove=(ACCEPT,)),),
        notes=(
            RouteNote(BUILDER, REJECT, "kept: the last entry of the list"),
            RouteNote(REVIEWER, REJECT, "kept: the last entry of the list"),
        ),
    )
    # When the same change adds an entry, the removed one is not the list's last.
    plan = plan_routes(
        route_file({REJECT: [BUILDER]}),
        route_file({ACCEPT: [BUILDER]}),
        {BUILDER: configs[BUILDER]},
    )
    assert plan == RoutePlan(
        changes=(RouteChange(BUILDER, add=(ACCEPT,), remove=(REJECT,)),), notes=()
    )
    assert gateway_view(apply_routes(configs[BUILDER], plan.changes[0])) == (
        _ROUTES_ALLOWLIST,
        [ACCEPT],
    )


def test_an_old_prefix_is_never_removed():
    prev = route_file({ACCEPT: [BUILDER, REVIEWER], MODELS: [BUILDER]})
    new = route_file({MODELS: [BUILDER]})
    configs = {
        BUILDER: config(["/v1/owner-workspace", MODELS]),
        REVIEWER: config(["/v1/owner-workspace", ACCEPT]),
    }

    plan = plan_routes(prev, new, configs)

    assert plan == RoutePlan(changes=(RouteChange(REVIEWER, add=(), remove=(ACCEPT,)),), notes=())
    assert gateway_view(apply_routes(configs[REVIEWER], plan.changes[0])) == (
        _ROUTES_ALLOWLIST,
        ["/v1/owner-workspace"],
    )


def test_a_role_dropped_from_a_living_route_keeps_its_entry():
    # Narrowing a role is a person's decision: nothing is removed, and the owner line names it.
    prev = route_file({ACCEPT: [BUILDER, REVIEWER]})
    new = route_file({ACCEPT: [REVIEWER]})
    configs = {BUILDER: config([MODELS, ACCEPT]), REVIEWER: config([ACCEPT])}

    plan = plan_routes(prev, new, configs)

    assert plan == RoutePlan(
        changes=(), notes=(RouteNote(BUILDER, ACCEPT, "kept: the route no longer names this role"),)
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(config(["/v1/owner-workspace", MODELS]), id="prefix-covered"),
        pytest.param(config([MODELS]), id="allowlist-without-the-entry"),
        pytest.param(config([]), id="unrestricted-empty-list"),
        pytest.param(config(), id="unrestricted-no-key"),
        pytest.param(None, id="unrestricted-no-file"),
        pytest.param(config(True), id="deny-all"),
    ],
)
def test_a_role_dropped_from_a_living_route_is_named_whatever_its_list(text):
    # The note comes from the route file's role delta, whatever admits the route in that role's
    # list; the list itself stays as it is.
    prev = route_file({ACCEPT: [BUILDER, REVIEWER]})
    new = route_file({ACCEPT: [REVIEWER]})

    plan = plan_routes(prev, new, {BUILDER: text})

    assert plan == RoutePlan(
        changes=(), notes=(RouteNote(BUILDER, ACCEPT, "kept: the route no longer names this role"),)
    )


RUN = "GET /v1/runs/{run_id}"


@pytest.mark.parametrize(
    ("entry", "admitted"),
    [
        pytest.param(RUN, True, id="exact-rule"),
        pytest.param("get /v1/runs/{run_id}/", True, id="exact-rule-as-written"),
        pytest.param("/v1/runs", True, id="covering-prefix"),
        pytest.param("/v1/runs/", True, id="covering-prefix-with-slash"),
        pytest.param("/v1/runs/{run_id}", False, id="literal-placeholder-prefix"),
        pytest.param("/v1/run", False, id="not-a-segment-prefix"),
        pytest.param("POST /v1/runs/{run_id}", False, id="other-method"),
    ],
)
def test_already_admitted_is_the_gateway_matcher_on_a_real_request(entry, admitted):
    # The gateway matches a prefix against the request's concrete path, where a placeholder
    # holds a real value: only a prefix of the template's literal segments covers every request.
    plan = plan_routes(route_file({}), route_file({RUN: [BUILDER]}), {BUILDER: config([entry])})

    if admitted:
        assert plan == RoutePlan(changes=(), notes=(RouteNote(BUILDER, RUN, "already admitted"),))
    else:
        assert plan == RoutePlan(changes=(RouteChange(BUILDER, add=(RUN,), remove=()),), notes=())


# --- the rewrite of the plain form (owner facts points 9 to 15, 24 and 27) ---------------------

# The plain form as the owner read it on the production host (facts point 29): `api_server:` at 2
# spaces, `allowed_routes:` at 4 and the entries at 6, deeper than their key line. Comment lines and
# blank lines stand around the list, and one comment line follows its last entry.
_DEEPER = b"""\
# Hermes config of the builder role.

model:
  default: anthropic/claude-sonnet-5   # the model
gateway:
  # The API server of this role.
  api_server:
    enabled: true
    allowed_routes:
      - GET /v1/models
      - POST /v1/owner-workspace/decisions/{decision_ref}/reject
    # The port is fixed.

    port: 8642
  telegram:
    enabled: false
"""

# The same list with its entries at the indent of their key line.
_AT_KEY = b"""\
gateway:
  api_server:
    allowed_routes:
    - GET /v1/models
    - POST /v1/owner-workspace/decisions/{decision_ref}/reject
    # after the list

    port: 8642
# the end
"""

_PLAIN_REWRITES = [
    pytest.param(
        _DEEPER,
        RouteChange(BUILDER, add=(RUNS,), remove=()),
        b"""\
# Hermes config of the builder role.

model:
  default: anthropic/claude-sonnet-5   # the model
gateway:
  # The API server of this role.
  api_server:
    enabled: true
    allowed_routes:
      - GET /v1/models
      - POST /v1/owner-workspace/decisions/{decision_ref}/reject
      - POST /v1/runs
    # The port is fixed.

    port: 8642
  telegram:
    enabled: false
""",
        [MODELS, REJECT, RUNS],
        id="deeper-add",
    ),
    pytest.param(
        _DEEPER,
        RouteChange(BUILDER, add=(), remove=(REJECT,)),
        b"""\
# Hermes config of the builder role.

model:
  default: anthropic/claude-sonnet-5   # the model
gateway:
  # The API server of this role.
  api_server:
    enabled: true
    allowed_routes:
      - GET /v1/models
    # The port is fixed.

    port: 8642
  telegram:
    enabled: false
""",
        [MODELS],
        id="deeper-remove",
    ),
    pytest.param(
        _DEEPER,
        RouteChange(BUILDER, add=(RUNS, ACCEPT), remove=(MODELS,)),
        b"""\
# Hermes config of the builder role.

model:
  default: anthropic/claude-sonnet-5   # the model
gateway:
  # The API server of this role.
  api_server:
    enabled: true
    allowed_routes:
      - POST /v1/owner-workspace/decisions/{decision_ref}/reject
      - POST /v1/runs
      - POST /v1/owner-workspace/decisions/{decision_ref}/accept
    # The port is fixed.

    port: 8642
  telegram:
    enabled: false
""",
        [REJECT, RUNS, ACCEPT],
        id="deeper-add-and-remove",
    ),
    pytest.param(
        _AT_KEY,
        RouteChange(BUILDER, add=(RUNS,), remove=()),
        b"""\
gateway:
  api_server:
    allowed_routes:
    - GET /v1/models
    - POST /v1/owner-workspace/decisions/{decision_ref}/reject
    - POST /v1/runs
    # after the list

    port: 8642
# the end
""",
        [MODELS, REJECT, RUNS],
        id="at-key-add",
    ),
    pytest.param(
        _AT_KEY,
        RouteChange(BUILDER, add=(), remove=(MODELS,)),
        b"""\
gateway:
  api_server:
    allowed_routes:
    - POST /v1/owner-workspace/decisions/{decision_ref}/reject
    # after the list

    port: 8642
# the end
""",
        [REJECT],
        id="at-key-remove",
    ),
    pytest.param(
        _AT_KEY,
        RouteChange(BUILDER, add=(ACCEPT,), remove=(REJECT,)),
        b"""\
gateway:
  api_server:
    allowed_routes:
    - GET /v1/models
    - POST /v1/owner-workspace/decisions/{decision_ref}/accept
    # after the list

    port: 8642
# the end
""",
        [MODELS, ACCEPT],
        id="at-key-add-and-remove",
    ),
]


@pytest.mark.parametrize(("text", "change", "expected", "routes"), _PLAIN_REWRITES)
def test_a_plain_list_is_rewritten_in_its_entry_lines_only(text, change, expected, routes):
    # Byte for byte: the comment lines, the blank lines, the spacing and the text around the list
    # stay; the new entry lines go after the last entry line, with its indent, in the order of add.
    rewritten = apply_routes(text, change)

    assert rewritten == expected
    assert gateway_view(rewritten) == (_ROUTES_ALLOWLIST, routes)
    # The plan makes the same change from the route files, its entries sorted, and its dry run on
    # the same text passes.
    prev = route_file({route: [BUILDER] for route in change.remove})
    new = route_file({route: [BUILDER] for route in change.add})
    planned = RouteChange(BUILDER, tuple(sorted(change.add)), tuple(sorted(change.remove)))
    assert plan_routes(prev, new, {BUILDER: text}) == RoutePlan(changes=(planned,), notes=())


def test_every_other_byte_stays_as_it_is():
    # Non-ASCII text, a tab, spaces at line ends, a byte order mark in front of a comment line, and
    # a list that ends the text: only the entry lines change.
    text = (
        "\ufeff# Café config, édité à la main.\t\n"
        "model:   \n"
        "  default: anthropic/claude-sonnet-5\n"
        "  note: \"naïve – ok\"  \n"
        "\n"
        "gateway:\n"
        "  api_server:\n"
        "    port: 8642\n"
        "    allowed_routes:\n"
        "      - get /v1/models/\n"
        "      - /v1/owner-workspace\n"
    ).encode()
    change = RouteChange(BUILDER, add=(RUNS,), remove=(MODELS,))

    rewritten = apply_routes(text, change)

    assert rewritten == (
        "\ufeff# Café config, édité à la main.\t\n"
        "model:   \n"
        "  default: anthropic/claude-sonnet-5\n"
        "  note: \"naïve – ok\"  \n"
        "\n"
        "gateway:\n"
        "  api_server:\n"
        "    port: 8642\n"
        "    allowed_routes:\n"
        "      - /v1/owner-workspace\n"
        "      - POST /v1/runs\n"
    ).encode()
    assert gateway_view(rewritten) == (_ROUTES_ALLOWLIST, ["/v1/owner-workspace", RUNS])


def test_a_removed_entry_takes_out_each_line_that_normalises_to_it():
    text = b"""\
gateway:
  api_server:
    allowed_routes:
      - POST /v1/runs
      - GET /v1/models
      - get /v1/models/
      - GET   /v1/models
"""
    assert gateway_view(text) == (_ROUTES_ALLOWLIST, [RUNS, MODELS, MODELS, MODELS])

    rewritten = apply_routes(text, RouteChange(BUILDER, add=(), remove=(MODELS,)))

    assert rewritten == b"""\
gateway:
  api_server:
    allowed_routes:
      - POST /v1/runs
"""


def test_an_added_entry_the_list_holds_is_not_added_again():
    text = config([MODELS, "post /v1/runs/"])

    rewritten = apply_routes(text, RouteChange(BUILDER, add=(RUNS, ACCEPT, ACCEPT), remove=()))

    assert rewritten == config([MODELS, "post /v1/runs/", ACCEPT])


def test_an_added_entry_outside_condition_f_refuses():
    # "GET /v1/files:upload" is an exact route, but a colon is not a character of the plain form.
    text = config([MODELS])
    added = "GET /v1/files:upload"

    with pytest.raises(RoutePlanError) as refused:
        apply_routes(text, RouteChange(BUILDER, add=(RUNS, added), remove=()))

    assert refused.value.reason == (
        f"the change for role {BUILDER} adds {added}, which is not an entry of the plain form "
        "(condition f)"
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(config([MODELS, REJECT]), id="list"),
        pytest.param(_AT_KEY, id="entries-at-the-key-indent"),
    ],
)
def test_applying_the_same_change_twice_adds_nothing_twice(text):
    change = RouteChange(BUILDER, add=(ACCEPT, RUNS), remove=(REJECT,))

    once = apply_routes(text, change)

    assert apply_routes(once, change) == once
    assert gateway_view(once) == (_ROUTES_ALLOWLIST, [MODELS, ACCEPT, RUNS])


@pytest.mark.parametrize(
    ("text", "change"),
    [
        pytest.param(config([REJECT]), RouteChange(BUILDER, (), (REJECT,)), id="the-only-entry"),
        pytest.param(
            config([REJECT, "post /v1/owner-workspace/decisions/{decision_ref}/reject/"]),
            RouteChange(BUILDER, (), (REJECT,)),
            id="every-line-of-the-entry",
        ),
        pytest.param(
            config([REJECT, MODELS]), RouteChange(BUILDER, (), (REJECT, MODELS)), id="every-entry"
        ),
    ],
)
def test_a_change_that_would_leave_no_entry_refuses(text, change):
    # An empty list means "not limited": a list with no entry would open every route.
    with pytest.raises(RoutePlanError) as refused:
        apply_routes(text, change)

    assert refused.value.reason == (
        f"the change for role {BUILDER} would leave its route list with no entry, "
        "which would open every route"
    )


def _rich_config(routes: list[str]) -> bytes:
    """A config.yaml with many kinds of values around the route list, as hermes writes it."""
    return _hermes_yaml(
        {
            "model": {
                "default": "anthropic/claude-sonnet-5",
                "base_url": None,
                "max_tokens": 64000,
            },
            "agent": {
                "max_turns": 90,
                "verbose": False,
                "temperature": 0.7,
                "system_prompt": "You build what the owner decided, and nothing else. " * 4,
                "notes": "line one\nline two\n",
            },
            "toolsets": ["web", "terminal", "file"],
            "display": {"spinner": "✨ thinking ✨", "skin": None},
            "looks_like": {
                "yes": "yes",
                "octal": "0755",
                "float": "1e3",
                "null": "null",
                "empty": "",
            },
            "since": datetime.date(2026, 11, 26),
            "gateway": {
                "api_server": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8642,
                    "key": None,
                    "allowed_routes": routes,
                    "cors_origins": ["http://localhost:5173"],
                },
                "telegram": {"enabled": False, "allowed_users": []},
            },
            "mcp_servers": {
                "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
            },
        }
    )


def test_the_rewrite_never_touches_any_key_outside_the_route_list():
    text = _rich_config([MODELS, REJECT])

    rewritten = apply_routes(text, RouteChange(BUILDER, add=(ACCEPT,), remove=(REJECT,)))

    assert rewritten == _rich_config([MODELS, ACCEPT])
    old, new = yaml.safe_load(text), yaml.safe_load(rewritten)
    del old["gateway"]["api_server"]["allowed_routes"]
    del new["gateway"]["api_server"]["allowed_routes"]
    assert new == old


# --- texts outside the plain form (facts points 10 to 13, 25 and 26) -----------------------------

# Each text below breaks one condition of the plain form, and the gateway reads each as an
# allowlist. The id names the condition and, where there is one, the example of facts point 12.
_NOT_PLAIN = [
    # a. The text is UTF-8, holds no carriage return and ends with a line feed.
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models",
        "a",
        id="a-no-line-feed-at-the-end",
    ),
    pytest.param(
        b"gateway:\r\n  api_server:\r\n    allowed_routes:\r\n      - GET /v1/models\r\n",
        "a",
        id="a-carriage-returns",
    ),
    # b. Exactly one line of the text is `gateway:`, with no indent and nothing after the colon.
    pytest.param(
        b"""\
gateway:
  telegram:
    enabled: false
gateway:
  api_server:
    allowed_routes:
      - GET /v1/models
""",
        "b",
        id="b-path-key-written-two-times",
    ),
    pytest.param(
        b"gateway:  # the gateway\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n",
        "b",
        id="b-comment-at-the-end-of-a-key-line",
    ),
    pytest.param(
        b"gateway: &gateway\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n",
        "b",
        id="b-anchor-on-the-path",
    ),
    pytest.param(
        b"gateway: !!map\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n",
        "b",
        id="b-tag-on-the-path",
    ),
    pytest.param(
        b"gateway: {api_server: {allowed_routes: [GET /v1/models]}}\n",
        "b",
        id="b-path-in-braces",
    ),
    # c. Exactly one direct child line of `gateway:` is `api_server:`, with nothing after the colon.
    pytest.param(
        b"""\
gateway:
  api_server:
    port: 8642
  api_server:
    allowed_routes:
      - GET /v1/models
""",
        "c",
        id="c-path-key-written-two-times",
    ),
    pytest.param(
        b"gateway:\n  api_server:  # c\n    allowed_routes:\n      - GET /v1/models\n",
        "c",
        id="c-comment-at-the-end-of-a-key-line",
    ),
    pytest.param(
        b"""\
shared: &api
  allowed_routes:
    - GET /v1/models
gateway:
  api_server: *api
""",
        "c",
        id="c-alias-on-the-path",
    ),
    # d. Exactly one direct child line of `api_server:` is `allowed_routes:`, nothing after it.
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes: [GET /v1/models]\n",
        "d",
        id="d-list-in-brackets",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes: GET /v1/models\n",
        "d",
        id="d-one-text-in-place-of-a-list",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:  # c\n      - GET /v1/models\n",
        "d",
        id="d-comment-at-the-end-of-a-key-line",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes: &routes\n      - GET /v1/models\n",
        "d",
        id="d-anchor-on-the-path",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes: !!seq\n      - GET /v1/models\n",
        "d",
        id="d-tag-on-the-path",
    ),
    pytest.param(
        b"""\
routes: &routes
  - GET /v1/models
gateway:
  api_server:
    allowed_routes: *routes
""",
        "d",
        id="d-alias-on-the-path",
    ),
    pytest.param(
        b"""\
shared: &api
  allowed_routes:
    - GET /v1/models
gateway:
  api_server:
    <<: *api
    port: 8642
""",
        "d",
        id="d-merge-key-on-the-path",
    ),
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
      - POST /v1/runs
    allowed_routes:
      - GET /v1/models
""",
        "d",
        id="d-path-key-written-two-times",
    ),
    # e. Entry lines follow `allowed_routes:` directly, all at one indent, not left of the key.
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n\n      - GET /v1/models\n",
        "e",
        id="e-blank-line-before-the-first-entry",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      # c\n      - GET /v1/models\n",
        "e",
        id="e-comment-line-before-the-first-entry",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      [GET /v1/models]\n",
        "e",
        id="e-list-in-brackets-on-the-next-line",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      GET /v1/models\n",
        "e",
        id="e-one-text-on-the-next-line",
    ),
    # f. An entry starts with a letter or a slash, holds only letters, digits, spaces and
    # / _ . { } * -, and does not end with a space.
    pytest.param(
        b'gateway:\n  api_server:\n    allowed_routes:\n      - "GET /v1/models"\n',
        "f",
        id="f-quoted-entry",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - 'GET /v1/models'\n",
        "f",
        id="f-single-quoted-entry",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models  # c\n",
        "f",
        id="f-comment-at-the-end-of-an-entry-line",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models \n",
        "f",
        id="f-space-at-the-end",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      -  GET /v1/models\n",
        "f",
        id="f-two-spaces-after-the-dash",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - &models GET /v1/models\n",
        "f",
        id="f-anchor-in-the-list",
    ),
    pytest.param(
        b"""\
models: &models GET /v1/models
gateway:
  api_server:
    allowed_routes:
      - *models
""",
        "f",
        id="f-alias-in-the-list",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - !!str GET /v1/models\n",
        "f",
        id="f-tag-in-the-list",
    ),
    pytest.param(
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n      - /v1/~x\n",
        "f",
        id="f-character-outside-the-set",
    ),
    # g. The first line after the last entry line that is not blank and not a comment line does
    # not belong to the list.
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
      - GET /v1/models
      # c
      - GET /v1/models
""",
        "g",
        id="g-comment-line-between-two-entries",
    ),
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
      - GET /v1/models

      - GET /v1/models
""",
        "g",
        id="g-blank-line-between-two-entries",
    ),
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
    - GET /v1/models

    - GET /v1/models
""",
        "g",
        id="g-blank-line-between-two-entries-at-the-key-indent",
    ),
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
    - GET /v1/models
    # c
    - GET /v1/models
""",
        "g",
        id="g-comment-line-between-two-entries-at-the-key-indent",
    ),
    # h. The lines and the gateway agree: here the gateway reads the last of two keys that are
    # equal once read, and the lines see only the plain one.
    pytest.param(
        b"""\
gateway:
  api_server:
    allowed_routes:
      - POST /v1/runs
    "allowed_routes":
      - GET /v1/models
""",
        "h",
        id="h-a-quoted-twin-of-the-list-key",
    ),
]


def _refused_before_any_edit(monkeypatch, text: bytes, change: RouteChange) -> str:
    """The reason apply_routes refuses `text` for; the refusal comes before the edit step runs,
    and the input is the same bytes afterwards."""
    before = bytes(text)
    edits: list[object] = []
    edit = release_routes._edit
    monkeypatch.setattr(release_routes, "_edit", lambda *args: edits.append(args) or edit(*args))

    with pytest.raises(RoutePlanError) as refused:
        apply_routes(text, change)

    assert edits == []
    assert text == before
    return refused.value.reason


@pytest.mark.parametrize(("text", "condition"), _NOT_PLAIN)
def test_a_text_outside_the_plain_form_refuses_before_any_edit(monkeypatch, text, condition):
    assert gateway_view(text)[0] == _ROUTES_ALLOWLIST

    reason = _refused_before_any_edit(monkeypatch, text, RouteChange(BUILDER, (ACCEPT,), ()))

    assert reason.startswith(
        f"the config of role {BUILDER} is not in the plain form: "
        f"condition {condition} does not hold: "
    )
    # The plan applies the change it makes to the same text, so it refuses before the cutover.
    with pytest.raises(RoutePlanError) as refused:
        plan_routes(route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: text})
    assert refused.value.reason == (
        f"the change for role {BUILDER} does not apply to its config: {reason}"
    )
    # A role with no change is never refused for its written form.
    configs = {BUILDER: text, REVIEWER: config([MODELS])}
    plan = plan_routes(route_file({}), route_file({ACCEPT: [REVIEWER]}), configs)
    assert plan == RoutePlan(changes=(RouteChange(REVIEWER, (ACCEPT,), ()),), notes=())


def test_a_second_document_refuses(monkeypatch):
    # The gateway reads no list from a text of two documents: it denies all, and so do both
    # functions when they would change the list.
    text = config([MODELS]) + b"---\nother: 1\n"
    assert gateway_view(text) == (_ROUTES_DENY_ALL, [])

    reason = _refused_before_any_edit(monkeypatch, text, RouteChange(BUILDER, (ACCEPT,), ()))

    assert reason == f"role {BUILDER} has a route list that is not valid"
    with pytest.raises(RoutePlanError):
        plan_routes(route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: text})
    assert plan_routes(route_file({}), route_file({}), {BUILDER: text}) == RoutePlan((), ())


def test_lines_the_gateway_reads_otherwise_refuse(monkeypatch):
    # Conditions a to g hold for this text; when the gateway's reading gives another list, the
    # lines and the gateway do not agree (condition h).
    text = config([MODELS])
    monkeypatch.setattr(
        release_routes, "_route_list", lambda _text: (_ROUTES_ALLOWLIST, [MODELS, REJECT])
    )

    reason = _refused_before_any_edit(monkeypatch, text, RouteChange(BUILDER, (ACCEPT,), ()))

    assert reason == (
        f"the config of role {BUILDER} is not in the plain form: condition h does not hold: "
        "the gateway reads another list than the entry lines"
    )


@pytest.mark.parametrize(
    ("text", "words"),
    [
        pytest.param(
            config([MODELS]).replace(b"claude-sonnet-5", b"claude-sonnet-\xff"),
            "condition a does not hold: the text is not UTF-8",
            id="a-not-utf-8",
        ),
        pytest.param(
            b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n"
            b"    - POST /v1/runs\n",
            "condition e does not hold: the entry lines do not all have the same indent",
            id="e-two-indents",
        ),
        pytest.param(
            b"gateway:\n  api_server:\n    allowed_routes:\n  - GET /v1/models\n",
            "condition e does not hold: the entry lines are less deep than 'allowed_routes:'",
            id="e-less-deep-than-the-key",
        ),
    ],
)
def test_texts_the_gateway_does_not_read_as_a_list_are_refused_by_the_lines_too(
    monkeypatch, text, words
):
    # The gateway denies all for these texts, so apply_routes refuses them before it reads lines;
    # with the gateway's reading patched to a list, the line reader refuses them on its own.
    assert gateway_view(text) == (_ROUTES_DENY_ALL, [])
    monkeypatch.setattr(release_routes, "_route_list", lambda _text: (_ROUTES_ALLOWLIST, [MODELS]))

    reason = _refused_before_any_edit(monkeypatch, text, RouteChange(BUILDER, (ACCEPT,), ()))

    assert reason == f"the config of role {BUILDER} is not in the plain form: {words}"


def test_the_reasons_name_the_first_condition_in_plain_words():
    cases = {
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models": (
            "condition a does not hold: the text does not end with a line feed"
        ),
        b"gateway:\r\n  api_server:\r\n    allowed_routes:\r\n      - GET /v1/models\r\n": (
            "condition a does not hold: the text holds a carriage return"
        ),
        b"gateway: &g\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n": (
            "condition b does not hold: no line of the text is 'gateway:' with no indent and "
            "nothing after the colon"
        ),
        b"gateway:\n  api_server:\n    allowed_routes:\n      # c\n      - GET /v1/models\n": (
            "condition e does not hold: the line after 'allowed_routes:' is not an entry line"
        ),
        b'gateway:\n  api_server:\n    allowed_routes:\n      - "GET /v1/models"\n': (
            "condition f does not hold: the entry on line 4 does not start with a letter or a "
            "slash"
        ),
        b"gateway:\n  api_server:\n    allowed_routes:\n      - GET /v1/models\n"
        b"      # c\n      - GET /v1/models\n": (
            "condition g does not hold: line 6, after the last entry line, is deeper than "
            "'allowed_routes:'"
        ),
    }
    for text, words in cases.items():
        with pytest.raises(RoutePlanError) as refused:
            apply_routes(text, RouteChange(BUILDER, (ACCEPT,), ()))
        assert refused.value.reason == (
            f"the config of role {BUILDER} is not in the plain form: {words}"
        )


def test_a_rewrite_that_would_change_another_byte_refuses(monkeypatch):
    # The check after the edit: no text comes back unless only its entry lines changed.
    real_edit = release_routes._edit
    monkeypatch.setattr(
        release_routes, "_edit", lambda *args: real_edit(*args).replace(b"8642", b"8643")
    )

    with pytest.raises(RoutePlanError) as refused:
        apply_routes(_DEEPER, RouteChange(BUILDER, (ACCEPT,), ()))

    assert refused.value.reason == (
        f"the rewrite of role {BUILDER} would change more than the entry lines of its route list"
    )


def test_a_rewrite_the_gateway_reads_otherwise_refuses(monkeypatch):
    # The other check after the edit: the gateway's reading of the result is the planned list.
    real_edit = release_routes._edit
    monkeypatch.setattr(
        release_routes, "_edit", lambda *args: real_edit(*args).replace(b"- POST", b"- PUT")
    )

    with pytest.raises(RoutePlanError) as refused:
        apply_routes(config([MODELS]), RouteChange(BUILDER, (RUNS,), ()))

    assert refused.value.reason == (
        f"the rewritten route list of role {BUILDER} is not the planned one"
    )


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "which",
    ["NEW", "PREV"],
)
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{", id="not-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param(json.dumps({ACCEPT: BUILDER}), id="roles-not-a-list"),
        pytest.param(json.dumps({ACCEPT: [7]}), id="role-not-a-string"),
    ],
)
def test_a_route_file_that_does_not_parse_refuses(which, text):
    good = route_file({ACCEPT: [BUILDER]})
    prev, new = (good, text) if which == "NEW" else (text, good)

    with pytest.raises(RoutePlanError) as refused:
        plan_routes(prev, new, {BUILDER: config([MODELS])})

    assert refused.value.reason == f"{which}'s route file does not parse"
    if which == "NEW":
        # NEW's file is read even on the first release that carries it.
        with pytest.raises(RoutePlanError):
            plan_routes(None, new, {BUILDER: config([MODELS])})


@pytest.mark.parametrize(
    "key",
    [
        "FETCH /v1/models",
        "GET v1/models",
        "GET /v1/models?all=1",
        "",
        "get /v1/models",
        "GET /v1/models/",
        "/v1/models",
    ],
)
def test_a_key_that_does_not_normalise_refuses(key):
    # No entry is ever written for a key that is not a route rule as the gateway writes it, and a
    # written entry is exact, never a path prefix.
    with pytest.raises(RoutePlanError) as refused:
        plan_routes(route_file({}), route_file({key: [BUILDER]}), {BUILDER: config([RUNS])})

    assert refused.value.reason == f"NEW's route file has a key that does not normalise: {key!r}"


_DUPLICATE_KEYS = b"""\
gateway:
  api_server:
    allowed_routes:
      - GET /v1/models
    port: 8642
    port: 8643
"""


def test_a_duplicate_key_beside_the_route_list_is_left_as_it_is():
    # The gateway reads the last of two equal keys; one off the route list's path changes nothing.
    rewritten = apply_routes(_DUPLICATE_KEYS, RouteChange(BUILDER, add=(RUNS,), remove=()))

    assert rewritten == _DUPLICATE_KEYS.replace(
        b"      - GET /v1/models\n", b"      - GET /v1/models\n      - POST /v1/runs\n"
    )
    assert gateway_view(rewritten) == (_ROUTES_ALLOWLIST, [MODELS, RUNS])


@pytest.mark.parametrize(
    ("text", "change", "reason"),
    [
        pytest.param(
            config(),
            RouteChange(BUILDER, add=(ACCEPT,), remove=()),
            f"role {BUILDER} has a route list that does not limit routes",
            id="unrestricted",
        ),
        pytest.param(
            config(True),
            RouteChange(BUILDER, add=(ACCEPT,), remove=()),
            f"role {BUILDER} has a route list that is not valid",
            id="deny-all",
        ),
        pytest.param(
            config([REJECT]),
            RouteChange(BUILDER, add=(), remove=(REJECT,)),
            f"the change for role {BUILDER} would leave its route list with no entry, "
            "which would open every route",
            id="the-last-entry",
        ),
        pytest.param(
            config([MODELS]),
            RouteChange(BUILDER, add=("/v1/owner-workspace",), remove=()),
            f"the change for role {BUILDER} has an entry that is not an exact route: "
            "'/v1/owner-workspace'",
            id="adds-a-prefix",
        ),
        pytest.param(
            config(["/v1/owner-workspace", MODELS]),
            RouteChange(BUILDER, add=(), remove=("/v1/owner-workspace",)),
            f"the change for role {BUILDER} has an entry that is not an exact route: "
            "'/v1/owner-workspace'",
            id="removes-a-prefix",
        ),
        pytest.param(
            config([MODELS]),
            RouteChange(BUILDER, add=("post /v1/runs",), remove=()),
            f"the change for role {BUILDER} has an entry that is not an exact route: "
            "'post /v1/runs'",
            id="not-normalised",
        ),
        pytest.param(
            config([MODELS]),
            RouteChange(BUILDER, add=(RUNS,), remove=(RUNS,)),
            f"the change for role {BUILDER} both adds and removes {RUNS}",
            id="adds-and-removes",
        ),
    ],
)
def test_apply_refuses_unless_the_result_is_the_planned_allowlist(text, change, reason):
    with pytest.raises(RoutePlanError) as refused:
        apply_routes(text, change)

    assert refused.value.reason == reason


def test_apply_with_nothing_to_change_returns_the_text_unchanged():
    text = b"gateway:\n  api_server:\n    allowed_routes: GET /v1/models   # as written\n"

    assert apply_routes(text, RouteChange(BUILDER, add=(MODELS,), remove=(REJECT,))) == text


# --- K6 and purity -------------------------------------------------------------------------------

# Calls that reach every branch of both functions, so the two tests below cover all of them.
_CALLS = (
    ("plan", (None, route_file({ACCEPT: [BUILDER]}), {BUILDER: config([MODELS])})),
    (
        "plan",
        (
            route_file({ACCEPT: [BUILDER, REVIEWER], REJECT: [BUILDER, "raphael-ops"]}),
            route_file(
                {
                    ACCEPT: [REVIEWER],
                    MODELS: [BUILDER, REVIEWER, "raphael-elsewhere"],
                    RUNS: [BUILDER, "default"],
                }
            ),
            {
                "default": None,
                BUILDER: config(["/v1/models", ACCEPT, REJECT, RUNS]),
                REVIEWER: config([REJECT]),
                "raphael-ops": config([REJECT]),
            },
        ),
    ),
    ("plan", (route_file({}), route_file({ACCEPT: [BUILDER]}), {BUILDER: b"gateway: ["})),
    ("plan", (route_file({}), "{", {})),
    ("plan", ("[]", route_file({}), {})),
    ("plan", (route_file({}), route_file({"GET v1": []}), {})),
    *(("apply", (layout.values[0], layout.values[1])) for layout in _PLAIN_REWRITES),
    *(
        ("apply", (layout.values[0], RouteChange(BUILDER, add=(ACCEPT,), remove=())))
        for layout in _NOT_PLAIN
    ),
    ("apply", (config([MODELS]), RouteChange(BUILDER, add=("GET /v1/files:upload",), remove=()))),
    (
        "apply",
        (
            b"gateway: {api_server: {allowed_routes: [GET /v1/models, GET /v1/x]}}  # flow\n",
            RouteChange(BUILDER, add=(RUNS,), remove=("GET /v1/x",)),
        ),
    ),
    (
        "apply",
        (_rich_config([MODELS, REJECT]), RouteChange(BUILDER, add=(ACCEPT,), remove=(REJECT,))),
    ),
    ("apply", (config([MODELS]), RouteChange(BUILDER, add=(MODELS,), remove=()))),
    ("apply", (config(), RouteChange(BUILDER, add=(ACCEPT,), remove=()))),
    ("apply", (config([REJECT]), RouteChange(BUILDER, add=(), remove=(REJECT,)))),
    ("apply", (config([MODELS]), RouteChange(BUILDER, add=("/v1",), remove=()))),
    ("apply", (config([MODELS]), RouteChange(BUILDER, add=(RUNS,), remove=(RUNS,)))),
    ("apply", (_DUPLICATE_KEYS, RouteChange(BUILDER, add=(RUNS,), remove=()))),
    (
        "apply",
        (
            b"shared: &r\n  - GET /v1/models\ngateway:\n  api_server:\n    allowed_routes: *r\n",
            RouteChange(BUILDER, add=(RUNS,), remove=()),
        ),
    ),
)


def _run(calls) -> list[object]:
    results: list[object] = []
    for name, args in calls:
        try:
            results.append(plan_routes(*args) if name == "plan" else apply_routes(*args))
        except RoutePlanError as refused:
            results.append(refused.reason)
    return results


# The child imports only the module and what reads its input, then calls both functions with the
# same calls as _CALLS. Bytes travel as latin-1 text, which maps every byte to one character.
_CHILD = """\
import json, sys
from hermes_cli.release_routes import RouteChange, RoutePlanError, apply_routes, plan_routes
calls = json.loads(sys.stdin.read())
before = set(sys.modules)
for name, args in calls:
    try:
        if name == "plan":
            prev, new, configs = args
            configs = {k: None if v is None else v.encode("latin-1") for k, v in configs.items()}
            plan_routes(prev, new, configs)
        else:
            text, role, add, remove = args
            apply_routes(text.encode("latin-1"), RouteChange(role, tuple(add), tuple(remove)))
    except RoutePlanError:
        pass
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def _calls_as_json(calls) -> str:
    out = []
    for name, args in calls:
        if name == "plan":
            prev, new, configs = args
            configs = {k: None if v is None else v.decode("latin-1") for k, v in configs.items()}
            out.append([name, [prev, new, configs]])
        else:
            text, change = args
            out.append([name, [text.decode("latin-1"), change.role, change.add, change.remove]])
    return json.dumps(out)


def test_the_module_imports_nothing_inside_a_function():
    # K6: everything the functions need is imported when the module loads. A fresh interpreter
    # makes this a real check: in this process an earlier test may already have imported what a
    # call would import.
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=_calls_as_json(_CALLS),
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=240,
    )

    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == []


_AUDITED: list[str] = []
_RECORDING: list[bool] = []
_HOOKED: list[bool] = []


def _record(event: str, args: tuple) -> None:
    if _RECORDING:
        _AUDITED.append(event)


class _WatchedEnviron(MutableMapping):
    """os.environ that notes each key read or written while _RECORDING is set."""

    def __init__(self, real, touched: list[object]) -> None:
        self._real = real
        self._touched = touched

    def _note(self, key: object) -> None:
        if _RECORDING:
            self._touched.append(key)

    def __getitem__(self, key):
        self._note(key)
        return self._real[key]

    def __setitem__(self, key, value):
        self._note(key)
        self._real[key] = value

    def __delitem__(self, key):
        self._note(key)
        del self._real[key]

    def __iter__(self):
        self._note("<every key>")
        return iter(self._real)

    def __len__(self):
        self._note("<every key>")
        return len(self._real)

    def copy(self):
        self._note("<every key>")
        return self._real.copy()


def test_the_functions_do_no_file_network_process_or_environment_access(monkeypatch):
    # Every file, socket, process and import step raises an audit event; environment reads do
    # not, so os.environ is watched as well. Nothing but the introspection _record allows may
    # happen while the functions run.
    if not _HOOKED:
        # Audit hooks cannot be removed, so this one records only while _RECORDING is set.
        sys.addaudithook(_record)
        _HOOKED.append(True)
    touched: list[object] = []
    monkeypatch.setattr(os, "environ", _WatchedEnviron(os.environ, touched))
    if hasattr(os, "environb"):
        # Only where the interpreter keeps a bytes environment (not on Windows).
        monkeypatch.setattr(os, "environb", _WatchedEnviron(os.environb, touched))
    configs = _CALLS[1][1][2]
    configs_before = dict(configs)
    _AUDITED.clear()
    _RECORDING.append(True)
    try:
        first = _run(_CALLS)
        second = _run(_CALLS)
    finally:
        _RECORDING.clear()

    assert _AUDITED == []
    assert touched == []
    # The result depends only on the arguments, and the arguments are left as they were.
    assert first == second
    assert configs == configs_before
