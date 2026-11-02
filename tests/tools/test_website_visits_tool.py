"""``website_visits`` — Cloudflare Web Analytics visits and the LinkedIn share.

Every test dispatches through the real registry against this test's
``HERMES_HOME``: a real ``config.yaml`` read by ``load_config()`` and a real
token file. Cloudflare is a recorded answer served at the httpx transport and
sockets refuse to connect, so no test can reach Cloudflare.
"""

import json
import logging
import socket

import httpx
import pytest
import yaml

from hermes_constants import get_hermes_home

NOT_SET_UP = "Website visits are not set up yet."
TOKEN = "recorded-fake-token-5f3c9a1e7b2d"
ACCOUNT_TAG = "4f9a2c7e1b3d5f6a8c0e2b4d6f8a0c1e"
SITE_TAG = "9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a"
START = "2026-10-25T00:00:00Z"
END = "2026-11-01T00:00:00Z"
GRAPHQL_URL = "https://api.cloudflare.com/client/v4/graphql"

# The total node's answer. It is deliberately not the sum of the referrer
# rows, so the tests can tell which node a count came from.
RECORDED_TOTAL = 1480
# The limit-100 node's rows, most visits first as orderBy sum_visits_DESC
# returns them. "" is the row for visits that sent no referrer.
RECORDED_REFERRERS = [
    ("", 640),
    ("www.google.com", 310),
    ("www.linkedin.com", 120),
    ("example.com", 95),
    ("lnkd.in", 40),
    ("duckduckgo.com", 33),
    ("LinkedIn.com", 25),
    ("news.ycombinator.com", 21),
    ("www.bing.com", 18),
    ("t.co", 16),
    ("github.com", 14),
    ("www.reddit.com", 13),
    ("linkedin-clone.example", 12),
    ("search.brave.com", 11),
    ("www.ecosia.org", 10),
    ("mail.google.com", 9),
    ("outlook.live.com", 8),
    ("www.facebook.com", 7),
    ("m.facebook.com", 6),
    ("notlnkd.in", 5),
    ("www.xing.com", 4),
    ("l.instagram.com", 3),
    ("www.linkedin.cn", 2),
    ("yandex.ru", 1),
]
# www.linkedin.com + lnkd.in + LinkedIn.com + linkedin-clone.example
# + www.linkedin.cn (past the 20 listed hosts, still a LinkedIn host).
# example.com and notlnkd.in do not count.
RECORDED_LINKEDIN = 120 + 40 + 25 + 12 + 2

FORBIDDEN_BODY = {
    "success": False,
    "errors": [{"code": 10000, "message": "Authentication error"}],
    "messages": [],
    "result": None,
}
GRAPHQL_ERRORS_BODY = {
    "data": None,
    "errors": [
        {
            "message": "not authorized for that account",
            "path": ["viewer", "accounts", "0", "total"],
            "extensions": {"code": "authz", "timestamp": "2026-11-01T09:00:00Z"},
        }
    ],
}


def _recorded_answer(total=None, referrers=None):
    """Cloudflare's answer to the query's two rumPageloadEventsAdaptiveGroups nodes."""
    total = [{"sum": {"visits": RECORDED_TOTAL}}] if total is None else total
    referrers = RECORDED_REFERRERS if referrers is None else referrers
    return {
        "data": {
            "viewer": {
                "accounts": [
                    {
                        "total": total,
                        "topReferers": [
                            {"sum": {"visits": visits}, "dimensions": {"refererHost": host}}
                            for host, visits in referrers
                        ],
                    }
                ]
            }
        },
        "errors": None,
    }


def _respond(status, **content):
    return lambda request: httpx.Response(status, **content)


def _raise(exc_type, message):
    def answer(request):
        raise exc_type(message, request=request)

    return answer


class _RecordedCloudflare:
    """Answers every request with ``answer`` and keeps every request it got."""

    def __init__(self):
        self.requests = []
        self.answer = _respond(200, json=_recorded_answer())

    def handle(self, request):
        self.requests.append(request)
        return self.answer(request)


def _refuse_network(sock, address):
    raise AssertionError(f"a test tried to open a network connection to {address!r}")


@pytest.fixture
def cloudflare(monkeypatch):
    fake = _RecordedCloudflare()
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda transport, request: fake.handle(request)
    )
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    return fake


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "secrets" / "cloudflare-analytics.token"
    path.parent.mkdir()
    path.write_text(f"{TOKEN}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.fixture
def configured(token_file):
    _write_settings(account_tag=ACCOUNT_TAG, site_tag=SITE_TAG, token_file=str(token_file))
    return token_file


def _write_settings(**settings):
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"website_visits": settings}), encoding="utf-8"
    )


def _dispatch_text(start=START, end=END):
    """Call the tool the way the agent does: through the real registry."""
    import tools.website_visits_tool  # noqa: F401 - ensure registered
    from tools.registry import registry

    return registry.dispatch("website_visits", {"start": start, "end": end})


def _dispatch(start=START, end=END):
    return json.loads(_dispatch_text(start, end))


def _assert_not_set_up(out, cloudflare):
    assert out == {"error": NOT_SET_UP}
    assert cloudflare.requests == []


# (a) ---------------------------------------------------------------------


def test_recorded_answer_gives_total_and_linkedin_count(cloudflare, configured):
    out = _dispatch()

    assert set(out) == {"start", "end", "visits", "linkedin_visits", "hosts", "note"}
    assert (out["start"], out["end"]) == (START, END)
    assert out["visits"] == RECORDED_TOTAL
    assert out["linkedin_visits"] == RECORDED_LINKEDIN
    assert out["hosts"] == [
        {"host": host, "visits": visits} for host, visits in RECORDED_REFERRERS[:20]
    ]
    note = out["note"].lower()
    assert "floor" in note
    assert "ad blocker" in note and "linkedin app" in note


def test_window_without_visits_counts_zero(cloudflare, configured):
    cloudflare.answer = _respond(200, json=_recorded_answer(total=[], referrers=[]))

    out = _dispatch()

    assert (out["visits"], out["linkedin_visits"], out["hosts"]) == (0, 0, [])


# (b) ---------------------------------------------------------------------


def test_sent_query_holds_account_site_and_both_times(cloudflare, configured):
    _dispatch()

    assert len(cloudflare.requests) == 1
    request = cloudflare.requests[0]
    assert request.method == "POST"
    assert str(request.url) == GRAPHQL_URL
    # The token file ends in a newline; the header carries the trimmed token.
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.extensions["timeout"] == {
        "connect": 20.0, "read": 20.0, "write": 20.0, "pool": 20.0,
    }
    sent = request.content.decode()
    assert "rumPageloadEventsAdaptiveGroups" in json.loads(sent)["query"]
    for value in (ACCOUNT_TAG, SITE_TAG, START, END):
        assert value in sent


# (c) ---------------------------------------------------------------------


def test_no_settings_is_not_set_up(cloudflare):
    _assert_not_set_up(_dispatch(), cloudflare)


@pytest.mark.parametrize("missing", ["account_tag", "site_tag", "token_file"])
def test_missing_setting_is_not_set_up(cloudflare, token_file, missing):
    settings = {"account_tag": ACCOUNT_TAG, "site_tag": SITE_TAG, "token_file": str(token_file)}
    del settings[missing]
    _write_settings(**settings)

    _assert_not_set_up(_dispatch(), cloudflare)


def test_missing_token_file_is_not_set_up(cloudflare, tmp_path):
    _write_settings(
        account_tag=ACCOUNT_TAG, site_tag=SITE_TAG, token_file=str(tmp_path / "absent.token")
    )

    _assert_not_set_up(_dispatch(), cloudflare)


def test_relative_token_file_is_not_set_up(cloudflare):
    _write_settings(account_tag=ACCOUNT_TAG, site_tag=SITE_TAG, token_file="cloudflare.token")

    _assert_not_set_up(_dispatch(), cloudflare)


def test_unreadable_token_file_is_not_set_up(cloudflare, tmp_path):
    # Root reads a file whatever its mode, and a mode-0000 file would trip
    # the 0600 rule first. A directory with mode 0600 passes the mode rule
    # and still cannot be read as a file, by root or by a normal CI user.
    unreadable = tmp_path / "cloudflare-analytics.token"
    unreadable.mkdir()
    unreadable.chmod(0o600)
    _write_settings(account_tag=ACCOUNT_TAG, site_tag=SITE_TAG, token_file=str(unreadable))

    _assert_not_set_up(_dispatch(), cloudflare)


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604], ids=["0644", "0640", "0604"])
def test_token_file_mode_other_than_0600_is_not_set_up(cloudflare, configured, mode):
    configured.chmod(mode)

    _assert_not_set_up(_dispatch(), cloudflare)


@pytest.mark.parametrize(
    "content",
    [b"", b"\n", f"Bearer {TOKEN}\n".encode(), f"{TOKEN}\n{TOKEN}\n".encode(), b"\xff\xfe\x00"],
    ids=["empty", "blank", "header-pasted", "two-lines", "not-text"],
)
def test_token_file_without_one_token_is_not_set_up(cloudflare, configured, content):
    configured.write_bytes(content)

    _assert_not_set_up(_dispatch(), cloudflare)


# (d) ---------------------------------------------------------------------


def test_403_answers_fixed_sentence_with_status_and_no_body(cloudflare, configured):
    cloudflare.answer = _respond(403, json=FORBIDDEN_BODY)
    out = _dispatch()

    assert set(out) == {"error"}
    assert "403" in out["error"]
    for fragment in ("Authentication error", "10000", json.dumps(FORBIDDEN_BODY)):
        assert fragment not in out["error"]

    # Fixed: a different 403 body gives the same answer.
    cloudflare.answer = _respond(403, text="<html>403 Forbidden by edge rule 77</html>")
    assert _dispatch() == out


def test_graphql_errors_answer_the_same_sentence_with_status_200(cloudflare, configured):
    cloudflare.answer = _respond(403, json=FORBIDDEN_BODY)
    forbidden = _dispatch()["error"]

    cloudflare.answer = _respond(200, json=GRAPHQL_ERRORS_BODY)
    out = _dispatch()

    assert out == {"error": forbidden.replace("403", "200")}
    assert "not authorized" not in out["error"]


# (e) ---------------------------------------------------------------------

# Every answer below carries the token in its body or exception text, so a
# path that echoed a response or an exception would leak it.
TOKEN_CARRYING_ANSWERS = {
    "success": _respond(200, json=_recorded_answer()),
    "status-403": _respond(403, text=f"denied: Authorization: Bearer {TOKEN}"),
    "graphql-errors": _respond(
        200, json={"data": None, "errors": [{"message": f"bad token {TOKEN}"}]}
    ),
    "unreadable-200": _respond(200, text=f"<html>proxy saw Bearer {TOKEN}</html>"),
    "timeout": _raise(httpx.ReadTimeout, f"read timed out, request had Bearer {TOKEN}"),
    "connection-error": _raise(httpx.ConnectError, f"connect failed, request had Bearer {TOKEN}"),
}


@pytest.mark.parametrize("path", list(TOKEN_CARRYING_ANSWERS))
def test_log_capture_and_answer_never_hold_the_token(cloudflare, configured, caplog, path):
    cloudflare.answer = TOKEN_CARRYING_ANSWERS[path]

    with caplog.at_level(logging.DEBUG):
        text = _dispatch_text()

    assert len(cloudflare.requests) == 1  # one call, no retry
    assert TOKEN not in text
    assert TOKEN not in caplog.text
    if path != "success":
        assert set(json.loads(text)) == {"error"}


def test_refusal_logs_never_hold_the_token(cloudflare, configured, caplog):
    configured.chmod(0o644)

    with caplog.at_level(logging.DEBUG):
        text = _dispatch_text()

    _assert_not_set_up(json.loads(text), cloudflare)
    assert TOKEN not in caplog.text


# (f) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-10-01T00:00:00Z", "2026-11-01T00:00:01Z"),
        ("2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        ("2026-11-01T00:00:00Z", "2026-10-25T00:00:00Z"),
        ("2026-10-25T00:00:00Z", "2026-10-25T00:00:00Z"),
        ("2026-10-25", "2026-11-01"),
        ("2026-10-25T00:00:00", "2026-11-01T00:00:00"),
        ("2026-10-25T00:00:00+02:00", "2026-11-01T00:00:00+02:00"),
        ("last week", "today"),
        (None, END),
    ],
    ids=[
        "31-days-and-1-second",
        "two-months",
        "end-before-start",
        "end-equals-start",
        "dates-without-time",
        "no-utc-designator",
        "not-utc",
        "not-iso-8601",
        "missing-start",
    ],
)
def test_bad_window_is_refused_before_any_call(cloudflare, configured, start, end):
    out = _dispatch(start, end)

    assert set(out) == {"error"}
    assert cloudflare.requests == []


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-10-01T00:00:00Z", "2026-11-01T00:00:00Z"),
        ("2026-10-25T00:00:00+00:00", "2026-11-01T00:00:00+00:00"),
    ],
    ids=["exactly-31-days", "utc-as-offset"],
)
def test_window_up_to_31_days_is_queried(cloudflare, configured, start, end):
    out = _dispatch(start, end)

    assert out["visits"] == RECORDED_TOTAL
    assert len(cloudflare.requests) == 1


# (g) ---------------------------------------------------------------------


def test_website_visits_toolset_is_in_no_other_toolset():
    import tools.website_visits_tool  # noqa: F401 - ensure registered
    from toolsets import TOOLSETS, resolve_toolset

    assert resolve_toolset("website_visits") == ["website_visits"]
    for name, toolset in TOOLSETS.items():
        if name == "website_visits":
            continue
        assert "website_visits" not in toolset.get("includes", []), name
        assert "website_visits" not in resolve_toolset(name), name
