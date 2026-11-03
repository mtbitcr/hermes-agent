"""Website visits from Cloudflare Web Analytics, with the LinkedIn share.

Read-only. One GraphQL POST per call counts the visits in a UTC window and the
visits whose referrer host is LinkedIn. The profile's config.yaml names the
account, the site and a mode-0600 token file (``website_visits.account_tag``,
``site_tag``, ``token_file``). The token goes only into the Authorization
header: answers and log lines carry fixed sentences, never a response body or
an exception text.
"""

import json
import logging
import os
import re
import stat
from datetime import datetime, timedelta, timezone

import httpx

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.cloudflare.com/client/v4/graphql"
TIMEOUT_SECONDS = 20.0
MAX_WINDOW = timedelta(days=31)
MAX_HOSTS = 20

NOT_SET_UP = "Website visits are not set up yet."
UNREACHABLE = "Cloudflare Web Analytics could not be reached."
FLOOR_NOTE = (
    "These counts are a floor: ad blockers and the LinkedIn app can send no "
    "referrer, so real visits and LinkedIn visits can be higher."
)

# RFC 6750 b64token, the only characters a Bearer token may hold. Anything else
# (a pasted "Bearer " prefix, a second line) is refused rather than sent, so
# the token never reaches an HTTP library's error text.
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9\-._~+/]+=*")

# The one form start and end may take: YYYY-MM-DDTHH:MM:SSZ, whole seconds in
# UTC and nothing else. Every other form is refused rather than read, so the
# window checked and sent is exactly the window asked for.
_UTC_DATE_TIME = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z")

# One request, two aliased nodes over the same filter: the total, and the
# visits per referrer host.
_QUERY = """query {
  viewer {
    accounts(filter: {accountTag: %(account_tag)s}) {
      total: rumPageloadEventsAdaptiveGroups(limit: 1, filter: %(window)s) {
        sum { visits }
      }
      topReferers: rumPageloadEventsAdaptiveGroups(limit: 100, orderBy: [sum_visits_DESC], filter: %(window)s) {
        sum { visits }
        dimensions { refererHost }
      }
    }
  }
}"""


def _utc(value) -> datetime | None:
    """Parse a date-time in exactly the form YYYY-MM-DDTHH:MM:SSZ; None for any
    other form and for a date or time that does not exist."""
    match = _UTC_DATE_TIME.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    try:
        return datetime(*map(int, match.groups()), tzinfo=timezone.utc)
    except ValueError:
        return None


def _rfc3339(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _settings() -> dict | None:
    """The three website_visits settings, or None when any is missing."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("website_visits")
    except Exception:
        return None
    if not isinstance(section, dict):
        return None
    settings = {}
    for key in ("account_tag", "site_tag", "token_file"):
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        settings[key] = value.strip()
    return settings


def _read_token(token_file: str) -> str | None:
    """The trimmed token, or None for a relative path, a missing, unreadable or
    non-regular file, a mode other than 0600, or content that is not one token."""
    if not os.path.isabs(token_file):
        return None
    try:
        info = os.stat(token_file)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            return None
        with open(token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    except (OSError, ValueError):
        return None
    return token if _BEARER_TOKEN.fullmatch(token) else None


def _visits(row) -> int | None:
    """A row's sum.visits when it is a JSON integer of 0 or more; None otherwise."""
    group_sum = row.get("sum") if isinstance(row, dict) else None
    visits = group_sum.get("visits") if isinstance(group_sum, dict) else None
    # bool is a subclass of int: a JSON true must not count as one visit.
    if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
        return None
    return visits


def _referer_host(row) -> str | None:
    """A topReferers row's dimensions.refererHost, with null as ""; None when
    the row holds no such string or null."""
    dimensions = row.get("dimensions") if isinstance(row, dict) else None
    if not isinstance(dimensions, dict) or "refererHost" not in dimensions:
        return None
    host = dimensions["refererHost"]
    if host is None:
        return ""
    return host if isinstance(host, str) else None


def _counts(response: httpx.Response) -> tuple[int, list[dict]] | None:
    """(total visits, visits per referrer host) from a 200 answer in the one
    accepted shape; None for any other answer.

    The shape: errors absent or null; data.viewer.accounts a list of exactly
    one object, which holds the lists total (zero or one row) and topReferers.
    Every row has sum.visits, a JSON integer of 0 or more, and every
    topReferers row has dimensions.refererHost, a string or null.
    """
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except (ValueError, RecursionError):  # not JSON, or nested too deep to decode
        return None
    if not isinstance(payload, dict) or payload.get("errors") is not None:
        return None
    data = payload.get("data")
    viewer = data.get("viewer") if isinstance(data, dict) else None
    accounts = viewer.get("accounts") if isinstance(viewer, dict) else None
    if not isinstance(accounts, list) or len(accounts) != 1 or not isinstance(accounts[0], dict):
        return None
    total, referers = accounts[0].get("total"), accounts[0].get("topReferers")
    if not isinstance(total, list) or len(total) > 1 or not isinstance(referers, list):
        return None
    visits = _visits(total[0]) if total else 0  # no total row: no visits
    if visits is None:
        return None
    hosts = []
    for row in referers:
        host, host_visits = _referer_host(row), _visits(row)
        if host is None or host_visits is None:
            return None
        hosts.append({"host": host, "visits": host_visits})
    return visits, hosts


def _is_linkedin(host: str) -> bool:
    host = host.casefold()
    return "linkedin" in host or host == "lnkd.in"


def website_visits(start, end) -> str:
    """Visits and LinkedIn visits from ``start`` (inclusive) to ``end`` (exclusive)."""
    begin, finish = _utc(start), _utc(end)
    if begin is None or finish is None:
        return tool_error(
            "start and end must be UTC date-times in exactly the form YYYY-MM-DDTHH:MM:SSZ, "
            "such as 2026-10-25T00:00:00Z."
        )
    if finish <= begin:
        return tool_error("end must be after start.")
    if finish - begin > MAX_WINDOW:
        return tool_error("end must be at most 31 days after start.")

    settings = _settings()
    token = _read_token(settings["token_file"]) if settings else None
    if token is None:
        return tool_error(NOT_SET_UP)

    start, end = _rfc3339(begin), _rfc3339(finish)
    window = "{siteTag: %s, datetime_geq: %s, datetime_lt: %s}" % (
        json.dumps(settings["site_tag"]), json.dumps(start), json.dumps(end),
    )
    query = _QUERY % {"account_tag": json.dumps(settings["account_tag"]), "window": window}
    try:
        response = httpx.post(
            GRAPHQL_URL,
            json={"query": query},
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        # The class name only: an exception's text can carry request details.
        logger.warning("website_visits: Cloudflare request failed (%s)", type(exc).__name__)
        return tool_error(UNREACHABLE)

    counts = _counts(response)
    if counts is None:
        logger.warning(
            "website_visits: Cloudflare answered HTTP %s without visit counts",
            response.status_code,
        )
        return tool_error(
            f"Cloudflare did not return visit counts (HTTP status {response.status_code})."
        )
    total, hosts = counts
    hosts.sort(key=lambda row: row["visits"], reverse=True)
    return tool_result(
        start=start,
        end=end,
        visits=total,
        linkedin_visits=sum(row["visits"] for row in hosts if _is_linkedin(row["host"])),
        hosts=hosts[:MAX_HOSTS],
        note=FLOOR_NOTE,
    )


WEBSITE_VISITS_SCHEMA = {
    "name": "website_visits",
    "description": (
        "Count website visits in a UTC time window from Cloudflare Web Analytics, and how "
        "many came from LinkedIn by referrer host. Read-only. The window runs from start "
        "(inclusive) to end (exclusive) and spans at most 31 days. Returns the total, the "
        "LinkedIn visits, the top 20 referrer hosts by visits, and a note that the counts "
        "are a floor."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {
                "type": "string",
                "description": (
                    "Window start, a UTC date-time in exactly the form YYYY-MM-DDTHH:MM:SSZ, "
                    "such as 2026-10-25T00:00:00Z."
                ),
            },
            "end": {
                "type": "string",
                "description": (
                    "Window end, a UTC date-time in exactly the form YYYY-MM-DDTHH:MM:SSZ, "
                    "after start and at most 31 days later, such as 2026-11-01T00:00:00Z."
                ),
            },
        },
        "required": ["start", "end"],
    },
}


registry.register(
    name="website_visits",
    toolset="website_visits",
    schema=WEBSITE_VISITS_SCHEMA,
    handler=lambda args, **kw: website_visits(args.get("start"), args.get("end")),
    emoji="📈",
)
