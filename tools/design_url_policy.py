"""Address policy for the look-only design browser (design-loop plan v2, D1).

:func:`design_url_refusal` says whether the design browser may load a URL. ``is_safe_url`` and
``is_always_blocked_url`` from :mod:`tools.url_safety` run first; the design check then allows only
https on the default port, no user info, a host the ``design_browser.allowed_hosts`` setting
permits, and public DNS answers, even with ``security.allow_private_urls`` on. The Cloudflare Web
Analytics visitor counter stays blocked even when listed, so design visits never reach the site's
visit counts. An empty or malformed setting allows nothing.

The caller passes the allow-list or the loaded config, so importing this module loads no config and
touches no network.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from tools.url_safety import (
    _getaddrinfo,
    _iter_resolved_ips,
    _resolved_ip_block_reason,
    is_always_blocked_url,
    is_safe_url,
    sensitive_query_param_name,
)

# The visitor counter: its script and beacon hosts (and every host below them), and the path it
# posts to on the site's own host.
_COUNTER_HOSTS = ("static.cloudflareinsights.com", "cloudflareinsights.com")
_COUNTER_PATH_PREFIX = "/cdn-cgi/rum"
_HOST_LABEL = re.compile(r"[a-z0-9-]+")
# A server undoes a path's percent escapes once, a proxy in front of it perhaps once more. Each pass
# rescans the whole path and every two characters can nest one more layer, so decoding is capped at
# this many passes rather than run until nothing changes.
_MAX_DECODE_PASSES = 4


def _ascii_host(host: str) -> str | None:
    """``host`` in punycode form (ASCII, lowercase), or None unless it is a clean DNS host name.

    Clean means lowercasing alone already gives the punycode form or its Unicode form. Python's
    IDNA codec quietly maps fullwidth dots and letters, ``ß`` and zero-width joiners onto other
    hosts, and browsers do not always map them the same way, so such hosts are refused, as is
    punycode that does not round-trip, an IP literal, and any character but letters, digits and
    hyphens."""
    host = host.lower()
    try:
        ascii_host = host.encode("idna").decode("ascii")
        unicode_host = ascii_host.encode("ascii").decode("idna")
    except UnicodeError:
        return None
    labels = ascii_host.split(".")
    if host not in (ascii_host, unicode_host) or not all(map(_HOST_LABEL.fullmatch, labels)):
        return None
    # Browsers read a host whose last label is numeric as an IPv4 address.
    return ascii_host if labels[-1][0].isalpha() else None


def _path_refusal(path: str) -> str | None:
    """Why ``path`` must not be loaded, or None. It must not when it reaches the counter prefix as
    a server might read it: percent escapes undone, backslashes and repeated slashes as one ``/``,
    dot segments resolved, in any case. Nor when escapes remain after ``_MAX_DECODE_PASSES``
    passes, because decoding stops there and where such a path leads goes unchecked."""
    for _ in range(_MAX_DECODE_PASSES + 1):
        if (decoded := unquote(path)) == path:
            break
        path = decoded
    else:
        return "path has too many layers of percent escapes"
    path = posixpath.normpath(re.sub(r"[/\\]+", "/", "/" + path)).lower()
    if path.startswith(_COUNTER_PATH_PREFIX):
        return "the visitor counter is always blocked"
    return None


def _allowed_hosts(policy: Any) -> list[tuple[str, bool]]:
    """``(host, is_wildcard)`` for each usable allow-list entry. A missing ``design_browser``
    section, a missing or non-list ``allowed_hosts`` and entries that are not clean host names add
    nothing, so a broken setting allows nothing."""
    if isinstance(policy, Mapping):
        section = policy.get("design_browser")
        policy = section.get("allowed_hosts") if isinstance(section, Mapping) else None
    if not isinstance(policy, list):
        return []
    allowed = []
    for entry in policy:
        if isinstance(entry, str):
            wildcard = entry.startswith("*.")
            host = _ascii_host(entry[2:] if wildcard else entry)
            if host is not None:
                allowed.append((host, wildcard))
    return allowed


def _permits(allowed: list[tuple[str, bool]], host: str) -> bool:
    """A plain entry matches exactly that host; ``*.<base>`` matches exactly one more label below
    ``<base>``, never ``<base>`` itself and never two or more labels."""
    parent = host.partition(".")[2]
    return any(parent == base if wildcard else host == base for base, wildcard in allowed)


def _resolution_refusal(host: str) -> str | None:
    """Resolve ``host`` again and refuse unless every answer is public. ``is_safe_url`` lets private
    answers through with ``security.allow_private_urls`` on, and a name it cannot resolve through
    when a proxy is set; the design browser takes neither."""
    try:
        answers = list(_iter_resolved_ips(_getaddrinfo(host)))
    except OSError:
        return "host does not resolve"
    if not answers:
        return "host does not resolve"
    for _raw, _ip_str, ip in answers:
        if ip is None:
            return "host resolves to an unparseable address"
        reason = _resolved_ip_block_reason(ip, allow_private=False)
        if reason is not None:
            return f"host resolves to a {reason}"
    return None


def design_url_refusal(
    url: str,
    policy: Mapping[str, Any] | list[str] | None,
    *,
    builder_host: str | None = None,
) -> str | None:
    """Why the design browser must not load ``url``, or None when it may.

    ``policy`` is the ``design_browser.allowed_hosts`` list or the loaded config that holds it. A
    plain entry is one exact host; ``*.<host>`` is exactly one more label below ``<host>``, never
    ``<host>`` itself, so the apex is listed on its own. ``builder_host`` narrows the policy to that
    one host if the allow-list permits it, and to nothing if not. The reason is short and never
    echoes a query value."""
    if not isinstance(url, str) or not is_safe_url(url) or is_always_blocked_url(url):
        return "blocked by the URL safety check"
    # Python's parser and browsers treat these differently (a browser reads a backslash as "/").
    if any(ch <= " " or ch in "\\\x7f" for ch in url):
        return "not a clean URL"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "not a clean URL"
    if parts.scheme != "https":
        return "only https is allowed"
    if "@" in parts.netloc:
        return "user info is not allowed"
    host_text, colon, port = parts.netloc.partition(":")
    host = _ascii_host(host_text)
    if host is None:
        return "not a clean host name"
    if colon and port != "443":
        return "only the default https port is allowed"
    if any(host == counter or host.endswith("." + counter) for counter in _COUNTER_HOSTS):
        return "the visitor counter is always blocked"
    if (refusal := _path_refusal(parts.path)) is not None:
        return refusal
    allowed = _allowed_hosts(policy)
    if builder_host is not None:
        own = _ascii_host(builder_host) if isinstance(builder_host, str) else None
        allowed = [(own, False)] if own is not None and _permits(allowed, own) else []
    if not _permits(allowed, host):
        return "host is not on the design allow-list"
    secret = sensitive_query_param_name(url)
    if secret:
        return f"secret-named query parameter: {secret}"
    return _resolution_refusal(host)
