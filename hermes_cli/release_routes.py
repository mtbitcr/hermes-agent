"""Release routes: what a release changes in the API route list of each role served here.

gateway/platforms/api_server_route_profiles.json, the route file, names the roles each API route
the gateway registers is meant for. When a release changes that file, each role the gateway serves
on this host gains the exact entries of its new routes, and loses the exact entries of routes that
no longer exist (plan section 2.2). plan_routes makes that plan from the route file of PREV and of
NEW and the config.yaml of each served role; apply_routes rewrites one config.yaml by one role's
part of it, touching only the entry lines of gateway.api_server.allowed_routes. A route list is
read exactly as the gateway reads it: the same reading of the text, then the gateway's own
_classify_allowed_routes. plan_routes applies each change it plans to the same config text, so a
plan holds only changes that apply.

The rewrite reads the text by lines and edits one written form, the plain form of the owner facts
for this card (point 10, and condition i of their addition): a UTF-8 text with no carriage return
that ends with a line feed; one line `gateway:`, one direct child line `api_server:` of it and one
direct child line `allowed_routes:` of that, each with nothing after the colon; right after it, one
entry line or more ("- " and the entry, all at one indent, not less deep than the key), whose
entries hold only the ASCII letters and digits, spaces and / _ . { } * -; after the last of them,
no line that belongs to the list; the gateway reading the same entries from the text; and the
events of the parser the gateway reads with agreeing with these lines: one document, each key of
the path exactly once in its block mapping, plain and on its line, a block sequence with one plain
item on each entry line, and no anchor or tag on any of them. Each other text is refused before any
edit, with the first condition it breaks. The edit takes out the lines of the removed entries and
puts a line for each added entry after the last entry line. The result is checked: every byte
outside the entry lines is the input's, and the gateway reads the planned list from it.

Both functions only compute. Their result depends only on their arguments; they read no file and
no environment, make no network or process call and change no global state, and everything they
need is imported when this module loads (K6). Nothing calls this module yet; the release steps
that use the plan belong to later cards.
"""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from gateway.platforms.api_server import (
    _CONFIG_ROOT_ABSENT,
    _ROUTES_ALLOWLIST,
    _ROUTES_DENY_ALL,
    _normalize_api_route_rule,
    _route_matches_any,
)
from gateway.platforms.api_server_route_list import _classify_allowed_routes
from utils import _fast_yaml_loader, fast_safe_load

# Why an entry is left as it is, as the owner line shows it.
NOT_SERVED_HERE = "not served here"
NOT_LIMITED = "this role's list does not limit routes"
ALREADY_ADMITTED = "already admitted"
KEPT_LAST_ENTRY = "kept: the last entry of the list"
KEPT_ROLE_DROPPED = "kept: the route no longer names this role"

# Condition f of the plain form: the first character of an entry, and every character of one.
_ENTRY_START = frozenset(string.ascii_letters + "/")
_ENTRY_CHARACTERS = frozenset(string.ascii_letters + string.digits + " /_.{}*-")


class RoutePlanError(Exception):
    """The route plan cannot be made or applied; `reason` says why in plain words.

    The reason quotes no config text and keeps nothing of an underlying error, since a
    config.yaml may hold secrets.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RouteChange:
    """One role's part of the plan: the exact entries its route list gains and loses."""

    role: str
    add: tuple[str, ...]
    remove: tuple[str, ...]


@dataclass(frozen=True)
class RouteNote:
    """A route entry the plan leaves as it is, and why."""

    role: str
    entry: str
    reason: str


@dataclass(frozen=True)
class RoutePlan:
    """One change for each role whose list changes, and the notes, both in role order."""

    changes: tuple[RouteChange, ...]
    notes: tuple[RouteNote, ...]

    def added(self) -> tuple[tuple[str, str], ...]:
        """Every (role, entry) the plan adds."""
        return tuple((change.role, entry) for change in self.changes for entry in change.add)


def _is_exact_route(entry: object) -> bool:
    # An exact route rule written as the gateway writes one, "METHOD /template"; never a path
    # prefix, which may cover other routes.
    return (
        isinstance(entry, str)
        and not entry.startswith("/")
        and _normalize_api_route_rule(entry) == entry
    )


def _route_file(text: str, which: str) -> dict[str, frozenset[str]]:
    # The route file is one JSON object: route -> the roles it is meant for. A key that is not an
    # exact route rule could only be written as a different or a prefix entry, so it refuses.
    try:
        routes = json.loads(text)
    except ValueError:
        raise RoutePlanError(f"{which}'s route file does not parse") from None
    if not isinstance(routes, dict) or not all(
        isinstance(roles, list) and all(isinstance(role, str) for role in roles)
        for roles in routes.values()
    ):
        raise RoutePlanError(f"{which}'s route file does not parse")
    for route in routes:
        if not _is_exact_route(route):
            raise RoutePlanError(
                f"{which}'s route file has a key that does not normalise: {route!r}"
            )
    return {route: frozenset(roles) for route, roles in routes.items()}


def _decoded(text: bytes) -> str:
    # As the gateway's read_text reads config.yaml: strict UTF-8, with universal newlines.
    return text.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _root(text: bytes) -> Any:
    # The config.yaml root as the gateway's _read_raw_config_root reads it from the file, which it
    # cannot be given as text: a text with no YAML content (empty, blank or only comments) is no
    # config at all.
    decoded = _decoded(text)
    if not any(
        line.strip() and not line.lstrip().startswith("#") for line in decoded.splitlines()
    ):
        return _CONFIG_ROOT_ABSENT
    return fast_safe_load(decoded)


def _route_list(text: bytes | None) -> tuple[str, list[str]]:
    # The (mode, patterns) the gateway's _resolve_api_server_allowed_routes gives for a
    # config.yaml holding `text`; None is no config.yaml. A text that cannot be read or parsed
    # denies all.
    if text is None:
        return _classify_allowed_routes(_CONFIG_ROOT_ABSENT)
    try:
        root = _root(text)
    except Exception:
        return (_ROUTES_DENY_ALL, [])
    return _classify_allowed_routes(root)


def _admitted(route: str, patterns: list[str]) -> bool:
    # Whether a list in allowlist mode already admits every request the gateway routes to `route`,
    # as its _route_matches_any checks a real request: an exact rule by method and template, a
    # path prefix by the request's path. Every path of a template starts with the template's
    # complete segments before its first placeholder, so a prefix of those covers the route; a
    # prefix holding placeholder text ("/v1/runs/{run_id}") matches no real path.
    method, template = route.split(" ", 1)
    exact = [pattern for pattern in patterns if not pattern.startswith("/")]
    if _route_matches_any(template, exact, method=method, route_template=template):
        return True
    literal = template.split("{", 1)[0].rsplit("/", 1)[0] if "{" in template else template
    prefixes = [pattern for pattern in patterns if pattern.startswith("/")]
    return bool(literal) and _route_matches_any(literal, prefixes)


def plan_routes(
    prev_file: str | None, new_file: str, configs: Mapping[str, bytes | None]
) -> RoutePlan:
    """Plan what this release changes in the route list of each role the gateway serves here.

    `prev_file` and `new_file` are the route file's text at PREV and at NEW; `prev_file` is None
    when PREV has no route file, and then nothing is added and nothing is removed. `configs` is
    the config.yaml text of every role the gateway serves on this host, keyed by role ("default"
    for the root), None for a role with no config.yaml. The rules are plan section 2.2's:

    - an added pair is a (route, role) pair in NEW's file and not in PREV's. It is skipped when
      the role is not served here, when the role's list does not limit routes, or when the list
      already admits every request to the route, as the gateway's _route_matches_any matches a
      real request: by its exact entry, or by an old path prefix of the route's complete
      segments before its first placeholder (a prefix holding placeholder text matches no real
      path); otherwise its exact entry is added. A list that is not valid (deny-all) cannot take
      the entry, so it refuses;
    - a removed route is a key of PREV's file that NEW's file no longer has. Its exact entry
      leaves every served list in allowlist mode that holds it, unless it is the list's last
      entry, since an empty list means "not limited". A path prefix is never removed;
    - a route that lives on but no longer names a role changes nothing in that role's list,
      since narrowing a role is a person's decision; a note names it for every role served
      here, whatever its list's mode.

    Raises RoutePlanError for a deny-all list the plan must change, a route file that does not
    parse, or a route file key that is not an exact route rule as the gateway writes one.
    """
    new = _route_file(new_file, "NEW")
    if prev_file is None:
        return RoutePlan((), ())
    prev = _route_file(prev_file, "PREV")
    removed = sorted(set(prev) - set(new))
    living = sorted(set(prev) & set(new))
    lists = {role: _route_list(text) for role, text in configs.items()}
    named = {role for roles in new.values() for role in roles}
    changes: list[RouteChange] = []
    notes: list[RouteNote] = []
    for role in sorted(set(lists) | named):
        added = sorted(
            route for route, roles in new.items() if role in roles - prev.get(route, frozenset())
        )
        if role not in lists:
            notes += [RouteNote(role, route, NOT_SERVED_HERE) for route in added]
            continue
        # A route that lives on but no longer names this role: whatever the list holds for it
        # stays, in every mode, and the owner line names it.
        dropped = [
            RouteNote(role, route, KEPT_ROLE_DROPPED)
            for route in living
            if role in prev[route] - new[route]
        ]
        mode, patterns = lists[role]
        if mode == _ROUTES_DENY_ALL:
            if added:
                raise RoutePlanError(f"role {role} has a route list that is not valid")
            notes += dropped
            continue
        if mode != _ROUTES_ALLOWLIST:
            notes += [RouteNote(role, route, NOT_LIMITED) for route in added]
            notes += dropped
            continue
        add: list[str] = []
        for route in added:
            if _admitted(route, patterns):
                notes.append(RouteNote(role, route, ALREADY_ADMITTED))
            else:
                add.append(route)
        # The entries the list holds once this change is applied, entries it adds included.
        remaining = [*patterns, *add]
        remove: list[str] = []
        for route in removed:
            if route not in patterns:
                continue
            if all(entry == route for entry in remaining):
                notes.append(RouteNote(role, route, KEPT_LAST_ENTRY))
                continue
            remove.append(route)
            remaining = [entry for entry in remaining if entry != route]
        notes += dropped
        if add or remove:
            change = RouteChange(role, tuple(add), tuple(remove))
            # A dry run on the same config text: the plan holds only changes that apply.
            try:
                apply_routes(configs[role], change)
            except RoutePlanError as error:
                raise RoutePlanError(
                    f"the change for role {role} does not apply to its config: {error.reason}"
                ) from None
            changes.append(change)
    return RoutePlan(tuple(changes), tuple(notes))


@dataclass(frozen=True)
class _PlainList:
    # The route list of a text in the plain form: its entry lines are the bytes text[start:end],
    # each at `indent`, and they hold `entries` in order, as written.
    start: int
    end: int
    indent: int
    entries: tuple[str, ...]


def _not_plain(role: str, condition: str, words: str) -> RoutePlanError:
    return RoutePlanError(
        f"the config of role {role} is not in the plain form: "
        f"condition {condition} does not hold: {words}"
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    # Neither a blank line (spaces and tabs at most) nor a comment line ("#" after the indent).
    return bool(line.strip(" \t")) and not line.lstrip(" ").startswith("#")


def _children(lines: list[str], key: int) -> list[int]:
    # The direct child lines of the key on line `key` (owner facts point 11): the content lines of
    # its block, which ends at the next content line not deeper than the key line, that have the
    # indent of the first of them.
    block: list[int] = []
    for index in range(key + 1, len(lines)):
        if not _is_content(lines[index]):
            continue
        if _indent(lines[index]) <= _indent(lines[key]):
            break
        block.append(index)
    return [index for index in block if _indent(lines[index]) == _indent(lines[block[0]])]


def _how_many(found: list[int]) -> str:
    return "no" if not found else "more than one"


def _entry_fault(entry: str) -> str | None:
    # Which part of condition f `entry` breaks, or None when it is an entry of the plain form.
    if not entry or entry[0] not in _ENTRY_START:
        return "does not start with a letter or a slash"
    if not set(entry) <= _ENTRY_CHARACTERS:
        return "holds a character other than letters, digits, spaces and / _ . { } * -"
    if entry.endswith(" "):
        return "ends with a space"
    return None


def _is_dash_line(line: str) -> bool:
    # After the indent, a dash followed by a space or by the end of the line (owner addition
    # point 11).
    content = line.lstrip(" ")
    return content == "-" or content.startswith("- ")


# Condition i: the keys of the path, each with the mapping that holds it.
_PATH = (
    ("gateway", "in the root mapping"),
    ("api_server", "under 'gateway'"),
    ("allowed_routes", "under 'api_server'"),
)


def _line(text: str, event: yaml.Event) -> int:
    # The line, counted from 0 as the lines of the plain form are, that `event` starts on. The
    # parser also ends a line at a next-line, line or paragraph separator; these lines do not.
    return text.count("\n", 0, event.start_mark.index)


def _node_end(events: list[yaml.Event], index: int) -> int:
    # The index of the first event after the node that starts at events[index].
    depth = 0
    while True:
        if isinstance(events[index], yaml.CollectionStartEvent):
            depth += 1
        elif isinstance(events[index], yaml.CollectionEndEvent):
            depth -= 1
        index += 1
        if depth == 0:
            return index


def _nodes(events: list[yaml.Event], start: int) -> list[int]:
    # The indexes of the nodes in the collection that starts at events[start], in order; in a
    # mapping, each key is followed by its value.
    nodes = []
    index = start + 1
    while not isinstance(events[index], yaml.CollectionEndEvent):
        nodes.append(index)
        index = _node_end(events, index)
    return nodes


def _key_value(events: list[yaml.Event], index: int) -> str | None:
    # The value of the key at events[index] when the key is a scalar or an alias of one.
    event = events[index]
    if isinstance(event, yaml.AliasEvent):
        event = next(
            (
                earlier
                for earlier in reversed(events[:index])
                if isinstance(earlier, (yaml.ScalarEvent, yaml.CollectionStartEvent))
                and earlier.anchor == event.anchor
            ),
            None,
        )
    return event.value if isinstance(event, yaml.ScalarEvent) else None


def _properties_fault(event: yaml.NodeEvent, name: str) -> str | None:
    if event.anchor is not None:
        return f"the parser shows an anchor &{event.anchor} on {name}"
    if event.tag is not None:
        return f"the parser shows a tag {event.tag!r} on {name}"
    return None


def _collection_fault(event: yaml.Event, kind: type, name: str) -> str | None:
    # Whether the node `event` starts is a block collection of `kind` with no anchor and no tag;
    # a block collection is one that is not in flow style.
    if not isinstance(event, kind) or event.flow_style:
        shape = "mapping" if kind is yaml.MappingStartEvent else "sequence"
        return f"the parser shows that {name} is not a block {shape}"
    return _properties_fault(event, name)


def _parser_fault(decoded: str, key_lines: list[int], entry_lines: range) -> str | None:
    # What the events of the parser show against the lines of the plain form (condition i of the
    # owner addition, point 4), or None when they agree. `key_lines` are the lines of conditions b,
    # c and d, `entry_lines` the entry lines. The parser is the gateway's own; it skips a byte
    # order mark and counts no position for it, so the text is read without one.
    text = decoded.removeprefix("\ufeff")
    try:
        events = list(yaml.parse(text, Loader=_fast_yaml_loader))
    except yaml.YAMLError:
        return "the parser cannot read the text"
    documents = sum(isinstance(event, yaml.DocumentStartEvent) for event in events)
    if documents != 1:
        return f"the parser shows {documents} documents"
    node, name = 2, "the root"
    for (key, where), key_line in zip(_PATH, key_lines):
        fault = _collection_fault(events[node], yaml.MappingStartEvent, name)
        if fault is not None:
            return fault
        found = [index for index in _nodes(events, node)[::2] if _key_value(events, index) == key]
        if len(found) != 1:
            return f"the parser shows {'a second' if found else 'no'} key {key!r} {where}"
        event = events[found[0]]
        if not isinstance(event, yaml.ScalarEvent) or event.style:
            return f"the parser shows that the key {key!r} is not plain"
        fault = _properties_fault(event, f"the key {key!r}")
        if fault is not None:
            return fault
        if _line(text, event) != key_line:
            return (
                f"the parser shows the key {key!r} on line {_line(text, event) + 1}, not on "
                f"line {key_line + 1}"
            )
        node, name = _node_end(events, found[0]), f"the value of {key!r}"
    fault = _collection_fault(events[node], yaml.SequenceStartEvent, name)
    if fault is not None:
        return fault
    items = _nodes(events, node)
    item_lines = [_line(text, events[index]) for index in items]
    for line in item_lines:
        if line not in entry_lines:
            return f"the parser shows an item on line {line + 1}, which is not an entry line"
    for line in entry_lines:
        if item_lines.count(line) != 1:
            return (
                f"the parser shows {'a second' if line in item_lines else 'no'} item on the "
                f"entry line {line + 1}"
            )
    for index, line in zip(items, item_lines):
        event = events[index]
        if not isinstance(event, yaml.ScalarEvent) or event.style:
            return f"the parser shows that the item on line {line + 1} is not plain"
        fault = _properties_fault(event, f"the item on line {line + 1}")
        if fault is not None:
            return fault
    return None


def _plain_list(text: bytes, role: str) -> _PlainList:
    # The route list of `text` when the text is in the plain form (owner facts point 10 and
    # condition i of their addition); otherwise RoutePlanError, naming the first condition that
    # does not hold.
    try:
        decoded = text.decode("utf-8")
    except UnicodeDecodeError:
        raise _not_plain(role, "a", "the text is not UTF-8") from None
    if "\r" in decoded:
        raise _not_plain(role, "a", "the text holds a carriage return")
    if not decoded.endswith("\n"):
        raise _not_plain(role, "a", "the text does not end with a line feed")
    lines = decoded.split("\n")[:-1]
    found = [index for index, line in enumerate(lines) if line == "gateway:"]
    if len(found) != 1:
        raise _not_plain(
            role,
            "b",
            f"{_how_many(found)} line of the text is 'gateway:' with no indent and nothing "
            "after the colon",
        )
    path = [found[0]]
    found = [
        index for index in _children(lines, found[0]) if lines[index].lstrip(" ") == "api_server:"
    ]
    if len(found) != 1:
        raise _not_plain(
            role,
            "c",
            f"{_how_many(found)} direct child line of 'gateway:' is 'api_server:' with nothing "
            "after the colon",
        )
    path.append(found[0])
    found = [
        index
        for index in _children(lines, found[0])
        if lines[index].lstrip(" ") == "allowed_routes:"
    ]
    if len(found) != 1:
        raise _not_plain(
            role,
            "d",
            f"{_how_many(found)} direct child line of 'api_server:' is 'allowed_routes:' with "
            "nothing after the colon",
        )
    key = found[0]
    path.append(key)
    first = end = key + 1
    while end < len(lines) and lines[end].lstrip(" ").startswith("- "):
        end += 1
    if end == first:
        raise _not_plain(role, "e", "the line after 'allowed_routes:' is not an entry line")
    indents = {_indent(lines[index]) for index in range(first, end)}
    if len(indents) > 1:
        raise _not_plain(role, "e", "the entry lines do not all have the same indent")
    indent = indents.pop()
    if indent < _indent(lines[key]):
        raise _not_plain(role, "e", "the entry lines are less deep than 'allowed_routes:'")
    entries = tuple(lines[index][indent + 2 :] for index in range(first, end))
    for index, entry in enumerate(entries, start=first):
        fault = _entry_fault(entry)
        if fault is not None:
            raise _not_plain(role, "f", f"the entry on line {index + 1} {fault}")
    after = next((index for index in range(end, len(lines)) if _is_content(lines[index])), None)
    if after is not None and _indent(lines[after]) > _indent(lines[key]):
        raise _not_plain(
            role,
            "g",
            f"line {after + 1}, after the last entry line, is deeper than 'allowed_routes:'",
        )
    if after is not None and _indent(lines[after]) == indent and _is_dash_line(lines[after]):
        raise _not_plain(
            role,
            "g",
            f"line {after + 1}, after the last entry line, is a dash line at the indent of the "
            "entries",
        )
    if _route_list(text) != (
        _ROUTES_ALLOWLIST,
        [_normalize_api_route_rule(entry) for entry in entries],
    ):
        raise _not_plain(role, "h", "the gateway reads another list than the entry lines")
    fault = _parser_fault(decoded, path, range(first, end))
    if fault is not None:
        raise _not_plain(role, "i", fault)
    raw = text.split(b"\n")
    start = sum(len(line) + 1 for line in raw[:first])
    return _PlainList(start, start + sum(len(line) + 1 for line in raw[first:end]), indent, entries)


def _edit(text: bytes, plain: _PlainList, add: list[str], remove: tuple[str, ...]) -> bytes:
    # Owner facts point 14: the line of each entry that normalises to a removed entry is taken
    # out, and each added entry gets a new line after the last entry line, at the indent of the
    # entry lines, in the order of `add`. Every other byte is copied as it is.
    lines = text[plain.start : plain.end].split(b"\n")[:-1]
    kept = [
        line + b"\n"
        for line, entry in zip(lines, plain.entries)
        if _normalize_api_route_rule(entry) not in remove
    ]
    new = [b" " * plain.indent + b"- " + entry.encode("utf-8") + b"\n" for entry in add]
    return text[: plain.start] + b"".join(kept + new) + text[plain.end :]


def _only_entry_lines_changed(text: bytes, plain: _PlainList, rewritten: bytes) -> bool:
    # Owner facts point 15a: `rewritten` is the bytes of `text` before its first entry line, then
    # entry lines at the indent of the old ones, then the bytes of `text` after its last entry line.
    head, tail = text[: plain.start], text[plain.end :]
    middle = rewritten[len(head) : len(rewritten) - len(tail)]
    if rewritten != head + middle + tail or not middle.endswith(b"\n"):
        return False
    try:
        lines = middle.decode("utf-8").split("\n")[:-1]
    except UnicodeDecodeError:
        return False
    prefix = " " * plain.indent + "- "
    return all(
        line.startswith(prefix) and _entry_fault(line[len(prefix) :]) is None for line in lines
    )


def apply_routes(text: bytes, change: RouteChange) -> bytes:
    """Rewrite one role's config.yaml text by its part of the plan, and return the new text.

    The text must be in the plain form (see the module docstring). The line of every entry that
    normalises to an entry of `change.remove` is taken out, and each entry of `change.add` the
    list does not hold yet gets a new line after the last entry line, at the indent of the entry
    lines, in the order of `change.add`; every other byte of `text` is copied as it is. When
    nothing is left to add or remove, `text` itself is returned, so a change applied twice adds
    nothing twice.

    Raises RoutePlanError, before any edit and in this order, for a change whose entries are not
    exact route rules or that both adds and removes an entry, for a list that is not in allowlist
    mode, for a text that is not in the plain form (naming the first condition that does not
    hold), for a change that would leave the list with no entry and for an added entry that the
    plain form cannot hold; and after the edit, unless every byte outside the entry lines is the
    input's and the gateway's reading of the result is allowlist mode with exactly the old
    patterns, without the removed entries and with the added ones at the end.
    """
    role = change.role
    for entry in (*change.add, *change.remove):
        if not _is_exact_route(entry):
            raise RoutePlanError(
                f"the change for role {role} has an entry that is not an exact route: {entry!r}"
            )
    for entry in change.add:
        if entry in change.remove:
            raise RoutePlanError(f"the change for role {role} both adds and removes {entry}")
    mode, old = _route_list(text)
    if mode == _ROUTES_DENY_ALL:
        raise RoutePlanError(f"role {role} has a route list that is not valid")
    if mode != _ROUTES_ALLOWLIST:
        raise RoutePlanError(f"role {role} has a route list that does not limit routes")
    add = [entry for entry in dict.fromkeys(change.add) if entry not in old]
    if not add and not any(entry in old for entry in change.remove):
        return text
    plain = _plain_list(text, role)
    planned = [entry for entry in old if entry not in change.remove] + add
    if not planned:
        # An empty list means "not limited".
        raise RoutePlanError(
            f"the change for role {role} would leave its route list with no entry, "
            "which would open every route"
        )
    for entry in add:
        if _entry_fault(entry) is not None:
            raise RoutePlanError(
                f"the change for role {role} adds {entry}, which is not an entry of the plain "
                "form (condition f)"
            )
    rewritten = _edit(text, plain, add, change.remove)
    if not _only_entry_lines_changed(text, plain, rewritten):
        raise RoutePlanError(
            f"the rewrite of role {role} would change more than the entry lines of its route list"
        )
    if _route_list(rewritten) != (_ROUTES_ALLOWLIST, planned):
        raise RoutePlanError(f"the rewritten route list of role {role} is not the planned one")
    return rewritten
