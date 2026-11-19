"""The learned effort record: the levels a provider was seen to accept or refuse, per model.

The record is ``model_efforts.json`` at the root Hermes home, beside the shared board, so every role reads
the same file. It is data written elsewhere; this module only reads it, from disk with no network, and
caches it by the stamp of the file it opened. A missing, faulty or other-version file reads as no record;
a faulty entry reads as absent and the other entries stay. Nothing here raises into a request.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

RECORD_NAME = "model_efforts.json"
_SCHEMA_VERSION = 1
_SOURCES = ("account_list", "probe")
_THINKING_FIELDS = ("adaptive", "manual", "disable")
_LAST_TIMESTAMP = 253_402_300_799  # 9999-12-31T23:59:59Z
_CONTEXT_VARIANT_SUFFIX = "-900k"  # Hermes-side picker alias; the record keys the base slug
# Integer text longer than this is outside every valid value. It is not converted: ``int()`` of more than
# 4300 digits raises, which would drop the whole file instead of the one faulty entry.
_LONGEST_INT_TEXT = 20


@dataclass(frozen=True)
class EffortEntry:
    """One model's entry: the levels proven accepted and the levels proven refused, as written."""

    efforts: tuple[str, ...]
    refused: tuple[str, ...]


def record_path() -> Path:
    """The record in the root Hermes home; a worker whose ``HERMES_HOME`` is a profile home gets the root."""
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / RECORD_NAME


def lookup(provider: str, model: Optional[str]) -> Optional[EffortEntry]:
    """The valid entry of ``model`` (lower case, no ``vendor/`` prefix, no ``-900k``) for ``provider``."""
    if not isinstance(model, str):
        return None
    bare = model.strip().lower().rsplit("/", 1)[-1]
    if bare.endswith(_CONTEXT_VARIANT_SUFFIX):
        bare = bare[: -len(_CONTEXT_VARIANT_SUFFIX)]
    return _entries().get(f"{provider}:{bare}")


def select_effort(
    asked: Any, name_levels: Sequence[str], entry: EffortEntry, route_levels: Sequence[str]
) -> Optional[str]:
    """The level to send under ``entry``, or None for no effort field.

    The allowed levels are the route's ordered levels (weakest first) that the name rule or the entry's
    ``efforts`` hold and its ``refused`` does not. The ask is sent when allowed, else the nearest weaker
    allowed level, else the weakest allowed one. Levels compare in lower case; an entry level off the
    route's levels is ignored, and an ask off them gets no effort field.
    """
    route = [level.lower() for level in route_levels]
    accepted = {level.lower() for level in (*name_levels, *entry.efforts)}
    refused = {level.lower() for level in entry.refused}
    allowed = [level for level in route if level in accepted and level not in refused]
    ask = asked.lower() if isinstance(asked, str) else None
    if ask not in route:
        return None
    if ask in allowed:
        return ask
    weaker = [level for level in allowed if route.index(level) < route.index(ask)]
    if weaker:
        return weaker[-1]
    return allowed[0] if allowed else None


# (stamp of the file read, its valid entries), replaced as a whole so a request sees one snapshot.
_last_read: tuple[Optional[tuple], dict[str, EffortEntry]] = (None, {})
_fill_lock = threading.Lock()


def _entries() -> dict[str, EffortEntry]:
    """The valid entries of the record; the file is read again only when its stamp changes."""
    global _last_read
    try:
        path = record_path()
    except Exception:  # no root home can be resolved: no record
        return {}
    stamp, entries = _last_read
    if _stamp_on_disk(path) == stamp:
        return entries
    with _fill_lock:
        stamp, entries = _last_read
        on_disk = _stamp_on_disk(path)
        if on_disk == stamp:  # filled by another request while this one waited
            return entries
        _last_read = _read(path) if on_disk is not None else (None, {})
        return _last_read[1]


def _stamp(path: Path, status: os.stat_result) -> Optional[tuple]:
    """What tells one version of the file from another; None for anything but a regular file."""
    if not stat.S_ISREG(status.st_mode):
        return None
    return (str(path), status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _stamp_on_disk(path: Path) -> Optional[tuple]:
    try:
        return _stamp(path, os.stat(path))
    except (OSError, ValueError):
        return None


def _read(path: Path) -> tuple[Optional[tuple], dict[str, EffortEntry]]:
    """(stamp of the file opened, its valid entries); the stamp is taken from the open file itself."""
    stamp, data = None, None
    try:
        with open(path, encoding="utf-8") as handle:
            stamp = _stamp(path, os.fstat(handle.fileno()))
            if stamp is not None:
                data = json.load(handle, parse_int=_bounded_int)
    except (OSError, ValueError, RecursionError):
        data = None
    return stamp, _valid_entries(data)


def _bounded_int(text: str) -> Any:
    return int(text) if len(text) <= _LONGEST_INT_TEXT else float("inf")


def _valid_entries(data: Any) -> dict[str, EffortEntry]:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        return {}
    models = data.get("models")
    if data["schema_version"] != _SCHEMA_VERSION or not isinstance(models, dict):
        return {}
    entries = {}
    for key, raw in models.items():
        entry = _entry(raw)
        if entry is not None:
            entries[key] = entry
    return entries


def _entry(raw: Any) -> Optional[EffortEntry]:
    """The entry when its shape is exact, else None."""
    if not isinstance(raw, dict):
        return None
    efforts, refused, thinking = raw.get("efforts"), raw.get("refused"), raw.get("thinking")
    if not (_is_level_list(efforts) and _is_level_list(refused)):
        return None
    if not isinstance(thinking, dict) or not all(
        field in thinking and (thinking[field] is None or isinstance(thinking[field], bool))
        for field in _THINKING_FIELDS
    ):
        return None
    observed_at = raw.get("observed_at")
    if raw.get("source") not in _SOURCES or type(observed_at) is not int:
        return None
    if not 0 <= observed_at <= _LAST_TIMESTAMP:
        return None
    return EffortEntry(efforts=tuple(efforts), refused=tuple(refused))


def _is_level_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(level, str) for level in value)
