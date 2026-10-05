"""Shared setup for the platform half of the owner payload contract tests.

Each test saves one example of an owner payload, built by the real producer,
as ``owner_payloads/<family>/<record_kind>.json`` for the owner app's checks
to read. Tests assert relationships only: a saved example has the shape the
producer returns now, record by record, and every value of each closed
vocabulary appears in some example. Examples are rewritten only when
``--write-owner-payloads`` is given and are never compared byte for byte.

The frozen clock and frozen ids only keep a rewritten example stable; no test
asserts their values. The clock reaches the ``datetime`` bindings of the
repository modules already imported when a test starts, so a test module
imports its producers at the top.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

import pytest

import hermes_time  # noqa: F401  (its ``datetime`` binding is the cron clock)

OWNER_PAYLOADS = Path(__file__).resolve().parent / "owner_payloads"
REPOSITORY = Path(__file__).resolve().parents[2]
WRITE_OPTION = "--write-owner-payloads"
FROZEN_TIME = 1_790_000_000.0
_REAL_DATETIME = datetime.datetime


def pytest_addoption(parser):
    parser.addoption(
        WRITE_OPTION,
        action="store_true",
        default=False,
        help="Rewrite tests/contracts/owner_payloads from the real producers.",
    )


def _shape(value):
    """The keys of a JSON value, record by record, without its values.

    Each record of a list keeps its own shape and is paired by position with
    the record in the same place among those of its event kind (records with
    no ``event`` pair among themselves), so a record that loses a key cannot
    hide behind another record's shape. The kind only pairs records; events of
    different kinds may interleave differently from run to run. A scalar in a
    list has no keys and is left out. A string holding JSON, like a stored
    owner reply, is compared by its decoded shape.
    """
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        records: dict = {}
        for item in value:
            shape = _shape(item)
            if shape is None:
                continue
            kind = item.get("event") if isinstance(item, dict) else None
            records.setdefault(kind if isinstance(kind, str) else None, []).append(shape)
        return records
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return _shape(json.loads(value))
        except ValueError:
            return None
    return None


@pytest.fixture
def owner_payload_example(request):
    """``example(family, record_kind, payload)`` returns the saved example.

    With the write option the live ``payload`` is written first. Either way
    the saved file is read back, and its shape, record by record, must equal
    the shape of the live payload the real producer just returned.
    """
    write = request.config.getoption(WRITE_OPTION, default=False)

    def example(family: str, record_kind: str, payload):
        payload = json.loads(json.dumps(payload))
        path = OWNER_PAYLOADS / family / f"{record_kind}.json"
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        assert path.is_file(), (
            f"no saved example {family}/{record_kind}.json; "
            f"write it with {WRITE_OPTION}"
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert _shape(saved) == _shape(payload), f"{family}/{record_kind}"
        return saved

    return example


class _FrozenClock(type):
    """Lets ``isinstance`` and ``issubclass`` treat the stand-in as ``datetime``."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, _REAL_DATETIME)

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, _REAL_DATETIME)


class _FrozenDatetime(_REAL_DATETIME, metaclass=_FrozenClock):
    """``datetime`` whose ``now`` reads ``time.time``; it builds real instances."""

    def __new__(cls, *args, **kwargs):
        return _REAL_DATETIME(*args, **kwargs)

    @classmethod
    def now(cls, tz=None):
        return _REAL_DATETIME.fromtimestamp(time.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.now(datetime.timezone.utc).replace(tzinfo=None)


def _repository_bindings(value):
    """``(module, name)`` for each global of an imported repository module bound to ``value``."""
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None) or {}
        path = str(namespace.get("__file__") or "")
        if namespace is globals() or "site-packages" in path or not path.startswith(
            f"{REPOSITORY}{os.sep}"
        ):
            continue
        for name, bound in list(namespace.items()):
            if bound is value:
                yield module, name


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Wall-clock reads return one instant; monotonic time still moves.

    ``time.time`` is frozen, and ``datetime.now`` wherever an imported
    repository module bound the ``datetime`` class, ``hermes_time.now``
    included. The ``datetime`` module itself is left alone.
    """
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)
    for module, name in list(_repository_bindings(_REAL_DATETIME)):
        monkeypatch.setattr(module, name, _FrozenDatetime)


@pytest.fixture(autouse=True)
def frozen_ids(monkeypatch):
    """``uuid4`` and ``token_hex`` hash a counter into every byte they return.

    Ids repeat from run to run, yet any prefix a producer keeps still differs
    between draws, and a ``uuid4`` keeps its version and variant bits.
    """
    counter = itertools.count(1)

    def _drawn(size: int) -> bytes:
        return hashlib.shake_256(str(next(counter)).encode()).digest(size)

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(bytes=_drawn(16), version=4))
    monkeypatch.setattr(
        secrets, "token_hex",
        lambda nbytes=None: _drawn(32 if nbytes is None else nbytes).hex(),
    )
