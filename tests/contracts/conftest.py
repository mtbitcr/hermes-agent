"""Shared setup for the platform half of the owner payload contract tests.

Each test saves one example of an owner payload, built by the real producer,
as ``owner_payloads/<family>/<record_kind>.json`` for the owner app's checks
to read. Tests assert relationships only: a saved example's key set equals the
key set the producer returns now, and every value of each closed vocabulary
appears in some example. Examples are rewritten only when
``--write-owner-payloads`` is given and are never compared byte for byte.

The frozen clock and frozen ids only keep a rewritten example stable; no test
asserts their values.
"""

from __future__ import annotations

import itertools
import json
import secrets
import time
import uuid
from pathlib import Path

import pytest

OWNER_PAYLOADS = Path(__file__).resolve().parent / "owner_payloads"
WRITE_OPTION = "--write-owner-payloads"
FROZEN_TIME = 1_790_000_000.0


def pytest_addoption(parser):
    parser.addoption(
        WRITE_OPTION,
        action="store_true",
        default=False,
        help="Rewrite tests/contracts/owner_payloads from the real producers.",
    )


def _key_paths(value, prefix: str = "") -> set[str]:
    """Every key path of a JSON value; all items of a list share ``[]``."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= _key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= _key_paths(item, f"{prefix}[]")
    return paths


@pytest.fixture
def owner_payload_example(request):
    """``example(family, record_kind, payload)`` returns the saved example.

    With the write option the live ``payload`` is written first. Either way
    the saved file is read back, and its key set must equal the key set of
    the live payload the real producer just returned.
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
        assert _key_paths(saved) == _key_paths(payload), f"{family}/{record_kind}"
        return saved

    return example


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Wall-clock reads return one instant; monotonic time still moves."""
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)


@pytest.fixture(autouse=True)
def frozen_ids(monkeypatch):
    """``uuid4`` and ``token_hex`` count up from one instead of drawing randomly."""
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter), version=4))
    monkeypatch.setattr(
        secrets,
        "token_hex",
        lambda nbytes=None: f"{next(counter):0{2 * (32 if nbytes is None else nbytes)}x}",
    )
