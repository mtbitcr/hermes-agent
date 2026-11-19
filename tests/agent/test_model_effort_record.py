"""The reader of the learned effort record (``agent/model_effort_record.py``).

Each test proves the reader on the final request of the Codex backend route, the one route that reads the
record: the real transport builds the request and the real OpenAI SDK serialises it, captured offline by an
in-process mock HTTP transport, so no provider is called. The record is ``model_efforts.json`` at the root
Hermes home; a test's HERMES_HOME is that root unless the test runs as a worker with a profile home.
"""

import builtins
import itertools
import json
import os
import threading
from pathlib import Path

import pytest

from agent.transports import get_transport

CODEX_BACKEND = "https://chatgpt.com/backend-api/codex"
EVERY_LEVEL = ["none", "low", "medium", "high", "xhigh", "max"]
_record_versions = itertools.count(1)


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


def _root_record():
    return Path(os.environ["HERMES_HOME"]) / "model_efforts.json"


def _write_raw(content, path=None):
    """Replace the record atomically with ``content`` (text or bytes); each version gets a modification time
    of its own, so it is a new file to the reader."""
    path = path or _root_record()
    staged = path.with_name(path.name + ".new")
    if isinstance(content, bytes):
        staged.write_bytes(content)
    else:
        staged.write_text(content, encoding="utf-8")
    mtime_ns = (1_800_000_000 + next(_record_versions)) * 1_000_000_000
    os.utime(staged, ns=(mtime_ns, mtime_ns))
    os.replace(staged, path)
    return path


def _write_record(models, path=None):
    return _write_raw(json.dumps({"schema_version": 1, "models": models}), path)


def _entry(efforts=(), refused=()):
    """A valid version-1 entry that accepts ``efforts`` and refuses ``refused``."""
    return {
        "efforts": list(efforts), "refused": list(refused),
        "thinking": {"adaptive": None, "manual": None, "disable": None},
        "source": "account_list", "observed_at": 1795070000,
    }


def _sent_effort(transport, model, ask):
    """The effort of the final serialized request of a Codex backend role on ``model`` that asks for ``ask``,
    or None when the request carries no effort."""
    import httpx
    import openai

    kwargs = transport.build_kwargs(
        model=model, messages=[{"role": "user", "content": "Hi"}], tools=[], provider="openai-codex",
        base_url=CODEX_BACKEND, is_codex_backend=True, reasoning_config={"enabled": True, "effort": ask},
    )
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "offline", "type": "invalid_request_error"}})

    client = openai.OpenAI(
        api_key="placeholder", base_url=CODEX_BACKEND, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.APIStatusError):
        client.responses.create(**transport.preflight_kwargs(kwargs))
    return (sent[0].get("reasoning") or {}).get("effort")


def _without(mapping, key):
    return {name: value for name, value in mapping.items() if name != key}


# Each fault of an entry's shape, applied to an entry that would let gpt-5.7 send max.
ENTRY_FAULTS = {
    "efforts is text": lambda e: {**e, "efforts": "max"},
    "efforts holds a number": lambda e: {**e, "efforts": [*e["efforts"], 6]},
    "efforts is missing": lambda e: _without(e, "efforts"),
    "refused is text": lambda e: {**e, "refused": "none"},
    "refused holds null": lambda e: {**e, "refused": [None]},
    "refused is missing": lambda e: _without(e, "refused"),
    "source is another value": lambda e: {**e, "source": "guess"},
    "source is missing": lambda e: _without(e, "source"),
    "observed_at is text": lambda e: {**e, "observed_at": "yesterday"},
    "observed_at is a fraction": lambda e: {**e, "observed_at": 1795070000.5},
    "observed_at is a boolean": lambda e: {**e, "observed_at": True},
    "observed_at is negative": lambda e: {**e, "observed_at": -1},
    "observed_at is after the year 9999": lambda e: {**e, "observed_at": 253402300800},
    "observed_at is missing": lambda e: _without(e, "observed_at"),
    "thinking is a list": lambda e: {**e, "thinking": [None, None, None]},
    "thinking is missing": lambda e: _without(e, "thinking"),
    "thinking.adaptive is missing": lambda e: {**e, "thinking": _without(e["thinking"], "adaptive")},
    "thinking.manual is text": lambda e: {**e, "thinking": {**e["thinking"], "manual": "yes"}},
    "thinking.disable is a number": lambda e: {**e, "thinking": {**e["thinking"], "disable": 0}},
    "the entry is a list": lambda e: list(e),
}

_SIBLING = '"openai-codex:gpt-5.6": ' + json.dumps(_entry(refused=["max"]))
# Each fault of the file; the entry in it (gpt-5.6 refuses max) would make gpt-5.6 send xhigh for max.
FILE_FAULTS = {
    "an empty file": "",
    "not JSON": '{"schema_version": 1, "models": {' + _SIBLING,
    "not UTF-8": b'{"schema_version": 1, "models": {"\xff": 1}}',
    "a list": "[" + json.dumps({"schema_version": 1, "models": {}}) + "]",
    "schema_version true": '{"schema_version": true, "models": {' + _SIBLING + "}}",
    "schema_version 1.0": '{"schema_version": 1.0, "models": {' + _SIBLING + "}}",
    "schema_version 2": '{"schema_version": 2, "models": {' + _SIBLING + "}}",
    "schema_version text": '{"schema_version": "1", "models": {' + _SIBLING + "}}",
    "schema_version missing": '{"models": {' + _SIBLING + "}}",
    "schema_version past the digit limit": '{"schema_version": 1' + "0" * 5000 + ', "models": {' + _SIBLING + "}}",
    "models is a list": '{"schema_version": 1, "models": [{' + _SIBLING + "}]}",
    "nested past the recursion limit": (
        '{"schema_version": 1, "models": {' + _SIBLING + '}, "deep": ' + "[" * 100_000 + "]" * 100_000 + "}"
    ),
}


class TestEntryShape:
    """A faulty entry reads as absent and the other entries stay in use; no fault raises into a request."""

    @pytest.mark.parametrize("fault", sorted(ENTRY_FAULTS))
    def test_a_faulty_entry_is_absent_and_a_valid_second_entry_stays(self, transport, fault):
        _write_record({
            "openai-codex:gpt-5.7": ENTRY_FAULTS[fault](_entry(efforts=EVERY_LEVEL)),
            "openai-codex:gpt-5.6": _entry(refused=["max"]),
        })

        assert _sent_effort(transport, "gpt-5.7", "max") == "xhigh"  # today's level: the entry is absent
        assert _sent_effort(transport, "gpt-5.6", "max") == "xhigh"  # the valid entry is in use

    @pytest.mark.parametrize("digits", [310, 5000])
    def test_an_integer_time_stamp_too_large_raises_nothing(self, transport, digits):
        too_large = json.dumps(_entry(efforts=EVERY_LEVEL)).replace("1795070000", "1" + "0" * digits)
        _write_raw('{"schema_version": 1, "models": {"openai-codex:gpt-5.7": ' + too_large + ", " + _SIBLING + "}}")

        assert _sent_effort(transport, "gpt-5.7", "max") == "xhigh"
        assert _sent_effort(transport, "gpt-5.6", "max") == "xhigh"

    @pytest.mark.parametrize("fault", sorted(FILE_FAULTS))
    def test_a_faulty_file_is_no_record(self, transport, fault):
        _write_raw(FILE_FAULTS[fault])
        assert _sent_effort(transport, "gpt-5.6", "max") == "max"

        _write_raw('{"schema_version": 1, "models": {' + _SIBLING + "}}")
        assert _sent_effort(transport, "gpt-5.6", "max") == "xhigh"

    def test_a_directory_in_place_of_the_file_is_no_record(self, transport):
        _root_record().mkdir()
        assert _sent_effort(transport, "gpt-5.6", "max") == "max"

        _root_record().rmdir()
        _write_raw('{"schema_version": 1, "models": {' + _SIBLING + "}}")
        assert _sent_effort(transport, "gpt-5.6", "max") == "xhigh"


class _WatchedLock:
    """Wraps the reader's fill lock and sets ``waiting`` when the thread named ``second`` asks for it."""

    def __init__(self, lock, waiting):
        self._lock, self._waiting = lock, waiting

    def _note(self):
        if threading.current_thread().name == "second":
            self._waiting.set()

    def __enter__(self):
        self._note()
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)

    def acquire(self, *args, **kwargs):
        self._note()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        return self._lock.release()


class TestReads:
    """One request reads the record once; the cache keeps the stamp of the file it opened, and requests on
    a cold cache share one read under a lock."""

    def _run_two_requests(self, transport, monkeypatch, path, on_first_open=None):
        """Run requests ``first`` and ``second`` for gpt-5.7 at max; ``second`` starts while ``first`` opens
        the record and holds the fill. Returns (thread names that opened the record, effort per request).
        The reader's lock is watched only from that open on; until then the test sees just the file opens."""
        second_waits = threading.Event()
        opened, sent, threads = [], {}, []
        real_open = builtins.open

        def request(name):
            sent[name] = _sent_effort(transport, "gpt-5.7", "max")

        def open_record(file, *args, **kwargs):
            if str(file) == str(path):
                opened.append(threading.current_thread().name)
                if len(opened) == 1:
                    from agent import model_effort_record  # loaded: its reader is opening the record

                    monkeypatch.setattr(
                        model_effort_record, "_fill_lock", _WatchedLock(model_effort_record._fill_lock, second_waits)
                    )
                    if on_first_open:
                        on_first_open()
                    second = threading.Thread(target=request, args=("second",), name="second")
                    threads.append(second)
                    second.start()
                    assert second_waits.wait(timeout=10)  # the second request now waits for this fill
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", open_record)
        first = threading.Thread(target=request, args=("first",), name="first")
        first.start()
        first.join(timeout=30)
        for thread in threads:
            thread.join(timeout=30)
        return opened, sent

    def test_two_requests_on_a_cold_cache_make_one_read(self, transport, monkeypatch):
        path = _write_record({"openai-codex:gpt-5.7": _entry(efforts=EVERY_LEVEL)})

        opened, sent = self._run_two_requests(transport, monkeypatch, path)

        assert sent == {"first": "max", "second": "max"}
        assert opened == ["first"]

    def test_a_file_replaced_between_the_stat_and_the_open_is_read_once(self, transport, monkeypatch):
        path = _write_record({"openai-codex:gpt-5.7": _entry(efforts=EVERY_LEVEL)})

        def replace():  # the atomic replace lands after the reader's stat and before its open
            _write_record({"openai-codex:gpt-5.7": _entry(refused=["max", "xhigh"])}, path)

        opened, sent = self._run_two_requests(transport, monkeypatch, path, on_first_open=replace)

        assert sent == {"first": "high", "second": "high"}
        assert opened == ["first"]

    def test_one_request_reads_one_snapshot(self, transport, monkeypatch):
        path = _write_record({"openai-codex:gpt-5.7": _entry(refused=["high"])})
        opened = []
        real_open = builtins.open

        def open_then_replace(file, *args, **kwargs):
            handle = real_open(file, *args, **kwargs)
            if str(file) == str(path):
                opened.append(file)
                if len(opened) == 1:  # a new version lands while the first request reads the old one
                    _write_record({"openai-codex:gpt-5.7": _entry(refused=["medium", "high"])}, path)
            return handle

        monkeypatch.setattr(builtins, "open", open_then_replace)

        assert _sent_effort(transport, "gpt-5.7", "high") == "medium"
        assert len(opened) == 1
        assert _sent_effort(transport, "gpt-5.7", "high") == "low"  # the next request reads the new version
        assert len(opened) == 2
        assert _sent_effort(transport, "gpt-5.7", "high") == "low"  # an unchanged file is not read again
        assert len(opened) == 2


class TestStorePath:
    def test_a_worker_with_a_profile_home_reads_the_record_of_the_root_home(self, transport, tmp_path, monkeypatch):
        root = tmp_path / "hermes-root"
        profile_home = root / "profiles" / "coder"
        board = root / "kanban" / "boards" / "proj-alpha"
        profile_home.mkdir(parents=True)
        board.mkdir(parents=True)
        # The environment the dispatcher gives a worker of the board proj-alpha with the profile coder.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HERMES_PROFILE", "coder")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "proj-alpha")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(board / "kanban.db"))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board / "workspaces"))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0001")
        expected = root / "model_efforts.json"
        _write_record({"openai-codex:gpt-5.7": _entry(efforts=EVERY_LEVEL)}, expected)
        # A record in the profile's own home is not the record.
        _write_record({"openai-codex:gpt-5.7": _entry(refused=["max", "xhigh"])}, profile_home / "model_efforts.json")
        opened = []
        real_open = builtins.open

        def watched_open(file, *args, **kwargs):
            if str(file).endswith("model_efforts.json"):
                opened.append(Path(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", watched_open)

        assert _sent_effort(transport, "gpt-5.7", "max") == "max"
        assert opened == [expected]
