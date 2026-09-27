"""Gateway housekeeping no longer drains restart-safe cron deliveries."""

from types import SimpleNamespace

import pytest

import gateway.run as gateway_run

DRAIN_LABEL = "Cron durable delivery queue drain"


class _TickLimitStopEvent:
    """Let the housekeeping loop run a fixed number of ticks without sleeping."""

    def __init__(self, ticks):
        self.remaining = ticks

    def is_set(self):
        return self.remaining <= 0

    def wait(self, timeout=None):
        self.remaining -= 1
        return self.remaining <= 0


def _registered_chores(monkeypatch, **kwargs):
    """Labels of every chore the loop runs over 60 ticks (the longest cadence); no chore body runs."""
    labels = []
    monkeypatch.setattr(
        gateway_run, "_housekeeping_chore", lambda label, fn, *args, **kw: labels.append(label)
    )
    gateway_run._start_gateway_housekeeping(_TickLimitStopEvent(60), interval=0, **kwargs)
    return labels


def _runner():
    return SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters={},
        _profile_adapters={"secondary": {}},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"adapters": {"discord": object()}, "loop": object()}, id="live-adapters"),
        pytest.param({"adapters": {}, "loop": object()}, id="no-connected-adapters"),
        pytest.param({"loop": object(), "runner": _runner()}, id="runner-only"),
        pytest.param(
            {"adapters": {"slack": object()}, "loop": object(), "runner": _runner()},
            id="adapters-and-runner",
        ),
    ],
)
def test_gateway_housekeeping_does_not_register_cron_delivery_drain(monkeypatch, kwargs):
    labels = _registered_chores(monkeypatch, **kwargs)

    assert "Channel directory refresh" in labels
    assert DRAIN_LABEL not in labels
    assert not [label for label in labels if "delivery queue" in label.lower()]
