"""The gateway housekeeper re-drives non-terminal board removals every tick."""
import gateway.run as gateway_run


class _OneTickStopEvent:
    """Run one housekeeping tick without a sleep or background thread."""

    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_resumes_removal_operations(monkeypatch):
    import hermes_cli.owner_workspace as ow

    calls = []
    monkeypatch.setattr(ow, "resume_removal_operations", lambda: calls.append(True))
    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)
    assert calls == [True]
