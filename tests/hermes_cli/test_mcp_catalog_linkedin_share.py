"""LinkedIn Share ships as a catalogue entry, the way Drive Work does.

Owner decisions of 31 October: the connection lives in the business role's own
profile, as Claude Design lives in the designer's. The connector is the owner
app's existing Worker, so the entry is an http transport with an oauth sign-in
and no install step.
"""

from pathlib import Path


def _entry():
    from hermes_cli.mcp_catalog import _parse_manifest

    manifest = Path(__file__).parents[2] / "optional-mcps" / "linkedin-share" / "manifest.yaml"
    return _parse_manifest(manifest)


def test_manifest_parses_with_exactly_the_worker_fields():
    entry = _entry()

    assert entry.name == "linkedin-share"
    assert entry.transport.type == "http"
    assert entry.transport.url == "https://linkedin-share.agentsbridge.net/mcp"
    assert entry.auth.type == "oauth"
    assert entry.source == (
        "https://learn.microsoft.com/en-us/linkedin/consumer/integrations/"
        "self-serve/share-on-linkedin"
    )
    assert entry.tools.default_enabled == ["read_profile_name", "create_post", "delete_post"]
    assert entry.post_install.strip()


def test_manifest_needs_no_install_step_key_or_local_process():
    entry = _entry()

    assert entry.install is None
    assert entry.auth.env == []
    assert entry.auth.provider is None
    assert entry.transport.command is None
    assert entry.transport.env == {}
    assert entry.tools.default_excluded is None
    assert entry.suggest is None
