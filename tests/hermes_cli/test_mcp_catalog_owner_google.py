"""Gmail Work and Calendar Work ship as catalogue entries, the way Drive Work does.

Plan B3 (owner decision D1): the Connections page acts only on catalogue
entries, so each of Google's official hosted servers gets a manifest with an
http transport, an oauth sign-in and no install step. Only read tools are
pre-checked.
"""

from pathlib import Path

import pytest

# Addresses and sources from Google's official documentation, read 2026-10-31.
# The operator checked that each address equals the address of the installed
# root entry of the same name, so a reconnect does not switch endpoints.
EXPECTED = {
    "google-gmail-work": {
        "source": "https://developers.google.com/workspace/gmail/api/reference/mcp",
        "url": "https://gmailmcp.googleapis.com/mcp/v1",
        "read_tools": [
            "get_message",
            "get_thread",
            "list_drafts",
            "list_labels",
            "search_threads",
        ],
    },
    "google-calendar-work": {
        "source": "https://developers.google.com/workspace/calendar/api/guides/configure-mcp-server",
        "url": "https://calendarmcp.googleapis.com/mcp/v1",
        "read_tools": [
            "get_event",
            "list_calendars",
            "list_events",
        ],
    },
}

# Tools that change mail or calendar data (create_draft, label_thread,
# unlabel_message, update_event, delete_event, respond_to_event, ...) are never
# pre-checked.
WRITE_TOOL_PREFIXES = ("create_", "label_", "unlabel_", "update_", "delete_", "respond_")


def _entry(name):
    from hermes_cli.mcp_catalog import _parse_manifest

    manifest = Path(__file__).parents[2] / "optional-mcps" / name / "manifest.yaml"
    return _parse_manifest(manifest)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_manifest_parses_with_http_oauth_and_no_install(name):
    entry = _entry(name)

    assert entry.name == name
    assert entry.transport.type == "http"
    assert entry.auth.type == "oauth"
    assert entry.install is None


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_address_equals_google_official_address(name):
    entry = _entry(name)

    assert entry.source == EXPECTED[name]["source"]
    assert entry.transport.url == EXPECTED[name]["url"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_default_enabled_is_exactly_the_read_tools(name):
    enabled = _entry(name).tools.default_enabled

    assert enabled == EXPECTED[name]["read_tools"]
    assert [tool for tool in enabled if tool.startswith(WRITE_TOOL_PREFIXES)] == []
