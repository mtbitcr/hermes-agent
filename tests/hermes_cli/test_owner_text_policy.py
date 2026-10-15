"""The one owner-text policy and its shared checker (plan section F, card 1).

hermes_cli/owner_text_policy.json holds the detection rules, the fixed
sentences of plan section D and the shared cases, and the owner app keeps an
exact copy of it. These tests hold the platform's checker to that file and to
the section D sentences, word for word.
"""

import json
from pathlib import Path

import pytest

from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text

_POLICY = json.loads(
    Path(ow.__file__).with_name("owner_text_policy.json").read_text(encoding="utf-8")
)
_SENTENCES = {
    "title": "This piece of work has a title Raphael cannot show here.",
    "project_name": "This project has a name Raphael cannot show here.",
    "stop_reason_decision": "Raphael needs your answer before this work can continue.",
    "stop_reason_receipt": "Work stopped and needs attention.",
    "planner_prose": (
        "Raphael wrote part of this in a way it cannot show you. "
        "Ask Raphael to say it again in plain words."
    ),
    "stored_failure": (
        "Something went wrong inside Raphael while doing this work. "
        "Nothing you did caused it."
    ),
    "retry_reason": "Raphael could not try this work again.",
    "browser_network_error": (
        "Raphael could not be reached. Check your connection, then try again."
    ),
    "automation_start": "That start time has already passed. Tell Raphael a new time.",
}


def test_each_kind_falls_back_to_its_section_d_sentence():
    assert _POLICY["sentences"] == _SENTENCES
    for kind, sentence in _SENTENCES.items():
        assert check_owner_text(None, kind) == sentence


def test_every_shared_case_holds():
    cases = _POLICY["cases"]
    assert {case["kind"] for case in cases} == set(_SENTENCES)
    wrong = [
        case
        for case in cases
        if check_owner_text(case["text"], case["kind"]) != case["expected"]
    ]
    assert wrong == []


@pytest.mark.parametrize(
    "text",
    [
        "Cannot open /srv/place\nholder for reading",
        "Cannot open /srv/\nplaceholder for reading",
    ],
    ids=["break-inside-a-segment", "break-after-a-slash"],
)
def test_wrapped_path_falls_back(text):
    decision = _SENTENCES["stop_reason_decision"]
    receipt = _SENTENCES["stop_reason_receipt"]
    assert check_owner_text(text, "stop_reason_decision") == decision
    assert check_owner_text(text, "stop_reason_receipt") == receipt
    assert ow.owner_stop_reason(text) == decision
    assert ow.owner_stop_reason(text, fallback=receipt) == receipt


@pytest.mark.parametrize(
    "text",
    [
        "Failed to fetch",
        "Traceback (most recent call last):",
        "[Errno 111] Connection refused",
        "KeyError: 'name'",
    ],
)
def test_exception_text_falls_back(text):
    assert check_owner_text(text, "stored_failure") == _SENTENCES["stored_failure"]


@pytest.mark.parametrize(
    "text",
    [
        "I sent the plan to owner@example.test this morning.",
        "I asked @placeholder_owner to confirm the room.",
        "I posted the plan in chat -1001234567890 for you.",
    ],
    ids=["e-mail", "chat-handle", "chat-id"],
)
def test_addresses_fall_back(text):
    assert check_owner_text(text, "planner_prose") == _SENTENCES["planner_prose"]


@pytest.mark.parametrize(
    "text",
    [
        "Claude Opus drafted the plan for you.",
        "I checked the dates with GPT-5 first.",
        "Anthropic wrote the first draft.",
    ],
)
def test_model_and_provider_names_fall_back(text):
    assert check_owner_text(text, "planner_prose") == _SENTENCES["planner_prose"]


def test_owner_title_hides_a_leaking_title():
    title = "Follow up on t_0000abcd before Friday"
    assert ow.owner_title(title) == _SENTENCES["title"]
    assert check_owner_text(title, "title") == _SENTENCES["title"]
    name = "Archive /srv/placeholder"
    assert check_owner_text(name, "project_name") == _SENTENCES["project_name"]


@pytest.mark.parametrize(
    "title",
    ["Ship the thing", "Café naïve 日本語", "Ship the thing \U0001F680"],
    ids=["plain", "accented", "emoji"],
)
def test_plain_accented_and_emoji_titles_come_back_unchanged(title):
    assert ow.owner_title(title) == title
    assert check_owner_text(title, "title") == title
    assert check_owner_text(title, "project_name") == title
