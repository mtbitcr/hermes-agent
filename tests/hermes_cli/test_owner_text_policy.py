"""The one owner-text policy and its shared checker (plan section F, card 1).

hermes_cli/owner_text.py holds the policy as one constant: the detection
rules, the fixed sentences of plan section D and the shared cases. The owner
app's copy is the module's canonical JSON output. These tests hold the
platform's checker and its owner-facing wrappers to that constant and to the
section D sentences, word for word.
"""

import json
import re
from pathlib import Path

import pytest

from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import (
    OWNER_TEXT_POLICY,
    check_owner_text,
    owner_text_policy_json,
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
    assert OWNER_TEXT_POLICY["sentences"] == _SENTENCES
    for kind, sentence in _SENTENCES.items():
        assert check_owner_text(None, kind) == sentence


def test_every_shared_case_holds():
    cases = OWNER_TEXT_POLICY["cases"]
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
    assert ow.owner_project_name(name) == _SENTENCES["project_name"]


@pytest.mark.parametrize(
    "title",
    ["Ship the thing", "Café naïve 日本語", "Ship the thing \U0001F680"],
    ids=["plain", "accented", "emoji"],
)
def test_plain_accented_and_emoji_titles_come_back_unchanged(title):
    assert ow.owner_title(title) == title
    assert ow.owner_project_name(title) == title
    assert check_owner_text(title, "title") == title
    assert check_owner_text(title, "project_name") == title


_PAST_THE_TITLE_BOUND = "Plan the shop " + "x" * 240 + " t_0000abcd"
_PAST_THE_NAME_BOUND = "Shoe Shop " + "z" * 160 + " t_0000abcd"


@pytest.mark.parametrize(
    "text",
    [
        "Archive /srv/place\nholder",
        "Archive /srv/\nplaceholder",
        "Follow up on t_0000\nabcd before Friday",
        "Rebuild 0a1b\n2c3 tonight",
        _PAST_THE_TITLE_BOUND,
    ],
    ids=[
        "path-break-inside-a-segment",
        "path-break-after-a-slash",
        "wrapped-kernel-id",
        "wrapped-commit-id",
        "id-past-the-bound",
    ],
)
def test_owner_title_checks_the_whole_input(text):
    assert ow.owner_title(text) == _SENTENCES["title"]


@pytest.mark.parametrize(
    "text",
    [
        "Archive /srv/place\nholder",
        "Archive /srv/\nplaceholder",
        "Notes t_0000\nabcd",
        _PAST_THE_NAME_BOUND,
    ],
    ids=[
        "path-break-inside-a-segment",
        "path-break-after-a-slash",
        "wrapped-kernel-id",
        "id-past-the-bound",
    ],
)
def test_owner_project_name_checks_the_whole_input(text):
    assert ow.owner_project_name(text) == _SENTENCES["project_name"]


@pytest.mark.parametrize(
    "title",
    [
        "Archive /srv/placeholder",
        "Check run_12345678 tonight",
        "Check 123e4567-e89b-12d3-a456-426614174000",
        "Ask Anthropic for the draft",
        "Fix password: placeholder",
    ],
    ids=["path", "run-id", "uuid", "provider", "secret"],
)
def test_owner_title_private_pattern_matches_get_the_title_sentence(title):
    assert ow.owner_title(title) == _SENTENCES["title"]


def test_wrappers_keep_their_blank_answers_and_long_safe_text_is_still_cut():
    assert ow.owner_title("   ") == "Untitled work item"
    assert ow.owner_project_name("") == "Untitled Project"
    long_title = "Plan the shop " + "x" * 300
    assert ow.owner_title(long_title) == long_title[:240]
    long_name = "Shoe Shop " + "z" * 200
    assert ow.owner_project_name(long_name) == long_name[:160]
    assert ow.owner_project_name("B03 — Shoe Shop") == "B03 — Shoe Shop"


@pytest.mark.parametrize(
    "text",
    [
        "Open config/settings.py",
        "Open (/srv/placeholder)",
        "Open config∕settings.py",
        "Open config＼settings.py",
        "Use https://example.test/help",
        "Visit www.example.test today",
        "Read example.test today",
        "Publish revision 0a1b2c3",
        "Publish revision abcdefa",
    ],
    ids=[
        "relative-path",
        "bracketed-path",
        "division-slash",
        "fullwidth-backslash",
        "web-address",
        "www-host",
        "bare-domain",
        "short-commit-id",
        "letter-only-commit-id",
    ],
)
def test_titles_and_project_names_refuse_paths_addresses_and_commit_ids(text):
    assert check_owner_text(text, "title") == _SENTENCES["title"]
    assert check_owner_text(text, "project_name") == _SENTENCES["project_name"]
    assert ow.owner_title(text) == _SENTENCES["title"]
    assert ow.owner_project_name(text) == _SENTENCES["project_name"]


@pytest.mark.parametrize(
    "text",
    [
        "Use Azure for the draft.",
        "Ask OpenRouter for the draft.",
        "OpenAIで確認した。",
        "確認はClaudeで。",
        "Plan it with Together AI.",
        "Draft it with gemma-2.",
    ],
    ids=["azure", "openrouter", "before-japanese", "after-japanese", "qualified", "versioned"],
)
def test_provider_and_model_names_are_hidden_next_to_any_text(text):
    assert check_owner_text(text, "planner_prose") == _SENTENCES["planner_prose"]
    assert check_owner_text(text, "title") == _SENTENCES["title"]
    assert ow.owner_title(text) == _SENTENCES["title"]
    assert ow.owner_project_name(text) == _SENTENCES["project_name"]


@pytest.mark.parametrize(
    "text",
    ["Plan the trip together", "Update the Google Calendar", "Book the fireworks"],
)
def test_everyday_words_that_name_a_provider_stay_readable(text):
    assert ow.owner_title(text) == text
    assert ow.owner_project_name(text) == text
    assert check_owner_text(text, "planner_prose") == text


def test_policy_json_is_canonical_and_loads_back_to_the_constant():
    text = owner_text_policy_json()
    assert json.loads(text) == OWNER_TEXT_POLICY
    assert owner_text_policy_json() == text
    assert text == json.dumps(
        OWNER_TEXT_POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert not Path(ow.__file__).with_name("owner_text_policy.json").exists()


def _policy_patterns():
    rules = OWNER_TEXT_POLICY["rules"]
    yield "line_break", {"pattern": rules["line_break"], "flags": "u"}
    yield "wrapped_line_break", {"pattern": rules["wrapped_line_break"], "flags": "u"}
    for group, group_rules in rules["groups"].items():
        for name, rule in group_rules.items():
            yield f"{group}.{name}", rule


def _portability_problems(pattern):
    """What in ``pattern`` Python's re and JavaScript's RegExp (u flag) could
    read differently."""
    from re import _constants as sre
    from re import _parser

    problems = []
    if re.search(r"\(\?(?![:=!]|<[=!])", pattern):
        problems.append("group other than (?: (?= (?! (?<= (?<!")
    for escape in re.finditer(r"\\(u[0-9a-f]{4}|.)", pattern):
        if escape.group(1)[0] != "u" and escape.group(1) not in "tnr\\.[](){}*+?|^$/":
            problems.append(f"escape \\{escape.group(1)}")
    parsed = _parser.parse(pattern)
    if parsed.state.flags & ~sre.SRE_FLAG_UNICODE or parsed.state.groupdict:
        problems.append("flags or named groups")

    def walk(subpattern):
        for op, av in subpattern:
            if op in (
                sre.AT, sre.ANY, sre.CATEGORY, sre.GROUPREF, sre.GROUPREF_EXISTS,
                sre.ATOMIC_GROUP, sre.POSSESSIVE_REPEAT,
            ):
                problems.append(str(op))
            elif op is sre.IN:
                problems.extend(str(item) for item, _ in av if item is sre.CATEGORY)
            elif op is sre.SUBPATTERN:
                if av[1] or av[2]:
                    problems.append("inline flags")
                walk(av[3])
            elif op in (sre.ASSERT, sre.ASSERT_NOT):
                low, high = av[1].getwidth()
                if low != high:
                    problems.append("lookaround of varying length")
                walk(av[1])
            elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT):
                walk(av[2])
            elif op is sre.BRANCH:
                for branch in av[1]:
                    walk(branch)

    walk(parsed)
    return problems


def test_policy_patterns_use_only_portable_constructs():
    found = {
        name: problems
        for name, rule in _policy_patterns()
        if (problems := _portability_problems(rule["pattern"]) + (
            [] if rule["flags"] == "u" else [f"flags {rule['flags']!r}"]
        ))
    }
    assert found == {}
