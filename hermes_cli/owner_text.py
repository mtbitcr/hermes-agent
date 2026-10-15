"""The one owner-text policy and its checker.

:data:`OWNER_TEXT_POLICY` is the policy: the detection rules, the fixed
sentences of plan section D and the shared cases that pin both.
:func:`owner_text_policy_json` returns it as canonical JSON text; the owner
app's copy is that output, and its CI compares the copy with the output at
the pinned platform commit. :func:`check_owner_text` is the one way to apply
the policy.

Every pattern gives the same answer in Python's ``re`` (compiled with no
flag) and in JavaScript's ``RegExp`` (compiled with the ``u`` flag): it uses
explicit ASCII character classes and fixed-length lookarounds instead of
``\\b``, ``\\w``, ``\\d``, ``\\s``, their negations and ``.``, and no inline
flag, named group, ``^``, ``$``, possessive or atomic form. A word matched in
either case is spelt one class per letter.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

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


def _cased(word: str, folds: dict[str, str] | None = None) -> str:
    """``word`` with each ASCII letter spelt as a class of both its cases,
    plus whatever ``folds`` adds for that letter."""
    return "".join(
        f"[{char.upper()}{char.lower()}{(folds or {}).get(char.lower(), '')}]"
        if char.isascii() and char.isalpha()
        else char
        for char in word
    )


def _any_of(words: list[str], folds: dict[str, str] | None = None) -> str:
    return "|".join(_cased(word, folds) for word in words)


_SPACE_CHARS = (
    r"\t\n\u000b\u000c\r\u001c-\u001f \u0085\u00a0\u1680\u2000-\u200a"
    r"\u2028\u2029\u202f\u205f\u3000"
)
_SPACE = f"[{_SPACE_CHARS}]"
_NOT_SPACE = f"[^{_SPACE_CHARS}]"
_HEX = "[0-9A-Fa-f]"
_ALNUM = "[0-9A-Za-z]"
_WORD = "[0-9A-Z_a-z]"

# Stop reasons keep the answers they had before this policy (plan C3), which
# Python's Unicode-aware \b, \w, \d and \s and its re.IGNORECASE decided. The
# word stand-ins (see "word_stand_ins" below) reduce every letter and digit
# outside ASCII to one letter or one digit first, so these few classes give
# exactly those answers in both engines. Case-insensitive Python also lets
# U+0130 and U+0131 match "i", U+017F "s" and U+212A "k".
_LEGACY_FOLDS = {"i": r"\u0130\u0131", "k": r"\u212a", "s": r"\u017f"}
_LEGACY_WORD = r"[0-9A-Z_a-z\u00aa\u0130\u0131\u0660\u017f\u212a]"
_LEGACY_ALNUM = r"[0-9A-Za-z\u00aa\u0130\u0131\u0660\u017f\u212a]"
_LEGACY_DIGIT = r"[0-9\u0660]"
_LEGACY_LETTER = r"[A-Za-z\u0130\u0131\u017f\u212a]"
_LEGACY_LETTER_OR_DIGIT = r"[0-9A-Za-z\u0130\u0131\u017f\u212a]"
_LEGACY_TOKEN = r"[.0-9A-Z_a-z\u0130\u0131\u017f\u212a-]"
_LEGACY_START = f"(?<!{_LEGACY_WORD})"
_LEGACY_END = f"(?!{_LEGACY_WORD})"
_LEGACY_EDGE = (
    f"(?:(?<={_LEGACY_WORD})(?!{_LEGACY_WORD})|(?<!{_LEGACY_WORD})(?={_LEGACY_WORD}))"
)
_LEGACY_SECRET_NAMES = (
    f"(?:{_cased('api', _LEGACY_FOLDS)}[_-]?{_cased('key', _LEGACY_FOLDS)}"
    f"|{_any_of(['secret', 'password', 'passwd'], _LEGACY_FOLDS)}"
    f"|{_cased('access', _LEGACY_FOLDS)}[_-]?{_cased('token', _LEGACY_FOLDS)})"
)
_LEGACY_MODEL_FAMILIES = _any_of(
    ["gpt", "claude", "gemini", "llama", "mistral", "deepseek", "qwen", "grok"],
    _LEGACY_FOLDS,
)
_LEGACY_PROVIDERS = _any_of(
    ["openai", "anthropic", "google", "aws", "azure", "bedrock"], _LEGACY_FOLDS
)
_LEGACY_NAMES = _any_of(
    [
        "openai", "anthropic", "claude", "gpt", "gemini", "llama", "mistral",
        "deepseek", "qwen", "grok", "bedrock", "opus", "sonnet", "haiku",
    ],
    _LEGACY_FOLDS,
)

# Model and provider names hidden from every other kind (plan C1.6 and the
# owner decision of 2026-10-14). A bare name is refused on its own; a
# provider whose name is also an everyday word or name is refused only with
# its qualifier, so "together", "fireworks", "Google Calendar" or a friend
# called Kimi stay readable.
_BARE_NAMES = [
    "openai", "anthropic", "claude", "gpt", "chatgpt", "gemini", "gemma",
    "llama", "mistral", "mixtral", "deepseek", "qwen", "grok", "bedrock",
    "opus", "sonnet", "haiku", "azure", "openrouter", "deepinfra",
    "huggingface", "ollama", "nvidia", "minimax", "novita", "nebius", "arcee",
    "stepfun", "kilocode", "opencode", "commandcode", "xai",
]
_QUALIFIED_NAMES = [
    ["together", "ai"], ["fireworks", "ai"], ["vertex", "ai"], ["upstage", "ai"],
    ["moonshot", "ai"], ["meta", "ai"], ["google", "ai"], ["alibaba", "cloud"],
    ["gmi", "cloud"], ["xiaomi", "mimo"], ["github", "copilot"],
    ["microsoft", "copilot"], ["nous", "research"], ["kimi", "k2"],
    ["hugging", "face"],
]
_VERSIONED_FAMILIES = ["glm", "kimi", "phi", "mimo", "codex", "nemotron"]
_DOMAIN_ENDINGS = [
    "com", "net", "org", "io", "dev", "app", "ai", "co", "test", "example",
    "invalid", "localhost", "local", "internal", "info", "biz", "me", "xyz",
    "cloud", "site", "edu", "gov", "uk", "de",
]


def _rule(pattern: str) -> dict[str, str]:
    return {"pattern": pattern, "flags": "u"}


_GROUPS = {
    "secrets": {
        "private_key": _rule("-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "aws_access_key": _rule(f"(?<!{_WORD})AKIA[0-9A-Z]{{16}}"),
        "repository_token": _rule(f"(?<!{_WORD})gh[opsu]_[0-9A-Za-z]{{20}}"),
        "chat_token": _rule(f"(?<!{_WORD})xox[abprs]-[0-9A-Za-z-]{{10}}"),
        "secret_key_prefix": _rule(f"(?<!{_WORD})sk-[0-9A-Z_a-z-]{{16}}"),
        "bearer_token": _rule(
            f"(?<!{_WORD}){_cased('bearer')}{_SPACE}+[.0-9A-Z_a-z-]{{16}}"
        ),
        "key_value_secret": _rule(
            f"(?<!{_WORD})(?:{_cased('api')}[_-]?{_cased('key')}"
            f"|{_any_of(['secret', 'password', 'passwd'])}"
            f"|{_cased('access')}[_-]?{_cased('token')})"
            f"{_SPACE}*[:=]{_SPACE}*{_NOT_SPACE}"
        ),
    },
    "native_ids": {
        "uuid": _rule(f"{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}"),
        "task_run_response_claim_id": _rule(
            f"(?<!{_ALNUM})(?:{_any_of(['task', 'run', 'resp', 'claim'])})[_-]"
            f"(?={_ALNUM}{{6}})[A-Za-z]*[0-9]"
        ),
        "profile_name": _rule(
            f"(?<!{_ALNUM}){_cased('raphael')}-{_ALNUM}[0-9A-Z_a-z-]{{2}}"
        ),
        "kernel_card_id": _rule(f"[Tt]_{_HEX}{{8}}"),
        "project_id": _rule(f"[Pp]_{_HEX}{{8}}"),
    },
    "commit_ids": {
        "full_commit_id": _rule(f"{_HEX}{{40}}"),
        "hex_run_with_digit": _rule(f"(?<!{_HEX})(?={_HEX}{{7}})[A-Fa-f]*[0-9]"),
        "hex_word": _rule(f"(?<!{_ALNUM}){_HEX}{{7}}(?!{_ALNUM})|(?<!{_ALNUM}){_HEX}{{8,}}(?!{_ALNUM})"),
    },
    "host_paths": {
        "separator": _rule(
            r"[/\\\u2044\u2215\u2216\u2571\u2572\u27cb\u27cd\u29f5\u29f6\u29f7"
            r"\u29f8\u29f9\u2afb\u2afd\u2e4a\ufe68\uff0f\uff3c\u0337\u0338"
            r"\u20e5\u20eb]"
        ),
        "drive_letter": _rule(f"(?<!{_WORD})[A-Za-z]:(?={_NOT_SPACE})"),
    },
    "web_addresses": {
        "www_host": _rule(f"(?<!{_ALNUM}){_cased('www')}\\.{_ALNUM}"),
        "mail_link": _rule(f"(?<!{_ALNUM}){_cased('mailto')}:"),
        "domain_name": _rule(
            f"{_ALNUM}\\.(?:{_any_of(_DOMAIN_ENDINGS)})(?![0-9A-Za-z-])"
        ),
        "ip_address": _rule(
            "(?<![.0-9])[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}"
        ),
        "localhost": _rule(f"(?<!{_ALNUM}){_cased('localhost')}"),
    },
    "addresses": {
        "email": _rule("[%+.0-9A-Z_a-z-]@[0-9A-Za-z-]+\\.[0-9A-Za-z-]"),
        "chat_handle": _rule("(?<![%*+./0-9:=A-Za-z-])@[0-9A-Z_a-z]"),
        "chat_mention": _rule("<[!#@][!&]?[0-9A-Za-z]"),
        "chat_id_or_phone": _rule(f"(?<!{_ALNUM})[+-][0-9]{{7}}"),
    },
    "exception_text": {
        "traceback": _rule("Traceback"),
        "file_line": _rule('"[^"]+", line [0-9]'),
        "error_word": _rule("(?:Error|Exception):"),
        "error_number": _rule("\\[Errno"),
        "browser_error": _rule(
            "Failed to fetch|NetworkError|Load failed|Unexpected token"
            "|JSON\\.parse|is not valid JSON"
        ),
        "stack_frame": _rule(f"(?<!{_WORD})at [^ ()]+ \\([^()]*:[0-9]+:[0-9]+\\)"),
        "connection_error": _rule(
            "E(?:CONNREFUSED|CONNRESET|CONNABORTED|TIMEDOUT|HOSTUNREACH"
            "|NETUNREACH|NOTFOUND|AI_AGAIN)"
        ),
        "process_id": _rule(f"(?<!{_WORD}){_cased('pid')}[ :=]+[0-9]"),
    },
    "model_names": {
        "bare_name": _rule(f"(?<!{_ALNUM})(?:{_any_of(_BARE_NAMES)})(?![A-Za-z])"),
        "qualified_name": _rule(
            f"(?<!{_ALNUM})(?:"
            + "|".join(
                f"{_cased(name)}[\\t ._-]*{_cased(qualifier)}"
                for name, qualifier in _QUALIFIED_NAMES
            )
            + ")(?![A-Za-z])"
        ),
        "versioned_name": _rule(
            f"(?<!{_ALNUM})(?:{_any_of(_VERSIONED_FAMILIES)})-[.0-9A-Za-z]*[0-9]"
        ),
        "provider_model": _rule(
            f"(?<!{_ALNUM})(?:{_any_of(['google', 'aws'])}){_SPACE}*:{_SPACE}*{_ALNUM}"
        ),
    },
    # The stop-reason rules as they stood before this policy, one for one,
    # plus the commit-id rule. They run on the word stand-in form of the text.
    "stop_reason_compatibility": {
        "private_key": _rule("-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "aws_access_key": _rule(f"{_LEGACY_START}AKIA[0-9A-Z]{{16}}{_LEGACY_END}"),
        "repository_token": _rule(
            f"{_LEGACY_START}gh[opsu]_[0-9A-Za-z]{{20,}}{_LEGACY_END}"
        ),
        "chat_token": _rule(f"{_LEGACY_START}xox[abprs]-[0-9A-Za-z-]{{10,}}{_LEGACY_EDGE}"),
        "secret_key_prefix": _rule(
            f"{_LEGACY_START}sk-[0-9A-Z_a-z-]{{16,}}{_LEGACY_EDGE}"
        ),
        "bearer_token": _rule(
            f"{_LEGACY_START}{_cased('bearer', _LEGACY_FOLDS)}{_SPACE}+{_LEGACY_TOKEN}{{16,}}"
        ),
        "key_value_secret": _rule(
            f"{_LEGACY_START}{_LEGACY_SECRET_NAMES}{_SPACE}*[:=]{_SPACE}*{_NOT_SPACE}"
        ),
        "hex_64": _rule(f"{_LEGACY_START}{_HEX}{{64}}{_LEGACY_END}"),
        "uuid": _rule(
            f"{_LEGACY_START}{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}"
            f"{_LEGACY_END}"
        ),
        "task_run_response_claim_id": _rule(
            f"{_LEGACY_START}(?:{_any_of(['task', 'run', 'resp', 'claim'], _LEGACY_FOLDS)})"
            f"[_-](?={_LEGACY_LETTER_OR_DIGIT}{{6}}){_LEGACY_LETTER}*[0-9]"
            f"{_LEGACY_LETTER_OR_DIGIT}*{_LEGACY_END}"
        ),
        "profile_name": _rule(
            f"{_LEGACY_START}{_cased('raphael', _LEGACY_FOLDS)}-{_LEGACY_LETTER_OR_DIGIT}"
            f"[0-9A-Z_a-z\\u0130\\u0131\\u017f\\u212a-]{{2,}}{_LEGACY_EDGE}"
        ),
        "absolute_path": _rule(
            f"(?<!{_NOT_SPACE})(?:/[.0-9A-Z_a-z-]+){{2,}}(?!{_NOT_SPACE})"
        ),
        "windows_path": _rule(
            f"{_LEGACY_START}[A-Za-z]:\\\\(?:[^\\\\{_SPACE_CHARS}]+\\\\)+"
            f"[^\\\\{_SPACE_CHARS}]+"
        ),
        "claude_model": _rule(
            f"{_LEGACY_START}{_cased('claude', _LEGACY_FOLDS)}{_SPACE}+"
            f"(?:{_any_of(['opus', 'sonnet', 'haiku'], _LEGACY_FOLDS)})"
            f"(?:{_SPACE}*{_LEGACY_DIGIT}+(?:\\.{_LEGACY_DIGIT}+)*)?{_LEGACY_END}"
        ),
        "gpt_model": _rule(
            f"{_LEGACY_START}{_cased('gpt', _LEGACY_FOLDS)}"
            f"[{_SPACE_CHARS}-]?{_LEGACY_DIGIT}{_LEGACY_TOKEN}*{_LEGACY_EDGE}"
        ),
        "versioned_model": _rule(
            f"{_LEGACY_START}(?:{_LEGACY_MODEL_FAMILIES})-{_LEGACY_TOKEN}*"
            f"{_LEGACY_DIGIT}{_LEGACY_TOKEN}*{_LEGACY_EDGE}"
        ),
        "provider_model": _rule(
            f"{_LEGACY_START}(?:{_LEGACY_PROVIDERS}){_SPACE}*[:/]{_SPACE}*"
            f"{_LEGACY_TOKEN}+{_LEGACY_EDGE}"
        ),
        "model_or_provider_name": _rule(
            f"{_LEGACY_START}(?:{_LEGACY_NAMES}){_LEGACY_END}"
        ),
        "kernel_card_id": _rule(f"[Tt]_{_HEX}{{8}}"),
        "full_commit_id": _rule(f"{_HEX}{{40}}"),
        "file_link": _rule(f"{_LEGACY_START}{_cased('file', _LEGACY_FOLDS)}:/"),
        "drive_relative_path": _rule(f"{_LEGACY_START}[A-Za-z]:(?={_NOT_SPACE})"),
        "project_id": _rule(f"[Pp]_{_HEX}{{8}}"),
        "commit_id_with_digit": _rule(f"(?<!{_HEX})(?={_HEX}{{7}})[A-Fa-f]*[0-9]"),
        "commit_id_word": _rule(
            f"(?<!{_LEGACY_ALNUM}){_HEX}{{7,}}(?!{_LEGACY_ALNUM})"
        ),
    },
}

_TEXT_GROUPS = [
    "secrets", "native_ids", "commit_ids", "host_paths", "web_addresses",
    "addresses", "exception_text", "model_names",
]


def _kind(groups, *, characters, join_wrapped_lines, word_stand_ins=False, max_length=None):
    return {
        "groups": groups,
        "characters": characters,
        "join_wrapped_lines": join_wrapped_lines,
        "word_stand_ins": word_stand_ins,
        "max_length": max_length,
        "fixed": False,
    }


_STOP_REASON = _kind(
    ["stop_reason_compatibility"], characters=True, join_wrapped_lines=False,
    word_stand_ins=True,
)
_PROSE = _kind(_TEXT_GROUPS, characters=True, join_wrapped_lines=True)

# (kind, text, whether the text passes); a refused text expects its kind's sentence.
_CASES = [
    ("title", "Ship the thing", True),
    ("title", "Caf\xe9 na\xefve \u65e5\u672c\u8a9e", True),
    ("title", "Ship the thing \U0001f680", True),
    ("title", "Plan the A/B test", False),
    ("title", "Order 12345678 for the shop", False),
    ("title", "Retry https://api.example.com/v1/sync?token=***&page=2", False),
    ("title", "Mirror https://***@git.example.com/repo.git", False),
    ("title", "x" * 240, True),
    ("title", "x" * 241, False),
    ("title", "Follow up on t_0000abcd before Friday", False),
    ("title", "Follow up on t_0000\nabcd before Friday", False),
    ("title", "Move p_0000abcd0000 into the archive", False),
    ("title", "Archive /srv/placeholder", False),
    ("title", "Archive /srv/\nplaceholder", False),
    ("title", "Open ~/notes.txt", False),
    ("title", "Email owner@example.test the plan", False),
    ("title", "Ask @placeholder_owner about the room", False),
    ("title", "Call +15550100 about the room", False),
    ("title", "Fix KeyError: 'name' in the importer", False),
    ("title", "Use Claude Opus for the draft", False),
    ("title", "   ", False),
    ("title", None, False),
    ("project_name", "Shoe Shop", True),
    ("project_name", "Caf\xe9 \U0001f680", True),
    ("project_name", "z" * 160, True),
    ("project_name", "z" * 161, False),
    ("project_name", "Archive /srv/placeholder", False),
    ("project_name", "Project p_0000abcd", False),
    ("project_name", "Use OpenAI for review", False),
    ("stop_reason_decision", "Please connect the placeholder calendar account so I can book the room.", True),
    ("stop_reason_decision", "Please confirm the placeholder plan. " * 30, True),
    ("stop_reason_decision", "Need the deploy key sk-aaaaaaaaaaaaaaaaaaaaaaaa to continue", False),
    ("stop_reason_decision", "Cannot open /srv/placeholder/config.yaml for reading", False),
    ("stop_reason_decision", "Waiting on task_1234abcd before continuing", False),
    ("stop_reason_decision", "", False),
    ("stop_reason_decision", None, False),
    ("stop_reason_decision", "Waiting for the owner to approve t_0000abcd before booking", False),
    ("stop_reason_decision", "Cannot read /srv/placeholder/config.yaml: permission denied", False),
    ("stop_reason_decision", "[Errno 2] No such file or directory: '/srv/placeholder/x'", False),
    ("stop_reason_decision", "Cannot open ~/notes.txt for reading", False),
    ("stop_reason_decision", "Should I publish the draft at commit 0000abcd0000abcd0000abcd0000abcd0000abcd?", False),
    ("stop_reason_decision", "Cannot open file:///srv/placeholder/index.html: the browser is not available", False),
    ("stop_reason_decision", "Should I move this into p_0000abcd first?", False),
    ("stop_reason_decision", "Waiting for _t_0000abcd_ to finish first", False),
    ("stop_reason_decision", "Please check notes\u29f5draft.md", False),
    ("stop_reason_decision", "Cannot open //srv/placeholder/x for reading", False),
    ("stop_reason_decision", "The upload failed (see /srv//placeholder/x)", False),
    ("stop_reason_decision", "Saved the list to srv/placeholder/list.txt, please check it", False),
    ("stop_reason_decision", "Please check config/settings.py", False),
    ("stop_reason_decision", "Cannot read config\\settings.py", False),
    ("stop_reason_decision", "Please check config/application.properties", False),
    ("stop_reason_decision", "Please check config/.sample", False),
    ("stop_reason_decision", "Please check config/archive.7z", False),
    ("stop_reason_decision", "Please check config/my notes.py", False),
    ("stop_reason_decision", "Cannot open A/B: permission denied", False),
    ("stop_reason_decision", "Rebased onto 1234567 first", False),
    ("stop_reason_decision", "Rebased onto abcdefa first", False),
    ("stop_reason_decision", "Please confirm the placeholder plan. " * 14 + "Then open /srv/placeholder/x", False),
    ("stop_reason_decision", "Cannot open A\u20e5B: permission denied", False),
    ("stop_reason_decision", "Cannot open A\u20ebB: permission denied", False),
    ("stop_reason_decision", "Cannot read C:placeholder.txt", False),
    ("stop_reason_decision", "Please confirm 1\ufe0f.", False),
    ("stop_reason_decision", "Should we book it for the fifth or the nineteenth?", True),
    ("stop_reason_decision", "Which file: the first or the second?", True),
    ("stop_reason_decision", "Should we book it for next Monday or next Tuesday?", True),
    ("stop_reason_decision", "Should the placeholder room be booked for two hours?", True),
    ("stop_reason_decision", "Should the placeholder lunch start at noon?", True),
    ("stop_reason_decision", "Should the placeholder dinner start at seven?", True),
    ("stop_reason_decision", "Open ./settings.py first", False),
    ("stop_reason_decision", "The draft is in notes\\placeholder\\drafts", False),
    ("stop_reason_decision", "Saved it as 'notes/list.txt' for you", False),
    ("stop_reason_decision", "Look in \u043a\u043e\u043d\u0444\u0438\u0433/\u0444\u0430\u0439\u043b for it", False),
    ("stop_reason_decision", "See https://example.test/help/owner for the steps", False),
    ("stop_reason_decision", "Saved the placeholder list to ~/placeholder/list.txt, please check it", False),
    ("stop_reason_decision", "The upload failed (see /srv/placeholder/x)", False),
    ("stop_reason_decision", "Should I publish file:///srv/placeholder/draft.html as it is?", False),
    ("stop_reason_decision", "Project p_0000abcd0000abcd0000abcd has no repository configured", False),
    ("stop_reason_decision", "Should we book it for 12/05 or 19/05?", False),
    ("stop_reason_decision", "Book it for 12/05/2026.", False),
    ("stop_reason_decision", "Python 3.11/3.12 both work", False),
    ("stop_reason_decision", "Use v1.2/v1.3", False),
    ("stop_reason_decision", "Plan the A/B test", False),
    ("stop_reason_decision", "Is it yes/no.", False),
    ("stop_reason_decision", "Is it and/or.", False),
    ("stop_reason_decision", "Cannot open A\u2215B: permission denied", False),
    ("stop_reason_decision", "Cannot open A\uff0fB: permission denied", False),
    ("stop_reason_decision", "The draft is in notes\uff3cdrafts", False),
    ("stop_reason_decision", "Cannot open A\u2afdB: permission denied", False),
    ("stop_reason_decision", "The draft is in notes\u27cddrafts", False),
    ("stop_reason_decision", "Cannot open A\u0338B: permission denied", False),
    ("stop_reason_decision", "Project _p_0000abcd_ has no repository", False),
    ("stop_reason_decision", "Rebuilt _0000abcd0000abcd0000abcd0000abcd0000abcd_ first", False),
    ("stop_reason_decision", "Rebased onto 0a1b2c3d first", False),
    ("stop_reason_decision", "Ask owner@example.test first", False),
    ("stop_reason_decision", "Should p_values be reported?", False),
    ("stop_reason_decision", "Waiting on ABCDEFA", False),
    ("stop_reason_decision", "Waiting on _abcdefa_", False),
    ("stop_reason_decision", "Cannot open A\u0335B: permission denied", False),
    ("stop_reason_decision", "Ask about A\u20dd first", False),
    ("stop_reason_decision", "Cannot open A\U0001d167B: permission denied", False),
    ("stop_reason_decision", "Cannot read d:notes first", False),
    ("stop_reason_decision", "Please confirm A\U000e0100.", False),
    ("stop_reason_decision", "Is it yes or no?", True),
    ("stop_reason_decision", "Book it for the fifth of December.", True),
    ("stop_reason_decision", "\u041c\u043e\u0436\u043d\u043e \u043b\u0438 \u043f\u0435\u0440\u0435\u043d\u0435\u0441\u0442\u0438 \u0432\u0441\u0442\u0440\u0435\u0447\u0443 \u2014 \xab\u043f\u043e\u0441\u043b\u0435 \u043e\u0431\u0435\u0434\u0430\xbb?", True),
    ("stop_reason_decision", "Ist der Termin f\xfcr Montag geplant?", True),
    ("stop_reason_decision", "Is 50% of the budget enough?", True),
    ("stop_reason_decision", "Should R&D approve it first?", True),
    ("stop_reason_decision", "Waiting for your feedback on the draft.", True),
    ("stop_reason_decision", "Is the Cafe\u0301 booked for noon?", True),
    ("stop_reason_decision", "Meet at 12:30 in room A.", True),
    ("stop_reason_decision", "Note: the placeholder plan is ready.", True),
    ("stop_reason_decision", "Is option A: yes or no?", True),
    ("stop_reason_decision", "\u0915\u094d\u092f\u093e \u092f\u0939 \u0920\u0940\u0915 \u0939\u0948?", True),
    ("stop_reason_decision", "Raphael needs your answer before this work can continue.", True),
    ("stop_reason_decision", "Work stopped and needs attention.", True),
    ("stop_reason_decision", "Please confirm\nthe placeholder plan.", True),
    ("stop_reason_decision", "Is sk-aaaaaaaaaaaaaaaaaaaaaaaa the right key?", False),
    ("stop_reason_decision", "Please connect the placeholder calendar account.", True),
    ("stop_reason_decision", "Tighten the placeholder section", True),
    ("stop_reason_decision", "Wait for the placeholder date", True),
    ("stop_reason_decision", "Next month", True),
    ("stop_reason_decision", "Later", True),
    ("stop_reason_decision", "Which placeholder venue should we book?", True),
    ("stop_reason_decision", "Which placeholder date?", True),
    ("stop_reason_decision", "Which placeholder menu?", True),
    ("stop_reason_decision", "Which placeholder venue?", True),
    ("stop_reason_decision", "Please connect the placeholder calendar.", True),
    ("stop_reason_decision", "Please connect the placeholder booking account.", True),
    ("stop_reason_decision", "Cannot open /srv/place\nholder for reading", False),
    ("stop_reason_decision", "Cannot open /srv/\nplaceholder for reading", False),
    ("stop_reason_decision", "Rebased onto abc\ndefa first", True),
    ("stop_reason_receipt", "Is sk-aaaaaaaaaaaaaaaaaaaaaaaa the right key?", False),
    ("stop_reason_receipt", "Please connect the placeholder calendar account.", True),
    ("stop_reason_receipt", "Cannot open /srv/place\nholder for reading", False),
    ("stop_reason_receipt", "Cannot open /srv/\nplaceholder for reading", False),
    ("stop_reason_receipt", "Please confirm\nthe placeholder plan.", True),
    ("stop_reason_receipt", "Waiting for _t_0000abcd_ to finish first", False),
    ("stop_reason_receipt", "Rebased onto abcdefa first", False),
    ("stop_reason_receipt", None, False),
    ("planner_prose", "I booked the room for Tuesday at noon.", True),
    ("planner_prose", "The plan has three steps:\nbook the room, invite the team, send the agenda.", True),
    ("planner_prose", "Ist der Termin f\xfcr Montag geplant?", True),
    ("planner_prose", "I sent the plan to owner@example.test this morning.", False),
    ("planner_prose", "I asked @placeholder_owner to confirm the room.", False),
    ("planner_prose", "I posted the plan in chat -1001234567890 for you.", False),
    ("planner_prose", "Call me on +15550100 tomorrow.", False),
    ("planner_prose", "Ping <@U012AB3CD> about the room.", False),
    ("planner_prose", "Claude Opus drafted the plan for you.", False),
    ("planner_prose", "I checked the dates with GPT-5 first.", False),
    ("planner_prose", "Anthropic wrote the first draft.", False),
    ("planner_prose", "I saved the plan to /srv/place\nholder for you.", False),
    ("planner_prose", "Ship it \U0001f680", False),
    ("stored_failure", "The calendar was busy, so the booking did not finish.", True),
    ("stored_failure", "Failed to fetch", False),
    ("stored_failure", "Traceback (most recent call last):", False),
    ("stored_failure", "[Errno 111] Connection refused", False),
    ("stored_failure", "KeyError: 'name'", False),
    ("stored_failure", 'File "main.py", line 3, in main', False),
    ("stored_failure", "at handler (server.js:10:5)", False),
    ("stored_failure", "connect ECONNREFUSED 127.0.0.1:5432", False),
    ("stored_failure", "The worker with pid 4242 stopped.", False),
    ("stored_failure", "NetworkError when attempting to fetch resource.", False),
    ("retry_reason", "Please try the booking again tomorrow.", True),
    ("retry_reason", "Retry after KeyError: 'name'", False),
    ("retry_reason", "Retry with gpt-4o", False),
    ("retry_reason", "Retry task_1234abcd tonight", False),
    ("browser_network_error", "Please try again.", False),
    ("browser_network_error", "Failed to fetch", False),
    ("browser_network_error", None, False),
    ("automation_start", "Start on Monday at nine.", True),
    ("automation_start", "Start when t_0000abcd finishes", False),
    ("automation_start", "Start at 09:00 on 2026/10/20", False),
    ("title", "Open config/settings.py", False),
    ("title", "Open (/srv/placeholder)", False),
    ("title", "Open config\u2215settings.py", False),
    ("title", "Open config\uff0fsettings.py", False),
    ("title", "Open D:notes.txt", False),
    ("title", "Use https://example.test/help", False),
    ("title", "Visit www.example.test today", False),
    ("title", "Read example.test today", False),
    ("title", "Publish revision 0a1b2c3", False),
    ("title", "Publish revision abcdefa", False),
    ("title", "Use Azure for the draft.", False),
    ("title", "Ask OpenRouter for the draft.", False),
    ("title", "OpenAI\u3067\u78ba\u8a8d\u3057\u305f\u3002", False),
    ("title", "\u78ba\u8a8dClaude\u3067", False),
    ("title", "Plan with Together AI", False),
    ("title", "Draft it with gemma-2", False),
    ("title", "Plan the trip together", True),
    ("title", "Update the Google Calendar", True),
    ("title", "Book the fireworks for Friday", True),
    ("title", "Clean the feedback list", True),
    ("title", "Meet at 12:30 in room A", True),
    ("project_name", "Open config/settings.py", False),
    ("project_name", "Archive (/srv/placeholder)", False),
    ("project_name", "Notes config\u2215settings.py", False),
    ("project_name", "https://example.test/help", False),
    ("project_name", "Revision 0a1b2c3", False),
    ("project_name", "Azure rollout", False),
    ("project_name", "OpenRouter trial", False),
    ("project_name", "OpenAI\u3067\u78ba\u8a8d", False),
    ("project_name", "Fireworks for the shop", True),
    ("project_name", "Caf\xe9 \u65e5\u672c\u8a9e", True),
    ("planner_prose", "Use Azure for the draft.", False),
    ("planner_prose", "Ask OpenRouter for the draft.", False),
    ("planner_prose", "OpenAI\u3067\u78ba\u8a8d\u3057\u305f\u3002", False),
    ("planner_prose", "Together AI wrote the first draft.", False),
    ("planner_prose", "See example.test for the steps.", False),
    ("planner_prose", "We will travel together.", True),
    ("stop_reason_decision", "OpenAI\u3067\u78ba\u8a8d\u3057\u305f\u3002", True),
    ("stop_reason_decision", "Use Azure for the draft.", True),
]

OWNER_TEXT_POLICY: dict[str, Any] = {
    "engines": {
        "flags": (
            "Every pattern's flags are \"u\": JavaScript compiles it with "
            "new RegExp(pattern, \"u\"); Python compiles it with re.compile(pattern) "
            "and no flag."
        ),
    },
    "rules": {
        # A kind that checks characters lets through only letters, marks and
        # digits of any script, spaces and plain punctuation, so a slash, a
        # backslash, every look-alike of either and every other symbol (emoji
        # among them) makes the text fall back. Accents are composed into
        # their letters first; a mark that remains is refused when it encloses
        # what precedes it, when its Unicode name says COMBINING (every
        # overlay) or VARIATION SELECTOR, or when it has no name.
        "characters": {
            "normalization": "NFC",
            "allowed_categories": ["Ll", "Lm", "Lo", "Lt", "Lu", "Nd", "Nl", "No", "Zl", "Zp", "Zs"],
            "allowed_characters": (
                ".,;:!?'\"()-%&\u00ab\u00bb\u2013\u2014\u2018\u2019\u201c\u201d\u2026"
                "\u3001\u3002\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f"
                "\t\n\u000b\f\r\u001c\u001d\u001e\u001f\u0085"
            ),
            "mark_categories": ["Mc", "Mn"],
            "refused_mark_names": ["COMBINING", "VARIATION SELECTOR"],
        },
        "line_break": "\\r\\n|[\\n\\r\\u000b\\u000c\\u0085\\u2028\\u2029]",
        "wrapped_line_break": (
            "[ \\t\\u00ad]*(?:\\r\\n|[\\n\\r\\u000b\\u000c\\u0085\\u2028\\u2029])[ \\t\\u00ad]*"
        ),
        # Before the patterns of a kind with "word_stand_ins" run, each
        # character outside ASCII and outside "kept" is replaced: one of the
        # "digit_categories" by "digit", one of the "letter_categories" by
        # "letter"; every other character stays. Only the check uses this form.
        "word_stand_ins": {
            "kept": "\u0130\u0131\u017f\u212a",
            "digit": "\u0660",
            "digit_categories": ["Nd"],
            "letter": "\u00aa",
            "letter_categories": ["Ll", "Lm", "Lo", "Lt", "Lu", "Nl", "No"],
        },
        "groups": _GROUPS,
        "kinds": {
            "title": _kind(
                _TEXT_GROUPS, characters=False, join_wrapped_lines=True, max_length=240
            ),
            "project_name": _kind(
                _TEXT_GROUPS, characters=False, join_wrapped_lines=True, max_length=160
            ),
            "stop_reason_decision": _STOP_REASON,
            "stop_reason_receipt": _STOP_REASON,
            "planner_prose": _PROSE,
            "stored_failure": _PROSE,
            "retry_reason": _PROSE,
            "browser_network_error": {**_PROSE, "fixed": True},
            "automation_start": _PROSE,
        },
        "model_and_provider_names": {
            "bare": _BARE_NAMES,
            "with_qualifier": _QUALIFIED_NAMES,
            "with_version": _VERSIONED_FAMILIES,
            "why_some_need_a_qualifier": (
                "Each of these is also an everyday word or a common name "
                "(together, fireworks, vertex, upstage, moonshot, meta, google, "
                "alibaba, gmi, xiaomi, copilot, nous, kimi, hugging face), so it is "
                "refused only with its qualifier or a version."
            ),
        },
    },
    "sentences": _SENTENCES,
    "cases": [
        {"kind": kind, "text": text, "expected": text if passes else _SENTENCES[kind]}
        for kind, text, passes in _CASES
    ],
}

_RULES = OWNER_TEXT_POLICY["rules"]
_CHARACTERS = _RULES["characters"]
_ALLOWED_CATEGORIES = frozenset(_CHARACTERS["allowed_categories"])
_ALLOWED_CHARACTERS = frozenset(_CHARACTERS["allowed_characters"])
_MARK_CATEGORIES = frozenset(_CHARACTERS["mark_categories"])
_REFUSED_MARK_NAMES = tuple(_CHARACTERS["refused_mark_names"])
_STAND_INS = _RULES["word_stand_ins"]
_LINE_BREAK = re.compile(_RULES["line_break"])
_WRAPPED_LINE_BREAK = re.compile(_RULES["wrapped_line_break"])


def _compile(rule: dict[str, str]) -> re.Pattern[str]:
    if rule["flags"] != "u":
        raise ValueError(f"unsupported owner-text pattern flags: {rule['flags']!r}")
    return re.compile(rule["pattern"])


_KINDS = {
    kind: (
        spec,
        tuple(
            _compile(rule)
            for group in spec["groups"]
            for rule in _RULES["groups"][group].values()
        ),
    )
    for kind, spec in _RULES["kinds"].items()
}


def owner_text_policy_json() -> str:
    """Return :data:`OWNER_TEXT_POLICY` as canonical JSON text: keys sorted,
    no spaces between items, characters written as themselves."""
    return json.dumps(
        OWNER_TEXT_POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _plain_characters(text: str) -> bool:
    """Whether every character of ``text`` is one the policy lets through."""
    for char in unicodedata.normalize(_CHARACTERS["normalization"], text):
        category = unicodedata.category(char)
        if category in _ALLOWED_CATEGORIES or char in _ALLOWED_CHARACTERS:
            continue
        name = unicodedata.name(char, "")
        if (
            category in _MARK_CATEGORIES
            and name
            and not any(word in name for word in _REFUSED_MARK_NAMES)
        ):
            continue
        return False
    return True


def _with_word_stand_ins(text: str) -> str:
    """``text`` with the policy's word stand-ins in place."""
    return "".join(
        char
        if char.isascii() or char in _STAND_INS["kept"]
        else _STAND_INS["digit"]
        if unicodedata.category(char) in _STAND_INS["digit_categories"]
        else _STAND_INS["letter"]
        if unicodedata.category(char) in _STAND_INS["letter_categories"]
        else char
        for char in text
    )


def check_owner_text(text: Any, kind: str, *, bounded: bool = True) -> str:
    """Return ``text`` when the policy lets the owner see it as ``kind``.

    Otherwise return the kind's fixed sentence, whole: for text that is not a
    string or is blank, for every text of a ``fixed`` kind, and for text longer
    than the kind allows, holding a character the kind refuses or matching a
    rule in one of the kind's groups. A kind that joins wrapped lines checks the
    text with each line break closed up and again with each shown as a space,
    so a path or id split across lines is still caught. ``bounded=False`` skips
    the length bound only, for a caller that checks a whole input it then cuts
    to the bound itself. An unknown ``kind`` raises :class:`KeyError`.
    """
    spec, patterns = _KINDS[kind]
    sentence = _SENTENCES[kind]
    if spec["fixed"] or not isinstance(text, str) or not text.strip():
        return sentence
    limit = spec["max_length"]
    if bounded and limit is not None and len(text) > limit:
        return sentence
    forms = [text]
    if spec["join_wrapped_lines"]:
        forms = [_WRAPPED_LINE_BREAK.sub("", text), _LINE_BREAK.sub(" ", text)]
    for form in forms:
        if spec["characters"] and not _plain_characters(form):
            return sentence
        if spec["word_stand_ins"]:
            form = _with_word_stand_ins(form)
        if any(pattern.search(form) for pattern in patterns):
            return sentence
    return text
