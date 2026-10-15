"""The one owner-text policy and its checker.

``owner_text_policy.json`` beside this module holds the policy: the detection
rules, the fixed sentences of plan section D and the shared cases that pin
both. The owner app keeps an exact copy of that file, and its CI compares the
copy with this one at the pinned platform commit. The file is read once, at
import, and :func:`check_owner_text` is the one way to apply it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

_POLICY = json.loads(
    Path(__file__).with_name("owner_text_policy.json").read_text(encoding="utf-8")
)
_RULES = _POLICY["rules"]
_SENTENCES: dict[str, str] = _POLICY["sentences"]
# A kind that checks characters lets through only letters, marks and digits of
# any script, spaces and plain punctuation, so a slash, a backslash, every
# look-alike of either and every other symbol makes the text fall back. Accents
# are composed into their letters first; a mark that remains is refused when it
# encloses what precedes it, when its Unicode name says COMBINING (every
# overlay) or VARIATION SELECTOR, or when it has no name, so what stays is a
# script's own vowel signs and the like.
_CHARACTERS = _RULES["characters"]
_ALLOWED_CATEGORIES = frozenset(_CHARACTERS["allowed_categories"])
_ALLOWED_CHARACTERS = frozenset(_CHARACTERS["allowed_characters"])
_MARK_CATEGORIES = frozenset(_CHARACTERS["mark_categories"])
_REFUSED_MARK_NAMES = tuple(_CHARACTERS["refused_mark_names"])
_LINE_BREAK = re.compile(_RULES["line_break"])
_WRAPPED_LINE_BREAK = re.compile(_RULES["wrapped_line_break"])
_PATTERNS = {
    group: tuple(
        re.compile(
            rule["pattern"], re.IGNORECASE if "i" in rule.get("flags", "") else 0
        )
        for rule in rules.values()
    )
    for group, rules in _RULES["groups"].items()
}
_KINDS = {
    kind: (
        spec,
        tuple(
            pattern
            for group in spec.get("groups", ())
            for pattern in _PATTERNS[group]
        ),
    )
    for kind, spec in _RULES["kinds"].items()
}


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


def check_owner_text(text: Any, kind: str) -> str:
    """Return ``text`` when the policy lets the owner see it as ``kind``.

    Otherwise return the kind's fixed sentence, whole: for text that is not a
    string or is blank, for every text of a ``fixed`` kind, and for text longer
    than the kind allows, holding a character the kind refuses or matching a
    rule in one of the kind's groups. A kind that joins wrapped lines checks the
    text with each line break closed up and again with each shown as a space,
    so a path or id split across lines is still caught. An unknown ``kind``
    raises :class:`KeyError`.
    """
    spec, patterns = _KINDS[kind]
    sentence = _SENTENCES[kind]
    if spec.get("fixed") or not isinstance(text, str) or not text.strip():
        return sentence
    limit = spec.get("max_length")
    if limit is not None and len(text) > limit:
        return sentence
    forms = [text]
    if spec.get("join_wrapped_lines"):
        forms = [_WRAPPED_LINE_BREAK.sub("", text), _LINE_BREAK.sub(" ", text)]
    for form in forms:
        if spec.get("characters") and not _plain_characters(form):
            return sentence
        if any(pattern.search(form) for pattern in patterns):
            return sentence
    return text
