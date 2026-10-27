"""A work-item title naming the product Claude Design reads to the owner.

The owner-text policy lets Claude Design through (owner decision of
2026-10-26); the private work-item patterns must not refuse it, and must keep
refusing every other form of the word Claude.
"""

import pytest

from hermes_cli import owner_workspace as ow

_TITLE = "Draw the first screens in Claude Design"
_REFUSED = "This piece of work has a title Raphael cannot show here."

_REFUSED_TITLES = {
    "alone": _TITLE.replace("Claude Design", "Claude"),
    "version": _TITLE.replace("Claude Design", "Claude 4"),
    "model-family": _TITLE.replace("Claude Design", "Claude Opus"),
    "longer-word": _TITLE.replace("Claude Design", "Claude Designer"),
    "two-spaces": _TITLE.replace("Claude Design", "Claude  Design"),
    "tab": _TITLE.replace("Claude Design", "Claude\tDesign"),
    "line-break": _TITLE.replace("Claude Design", "Claude\nDesign"),
    "invisible-joint": _TITLE.replace("Claude Design", "Claude \u200bDesign"),
    "beside-another": _TITLE + " and Claude",
    # Display breaks the first name and makes a second out of the other one.
    "name-lost-and-made": _TITLE.replace(
        "Claude Design", "e\u200bClaude Design and Claude  Design"
    ),
    "name-glued-by-display": _TITLE.replace("Claude Design", "Claude Design\u200bs"),
}


def test_owner_title_gives_back_a_title_naming_claude_design():
    assert ow.owner_title(_TITLE) == _TITLE


def test_native_owner_title_stores_a_title_naming_claude_design():
    assert ow._native_owner_title(_TITLE, "changes[0].title") == _TITLE


def test_claude_design_reads_while_every_other_form_of_claude_is_refused():
    assert ow.owner_title(_TITLE) == _TITLE
    assert ow._native_owner_title(_TITLE, "title") == _TITLE
    for form, title in _REFUSED_TITLES.items():
        assert ow.owner_title(title) == _REFUSED, form
        with pytest.raises(ow.OwnerWorkspaceError):
            ow._native_owner_title(title, "title")
