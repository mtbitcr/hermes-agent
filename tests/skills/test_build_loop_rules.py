"""Behavioral contract for the build-loop additions to the builder's rule file.

The rule file `running-build-work/SKILL.md` carries four additions: a point
table item and a boundary-case item in the prompt-file list, one sentence at
the end of step 6, and a section that tells when to ask the owner. These tests
hold each text exactly, at its place, and hold that no other byte changed.
"""

import hashlib
import re
from pathlib import Path

import toolsets


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = (
    REPO_ROOT
    / "role-skills"
    / "raphael-claude-worker"
    / "team-operations"
    / "running-build-work"
    / "SKILL.md"
)

ORIGINAL_SHA256 = "a0489c31f4ea2f9fb873b9729b9d4804b29e752c40373e80e506311464441a77"

ADDITION_A = (
    '- Add "Before the handover, write `/workspace/task/POINT_TABLE.md`: each numbered '
    "point of the facts file and of the card's acceptance, each with the name of the "
    "test that holds it or 'not held: <reason>'. A test that holds the opposite of a "
    'point is a defect to fix before the handover."'
)

ADDITION_B = (
    '- Add "For each new input or read, write a focused test for each case that '
    "applies: negative, zero, empty, missing, a read that raises, a write that fails "
    "after a successful open, multibyte and NUL text, and a record stored by the "
    "previous version. List each case that does not apply in the point table.\""
)

ADDITION_C = (
    "Before the completion, export `/workspace/task/POINT_TABLE.md` as one more "
    "attachment of the same run with the call of step 5, and name its attachment id "
    "in the completion metadata as `point_table`; when the file does not exist, set "
    "`point_table` to `none`."
)

ADDITION_D_HEADING = "## When the card needs the owner"

ADDITION_D_PARAGRAPH = (
    "When the card cannot go on without a decision of the owner, do not complete and "
    "do not guess. That is the case when the card or its facts file says that the "
    "build stops, when two rules of the card contradict each other, or when a file "
    "outside the owned paths must change. First write the technical detail, such as "
    "the name of the test or of the file, as a comment on the card. Then call "
    "`kanban_ask_owner` one time: the question in plain words for an owner who is not "
    "a programmer, with what stopped the work; at most four choices, each one line "
    "that says what happens next; no file path, no card id and no model name in the "
    "question or in a choice. The call holds the card for the owner, so end the turn "
    "after it. Use `kanban_block` only for a stop that is not a question to the owner."
)

LAST_ITEM_BEFORE_ADDITIONS = (
    "- When the change adds or touches a kernel tool that resolves a store path"
)
STEP_6_START = "6. Complete with `kanban_complete`"
STEP_6_OLD_END = "a review request alone leaves the branch without the change."
LIST_HEADING = "## The prompt-file list"
PITFALLS_HEADING = "## Pitfalls seen on the build machine"

BOUNDARY_CASES = (
    "negative",
    "zero",
    "empty",
    "missing",
    "a read that raises",
    "a write that fails after a successful open",
    "multibyte and NUL text",
    "a record stored by the previous version",
)


def _text() -> str:
    return SKILL.read_bytes().decode("utf-8")


def _lines() -> list[str]:
    return _text().split("\n")


def _prompt_file_items() -> list[str]:
    lines = _lines()
    start = lines.index(LIST_HEADING) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("## "))
    return [line for line in lines[start:end] if line.startswith("- ")]


def _step_6() -> str:
    steps = [line for line in _lines() if line.startswith(STEP_6_START)]
    assert len(steps) == 1
    return steps[0]


def test_builder_skill_asks_for_the_point_table():
    items = _prompt_file_items()
    assert items[-2] == ADDITION_A
    assert items[-3].startswith(LAST_ITEM_BEFORE_ADDITIONS)
    assert items[-1] == ADDITION_B

    step_6 = _step_6()
    assert step_6.endswith(f"{STEP_6_OLD_END} {ADDITION_C}")

    text = _text()
    assert text.count(ADDITION_A) == 1
    assert text.count(ADDITION_C) == 1


def test_builder_skill_names_every_boundary_case():
    items = _prompt_file_items()
    assert items[-1] == ADDITION_B
    for case in BOUNDARY_CASES:
        assert case in items[-1], case
    assert _text().count(ADDITION_B) == 1


def test_builder_skill_tells_when_to_ask_the_owner():
    lines = _lines()
    pitfalls = lines.index(PITFALLS_HEADING)
    assert lines[pitfalls - 5 : pitfalls] == [
        "",
        ADDITION_D_HEADING,
        "",
        ADDITION_D_PARAGRAPH,
        "",
    ]
    assert lines.count(ADDITION_D_HEADING) == 1
    assert _text().count(ADDITION_D_PARAGRAPH) == 1

    tool_names = set(re.findall(r"`(kanban_[a-z_]+)`", ADDITION_D_PARAGRAPH))
    assert tool_names == {"kanban_ask_owner", "kanban_block"}
    kanban_tools = toolsets.TOOLSETS["kanban"]["tools"]
    for name in tool_names:
        assert name in kanban_tools, name


def test_builder_skill_changed_only_by_the_build_loop_additions():
    text = _text()

    for addition in (
        f"{ADDITION_A}\n{ADDITION_B}\n",
        f" {ADDITION_C}\n",
        f"{ADDITION_D_HEADING}\n\n{ADDITION_D_PARAGRAPH}\n\n",
    ):
        assert text.count(addition) == 1

    remainder = text.replace(f"{ADDITION_A}\n{ADDITION_B}\n", "", 1)
    remainder = remainder.replace(f" {ADDITION_C}\n", "\n", 1)
    remainder = remainder.replace(
        f"{ADDITION_D_HEADING}\n\n{ADDITION_D_PARAGRAPH}\n\n", "", 1
    )
    assert hashlib.sha256(remainder.encode("utf-8")).hexdigest() == ORIGINAL_SHA256
