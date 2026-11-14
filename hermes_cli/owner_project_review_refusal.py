"""Shared refusal detail of the daily reviews: the log tokens and the sentences of a refusal, built only from kernel-made
fields, and the claim rule that the harvest and the hand-over both run against the frozen input.

Pure functions. They import nothing from the review modules, and they print only field paths, numbers and kernel ids.
"""

from __future__ import annotations

from typing import Any, Mapping

_TEXT_CODES = ("review_text_length_invalid", "review_text_policy_invalid")
_COVERAGE_CODE = "review_class_coverage_invalid"
_PORTFOLIO_TEXT_CODE = "portfolio_result_text_invalid"
_PORTFOLIO_LIST_CODE = "portfolio_result_list_invalid"
_MAX_IDS = 10
_MAX_SENTENCES = 12
_STILL_RUNNING = "Your task is still running. Correct the result and call kanban_complete again."
_SENTENCES = {
    "review_result_keys_invalid": "project_review must hold exactly the keys of the shape in the task instructions.",
    "review_result_too_large": "The result is larger than the kernel accepts. Shorten it.",
    "review_result_data_invalid": "The result must be plain JSON data.",
    "review_text_keys_invalid": "summary and brief must hold exactly the keys of the shape in the task instructions.",
    "review_text_length_invalid": "A text of the result is empty, too long or not text.",
    "review_text_policy_invalid": "A text of the result breaks the owner text rule. Write it again in plain words.",
    "review_brief_generic": "The brief headline is too general, or what_changed names no work title from the input.",
    "review_class_list_invalid": "classification must be a list.",
    "review_class_evidence_invalid": (
        "Each classification must hold exactly item, class and evidence, name its item once, use a class from the task "
        "instructions and cite only references from the input."
    ),
    "review_class_coverage_invalid": "Classify each expected input item exactly once, and no other item.",
    "review_closure_list_invalid": "proposed_closures must be a list with no more closures than the task instructions allow.",
    "review_closure_keys_invalid": (
        "Each proposed closure must hold exactly recommendation_id, card_id, reason and evidence, with text ids."
    ),
    "portfolio_result_keys_invalid": "portfolio_review must hold exactly conflicts, priorities and shared_blockers.",
    "portfolio_result_list_invalid": "Each list of the result must be a list within the task instructions' limit.",
    "portfolio_result_text_invalid": "A line of the result is empty, too long, not plain owner text or has no Project name.",
}


def _ids(items: Any) -> str:
    items = [str(item) for item in items]
    if not items:
        return "none"
    shown = ",".join(items[:_MAX_IDS])
    return shown if len(items) <= _MAX_IDS else f"{shown},+{len(items) - _MAX_IDS}"


def _breach(path: str, breach: Mapping[str, Any]) -> str:
    token = f"{breach[path]}:{breach['rule']}"
    if breach["rule"] == "length" and "length" in breach and "limit" in breach:
        token += f":{int(breach['length'])}/{int(breach['limit'])}"
    return token


def refusal_tokens(code: str, detail: Mapping[str, Any]) -> str:
    """The ``key=value`` tokens that name the field, limit or items of a refusal; empty for a detail it does not know."""
    try:
        if code in _TEXT_CODES and detail.get("fields"):
            return "fields=" + ",".join(_breach("field", item) for item in detail["fields"])
        if code == _COVERAGE_CODE and "missing" in detail:
            return f"missing={_ids(detail['missing'])} extra={_ids(detail['extra'])} extra_other={int(detail['extra_other'])}"
        if code == _PORTFOLIO_TEXT_CODE and detail.get("lines"):
            return "lines=" + ",".join(_breach("line", item) for item in detail["lines"])
        if code == _PORTFOLIO_LIST_CODE and detail.get("lists"):
            return "lists=" + ",".join(
                f"{item['list']}:{int(item['count'])}/{int(item['limit'])}" if "count" in item else f"{item['list']}:type"
                for item in detail["lists"]
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return ""
    return ""


def _listed(items: Any) -> str:
    items = [str(item) for item in items]
    shown = ", ".join(items[:_MAX_IDS])
    return shown if len(items) <= _MAX_IDS else f"{shown} and {len(items) - _MAX_IDS} more in refusal.detail"


def _line_name(path: Any) -> str:
    """A kernel line path such as priorities[1], told as priorities line 2."""
    name, _, index = str(path).partition("[")
    return f"{name} line {int(index.rstrip(']')) + 1}"


def _breach_sentence(name: str, breach: Mapping[str, Any]) -> str:
    rule = breach["rule"]
    if rule == "length" and int(breach["length"]) < 1:
        return f"{name} must not be empty."
    if rule == "length":
        return f"{name} has {int(breach['length'])} characters; at most {int(breach['limit'])}."
    if rule == "type":
        return f"{name} must be text."
    if rule == "owner_text":
        return f"{name} breaks the owner text rule. Write it again in plain words."
    if rule == "project_prefix":
        return f"{name} must start with one of the input Project names, a colon and a space."
    raise ValueError(rule)


def _detail_sentences(code: str, detail: Mapping[str, Any]) -> list[str]:
    if code in _TEXT_CODES and detail.get("fields"):
        return [_breach_sentence(str(item["field"]), item) for item in detail["fields"]]
    if code == _COVERAGE_CODE and "missing" in detail:
        sentences = []
        if detail["missing"]:
            sentences.append(f"Classify each of these items exactly once: {_listed(detail['missing'])}.")
        if detail["extra"]:
            sentences.append(f"Remove these items: {_listed(detail['extra'])}.")
        other = int(detail["extra_other"])
        if other:
            sentences.append(f"Remove {other} other classified item{'' if other == 1 else 's'} not found in the input.")
        return sentences
    sentences = []
    if code == _PORTFOLIO_LIST_CODE:
        for item in detail.get("lists") or []:
            if "count" in item:
                sentences.append(f"{item['list']} has {int(item['count'])} lines; at most {int(item['limit'])}.")
            else:
                sentences.append(f"{item['list']} must be a list.")
    if code in (_PORTFOLIO_LIST_CODE, _PORTFOLIO_TEXT_CODE):
        sentences += [_breach_sentence(_line_name(item["line"]), item) for item in detail.get("lines") or []]
    return sentences


def refusal_sentences(code: str, detail: Mapping[str, Any]) -> str:
    """Plain sentences that tell a reviewer what to correct: field paths, numbers and kernel ids, never its own text."""
    try:
        sentences = _detail_sentences(code, detail or {})
    except (AttributeError, KeyError, TypeError, ValueError):
        sentences = []
    if len(sentences) > _MAX_SENTENCES:
        sentences = sentences[:_MAX_SENTENCES] + [f"{len(sentences) - _MAX_SENTENCES} more are in refusal.detail."]
    if not sentences:
        sentences = [_SENTENCES.get(str(code), "The result breaks a rule of the task instructions.")]
    return " ".join([f"The result was refused ({code}).", *sentences, _STILL_RUNNING])


def class_history(sources: Mapping[str, Any], classes: Any, refused: type) -> list:
    """The first earlier review's classification, every item a known class. ``refused`` is the facade's refusal."""
    previous = sources["earlier_reviews"][0].get("classification", []) if sources["earlier_reviews"] else []
    if not isinstance(previous, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("item"), str)
        or not isinstance(item.get("class"), str) or item["class"] not in classes
        for item in previous
    ):
        raise refused("review_class_history_invalid")
    return previous


def claim_coverage(sources: Mapping[str, Any], claims: Any, classes: Any, refused: type) -> tuple[set, dict, set]:
    """The claim structure and coverage against the frozen input alone, raised as ``refused``: one rule, read by the
    harvest through ``check_claims`` of the facade and by the hand-over."""
    suggestions = {item["id"]: item for item in sources["suggestions"]}
    decisions = sources["decisions"]
    previous = class_history(sources, classes, refused)
    expected = set(suggestions) | {item["task_id"] for item in decisions if item.get("task_id")}
    expected.update(item["item"] for item in previous if item["class"] == "still actionable")
    references = {f"tasks:{item['id']}" for item in sources["tasks"]}
    references.update(f"suggestions:{tid}" for tid in suggestions)
    references.update(f"proposals:{item['id']}" for item in sources.get("proposals", []))
    references.update(
        f"decisions:event:{item['event_id']}" if "event_id" in item else f"decisions:wait:{item['task_id']}"
        for item in decisions if "event_id" in item or item.get("task_id")
    )
    if not isinstance(claims, list):
        raise refused("review_class_list_invalid")
    by_item = {}
    for claim in claims:
        if (not isinstance(claim, dict) or set(claim) != {"item", "class", "evidence"}
                or not isinstance(claim["item"], str) or claim["item"] in by_item
                or not isinstance(claim["class"], str) or claim["class"] not in classes
                or not isinstance(claim["evidence"], list)
                or not all(isinstance(ref, str) and ref in references for ref in claim["evidence"])):
            raise refused("review_class_evidence_invalid")
        by_item[claim["item"]] = claim
    if set(by_item) != expected:
        known = {item["id"] for item in sources["tasks"]} | set(suggestions)
        extra = set(by_item) - expected
        raise refused("review_class_coverage_invalid", {
            "missing": sorted(expected - set(by_item)),
            "extra": sorted(item for item in extra if item in known),
            "extra_other": sum(item not in known for item in extra),
        })
    return expected, by_item, references
