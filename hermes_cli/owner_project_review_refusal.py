"""Shared refusal detail of the daily reviews: the log tokens of a refusal, built only from kernel-made fields.

Pure functions. They import nothing from the review modules, and they print only field paths, numbers and kernel ids.
"""

from __future__ import annotations

from typing import Any, Mapping

_TEXT_CODES = ("review_text_length_invalid", "review_text_policy_invalid")
_COVERAGE_CODE = "review_class_coverage_invalid"
_PORTFOLIO_TEXT_CODE = "portfolio_result_text_invalid"
_PORTFOLIO_LIST_CODE = "portfolio_result_list_invalid"
_MAX_IDS = 10


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
