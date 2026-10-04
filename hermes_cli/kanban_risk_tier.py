"""The risk tier rules: one copy, shared by the authority, the apply path and
the kernel.

An approved plan states a risk tier for every card it creates or replaces: the
integer 0, 1 or 2, never a boolean or a string. The tier is stored with the
card (``tasks.risk_tier``). A card without a recorded tier -- every card from
before the tier existed, and a card made outside an approved plan -- holds
null, counts as tier 2 and is treated as high risk (owner answer 3).
"""

from __future__ import annotations

from typing import Any, Optional

RISK_TIERS = (0, 1, 2)
# The tier a card without a recorded tier counts as (owner answer 3).
UNRECORDED_RISK_TIER = 2
# What a card without a recorded tier reads (owner answer 3).
RISK_NOT_RECORDED = "Risk not recorded. Treated as high risk."


def parse_risk_tier(value: Any) -> int:
    """Return *value* as a risk tier, or raise ``ValueError``.

    Only the integers 0, 1 and 2 are tiers. A boolean is refused although
    Python counts it as an integer, and so is every string, float and null.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value not in RISK_TIERS:
        raise ValueError("risk_tier must be the integer 0, 1 or 2")
    return value


def effective_risk_tier(value: Optional[int]) -> int:
    """The tier a card counts as: its own, or tier 2 when none is recorded."""
    if value is None:
        return UNRECORDED_RISK_TIER
    return parse_risk_tier(value)


# The only efforts a card is ever pinned at. The model policy admits exactly
# these next to each other on a lane's own model, and nothing wider.
PINNED_EFFORTS = ("high", "max")
# Responsibility "Security review" runs at max whatever its tier (decision 6).
SECURITY_REVIEW_RESPONSIBILITY = "R12"


def pinned_reasoning_effort(tier: Optional[int], responsibility: Optional[str]) -> str:
    """The effort a new card is pinned at: max for tier 2 and for a security
    review, high for everything else."""
    if effective_risk_tier(tier) == 2:
        return "max"
    if str(responsibility or "").strip().upper() == SECURITY_REVIEW_RESPONSIBILITY:
        return "max"
    return "high"


# Seconds per run for each kind of work (the plan's owner summary and
# decision 2).
_TIME_BOX_SECONDS = {
    "build": 7200,
    "analysis": 2700,
    "proposal": 1800,
    "review": 2700,
    "release": 2700,
    "coordinator_deep": 2700,
    "coordinator_routine": 1800,
}
_COORDINATOR = "default"
_REVIEWER = "raphael-verifier"
# The builder integrates verified work and operates infrastructure, so its
# cards are release, integration and infrastructure work.
_RELEASE = "raphael-builder"


def card_work_kind(
    assignee: Optional[str], owned_paths: Optional[list], execution_tier: Optional[str],
) -> str:
    """Classify a card's work for its time box.

    The role decides first: the coordinator, the independent reviewer and the
    builder have their own boxes. Any other card is a build when it may write
    (``owned_paths`` None is legacy whole-repository ownership), and read-only
    work is an analysis when deep and a proposal when routine.
    """
    deep = str(execution_tier or "").strip().lower() == "deep"
    role = str(assignee or "").strip()
    if role == _COORDINATOR:
        return "coordinator_deep" if deep else "coordinator_routine"
    if role == _REVIEWER:
        return "review"
    if role == _RELEASE:
        return "release"
    if owned_paths is not None and not list(owned_paths):
        return "analysis" if deep else "proposal"
    return "build"


def pinned_time_box_seconds(kind: str) -> int:
    """The per-run time box, in seconds, for one kind of work."""
    try:
        return _TIME_BOX_SECONDS[kind]
    except KeyError:
        raise ValueError(f"unknown kind of work {kind!r}") from None
