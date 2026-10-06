"""The risk tier rules: one copy, shared by the authority, the apply path and
the kernel.

An approved plan states a risk tier for every card it creates or replaces: the
integer 0, 1 or 2, never a boolean or a string. The tier is stored with the
card (``tasks.risk_tier``). A card without a recorded tier -- every card from
before the tier existed, and a card made without a route lock -- holds
null, counts as tier 2 and is treated as high risk (owner answer 3).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

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


def check_raise(current: Optional[int], proposed: Any) -> Optional[int]:
    """The tier a reviewer's raise records, or None when the card already
    counts as *proposed*. A tier may only be raised: a lower one, or anything
    that is not a tier, raises ``ValueError``."""
    tier = parse_risk_tier(proposed)
    counted = effective_risk_tier(current)
    if tier < counted:
        raise ValueError(
            f"risk_tier may only be raised: the card counts as tier {counted}, "
            f"so tier {tier} would lower it"
        )
    return tier if tier > counted else None


def highest_risk_tier(tiers: Iterable[Optional[int]]) -> int:
    """The highest tier among *tiers*, a card without one counting as tier 2.

    The floor of a replacement, split or merge (the cards it replaces) and the
    tier of a new Project's root card (its tasks).
    """
    return max(effective_risk_tier(tier) for tier in tiers)


def recovered_root_risk_tier(recorded: Optional[int], derived: int) -> int:
    """The tier a new Project's root card, found already written when its
    commit is replayed, is verified at.

    The root is created at *derived*, the highest tier of its tasks. The code
    before that created it without a tier, which the creation pin records as
    tier 2 with tier 2's effort and seal. A root recorded at tier 2 is
    verified at tier 2, so that root is accepted exactly as written and never
    rewritten; any other root is verified at *derived*.
    """
    if recorded == UNRECORDED_RISK_TIER:
        return UNRECORDED_RISK_TIER
    return derived


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


def raised_reasoning_effort(
    pinned_effort: Optional[str], tier: int, responsibility: Optional[str],
) -> Optional[str]:
    """The effort a raise to *tier* re-pins the role holding a card at, or None
    when it re-pins nothing.

    A card pinned at creation (its ``pinned_effort`` recorded) runs every later
    run at the effort its new tier pins, whichever verdict follows the raise
    (decision 1). A card without that record keeps each role's own effort.
    """
    if pinned_effort is None:
        return None
    return pinned_reasoning_effort(tier, responsibility)


# Seconds per run for each kind of work (the plan's owner summary and
# decision 2).
_TIME_BOX_SECONDS = {
    "build": 7200,
    "analysis": 2700,
    "proposal": 1800,
    "review": 5400,
    "release": 2700,
    "coordinator_deep": 2700,
    "coordinator_routine": 1800,
}
_COORDINATOR = "default"
_REVIEWER = "raphael-verifier"
# The builder integrates verified work and operates infrastructure, so its
# cards are release, integration and infrastructure work.
_RELEASE = "raphael-builder"
# The roles that only read: the planner shapes the plan and the reviewer is the
# review, so neither is work awaiting a review.
_READ_ONLY = frozenset({"raphael-planner", _REVIEWER})


def card_work_kind(
    assignee: Optional[str], owned_paths: Optional[list], execution_tier: Optional[str],
    integrates_parent_heads: bool = False,
) -> str:
    """Classify a card's work for its time box.

    A card marked to integrate its parents' heads is integration work,
    whichever profile runs it. Otherwise the role decides first: the
    coordinator, the independent reviewer and the builder have their own
    boxes. Any other card is a build when it may write
    (``owned_paths`` None is legacy whole-repository ownership), and read-only
    work is an analysis when deep and a proposal when routine.
    """
    deep = str(execution_tier or "").strip().lower() == "deep"
    role = str(assignee or "").strip()
    if integrates_parent_heads:
        return "release"
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


def review_run_box_seconds(card_box: Optional[int], route_locked: bool) -> Optional[int]:
    """The time box of one review run of a card, pinned when it is claimed.

    Every review run of a pinned card gets the review box (decision 2), while
    the card keeps its own box for a handback to its implementer. A card
    without a route lock, or without a box, keeps what it has today.
    """
    if route_locked and card_box is not None:
        return _TIME_BOX_SECONDS["review"]
    return card_box


def active_run_box_seconds(
    card_box: Optional[int], run_box: Optional[int], review_run: bool,
) -> Optional[int]:
    """The time box a running attempt is held to.

    A review run is held to the box pinned on it when it was claimed; any
    other run, and a review run that carries no box, to its card's box.
    """
    if review_run and run_box is not None:
        return run_box
    return card_box


def requires_independent_review(assignee: Optional[str], kind: str) -> bool:
    """Whether a new card's work is independently reviewed before it is done.

    Every build is (decision 4): work saved at two hours goes to the reviewer
    and is not retried. A role that only reads never is.
    """
    return kind == "build" and str(assignee or "").strip() not in _READ_ONLY
