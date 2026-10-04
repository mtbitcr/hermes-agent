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
