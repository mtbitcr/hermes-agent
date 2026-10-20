"""Keep the platform-written run receipts when a completed result is edited.

Owner decision of 2026-10-15: a result edit never rewrites the receipts the
platform writes on a run (the session link, the model and capability receipt
and both code-change receipts). An edit changes only the ordinary keys.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Written by the kernel alone; an edit keeps their stored value or leaves them absent.
KERNEL_KEYS = (
    "runtime_receipt",
    "worker_session_id",
    "execution_receipt",
    "worktree_materialization",
)


def _stored_dict(stored: Any) -> dict:
    """The stored metadata as a dict; absent, unreadable or non-object reads as empty."""
    if isinstance(stored, (str, bytes, bytearray)):
        try:
            stored = json.loads(stored)
        except ValueError:
            return {}
    return dict(stored) if isinstance(stored, dict) else {}


def merge_edited_run_metadata(stored: Any, edited: Optional[dict]) -> dict:
    """The stored metadata updated with the edited keys, kernel keys kept as stored.

    Stored keys keep their order, so each kernel receipt serializes to the
    same bytes it was written with.
    """
    merged = _stored_dict(stored)
    if isinstance(edited, dict):
        for key, value in edited.items():
            if key not in KERNEL_KEYS:
                merged[key] = value
    return merged


def strip_kernel_keys(metadata: Optional[dict]) -> Optional[dict]:
    """``metadata`` without the kernel keys, for the run an edit itself creates."""
    if not isinstance(metadata, dict):
        return metadata
    return {key: value for key, value in metadata.items() if key not in KERNEL_KEYS}
