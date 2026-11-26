"""Check that a card's patch applies on a repository before the card is created.

``check_patch`` runs ``git apply --check --verbose`` and writes nothing: the working
tree, the index and ``.git`` are left as they were.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

APPLIES = "applies"


def check_patch(root: str | os.PathLike, patch_path: str | os.PathLike) -> str:
    """Return ``"applies"``, or git's refusal text, for the patch against ``root``."""
    done = subprocess.run(
        ["git", "apply", "--check", "--verbose", str(Path(patch_path).resolve())],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        env={**os.environ, "LC_ALL": "C"},
    )
    if done.returncode == 0:
        return APPLIES
    return (done.stderr or done.stdout).strip() or f"git apply exited with status {done.returncode}"
