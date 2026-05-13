"""User-identity resolution.

`user_id` is derived from `git config user.email` at hook runtime —
every engineer already has this set for commits, so we inherit a free
per-machine identity without any per-user setup.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_user_id(cwd: Path) -> str:
    """Return the user's identity, falling back to OS user with a WARN."""
    # `git` will raise FileNotFoundError if `cwd` itself was removed
    # (stale .env parent after a `mv`). Pre-check so we fall through
    # cleanly to the OS-user fallback.
    if cwd.exists():
        try:
            out = subprocess.check_output(
                ["git", "config", "user.email"],
                cwd=str(cwd),
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            if out:
                return out
        except Exception as exc:
            log.warning("git config user.email unavailable: %s", exc)
    else:
        log.warning("resolve_user_id: cwd %s does not exist; skipping git.", cwd)

    fallback = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    log.warning("Falling back to OS user %r (no git email).", fallback)
    return fallback
