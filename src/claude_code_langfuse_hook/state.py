"""Per-session state for incremental transcript reading.

Each invocation of the Stop hook needs to remember:
  - how many bytes of the transcript we've already processed (`offset`)
  - any partial trailing line we couldn't parse (`buffer`, raw bytes)
  - the running turn count, so trace names stay monotonic (`turn_count`)

State is persisted to `~/.claude/state/claude_langfuse_state.json` and
keyed by sha256(session_id::transcript_path) so colliding session IDs
in different repos don't trample each other.

The buffer is kept as `bytes` (and base64-encoded on disk) rather than
a UTF-8-decoded string. That way, if a chunk read happens to land in
the middle of a multi-byte UTF-8 sequence, we never apply
`errors="replace"` to incomplete bytes — the partial sequence is
carried forward to the next invocation as bytes and only decoded once
the full line is in hand.

Concurrency: `fcntl.flock` gives us best-effort exclusion when two
Claude Code sessions fire their Stop hook simultaneously. Atomic
writes (`tmp.write_text(...); os.replace(tmp, target)`) ensure a
crashed write can never corrupt the state file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".claude" / "state"
STATE_FILE = STATE_DIR / "claude_langfuse_state.json"
LOCK_FILE = STATE_DIR / "claude_langfuse_state.lock"


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    offset: int = 0
    buffer: bytes = b""
    turn_count: int = 0
    # Messages that belong to a turn whose assistant response hasn't
    # arrived yet (or hasn't been bounded by the *next* user message).
    # Carried forward across hook fires so a turn that straddles two
    # Stop events is committed exactly once.
    pending_msgs: list = field(default_factory=list)


def state_key(session_id: str, transcript_path: str) -> str:
    """Stable key for a (session, transcript) pair."""
    raw = f"{session_id}::{transcript_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_session_state(global_state: dict, key: str) -> SessionState:
    entry = global_state.get(key, {})
    buf_b64 = entry.get("buffer_b64") or ""
    try:
        buf = base64.b64decode(buf_b64) if buf_b64 else b""
    except (ValueError, TypeError):
        buf = b""
    pending = entry.get("pending_msgs") or []
    if not isinstance(pending, list):
        pending = []
    return SessionState(
        offset=int(entry.get("offset", 0)),
        buffer=buf,
        turn_count=int(entry.get("turn_count", 0)),
        pending_msgs=pending,
    )


def write_session_state(global_state: dict, key: str, ss: SessionState) -> None:
    global_state[key] = {
        "offset": ss.offset,
        # Bytes can't go straight into JSON — base64 round-trips them
        # without losing partial UTF-8 sequences.
        "buffer_b64": base64.b64encode(ss.buffer).decode("ascii") if ss.buffer else "",
        "turn_count": ss.turn_count,
        "pending_msgs": ss.pending_msgs,
        "updated": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Global state file — atomic load/save
# ---------------------------------------------------------------------------
def load_state(path: Path = STATE_FILE) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load state file %s: %s — starting fresh.", path, exc)
        return {}


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    """Atomic write — never leaves a half-written state file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("save_state failed for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Cross-process lock (best effort)
# ---------------------------------------------------------------------------
class FileLock:
    """Best-effort exclusive lock via `fcntl.flock`.

    Falls through silently on platforms without fcntl (Windows) — better
    a rare state collision than crashing the hook.
    """

    def __init__(self, path: Path = LOCK_FILE, timeout_s: float = 5.0) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self._fh = None  # type: ignore[assignment]
        self.acquired = False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            import fcntl  # type: ignore[import-not-found]

            deadline = time.time() + self.timeout_s
            while True:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.acquired = True
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        log.warning(
                            "FileLock timeout on %s after %.1fs — proceeding without lock.",
                            self.path, self.timeout_s,
                        )
                        break
                    time.sleep(0.05)
        except ImportError:
            # Windows or other platforms without fcntl — proceed unlocked.
            # Treat as acquired so callers don't skip work entirely.
            self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            import fcntl  # type: ignore[import-not-found]

            if self._fh is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Incremental JSONL reader
# ---------------------------------------------------------------------------
def read_new_jsonl(
    transcript_path: Path,
    ss: SessionState,
) -> tuple[list[dict[str, Any]], SessionState]:
    """Read only bytes past `ss.offset`. Carry the partial last line forward.

    Operates in bytes throughout: we only decode a line as UTF-8 once
    we've seen its terminating newline. Bytes after the last newline are
    carried forward in `ss.buffer` and re-joined with the next read.
    """
    if not transcript_path.exists():
        return [], ss

    try:
        file_size = transcript_path.stat().st_size
        # If the file shrank below our offset, it was rotated, truncated,
        # or replaced (e.g., by Claude Code compaction). Reset to 0 and
        # drop the stale buffer — otherwise we'd silently skip every
        # byte until the file grows past the old offset again.
        if file_size < ss.offset:
            log.warning(
                "Transcript %s shrank (%d < %d) — resetting offset.",
                transcript_path, file_size, ss.offset,
            )
            ss.offset = 0
            ss.buffer = b""
        with open(transcript_path, "rb") as fh:
            fh.seek(ss.offset)
            chunk = fh.read()
            new_offset = fh.tell()
    except OSError as exc:
        log.warning("read_new_jsonl failed for %s: %s", transcript_path, exc)
        return [], ss

    if not chunk and not ss.buffer:
        return [], ss

    combined = ss.buffer + chunk
    last_nl = combined.rfind(b"\n")
    if last_nl == -1:
        # Nothing terminated yet — keep everything in the buffer and
        # wait for the next hook fire.
        ss.buffer = combined
        ss.offset = new_offset
        return [], ss

    parseable = combined[: last_nl + 1]
    ss.buffer = combined[last_nl + 1 :]
    ss.offset = new_offset

    parsed: list[dict[str, Any]] = []
    for line_bytes in parseable.split(b"\n"):
        line_bytes = line_bytes.strip()
        if not line_bytes:
            continue
        try:
            # json.loads accepts bytes since Python 3.6; if a multi-byte
            # sequence is somehow broken inside a complete line, skip it.
            parsed.append(json.loads(line_bytes))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return parsed, ss
