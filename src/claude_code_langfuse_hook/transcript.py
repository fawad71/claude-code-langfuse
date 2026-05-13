"""Parse Claude Code's JSONL transcript into structured turns.

A "turn" is one user message → one or more assistant messages →
(possibly interleaved) tool_result rows. The transcript stores these
as a flat sequence of role-tagged entries; we assemble them with two
deduplication rules:

  - Assistant messages that share the same `message.id` are partial
    streaming rows; latest-wins replaces earlier copies.
  - Tool results are keyed by `tool_use_id`; latest-wins.

The dedup matters because Claude Code may write multiple rows for the
same assistant message as it streams in.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field accessors — Claude Code's schema has two shapes for some fields.
# ---------------------------------------------------------------------------
def get_role(msg: dict) -> Optional[str]:
    """Resolve role from either `type` or `message.role`."""
    if not isinstance(msg, dict):
        return None
    t = msg.get("type")
    if t in ("user", "assistant"):
        return t
    m = msg.get("message")
    if isinstance(m, dict):
        r = m.get("role")
        if r in ("user", "assistant"):
            return r
    return None


def get_content(msg: dict) -> Any:
    if not isinstance(msg, dict):
        return None
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("content")
    return msg.get("content")


def get_message_id(msg: dict) -> Optional[str]:
    m = msg.get("message")
    if isinstance(m, dict):
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            return mid
    return None


def get_model(msg: dict) -> str:
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("model") or "claude"
    return "claude"


def get_usage(msg: dict) -> dict:
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("usage") or {}
    return {}


# ---------------------------------------------------------------------------
# Content-block helpers
# ---------------------------------------------------------------------------
def is_tool_result(msg: dict) -> bool:
    if get_role(msg) != "user":
        return False
    content = get_content(msg)
    if isinstance(content, list):
        return any(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in content
        )
    return False


def iter_tool_uses(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"
        ]
    return []


def iter_tool_results(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ]
    return []


def extract_text(content: Any) -> str:
    """Pull plain text out of a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return ""


def extract_thinking(content: Any) -> str:
    """Pull extended-thinking blocks out of a content list.

    Claude's extended-thinking responses emit `type=thinking` blocks
    alongside the `type=text` and `type=tool_use` blocks. We surface
    this separately (as generation metadata) so engineers can audit
    Claude's reasoning without bloating the trace's primary
    input/output view.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                parts.append(item.get("thinking") or "")
        return "\n".join(p for p in parts if p)
    return ""


# ---------------------------------------------------------------------------
# Truncation — caps oversized payloads but preserves identity in metadata
# ---------------------------------------------------------------------------
def truncate_text(s: Optional[str], max_chars: int) -> tuple[str, dict]:
    """Cap `s` at `max_chars`; return (text, metadata).

    Metadata always includes `truncated` and `orig_len`. When truncated,
    it also includes `kept_len` and the sha256 of the original so we can
    recognize duplicates or look up the full content elsewhere.
    """
    if s is None:
        return "", {"truncated": False, "orig_len": 0}
    orig_len = len(s)
    if orig_len <= max_chars:
        return s, {"truncated": False, "orig_len": orig_len}
    head = s[:max_chars]
    return head, {
        "truncated": True,
        "orig_len": orig_len,
        "kept_len": len(head),
        "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
    }


def truncate_value(value: Any, max_chars: int) -> tuple[Any, Optional[dict]]:
    """Truncate string-like values; pass dicts/lists through as-is.

    For dict/list inputs we JSON-stringify only if doing so exceeds the
    limit — otherwise the structured form is more useful in Langfuse.
    """
    if isinstance(value, str):
        text, meta = truncate_text(value, max_chars)
        return text, (meta if meta.get("truncated") else None)
    if isinstance(value, (dict, list)):
        try:
            as_str = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return value, None
        if len(as_str) <= max_chars:
            return value, None
        text, meta = truncate_text(as_str, max_chars)
        return text, meta
    return value, None


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    user_msg: dict
    assistant_msgs: list[dict]
    tool_results_by_id: dict[str, Any]


def split_for_commit(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split messages into (committed, pending).

    The last "turn" is only safe to emit once we've seen at least one
    assistant message bounded to it — otherwise the assistant response
    may still be streaming in and another hook fire is about to add to
    the same turn. We carry the trailing user/tool_result rows forward
    as `pending` until that boundary lands.

    Rules:
      - Find the index of the last non-tool-result user message.
      - If no real user msg exists, everything is pending.
      - If the segment from that user msg onward contains at least one
        assistant msg, the whole list is committed (the turn is done;
        a future user msg only opens a *new* turn).
      - Otherwise (user msg with no assistant yet) we commit everything
        before it and pend from that user msg onward.
    """
    last_user = -1
    for i, m in enumerate(messages):
        if get_role(m) == "user" and not is_tool_result(m):
            last_user = i
    if last_user == -1:
        return [], list(messages)
    has_assistant = any(
        get_role(m) == "assistant" for m in messages[last_user:]
    )
    if has_assistant:
        return list(messages), []
    return list(messages[:last_user]), list(messages[last_user:])


def build_turns(messages: list[dict]) -> list[Turn]:
    """Group a list of transcript entries into Turns.

    Rules:
      - A user entry that ISN'T a tool_result closes the previous turn
        and opens a new one.
      - tool_result rows are pinned to the current turn, keyed by
        `tool_use_id`, latest-wins.
      - Assistant rows are accumulated into the current turn. Multiple
        rows sharing the same `message.id` are deduped latest-wins.
    """
    turns: list[Turn] = []

    current_user: Optional[dict] = None
    assistant_order: list[str] = []
    assistant_latest: dict[str, dict] = {}
    tool_results_by_id: dict[str, Any] = {}

    def flush() -> None:
        nonlocal current_user
        if current_user is None or not assistant_latest:
            return
        ordered_assistants = [
            assistant_latest[mid]
            for mid in assistant_order
            if mid in assistant_latest
        ]
        turns.append(
            Turn(
                user_msg=current_user,
                assistant_msgs=ordered_assistants,
                tool_results_by_id=dict(tool_results_by_id),
            )
        )

    for msg in messages:
        role = get_role(msg)

        if is_tool_result(msg):
            for tr in iter_tool_results(get_content(msg)):
                tid = tr.get("tool_use_id")
                if tid:
                    tool_results_by_id[str(tid)] = tr.get("content")
            continue

        if role == "user":
            flush()
            current_user = msg
            assistant_order = []
            assistant_latest = {}
            tool_results_by_id = {}
            continue

        if role == "assistant":
            if current_user is None:
                continue
            mid = get_message_id(msg) or f"noid:{len(assistant_order)}"
            if mid not in assistant_latest:
                assistant_order.append(mid)
            assistant_latest[mid] = msg
            continue

        # Unknown row types are ignored on purpose.

    flush()
    return turns
