"""Tests for the turn-assembly + truncation pipeline."""

from __future__ import annotations

import hashlib
import json

from claude_code_langfuse_hook import transcript


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------
def test_simple_user_then_assistant_pairs_into_one_turn() -> None:
    msgs = [
        {"type": "user", "message": {"content": "hi"}},
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
    ]
    turns = transcript.build_turns(msgs)
    assert len(turns) == 1
    assert transcript.extract_text(transcript.get_content(turns[0].user_msg)) == "hi"
    assert len(turns[0].assistant_msgs) == 1


def test_two_user_messages_create_two_turns() -> None:
    msgs = [
        {"type": "user", "message": {"content": "first"}},
        {"type": "assistant", "message": {"id": "a1", "content": "ok"}},
        {"type": "user", "message": {"content": "second"}},
        {"type": "assistant", "message": {"id": "a2", "content": "done"}},
    ]
    assert len(transcript.build_turns(msgs)) == 2


def test_dedup_assistant_rows_by_message_id() -> None:
    """Streaming partial rows share message.id — latest one wins."""
    msgs = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "message": {"id": "m1", "content": "partial"}},
        {"type": "assistant", "message": {"id": "m1", "content": "complete"}},
    ]
    turns = transcript.build_turns(msgs)
    assert len(turns) == 1
    # Only one assistant entry kept, and it's the latest one.
    assert len(turns[0].assistant_msgs) == 1
    assert transcript.get_content(turns[0].assistant_msgs[0]) == "complete"


def test_tool_result_rows_attach_to_current_turn() -> None:
    msgs = [
        {"type": "user", "message": {"content": "read x"}},
        {
            "type": "assistant",
            "message": {
                "id": "a1",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/x"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file body"},
                ],
            },
        },
    ]
    turns = transcript.build_turns(msgs)
    assert len(turns) == 1
    assert turns[0].tool_results_by_id == {"t1": "file body"}


def test_tool_result_latest_wins_on_collision() -> None:
    msgs = [
        {"type": "user", "message": {"content": "x"}},
        {
            "type": "assistant",
            "message": {
                "id": "a1",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "first"}]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "second"}]
            },
        },
    ]
    turns = transcript.build_turns(msgs)
    assert turns[0].tool_results_by_id == {"t1": "second"}


def test_split_for_commit_keeps_dangling_user_pending() -> None:
    """A user msg with no assistant yet must stay pending so the next
    fire (when the assistant arrives) commits the turn exactly once."""
    msgs = [
        {"type": "user", "message": {"content": "first"}},
        {"type": "assistant", "message": {"id": "a1", "content": "ok"}},
        {"type": "user", "message": {"content": "second"}},
    ]
    commit, pending = transcript.split_for_commit(msgs)
    assert len(commit) == 2
    assert commit[0]["message"]["content"] == "first"
    assert len(pending) == 1
    assert pending[0]["message"]["content"] == "second"


def test_split_for_commit_commits_completed_last_turn() -> None:
    msgs = [
        {"type": "user", "message": {"content": "first"}},
        {"type": "assistant", "message": {"id": "a1", "content": "ok"}},
    ]
    commit, pending = transcript.split_for_commit(msgs)
    assert commit == msgs
    assert pending == []


def test_split_for_commit_keeps_orphan_tool_results_pending() -> None:
    """Tool-result rows with no preceding user msg are pending until
    one of (real user, full turn) lands."""
    msgs = [
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x"}
        ]}},
    ]
    commit, pending = transcript.split_for_commit(msgs)
    assert commit == []
    assert pending == msgs


def test_assistant_without_preceding_user_is_dropped() -> None:
    msgs = [
        {"type": "assistant", "message": {"id": "a1", "content": "orphan"}},
    ]
    assert transcript.build_turns(msgs) == []


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def test_truncate_text_under_limit_unchanged() -> None:
    text, meta = transcript.truncate_text("hello", max_chars=100)
    assert text == "hello"
    assert meta == {"truncated": False, "orig_len": 5}


def test_truncate_text_over_limit_captures_hash_and_length() -> None:
    big = "x" * 5_000
    text, meta = transcript.truncate_text(big, max_chars=100)
    assert text == "x" * 100
    assert meta["truncated"] is True
    assert meta["orig_len"] == 5_000
    assert meta["kept_len"] == 100
    assert meta["sha256"] == hashlib.sha256(big.encode()).hexdigest()


def test_truncate_value_passes_through_small_dicts() -> None:
    value = {"path": "/x"}
    out, meta = transcript.truncate_value(value, max_chars=100)
    assert out == value
    assert meta is None


def test_truncate_value_serializes_oversized_dicts() -> None:
    value = {"data": "y" * 5_000}
    out, meta = transcript.truncate_value(value, max_chars=100)
    assert isinstance(out, str)
    assert meta is not None and meta["truncated"] is True


# ---------------------------------------------------------------------------
# Extended thinking
# ---------------------------------------------------------------------------
def test_extract_thinking_pulls_thinking_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "let me reason..."},
        {"type": "text", "text": "the answer is 42"},
    ]
    assert transcript.extract_thinking(content) == "let me reason..."
    # And extract_text still only sees the text block.
    assert transcript.extract_text(content) == "the answer is 42"


def test_extract_thinking_empty_for_no_thinking_blocks() -> None:
    assert transcript.extract_thinking([{"type": "text", "text": "hi"}]) == ""
    assert transcript.extract_thinking("plain string") == ""
    assert transcript.extract_thinking(None) == ""


def test_extract_thinking_joins_multiple_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "step 1"},
        {"type": "thinking", "thinking": "step 2"},
    ]
    assert transcript.extract_thinking(content) == "step 1\nstep 2"
