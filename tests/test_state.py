"""Tests for the state-file + incremental JSONL reader."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from claude_code_langfuse_hook import state as state_mod


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def test_state_round_trip_via_write_and_load(tmp_path: Path) -> None:
    """Use the canonical writer/loader (not raw JSON) to assert round-trip."""
    path = tmp_path / "state.json"
    state: dict = {}
    state_mod.write_session_state(
        state,
        "abc",
        state_mod.SessionState(offset=100, buffer=b"partial\xc3", turn_count=3),
    )
    state_mod.save_state(state, path)

    reloaded = state_mod.load_state(path)
    ss = state_mod.load_session_state(reloaded, "abc")
    assert ss.offset == 100
    assert ss.buffer == b"partial\xc3"   # partial UTF-8 byte preserved exactly
    assert ss.turn_count == 3


def test_state_round_trips_pending_msgs(tmp_path: Path) -> None:
    """Pending-msgs carry-over must survive a save/load cycle."""
    path = tmp_path / "state.json"
    state: dict = {}
    pending = [{"type": "user", "message": {"content": "dangling"}}]
    state_mod.write_session_state(
        state, "k",
        state_mod.SessionState(offset=42, buffer=b"", turn_count=1, pending_msgs=pending),
    )
    state_mod.save_state(state, path)
    reloaded = state_mod.load_state(path)
    ss = state_mod.load_session_state(reloaded, "k")
    assert ss.pending_msgs == pending


def test_state_uses_base64_for_buffer_on_disk(tmp_path: Path) -> None:
    """Sanity-check the on-disk shape — base64 keeps bytes round-trip-safe."""
    path = tmp_path / "state.json"
    state: dict = {}
    state_mod.write_session_state(
        state, "k", state_mod.SessionState(offset=0, buffer=b"\xc3\xa9", turn_count=0)
    )
    state_mod.save_state(state, path)
    raw = json.loads(path.read_text())
    assert "buffer_b64" in raw["k"]
    assert base64.b64decode(raw["k"]["buffer_b64"]) == b"\xc3\xa9"


def test_load_state_missing_file_returns_empty(tmp_path: Path) -> None:
    assert state_mod.load_state(tmp_path / "nope.json") == {}


def test_state_key_is_stable_and_distinct() -> None:
    a = state_mod.state_key("s1", "/tmp/a.jsonl")
    b = state_mod.state_key("s1", "/tmp/a.jsonl")
    c = state_mod.state_key("s1", "/tmp/b.jsonl")
    d = state_mod.state_key("s2", "/tmp/a.jsonl")
    assert a == b
    assert a != c
    assert a != d


def test_atomic_save_does_not_leave_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state_mod.save_state({"x": 1}, path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Incremental reader
# ---------------------------------------------------------------------------
def _write_lines(path: Path, lines: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def test_read_new_jsonl_advances_offset(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_lines(transcript, [{"a": 1}, {"a": 2}])
    ss = state_mod.SessionState()

    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"a": 1}, {"a": 2}]
    first_offset = ss.offset
    assert first_offset == transcript.stat().st_size

    # No new bytes — nothing to return.
    msgs2, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs2 == []
    assert ss.offset == first_offset

    # Append, re-read — only the new line comes back.
    _write_lines(transcript, [{"a": 3}])
    msgs3, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs3 == [{"a": 3}]
    assert ss.offset == transcript.stat().st_size


def test_read_new_jsonl_buffers_partial_line(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    # Complete line + incomplete line (no trailing newline).
    transcript.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2', encoding="utf-8")
    ss = state_mod.SessionState()

    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"a": 1}]
    assert ss.buffer == b'{"a": 2'

    # Complete the partial line and re-read.
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write("}\n")
    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"a": 2}]
    assert ss.buffer == b""


def test_read_new_jsonl_survives_multibyte_utf8_boundary(tmp_path: Path) -> None:
    """A multi-byte UTF-8 char split across two reads must NOT corrupt the line."""
    transcript = tmp_path / "session.jsonl"

    # First read: a complete line, then ONE byte (0xC3) of an 'é'
    # (UTF-8: 0xC3 0xA9) on the next line. The next-line newline isn't
    # there yet — so the partial byte must be buffered as bytes.
    transcript.write_bytes(b'{"a": 1}\n{"b": "\xc3')
    ss = state_mod.SessionState()
    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"a": 1}]
    assert ss.buffer == b'{"b": "\xc3'

    # Now append the second byte of 'é' plus the rest of the line.
    with transcript.open("ab") as fh:
        fh.write(b'\xa9llo"}\n')

    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    # The 'é' round-trips intact because we never `decode(errors="replace")`'d it.
    assert msgs == [{"b": "éllo"}]
    assert ss.buffer == b""


def test_read_new_jsonl_resets_offset_when_file_shrinks(tmp_path: Path) -> None:
    """If the transcript is rotated/truncated below ss.offset we must
    reset, not silently skip every byte until it grows again."""
    transcript = tmp_path / "session.jsonl"
    _write_lines(transcript, [{"a": 1}, {"a": 2}, {"a": 3}])
    ss = state_mod.SessionState()
    _, ss = state_mod.read_new_jsonl(transcript, ss)
    stale_offset = ss.offset
    assert stale_offset > 0

    # Replace file with shorter content (compaction / rotation).
    transcript.write_text(json.dumps({"b": 1}) + "\n", encoding="utf-8")
    assert transcript.stat().st_size < stale_offset

    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"b": 1}]
    assert ss.offset == transcript.stat().st_size
    assert ss.buffer == b""


def test_filelock_signals_acquisition(tmp_path: Path) -> None:
    """Single holder acquires; nested attempt within timeout fails — caller
    can then choose to skip work instead of writing split-brain state."""
    lock_path = tmp_path / "lock"
    with state_mod.FileLock(lock_path, timeout_s=0.1) as outer:
        assert outer.acquired is True
        with state_mod.FileLock(lock_path, timeout_s=0.1) as inner:
            # On Linux/macOS with fcntl, a second exclusive lock from a
            # *different file descriptor* in the same process is denied.
            assert inner.acquired is False


def test_read_new_jsonl_skips_malformed(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "not json\n" + json.dumps({"a": 1}) + "\n",
        encoding="utf-8",
    )
    ss = state_mod.SessionState()
    msgs, ss = state_mod.read_new_jsonl(transcript, ss)
    assert msgs == [{"a": 1}]
