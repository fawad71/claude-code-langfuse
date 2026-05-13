"""End-to-end test for hook.run() — pending-msgs straddle case.

If a turn straddles two Stop hook fires (user appended in fire 1, the
assistant only arrives in fire 2), the turn must be emitted exactly
once — not zero times, not twice.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from claude_code_langfuse_hook import cli, hook as hook_mod, state as state_mod


class _RecObs:
    def __init__(self, recorder, kwargs):
        self.kwargs = dict(kwargs)
        self.kwargs["updates"] = []
        recorder.append(self.kwargs)

    def update(self, **kw):
        self.kwargs["updates"].append(kw)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeLF:
    def __init__(self, **_):
        self.calls = []

    def start_as_current_observation(self, **kw):
        return _RecObs(self.calls, kw)

    def flush(self):
        pass

    def shutdown(self):
        pass


@contextmanager
def _patched_langfuse():
    import langfuse

    @contextmanager
    def fake_propagate(**_kw):
        yield

    with mock.patch.object(langfuse, "Langfuse", _FakeLF), \
         mock.patch.object(langfuse, "propagate_attributes", fake_propagate):
        yield


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_turn_straddling_two_fires_emits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # state module captured Path.home() at import time; redirect explicitly.
    state_dir = fake_home / ".claude" / "state"
    monkeypatch.setattr(state_mod, "STATE_DIR", state_dir)
    new_state_file = state_dir / "claude_langfuse_state.json"
    monkeypatch.setattr(state_mod, "STATE_FILE", new_state_file)
    monkeypatch.setattr(state_mod, "LOCK_FILE", state_dir / "claude_langfuse_state.lock")
    # `load_state`/`save_state` capture STATE_FILE as a default arg at
    # def time — override the bound defaults so the hook actually writes
    # to our tmp path instead of the real ~/.claude/state.
    monkeypatch.setattr(state_mod.load_state, "__defaults__", (new_state_file,))
    monkeypatch.setattr(state_mod.save_state, "__defaults__", (new_state_file,))
    monkeypatch.setattr(hook_mod, "LOG_PATH", state_dir / "claude_langfuse.log")

    monkeypatch.setenv("CC_TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("CC_PROJECT_NAME", "demo")
    monkeypatch.setenv("CC_LANGFUSE_BASE_URL", "https://lf.example")
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("CC_LANGFUSE_SECRET_KEY", "sk")

    project = tmp_path / "proj"
    project.mkdir()
    transcript = project / "session.jsonl"

    # Fire 1: a complete turn, plus a *dangling* user message (assistant
    # hasn't arrived yet).
    _write_jsonl(transcript, [
        {"type": "user", "message": {"content": "hi"}},
        {
            "type": "assistant",
            "message": {
                "id": "a1",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {"type": "user", "message": {"content": "second"}},
    ])

    payload = json.dumps({
        "session_id": "sess-1",
        "transcript_path": str(transcript),
        "cwd": str(project),
    })

    with _patched_langfuse():
        # Fire 1
        monkeypatch.setattr("sys.stdin", _Stdin(payload))
        assert hook_mod.run() == 0

        # Append the assistant for turn 2 and fire again.
        _write_jsonl(transcript, [
            {
                "type": "assistant",
                "message": {
                    "id": "a2",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "world"}],
                    "usage": {"input_tokens": 2, "output_tokens": 2},
                },
            },
        ])
        monkeypatch.setattr("sys.stdin", _Stdin(payload))
        assert hook_mod.run() == 0

    # Read state directly to count how many turns were emitted in total.
    final = state_mod.load_state(state_mod.STATE_FILE)
    key = state_mod.state_key("sess-1", str(transcript))
    ss = state_mod.load_session_state(final, key)
    # Two complete turns should have been committed exactly once each.
    assert ss.turn_count == 2, f"expected 2 turns committed, got {ss.turn_count}"
    assert ss.pending_msgs == []


def test_subagent_transcript_is_skipped_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transcripts under a `subagents/` path component are skipped unless
    CC_TRACE_SUBAGENTS=true — the main session's traces shouldn't be
    crowded with nested-agent loops."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    state_dir = fake_home / ".claude" / "state"
    new_state_file = state_dir / "claude_langfuse_state.json"
    monkeypatch.setattr(state_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(state_mod, "STATE_FILE", new_state_file)
    monkeypatch.setattr(state_mod, "LOCK_FILE", state_dir / "claude_langfuse_state.lock")
    monkeypatch.setattr(state_mod.load_state, "__defaults__", (new_state_file,))
    monkeypatch.setattr(state_mod.save_state, "__defaults__", (new_state_file,))
    monkeypatch.setattr(hook_mod, "LOG_PATH", state_dir / "claude_langfuse.log")

    monkeypatch.setenv("CC_TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("CC_PROJECT_NAME", "demo")
    monkeypatch.setenv("CC_LANGFUSE_BASE_URL", "https://lf.example")
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("CC_LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("CC_TRACE_SUBAGENTS", raising=False)

    project = tmp_path / "proj"
    sub_dir = project / "subagents" / "agent1"
    sub_dir.mkdir(parents=True)
    transcript = sub_dir / "session.jsonl"
    _write_jsonl(transcript, [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "message": {
            "id": "a1", "model": "claude", "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }},
    ])
    payload = json.dumps({
        "session_id": "sub-sess",
        "transcript_path": str(transcript),
        "cwd": str(project),
    })

    with _patched_langfuse():
        monkeypatch.setattr("sys.stdin", _Stdin(payload))
        assert hook_mod.run() == 0

    # Nothing should have been written to state — we exited before the lock.
    assert not new_state_file.exists()


class _Stdin:
    """Minimal stdin stand-in for `json.load(sys.stdin)`."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> str:
        return self._payload

    def isatty(self) -> bool:
        return False
