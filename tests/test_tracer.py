"""Live-SDK integration tests for the tracer.

Two layers of safety:

1. A signature smoke test that imports the *real* Langfuse v3 SDK and
   asserts the kwargs we pass to `propagate_attributes` and
   `start_as_current_observation` are accepted. This would have caught
   the `usage` → `usage_details` rename before publishing.

2. An end-to-end emission test that pipes a synthetic Turn through
   `emit_turn` against a mocked `Langfuse` client and inspects the
   recorded calls. Verifies the trace structure, the usage_details
   keys (including the four Anthropic token types), and the
   thinking-block metadata.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from claude_code_langfuse_hook import config as config_mod
from claude_code_langfuse_hook import tracer, transcript


# ---------------------------------------------------------------------------
# 1. Signature smoke test against the installed Langfuse SDK
# ---------------------------------------------------------------------------
def test_langfuse_v3_signatures_accept_our_kwargs() -> None:
    """If Langfuse v4 renames a kwarg, this test fails before publish."""
    from langfuse import Langfuse, propagate_attributes

    propagate_params = set(inspect.signature(propagate_attributes).parameters)
    for kw in ("session_id", "user_id", "trace_name", "tags"):
        assert kw in propagate_params, f"propagate_attributes lost kwarg: {kw}"

    observe_params = set(
        inspect.signature(Langfuse.start_as_current_observation).parameters
    )
    for kw in (
        "name",
        "as_type",
        "input",
        "output",
        "metadata",
        "model",
        "usage_details",
    ):
        assert kw in observe_params, (
            f"start_as_current_observation lost kwarg: {kw}"
        )


# ---------------------------------------------------------------------------
# 2. End-to-end emission test with a mocked Langfuse client
# ---------------------------------------------------------------------------
class _RecordingObservation:
    """Captures kwargs from start_as_current_observation calls."""

    def __init__(self, recorder: list[dict], kwargs: dict) -> None:
        self._recorder = recorder
        self._call: dict = dict(kwargs)
        self._call["updates"] = []
        recorder.append(self._call)

    def update(self, **kwargs):
        self._call["updates"].append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeLangfuse:
    """Stand-in that records every start_as_current_observation call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def start_as_current_observation(self, **kwargs) -> _RecordingObservation:
        return _RecordingObservation(self.calls, kwargs)


@contextmanager
def _capture_propagate():
    """Capture kwargs passed to `propagate_attributes`.

    `tracer.emit_turn` does `from langfuse import propagate_attributes`
    lazily, so we patch it on the `langfuse` module itself rather than
    on the tracer module.
    """
    import langfuse

    recorded: list[dict] = []

    @contextmanager
    def fake_propagate(**kwargs):
        recorded.append(kwargs)
        yield

    with mock.patch.object(langfuse, "propagate_attributes", fake_propagate):
        yield recorded


def _make_cfg(tmp_path: Path) -> config_mod.Config:
    return config_mod.Config(
        project_root=tmp_path,
        env_path=None,
        trace_enabled=True,
        project_name="my-project",
        langfuse_base_url="https://lf.example",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        debug=False,
        max_chars=20_000,
        trace_subagents=False,
    )


def _make_turn() -> transcript.Turn:
    """One user msg → assistant with thinking + text + tool_use, plus tool_result."""
    user = {"type": "user", "message": {"content": "explain caching"}}
    assistant = {
        "type": "assistant",
        "message": {
            "id": "m1",
            "model": "claude-sonnet-4-6-20250827",
            "content": [
                {"type": "thinking", "thinking": "I should reason about prompt caches."},
                {"type": "text", "text": "Prompt caching lets Claude reuse context."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/x"}},
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
            },
            "stop_reason": "tool_use",
        },
    }
    return transcript.Turn(
        user_msg=user,
        assistant_msgs=[assistant],
        tool_results_by_id={"t1": "file body"},
    )


def test_emit_turn_records_generation_with_all_four_token_types(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    turn = _make_turn()
    fake = _FakeLangfuse()

    with _capture_propagate() as propagate_calls:
        tracer.emit_turn(
            langfuse=fake,
            cfg=cfg,
            user_id="demo@example.com",
            session_id="sess-1",
            turn_num=1,
            turn=turn,
            transcript_path=tmp_path / "session.jsonl",
        )

    # propagate_attributes received our identity kwargs.
    assert len(propagate_calls) == 1
    pc = propagate_calls[0]
    assert pc["session_id"] == "sess-1"
    assert pc["user_id"] == "demo@example.com"
    assert pc["trace_name"] == "Claude Code - Turn 1"
    assert "project:my-project" in pc["tags"]
    assert "model:claude-sonnet-4-6-20250827" in pc["tags"]

    # We made three observation calls: root span, generation, one tool.
    assert len(fake.calls) == 3
    root, generation, tool = fake.calls

    assert root["name"] == "Claude Code - Turn 1"
    assert generation["name"] == "Claude Response"
    assert generation["as_type"] == "generation"
    assert generation["model"] == "claude-sonnet-4-6-20250827"

    # All four token categories landed in usage_details, using
    # Anthropic's native field names so they match Langfuse's default
    # price catalog out of the box.
    usage = generation["usage_details"]
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] == 20
    assert usage["cache_creation_input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 200
    assert "total" not in usage         # Langfuse derives total itself

    # Thinking content is preserved on the generation metadata, not the
    # input/output payload (so the primary trace view stays clean).
    meta = generation["metadata"]
    assert meta["thinking"] == "I should reason about prompt caches."
    assert meta["tool_count"] == 1
    assert meta["stop_reason"] == "tool_use"

    # Tool call became its own observation.
    assert tool["name"] == "Tool: Read"
    assert tool["as_type"] == "tool"
    assert tool["input"] == {"path": "/x"}
    # Tool output was attached via .update(output=...).
    update_outputs = [u for u in tool["updates"] if "output" in u]
    assert any(u["output"] == "file body" for u in update_outputs)


def test_emit_turn_dedups_tool_use_blocks_by_id(tmp_path: Path) -> None:
    """If two assistant rows in the same turn carry the same tool_use id,
    we emit one Tool observation, not two."""
    cfg = _make_cfg(tmp_path)
    user = {"type": "user", "message": {"content": "x"}}
    asst_a = {
        "type": "assistant",
        "message": {
            "id": "m1",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    asst_b = {
        "type": "assistant",
        "message": {
            "id": "m2",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    turn = transcript.Turn(
        user_msg=user, assistant_msgs=[asst_a, asst_b], tool_results_by_id={"t1": "ok"}
    )
    fake = _FakeLangfuse()
    with _capture_propagate():
        tracer.emit_turn(
            langfuse=fake, cfg=cfg, user_id="u", session_id="s",
            turn_num=1, turn=turn, transcript_path=tmp_path / "t.jsonl",
        )
    # root + generation + 1 tool (not 2)
    assert len(fake.calls) == 3
    assert fake.calls[2]["name"] == "Tool: Read"


def test_emit_turn_handles_non_numeric_usage(tmp_path: Path) -> None:
    """A typo'd usage payload must not crash emission; tokens default to 0."""
    cfg = _make_cfg(tmp_path)
    user = {"type": "user", "message": {"content": "hi"}}
    asst = {
        "type": "assistant",
        "message": {
            "id": "m1",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": "n/a", "output_tokens": None},
            "stop_reason": "end_turn",
        },
    }
    turn = transcript.Turn(user_msg=user, assistant_msgs=[asst], tool_results_by_id={})
    fake = _FakeLangfuse()
    with _capture_propagate():
        tracer.emit_turn(
            langfuse=fake, cfg=cfg, user_id="u", session_id="s",
            turn_num=1, turn=turn, transcript_path=tmp_path / "t.jsonl",
        )
    usage = fake.calls[1]["usage_details"]
    assert usage == {"input_tokens": 0, "output_tokens": 0}


def test_emit_turn_omits_cache_keys_when_not_present(tmp_path: Path) -> None:
    """Sessions without prompt caching should NOT include cache_* keys."""
    cfg = _make_cfg(tmp_path)
    user = {"type": "user", "message": {"content": "hi"}}
    assistant = {
        "type": "assistant",
        "message": {
            "id": "m1",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        },
    }
    turn = transcript.Turn(user_msg=user, assistant_msgs=[assistant], tool_results_by_id={})
    fake = _FakeLangfuse()

    with _capture_propagate():
        tracer.emit_turn(
            langfuse=fake,
            cfg=cfg,
            user_id="u",
            session_id="s",
            turn_num=1,
            turn=turn,
            transcript_path=tmp_path / "t.jsonl",
        )

    generation = fake.calls[1]
    usage = generation["usage_details"]
    assert usage == {"input_tokens": 10, "output_tokens": 5}
    assert "thinking" not in generation["metadata"]
