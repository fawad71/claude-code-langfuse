"""Smoke tests for the CLI: install/uninstall idempotency + init snippet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_code_langfuse_hook import cli


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.claude/settings.json into tmp_path for every test."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS", fake_home / ".claude" / "settings.json")
    return fake_home


def test_install_creates_hook_entry() -> None:
    rc = cli.main(["install"])
    assert rc == 0
    data = json.loads(cli.CLAUDE_SETTINGS.read_text())
    commands = [
        h["command"]
        for entry in data["hooks"]["Stop"]
        for h in entry["hooks"]
    ]
    assert commands == [cli.HOOK_COMMAND]


def test_install_is_idempotent() -> None:
    cli.main(["install"])
    cli.main(["install"])
    data = json.loads(cli.CLAUDE_SETTINGS.read_text())
    assert len(data["hooks"]["Stop"]) == 1


def test_uninstall_removes_entry() -> None:
    cli.main(["install"])
    rc = cli.main(["uninstall"])
    assert rc == 0
    data = json.loads(cli.CLAUDE_SETTINGS.read_text())
    assert data["hooks"]["Stop"] == []


def test_status_exit_code_signals_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """status returns 1 when not ready, 0 with --exit-zero, 0 when ready."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "CC_TRACE_TO_LANGFUSE",
        "CC_PROJECT_NAME",
        "CC_LANGFUSE_BASE_URL",
        "CC_LANGFUSE_PUBLIC_KEY",
        "CC_LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    assert cli.main(["status"]) == 1
    assert cli.main(["status", "--exit-zero"]) == 0

    monkeypatch.setenv("CC_TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("CC_PROJECT_NAME", "demo")
    monkeypatch.setenv("CC_LANGFUSE_BASE_URL", "https://x")
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("CC_LANGFUSE_SECRET_KEY", "sk")
    cli.main(["install"])
    assert cli.main(["status"]) == 0


def test_cmd_test_uses_v3_sdk_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Regression guard: `claude-langfuse test` must use the v3 SDK
    pattern (propagate_attributes + start_as_current_observation), not
    the removed v2 `client.trace(...)` API."""
    from contextlib import contextmanager
    from unittest import mock

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CC_TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("CC_PROJECT_NAME", "demo")
    monkeypatch.setenv("CC_LANGFUSE_BASE_URL", "https://lf.example")
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("CC_LANGFUSE_SECRET_KEY", "sk")

    # Stub the Langfuse class entirely so we don't hit the network.
    calls: list[dict] = []

    class FakeSpan:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.updates: list[dict] = []
            calls.append({"op": "start_as_current_observation", **kwargs, "updates": self.updates})
        def update(self, **kw): self.updates.append(kw)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class FakeLangfuse:
        def __init__(self, **kwargs):
            calls.append({"op": "init", **kwargs})
        def start_as_current_observation(self, **kw): return FakeSpan(kw)
        def flush(self): calls.append({"op": "flush"})
        def shutdown(self): calls.append({"op": "shutdown"})

    @contextmanager
    def fake_propagate(**kw):
        calls.append({"op": "propagate_attributes", **kw})
        yield

    import langfuse

    with mock.patch.object(langfuse, "Langfuse", FakeLangfuse), \
         mock.patch.object(langfuse, "propagate_attributes", fake_propagate):
        rc = cli.main(["test"])

    assert rc == 0
    ops = [c["op"] for c in calls]
    # The v3-correct call sequence: init → propagate → observe → flush → shutdown.
    assert ops == [
        "init",
        "propagate_attributes",
        "start_as_current_observation",
        "flush",
        "shutdown",
    ]

    # Verify the kwargs we promise users:
    propagate_call = next(c for c in calls if c["op"] == "propagate_attributes")
    assert propagate_call["session_id"] == "claude-langfuse-cli-test"
    assert propagate_call["trace_name"] == "claude-langfuse:test:demo"
    assert "project:demo" in propagate_call["tags"]

    observation = next(c for c in calls if c["op"] == "start_as_current_observation")
    assert observation["name"] == "claude-langfuse:test:demo"
    assert observation["input"] == {"role": "user", "content": "ping"}
    assert observation["updates"] == [{"output": {"role": "assistant", "content": "pong"}}]


def test_save_settings_is_atomic_and_leaves_no_tmp(tmp_path: Path) -> None:
    cli.main(["install"])
    assert cli.CLAUDE_SETTINGS.exists()
    tmp_sibling = cli.CLAUDE_SETTINGS.with_suffix(cli.CLAUDE_SETTINGS.suffix + ".tmp")
    assert not tmp_sibling.exists()


def test_init_prints_env_snippet(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["init"])
    assert rc == 0
    captured = capsys.readouterr().out
    for line in (
        "CC_TRACE_TO_LANGFUSE=true",
        "CC_PROJECT_NAME=",
        "CC_LANGFUSE_BASE_URL=",
        "CC_LANGFUSE_PUBLIC_KEY=",
        "CC_LANGFUSE_SECRET_KEY=",
    ):
        assert line in captured
