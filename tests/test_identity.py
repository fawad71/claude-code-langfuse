"""Tests for `identity.resolve_user_id`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from claude_code_langfuse_hook import identity


def test_returns_git_email_when_available(tmp_path: Path) -> None:
    with mock.patch.object(
        subprocess, "check_output", return_value="dev@example.com\n"
    ):
        assert identity.resolve_user_id(tmp_path) == "dev@example.com"


def test_falls_back_to_user_env_when_git_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "alice")
    with mock.patch.object(
        subprocess, "check_output", side_effect=FileNotFoundError("no git")
    ):
        assert identity.resolve_user_id(tmp_path) == "alice"


def test_falls_back_when_git_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "bob")
    with mock.patch.object(subprocess, "check_output", return_value="\n"):
        assert identity.resolve_user_id(tmp_path) == "bob"


def test_handles_missing_cwd_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale .env-parent directory must not crash; should hit OS-user fallback."""
    monkeypatch.setenv("USER", "carol")
    # subprocess is never called because cwd.exists() is False — assert that.
    with mock.patch.object(subprocess, "check_output") as m:
        result = identity.resolve_user_id(Path("/nonexistent/path/xyzzy"))
    assert result == "carol"
    m.assert_not_called()


def test_returns_unknown_when_nothing_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    with mock.patch.object(
        subprocess, "check_output", side_effect=FileNotFoundError("no git")
    ):
        assert identity.resolve_user_id(Path.cwd()) == "unknown"
