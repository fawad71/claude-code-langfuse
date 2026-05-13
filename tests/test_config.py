"""Tests for env-var resolution and the .env parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from claude_code_langfuse_hook import config as config_mod


def _write_env(root: Path, body: str) -> Path:
    target = root / config_mod.ENV_FILENAME
    target.write_text(textwrap.dedent(body))
    return target


@pytest.fixture(autouse=True)
def _clear_relevant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any host-env vars that would otherwise leak into tests."""
    for name in (
        *config_mod.ALL_VARS,
        # also clear the un-prefixed names so we can prove we don't read them
        "TRACE_TO_LANGFUSE",
        "PROJECT_NAME",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_reads_dotenv_in_cwd(tmp_path: Path) -> None:
    _write_env(tmp_path, """
        CC_TRACE_TO_LANGFUSE=true
        CC_PROJECT_NAME=demo
        CC_LANGFUSE_BASE_URL=https://lf.example
        CC_LANGFUSE_PUBLIC_KEY=pk-1
        CC_LANGFUSE_SECRET_KEY=sk-1
    """)
    cfg = config_mod.resolve(tmp_path)
    assert cfg.trace_enabled is True
    assert cfg.project_name == "demo"
    assert cfg.langfuse_public_key == "pk-1"
    assert cfg.is_complete


def test_walks_up_for_dotenv(tmp_path: Path) -> None:
    _write_env(tmp_path, """
        CC_TRACE_TO_LANGFUSE=true
        CC_PROJECT_NAME=demo
        CC_LANGFUSE_BASE_URL=https://lf.example
        CC_LANGFUSE_PUBLIC_KEY=pk-1
        CC_LANGFUSE_SECRET_KEY=sk-1
    """)
    nested = tmp_path / "src" / "deeply" / "nested"
    nested.mkdir(parents=True)
    cfg = config_mod.resolve(nested)
    assert cfg.env_path is not None
    assert cfg.project_root == tmp_path.resolve()


def test_os_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path, """
        CC_TRACE_TO_LANGFUSE=true
        CC_PROJECT_NAME=from-file
        CC_LANGFUSE_BASE_URL=https://file.example
        CC_LANGFUSE_PUBLIC_KEY=pk-file
        CC_LANGFUSE_SECRET_KEY=sk-file
    """)
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk-env")
    monkeypatch.setenv("CC_PROJECT_NAME", "from-env")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.langfuse_public_key == "pk-env"
    assert cfg.project_name == "from-env"
    # Untouched fields still come from the file:
    assert cfg.langfuse_secret_key == "sk-file"


def test_works_with_only_os_env_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CC_TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("CC_PROJECT_NAME", "pure-env")
    monkeypatch.setenv("CC_LANGFUSE_BASE_URL", "https://lf.example")
    monkeypatch.setenv("CC_LANGFUSE_PUBLIC_KEY", "pk-1")
    monkeypatch.setenv("CC_LANGFUSE_SECRET_KEY", "sk-1")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.env_path is None
    assert cfg.is_complete
    assert cfg.project_name == "pure-env"


def test_returns_disabled_when_nothing_set(tmp_path: Path) -> None:
    cfg = config_mod.resolve(tmp_path)
    assert cfg.env_path is None
    assert cfg.trace_enabled is False
    assert not cfg.is_complete


def test_unprefixed_langfuse_vars_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical: a sibling service's LANGFUSE_PUBLIC_KEY must NOT leak in."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-from-other-service")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-from-other-service")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://other.example")
    monkeypatch.setenv("TRACE_TO_LANGFUSE", "true")
    monkeypatch.setenv("PROJECT_NAME", "other-service")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.trace_enabled is False
    assert cfg.langfuse_public_key == ""
    assert cfg.langfuse_secret_key == ""
    assert cfg.langfuse_base_url == ""
    assert cfg.project_name == "unknown-project"


def test_debug_and_max_chars_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CC_LANGFUSE_DEBUG", "true")
    monkeypatch.setenv("CC_LANGFUSE_MAX_CHARS", "1234")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.debug is True
    assert cfg.max_chars == 1234


def test_trace_subagents_defaults_off(tmp_path: Path) -> None:
    cfg = config_mod.resolve(tmp_path)
    assert cfg.trace_subagents is False


def test_trace_subagents_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CC_TRACE_SUBAGENTS", "true")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.trace_subagents is True


def test_max_chars_falls_back_to_default_on_bad_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_LANGFUSE_MAX_CHARS", "not-a-number")
    cfg = config_mod.resolve(tmp_path)
    assert cfg.max_chars == config_mod.DEFAULT_MAX_CHARS


def test_missing_fields_reported(tmp_path: Path) -> None:
    _write_env(tmp_path, """
        CC_TRACE_TO_LANGFUSE=true
        CC_PROJECT_NAME=demo
    """)
    cfg = config_mod.resolve(tmp_path)
    assert set(cfg.missing_fields()) == {
        config_mod.BASE_URL_VAR,
        config_mod.PUBLIC_KEY_VAR,
        config_mod.SECRET_KEY_VAR,
    }


def test_find_env_file_stops_at_git_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.env` above the repo boundary (e.g., $HOME/.env) must not be
    adopted as the project's config — the `.git` marker stops the walk."""
    # tmp_path/outer/.env   <- must NOT be picked up
    # tmp_path/outer/repo/  <- has .git
    # tmp_path/outer/repo/src/
    outer = tmp_path / "outer"
    repo = outer / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (repo / ".git").mkdir()
    (outer / ".env").write_text("CC_TRACE_TO_LANGFUSE=true\n")
    # Pretend home is somewhere unrelated so the home guard isn't what stops us.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")

    found = config_mod.find_env_file(src)
    assert found is None


def test_find_env_file_stops_at_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.env` above $HOME must not be adopted."""
    home = tmp_path / "home"
    project = home / "work" / "proj"
    project.mkdir(parents=True)
    (tmp_path / ".env").write_text("CC_TRACE_TO_LANGFUSE=true\n")
    monkeypatch.setattr(Path, "home", lambda: home)

    found = config_mod.find_env_file(project)
    assert found is None


def test_env_parser_handles_quotes_comments_and_export(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        '# a comment\n'
        '\n'
        'CC_TRACE_TO_LANGFUSE=true\n'
        'CC_PROJECT_NAME="quoted name"\n'
        "CC_LANGFUSE_BASE_URL='https://x'\n"
        'export CC_LANGFUSE_PUBLIC_KEY=pk-exported\n'
        'CC_LANGFUSE_SECRET_KEY=sk-bare\n'
        'BAD_LINE_WITHOUT_EQUALS\n'
    )
    parsed = config_mod.parse_env_file(env)
    assert parsed["CC_PROJECT_NAME"] == "quoted name"
    assert parsed["CC_LANGFUSE_BASE_URL"] == "https://x"
    assert parsed["CC_LANGFUSE_PUBLIC_KEY"] == "pk-exported"
    assert parsed["CC_LANGFUSE_SECRET_KEY"] == "sk-bare"
    assert "BAD_LINE_WITHOUT_EQUALS" not in parsed
