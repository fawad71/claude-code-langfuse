"""Config resolution — env vars first, with an optional `.env` loader.

The hook reads five required variables and three optional knobs. Set
them however you already manage secrets (direnv, doppler, vault, a
plain `.env`, your shell rc, CI secrets, …) — env always wins. As a
convenience, if a `.env` file exists at the project root (or any
ancestor), we parse it as a fallback so engineers can just drop the
keys into their existing dotfile.

All variables use the `CC_` prefix to keep this hook isolated from any
other Langfuse-using service that might share the engineer's
environment (e.g. a Python app that reads the standard
`LANGFUSE_PUBLIC_KEY`). Unprefixed names are deliberately ignored.

Required:
  CC_TRACE_TO_LANGFUSE     "true" to enable
  CC_PROJECT_NAME          project slug tagged onto every trace
  CC_LANGFUSE_BASE_URL     Langfuse host
  CC_LANGFUSE_PUBLIC_KEY   Langfuse public key
  CC_LANGFUSE_SECRET_KEY   Langfuse secret key

Optional:
  CC_LANGFUSE_DEBUG        "true" for verbose logging
  CC_LANGFUSE_MAX_CHARS    truncation cap for prompt/response/tool bodies
                           (default 20000; original length + sha256 are
                           preserved in metadata)
  CC_TRACE_SUBAGENTS       "true" to also trace subagent transcripts;
                           default false skips them so the main
                           session's traces aren't crowded with nested
                           agent loops.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ENV_FILENAME = ".env"

# The five required variables this package reads.
# All names are CC_-prefixed so they cannot collide with the env vars of
# any other Langfuse-using service running in the same shell.
TRACE_ENABLED_VAR = "CC_TRACE_TO_LANGFUSE"
PROJECT_NAME_VAR = "CC_PROJECT_NAME"
BASE_URL_VAR = "CC_LANGFUSE_BASE_URL"
PUBLIC_KEY_VAR = "CC_LANGFUSE_PUBLIC_KEY"
SECRET_KEY_VAR = "CC_LANGFUSE_SECRET_KEY"

DEBUG_VAR = "CC_LANGFUSE_DEBUG"
MAX_CHARS_VAR = "CC_LANGFUSE_MAX_CHARS"
TRACE_SUBAGENTS_VAR = "CC_TRACE_SUBAGENTS"

ALL_VARS = (
    TRACE_ENABLED_VAR,
    PROJECT_NAME_VAR,
    BASE_URL_VAR,
    PUBLIC_KEY_VAR,
    SECRET_KEY_VAR,
    DEBUG_VAR,
    MAX_CHARS_VAR,
    TRACE_SUBAGENTS_VAR,
)


DEFAULT_MAX_CHARS = 20_000


@dataclass(frozen=True)
class Config:
    project_root: Path
    env_path: Optional[Path]
    trace_enabled: bool
    project_name: str
    langfuse_base_url: str
    langfuse_public_key: str
    langfuse_secret_key: str
    debug: bool
    max_chars: int
    trace_subagents: bool

    @property
    def is_complete(self) -> bool:
        return bool(
            self.trace_enabled
            and self.langfuse_base_url
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    def missing_fields(self) -> list[str]:
        out: list[str] = []
        if not self.langfuse_base_url:
            out.append(BASE_URL_VAR)
        if not self.langfuse_public_key:
            out.append(PUBLIC_KEY_VAR)
        if not self.langfuse_secret_key:
            out.append(SECRET_KEY_VAR)
        return out


def find_env_file(start: Path) -> Optional[Path]:
    """Walk up from `start` looking for a `.env` file.

    Stops at the user's home directory and at filesystem-root markers
    so we never adopt a stray `/.env` or `~/.env` as "the project root."
    A `.git` directory short-circuits the walk — that's the project
    boundary in practice.
    """
    start = start.resolve()
    try:
        home = Path.home().resolve()
    except (RuntimeError, OSError):
        home = None
    for directory in [start, *start.parents]:
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break
        if home is not None and directory == home:
            break
    return None


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal `.env` parser.

    Supports: `KEY=value`, `KEY="value"`, `KEY='value'`, leading
    `export `, `#` comments, blank lines. Does NOT support multi-line
    values, variable interpolation, or shell escapes — keep it simple
    and predictable.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve(start: Optional[Path] = None) -> Config:
    """Locate the project root, layer .env over env vars, return Config."""
    start = (start or Path.cwd()).resolve()
    env_path = find_env_file(start)

    file_vars: dict[str, str] = {}
    project_root = start
    if env_path is not None:
        file_vars = parse_env_file(env_path)
        project_root = env_path.parent

    def pick(name: str) -> str:
        """OS env wins; .env is the fallback. No aliases — CC_-prefixed only."""
        v = os.environ.get(name)
        if v:
            return v
        return file_vars.get(name, "")

    # Parse max_chars defensively — bad value falls back to the default.
    raw_max = pick(MAX_CHARS_VAR)
    try:
        max_chars = int(raw_max) if raw_max else DEFAULT_MAX_CHARS
    except ValueError:
        max_chars = DEFAULT_MAX_CHARS

    return Config(
        project_root=project_root,
        env_path=env_path,
        trace_enabled=_as_bool(pick(TRACE_ENABLED_VAR)),
        project_name=pick(PROJECT_NAME_VAR) or "unknown-project",
        langfuse_base_url=pick(BASE_URL_VAR),
        langfuse_public_key=pick(PUBLIC_KEY_VAR),
        langfuse_secret_key=pick(SECRET_KEY_VAR),
        debug=_as_bool(pick(DEBUG_VAR)),
        max_chars=max_chars,
        trace_subagents=_as_bool(pick(TRACE_SUBAGENTS_VAR)),
    )
