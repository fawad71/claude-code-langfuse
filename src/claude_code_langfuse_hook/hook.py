"""Stop-hook entry point.

Wired in `~/.claude/settings.json` as:

    {"type": "command", "command": "claude-langfuse hook"}

For each Stop event Claude Code fires, we:

  1. Parse the JSON payload from stdin (session_id, transcript_path, cwd).
  2. Resolve project config — env vars + optional `.env` walked up from cwd.
  3. Resolve user_id from `git config user.email` (fallback: $USER).
  4. Acquire an exclusive lock on the state file.
  5. Load the SessionState for this (session, transcript) pair —
     `offset` (bytes already processed), `buffer` (partial last line),
     `turn_count` (running total).
  6. Read only the new bytes since `offset`, parse them into messages,
     assemble messages into Turn objects with dedup.
  7. Emit one Langfuse trace per Turn, incrementing turn_count.
  8. Persist the updated state atomically.
  9. Flush + shutdown the Langfuse client.

Every step is wrapped in fail-open guards: any exception is logged to
`~/.claude/state/claude_langfuse.log` and the process exits 0 so
Claude Code never sees a hook failure (PRD R8).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from . import config as config_mod
from . import identity, state as state_mod, tracer, transcript

LOG_PATH = Path.home() / ".claude" / "state" / "claude_langfuse.log"


def _setup_logging(debug: bool = False) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _read_payload() -> dict:
    if sys.stdin.isatty():
        return {}
    try:
        return json.load(sys.stdin)
    except Exception as exc:
        logging.warning("Could not parse hook stdin payload: %s", exc)
        return {}


def _extract_session_and_transcript(payload: dict) -> tuple[Optional[str], Optional[Path]]:
    """Tolerate both camelCase and snake_case across hook payload versions."""
    session_id = (
        payload.get("session_id")
        or payload.get("sessionId")
        or (payload.get("session") or {}).get("id")
    )
    transcript_path_str = (
        payload.get("transcript_path")
        or payload.get("transcriptPath")
        or (payload.get("transcript") or {}).get("path")
    )
    transcript_path = (
        Path(transcript_path_str).expanduser().resolve()
        if transcript_path_str
        else None
    )
    return session_id, transcript_path


def _resolve_project_dir(payload: dict) -> Path:
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        return Path(env_dir)
    cwd = payload.get("cwd")
    if cwd:
        return Path(cwd)
    return Path.cwd()


def run() -> int:
    """Programmatic entry point. Always returns 0 (fail-open)."""
    start = time.time()
    _setup_logging()
    log = logging.getLogger("claude_code_langfuse_hook.hook")

    langfuse = None
    try:
        payload = _read_payload()
        project_dir = _resolve_project_dir(payload)
        cfg = config_mod.resolve(project_dir)

        # Re-init logging at the requested level once we know the config.
        if cfg.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        if not cfg.trace_enabled:
            log.debug("Tracing not enabled in %s — skipping.", project_dir)
            return 0

        if not cfg.is_complete:
            log.warning(
                "Tracing enabled but missing fields: %s — skipping.",
                cfg.missing_fields(),
            )
            return 0

        session_id, transcript_path = _extract_session_and_transcript(payload)
        if not session_id or not transcript_path or not transcript_path.exists():
            log.warning(
                "Missing or non-existent session_id / transcript_path on payload — skipping."
            )
            return 0

        # Subagent transcripts live under .../subagents/... — they're
        # one-off helper sessions Claude spins up. Default to skipping
        # them so the main session's trace doesn't get crowded.
        if "subagents" in transcript_path.parts and not cfg.trace_subagents:
            log.debug(
                "Skipping subagent transcript %s (set CC_TRACE_SUBAGENTS=true to include).",
                transcript_path,
            )
            return 0

        # Lazy import — Langfuse SDK is only needed when we're actually tracing.
        try:
            from langfuse import Langfuse
        except ImportError as exc:
            log.warning("langfuse SDK not installed: %s — skipping.", exc)
            return 0

        try:
            langfuse = Langfuse(
                public_key=cfg.langfuse_public_key,
                secret_key=cfg.langfuse_secret_key,
                host=cfg.langfuse_base_url,
            )
        except Exception as exc:
            log.error("Failed to initialize Langfuse client: %s", exc)
            return 0

        user_id = identity.resolve_user_id(cfg.project_root)

        # ---- locked state read / write -------------------------------
        emitted = 0
        with state_mod.FileLock() as lock:
            if not lock.acquired:
                log.warning(
                    "Could not acquire state lock — skipping this fire to "
                    "avoid split-brain writes; turns will be picked up on the next Stop."
                )
                return 0
            state = state_mod.load_state()
            key = state_mod.state_key(session_id, str(transcript_path))
            ss = state_mod.load_session_state(state, key)

            new_msgs, ss = state_mod.read_new_jsonl(transcript_path, ss)

            # Merge any rows from prior fires that didn't form a
            # complete turn yet, then split off the still-incomplete
            # trailing turn (if any) to carry forward again.
            all_msgs = list(ss.pending_msgs) + new_msgs
            commit_msgs, pending_msgs = transcript.split_for_commit(all_msgs)
            ss.pending_msgs = pending_msgs

            if not commit_msgs:
                state_mod.write_session_state(state, key, ss)
                state_mod.save_state(state)
                log.debug(
                    "No committable turns — pending=%d, exiting.",
                    len(pending_msgs),
                )
                return 0

            turns = transcript.build_turns(commit_msgs)
            if not turns:
                state_mod.write_session_state(state, key, ss)
                state_mod.save_state(state)
                log.debug("No complete turns in new bytes — state updated, exiting.")
                return 0

            for t in turns:
                turn_num = ss.turn_count + emitted + 1
                try:
                    tracer.emit_turn(
                        langfuse=langfuse,
                        cfg=cfg,
                        user_id=user_id,
                        session_id=session_id,
                        turn_num=turn_num,
                        turn=t,
                        transcript_path=transcript_path,
                    )
                    emitted += 1
                except Exception:
                    log.error("emit_turn failed:\n%s", traceback.format_exc())
                    # Keep going — one bad turn shouldn't sink the rest.

            ss.turn_count += emitted
            state_mod.write_session_state(state, key, ss)
            state_mod.save_state(state)

        try:
            langfuse.flush()
        except Exception as exc:
            log.warning("langfuse.flush() failed: %s", exc)

        duration = time.time() - start
        log.info(
            "Processed %d turns in %.2fs (session=%s, user=%s, project=%s)",
            emitted, duration, session_id, user_id, cfg.project_name,
        )
        if duration > 180:
            log.warning("Hook took %.1fs (>3min) — consider optimizing.", duration)

    except Exception:
        log.error("Hook failed:\n%s", traceback.format_exc())
    finally:
        if langfuse is not None:
            try:
                langfuse.shutdown()
            except Exception:
                pass

    return 0
