"""`claude-langfuse` CLI.

Commands:
  install   — register the Stop hook in ~/.claude/settings.json (per machine)
  uninstall — remove it
  init      — print the env-var block to drop into your .env / .env.example
  status    — diagnose: hook registered? env vars resolved? identity?
  test      — send a synthetic trace to verify Langfuse connectivity
  hook      — internal: the Stop-hook entry point Claude Code calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, config as config_mod, hook as hook_mod

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
HOOK_COMMAND = "claude-langfuse hook"
HOOK_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# install / uninstall — wires the Stop hook globally
# ---------------------------------------------------------------------------
def _load_settings() -> dict:
    if CLAUDE_SETTINGS.exists():
        try:
            return json.loads(CLAUDE_SETTINGS.read_text())
        except json.JSONDecodeError:
            print(
                f"warning: {CLAUDE_SETTINGS} was malformed; rewriting.",
                file=sys.stderr,
            )
        except OSError as exc:
            print(
                f"warning: could not read {CLAUDE_SETTINGS} ({exc}); using empty config.",
                file=sys.stderr,
            )
    return {}


def _save_settings(settings: dict) -> None:
    # Atomic write — a crash mid-write must never corrupt the user's
    # primary Claude Code config.
    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLAUDE_SETTINGS.with_suffix(CLAUDE_SETTINGS.suffix + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n")
    os.replace(tmp, CLAUDE_SETTINGS)


def _is_our_entry(entry: dict) -> bool:
    return any(
        HOOK_COMMAND in (h.get("command") or "")
        for h in entry.get("hooks", [])
    )


def cmd_install(_args: argparse.Namespace) -> int:
    settings = _load_settings()
    settings.setdefault("hooks", {}).setdefault("Stop", [])
    settings["hooks"]["Stop"] = [
        e for e in settings["hooks"]["Stop"] if not _is_our_entry(e)
    ]
    settings["hooks"]["Stop"].append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": HOOK_COMMAND,
                    "timeout": HOOK_TIMEOUT_SECONDS,
                }
            ],
        }
    )
    _save_settings(settings)
    print(f"✓ Registered Stop hook in {CLAUDE_SETTINGS}")
    print("  Next: add the env vars to your project's .env (see `claude-langfuse init`).")
    return 0


def cmd_uninstall(_args: argparse.Namespace) -> int:
    if not CLAUDE_SETTINGS.exists():
        print(f"Nothing to do — {CLAUDE_SETTINGS} doesn't exist.")
        return 0
    settings = _load_settings()
    stop = settings.get("hooks", {}).get("Stop", [])
    before = len(stop)
    settings.setdefault("hooks", {})["Stop"] = [e for e in stop if not _is_our_entry(e)]
    if len(settings["hooks"]["Stop"]) == before:
        print("No claude-langfuse Stop hook registered — nothing to remove.")
        return 0
    _save_settings(settings)
    print(f"✓ Removed Stop hook from {CLAUDE_SETTINGS}")
    return 0


# ---------------------------------------------------------------------------
# init — print the env-var block to copy into .env / .env.example
# ---------------------------------------------------------------------------
ENV_TEMPLATE = """\
# --- Claude Code → Langfuse tracing ---
# Required for claude-code-langfuse-hook. All names use the CC_ prefix
# so they can't collide with the env vars of any other Langfuse-using
# service in your repo. Commit the non-secret lines to .env.example;
# keep the keys out of git unless your repo is private.

CC_TRACE_TO_LANGFUSE=true
CC_PROJECT_NAME=REPLACE_ME
CC_LANGFUSE_BASE_URL=https://langfuse.internal.example.com
CC_LANGFUSE_PUBLIC_KEY=
CC_LANGFUSE_SECRET_KEY=
"""


def cmd_init(_args: argparse.Namespace) -> int:
    print("Add these to your project's .env (and commit the non-secret lines to .env.example):\n")
    print(ENV_TEMPLATE)
    print(
        "The hook picks them up from the OS env or from a .env file walked up "
        "from your project root.\n"
        "If you use direnv / doppler / vault, set them however you normally do — "
        "OS env always wins."
    )
    return 0


# ---------------------------------------------------------------------------
# status — diagnostics
# ---------------------------------------------------------------------------
def _mask(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-3:]}"


def cmd_status(args: argparse.Namespace) -> int:
    print(f"claude-code-langfuse-hook v{__version__}\n")

    # 1. Global hook registration
    print(f"Global hook ({CLAUDE_SETTINGS}):")
    settings = _load_settings() if CLAUDE_SETTINGS.exists() else {}
    stop = settings.get("hooks", {}).get("Stop", [])
    registered = any(_is_our_entry(e) for e in stop)
    print(f"  registered: {'yes' if registered else 'NO — run `claude-langfuse install`'}")

    # 2. Resolved config
    cfg = config_mod.resolve(Path.cwd())
    print("\nResolved config:")
    if cfg.env_path:
        print(f"  .env file:     {cfg.env_path}")
    else:
        print("  .env file:     (none found in cwd or parents — values must come from OS env)")
    print(f"  project_name:  {cfg.project_name}")
    print(f"  trace_enabled: {cfg.trace_enabled}")
    print(f"  base_url:      {cfg.langfuse_base_url or '(empty)'}")
    print(f"  public_key:    {_mask(cfg.langfuse_public_key)}")
    print(f"  secret_key:    {_mask(cfg.langfuse_secret_key)}")

    # 3. Identity
    from . import identity

    user_id = identity.resolve_user_id(cfg.project_root)
    print(f"\nIdentity:\n  user_id: {user_id}")

    ready = registered and cfg.trace_enabled and cfg.is_complete
    print(f"\nReady to trace: {'YES' if ready else 'no'}")
    if not ready and cfg.trace_enabled and not cfg.is_complete:
        print(f"  Missing: {', '.join(cfg.missing_fields())}")
    if getattr(args, "exit_zero", False):
        return 0
    return 0 if ready else 1


# ---------------------------------------------------------------------------
# test — synthetic trace
# ---------------------------------------------------------------------------
def cmd_test(_args: argparse.Namespace) -> int:
    cfg = config_mod.resolve(Path.cwd())
    if not cfg.is_complete:
        print(
            f"Config incomplete: missing {cfg.missing_fields()}. "
            "Set them in your .env or shell env.",
            file=sys.stderr,
        )
        return 1

    # Same v3 SDK pattern the hook itself uses — propagate_attributes
    # sets session_id / user_id / tags on the implicit trace, and
    # start_as_current_observation creates the root span.
    from langfuse import Langfuse, propagate_attributes

    from . import identity

    client = Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
        host=cfg.langfuse_base_url,
    )
    user_id = identity.resolve_user_id(cfg.project_root)
    trace_name = f"claude-langfuse:test:{cfg.project_name}"

    try:
        with propagate_attributes(
            session_id="claude-langfuse-cli-test",
            user_id=user_id,
            trace_name=trace_name,
            tags=[f"project:{cfg.project_name}", "synthetic", "claude-code"],
        ):
            with client.start_as_current_observation(
                name=trace_name,
                input={"role": "user", "content": "ping"},
                metadata={"source": "claude-langfuse test"},
            ) as span:
                span.update(output={"role": "assistant", "content": "pong"})
        client.flush()
    finally:
        try:
            client.shutdown()
        except Exception:
            pass

    print("✓ Sent synthetic trace.")
    print(f"  Look for it at: {cfg.langfuse_base_url}")
    print(f"  Trace name:     {trace_name}")
    print(f"  user_id:        {user_id}")
    print(f"  session_id:     claude-langfuse-cli-test")
    return 0


# ---------------------------------------------------------------------------
# hook — the Stop-hook entry point Claude Code calls
# ---------------------------------------------------------------------------
def cmd_hook(_args: argparse.Namespace) -> int:
    return hook_mod.run()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-langfuse",
        description="Claude Code → Langfuse tracer.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("install", help="Register the Stop hook in ~/.claude/settings.json").set_defaults(func=cmd_install)
    sub.add_parser("uninstall", help="Remove the Stop hook").set_defaults(func=cmd_uninstall)
    sub.add_parser("init", help="Print the env-var block to drop into your .env").set_defaults(func=cmd_init)
    status_p = sub.add_parser("status", help="Diagnose installation + resolved config")
    status_p.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit 0, even when not ready to trace (useful for interactive runs).",
    )
    status_p.set_defaults(func=cmd_status)
    sub.add_parser("test", help="Send a synthetic trace to verify connectivity").set_defaults(func=cmd_test)
    sub.add_parser("hook", help="Internal: Stop-hook entry point").set_defaults(func=cmd_hook)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
