"""Langfuse dispatch — emit one trace per turn using the v3 SDK.

Schema we produce per turn:

  trace                (named "Claude Code - Turn N", session_id, user_id, tags)
    └─ root span       "Claude Code - Turn N"  (input = user text)
        ├─ generation  "Claude Response"        (model, input/output, token usage)
        └─ tool span   "Tool: <name>"           (input + output, one per call)

Each turn becomes its own Langfuse trace; the shared `session_id`
glues them together in the Sessions view. Large prompt / response /
tool-output bodies are truncated; metadata records the original length
and a sha256 so identity is recoverable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import Config
from .transcript import (
    Turn,
    extract_text,
    extract_thinking,
    get_content,
    get_model,
    get_usage,
    iter_tool_uses,
    truncate_text,
    truncate_value,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool call assembly — pulls tool_use blocks out of the assistant messages
# and pairs them with their tool_result counterparts by id.
# ---------------------------------------------------------------------------
def _tool_calls_for_turn(turn: Turn, max_chars: int) -> list[dict]:
    calls: list[dict] = []
    seen_ids: set[str] = set()
    for am in turn.assistant_msgs:
        for tu in iter_tool_uses(get_content(am)):
            tid = str(tu.get("id") or "")
            if tid and tid in seen_ids:
                continue
            if tid:
                seen_ids.add(tid)
            tool_input = tu.get("input")
            trunc_input, in_meta = truncate_value(tool_input, max_chars)

            raw_output = turn.tool_results_by_id.get(tid)
            if raw_output is None:
                trunc_output: Any = None
                out_meta = None
            elif isinstance(raw_output, str):
                trunc_output, out_meta = truncate_text(raw_output, max_chars)
                if not out_meta.get("truncated"):
                    out_meta = None
            else:
                try:
                    as_str = json.dumps(raw_output, ensure_ascii=False)
                except (TypeError, ValueError):
                    as_str = str(raw_output)
                if len(as_str) <= max_chars:
                    trunc_output, out_meta = raw_output, None
                else:
                    trunc_output, out_meta = truncate_text(as_str, max_chars)

            calls.append({
                "id": tid,
                "name": tu.get("name") or "unknown",
                "input": trunc_input,
                "input_meta": in_meta,
                "output": trunc_output,
                "output_meta": out_meta,
            })
    return calls


def _sum_usage(assistant_msgs: list[dict]) -> dict:
    """Sum Anthropic usage across every assistant call in a turn.

    Each assistant message in `turn.assistant_msgs` corresponds to one
    Anthropic API call inside the agent loop, and each carries its own
    `usage` block. We sum the four token categories so the generation's
    `usage_details` reflects the whole turn, not just the final step.
    """
    totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    seen_ids: set[str] = set()
    for am in assistant_msgs:
        # Skip duplicate-id rows: streamed updates of the same message
        # carry cumulative usage on the latest row, so summing every
        # row with the same id would double-count.
        m = am.get("message") or {}
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
        u = m.get("usage") or {}
        for k in totals:
            v = u.get(k)
            try:
                totals[k] += int(v or 0)
            except (TypeError, ValueError):
                pass
    return totals


# ---------------------------------------------------------------------------
# Per-turn emission
# ---------------------------------------------------------------------------
def emit_turn(
    *,
    langfuse,
    cfg: Config,
    user_id: str,
    session_id: str,
    turn_num: int,
    turn: Turn,
    transcript_path: Path,
) -> None:
    """Emit one Langfuse trace for `turn`.

    The trace name encodes the turn number so each session reads as a
    natural sequence in the Langfuse UI.
    """
    from langfuse import propagate_attributes

    user_text_raw = extract_text(get_content(turn.user_msg))
    user_text, user_text_meta = truncate_text(user_text_raw, cfg.max_chars)

    last_assistant = turn.assistant_msgs[-1]
    assistant_text_raw = extract_text(get_content(last_assistant))
    assistant_text, assistant_text_meta = truncate_text(assistant_text_raw, cfg.max_chars)

    # Extended thinking — capture across all assistant messages in the turn
    # since long reasoning may be split across streamed message ids.
    thinking_raw = "\n".join(
        filter(None, (extract_thinking(get_content(am)) for am in turn.assistant_msgs))
    )
    thinking_text, thinking_text_meta = (
        truncate_text(thinking_raw, cfg.max_chars) if thinking_raw else ("", None)
    )

    model = get_model(turn.assistant_msgs[0])
    # A turn is an agent loop — each assistant message is its own
    # Anthropic API call with its own usage block. Sum across all of
    # them so cache reads / cache creations / input / output tokens
    # from intermediate tool-calling steps aren't dropped. Without
    # this, a turn that hits Read/Grep/etc. several times reports
    # only the *last* step's tokens, badly under-counting cost.
    usage = _sum_usage(turn.assistant_msgs)
    tool_calls = _tool_calls_for_turn(turn, cfg.max_chars)

    trace_name = f"Claude Code - Turn {turn_num}"
    # Build a composite `user_project:` tag (e.g.
    # "user_project:muhammad.fawad.ext-karzaty_api") so a single Langfuse
    # dashboard widget can group by user × project in one dimension.
    user_local = user_id.split("@", 1)[0] if "@" in user_id else user_id
    tags = [
        f"project:{cfg.project_name}",
        f"user_project:{user_local}-{cfg.project_name}",
        f"model:{model}",
        "claude-code",
    ]

    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=tags,
    ):
        with langfuse.start_as_current_observation(
            name=trace_name,
            input={"role": "user", "content": user_text},
            metadata={
                "source": "claude-code",
                "project": cfg.project_name,
                "session_id": session_id,
                "turn_number": turn_num,
                "transcript_path": str(transcript_path),
                "user_text": user_text_meta,
                "tool_call_count": len(tool_calls),
            },
        ) as trace_span:
            # The actual LLM call.
            # Pass Anthropic's native usage field names verbatim so the
            # keys match Langfuse's default model-price catalog exactly
            # (input_tokens, output_tokens, cache_creation_input_tokens,
            # cache_read_input_tokens). Renaming them would force every
            # user to add custom price aliases in Settings → Models.
            def _as_int(v: Any) -> int:
                try:
                    return int(v or 0)
                except (TypeError, ValueError):
                    return 0

            input_tokens = _as_int(usage.get("input_tokens"))
            output_tokens = _as_int(usage.get("output_tokens"))
            cache_creation = _as_int(usage.get("cache_creation_input_tokens"))
            cache_read = _as_int(usage.get("cache_read_input_tokens"))
            usage_details: dict[str, int] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_creation:
                usage_details["cache_creation_input_tokens"] = cache_creation
            if cache_read:
                usage_details["cache_read_input_tokens"] = cache_read

            generation_metadata: dict[str, Any] = {
                "assistant_text": assistant_text_meta,
                "stop_reason": (last_assistant.get("message") or {}).get("stop_reason"),
                "tool_count": len(tool_calls),
            }
            if thinking_raw:
                generation_metadata["thinking"] = thinking_text
                generation_metadata["thinking_meta"] = thinking_text_meta

            with langfuse.start_as_current_observation(
                name="Claude Response",
                as_type="generation",
                model=model,
                input={"role": "user", "content": user_text},
                output={"role": "assistant", "content": assistant_text},
                usage_details=usage_details,
                metadata=generation_metadata,
            ):
                pass

            # One observation per tool call.
            for tc in tool_calls:
                with langfuse.start_as_current_observation(
                    name=f"Tool: {tc['name']}",
                    as_type="tool",
                    input=tc["input"],
                    metadata={
                        "tool_name": tc["name"],
                        "tool_id": tc["id"],
                        "input_meta": tc["input_meta"],
                        "output_meta": tc["output_meta"],
                    },
                ) as tool_obs:
                    tool_obs.update(output=tc["output"])

            trace_span.update(output={"role": "assistant", "content": assistant_text})

    log.info(
        "Emitted turn user=%s project=%s session=%s turn=%d tools=%d "
        "in=%d out=%d cache_create=%d cache_read=%d thinking=%d",
        user_id,
        cfg.project_name,
        session_id,
        turn_num,
        len(tool_calls),
        input_tokens,
        output_tokens,
        cache_creation,
        cache_read,
        len(thinking_raw),
    )
