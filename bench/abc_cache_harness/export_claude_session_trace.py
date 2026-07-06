#!/usr/bin/env python3
"""Export real Claude Code user turns into replay_harness request JSONL.

The source transcript stays local. The generated trace contains only selected
human-readable user turns plus deterministic assistant placeholders, with the
same tool schema attached to every request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM = """You are a coding agent working in a local repository.
Use the shell tool when the user asks you to inspect files, verify status,
search, run commands, check logs, benchmark, poll progress, or gather evidence.
When using a tool, respond with exactly one tool_use block and no prose."""


SHELL_TOOL = {
    "name": "shell",
    "description": "Run one read-only shell command in the repository.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


def user_text_from_record(record: dict) -> str | None:
    """Return benchmark-relevant human user text, excluding command/meta rows."""
    if record.get("type") != "user":
        return None
    msg = record.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None

    content = msg.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("text", None):
                parts.append(str(block.get("text", "")))
        text = "\n".join(parts)

    text = text.strip()
    if not text or text.startswith("<"):
        return None
    if text.startswith("[tool_result"):
        return None
    if text.startswith("[Image:"):
        return None
    return text


def extract_user_turns(path: Path) -> list[str]:
    turns: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = user_text_from_record(record)
            if text is not None:
                turns.append(text)
    return turns


def assistant_placeholder(turn_idx: int) -> str:
    return (
        f"Recorded benchmark history placeholder after turn {turn_idx}. "
        "Continue using the repository context and call tools when needed."
    )


def build_trace(
    user_turns: list[str],
    *,
    start: int,
    turns: int,
    max_tokens: int,
    temperature: float,
    model: str,
) -> list[dict]:
    selected = user_turns[start : start + turns]
    if len(selected) < turns:
        raise SystemExit(
            f"requested {turns} turns from offset {start}, only {len(selected)} available"
        )

    history: list[dict] = []
    trace: list[dict] = []
    for idx, user_text in enumerate(selected, start=1):
        messages = [*history, {"role": "user", "content": user_text}]
        trace.append(
            {
                "model": model,
                "system": SYSTEM,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "tools": [SHELL_TOOL],
            }
        )
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_placeholder(idx)})
    return trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--turns", type=int, default=38)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model", default="luce-dflash")
    args = ap.parse_args()

    source_turns = extract_user_turns(args.session)
    trace = build_trace(
        source_turns,
        start=args.start,
        turns=args.turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model=args.model,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for request in trace:
            f.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(
        f"wrote {len(trace)} turns to {args.out} "
        f"from {args.session} start={args.start} source_turns={len(source_turns)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

