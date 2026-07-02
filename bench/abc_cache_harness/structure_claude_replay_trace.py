#!/usr/bin/env python3
"""Convert Claude Code replay/session traces into structured tool-call traces.

`build_trace_from_session.py` preserved context depth by flattening Claude Code
tool calls into text such as `[tool_use name=Bash input={...}]`. That is useful
for cache/deep-context replay, but it teaches chat-template models to imitate
Claude's textual marker instead of emitting structured Qwen/OpenAI tool calls.

This converter can either repair those flattened traces or build a clean
growing-prefix trace directly from raw Claude session JSONL. In both modes it
rewrites historical tool interactions into OpenAI-style `assistant.tool_calls`
and `tool` messages, labels whether the recorded next assistant turn used a
tool, and applies exact `tool_choice` on expected tool turns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any


TOOL_RESULT_RE = re.compile(
    r"^\[tool_result id=([^\s\]]+)(?:\s+is_error=true)?\s*([\s\S]*?)\]$"
)
TOOL_USE_PREFIX = "[tool_use name="
TOOL_USE_INPUT = " input="


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": True,
    }


def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": object_schema(properties, required),
        },
    }


TOOLS = [
    tool("Bash", "Run a shell command.", {
        "command": {"type": "string"},
        "description": {"type": "string"},
    }, ["command"]),
    tool("Read", "Read a local file.", {
        "file_path": {"type": "string"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
    }, ["file_path"]),
    tool("Write", "Write a local file.", {
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    }, ["file_path", "content"]),
    tool("Edit", "Edit a local file.", {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"},
    }, ["file_path", "old_string", "new_string"]),
    tool("Agent", "Launch a sub-agent for parallel work.", {
        "description": {"type": "string"},
        "prompt": {"type": "string"},
        "subagent_type": {"type": "string"},
        "run_in_background": {"type": "boolean"},
    }, ["description", "prompt"]),
    tool("Workflow", "Run a structured local workflow script.", {
        "script": {"type": "string"},
    }, ["script"]),
    tool("TaskOutput", "Wait for or fetch output from a background task.", {
        "task_id": {"type": "string"},
        "block": {"type": "boolean"},
        "timeout": {"type": "integer"},
    }, ["task_id"]),
    tool("TaskCreate", "Create a tracked task.", {
        "subject": {"type": "string"},
        "description": {"type": "string"},
    }, ["subject"]),
    tool("TaskUpdate", "Update a tracked task.", {
        "taskId": {"type": "string"},
        "status": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "activeForm": {"type": "string"},
        "blockedBy": {"type": "string"},
    }, ["taskId"]),
    tool("AskUserQuestion", "Ask the user a blocking or clarifying question.", {
        "question": {"type": "string"},
        "options": {"type": "array"},
    }, ["question"]),
]


def tool_def_name(tool_def: dict) -> str | None:
    if not isinstance(tool_def, dict):
        return None
    if isinstance(tool_def.get("function"), dict):
        name = tool_def["function"].get("name")
        return str(name) if name else None
    name = tool_def.get("name")
    return str(name) if name else None


def replay_tool(name: str) -> dict:
    return tool(
        name,
        "Replay schema for a tool observed in the source Claude session.",
        {},
    )


def tool_names_from_messages(messages: list[dict]) -> set[str]:
    names: set[str] = set()
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                names.add(str(fn["name"]))
    return names


def with_observed_tools(rows: list[dict]) -> list[dict]:
    """Ensure every expected/history tool name is present in each row schema."""
    observed: set[str] = set()
    for row in rows:
        expected = row.get("expected_tool_name")
        if expected:
            observed.add(str(expected))
        observed.update(tool_names_from_messages(row.get("messages", [])))

    known = {name for name in (tool_def_name(t) for t in TOOLS) if name}
    extra = [replay_tool(name) for name in sorted(observed - known)]
    if not extra:
        return rows

    expanded = [*TOOLS, *extra]
    for row in rows:
        row["tools"] = expanded
    return rows


def normalize_tool_args(value: Any) -> str:
    if not isinstance(value, dict):
        value = {"value": value}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype in ("text", "input_text", "output_text"):
            parts.append(str(block.get("text", "")))
        elif btype is None and "text" in block:
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)


def load_system_prompt(system_trace: Path) -> str:
    with system_trace.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                system = row.get("system", "")
                if not isinstance(system, str) or not system:
                    raise SystemExit(f"{system_trace} has no non-empty system prompt")
                return system
    raise SystemExit(f"{system_trace} is empty")


def extract_tool_uses(text: str) -> tuple[str, list[dict]]:
    calls: list[dict] = []
    parts: list[str] = []
    decoder = json.JSONDecoder()
    pos = 0

    while True:
        start = text.find(TOOL_USE_PREFIX, pos)
        if start < 0:
            parts.append(text[pos:])
            break

        name_start = start + len(TOOL_USE_PREFIX)
        input_sep = text.find(TOOL_USE_INPUT, name_start)
        if input_sep < 0:
            parts.append(text[pos:])
            break

        name = text[name_start:input_sep]
        json_start = input_sep + len(TOOL_USE_INPUT)
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1

        try:
            value, json_end_rel = decoder.raw_decode(text[json_start:])
        except json.JSONDecodeError:
            parts.append(text[pos:start])
            pos = start + len(TOOL_USE_PREFIX)
            continue

        json_end = json_start + json_end_rel
        if json_end >= len(text) or text[json_end] != "]":
            parts.append(text[pos:start])
            pos = start + len(TOOL_USE_PREFIX)
            continue

        parts.append(text[pos:start])
        calls.append({
            "name": name,
            "arguments": normalize_tool_args(value),
        })
        pos = json_end + 1

    return "".join(parts).strip(), calls


def convert_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending_ids: deque[str] = deque()
    call_seq = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            out.append(msg)
            continue

        if role == "assistant":
            text, calls = extract_tool_uses(content)
            if calls:
                tool_calls = []
                for call in calls:
                    call_seq += 1
                    call_id = f"call_{call_seq}"
                    pending_ids.append(call_id)
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    })
                converted = {"role": "assistant", "content": text, "tool_calls": tool_calls}
                out.append(converted)
                continue

        if role == "user":
            m = TOOL_RESULT_RE.match(content.strip())
            if m:
                tool_call_id = pending_ids.popleft() if pending_ids else m.group(1)
                out.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": m.group(2),
                })
                continue

        out.append({"role": role, "content": content})

    return out


def first_new_tool_call(prev: list[dict], nxt: list[dict]) -> str | None:
    for msg in nxt[len(prev):]:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            _, calls = extract_tool_uses(content)
            if calls:
                return calls[0]["name"]
    return None


def assistant_message_from_claude(content: Any) -> dict | None:
    if isinstance(content, str):
        text, calls = extract_tool_uses(content)
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = [
                {
                    "id": f"call_flat_{i}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for i, call in enumerate(calls, start=1)
            ]
        return msg if text or calls else None

    if not isinstance(content, list):
        text = "" if content is None else str(content)
        return {"role": "assistant", "content": text} if text else None

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            call_id = str(block.get("id") or f"call_{len(tool_calls) + 1}")
            name = str(block.get("name") or "")
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": normalize_tool_args(block.get("input", {})),
                },
            })
        elif btype == "thinking":
            continue
        elif btype:
            text_parts.append(f"[{btype}]")

    text = "\n".join(p for p in text_parts if p).strip()
    if not text and not tool_calls:
        return None
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def user_messages_from_claude(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": "" if content is None else str(content)}]

    out: list[dict] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        text = "\n".join(p for p in text_parts if p).strip()
        text_parts.clear()
        if text:
            out.append({"role": "user", "content": text})

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type")
        if btype in ("text", None):
            text = block.get("text")
            if text is None and btype is None:
                text = block.get("content", "")
            text_parts.append(str(text or ""))
        elif btype == "tool_result":
            flush_text()
            out.append({
                "role": "tool",
                "tool_call_id": str(block.get("tool_use_id") or ""),
                "content": text_from_content(block.get("content", "")),
            })
        else:
            text_parts.append(text_from_content([block]) or f"[{btype}]")

    flush_text()
    return out


def merge_assistant_message(prev: dict, msg: dict) -> None:
    if msg.get("content"):
        if prev.get("content"):
            prev["content"] = f"{prev['content']}\n{msg['content']}"
        else:
            prev["content"] = msg["content"]
    if msg.get("tool_calls"):
        prev.setdefault("tool_calls", [])
        prev["tool_calls"].extend(msg["tool_calls"])


def raw_session_messages(rows: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for row in rows:
        if row.get("isSidechain"):
            continue
        msg = row.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            converted = assistant_message_from_claude(msg.get("content", ""))
            if converted is None:
                continue
            if messages and messages[-1].get("role") == "assistant":
                merge_assistant_message(messages[-1], converted)
            else:
                messages.append(converted)
        elif role == "user":
            messages.extend(user_messages_from_claude(msg.get("content", "")))
    return messages


def build_trace_from_raw_session(
    rows: list[dict],
    *,
    system_prompt: str,
    turns: int,
    max_tokens: int,
    model: str,
) -> list[dict]:
    history: list[dict] = []
    out_rows: list[dict] = []
    messages = raw_session_messages(rows)

    for msg in messages:
        if msg.get("role") == "assistant":
            if history and history[-1].get("role") in ("user", "tool"):
                tool_calls = msg.get("tool_calls") or []
                row = {
                    "model": model,
                    "system": system_prompt,
                    "messages": list(history),
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": False,
                    "tools": TOOLS,
                    "expect_tool_call": bool(tool_calls),
                }
                if tool_calls:
                    name = tool_calls[0].get("function", {}).get("name", "")
                    row["expected_tool_name"] = name
                    row["tool_choice"] = {"type": "function", "function": {"name": name}}
                out_rows.append(row)
                if len(out_rows) >= turns:
                    break
            history.append(msg)
        else:
            history.append(msg)

    return with_observed_tools(out_rows)


def build_trace_from_flattened(
    rows: list[dict],
    *,
    turns: int,
    max_tokens: int,
) -> list[dict]:
    selected = rows[:turns]
    out_rows = []
    for i, row in enumerate(selected):
        converted = dict(row)
        messages = row.get("messages", [])
        converted["messages"] = convert_messages(messages)
        converted["tools"] = TOOLS
        converted["max_tokens"] = max_tokens
        converted["temperature"] = 0
        converted["stream"] = False

        expected_tool = first_new_tool_call(
            row.get("messages", []),
            rows[i + 1].get("messages", []) if i + 1 < len(rows) else [],
        )
        converted["expect_tool_call"] = expected_tool is not None
        if expected_tool is not None:
            converted["expected_tool_name"] = expected_tool
            converted["tool_choice"] = {
                "type": "function",
                "function": {"name": expected_tool},
            }
        else:
            converted.pop("tool_choice", None)

        out_rows.append(converted)
    return with_observed_tools(out_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--turns", type=int, default=38)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--model", default="luce-dflash")
    ap.add_argument(
        "--source-kind",
        choices=("auto", "raw-session", "flattened-trace"),
        default="auto",
    )
    ap.add_argument(
        "--system-trace",
        type=Path,
        default=Path(__file__).resolve().parent / "traces" / "real_session_long_38.jsonl",
    )
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in args.src.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{args.src} is empty")

    source_kind = args.source_kind
    if source_kind == "auto":
        first = rows[0]
        source_kind = "raw-session" if "message" in first else "flattened-trace"

    if source_kind == "raw-session":
        system_prompt = load_system_prompt(args.system_trace)
        out_rows = build_trace_from_raw_session(
            rows,
            system_prompt=system_prompt,
            turns=args.turns,
            max_tokens=args.max_tokens,
            model=args.model,
        )
    else:
        out_rows = build_trace_from_flattened(
            rows,
            turns=args.turns,
            max_tokens=args.max_tokens,
        )

    if len(out_rows) < args.turns:
        raise SystemExit(
            f"requested {args.turns} turns from {args.src}, produced {len(out_rows)}"
        )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    expected = sum(1 for row in out_rows if row.get("expect_tool_call"))
    max_chars = max(len(json.dumps(row.get("messages", []), ensure_ascii=False)) for row in out_rows)
    print(f"wrote {len(out_rows)} {source_kind} rows to {args.dst}")
    print(f"expected_tool_calls={expected}")
    print(f"max_messages_chars={max_chars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
