"""Python side of the A1 tool-parser program: `parse(text, tools)` -> {"cleaned_text", "tool_calls": [{"id","name","arguments"}]}.

The binary is built from the server's tool_parser.cpp (tool_parser/Makefile). Path: $A1_TOOL_PARSER, else
$A1_TP_BUILD/tool_parser_cli, else tool_parser/build/tool_parser_cli next to this file. One JSON line in, one JSON line out.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def binary_path():
    if os.environ.get("A1_TOOL_PARSER"):
        return os.environ["A1_TOOL_PARSER"]
    return os.path.join(os.environ.get("A1_TP_BUILD") or os.path.join(HERE, "tool_parser", "build"), "tool_parser_cli")


def parse_batch(items, binary=None):
    """items: [(text, tools)] -> [result dict] (same order), one process for the whole batch."""
    lines = "".join(json.dumps({"text": t, "tools": tools}) + "\n" for t, tools in items)
    r = subprocess.run([binary or binary_path()], input=lines, capture_output=True, text=True, check=True)
    out = [json.loads(l) for l in r.stdout.splitlines()]
    if len(out) != len(items):
        raise RuntimeError(f"tool_parser_cli returned {len(out)} lines for {len(items)} inputs")
    return out


def parse(text, tools, binary=None):
    return parse_batch([(text, tools)], binary)[0]


def to_message(parsed):
    """Parser result -> the OpenAI-style message fields the harness scorer reads (content, tool_calls)."""
    tcs = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
           for c in parsed["tool_calls"]]
    return {"content": parsed["cleaned_text"], "tool_calls": tcs or None}
