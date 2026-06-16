#!/usr/bin/env python3
"""DiffusionGemma vs dense Gemma4 agentic protocol benchmark.

This is the first gate for making DiffusionGemma beat dense Gemma4 on long
agentic coding sessions. It deliberately measures the failure mode we found:
large code payloads inside structured tool arguments are fragile, while plain
content/code-block output plus a small metadata tool call is the intended
protocol. The strongest candidate path is two-turn: generate a diff as content
with no tools available, then make a small apply call after the harness has the
diff text.

The script runs one model at a time through dflash_server's OpenAI-compatible
/v1/chat/completions path. Use --self-test to validate scoring logic without
loading any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_BIN = ROOT / "server" / "build" / "dflash_server"
DEFAULT_DIFFUSION_MODEL = Path(
    "/home/peppi/models/diffusiongemma-26b/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
)
DEFAULT_DENSE_MODEL = Path(
    "/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)
DEFAULT_OUT = Path("/tmp/dg_agentic_pair/summary.json")

LEAN_SYSTEM = """You are a coding agent. Solve the task by editing files safely.
Use the provided tools exactly. Keep code changes minimal. Do not invent files.
If the protocol asks for a diff in content, put the full unified diff in your
assistant content and use the tool call only for metadata."""


@dataclass(frozen=True)
class Task:
    name: str
    path: str
    prompt: str
    required_fragments: tuple[str, ...]


TASKS = [
    Task(
        name="stable_dedupe",
        path="src/stable_dedupe.py",
        prompt=(
            "Create a Python function stable_dedupe(items) that returns a list "
            "with duplicates removed while preserving first occurrence order. "
            "It must handle unhashable items by equality comparison."
        ),
        required_fragments=("def stable_dedupe", "seen", "result"),
    ),
    Task(
        name="parse_kv_lines",
        path="src/parse_kv.py",
        prompt=(
            "Create parse_kv_lines(text) that parses KEY=VALUE lines, ignores "
            "blank lines and lines starting with '#', strips whitespace, and "
            "raises ValueError for non-comment lines without '='."
        ),
        required_fragments=("def parse_kv_lines", "ValueError", "split"),
    ),
    Task(
        name="retry_backoff",
        path="src/retry_backoff.py",
        prompt=(
            "Create retry_backoff(delays, max_delay) that yields cumulative "
            "retry delays capped at max_delay and never mutates the input."
        ),
        required_fragments=("def retry_backoff", "yield", "max_delay"),
    ),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def command_status(name: str) -> str:
    path = shutil.which(name)
    return path if path else "missing"


def context_seed() -> str:
    candidates = [
        ROOT / "server" / "src" / "server" / "tool_parser.cpp",
        ROOT / "server" / "src" / "diffusion" / "diffusion_decoder.cpp",
        ROOT / "harness" / "clients" / "common.sh",
        ROOT / "thoughts" / "RECAP_2026-06-15_diffusion_diffusiongemma.md",
    ]
    chunks: list[str] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            continue
        chunks.append(f"### read_file {path.relative_to(ROOT)}\n{text[:12000]}")
    return "\n\n".join(chunks) or "No local context seed files were found."


def expand_context(target_chars: int) -> str:
    seed = context_seed()
    parts: list[str] = []
    while sum(len(p) for p in parts) < target_chars:
        idx = len(parts) + 1
        parts.append(f"\n\n<tool_result tool=\"read_file\" idx=\"{idx}\">\n{seed}\n</tool_result>")
    return "".join(parts)[:target_chars]


def protocol_tools(protocol: str) -> list[dict[str, Any]]:
    if protocol == "json_arg":
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write complete file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]
    if protocol in {"content_diff", "two_turn_diff"}:
        return [
            {
                "type": "function",
                "function": {
                    "name": "apply_content_patch",
                    "description": "Apply the unified diff emitted in assistant content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "format": {"type": "string", "enum": ["unified_diff"]},
                            "content_sha256": {"type": "string"},
                        },
                        "required": ["path", "format"],
                    },
                },
            }
        ]
    raise ValueError(f"unknown protocol: {protocol}")


def build_messages(task: Task, protocol: str, context_chars: int) -> list[dict[str, str]]:
    context = expand_context(context_chars)
    if protocol == "json_arg":
        request = (
            f"{task.prompt}\n\n"
            f"Target path: {task.path}\n"
            "Return exactly one write_file tool call. Put the entire source code "
            "in write_file.content."
        )
    elif protocol == "content_diff":
        request = (
            f"{task.prompt}\n\n"
            f"Target path: {task.path}\n"
            "Return a unified diff in assistant content. Then call "
            "apply_content_patch with only path, format='unified_diff', and "
            "optionally content_sha256 of the emitted diff. Do not put source "
            "code or patch text inside tool arguments."
        )
    elif protocol == "two_turn_diff":
        request = (
            f"{task.prompt}\n\n"
            f"Target path: {task.path}\n"
            "Return only a unified diff in assistant content. Do not call any "
            "tools in this turn. The harness will apply the diff after a "
            "separate metadata confirmation turn."
        )
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return [
        {"role": "system", "content": LEAN_SYSTEM},
        {
            "role": "user",
            "content": (
                "The following is long agentic session context gathered from prior "
                f"tool calls. Use it as background only.\n{context}\n\n{request}"
            ),
        },
    ]


def post_chat(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def wait_healthy(base_url: str, proc: subprocess.Popen[str], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with rc={proc.returncode}")
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2).read()
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy at {base_url}")


class Server:
    def __init__(self, args: argparse.Namespace, label: str, model: Path, model_id: str):
        self.args = args
        self.label = label
        self.model = model
        self.model_id = model_id
        self.proc: subprocess.Popen[str] | None = None
        self.log_path = Path(args.out).with_suffix(f".{label}.server.log")

    @property
    def base_url(self) -> str:
        return f"http://{self.args.host}:{self.args.port}"

    def __enter__(self) -> "Server":
        if self.args.no_launch:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = self.log_path.open("w", encoding="utf-8")
        cmd = [
            "flock",
            self.args.gpu_lock,
            str(self.args.server_bin),
            str(self.model),
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            "--max-ctx",
            str(self.args.max_ctx),
            "--max-tokens",
            str(self.args.max_tokens),
            "--model-name",
            self.model_id,
            "--prefix-cache-slots",
            str(self.args.prefix_cache_slots),
        ]
        if self.args.kv_cache_dir:
            cmd.extend(["--kv-cache-dir", self.args.kv_cache_dir])
        if self.args.extra_server_args:
            cmd.extend(self.args.extra_server_args)
        log(f"[{self.label}] starting: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_f.close()
        wait_healthy(self.base_url, self.proc, self.args.startup_timeout)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def parse_tool_args(call: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    fn = call.get("function", {}) if isinstance(call, dict) else {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return name, raw_args, True
    if not isinstance(raw_args, str):
        return name, {}, False
    try:
        args = json.loads(raw_args) if raw_args.strip() else {}
        return name, args if isinstance(args, dict) else {}, isinstance(args, dict)
    except json.JSONDecodeError:
        return name, {}, False


def extract_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {"content": "", "tool_calls": []}
    msg = choices[0].get("message") or {}
    return msg if isinstance(msg, dict) else {"content": "", "tool_calls": []}


def has_diff(text: str) -> bool:
    return (
        "```diff" in text
        or "diff --git " in text
        or ("--- " in text and "+++ " in text and "@@" in text)
    )


def score_message(task: Task, protocol: str, message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    parsed = [parse_tool_args(call) for call in tool_calls if isinstance(call, dict)]
    arg_lengths = {
        key: len(str(value))
        for _, args, ok in parsed
        if ok
        for key, value in args.items()
        if key in {"content", "code", "patch", "diff"}
    }
    max_arg_payload = max(arg_lengths.values(), default=0)
    required_in_content = all(fragment in content for fragment in task.required_fragments)

    if protocol == "json_arg":
        expected = [p for p in parsed if p[0] == "write_file"]
        args_ok = bool(expected and expected[0][2])
        payload = expected[0][1].get("content", "") if expected else ""
        payload_text = payload if isinstance(payload, str) else str(payload)
        required_in_payload = all(fragment in payload_text for fragment in task.required_fragments)
        passed = bool(args_ok and len(payload_text) >= 160 and required_in_payload)
        return {
            "passed": passed,
            "tool_call_valid": args_ok,
            "expected_tool_seen": bool(expected),
            "content_chars": len(content),
            "json_content_chars": len(payload_text),
            "empty_long_code_write": bool(expected and len(payload_text.strip()) < 32),
            "required_fragments_seen": required_in_payload,
            "max_arg_payload_chars": max_arg_payload,
        }

    expected = [p for p in parsed if p[0] == "apply_content_patch"]
    args_ok = bool(expected and expected[0][2])
    diff_ok = has_diff(content)
    light_args_ok = max_arg_payload <= 256
    passed = bool(args_ok and diff_ok and light_args_ok)
    return {
        "passed": passed,
        "tool_call_valid": args_ok,
        "expected_tool_seen": bool(expected),
        "content_chars": len(content),
        "content_diff_seen": diff_ok,
        "required_fragments_seen": required_in_content,
        "lightweight_args": light_args_ok,
        "max_arg_payload_chars": max_arg_payload,
    }


def score_two_turn(
    task: Task,
    first_message: dict[str, Any],
    second_message: dict[str, Any],
) -> dict[str, Any]:
    first_content = first_message.get("content") or ""
    first_tool_calls = first_message.get("tool_calls") or []
    second_tool_calls = second_message.get("tool_calls") or []
    parsed = [parse_tool_args(call) for call in second_tool_calls if isinstance(call, dict)]
    expected = [p for p in parsed if p[0] == "apply_content_patch"]
    arg_payloads = [
        len(str(value))
        for _, args, ok in parsed
        if ok
        for key, value in args.items()
        if key in {"content", "code", "patch", "diff"}
    ]
    max_arg_payload = max(arg_payloads, default=0)
    diff_ok = has_diff(first_content)
    required_seen = all(fragment in first_content for fragment in task.required_fragments)
    args_ok = bool(expected and expected[0][2])
    no_first_tool = not first_tool_calls
    light_args_ok = max_arg_payload <= 256
    return {
        "passed": bool(diff_ok and args_ok and no_first_tool and light_args_ok),
        "first_turn_no_tool_calls": no_first_tool,
        "tool_call_valid": args_ok,
        "expected_tool_seen": bool(expected),
        "content_chars": len(first_content),
        "content_diff_seen": diff_ok,
        "required_fragments_seen": required_seen,
        "lightweight_args": light_args_ok,
        "max_arg_payload_chars": max_arg_payload,
    }


def cache_stats(log_path: Path) -> dict[str, int]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    patterns = {
        "disk_hit": r"disk_hit=true",
        "restore": r"restore=true|restore delta|snapshot_adopt",
        "inline_snap": r"inline-snap",
        "cache_save": r"snapshot saved|snapshot_save",
    }
    return {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}


def run_case(
    server: Server,
    task: Task,
    protocol: str,
    context_chars: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if protocol == "two_turn_diff":
        return run_two_turn_case(server, task, context_chars, args)

    payload = {
        "model": server.model_id,
        "messages": build_messages(task, protocol, context_chars),
        "tools": protocol_tools(protocol),
        "tool_choice": "auto",
        "temperature": 0,
        "stream": False,
        "max_tokens": args.max_tokens,
        "extra_body": {"enable_thinking": False},
    }
    start = time.perf_counter()
    response = post_chat(server.base_url, payload, args.request_timeout)
    wall_s = time.perf_counter() - start
    message = extract_message(response)
    score = score_message(task, protocol, message)
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {
        "task": task.name,
        "protocol": protocol,
        "context_chars": context_chars,
        "wall_s": wall_s,
        "usage": usage,
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "score": score,
        "message_preview": (message.get("content") or "")[:500],
        "tool_calls": message.get("tool_calls") or [],
    }


def run_two_turn_case(
    server: Server,
    task: Task,
    context_chars: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = build_messages(task, "two_turn_diff", context_chars)
    first_payload = {
        "model": server.model_id,
        "messages": messages,
        "temperature": 0,
        "stream": False,
        "max_tokens": args.max_tokens,
        "extra_body": {"enable_thinking": False},
    }
    start = time.perf_counter()
    first_response = post_chat(server.base_url, first_payload, args.request_timeout)
    first_wall_s = time.perf_counter() - start
    first_message = extract_message(first_response)
    diff_text = first_message.get("content") or ""
    digest = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()

    second_messages = [
        *messages,
        {"role": "assistant", "content": diff_text},
        {
            "role": "user",
            "content": (
                "Now call apply_content_patch with path, format='unified_diff', "
                f"and content_sha256={digest}. Do not repeat the diff."
            ),
        },
    ]
    second_payload = {
        "model": server.model_id,
        "messages": second_messages,
        "tools": protocol_tools("two_turn_diff"),
        "tool_choice": "auto",
        "temperature": 0,
        "stream": False,
        "max_tokens": min(args.max_tokens, 256),
        "extra_body": {"enable_thinking": False},
    }
    start = time.perf_counter()
    second_response = post_chat(server.base_url, second_payload, args.request_timeout)
    second_wall_s = time.perf_counter() - start
    second_message = extract_message(second_response)
    score = score_two_turn(task, first_message, second_message)
    return {
        "task": task.name,
        "protocol": "two_turn_diff",
        "context_chars": context_chars,
        "wall_s": first_wall_s + second_wall_s,
        "turn_wall_s": {"first": first_wall_s, "second": second_wall_s},
        "usage": {
            "first": first_response.get("usage", {}),
            "second": second_response.get("usage", {}),
        },
        "finish_reason": {
            "first": (first_response.get("choices") or [{}])[0].get("finish_reason"),
            "second": (second_response.get("choices") or [{}])[0].get("finish_reason"),
        },
        "score": score,
        "message_preview": diff_text[:500],
        "tool_calls": second_message.get("tool_calls") or [],
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n": 0, "pass_rate": 0.0}
    passed = sum(1 for r in results if r["score"].get("passed"))
    wall = sum(float(r.get("wall_s", 0.0)) for r in results)
    empty_writes = sum(1 for r in results if r["score"].get("empty_long_code_write"))
    valid_tools = sum(1 for r in results if r["score"].get("tool_call_valid"))
    return {
        "n": len(results),
        "pass_rate": passed / len(results),
        "tool_valid_rate": valid_tools / len(results),
        "empty_long_code_writes": empty_writes,
        "wall_s_total": wall,
        "wall_s_mean": wall / len(results),
    }


def run_backend(label: str, model: Path, model_id: str, args: argparse.Namespace) -> dict[str, Any]:
    protocols = (
        ["json_arg", "content_diff", "two_turn_diff"]
        if args.protocol == "both"
        else [args.protocol]
    )
    contexts = [args.context_chars] if args.context_chars else args.context_buckets
    all_results: list[dict[str, Any]] = []
    with Server(args, label, model, model_id) as server:
        for context_chars in contexts:
            for protocol in protocols:
                for task in TASKS[: args.tasks]:
                    for _ in range(args.repeats):
                        log(
                            f"[{label}] task={task.name} protocol={protocol} "
                            f"context_chars={context_chars}"
                        )
                        all_results.append(run_case(server, task, protocol, context_chars, args))
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for result in all_results:
        by_protocol.setdefault(result["protocol"], []).append(result)
    return {
        "label": label,
        "model": str(model),
        "results": all_results,
        "aggregate": aggregate(all_results),
        "by_protocol": {k: aggregate(v) for k, v in by_protocol.items()},
        "cache_stats": cache_stats(Path(args.out).with_suffix(f".{label}.server.log")),
    }


def self_test() -> int:
    json_empty = {
        "content": "",
        "tool_calls": [
            {"function": {"name": "write_file", "arguments": json.dumps({"path": "x.py", "content": ""})}}
        ],
    }
    diff_text = """```diff
--- /dev/null
+++ b/src/stable_dedupe.py
@@
+def stable_dedupe(items):
+    seen = []
+    result = []
+    return result
```"""
    diff_ok = {
        "content": diff_text,
        "tool_calls": [
            {
                "function": {
                    "name": "apply_content_patch",
                    "arguments": json.dumps(
                        {
                            "path": "src/stable_dedupe.py",
                            "format": "unified_diff",
                            "content_sha256": hashlib.sha256(diff_text.encode()).hexdigest(),
                        }
                    ),
                }
            }
        ],
    }
    task = TASKS[0]
    scores = {
        "json_empty": score_message(task, "json_arg", json_empty),
        "content_diff": score_message(task, "content_diff", diff_ok),
        "two_turn_diff": score_two_turn(task, {"content": diff_text}, diff_ok),
    }
    print(json.dumps(scores, indent=2))
    if not scores["json_empty"]["empty_long_code_write"]:
        print("self-test failed: empty write was not detected", file=sys.stderr)
        return 1
    if scores["json_empty"]["passed"]:
        print("self-test failed: empty write passed", file=sys.stderr)
        return 1
    if not scores["content_diff"]["passed"]:
        print("self-test failed: content diff did not pass", file=sys.stderr)
        return 1
    if not scores["two_turn_diff"]["passed"]:
        print("self-test failed: two-turn diff did not pass", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    p.add_argument("--diffusion-model", type=Path, default=DEFAULT_DIFFUSION_MODEL)
    p.add_argument("--dense-model", type=Path, default=DEFAULT_DENSE_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18200)
    p.add_argument("--max-ctx", type=int, default=86016)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--prefix-cache-slots", type=int, default=8)
    p.add_argument("--kv-cache-dir", default="")
    p.add_argument("--gpu-lock", default="/tmp/dg_gpu.lock")
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument(
        "--protocol",
        choices=["json_arg", "content_diff", "two_turn_diff", "both"],
        default="both",
    )
    p.add_argument("--context-buckets", type=int, nargs="+", default=[32000, 96000, 256000])
    p.add_argument("--context-chars", type=int, default=0)
    p.add_argument("--tasks", type=int, default=len(TASKS))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--backend", choices=["both", "diffusion", "dense"], default="both")
    p.add_argument("--no-launch", action="store_true", help="use an already-running server")
    p.add_argument("--extra-server-args", nargs="*", default=[])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    args.server_bin = args.server_bin.resolve()
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    env = {
        "kimi": command_status("kimi"),
        "opencode": command_status("opencode"),
        "server_bin": str(args.server_bin),
        "root": str(ROOT),
    }
    if not args.no_launch:
        for label, path in {
            "server_bin": args.server_bin,
            "diffusion_model": args.diffusion_model,
            "dense_model": args.dense_model,
        }.items():
            if args.backend != "both" and label == "dense_model" and args.backend != "dense":
                continue
            if args.backend != "both" and label == "diffusion_model" and args.backend != "diffusion":
                continue
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    report: dict[str, Any] = {
        "environment": env,
        "criteria": {
            "primary": "DiffusionGemma pass_rate >= dense Gemma4",
            "tie_break": "If pass rates tie, DiffusionGemma wall time <= dense / 1.5",
            "tool_valid_rate": ">= 0.98",
            "empty_long_code_writes": 0,
        },
        "backends": {},
    }
    if args.backend in {"both", "diffusion"}:
        report["backends"]["diffusion"] = run_backend(
            "diffusion", args.diffusion_model, "diffusion-gemma", args
        )
    if args.backend in {"both", "dense"}:
        report["backends"]["dense"] = run_backend("dense", args.dense_model, "dense-gemma4", args)

    if "diffusion" in report["backends"] and "dense" in report["backends"]:
        dg = report["backends"]["diffusion"]["aggregate"]
        dense = report["backends"]["dense"]["aggregate"]
        dg_beats_quality = dg["pass_rate"] >= dense["pass_rate"]
        dg_beats_time = (
            dg["wall_s_total"] > 0
            and dense["wall_s_total"] / dg["wall_s_total"] >= 1.5
        )
        report["verdict"] = {
            "diffusion_pass_rate": dg["pass_rate"],
            "dense_pass_rate": dense["pass_rate"],
            "quality_gate_met": dg_beats_quality,
            "time_gate_met": dg_beats_time,
            "beats_dense": bool(dg_beats_quality and (dg["pass_rate"] > dense["pass_rate"] or dg_beats_time)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"[bench] wrote {args.out}")
    print(json.dumps(report.get("verdict", report["backends"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
