"""Standalone repro for the empty-response correctness regression in
dflash/scripts/server_tools.py + dflash/scripts/prefix_cache.py.

Sends a SEQUENCE of multi-turn chat completions to one warm dflash server.
Each call extends the conversation by one full
  assistant(tool_call) → tool(result) → user(continue)
turn, so the prefix grows naturally and the cache populates inline
snapshots between calls — the same pattern that
dflash/scripts/bench_agent_loop.py walks through transcripts.

Then runs the SAME sequence against a slots=0 server (cache disabled)
as the control.

The tool result is generated programmatically (synthetic pylint-like
output, deterministic per turn index) so this script has no dependency
on any private session transcript.

Expected: every call produces non-empty content. Observed against
PR #59: starting at some call N (often N=3 around 12-15K char prefix),
warm responses become content_len=0 / reasoning_len=0 / completion_tokens=0,
finish_reason="stop". The first broken call still does the prefill (slow
wall time) but emits nothing; subsequent calls hit the cache and return
empty in <50 ms.

Usage:
    python3 repro_empty_response.py \\
        --target /path/to/Qwen3.6-27B*.gguf \\
        --draft  /path/to/qwen3.6-27b-dflash \\
        --bin    /path/to/dflash/build/test_dflash \\
        --server /path/to/dflash/scripts/server_tools.py
"""
import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Synthetic prompt (deterministic, no external data) ─────────────────

SYSTEM = "You are a helpful Python code reviewer."
INITIAL_USER = ("Run pylint on each subdirectory of src/ in turn and "
                "summarise the unused-variable warnings.")


def long_tool_result(target_chars: int, seed: int) -> str:
    """Synthetic pylint-style output, deterministic per `seed`."""
    head = f"pylint report (subdir #{seed}):\n"
    pieces = [head]
    n = len(head)
    i = 0
    while n < target_chars:
        line = (f"  [{seed:02d}-{i:05d}] src/sub_{seed}/module_{i:04d}.py:"
                f"{(i * 13) % 9999}: lint warning: unused variable "
                f"'tmp_{i}_value' (column {i % 80})\n")
        pieces.append(line)
        n += len(line)
        i += 1
    return "".join(pieces)


def build_call_sequence(n_turns: int, tool_chars: int) -> list[list[dict]]:
    """Return prefixes[0..n_turns-1] where each prefix is the messages array
    sent at call index i, and prefix[i+1] = prefix[i] + (recorded asst turn,
    tool result, next user). Mirrors how bench_agent_loop walks transcripts.
    """
    base: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": INITIAL_USER},
    ]
    prefixes = [list(base)]
    for i in range(1, n_turns):
        tool_id = f"call_{i:03d}"
        # Append a "recorded" assistant turn (tool_call) and its tool result,
        # then the next user message — exactly what the transcript replay
        # would do after receiving a server response.
        base = base + [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": tool_id, "type": "function",
                              "function": {"name": "run_pylint",
                                           "arguments": json.dumps(
                                               {"path": f"src/sub_{i}/"})}}]},
            {"role": "tool", "tool_call_id": tool_id,
             "content": long_tool_result(tool_chars, seed=i)},
            {"role": "user",
             "content": f"Continue with subdir #{i + 1} please."},
        ]
        prefixes.append(list(base))
    return prefixes


# ── HTTP plumbing ──────────────────────────────────────────────────────

def call_chat(port: int, messages: list, n_gen: int = 64,
              timeout: int = 600) -> dict:
    """Non-streaming POST. Returns dict with content, reasoning, finish,
    completion_tokens, dt_s, in_chars."""
    in_chars = sum(len(m.get("content") or "") for m in messages
                   if isinstance(m.get("content"), str))
    in_chars += sum(len((tc.get("function") or {}).get("arguments") or "")
                    for m in messages for tc in (m.get("tool_calls") or []))
    body = json.dumps({"model": "luce-dflash", "messages": messages,
                       "max_tokens": n_gen, "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    dt = time.perf_counter() - t0
    body_json = json.loads(raw)
    msg = body_json["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": body_json["choices"][0].get("finish_reason"),
        "completion_tokens": (body_json.get("usage") or {}).get("completion_tokens"),
        "dt_s": dt,
        "in_chars": in_chars,
    }


def wait_up(port: int, proc: subprocess.Popen, timeout: int = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                   timeout=2).read()
            return True
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(1)
    return False


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Bench driver ───────────────────────────────────────────────────────

def is_empty(r: dict) -> bool:
    """A response counts as empty if no content, no reasoning, and no
    tool_calls were produced. completion_tokens=0 is necessary but not
    sufficient (some servers under-report)."""
    return (not r["content"] and not r["reasoning"]
            and not r["tool_calls"])


def run_one(label: str, slots: int, port: int, args,
            prefixes: list) -> list[dict]:
    log_path = Path(f"/tmp/repro_{label}.log")
    log_f = open(log_path, "w")
    cmd = [sys.executable, "-u", str(args.server),
           "--target", str(args.target), "--draft", str(args.draft),
           "--bin", str(args.bin), "--max-ctx", str(args.max_ctx),
           "--port", str(port), "--prefix-cache-slots", str(slots)]
    import os as _os
    env = _os.environ.copy()
    env["GGML_NO_BACKTRACE"] = "1"
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    print(f"\n=== {label}: slots={slots}, port={port}, "
          f"calls={len(prefixes)} ===", flush=True)
    if not wait_up(port, proc):
        log_f.close()
        tail = log_path.read_text()[-2000:]
        stop(proc)
        raise RuntimeError(f"{label}: server didn't come up\n{tail}")
    results: list[dict] = []
    try:
        # Tiny warmup, discarded
        call_chat(port, [{"role": "user", "content": "ok"}], n_gen=4)
        for i, msgs in enumerate(prefixes, start=1):
            try:
                r = call_chat(port, msgs, n_gen=args.n_gen)
            except Exception as e:
                print(f"  call {i}: ERROR {e}", flush=True)
                results.append({"call": i, "error": str(e)})
                continue
            results.append(r)
            tag = " *** EMPTY ***" if is_empty(r) else ""
            print(f"  call {i}: in={r['in_chars']:>7,}  "
                  f"dt={r['dt_s']:>6.2f}s  comp_tok={r['completion_tokens']}  "
                  f"content_len={len(r['content']):>4}  "
                  f"reasoning_len={len(r['reasoning']):>4}  "
                  f"finish={r['finish']!r}{tag}", flush=True)
    finally:
        stop(proc)
        log_f.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--draft",  required=True, type=Path)
    ap.add_argument("--bin",    required=True, type=Path)
    ap.add_argument("--server", required=True, type=Path,
                    help="Path to dflash/scripts/server_tools.py")
    ap.add_argument("--max-ctx", type=int, default=24576)
    ap.add_argument("--n-turns", type=int, default=6,
                    help="Number of growing-prefix calls (default %(default)s)")
    ap.add_argument("--tool-chars", type=int, default=5000,
                    help="Approx chars of synthetic tool result per turn "
                         "(default %(default)s). Real-session triggers "
                         "appeared at ~14K char total prefix by turn 3.")
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2,
                    help="--prefix-cache-slots for warm config (default %(default)s)")
    args = ap.parse_args()

    prefixes = build_call_sequence(args.n_turns, args.tool_chars)
    sizes = [sum(len(m.get("content") or "") for m in p
                 if isinstance(m.get("content"), str))
             for p in prefixes]
    print(f"{args.n_turns} growing-prefix calls; "
          f"in_chars per call: {sizes}")

    cold = run_one("cold", slots=0, port=18290, args=args,
                   prefixes=prefixes)
    warm = run_one("warm", slots=args.slots, port=18291, args=args,
                   prefixes=prefixes)

    print("\n=== summary ===")
    print(f"  {'call':>4}  {'in_chars':>9}  "
          f"{'cold dt':>8} {'cold tok':>9} {'cold len':>9}  "
          f"{'warm dt':>8} {'warm tok':>9} {'warm len':>9}  flag")
    cold_empty = warm_empty = 0
    for n, (c, w) in enumerate(zip(cold, warm), start=1):
        if c.get("error") or w.get("error"):
            print(f"  {n:>4}  ERROR cold={c.get('error')} warm={w.get('error')}")
            continue
        flag = ""
        if is_empty(c):
            cold_empty += 1; flag += " COLD-EMPTY"
        if is_empty(w):
            warm_empty += 1; flag += " WARM-EMPTY"
        print(f"  {n:>4}  {c['in_chars']:>9,}  "
              f"{c['dt_s']:>7.2f}s {c['completion_tokens']:>9} "
              f"{len(c['content']):>9}  "
              f"{w['dt_s']:>7.2f}s {w['completion_tokens']:>9} "
              f"{len(w['content']):>9} {flag}")

    print(f"\n  cold empty responses: {cold_empty}/{len(cold)}")
    print(f"  warm empty responses: {warm_empty}/{len(warm)}")

    if cold_empty:
        print("\nUNEXPECTED: cold (slots=0) produced empty responses — "
              "different problem than the cache regression.")
        return 2
    if warm_empty == 0:
        print("\nDID NOT REPRO: every warm call produced output. "
              "Try increasing --n-turns or --tool-chars to push past the "
              "trigger threshold (real-session bug fired at ~14K chars / "
              "3rd multi-turn call).")
        return 1

    print("\nREPRO CONFIRMED:")
    print(f"  - Same {len(prefixes)} growing-prefix calls produced output "
          f"on slots=0")
    print(f"  - With prefix-cache slots={args.slots}, "
          f"{warm_empty} of {len(warm)} warm calls returned an empty body")
    print(f"  - The pattern is: one slow 'broken' call (real prefill, "
          f"empty output), then cascade of <100ms empty hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
