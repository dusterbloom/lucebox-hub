#!/usr/bin/env python3
"""Model-level forward differentials for the Qwen3.8-Flash-Next backend.

Self-contained A/B checks that a kernel unit test cannot cover because they
need the real weights and graph marking:

  hc16   LLAMA_MMB_HC16=2 (fused bf16 gate-mix / HC-down) vs =0 (f32 path) must
         generate the same greedy continuation.
  qsa    a prompt prefilled in one chunk vs split over smaller chunks must
         generate the same greedy continuation.
  mtp    MTP speculative decode (sidecar auto-discovered from <repo>/MTP/, or
         QWEN4EXP_MTP=<path>) vs --verify-width 1 must give byte-identical greedy
         replies, short and past the 2,052-token QSA budget; prints the
         acceptance rate and decode tok/s of both.

Both run on the gfx1151 prefill profile the backend defaults (QSA, HC16, MMB).

Requires the model and a built luce_server; run on the gfx1151 box:

    python3 server/scripts/qwen4exp_forward_diff.py hc16
    python3 server/scripts/qwen4exp_forward_diff.py qsa --chunks 4096,16384
    QWEN4EXP_MODEL=<UD-Q4_K_XL.gguf> python3 server/scripts/qwen4exp_forward_diff.py mtp --draft auto,1,2,3,4

Not part of ctest: model-backed and long-running.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = os.environ.get("QWEN4EXP_SERVER", str(REPO / "server/build-hip/luce_server"))
MODEL = os.environ.get("QWEN4EXP_MODEL", str(Path.home() / "models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf"))
PORT = int(os.environ.get("QWEN4EXP_DIFF_PORT", "8711"))

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the engineer studies the "
    "kernel launch counters and the cache residency of every intermediate tensor. "
    "Numbers follow numbers until the context is long enough to matter. "
)
# A long filler context with an unambiguous tail question, so greedy output is
# robust to the small numerical differences chunk boundaries introduce.
LONG_PROMPT = PARAGRAPH * 400 + "\n\nAnswer with a single word. What is the capital of France?"


def log(msg: str) -> None:
    print(msg, flush=True)


def chat_full(prompt: str, max_tokens: int, timeout: int, temperature: float = 0, seed: int = 0,
              thinking: bool | None = None) -> dict:
    body = json.dumps({
        "model": "luce",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
        **({"chat_template_kwargs": {"enable_thinking": thinking}} if thinking is not None else {}),
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(prompt: str, max_tokens: int, timeout: int) -> str:
    return chat_full(prompt, max_tokens, timeout)["choices"][0]["message"]["content"]


def wait_ready(proc: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def run_server(name: str, env_overrides: dict[str, str], extra_args: list[str],
               prompts: list[str], max_tokens: int, temperature: float = 0, seed: int = 0,
               thinking: list[bool] | None = None) -> list[dict]:
    """Spawn one server and return the full /v1/chat/completions response of each prompt, in order."""
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
    time.sleep(0.5)
    cmd = [SERVER_BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT),
           "--target-device", "hip:0", "--max-ctx", "40000", *extra_args]
    env = {**os.environ, **env_overrides}
    log(f"[{name}] spawn: {' '.join(cmd)}  env+={env_overrides} {extra_args}")
    logf = open(f"/tmp/qwen4exp_diff_{name}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(proc, timeout=600):
            raise RuntimeError(f"server did not become ready; see /tmp/qwen4exp_diff_{name}.log")
        log(f"[{name}] ready")
        replies = [chat_full(prompt, max_tokens, timeout=1800, temperature=temperature, seed=seed,
                             thinking=thinking[i] if thinking is not None else None)
                   for i, prompt in enumerate(prompts)]
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        logf.close()
    return replies


def run_config(name: str, env_overrides: dict[str, str], extra_args: list[str],
               prompt: str, max_tokens: int) -> str:
    return run_server(name, env_overrides, extra_args, [prompt], max_tokens)[0]["choices"][0]["message"]["content"]


def common_prefix_ratio(a: str, b: str) -> float:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n / max(1, max(len(a), len(b)))


def cmd_hc16(args: argparse.Namespace) -> int:
    # >512 prefill tokens so the HC marking (ggml_nrows(xn) >= 512) and the
    # HC-down GEMM (T >= mmb_min_t() = 512) actually engage.
    prompt = PARAGRAPH * 30 + "\n\nAnswer with a single word. What is the capital of France?"
    replies = {}
    for name, hc in (("hc16_on", "2"), ("hc16_off", "0")):
        replies[name] = run_config(name, {"LLAMA_MMB_HC16": hc}, ["--chunk", "16384"], prompt, args.max_tokens)
    same = replies["hc16_on"].strip() == replies["hc16_off"].strip()
    ratio = common_prefix_ratio(replies["hc16_on"], replies["hc16_off"])
    log(f"[hc16] on={replies['hc16_on']!r}")
    log(f"[hc16] off={replies['hc16_off']!r}")
    log(f"[hc16] exact={same} common_prefix={ratio:.3f}")
    return 0 if same or ratio >= 0.9 else 1


def cmd_qsa(args: argparse.Namespace) -> int:
    results = {}
    for chunk in [int(c) for c in args.chunks.split(",")]:
        name = f"qsa_chunk{chunk}"
        results[chunk] = run_config(name, {}, ["--chunk", str(chunk)], LONG_PROMPT, args.max_tokens)
        log(f"[qsa] chunk={chunk} reply={results[chunk][:80]!r}")
    chunks = sorted(results)
    base = results[chunks[-1]].strip()
    ok = True
    for chunk in chunks:
        reply = results[chunk].strip()
        if reply != base:
            ratio = common_prefix_ratio(reply, base)
            log(f"[qsa] chunk={chunk} reply differs from chunk={chunks[-1]} common_prefix={ratio:.3f}")
            ok = ok and ratio >= 0.9
    log(f"[qsa] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_mtp(args: argparse.Namespace) -> int:
    # Greedy output must not change with speculation: the verify forward reproduces plain decode bit for bit.
    # "cross" starts below the QSA block budget (2,052 tokens) and decodes across it; "long" decodes past it.
    question = "\n\nSummarize the text above in one sentence."
    prompts = {
        "warmup": "Write a short poem about the sea.",
        "short": "Write a short poem about the sea.",
        "cross": PARAGRAPH * 45 + question,
        "long": PARAGRAPH * 70 + question,
    }
    thinking = None
    if args.workloads:
        # Representative prompts, not the original external benchmark corpus.
        prompts = {
            "warmup": "Write a short poem about the sea.",
            "json": "Output only JSON: an array of 80 objects with id, name, city, and score fields.",
            "list": "List the integers from 1 to 200, one per line, with each integer's English name.",
            "code": "Write a complete Python LRU cache implementation with TTL and unit tests.",
            "math": "Derive the sum of squares formula, prove it by induction, then work through examples.",
            "translate": "Translate into Italian, preserving all detail:\n" + PARAGRAPH * 20,
            "explain": "Explain how a relational database executes joins, with examples and tradeoffs.",
            "reasoning": "Find all positive integers n for which n squared plus 2 divides n factorial plus 1. Prove your answer.",
            "summarize": "Summarize the following in a detailed structured report:\n" + PARAGRAPH * 70,
            "prose": "Write a long literary story about a lighthouse keeper who receives letters from the future.",
        }
        thinking = [name == "reasoning" for name in prompts]
    # Exercise the public flags; avoid an inherited legacy fixed-width override
    # silently turning the adaptive run into fixed k. Preserve sidecar discovery.
    env = {"QWEN4EXP_MTP_DRAFT": ""}
    off_run = run_server("mtp_off", env, ["--verify-width", "1"], list(prompts.values()),
                         args.max_tokens, args.temperature, args.seed, thinking)
    ok = True
    for k in args.draft:
        mode = f"mtp_{k}" if k == "auto" else f"mtp_k{k}"
        flags = ["--verify-width", str(0 if k == "auto" else int(k) + 1)]
        if os.environ.get("QWEN4EXP_MTP") not in (None, "", "0"):
            flags += ["--draft", os.environ["QWEN4EXP_MTP"]]
        on_run = run_server(mode, env, flags, list(prompts.values()),
                            args.max_tokens, args.temperature, args.seed, thinking)
        on_log = Path(f"/tmp/qwen4exp_diff_{mode}.log").read_text(errors="replace").splitlines()
        if not any("[qwen4exp] MTP sidecar:" in line for line in on_log):
            log(f"[mtp] FAIL: {mode} loaded no sidecar (set QWEN4EXP_MTP=<mtp-*.gguf>)")
            return 1
        if not any("[qwen4exp-mtp]" in line and
                   ("adaptive=1" in line if k == "auto" else f"k={k} drafts=" in line) for line in on_log):
            log(f"[mtp] FAIL: {mode} did not run speculative decoding")
            return 1
        for i, name in enumerate(prompts):
            off, on = off_run[i], on_run[i]
            text = [(r["choices"][0]["message"].get("reasoning_content") or "") + "\0" +
                    (r["choices"][0]["message"].get("content") or "") for r in (off, on)]
            usage = on.get("usage", {})
            tps = [r.get("usage", {}).get("timings", {}).get("decode_tokens_per_sec", 0.0) for r in (off, on)]
            same = text[0] == text[1] and off.get("usage", {}).get("completion_tokens") == usage.get("completion_tokens")
            ok = ok and same
            if name != "warmup":
                ran = usage.get("spec_decode_ran") is True and off.get("usage", {}).get("spec_decode_ran") is False
                ok = ok and ran
                if not ran:
                    log(f"[mtp] FAIL: {mode} {name} unexpected spec_decode_ran usage")
            log(f"[mtp] k={k} {name}: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
                f"exact={same} common_prefix={common_prefix_ratio(text[0], text[1]):.3f} "
                f"accept_rate={usage.get('accept_rate')} decode tok/s off={tps[0]} on={tps[1]} "
                f"speedup={tps[1] / tps[0] if tps[0] else 0.0:.3f}")
        for line in on_log:
            if "[qwen4exp-mtp]" in line:
                log(line.strip())
    log(f"[mtp] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def draft_list(value: str) -> list[str]:
    values = value.split(",")
    if any(k not in ("auto", "1", "2", "3", "4") for k in values):
        raise argparse.ArgumentTypeError("--draft must be a comma-separated list of auto and/or 1..4")
    return list(dict.fromkeys(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_hc = sub.add_parser("hc16")
    p_hc.add_argument("--max-tokens", type=int, default=24)
    p_hc.set_defaults(func=cmd_hc16)
    p_qsa = sub.add_parser("qsa")
    p_qsa.add_argument("--chunks", default="4096,16384")
    p_qsa.add_argument("--max-tokens", type=int, default=24)
    p_qsa.set_defaults(func=cmd_qsa)
    p_mtp = sub.add_parser("mtp")
    p_mtp.add_argument("--max-tokens", type=int, default=128)
    p_mtp.add_argument("--draft", type=draft_list, default=["auto", "1", "2", "3", "4"],
                       help="comma-separated lengths or auto (default: auto,1,2,3,4)")
    p_mtp.add_argument("--workloads", action="store_true", help="nine prompt types; thinking enabled only for reasoning")
    p_mtp.add_argument("--temperature", type=float, default=0, help="0 for greedy; e.g. 0.7 for seeded sampled A/B")
    p_mtp.add_argument("--seed", type=int, default=123, help="nonzero deterministic sampler seed")
    p_mtp.set_defaults(func=cmd_mtp)
    args = ap.parse_args()
    if args.cmd == "mtp" and args.seed == 0:
        ap.error("--seed must be nonzero for reproducible sampled A/B")
    if not Path(SERVER_BIN).exists():
        log(f"SKIP: server binary not found at {SERVER_BIN}")
        return 77
    if not Path(MODEL).exists():
        log(f"SKIP: model not found at {MODEL}")
        return 77
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
