#!/usr/bin/env python3
"""TTFT vs wall_total investigation at 32K — disambiguate n_gen variable.

Measures TTFT and total wall time for pflash OFF vs ALWAYS across
n_gen values {8, 64, 256} to separate prefill speedup (TTFT) from
decode contribution (total wall).

Usage:
    python dflash/bench/ttft_investigation.py \
        --out dflash/bench/results/2026-05-21_ttft_investigation \
        --server http://127.0.0.1:8080 \
        --mode off|always \
        --cases /tmp/ttft_cases_32k.jsonl \
        --n-gen 8,64,256

Designed to be called twice: once per mode (off, always).
Results are appended to out/raw_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def measure_request_streaming(
    host: str,
    port: int,
    path: str,
    payload: dict,
    timeout: float = 300.0,
) -> dict:
    """
    Send a streaming POST request over a raw TCP socket.
    Returns dict with ttft_s, wall_s, output_tokens, response_text, error.

    We use raw sockets instead of requests/httpx to avoid any buffering
    that would delay first-byte detection. The socket reads chunk-by-chunk
    and records the timestamp of the first SSE content delta.
    """
    body = json.dumps(payload).encode()
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body

    t_send = time.monotonic()
    ttft_s = None
    wall_s = None
    response_text = ""
    output_tokens = 0
    error = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(request)

        # Read response, detect HTTP header end, then parse SSE events.
        buf = b""
        header_done = False
        header_end = 0

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                error = "socket timeout"
                break
            if not chunk:
                break
            buf += chunk
            now = time.monotonic()

            if not header_done:
                # Find end of HTTP headers
                idx = buf.find(b"\r\n\r\n")
                if idx < 0:
                    continue
                header_end = idx + 4
                header_done = True

            # Parse SSE lines from the body portion
            body_so_far = buf[header_end:]

            # Process lines looking for first content delta
            lines = body_so_far.decode("utf-8", errors="replace").split("\n")
            for line in lines:
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    wall_s = time.monotonic() - t_send
                    break
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # OpenAI chat format: look for content delta
                if "choices" in evt:
                    choices = evt.get("choices", [])
                    for ch in choices:
                        delta = ch.get("delta", {})
                        content = delta.get("content", "")
                        if content and ttft_s is None:
                            ttft_s = now - t_send
                        response_text += content
                    # Usage chunk
                    usage = evt.get("usage", {})
                    if usage.get("completion_tokens"):
                        output_tokens = usage["completion_tokens"]

            if wall_s is not None:
                break

        sock.close()

    except Exception as e:
        error = str(e)

    if wall_s is None:
        wall_s = time.monotonic() - t_send

    return {
        "ttft_s": ttft_s,
        "wall_s": wall_s,
        "output_tokens": output_tokens,
        "response_text": response_text[:200],
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", required=True, choices=["off", "always"])
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--n-gen", default="8,64,256")
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Parse server URL
    url = args.server.rstrip("/")
    if url.startswith("http://"):
        url = url[7:]
    host, port_str = url.rsplit(":", 1)
    port = int(port_str)

    n_gen_list = [int(x) for x in args.n_gen.split(",")]

    # Load cases
    cases = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"[ttft] loaded {len(cases)} cases from {args.cases}", flush=True)

    raw_out = args.out / "raw_results.jsonl"
    results = []

    total = len(n_gen_list) * len(cases)
    done = 0

    for n_gen in n_gen_list:
        for ci, case in enumerate(cases):
            prompt = case["prompt"]
            answer = case["answer"]

            # Build OpenAI-format streaming payload
            payload = {
                "model": "dflash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": n_gen,
                "stream": True,
                "temperature": 0.0,
            }

            # For pflash=off mode: no extra_body needed (server started with off)
            # For pflash=always mode: server started with always, no override needed
            # (per-request override can't switch OFF<->ALWAYS, see http_server.cpp L551)

            print(f"[ttft] mode={args.mode} n_gen={n_gen} case={ci} ...", flush=True)
            t0 = time.monotonic()
            r = measure_request_streaming(host, port, "/v1/chat/completions", payload)
            elapsed = time.monotonic() - t0

            row = {
                "mode": args.mode,
                "n_gen": n_gen,
                "case_idx": ci,
                "prompt_len": case.get("n_tokens", len(prompt.split())),
                "answer": answer,
                "ttft_s": r["ttft_s"],
                "wall_s": r["wall_s"],
                "output_tokens": r["output_tokens"],
                "response_text": r["response_text"],
                "error": r["error"],
                "keep_ratio": args.keep_ratio,
            }
            results.append(row)
            done += 1

            ttft_str = f"{r['ttft_s']:.2f}s" if r['ttft_s'] is not None else "NaN"
            print(
                f"[ttft]   ttft={ttft_str} wall={r['wall_s']:.2f}s "
                f"out_tokens={r['output_tokens']} "
                f"({done}/{total})",
                flush=True,
            )

            if r["error"]:
                print(f"[ttft]   ERROR: {r['error']}", flush=True)

    # Append to raw results file
    with open(raw_out, "a") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"\n[ttft] wrote {len(results)} rows to {raw_out}")


if __name__ == "__main__":
    main()
