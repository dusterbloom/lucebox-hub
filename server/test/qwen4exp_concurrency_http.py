#!/usr/bin/env python3
"""Run four long prompts alone, then together; also check planted-fact recall.

Usage: python3 server/test/qwen4exp_concurrency_http.py [http://127.0.0.1:8080]
Run against the concurrency-enabled server with prefix caching disabled.
HTTP checks text/counts; the GPU C++ probes check token IDs and indexer state.
"""
import concurrent.futures
import json
import sys
import threading
import urllib.request


def post(base, path, body):
    request = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.load(response)


def payload(slot, lines):
    secret = ["VIOLET-7319", "AMBER-4826", "COBALT-9153", "JADE-2648"][slot]
    filler = "The archive entry describes ordinary weather and routine maintenance.\n"
    split = lines * (slot + 1) // 5
    prompt = (
        f"Archive {slot}. Read the records and answer the question at the end.\n"
        + filler * split
        + f"The vault access code is {secret}.\n"
        + filler * (lines - split)
        + "What is the vault access code? Reply with only the code."
    )
    return {
        "model": "luce", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 42, "max_tokens": 96,
        "reasoning_effort": "none", "stream": False,
    }, secret


def run(base, target):
    requests = []
    for slot in range(4):
        # Count with the server tokenizer, rather than assuming words == tokens.
        lo, hi = 0, target
        while lo < hi:
            mid = (lo + hi) // 2
            body, _ = payload(slot, mid)
            count = post(base, "/v1/messages/count_tokens", body)["input_tokens"]
            if count < target:
                lo = mid + 1
            else:
                hi = mid
        requests.append(payload(slot, lo))

    def generate(item):
        return post(base, "/v1/chat/completions", item[0])

    solo = [generate(item) for item in requests]
    barrier = threading.Barrier(4)

    def generate_parallel(item):
        barrier.wait()
        return generate(item)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(generate_parallel, requests))
    ok = True
    for slot, ((_, secret), a, b) in enumerate(zip(requests, solo, parallel)):
        text_a = a["choices"][0]["message"].get("content") or ""
        text_b = b["choices"][0]["message"].get("content") or ""
        counts = [r["usage"]["prompt_tokens"] for r in (a, b)]
        passed = (
            all(target <= n < target + 64 for n in counts)
            and a["choices"] == b["choices"]
            and a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"]
            and text_a.strip() == secret and text_b.strip() == secret
        )
        print(json.dumps({"target": target, "slot": slot, "pass": passed,
                          "solo": a, "parallel": b}), flush=True)
        ok = passed and ok
    return ok


if __name__ == "__main__":
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
    results = [run(base, target) for target in (2304, 16384)]
    sys.exit(0 if all(results) else 1)
