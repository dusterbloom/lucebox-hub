#!/usr/bin/env python3
"""Multi-turn HTTP chat client for composition benchmarks."""
import sys
import json
import time
import urllib.request
import urllib.error

base_url, model_id, out_path, timing_path = sys.argv[1:]

prompts = [
    (
        "Explain what this Python function does and give one concrete usage example. "
        "End your reply with exactly: OK_DONE\n\n"
        "def clamp(x, lo, hi):\n"
        "    if lo > hi:\n"
        "        raise ValueError('bad bounds')\n"
        "    return min(max(x, lo), hi)"
    ),
    (
        "Now apply clamp() to normalize a list of sensor readings to the range [0, 100]. "
        "Show a short Python snippet (15 lines or fewer). End your reply with exactly: OK_DONE"
    ),
    (
        "What would happen if someone passed lo=50, hi=10 to clamp()? "
        "What defensive patterns exist to avoid that mistake? "
        "End your reply with exactly: OK_DONE"
    ),
]

messages = []
turns = []
all_text = []

for i, prompt in enumerate(prompts):
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 400,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-lucebox",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        wall = time.time() - t0
        content = body["choices"][0]["message"]["content"]
        ok_done = "OK_DONE" in content
        usage = body.get("usage", {})
        turns.append({
            "turn": i + 1,
            "wall_s": round(wall, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "ok_done": ok_done,
        })
        messages.append({"role": "assistant", "content": content})
        all_text.append(f"=== Turn {i+1} ===\n{content}\n")
        print(
            f"[client] turn {i+1} wall={wall:.2f}s "
            f"tokens={usage.get('completion_tokens', 0)} ok_done={ok_done}",
            flush=True,
        )
    except Exception as e:
        wall = time.time() - t0
        turns.append({
            "turn": i + 1,
            "wall_s": round(wall, 3),
            "error": str(e),
            "ok_done": False,
        })
        all_text.append(f"=== Turn {i+1} ERROR: {e} ===\n")
        print(f"[client] turn {i+1} ERROR: {e}", flush=True)

with open(out_path, "w") as f:
    f.write("\n".join(all_text))
with open(timing_path, "w") as f:
    json.dump(turns, f, indent=2)

ok_count = sum(1 for t in turns if t.get("ok_done"))
print(
    f"[client] completed {len(turns)} turns, OK_DONE in {ok_count}/{len(turns)}",
    flush=True,
)
